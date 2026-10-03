# Cube（矩阵单元）张量化探针

> 结论先行：**fp32 Cube 在本 SoC（Ascend910_9382, `__NPU_ARCH__=2201`）上可用且已跑通全链路**，
> 纯 `Mmad` 下界 **109.4 GMAC/s ≈ 218.8 GFLOPS(fp32)**。
> 但它**必须跑在独立的 `__cube__`（AIC）内核里**，与现有纯矢量内核是**两套核**——
> 要吃到收益需要 split-core 架构（AIC kernel + AIV kernel + KFC 同步），不是加几行代码的事。

---

## 0. 复现

```bash
source scripts/env.sh
bash scripts/build.sh cube

# 正确性（a0b0 布局，确定性 PASS 15/15）
./build/cube_probe build/cube_probe.o 2 1 16 2 0 5

# 吞吐：mode 0=只LoadData 1=+Mmad 2=+Fixpipe 3=只Mmad循环
./build/cube_probe build/cube_probe.o 3 1 16 2 0 1 65536 5   # 纯 Mmad
./build/cube_probe build/cube_probe.o 1 1 16 2 0 1 65536 5   # LoadData + Mmad
./build/cube_probe build/cube_probe.o 2 1 16 2 0 1 65536 5   # 全链路（含 Fixpipe）

# profile
MSPROF=/usr/local/Ascend/cann-9.0.0/bin/msprof
$MSPROF --output=/tmp/op/cube --task-time=on --aic-metrics=PipeUtilization \
        ./build/cube_probe build/cube_probe.o 3 1 16 2 0 1 65536 5
```

文件：`src/ascendc/cube_probe.cpp`（内核 `kcube`）、`src/host/cube_probe.cpp`、build target `cube`。

---

## 1. 确定的语义（全部实证，非推断）

### 1.1 内核核型必须是 AIC
- `DataCopy GM→L1`（`DataCopyGM2L1Impl`）：`if ASCEND_IS_AIC {...} else if ASCEND_IS_AIV { ScmDataCopyMsg(...) /* 需 KFC */ }`
- `FixpipeL0cToOut` / `FixpipeL0cToL1`：`if ASCEND_IS_AIV { return; }` → **AIV 直接空转**
- 因此 `extern "C" __global__ __aicore__ __cube__ void kcube(...)`；
  我们原来的 `fft_radix2` 是 `__vector__` → **二者是两个不同的 kernel task**
  （msprof 里对应 `KERNEL_AICORE` vs AIV 指标，与原生 CANN 的 `AI_CORE` / `AI_VECTOR_CORE` 分裂一致）。

### 1.2 fp32 fractal = 16×8 = 512B
- `GemmTiling.blockSize = 16`（`const`），fractal 元素数 = `blockSize * c0Size`，恒 512B
  （int8: 16×32、fp16: 16×16、**fp32: 16×8**）。
- 证据：`loadRep=1`（只装 1 个 fractal）时结果**跨进程不确定**（L0A 后半是脏数据）；
  `loadRep=2` 时 **3 进程 × 5 重复 = 15/15 确定性 PASS**。

### 1.3 GM 布局（host 侧预打包，`LoadData` 直接从 GM 读 NZ）
| 矩阵 | 布局 | 说明 |
|---|---|---|
| A (M×K) | `[k/8][16 m × 8 k]` 行主序 | `out[kb*128 + r*8 + j] = A[r][kb*8+j]` |
| B (K×N) | `[k/8][16 n × 8 k]` 行主序 | `out[kb*128 + n*8 + j] = B[(kb*8+j)*N + n]`（即 **Bᵀ 的 NZ**） |
| C (M×N) | Fixpipe NZ→ND 直接写出 | `CFG_ROW_MAJOR` 内部 `nz2ndEn=true` |

16 个候选布局里只有 `a0b0` 正确（其余全 FAIL，且错误形态各异 → 说明扫描有效）。

### 1.4 各算子参数（16×16×16 验证通过的那组）
```cpp
LoadData2DParams lp(/*startIndex*/0, /*repeatTimes*/2, /*srcStride*/1,
                    /*sid*/0, /*dstGap*/0, /*ifTranspose*/false, /*addrMode*/0);
MmadParams       mp(/*m*/16, /*n*/16, /*k*/16, /*unitFlag*/0,
                    /*cmatrixSource*/false, /*cmatrixInitVal*/true);
FixpipeParamsV220 fp(/*nSize*/16, /*mSize*/16, /*srcStride*/16,
                     /*dstStride*/16 /* NZ2ND 时为元素单位的行距 */, /*reluEn*/false);
```
- `Mmad` fp32 合法性：`MmadCal` 的 `SupportType<Tuple<float,float,float>>` 显式列出；
  `GetGemmTiling` 非 int8 时硬编码 `c0=16, dSize=2`（fp16 取向）→ **不能用 `Gemm/GetGemmTiling`，要自己填 `GemmTiling` 或直接用 `LoadData+Mmad+Fixpipe`**。
- `Fixpipe float→float` 合法性：`CheckFixpipeL0C2GMParam` 的 `Tuple<float,float>` + `quantPre==NoQuant`。

### 1.5 收尾必须 `PipeBarrier<PIPE_ALL>()`
只用 `PIPE_MTE2` / `PIPE_M` / `PIPE_FIX` 分段栅栏时，输出**同参不同进程结果不同**
（全 0 / 垃圾 / 正确三种形态随机出现）。换成每段 `PIPE_ALL` 后完全确定。
> 这条也提醒 `fft_radix2`：任何"看着对但偶发错"的 bug 优先查收尾栅栏。

---

## 2. 吞吐实测（`nTile=65536`，每 tile 16×16×16 = 4096 MAC，合计 268.4 M MAC）

| mode | 内容 | 耗时/次 | 折算 | 备注 |
|---|---|---:|---:|---|
| **3** | **只 `Mmad` 循环（每条后 `PipeBarrier<PIPE_M>`）** | **2454.6 µs** | **109.4 GMAC/s = 218.8 GFLOPS** | **下界**（无双缓冲、每条都栅栏） |
| 1 | `LoadData(A,B)` + `Mmad` | 13291 µs | 20.2 GMAC/s | 每 tile 2×512B 小包 + `PIPE_ALL` → GM 仅 15 GB/s |
| 2 | 全链路 + `Fixpipe` | 31079 µs | 8.6 GMAC/s | **Fixpipe 每 tile 一次是最大开销** |
| 0 | 只 `LoadData` | 2402 µs(@16384) | GM ~10 GB/s | 栅栏把流水彻底串行化 |

单 tile 小规模（`nTile=1`）= 3.72 µs/call，基本是 launch 开销。

### msprof（mode=3，`KERNEL_AICORE` ×13，duration ≈2455 µs）
| 指标 | 本探针 | 原生 CANN matmul（对照） |
|---|---:|---:|
| `task_type` | **KERNEL_AICORE** | AI_CORE（与 Transpose 的 AI_VECTOR_CORE 分离） |
| `aic_mac_ratio` | **0.576** | 0.095 / 0.093 / 0.52（3 个 matmul） |
| `aic_scalar_ratio` | 0.089 | 0.54 / 0.32 |
| `aic_mte1/mte2/fixpipe_ratio` | 0 / 0 / 0 | mte2 ≈ **0.98**（原生被 GM 带宽卡死） |
| `aiv_*` | 全 0 | Transpose 那几个 task 才有 |
| `cube_utilization(%)` | 4.166 | 96–98 / 78–80 / 67–69 |

> 两份 `cube_utilization` 口径与 `aic_mac_ratio` 不同步增减，**跨 kernel 比较请用 `aic_mac_ratio`**。
> 关键对比：原生 matmul `mte2_ratio≈0.98` → **它把 Cube 当搬运工（GM 带宽受限，有效 418 GB/s，MAC 有效率 0.095）**；
> 我们的探针 `mte*=0` → **纯 MAC + 栅栏受限**。

---

## 3. 对 FFT 张量化的判断

1. **原生已经用 Cube，但没有因此赢**：n=4096/B=4096 下原生 2461 µs，我们 2031 µs（**1.21× 领先**）。
   原生 1 次调用 = 3×BatchMatMul(共 ~1325 µs) + 3×Transpose(**~1290 µs，纯矢量**)。
   → 原生的短板正是矢量 transpose；**只把 GEMM 搬上 Cube 是不够的**。
2. **我们的瓶颈不是算力**：当前 kernel `aiv_vec_ratio 0.813`、`aiv_scalar 0.168`，
   代价模型 `≈12.1 ns/op/batch + ~1167 µs 下限` → **矢量发射墙 + 标量/收尾下限**。
3. **要吃到 Cube 需要什么**：
   - **架构**：拆成 AIC kernel（做 group-DFT / 16 点稠密 matmul）+ AIV kernel（twiddle、bit-reverse、index、planar 段），
     两者并行 + KFC/事件同步；或用 `SPLIT_CORE_CUBE` / adv API `Matmul`（`asc/include/adv_api/matmul_intf.h`，
     2201 上需 `MatmulClient` + KFC）。
   - **数据通路**：原生证明 `GM→L1→L0→Cube` 能跑到 ~418 GB/s，而矢量路径 81–132 GB/s；
     **真正的机会是把"整张量过一遍"的搬运搬到 Cube 通路**，而不是指望 MAC。
   - **必须一次性大 Fixpipe**：本探针证明每 tile 一次 Fixpipe 会让吞吐从 109 掉到 8.6 GMAC/s
     → 设计上必须 `m×n` 足够大再写出（原生的 64×64×524288 就是这个形状）。
4. **优先级建议**：split-core 是大工程（估计比已做的 C1+C2 大一个量级）。
   在动手前应先收割**确定性收益**：C2b（planar `Mul`+`MulAddDst`，150→127 op）、
   scalar 占空 16.8% 的削减、`η` 重标定。

---

## 4. 已知边界 / 下一步探针没做完的部分
- 只验证了 **16×16×16** 单 tile；更大 `m/n/k`（如 64×64×16、64×64×4096）的 tiling 与
  `LoadData` fractal 顺序（`[m/16][k/8]` vs `[k/8][m/16]`）**未验证**。
- 未做双缓冲（`LoadData` 下一块与 `Mmad` 当前块重叠），mode=1 的 15 GB/s 是串行下界。
- 未尝试 `L0C→UB`（`FixpipeL0C2UBImpl` 在本设备 `ASCENDC_DEBUG_ASSERT(false)` → **不支持**，只能 L0C→L1/GM）。
- 未评估 `Gemm` 高层 API（deprecated，且 fp32 需自造 `GemmTiling`：`c0Size=8, dtypeSize=4`）。
