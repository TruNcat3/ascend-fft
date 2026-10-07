# 实数变换：r2c / c2r 半谱链路

> **一句话**：在不动 `kfft_fwd` 主体的前提下，用「短链后处理」把复数前向 FFT 复用成
> `r2c`（实数 → `n/2+1` 半谱）与 `c2r`（半谱 → 实数）两个方向，与 `torch.fft.rfft` /
> `torch.fft.irfft` 布局逐点可比；正确性 **98/98 PASS**（`maxRel ≤ 3.1e-7`），
> device-only 几何均值 r2c **65.2 µs**（vs torch **1.84×**、vs `aclRfft1D` **3.21×**）、
> c2r **60.5 µs**（vs torch **3.06×**）。
>
> **复现**：[`scripts/repro.sh r2c-c2r`](../scripts/repro.sh)（基准）·
> `scripts/one_click_test.sh`（门禁，含 `test_framework` 的 r2c/c2r 冒烟）。
> 逐点数据：[`results/r2c_c2r.json`](../results/r2c_c2r.json)。

---

## 1. 为什么

CANN 9.0.0 公开头里唯一可用的 FFT C API 是 `aclRfft1D`（实→复）与 `aclSTFT`，
**没有 c2r**，也没有复数→复数的 C API（见 [实验对比 图7](实验对比.md)）。
本仓库已经补齐复数前向（49:0 vs torch_npu `_fft_c2c`），但真实应用里还有一半入口是实信号：

- **实信号频谱分析**：输入 `n` 个实数，只要前 `n/2+1` 个 bin（numpy `rfft` 语义）；
- **频域处理回时域**：均衡 / 滤波 / 重建后再取实部（numpy `irfft` 语义）。

torch 两个方向都有，CANN 只有前者。所以这一步把两个方向都做成 kernel 链，
让 host 侧 C++ 一次 `Plan::runR2C` / `Plan::runC2R` 就能走完全程。

---

## 2. 链路设计：复用 `kfft_fwd`，不新写变换核

三个新 kernel（[`src/ascendc/fft_real.cpp`](../src/ascendc/fft_real.cpp)）都是
**后处理/预处理**，蝶形本体一行不改：

```
r2c :  x[n] 实数  ──视为交错复数──▶  kfft_fwd(n/2)  ──▶  kfft_r2c_post   ──▶  半谱 [n/2+1]
                                                        （W_n^k 拼接 + 共轭镜像）
c2r :  半谱 [n/2+1] ──▶ kfft_c2r_prep ──▶ 满谱 [n] ──▶ kfft_fwd(n, +i 约定) ──▶ kfft_c2r_post ──▶ x[n]
                                                     （旋转因子取反 + xflip）      （偶 bin 提取 × 1/n）
```

**r2c 的输入零拷贝**：实数行 `[x0..x_{n-1}]` 在内存里恰好就是长度 `n/2` 的交错复数
`(x0,x1),(x2,x3),…`。对它做一次 `n/2` 点复数 FFT 得到 `Z[k]`，再用
`X[k] = (Z[k] + conj Z[m-k]) / 2 + W_n^k · (Z[k] − conj Z[m-k]) / 2i`（`k ≤ m = n/2`）
拼出前半谱，`k > m` 由共轭对称补全 —— 这就是 `kfft_r2c_post` 做的事，
旋转因子表 `W_n^k, k ∈ [0, n/4]` 由 host 生成（槽上取整到 8，32B 对齐）。

**c2r 为什么反号**：`kfft_fwd` 的旋转因子是 `e^{−2πi·}`（正向 DFT）。
把表整体取反成 `e^{+2πi·}`，`kfft_fwd` 就直接输出 `n·IDFT`，
`kfft_c2r_post` 只需提偶数 bin 并乘 `1/n`。这比「正向算完再共轭回来」少一整趟核。

---

## 3. 布局约定

| 层 | 布局 | 说明 |
|---|---|---|
| **框架 API（稠密）** | `runR2C` 出 / `runC2R` 入 `[batch][n+2]` | `n/2+1` 个交错复数，同 `numpy.fft.rfft` / `irfft` |
| **设备行距** | `[batch][n+16]` float | 半谱行距 `hsStride(n) = n+16`，16 float 的 slack 供 `DataCopy` 越界读（见 §5） |
| **c2r 归一化** | 输出含 `1/n` | 同 `numpy.fft.irfft`（不是未归一 IDFT） |
| **Nyquist 虚部** | **必须为 0** | `kfft_c2r_prep` 的镜像会把它取负（`−0 = 0`）；非 0 输入与 numpy 语义（直接忽略）不同 |

设备行距与 API 稠密布局之间的拼装/拆行由框架 host 侧完成（`runR2C` 拆行、`runC2R` 拼行），
kernel 永远只看行距版。`fft_check` 的 `AB_DIR=r2c|c2r` 走同一套链，判据同 `maxRel ≤ 1e-4`。

---

## 4. 两个关键机制

### 4.1 xflip：`packArg` 第 16 位翻交叉旋转

c2r 需要 `kfft_fwd` 走 `+i` 约定，但表里所有量（`W`、`W²`、`W³`、`−ti`…）
都是查表自动共轭的，**唯一硬编码进蝶形的相位是 radix-4 融合里的交叉旋转 `±i = W_{4h}^h`**。
处理办法不是改表，而是打包参数加一位：

- `packArg` 第 16 位 = `xflip`（`fft_check` 打包 / `Plan` 的 `buildArgs` 同一位）；
- 内核在 `xflip=1` 时把 `±i` 交叉项的 **`Sub` 两个操作数对调**
  （`u0−u4−u2` ↔ `u0−u2−u4`，等价于 `s` 取反，还省两条 `Muls` 取反指令）。

试过就地 `Muls` 取反 `s`，在 Level-0 批折叠路径下不可靠（`uTA` 的 `srcStride=arRep ≠ tRep`，
跨步取反写坏），对调操作数在 Level-0 / Level-2 两条路径都成立。

### 4.2 行首 `PipeBarrier<PIPE_ALL>()`：首行 WAW 竞争

三个实数 kernel 都是「按行复用 UB 缓冲」的循环。首行的缓冲写与**上一次 launch 尚在途的
UB 写**构成 WAW —— 早期表现是每次 launch 的第 0~`nblk-1` 行偶发坏值（`maxRel` 在
0 ~ 3e-1 之间随机跳）。修法是每个行循环开头先排空一遍：

```cpp
for (int64_t b = blk; b < (int64_t)batch; b += nblk) {
    PipeBarrier<PIPE_ALL>();   // 行复用缓冲的首写前必须排空在途写
    ...
}
```

`kfft_c2r_prep` 早期还有一版独立 `hbuf` 中转，去掉后（输入行直落 `full[0..inSt)`）
n=8192 的四缓冲 UB 占用从 196.7 KB 降到 164 KB，才塞进 192 KB 上限。

---

## 5. 探针事实决定了实现形状

dav-c100 上的实测约束（[阶段0-1 §1](阶段0-1-发现与结果.md)、[性能优化 §10](性能优化-C2b与K择优.md)）：

| 事实 | 对实现的影响 |
|---|---|
| `DataCopy` 长度静默截断到 8 float 整数倍 | 半谱行距必须带 slack（`n+16`），否则行尾 Nyquist 读不到 |
| `Gather` dst/idx 必须 32B 对齐 | `kfft_r2c_post` 的 L/U 段表、`kfft_c2r_prep` 的 `[re|−im]` 区全部按 8 元素上取整 |
| `Scatter` 是 no-op 桩 | 镜像/拼接只能 `Gather` + 全缓冲写，不能散写 |
| `(float)uint32` 转换被编译器禁止 | 偏移全部走 `uint32` 表 + `Gather` |
| UB 192 KB | 守卫：r2c `n ∈ [128, 8192]`（内层 fwd ≤ 4096），c2r `n ∈ [64, 4096]`，c2c ≤ 4096 |

---

## 6. 框架接入（`Plan::runR2C` / `runC2R`）

[`include/butterfly/plan.hpp`](../include/butterfly/plan.hpp) 新增两个入口，与 `run()`
同族：`fft_real.o` 在 `Context::makePlan` 里**尽力加载**（与 `kernelPath` 同目录，
`AB_REAL_O` 可覆盖；缺文件不影响 c2c，实数入口返回 `-1`）。

- **`prepareSign(n, batch, sign)`**：`prepare()` 的实现体。旋转因子符号进了缓存键 ——
  同一 plan 上 c2c → c2r 切换会重新生成（取反）旋转因子表，切回再生成一遍，
  不会拿错符号的表继续跑。
- **`buildArgs(..., xflip)`**：第 16 位按方向置位；c2c 路径默认 `false`，逐字节不变。
- **`ensureBuf()`**：设备缓冲按容量扩容。旧 `run()` 只在首次 `malloc`，同一 plan 换大
  shape 复用会越界 —— 这次一并修掉。
- **双精度参考** `refR2CF32` / `refC2RF32` 进 [`reference.hpp`](../include/butterfly/reference.hpp)，
  与 `fft_check` 的公式逐点同式，`test_framework` 共用。

**验收分层**：

| 层 | 内容 | 结果 |
|---|---|---|
| `fft_check AB_DIR` 正确性网格 | r2c `n=128…8192`、c2r `n=64…4096` × `B=1…4096` | **98/98 PASS**，`maxRel ≤ 3.122e-7` |
| `test_framework` 冒烟 | 门禁 B 的 4 个抽样点自动带 r2c/c2r 检查 | 4/4 PASS（`≤ 3.1e-7`），`fft_real.o` 缺失时跳过 |
| 单 plan 压力 | c2c → r2c → c2r → r2c → c2c 方向反复切换 + n=128→8192 换 shape 扩容 + 越界守卫 | 全 PASS，守卫 `-8` |
| 门禁 + 矩阵 | `scripts/one_click_test.sh` 全量 | ALL PASS：矩阵正确性 **49/49**、比值 **49/49 ≥ 1×**、η 7/49 带外 |

---

## 7. 结果

49 点全网格（`n ∈ {64…4096}` × `B ∈ {1…4096}`，r2c 自 `n=128` 起 42 点），
`device-only`、同 reps 政策（`n·B ≤ 4096: 50`、`≤ 2²⁰: 30`、否则 `10`，逐点 min）：

| 方向 | 自研 geo | torch geo | 比值 | `aclRfft1D` geo | 比值 |
|---|---:|---:|---:|---:|---:|
| r2c（42 点，`n≥128`） | **65.17 µs** | 119.79 µs | **1.84×** | 208.87 µs | **3.21×** |
| c2r（49 点） | **60.49 µs** | 185.30 µs | **3.06×** | —（CANN 无 c2r） | — |

- `aclRfft1D` 一列来自 `results/e2e.json` 的 `bare_dev`（同卡同形状的裸 CANN device 时间）。
- torch 一列是 `torch.fft.rfft` / `torch.fft.irfft`（torch_npu），与自研同 reps 政策。
- 逐点数值、逐点比值见 [`results/r2c_c2r.json`](../results/r2c_c2r.json)。

**调试钩子**（`fft_check`，均不影响默认路径）：

| 环境变量 | 作用 |
|---|---|
| `AB_DIR=r2c\|c2r` | 走实数链（默认 `c2c`） |
| `AB_SGN=+1\|-1` | 强制旋转因子符号（默认 c2r 取反） |
| `AB_INPUT_FILE=<path>` | 输入改从文件读（与 torch 对拍） |
| `AB_DUMP=<prefix>` / `AB_DUMP2=<prefix>` | 导出输入输出 / 链路中间量（`.in.bin` `.out.bin` `.a.bin` `.b.bin`） |
| `AB_REAL_O=<path>` | 指定 `fft_real.o`（框架侧同名钩子） |
| `AB_FOLD_D=<k>` / `AB_PLANE_K=<8\|16\|32>` | 强制折叠系数 / 平面 K（A/B 用） |

---

## 8. 复现

```bash
scripts/repro.sh r2c-c2r            # 49 点基准 -> results/r2c_c2r.json + 汇总
scripts/one_click_test.sh --no-matrix   # 4 道门禁（门禁 B 自带 r2c/c2r 冒烟）
AB_DIR=r2c  ./build/fft_check 1024 64 30   # 单点：r2c 正确性 + device 时间
AB_DIR=c2r  ./build/fft_check 1024 64 30   # 单点：c2r
```

框架侧入口见 [README · API 概览](../README.md#api-概览)；
本仓文档总索引见 [`docs/README.md`](README.md)。
