# 性能优化（第二、三批）：C2b + K 择优 + merge 放宽 + MTE 重叠 + η 重标定 ／ 平面级 radix-4

> **复现脚本**：[`scripts/ab_test.py`](../scripts/ab_test.py) 批量 A/B 消融（§7/§9/§11）· [`scripts/baseline_o.sh`](../scripts/baseline_o.sh) 从提交重建历史基线 `.o` · [`scripts/calib_eta.py`](../scripts/calib_eta.py) η 标定 · [`scripts/hw_probe.sh`](../scripts/hw_probe.sh) §10 硬件探针 · [`scripts/one_click_test.sh`](../scripts/one_click_test.sh) 门禁+矩阵。
> **全量索引**：[`scripts/repro.sh --doc <文档名片段>`](../scripts/repro.sh) 列出本文件涉及的全部实验与命令；`scripts/repro.sh --list` 是全仓清单。
> 全部为 A/B 实测（每轮各 50 reps 取 **min**，3 轮取最优），交替运行以抵消共租户噪声。
> 背景：机器 `nproc=1`、load 常年 22~30，且 NPU 上有其他租户（`npu-smi` 可见 HBM/功耗非空），
> 单次运行会出现 **双峰**（~2100 µs / ~4100 µs，后者是共租户抢占），
> **只有「取 min + A/B 交替」才可比**，均值在此环境下不可用。

---

## 0. 结果总览

| 阶段 | n=4096/B=4096（min/50reps） | 增益 | 变更 |
|---|---:|---:|---|
| C1+C2（上一批） | ~2013 µs | — | 平面级 Axpy 合并、planar Level-0 折组 |
| **+ C2b** | **1975 µs** | **−1.9%** | planar 复数乘 6→4 算子（`MulAddDst`） |
| **+ K 择优修正** | **1677 µs** | **−15.1%** | `planeKFor` 改用当前真实算子公式 |
| **+ merge 条件放宽** | **1638 µs** | **−2.3%** | `h<=255` 是多余约束，按 stride 字段判 |
| **+ 去掉 4 个矢量管内屏障** | **1633 µs** | **−0.2%** | 同管顺序执行，`PIPE_ALL` 冗余（实测≈0） |
| **+ MTE3⊗MTE2 重叠** | **1560 µs** | **−4.5%** | 批间输出写与下一批输入读并发 |
| **累计（第二批）** | **1560 µs** | **−22.5%** | — |
| **+ 平面级 radix-4 融合（第三批，§9）** | **1481 µs** | **−5.3%** | 两阶段代数消元，平面级算子 226→164 |
| **累计（三批）** | **1481 µs** | **−27.0%** | 相对 C1+C2 的 2013 µs |

最终 `fft_check 4096 4096 50`：中位 **1481.3 µs** / 最优 **1476.1 µs**（插队 3 轮取中位）。

K 择优在小 n 上更猛：

| n (B=4096) | 旧 K → 新 K | 旧 µs | 新 µs | 倍率 |
|---:|:---|---:|---:|---:|
| 512 | 16 → **8** | 438.4 | **297.2** | **1.48×** |
| 1024 | 16 → **8** | 558.5 | **457.5** | **1.22×** |
| 2048 | 32 → **8** | 1152.9 | **752.5** | **1.53×** |
| 4096 | 32 → **16** | 1981.6 | **1677.3** | **1.18×** |

每格三轮数值：512 (438.2/440.2/438.4 → 300.6/295.0/298.0)、1024 (561.2/558.8/555.6 → 455.6/458.9/457.9)、
2048 (1152.7/1156.8/1149.1 → 750.8/753.9/752.9)、4096 (1978.8/1980.3/1985.8 → 1677.9/1674.1/1680.0)
—— 轮内极差 < 0.5%，结论稳固。

---

## 1. C2b：planar 复数乘 6 → 4 算子

### 1.1 手法
```
re = r1*tr - i1*ti  =  i1*(-ti) + r1*tr   -> Mul + MulAddDst   (2 op)
im = i1*tr + r1*ti                      -> Mul + MulAddDst   (2 op)
```
`MulAddDst(dst, src0, src1)` 即 `vmla`，语义 `dst = dst + src0*src1`（**实测确认**，
`fft_check` maxRel ≈ 1e-7；注意别与 `FusedMulAdd` 搞混——后者是 `src0*dst+src1`，角色相反）。

负号 twiddle `v = -ti` **每级只算一次**，存进**本级已经空闲的 `t2` 临时区**
（合并路径原本用 `t2` 当 `a2` 中转，新公式只用 `t0`/`t1` 两个目的区）→ **零额外 UB**。

合并路径（`BinaryRepeatParams` 折组）与非合并路径（Level-2）各单独 A/B：

| 变体 | 含义 | min/50reps × 3 轮 | 结论 |
|---|---|---|---|
| old | 两路都 6 op | 2060.3 / 2013.5 / 2013.1 → **2013.1** | 基线 |
| v1 | 只非合并改 4 op | 2003.1 / 1997.8 / 1995.0 → **1995.0** | −0.9% |
| v2 | 只合并改 4 op | 1996.6 / 1999.2 / (4072.9) → **1996.6** | −0.8%（第三个是共租户峰） |
| new | 两路都改 | 2095.3 / 1975.5 / 1975.9 → **1975.5** | **−1.9%**，近似可加 |

改动：`src/ascendc/fft_radix2.cpp` planar 段（`if (!merge)` 块与 `nSlice` 循环）。

> **教训**：初版 A/B 出现「new 恒定 4080 µs、old 恒定 2120 µs」的假象，
> 15 reps 撞上共租户抢占窗；换成 50 reps 取 min 后结论完全反转。
> **本环境任何单次计时都不可信。**

---

## 2. K 择优公式修正（主要收益）

### 2.1 病因
`planeKFor()`（`include/butterfly/fft_k.hpp`）用一个**写死的旧公式**在 {8,16,32} 里选 K，
但该公式停在 **C1/C2 之前**：
```
旧: ops = 3K + 5K(logK-1) + Σ 5n/2^s + 6
```
而内核当前的**真实**算子调用数是：
```
平面级 si（h=2^si, 共 K/2 对）:
    j2==0 的对 tw=(1,0) -> 2×DataCopy -> 6 算子；其余 -> 8 算子（复数乘 4 + 蝶形 4）
    该级 = (K/(2h))*6 + (K/2 - K/(2h))*8      // h=1 => 3K；h>=4 => ~4K，不是 5K
planar 级 h=2^s:
    merge ? 8*ceil(h/64) : 8*groups            // 不是 5n/h
    + 每级 1 次 Muls(-ti)
每 batch 收尾: 7（2 DataCopy + 5 Gather）
```
用旧公式会**高估平面级、低估 planar 级**，于是总往大 K 偏。

### 2.2 修正后的选择（n=4096：K=32 → K=16）

| 项 | K=32（旧） | K=16（新） |
|---|---:|---:|
| 平面级算子 | 578 | **226** |
| planar 级算子 | 159 | 168 |
| 收尾 | 7 | 7 |
| **合计 ops** | **744** | **401** |
| 合计 elems | ~241,900 | ~234,000 |

平面级每算子长度 `rows = n/K`：K 变小 → `rows` 翻倍但算子数腰斩，**净省 343 次调用**；
elems 基本不变。代价模型 `opNs=15.4` 下 ops 项从 11.5 µs/batch 掉到 6.2 µs/batch。

### 2.3 改动位置（三处必须同式）
1. `include/butterfly/fft_k.hpp` — `planeKFor()`（内核与宿主共用，`[aicore]` 修饰）
2. `src/framework/butterfly.cpp` — `estimate()` 的 `planeOps/planarOps` 计数
3. `scripts/calib_eta.py` — `plane_k_for()` + `counts()`

### 2.4 验证
- 正确性：n = 64/128/256/512/1024/2048/4096 × B=64 全 PASS，maxRel ≤ 2.0e-7；
  `fft_check 4096 4096 20` maxRel = 2.606e-07
- `test_framework` 全流程 PASS

---

## 3. η 重标定

### 3.1 计数公式与 `estimate()` 对齐
`counts()` / `estimate()` 同式，元素数按真实长度算：
- 平面级：`ops_si × rows`
- planar 级：每级 `8` 个算子各覆盖 `h*groups = n/2` 元素 → `4n`，另加 `Muls(-ti)` 的 `h`
- 收尾每 batch `10n` 元素（2 DataCopy + 5 Gather）
- R≠2 的结构缩放分母同步改成新基准（R=2 时 `opF = elF = 1`）：
  `g = 4(R-1) + 2R·l2`、`opF = (g/(R-1))/8`、`elF = (g/(R·l2))/4`

### 3.2 拟合（`python3 scripts/calib_eta.py`，21 点 reps=20）
```
fit: launchUs=15.269 us   opNs=15.402 ns   elemNs=0.0556 ns
residual mean=7.2%  max=32.6%      LOO mean=8.3%  max=35.1%
```
最差点是 n=128/B=48（µs 级小点，host 噪声），其余基本在 ±10% 内。
**旧标定是 +60%~+111% 的系统性高估，现在残差均值 7.2%。**

系数已回写 `estimate()`（`src/framework/butterfly.cpp`）。

### 3.3 选型循环的健壮性修复
新计数下 R=4/R=8 候选的 η 会低于 R=2，导致 **top-3 全被未实现的 R≠2 占满 →
`selected: none`**（`no feasible candidate`）。
原循环条件是 `i < k`（**看了前 k 个**），现改为 `measured < k`（**实测成功 k 个为止**），
被 `skip`/`reject` 的候选不再消耗配额。
`test_framework` 日志：`3 accepted after 21 examined, quota 3`，最终 `PASS`。

---

## 4. merge 条件放宽（+2.3%）

原条件里有一个**多余的** `(h <= 255)`：
```cpp
// 旧
(h >= 8) && (h <= 256) && ((2h)%8==0) && ((2h)/8 <= 255) && (h <= 255) && (groups <= 255)
```
`BinaryRepeatParams` 的 `*RepStride` 是 `uint8_t`，真正限制是
`pSt = 2h/8 <= 255`（h ≤ 1020）与 `vSt = h/8 <= 255`（h ≤ 2040）——
前者已经写在条件里了，`h <= 255` 只是**旧的保守写法**，直接把 h=256 挡掉。

放宽后还要加一条**收益判据**：两路的元素数都是 `4n`，差的只是算子调用数，
只有 `nSlice = ceil(h/64) <= groups` 时折组才不亏。
```cpp
nSlice = (h + 63) >> 6;
merge = (h >= 8) && ((2h)%8==0) && ((2h)/8 <= 255) && (h/8 <= 255)
        && (groups <= 255) && (nSlice <= groups);
```

效果（n=4096, K=16）：h=256 那级从 `8*groups=64` 变成 `8*nSlice=32`，planar 算子 168 → 136，
总 ops 401 → 369。

| n (B=4096) | before | after | 说明 |
|---:|---:|---:|:---|
| 1024 | 457.4 | 459.0 | **不变**（h=256 时 groups=2 < nSlice=4，本就不该折组）——差值即噪声 |
| 4096 | 1678.0 | **1637.6** | **−2.4%**（h=256: groups=8 ≥ nSlice=4，折组生效） |

正确性：n = 64/256/512/1024/2048/4096 全 PASS，maxRel ≤ 2.0e-7。

三处同式同步：`fft_k.hpp`、`butterfly.cpp::estimate()`、`calib_eta.py`（`plane_k_for` + `counts`）。
同步后 K 选择不变（n=4096 仍为 16）。

---

## 5. 去掉矢量管内的 `PipeBarrier`（实测≈0，但语义更干净）

内核里 7 个 `PipeBarrier<PIPE_ALL>()`，其中 4 个**两侧都是矢量管的指令**：

| 位置 | 前 | 后 | 是否跨管 |
|:---|:---|:---|:---|
| 批首 `DataCopy(plan, gin)` 之后 | MTE2 | `Gather` | **跨管，保留** |
| 位反转 2×`Gather` 之后 | vector | 平面级算子 | 同管 → **删** |
| 平面级循环之后 | vector | `Gather` | 同管 → **删** |
| plane→planar 2×`Gather` 之后 | vector | planar 算子 | 同管 → **删** |
| planar 循环之后 | vector | `Gather` | 同管 → **删** |
| `Gather(..., idxO)` 之后 | vector | `DataCopy(gout,·)` | **跨管(MTE3)，保留** |
| `DataCopy(gout,·)` 之后 | MTE3 | 下一批 MTE2 | 跨管 → 见 §6 |

依据：`Gather` 实现是 `vgather`（`kernel_operator_vec_gather_impl.h`），`DataCopy(Local,Local,count)`
无 `__inout_pipe__` 注解 → 矢量管；`DataCopy(Global,Local)` 标 `__inout_pipe__(MTE2)`、
`(Local,Global)` 标 `MTE3`。矢量管顺序执行，同管屏障冗余。

**实测：0%**（1636 → 1633 µs，噪声内）。保留删除（更简单），并做了 **960 次正确性压力（0 FAIL）**。

---

## 6. 批间 MTE3⊗MTE2 重叠（−4.5%）

原流程每批是严格串行的三段：

```
MTE2 读 plan  →  vector 全部算  →  MTE3 写 gout  →  屏障  →  下一批 MTE2
   ~11%             ~75%              ~8%
```
`DataCopy(gout, ar)` 之后的 `PipeBarrier<PIPE_ALL>()` 把 **本批的输出写** 和 **下一批的输入读**
强行串起来，而两者操作的是**不同 UB**（`ar` vs `plan`）、不同 GM 地址。

删掉这个屏障后：
- MTE3（读 `ar` 写 GM）与 MTE2（读 GM 写 `plan`）并发；
- 安全性：本批的 `Gather(...,idxO)` 已被**前一个** `PIPE_ALL` 排空，`plan` 此时已可写；
- 下一批的矢量写 `ar` 会被 `DataCopy(plan,·)` 之后的 `PIPE_ALL` 挡住（`PIPE_ALL` 会连在途的 MTE3 一起排空），故无写读竞争。

**A/B（min/50 reps × 3，交替，0 FAIL / 960 次压力）**：

| n | 屏障版 | 重叠版 | 增益 |
|---:|---:|---:|---:|
| 1024 | 455.0 / 450.5 / 451.0 | 442.4 / 439.7 / 441.0 | −2.5% |
| 4096 | 1630.5 / 1637.8 / 1626.5 | **1565.9 / 1565.1 / 1557.4** | **−4.3%** |

msprof（`--aic-metrics=PipeUtilization`, n=4096/B=4096）前后对比：

| | 改前 | 改后 |
|---|---:|---:|
| `aiv_vec_ratio` | 0.754 | **0.822** |
| `aiv_scalar_ratio` | 0.123 | 0.140 |
| `aiv_mte2_ratio` | 0.109 | 0.114 |
| `aiv_mte3_ratio` | 0.084 | 0.093 |
| 设备任务时长 | ~1630 µs | **~1560 µs** |

**结论：`vec 0.822` 是当时的墙**（第二批口径；A7 之后见 §11.6）；MTE 与 scalar 仍是可压项，但已经没有"删一个屏障"这种免费午餐了。

---

## 7. 复现

```bash
source scripts/env.sh
bash scripts/build.sh all

# 正确性
./build/fft_check 4096 4096 20        # maxRel 2.606e-07, min ~1567 us
./build/test_framework                # eta=55.6 vs measured=52.6 us  PASS

# A/B（关键：交替 + min/50reps；.o 用 AB_FFT_O 指定，不要重新编译）
AB_FFT_O=<o> ./build/fft_check <n> 4096 50
# 多点批量版（同样交替 + 3 轮 min）：
python3 scripts/ab_test.py --base <基线.o> --cand build/fft_radix2.o \
    --points 4096x4096,64x4096 --reps 50 --rounds 3

# η 标定（拟合常量要回写 estimate()）
python3 scripts/calib_eta.py

# 全网格
python3 scripts/matrix_test.py --reps 20 --out docs/matrix_test.md
```

---

## 8. 下一步（按性价比）
1. **隐藏 MTE2（约 −5%）**：`plan` 目前同时充当"输入"和"planar 结果"两个角色，
   所以下一批的输入读必须等本批**最后一个矢量算子**结束，没法垫到矢量工作底下。
   要把它垫进去需要给输入单独开 32 KB 缓冲，而 UB 只剩 **6.9 KiB**。
   可腾出的空间（n=4096）：`idxO` 换成共享的 n 项索引 −16 KB、
   planar 复数乘改为「re 暂存进 `i1`」省掉 `t1` −8 KB → 合计 24 KB，**刚好够但没有余量**。
2. ~~**radix-4**~~ —— **已做，见 §9**。原估「−30~40%」是错的（把中间量也当成了要搬的元素），
   实测 **−5.3%**。剩余大杠杆仍是「把 `#op` 摊到 0」（流水重构）。
3. **η 小 batch 残差**（n=128/B=384 −30.1%）：模型缺一个「每 launch 固定的设备空转」项，
   或该点纯属噪声——需在安静时段复测确认。
4. **R≠2 分箱**：`select()` 已能实测它们，但 `GenTwiddles` 等 transform 还只实现 R=2，
   补齐后 R=4/R=8 的 η 才有意义。
5. **Cube 张量化**（见 `Cube张量化探针.md`）：原生已用 Cube 却仍落后，因其短板是 3× Transpose ≈1290 µs（vector）；
   我们要吃到 Cube 收益必须 split-core（AI_CORE + AI_VECTOR_CORE 两个 task + KFC 同步），量级更大，暂缓。

---

## 9. 第三批：平面级 radix-4 融合（−5.3%）

> A/B 口径与 §0 相同：**插队 3 轮 × min/50 reps**。基线 `.o` = 第二批结束时的临时产物
> `overlap.o`（仓库不保留；用 `scripts/baseline_o.sh <当时提交>` 重建，或用 `scripts/ab_test.py --base <.o>`）。

### 9.1 先修正一个被写进文档的错误估计

§8 第 2 条曾写「radix-4 约 −30~40%，元素搬运量降 ~46%」。**那个数字是错的**：它把
「12 stage 砍到 6 stage」当成了元素搬运也减半，但**每个 stage 本来就是读一遍写一遍**，
把两级合并成一级后仍然是读一遍写一遍 —— **朴素融合的元素流量是 0% 变化**。

正确的账要分两笔：

| 项 | 朴素两级 DIT 融合 | 真 radix-4（本批实现） |
|---|---:|---:|
| 中间量 | 必须落 UB（多 2n 读写） | **不存在**，4 点一次成型 |
| 平面级算子调用数（n=4096, K=16） | 32 → **0%** | 226 → **164（−27.4%）** |
| 元素流量 | 0% | **−15,872（−6.8%）** |
| η 模型 | 0% | **−9.6%** |
| **实测** | — | **−5.3%** |

**朴素融合为什么是 0%**：把 stage (h, 2h) 直接接起来，(x0,x1,x2,x3) 上要做的仍是
「先按 W² 分组做两个 radix-2，再按 W 分组做两个 radix-2」——逐元素的复乘加减一条不少，
而中间结果 `x'` 必须从 UB 写回再读回，**反而多 2n 元素流量**。

**真 radix-4 才有得赚**：把两阶段代数消元成一个 4 点变换。

### 9.2 推导（已代数核对，直接照做）

融合 stage `h` 与其后 `2h`，基 `h = 1<<si`，4 点块 `{x0, x1=x0+h, x2=x0+2h, x3=x0+3h}`，
`j2 ∈ [0,h)`、块数 `K/(4h)`。由 `genTwiddles` 的 `slot_s[j] = e^{-iπj/s}`：

- `W  = slot_{2h}[j2]`（外层级 twiddle）
- `W² = slot_h[j2]`（内层级 twiddle，**恒等式**：`(e^{-iπj/2h})² = e^{-iπj/h}`）
- `W³ = W·W²`，核内标量算（6 条标量指令）

```
p = x0 + W²·x1        q = x0 − W²·x1
u2 = W·x2,  u3 = W³·x3
r = u2 + u3,  s = u2 − u3
y0 = p + r    y2 = p − r
y1 = q − i·s  y3 = q + i·s
```

**`−i` 是交叉旋转**：实/虚各在一个张量里 → `i·s` 只是 `s` 的 re/im 跨张量换位，
**零算子**。这一条是整个方案的收益来源；如果 re/im 交织存储，它就要 4 条算子。

外层级第二个 pair 的 twiddle 是 `slot_{2h}[j2+h] = −i·W` —— 正是它引入了 y1/y3 的
交叉旋转，与上面的 `q ∓ i·s` 完全吻合。

**代数核对**（两组具体值对两阶段组合）：
- `h=1, j2=0`（W=1）：两阶段 = DFT-4；融合平凡路给出
  `(x0+x1+x2+x3, x0−x1+x2−x3, x0+x1−x2−x3, x0−x1−x2+x3)` 的实/虚重排 ✓
- `h=4, j2=1`：`W = e^{-iπ/8}`、`W² = e^{-iπ/4}`、`W³ = e^{-i3π/8}`，
  代入两阶段展开与融合式逐项相等 ✓

### 9.3 算子序列（两路，都只用 t0..t5）

**非平凡 `j2 != 0`（28 算子）**：3 个复乘 12（`Muls`+`Axpy`×2，C2b 后是 4 算子/复乘）+ 蝶形 16。

| # | 序列 | 算子 |
|---|---|---|
| 1 | `u1=W²·x1 -> t0/t1` | Muls+Axpy ×2 |
| 2 | `u2=W·x2 -> t2/t3` | Muls+Axpy ×2 |
| 3 | `u3=W³·x3 -> t4/t5` | Muls+Axpy ×2 |
| 4 | `q=x0−u1 -> x1`；`p=x0+u1 -> x0`（t0/t1 仍是 u1） | Sub×2, Add×2 |
| 5 | `s=u2−u3 -> t0/t1`；`r=u2+u3 -> t2/t3` | Sub×2, Add×2 |
| 6 | `y2=p−r -> x2`；`y0=p+r -> x0` | Sub×2, Add×2 |
| 7 | `y3=q+i·s -> x3`；`y1=q−i·s -> x1` | Sub+Add, Add+Sub |

**全平凡 `j2==0`（16 算子）**：`W=W²=W³=1`，去掉全部复乘，蝶形 16 算子。

对照两阶段基线：平凡 26、非平凡 32 → 融合 16 / 28。

### 9.4 改动清单

**内核 `src/ascendc/fft_radix2.cpp`**
- 平面循环重写为融合对循环 `si += 2` + 尾部 radix-2 残级（`logK` 为奇数时）；
  K=8→logK=3、K=32→logK=5 都会走到残级，K=16 无残级。
- **`slot` 消耗顺序 `align8(h) + align8(2h)` 与 `genTwiddles` 逐字节一致 → 宿主 twiddle 表零改动**。
- `bTmp`：`(3*hmax + rows)` → `(3*hmax + 3*rows)`，新增 `t4 = t3[rows]`、`t5 = t4[rows]`。
  UB：n=4096/K=16 → **191,616 B（余 4,992）**；最坏 K=8/rows=512 → 194,688 B（余 1,920）。
  平面循环先于 planar 跑，且 planar 只用 t0/t1/t2（各 `hmax` 长）→ **不冲突**。
- 平凡路改成判 `j2 != 0` 而不是判 twiddle 值：少 2 次 `GetValue`，且 `slot_s[0]` 恒为 `(1,0)`。

**三处算子公式同步**（必须同式，否则 `select()` 与实测脱钩）
1. `include/butterfly/fft_k.hpp::planeKFor`
2. `src/framework/butterfly.cpp::estimate()`
3. `scripts/calib_eta.py::plane_k_for()` / `counts()`

```c
si = 0;
for (; si + 1 < logK; si += 2) {          // 融合对
    h = 1 << si;
    ops += (K / (4*h)) * (16 + (h-1) * 28);
}
if (si < logK) {                          // 残余奇数级
    h = 1 << si;
    ops += (K/(2h)) * 6 + (K/2 - K/(2h)) * 8;
}
```

K 择优在新公式下**仍然是 K=16**（n=4096）：K=8 被 planar 的 `groups>255` 卡到 8×256+1=2049 次，
K=32 平面级 454 太贵 → 16（307 次）胜。

**顺手修了 `estimate()` 里一个真 bug**
```c
旧：opF = (g / (R-1)) / 8     // R=4 -> 1.1667
新：opF = elF                 // R=4 -> 0.875
```
`g(R)` 把复数乘按 **6 个实数算术**计、蝶形却按 **矢量调用**计，**单位不一致**；
真实比值与 `elF` 同源（融合后覆盖 `log2R` 级的调用数/元素数），两者相等。
该分支 R≠2 候选仍因 `GenTwiddles` 未实现而被 `select()` skip，但会让 η 排序失真。

### 9.5 实测

**A/B（插队 3 轮 × min/50 reps）**

| 轮 | 基线（第二批 `.o`） | radix-4 |
|---|---:|---:|
| 1 | 1574.1 | 1481.3 |
| 2 | 1564.2 | 1481.3 |
| 3 | 1564.7 | **1476.1** |
| **中位** | **1564.7** | **1481.3** |

→ **−5.3%**（模型 −9.6%）。正确性 `maxRel = 2.256e-07`。

**为什么只拿到一半（−5.3% vs 模型 −9.6%）**
模型算的 Δ = `62×14.868 + 15872×0.0569` = 1825 ns/batch × 86 = **157 µs**，实测只有 84 µs。
根因不是内核，是**旧内核跑得比旧模型好**：旧模型 1635 µs vs 旧实测 1565 µs（−4.3%），
而新模型 1480 µs vs 新实测 1481 µs（**−0.1%**）。
即两批都在同一个模型误差带上，**Δ 的差额 = 71 µs 的模型旧欠账**，不是 radix-4 少赚了。
重标定后该欠账已并进系数（见 9.6）。

**msprof（n=4096/B=4096，`--aic-metrics=PipeUtilization`）**

| 指标 | 第二批 | 第三批 | 变化 |
|---|---:|---:|---|
| `aiv_vec_ratio` | 0.822 | **0.853** | ↑ 矢量仍是墙，且墙变矮了 |
| `aiv_scalar_ratio` | 0.140 | **0.075** | **−46%** |
| `aiv_mte2_ratio` | 0.114 | **0.096** | −16% |
| `aiv_mte3_ratio` | 0.093 | **0.069** | −26% |
| `cube_utilization` | 0 | 0 | — |
| task duration（加权均值） | ~1566 µs | **1472 µs** | −6% |

scalar 几乎腰斩的原因：平面级从 32 次蝶形降到 8 次，`twr.GetValue` 读取与
`wr != 1.0f` 浮点比较随之减少；而 W³ 的 4 条标量乘加只在 7 次非平凡蝶形里做。

**网格（7×7 = 49 点，reps=20，`docs/matrix_test_raw.md`）**
- **49/49 PASS**，比分 **44 胜 5 负**（第三批当时不变，5 个负点仍是 `B=4096` 的 n≤1024；**A4~A7 之后 49:0**，见 §11.6）
- 几何均值 **2.87×**；n=4096/B=4096 **1,492.0 µs vs 原生 2,568.8 µs = 1.72×**（第二批 1.53×）
  > 注：原生这轮读到 2568.8（第二批是 2408.2），共租户噪声；**自研自身 1573→1492 = −5.2%** 是干净的。

### 9.6 η 重标定

```
旧：launchUs=18.040  opNs=14.868  elemNs=0.0569   残差均值 7.7%  max 30.1%
新：launchUs=21.260  opNs=13.252  elemNs=0.0597   残差均值 7.2%  max 35.9%
```
`n=4096/B=4096` 的 η 误差从 **+3.9% → −0.1%**。
把标定网格从 `B≥48` 扩到 `B∈{1,4,16}` 再拟合，得 `21.074/13.355/0.0592`、残差 7.0% ——
**与上式统计等价**，说明 `launchUs ≈ 21.1` 是真值而非小 batch 拟合伪影，故保留先者。

49 点网格上 η 有符号均值 **+7.7%**（第二批是 −2.6%）—— **不是拟合变差**：
小 batch 点的实测本身低于 `launchUs` 地板（如 n=128/B=16 实测 16.0 µs < 21 µs，
且 B=1 的 n=4096 跨轮在 34.7 ~ 41.4 µs 之间跳），是**测量噪声**而非模型偏差。
`select()` 对候选一律复测，η 只作排序信号，故不影响选型。

### 9.7 与 `阶段0-1` §9.4「radix-4 已否决」的冲突

`docs/阶段0-1-发现与结果.md` §9.4 当时的结论是「radix-4 +13%，否决」，理由两条：
1. `g(R) = 6(R−1) + 2R·log2R` 把**复数乘按 6 个实数算术**计入，蝶形却按**矢量调用**计入
   —— **单位不一致**。C2b 把复数乘做成 4 个矢量调用后，非平凡 radix-4 是 **28** 不是 34。
2. 更关键：它假设「融合把长度 `2h`/`4h` 的长算子拆成长度 `h` 的短算子」，
   而 §8.1 实测 `#op` 是固定发射开销、与长度无关 → 短算子多就是输。
   **这条只对 planar 级成立**（那里的矢量长度 = group 尺寸）；
   **平面级的矢量长度是 `rows = n/K`，与 `h` 无关**，融合既不缩短算子也不增加调用次数。

所以 §9.4 的否决对 **planar 级**仍然有效（本批实测 planar 融合 = +51 算子 / −12,288 元素，
净亏 −2.1%，已按计划**不做**），但对**平面级**不适用。两边并不矛盾，是两个不同的问题。

### 9.8 复现

```bash
bash scripts/build.sh all                      # 必须 all（kernel+host 同源 fft_k.hpp）
# 正确性（49 格）
for n in 64 128 256 512 1024 2048 4096; do for b in 1 4 16 64 256 1024 4096; do
  ./build/fft_check $n $b 10; done; done
# A/B（插队 3 轮 × min/50 reps；基线 .o 用 AB_FFT_O 指定，**不要为了 A/B 去重编译**）
python3 scripts/ab_test.py --base <基线.o> --cand build/fft_radix2.o \
    --points 4096x4096,64x4096,128x4 --reps 50 --rounds 3

# 手工单点（口径与 ab_test 相同）：
AB_FFT_O=<基线.o> ./build/fft_check 4096 4096 50   # 本文旧值 1564.7
AB_FFT_O=build/fft_radix2.o ./build/fft_check 4096 4096 50  # 1481.3

# 历史基线 .o：本文出现过的 overlap.o 是第二批结束时的临时产物、已不再保留，
# 用 baseline_o.sh 从对应提交重建（不切分支、不动工作区）：
scripts/baseline_o.sh <当时提交>          # -> build/baseline_<rev>_fft_radix2.o

python3 scripts/calib_eta.py    # 结果回写 estimate()
python3 scripts/matrix_test.py --reps 20 --out docs/matrix_test.md
```

### 9.9 下一步（按性价比）—— 已在 §10 逐项核查
1. **I/O 的 Stockham 化 / 布局合同**：`10n` 收尾元素里有 `idxB` 2n + `idxT` 2n =
   **4n = 全部 elems 的 7.5%**（≈ −5% 模型）。若输入侧能直接产 plane 布局、
   输出侧能直接吃 planar 布局，这两段 Gather 可整段消失；`idxB`+`idxT` 还占 8n 字节 UB
   （n=4096 = 32 KB），**腾出来刚好够 §8.1 的 MTE2 预取** —— 一个改动解锁两个收益。
   > **§10.1 否决**：三条路各有结构性障碍，且天花板只有 −5.7%。
2. **平面级 `w = −i` 特判**（`j2 = h/2` 时 `W² = −i`）：4 算子 → 2，
   K=16 时省 14 算子 ≈ **−2.2%**，约 20 行。
   > **§10.3 下修为 −0.16%**：`W` 本身从不等于 ±i，只有 `W² = −i` 在 `j2 = h/2` 出现一次/config。
3. **planar 级 radix-4**：只对 (16,32)、(64,128)、(1024,2048) 三对划算，(256,512) 必须留 radix-2，
   净 −2.1%，且算子数反升 17%。性价比低，**暂缓**。
   > **§10.3 下修为 −0.9%**（原估漏算了融合级自身 28 算子）。
4. **merge 的 repeat 分块**（`groups > 255` 时切片重放）：只在 n≥8192 有感，不在网格内。
5. **Cube 张量化**：见 §8.5 —— 批场景也救不了它，稠密 DFT 比 FFT 多 **2N/log2N = 683×** 算术，
   Cube 的 `Mmad` 下界 109.4 GMAC/s 代入仍是 628 ms vs 自研 1.48 ms（**慢 400×**）；
   批只会摊薄 F 的装载，而我们本来就在 UB 里复用 twiddle。

---

## 10. 第四轮：候选可行性核查 + 硬件探针

上一批收尾后按 §9.9 逐项核查。**结论：1–3 全部否决或收益下修，主战场已到结构性下界。**

### 10.1 §9.9-1（Stockham / 布局合同）：否决 —— 三个结构性障碍

想消掉 `idxB`（n）与 `idxT`（n）两段 Gather，顺带腾出 8n 字节 UB。三条路都走不通：

1. **`idxT` 消不掉**。K-plane 布局下 `h < K` 的级是**跨面**蝶形（向量长 `rows`、步长 `rows`），
   `h ≥ K` 的级是**面内**蝶形（面 `v` 的第 `u` 行 ↔ `u+m`，`m = h/K`）。两段在 `h = K` 处切换
   是该布局的唯一自然切点。过渡级 `h = K` 恒有 `m = 1`，面内折行的 repeat 步长 = `2m` 个 float =
   **8 字节**，不是 32 B 整数倍；而 `BinaryRepeatParams` 的 `*RepStride` 是 `uint8_t`、单位 32 B
   （见 `kernel_struct_binary.h`），表达不了 → 退化成 `K·groups` 次逐行循环 = 16,384 算子，
   而整个 planar 级才 136。把布局反过来（让 `h < K` 变面内）只是把长度-1 的退级从 `h = K`
   挪到 `h = 1`，问题原样存在。
2. **`idxB` 现场重建更贵**。`idxB[c·rows+a] = bitrev(a·K+c) = bitrev_K(c)·(n/K) + bitrev_L(a)`
   是「每面一次位反转 ramp」，核内重建 = 再做一次位反转置换，调用次数远超全核的 307 算子。
   纯转置那半是 stride-`K` 等差数列（可用 strided `DataCopy`），但位反转那半不是 ——
   只有改用 DIF（输入自然序、输出位反转）才能把它折进本来就存在的 `idxO` Gather，
   那样省 `n` 元素 ≈ **−1.4%**，还要重排整个算法的 twiddle 结构。
3. **完整 Stockham（每级置换写）更差**：`2n·log2n ≈ 98K` 次元素搬移 vs 现在 `4n = 16K`，**约 6×**；
   `Scatter` 的索引同样要 32 B 对齐，省不下来。

附带核实：输出侧 `idxO`（2n）也不能换成两条 `DataCopy` —— 32 B 块里混着 4 个复数的 re/im，
交错粒度是 4 字节；`Gather` 的索引 `LocalTensor<uint32_t>` 定死
（`kernel_operator_vec_gather_intf.h`），`uint16` 压不了（最大值 32,764 其实装得下，但 API 不给）。
**天花板**：三个索引全部归零也只省 `4n` = 16,384 × 0.0597 ns ≈ **−5.7%**，何况做不到。

### 10.2 硬件探针：fp32 单 repeat 硬上限 = 64 元素（**新增 8 例，结论决定性**）

`stride_probe`（`src/host/stride_probe.cpp` / `src/ascendc/stride_probe.cpp`）新增 mask 上限用例，
语义模型 = `dst[dstRep·8·r + j] = src0[...]·src1[...]`，未写位置保持 `i+1`：

| 用例 | mask | repeat | 结果 |
|---|---:|---:|---|
| `m64_r1`（控制组） | 64 | 1 | **PASS** |
| `m100_r1`（非 2 的幂） | 100 | 1 | FAIL，`badWritten=36@64` → 实际只写了 0..63 |
| `m128_r1` / `m256_r1` / `m512_r1` | 128/256/512 | 1 | FAIL，写对的元素数恒为 **64** |
| `pl128_g16` / `pl256_g8` / `pl512_g4`（与 planar 合并路径逐字同参） | 128/256/512 | 16/8/4 | FAIL，`badWritten` = repeat×64 → 每 repeat 各只有 64 个写对 |

两条结论：
* **`mask` 是「个数」不是位图** —— `mask=64` 能对上 64 个元素（若按 64 位位图解释只能命中 1 个），
  而 `mask=64` 时 `badUntouched=0` 说明 2×64 全写对。
* **有效值被硬件截到 64**（256 B ÷ 4 B = 64）。`nSlice = (h+63)>>6` 的 64 切片是**强制**的，
  §9.9 里任何「把长度提到 `h` 再折 group/j2」的方案（plane 折 `j2`、planar 去切片）都被这条卡死。

**推论**：planar 每级 8 算子 / `4n` 元素、8 级 = `log2n − logK` 全是下界；
136 算子里相对理论下界 72 的 **64 算子超额，全部来自这条上限，不可回收**。

### 10.3 §9.9-2、-3 的收益下修

* **`w = −i` 特判**：`W = e^{-iπ j2/(2h)}` 本身在主值分支上从不等于 ±i；只有 `W² = e^{-iπ j2/h}`
  在 `j2 = h/2` 出现一次。K=16 的两个平面级对里只有 `h=4` 的 `j2=2` 命中、`blocks=1`
  → 每批省 **2 算子 / 307 ≈ −0.16%**（原估把 `W` 也算进去，得 −2.2% 是错的）。**不做。**
* **planar 级 radix-4**：逐对重算 —— 融合级 **28** 算子 vs 现在两级共 **16**
  （merge 已把 group 维折进 repeat，不是 32）；元素每对只省 `n`（两条独立 `4n` vs 融合 `7n`）
  外加 2 次负号写。净 **+68 算子 / −17,744 元素** = 901 − 1,059 ns/批 × 86 ≈ **−0.9%**。
  复杂度高、风险大，**不做**（原估 −2.1% 漏算了融合级自身的 28 算子）。

### 10.4 当前状态与仍开着的口子

n=4096/B=4096 成本结构（η 口径）：**307 算子** = plane 164 / planar 136 / I-O 7；
**218,096 元素** = planar 62% / plane 19% / I-O 19%；ops 占 24%、elems 占 76%。
plane 每对 16/28 算子在无 `FusedMulAdds` 时已是最简（复数乘 4 算子是下界），
planar 8 级 × `4n` 是原地读写下界。**主战场已到结构性下界，剩余可挖项全部 ≤1%。**

按性价比还开着的口子：

1. **`rows ≤ 64` 时把 `m`（块维）折进 repeat**（只对 n ≤ 512、K=8 成立）：
   `(1,2)` 对 `blocks×16 → 16`；再把残级的 `j2` 折进 repeat（需把该 slot 的 twiddle
   从 `align8(h)` 填充到 `8h`，t0..t5 还能顺手从 `3·hmax+3·rows` 压到 `2·hmax+4·rows` 省 7 KB）
   → 合计 **−38 算子**（n=64：96 → 58），η 约 −20~30%。
   **能把但翻不转** stdlib 的 3 个负点（`n=256/512/1024 @ B=4096` = 0.76/0.85/0.81×，
   实测 200.4/266.2/421.5 µs vs 原生 152.5/225.8/343.1 µs）—— 估算只能到 0.91~0.97×。
2. §9.9-1/-2/-3：全部 ≤1%（见上）。
3. **Cube 张量化**：见 §8.5，慢 400×，永久关闭。

---

## 11. 第五轮：批折叠 A3~A5 + stride 探针 + K 择优 + 屏障/η 收尾

§10.4 的三个「结构性下界」结论仍然成立（plane 16/28、planar `4n`/级 都动不了），
**本轮的收益全部来自把「每个 batch 单独过一遍」改成「一次过 D 个 batch」**，
即在不动算子序列的前提下，把 Level-0 的 `repeat` 槽位从 batch 维抢出来。

### 11.1 批折叠（A3~A5）设计

**核心式**：`foldDFor(n, batch, nblk=48)` 给出折叠系数 D（`kFoldCap = 4`），
一组 Level-0 repeat 同时覆盖 **D 个连续 batch**：

```cpp
// include/butterfly/fft_k.hpp
uint32_t D = kFoldCap = 4;
if ((n >> 3) > 255) return 1;                                  // arRep uint8 上限
while (D > 1 && D * groupsMax > 255) D--;                      // repeat 字段 255 封顶
while (D > 1 && ((batch - 1) / D + 1) < nblk) D--;             // 并发不足才折（ceil 安全）
```

三道闸依次把 D 收窄：**结构上限 4 → `repeat` 字段 255 → 48 核并发**。
D 只看 `n` 和 `batch`，**纯静态**，所以 kernel / host / framework 三处共用同一函数、永远一致。

实际 D 表（nblk=48）：`n ≤ 1024` 时 `B < 192 → D=1`、`B = 144 → D=3`（并发闸）、
`B ≥ 192 → D=4`；`n ≥ 2048` → `D = 1`（`(n>>3) > 255` 闸）。

**组划分**必须按**组号取模**，不能 `b += D*nblk`（会跳过 batch）：

```cpp
for (gk = blk; gk * D < batch; gk += nblk) {
    uint32_t b = gk * D;
    uint32_t g = min(D, batch - b);     // 尾组不足 D 批
}
```

**planar 级**（`fft_radix2.cpp` ~`:400`）的 merge 条件加 `|| g > 1`：

```cpp
(h >= 8) && ((2h) % 8 == 0) && (2h/8 <= 255) && (h/8 <= 255) &&
(rep <= 255) && (nSlice <= groups || g > 1)      // rep = D * groupsH
```

merge 后算子数 `1 + 8·nSlice`（**与 D 无关**），否则逐 `d` × 逐 group `1 + 8·D·groupsH`；
元素数两边相同（同一批数据、只是指令打包不同）= `4·n·D + h`。
`useL0 = (g > 1) && ((n>>3) <= 255)`，整个 stage 循环在 `for (d < g)` 内并以
`if (useL0) break;` 收尾 —— **Level-0 路径没有冗余的 d 迭代**。

**ABI（第 8 个字，56 B args）**：`packArg = (planeK << 8) | foldD`。
低 8 位 = D，高 8 位 = 平面级 K（0 ⇒ kernel 自己用 `planeKFor(n)` 算）。
host `buildArgs()` / `fft_check` 的 `AB_FOLD_D` / `AB_PLANE_K` 都走这一份打包。

**配套改动**：
| 位置 | 改动 |
|---|---|
| `genInterleave(dst, D)` | 按 D 生成 `2n·D` 索引；`D=1` 退化式与旧表**逐字节等价** |
| `ubBytes(n, D)` | UB 预算含折叠；`D=4` 下 n=1024 用 139,392 B（70.8%） |
| `prepare(n, batch)` | 缓存键加 D —— 换 D 必须重生成 `idxO` |
| `estimate()` | `foldDFor` + `groups = ceil(B/D)` + `iters = ceil(groups/blocks)` |

### 11.2 stride 探针（A2-2）：Level-0 的 `mask ≤ 64` 是硬上限

`src/host/stride_probe.cpp` × `src/ascendc/stride_probe.cpp`，30 个用例：

**23 PASS / 7 预期 FAIL**（一键脚本把这组数字当闸门）。7 个 FAIL **全部**是 `mask > 64`，
不是缺陷 —— Level-0 的 mask 只有 6 bit，超了会被**静默截断**（探针报
`badWritten`/`badUntouched` 非 0）。这条结论反推了两件事：

1. planar 级的 `nSlice` 上限只能靠 `repeat ≤ 255` 与 `|| g>1` 放宽，**不能**靠加大 mask；
2. 任何「把 `m`（块维）折进 repeat」的设想（§10.4-1）都受同一个 64 约束。

### 11.3 K 择优（A5）：`kMin` 规则

`planeKFor(n)` 三步（`include/butterfly/fft_k.hpp`）：

```cpp
kNeed = (n + 63) >> 6;                              // rows = n/K ≤ 64 的下界 K
kMin  = (n <= 1024 && kNeed > 8) ? kNeed : 8;       // 小 n 才抬 kMin，大 n 一律从 8 起
for (K = kMin; K <= 32; K <<= 1)  取 ops 最少者
```

**K 表**：`64~512 → K8`、**`1024 → K16`**、`2048 → K8`、`4096 → K16`。
关键是 `kMin` **只看 n**（不看 B、不看运行时）⇒ kernel 与 host 静态一致。

* **隐患修复**：plane 级位反转那一处 `Sub(i1, i0, t0)` 应为 `Sub(i1, i1, t0)` —— A5 一并修掉。
* **测试钩子**：`AB_FOLD_D=<k>` 强制 D、`AB_PLANE_K=<8|16|32>` 强制 K。
  K<8 → AIV **507035**（planar 首级 `h=K<8` 打破 32 B 对齐），所以 `kMin` 下界是 8。
* 回归：`test_limits` **18/18**、91 点扫描 PASS、stride **23/7**。

### 11.4 A6：屏障份额量化 + prologue 屏障**保留**

`msprof --aic-metrics=PipeUtilization` 对 `fft_check 1024 4096 3`（改造前后各 3 次）：

| 指标 | 改前 | 改后 |
|---|---|---|
| Task Duration | 297.9 / 299.1 / 299.5 µs | 298.1 / 299.3 / 298.8 µs |
| `aiv_vec_ratio` | ≈0.82 | ≈0.825 |
| `aiv_scalar_ratio` | ≈0.168 | ≈0.168 |
| `mte2` / `mte3` | ≈0.09 / ≈0.08 | ≈0.09 / ≈0.08 |

`vec + scalar ≈ 0.99` ⇒ **空闲/屏障份额 ≈1%，屏障没有可挖余量**。
试着把 prologue（`fft_radix2.cpp:123`）那次 `PipeBarrier<PIPE_ALL>` 并进每组的取数屏障，
收益 298.0 → 298.1 µs（**< 噪声**），**已回退**：

> batch 很小（如 1）时大量 block 不进组循环、直接 return，prologue 屏障是它们**唯一**的 UB
> 排空点。撤掉会让上一次 launch 在途的 UB 写跨任务污染下一次 launch。
> 「省 1 次 drain」vs「跨 launch UB 竞争」，后者风险更高 —— **该屏障不可省**，
> 原因已写进 `fft_radix2.cpp:118` 的注释。

三处 `PipeBarrier<PIPE_ALL>` 现状：`:123` prologue、`:147` 每组取数后、`:459` 每组 Gather 后 / 写回前。

### 11.5 A7：η 重标定（建模折叠 + 按 1/y 加权）

旧式 `η = launchUs + ceil(B/48)·(opNs·#op + elemNs·#elem)` **不含 D**：
折叠点把 `#op` 少算 D 倍 ⇒ `η/实测−1` 高达 **+67% ~ +104%**（`n=64/B=4096` 最坏 +90.7%）。

新式（`scripts/calib_eta.py` 与 `src/framework/butterfly.cpp::estimate()` 同一份公式）：

```
D      = foldDFor(n, B, 48)
groups = ceil(B / D)
blocks = min(48, groups)
iters  = ceil(groups / blocks)                 # 墙钟取最慢核
η      = launchUs + iters · (opNs · #op + elemNs · #elem)      # #op/#elem = counts(n, D)
```

**标定**：7 n × 7 B = 49 点，reps=20、每点 3 轮取 **min of means**（剔本机负载尖峰），
**按 `1/y` 加权**最小二乘（闸门是相对误差，未加权会被 `n=4096/B=4096` 的 1498 µs 主导）：

```
launchUs = 16.109406   opNs = 17.07683 ns   elemNs = 0.05016 ns
```

**独立验证**（系数拟合自标定集，验证集是另一轮矩阵）：`|η/实测−1|` **mean 7.7%、
median 6.7%、max 26.6%**，`>15%` 的 **6/49**。这 6 点全部满足 `mean/min ≥ 1.3`，
即**测量本身的抖动已大于模型误差**；其中 `n=128,256 @ B=256` 还多一层口径差 ——
`test_framework` 选中的是 `udCore=24`（`iters=3`），而 `fft_check` 恒按 48 块测。

> 闸门若按 `|η/实测−1| ≤ 15%` 卡「全 49 点」，本机 `loadavg 20~30` 下**测不准**：
> 同一 `(n,B)` 三点重测 `mean` 本身摆 10~30%（`n=64/B=16` 见过 14.0 µs，比 `launchUs` 还低）。
> 一键脚本因此按 **`>15%` 的点 ≤8 且均值 ≤10%** 判 η（实测 6 / 7.7%）。

### 11.6 现状与复现

`matrix_test.py --rounds 3`（每点 3 轮逐点取 min-of-means）：

| 项 | 结果 |
|---|---|
| 正确性 | **49/49**（worst `maxRel` 2.41e-7） |
| vs CANN 原生复数 FFT | **49:0**（最好 6.86× @ 128/4，最紧 1.04× @ 1024/4096） |
| vs `aclRfft1D` | **49/49** |
| vs 自研 v1 | **49/49**（中位 11.2×，最好 38.8×） |
| vs numpy / torch(CPU) | **36/49 / 38/49**；`B≥1024` 各 **14/14** |
| `n=4096/B=4096` | 自研 min **1,481.4 µs**（基线 1,483.2，**不回退**）；比值 1.60× → **1.64×** |
| η | mean 7.7%、6/49 >15% |

§10.4-1 里「**估算只能到 0.91~0.97×**、翻不转那 3 个负点」的问题，
**被批折叠直接解决了**（那 3 点现为 1.42× / 1.34× / 1.04×），
`rows ≤ 64` 那个口子仍开着、但优先级大幅下降。

```bash
# 一键：编译 → 门禁（limits / framework / stride / fft_check）→ 49 点矩阵 → 结论
scripts/one_click_test.sh
scripts/one_click_test.sh --quick          # 9 点抽样矩阵，约 5 min
scripts/one_click_test.sh --rounds 5       # 每点 5 轮，进一步压噪声

# 单点复核（D / K 可强制）
AB_FOLD_D=4 AB_FFT_O=build/fft_radix2.o ./build/fft_check 64 4096 30
# η 标定（系数回写 estimate()）
python3 scripts/calib_eta.py
```
