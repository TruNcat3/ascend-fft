# Trace / Profile 诊断：小尺寸 vs GPU、大尺寸 vs 自研原生

> 手段：`msprof`（CANN 9.0.0）打 `fft_check` 与 `torch.fft.fft(torch_npu)` 两份 profile，
> 取 `AscendTask.duration`（设备侧任务时长）、`op_summary`（每 op 管线占用率）、
> `api_statistic`（host 侧 API 耗时）。
>
> **复现脚本**：[`scripts/profile_test.sh`](../scripts/profile_test.sh)（采集+汇总）
> · [`scripts/sum_prof.py`](../scripts/sum_prof.py)（读已有 profile）
> · [`scripts/native_fft.py`](../scripts/native_fft.py) / [`scripts/time_native.py`](../scripts/time_native.py)（原生用例）
> · [`scripts/repro.sh --doc trace`](../scripts/repro.sh)（只列出 profile 类实验 `profile` / `profile-sum`；
>   本文其余实验见各节给出的具体命令）
> 全部路径由 [`scripts/env.sh`](../scripts/env.sh) 探测：`$AB_CANN` / `$AB_MSPROF` / `$AB_PY` / `$AB_WORK`。

---

## 0. 复现

```bash
source scripts/env.sh                 # 导出 $AB_MSPROF / $AB_PY / $AB_WORK

# 一键：5 个用例采集 + 汇总（最省事，等价于下面的手工命令）
scripts/profile_test.sh
scripts/profile_test.sh --only ours_b1,ours_b64,ours_b4k,nat
scripts/profile_test.sh --out results/profiles/p_a6_after

# 等价的手工命令（本文档里所有 msprof 调用的写法）：
# 自研 kernel，三个 batch
"$AB_MSPROF" --output="$AB_WORK/p_b1"  --task-time=on ./build/fft_check 4096 1    10
"$AB_MSPROF" --output="$AB_WORK/p_b64" --task-time=on ./build/fft_check 4096 64   10
"$AB_MSPROF" --output="$AB_WORK/p_b4k" --task-time=on --aic-metrics=PipeUtilization \
                             ./build/fft_check 4096 4096 10

# CANN 原生（torch_npu）—— scripts/native_fft.py 即 warmup 3 + 10 次 torch.fft.fft(4096x4096 c64)
"$AB_MSPROF" --output="$AB_WORK/p_nat" --task-time=on --aic-metrics=PipeUtilization \
                             "$AB_PY" scripts/native_fft.py --n 4096 --b 4096 --reps 10
```

取数（`scripts/sum_prof.py` 一次出齐，下面的 python 片段是它的数据来源）：
```bash
python3 scripts/sum_prof.py "$AB_WORK/p_b4k"           # 单份详表
python3 scripts/sum_prof.py "$AB_WORK/p_a6_before" \
                           "$AB_WORK/p_a6_after"        # 前后对照
python3 scripts/sum_prof.py <dir> --per 13 --drop 13           # 原生（每迭代 13 个 op）

python3 -c "import sqlite3,sys;c=sqlite3.connect('$P/device_0/sqlite/ascend_task.db');
[print(r) for r in c.execute('select host_task_type,duration from AscendTask')]"
# 管线占用率在 mindstudio_profiler_output/op_summary_*.csv 的
#   aiv_vec_ratio / aiv_scalar_ratio / aiv_mte2_ratio / aiv_mte3_ratio / cube_utilization(%)
```

---

## 1. 三个结论

| # | 问题 | 根因（profile 实测） | 证据 |
|---|:---|:---|:---|
| **A** | **小尺寸慢于 GPU** | ① **并行度**：`blocks = min(batch,48)`，**一个 transform 从不切分** → B=1 只有 **1/48 个 AIV 核**在干活；② **host** 每次 launch+sync ~13–28 µs（CUDA ~2–4 µs） | §2 |
| **B** | **大尺寸慢于自研原生** | 原生 = **3×BatchMatMul（跑在矩阵单元 cube 上，cube 利用率 83%）+ 2×Transpose**，只 5 次提交；我们是**单个纯矢量 kernel，cube 利用率 0%** | §3 |
| **B′** | **原生在失分点（n ≤ 1024）靠什么赢** | **不是 cube**：Transpose(AIV) 占设备时间 **55~84%**、cube 仅 16~45%；且原生 host 侧占墙钟 **20~63%**（n=64→1024 递减，mean 口径，见 §6.3.3；我们单 launch 省 ~45 µs） | §6.3 |
| **C** | **我们自己的瓶颈** | **Vector 管线占空 78.6%**、Scalar 19.6%、**MTE2 仅 5.5% / MTE3 4.3%**（11 次采样的最不利单点；均值 81.5% / 20.3% / 4.0% / 2.7% 见 §4.1）→ **不是内存墙，是矢量发射墙**；同时 **cube 单元 100% 闲置** | §4 |

---

## 2. 问题 A：小尺寸 vs GPU（B = 1）

### 2.1 n=4096 / B=1 拆账

| 项 | 我们 | cuFFT @ V100 | 说明 |
|---|---:|---:|:---|
| **设备侧 kernel**（`AscendTask.duration` min） | **38.7 µs** | 11.99 µs | 自研慢 **3.2×** |
| host 侧（墙钟 − 设备） | **27.6 µs** | ~2–4 µs | `aclrtLaunchKernelWithHostArgs` min 8.1 µs / 非首调均值 12.1 µs；`LaunchKernelV2` min 4.25 µs |
| **合计（矩阵测试 mean）** | **66.3 µs** | **11.99 µs** | 慢 **5.5×** |

> 即：**5.5× 的差距里，设备侧占 3.2×，host 启动/回同步占 27.6 µs（CUDA 侧只需 ~2–4 µs，多出 ~24 µs）。**
> GPU 那边是 kernel 时间；我们这边 66.3 µs 里有 27.6 µs 根本没碰到芯片。

### 2.2 设备侧为什么 38.7 µs：只有 1 个核在干活

`src/host/fft_check.cpp:168`：

```cpp
uint32_t blocks = batch<48u ? batch : 48u;     // 并行度 = min(batch, 48)
```

kernel 内层是批折叠后的 `for (gk = blk; gk*D < batch; gk += nblk)`（`fft_radix2.cpp:136`）——
**并行维只有 batch，单个 transform 内部（12 个 stage / 4096 点）完全串行**。

实测三方印证（设备侧任务时长）：

| batch | blocks | 最大批数/block | **设备时长** | 预测 = 最大批数 × 38 µs | AIV 核占用 |
|---:|---:|---:|---:|---:|:---|
| 1 | 1 | 1 | **38.7 µs** | 1 × 38 = 38 ✓ | **1/48 = 2%** |
| 64 | 48 | 2（16 核做 2 批、32 核做 1 批） | **76.1 µs** | 2 × 38 = 76 ✓ | 48/48，但 2/3 提前退出 → 有效 **67%** |
| 4096 | 48 | 86 | **3,252.0 µs** | 86 × 37.8 = 3251 ✓ | 48/48，有效 ~99% |

**模型完全闭合**：`device_time ≈ 该核分配到的最大批数 × 38 µs/批`（n=4096）。

这就是 §10.2 `#op` 模型的设备侧实证：
`2012 op × 12 ns + 270336 elem × 0.05 ns = 24.1 + 13.5 = 37.6 µs` ≈ 实测 37.8 µs。

### 2.3 GPU 为什么没有这个问题

cuFFT / cuButterfly 都会把**一个 transform 沿数据维切开**（cuButterfly 术语 `Ud`，见其 README 的
`data work D = Ud × Td`），所以 batch=1 也能喂满全部 SM。
我们的设计空间里有 `ud_core` 这根轴，但 `makePlan` 只实现了 R=2 + 按 batch 切块，
**`Ud` / `Ts` / `D` 三根轴全部枚举了却没落地**（§11.2：864 个候选里 816 个 Infeasible，剩下 48 个全等价）。

---

## 3. 问题 B：大尺寸 vs 自研原生（n=4096, B=4096）

### 3.1 原生算子的真实构成（`op_summary`）

一次 `torch.fft.fft(4096, 4096, complex64)` = **5 个 kernel**（13 次调用 / 39+26 个 op）：

| op | 类型 | 核 | 次数/次FFT | 设备时长/次FFT | 占比 |
|:---|:---|:---:|:---:|---:|---:|
| `aclnnMatmul_..._BatchMatMulV2` | **AI_CORE（矩阵单元）** | 3 | 3 | **1,455 µs** | **62%** |
| `aclnnInplaceCopy_Transpose` | AI_VECTOR_CORE | 2 | 2 | **879 µs** | **37%** |
| **合计** | | **5** | **5** | **2,334 µs** | 100% |

对照：矩阵测试同点 **原生墙钟 2,357 µs**，**自研设备侧 3,252 µs / 墙钟 3,288 µs** → **自研慢 1.39×**。

> **结构差异**：`4096 = 16³` → 原生走**三级 16 点 DFT = 3 个批量复矩阵乘 + 2 次转置**，
> 把蝶形算术搬进了**矩阵单元（Cube）**；我们把 12 级 radix-2 全部放在**矢量单元（AIV）**上。

### 3.2 原生自己也慢在哪（这说明什么）

原生的 37% 花在 **Transpose** 上，纯搬运；matmul 侧 `aic_mte2_ratio = 0.973`
（UB→Cube 的喂数管线 97% 忙）而 `aic_mac_ratio = 0.204`（乘加只忙 20%）
——**原生也是"喂不进去"，只是喂的是矩阵单元，不是矢量单元**。
所以"大尺寸比原生慢 1.39×"里，原生并不是一个理想基线，它自己还有 ~37% 的纯转置浪费。

---

## 4. 问题 C：我们自己的管线占用率（`--aic-metrics=PipeUtilization`）

### 4.1 三份 profile 并排

| 指标 | **自研 `kfft_fwd`** | 原生 `BatchMatMulV2` | 原生 `Transpose` |
|:---|---:|---:|---:|
| 执行单元 | AIV 矢量 | **AIC 矩阵** | AIV 矢量 |
| **`aiv_vec_ratio`（矢量管线占空）** | **0.815**（11 次采样 0.786–0.819） | — | 0.327 |
| `aiv_scalar_ratio` | **0.203**（0.196–0.204） | — | 0.193 |
| **`aiv_mte2_ratio`（GM→UB 取数）** | **0.040**（0.037–0.055） | 0.973 | 0.244 |
| **`aiv_mte3_ratio`（UB→GM 回写）** | **0.027**（0.025–0.043） | — | **0.418** |
| `aic_mac_ratio`（矩阵乘加） | — | 0.204 | — |
| `aic_fixpipe_ratio` | — | 0.228 | — |
| **`cube_utilization(%)`** | **0.000** | **83.0**（68.0–98.3） | — |

### 4.2 这张表说明三件事

1. **我们是矢量发射受限，不是内存受限。**
   Vector 管线 **81.5%** 忙（11 次采样 78.6–81.9%），而负责 GM 搬运的 MTE2/MTE3 只忙 **4.0% / 2.7%**。
   HBM 完全喂得动 —— 之前 §10.2 用 η 权重推的"63% 是 `#op`"，现在由硬件计数器直接证实。
   **结论：加大带宽、加双缓冲、换 layout 都救不了这一段；只有减少每次发射的算子数（`#op`）有效。**

2. **Scalar 管线 20.3% 忙，比预期高。**
   这是批循环内的地址/索引/循环控制，占到 1/5，是一个**没有在 η 模型里单独建模**的项。
   `#op` 只算了矢量算子，Scalar 那份是白捡的优化空间（例如把内层批循环展开、
   预计算 UB 步进、减少 `SetValue` 式标量操作）。

3. **Cube 矩阵单元 100% 闲置。**
   原生把 62% 的时间花在 cube 利用率 83% 的 matmul 上 —— 同一张卡上有一个我们完全没用的算力单元。
   这是"大尺寸 vs 原生"最大且最直接的结构性差异。

---

## 5. 对"学 cuButterfly 的搜索策略 / 开子图驻留"这个假设的评估

**方向对，但它只解决 A，不解决 B 和 C 的一半。**

cuButterfly README 里确实有这些（与我们设计空间一一对应）：

| cuButterfly | 我们的对应轴 | 实现状态 |
|:---|:---|:---|
| `residence`（驻留） | `local_exchange` / `coefficient_residency` | **枚举了，未实现**（shuffle 全部判 Infeasible，因为 SoC 无 SIMT） |
| `D = Ud × Td`（数据维空间×时间） | `ud_core` / `ts` | **枚举了，未实现**（`blocks=min(batch,48)` 是纯 `Td`） |
| `S = Us × Ts`（stage 维空间×时间） | `ts` / `fusion_level` | **枚举了，未实现** |
| Runtime Selector + `confirmed-median` 实测格 | `select()` top-K 实测 | **已实现**（§11.4） |
| 安装期跑标定写设备模型表 | `calib_eta.py` | **已实现**（`阶段0-1` §8.5） |

所以：**选型闭环我们不缺，缺的是 `Ud/Ts/Residence` 三种"物理映射"的 kernel 实现**。

**但要注意 cuButterfly 自己的结果**（README「What Is Frozen In v0.8」表，V100 同机）：

| shape | cuButterfly v0.8 | cuFFT | cuButterfly/cuFFT |
|:---|---:|---:|---:|
| FP32 logN18, batch 2 | 0.029819 ms | 0.031683 ms | **1.06× 更快** |
| FP32 logN18, batch 64 | 0.769270 ms | 0.707174 ms | **0.92× 更慢** |
| FP32 logN20, batch 2 | 0.119972 ms | 0.121467 ms | **1.01× 更快** |
| FP32 logN20, batch 16 | 0.767611 ms | 0.724091 ms | **0.94× 更慢** |

README 原文：*"FFT reaches cuFFT parity or better for selected shapes,
**while saturated long-batch FFT points remain below cuFFT**."*

> **即：饱和大 batch 是连 cuButterfly 也没打过的点。**
> 所以"开子图驻留就该很快"在**小/中 batch（并行度不足）**成立，
> 在**大 batch（我们 vs cuFFT 的 10×）**不成立 —— 那是另一堵墙。

---

## 6. 行动项（按性价比排序）

| # | 行动 | 打哪个问题 | 预期 | 成本 |
|---|:---|:---|:---|:---|
| 1 | **`Ud>1`：单 transform 内部按 stage/数据维切到多核** | **A** | B=1 时 1/48 → 48 核，设备侧 38.7 µs 有望降到 ~2–4 µs 量级（受 `#op` 下限约束） | 高（新 kernel + 归约合并） |
| 2 | **削 Scalar 管线 19.6%**：批循环展开 / 预计算步进 | **C** | `#op` 模型里没算的那 19.6% 归零 → 直接降 `device_time` | 低 |
| 3 | **`D>1` 批折叠**：一次矢量算子覆盖 D 批，`#op` ÷ D | **C + B** | 唯一能动 Vector 占空（78.6%，11 次采样的最不利单点；均值 81.5% 见 §4.1）的手段 | 中 |
| 4 | **把部分 stage 搬到 Cube（16 点 DFT = 16×16 复 matmul）** | **B** | 对齐原生的 83% cube 利用率 | 高（复数 matmul 分解） |
| 5 | host 侧压 launch+sync 到 <5 µs（批内合并、减少同步） | **A** | 砍掉 B≤64 区间的 ~13–28 µs 固定开销 | 中 |

**顺序建议：先 2（便宜且直接），再 3（唯一的大杠杆），再 1（救小尺寸），4/5 视收益。**

---

## 6.1 第二批优化结果（2026-09-30 更新）

上表 5 项**一项都还没动**——第二批走的是另一条更便宜的路：**修正模型与选型**，
外加两个局部算子优化。详细过程见 `docs/性能优化-C2b与K择优.md`。

| 改动 | n=4096/B=4096 | 说明 |
|---|---:|:---|
| 基线（C1+C2） | 2,013 µs | — |
| + planar 复数乘 6→4（`MulAddDst`） | 1,975 µs（−1.9%） | 负 twiddle 复用本级空闲的 `t2`，零额外 UB |
| + **K 择优公式修正** | 1,677 µs（−15.1%） | `planeKFor` 用的还是 C1 之前的算子数公式，K=32→16，`#op` 2012→401 |
| + **merge 条件放宽** | 1,638 µs（−2.3%） | 多余的 `h<=255` 挡掉了 h=256；按 `pSt/vSt` uint8 字段判 + `nSlice<=groups` 收益判 |
| + 去掉 4 个矢量管内 `PIPE_ALL` | 1,633 µs（−0.2%） | 实测≈0，但语义更干净（960 次压力 0 FAIL） |
| + **批间 MTE3⊗MTE2 重叠** | **1,573 µs（−4.5%）** | 删掉 `DataCopy(gout)` 之后的屏障，输出写与下一批输入读并发 |
| **累计** | **1,573 µs（−22%）** | `fft_check 4096 4096 20`：mean 1577.8 / min 1566.8 µs |

η 模型同步重标定：`launchUs=18.040, opNs=14.868, elemNs=0.0569`，
**49 点残差均值从 +60%~+111% 高估降到 −2.6%**（max \|·\| 29%）。

网格比分（7×7=49 点，reps=20）：**44 胜 5 负**（此前 40:9；**A4~A7 批折叠后现状 49:0**，见 `docs/matrix_test_a7.md`）；
vs 原生 @ n=4096/B=4096 从 1.21× → **1.53×**。

msprof 最新（n=4096/B=4096，`--aic-metrics=PipeUtilization`）：
**`aiv_vec_ratio 0.822`（墙）、`aiv_scalar_ratio 0.140`、`aiv_mte2 0.114`、`aiv_mte3 0.093`、`cube_utilization 0`**。

**对上面行动项的影响**：
- `#op` 已从 2012 降到 369，占模型耗时 29% → **`D>1` 批折叠（原 #3）性价比下降，从榜单撤下**；
- 屏障已清干净，`#2 削 Scalar` 的上限约 −14%，不再是"白捡"；
- **`#4 搬到 Cube` 仍需 split-core，量级大但对 75%+ 的矢量占空帮助有限**；
- ~~**新增首推：radix-4**（12 stage → 6 stage，元素搬运 −46%，预计 −30~40%）~~ ——
  该估计已作废，见 §6.2。

---

## 6.2 第三批优化结果：平面级 radix-4（2026-09-30 更新）

详细过程见 `docs/性能优化-C2b与K择优.md` §9。**先纠正一个数字**：上面说的
「radix-4 元素搬运 −46%、预计 −30~40%」是**错的** —— 每个 stage 本来就是读一遍写一遍，
两级合并成一级后仍然读一遍写一遍，**朴素融合的元素流量是 0% 变化**（中间量还要落 UB，反增 2n）。

真 radix-4 赚的是**算子调用数**，而且只在平面级赚：

| 改动 | n=4096/B=4096 | 说明 |
|---|---:|:---|
| 基线（第二批） | 1,564.7 µs（A/B 中位） | 第二批结束时的 `fft_radix2.o`（临时产物已不保留，`scripts/baseline_o.sh <当时提交>` 可重建） |
| + **平面级 radix-4 融合** | **1,481.3 µs（−5.3%）** | stage `(h,2h)` 代数消元成 4 点变换 |
| 平面级算子数 | 226 → **164**（−27.4%） | 平凡 16、非平凡 28（基线 26 / 32） |
| 元素流量 | 233,968 → **218,096**（−6.8%） | 模型 Δ = −9.6%，实测 −5.3% |

η 再标定：`launchUs=21.260, opNs=13.252, elemNs=0.0597`（旧 18.040 / 14.868 / 0.0569），
标定网格残差均值 7.7% → **7.2%**，**n=4096/B=4096 的 η 误差 +3.9% → −0.1%**。

网格比分 7×7=49：**49/49 PASS、44 胜 5 负**（当时不变；**A4~A7 之后 49:0**）；vs 原生 @ n=4096/B=4096 **1.53× → 1.72×**。

msprof（n=4096/B=4096）：
**`aiv_vec_ratio 0.853`（墙，↑）、`aiv_scalar_ratio 0.075`（0.140 → 腰斩）、
`aiv_mte2 0.096`、`aiv_mte3 0.069`、`cube_utilization 0`**。

**对上面行动项的影响**：
- **`#2 削 Scalar` 已经做掉一大半**：平面级从 32 次蝶形降到 8 次，
  `GetValue`/浮点比较随之减少，scalar 0.140 → 0.075；剩余上限很小；
- **`#4 搬到 Cube` 仍然是负 ROI**：见 `docs/性能优化-C2b与K择优.md` §8 第 5 条
  （683× / 628 ms 的推导在该文 §9.9 第 5 条）与 `Cube张量化探针.md` ——
  稠密 DFT 比 FFT 多 `2N/log2N = 683×` 算术，`Mmad` 下界 109.4 GMAC/s 代入
  是 628 ms vs 自研 1.48 ms（**慢 400×**），**批也救不了**（批只摊薄 F 的装载，
  而我们本来就在 UB 里复用 twiddle；算术比与 batch 无关）；
- **新的首推：I/O 的 Stockham 化 / 布局合同** —— `10n` 收尾元素里 `idxB` 2n + `idxT` 2n
  = **4n = 全部 elems 的 7.5%（模型 ≈ −5%）**，且这两张索引占 8n 字节 UB
  （n=4096 = 32 KB），**腾出来正好够 `docs/性能优化-C2b与K择优.md` §8 第 1 条的 MTE2 预取（≈ −5%）** —— 一个改动解锁两个收益；
- 平面级 `w = −i` 特判（`j2 = h/2`，4 算子 → 2）约 **−2.2%**、20 行，是下一个便宜项；
- planar 级 radix-4 只净 **−2.1%** 且算子数反升 17%，**暂缓**。

---

## 6.3 A2-1：原生在 4 个失分点的真实构成（2026-10-01，n=64/256/512/1024）

**结论先行：原生在 n ≤ 1024 的失分点上不是 cube 主导，而是 Transpose（AIV）主导** ——
Transpose 占设备时间 **55~84%**，cube 只占 **16~45%**。§3 那条"原生 = 3×BatchMatMul、cube 83%"
的观察是 **n=4096 专属**，搬到小 n 不成立；此前"`DFT_BORDER_VALUE=4096` ⇒ 小 n 走 DFT 矩阵路径"
的猜测**被证实**（每迭代确实 1~2 个 BatchMatMul），但**大头在配套的两次转置**，不在矩阵。

### 6.3.1 每迭代的任务构成

`--task-time=on --aic-metrics=PipeUtilization`，每迭代**全部串行**（上一任务 end == 下一任务 start）。

| n | 任务/迭代 | 构成（mean µs） | 设备侧合计 | Transpose | MM | cube% |
|---:|---:|:---|---:|---:|---:|---:|
| 64 | 3 | TP `4096,64,2` 19.65 + **MM `1,128,128;1,128,4096` 7.85** + TP `64,2,4096` 20.49 | **48.0** | 40.1 (**84%**) | 7.85 (16%) | 72.6 |
| 256 | 4 | TP `4096,256,2` 29.97 + MM `1,64,64;1,64,32768` 9.77 + MM `32,16,16;32,16,4096` 15.84 + TP `32,8,2,4096` 22.46 | **78.0** | 52.4 (67%) | 25.6 (33%) | 78.8 |
| 512 | 4 | TP `4096,512,2` 48.38 + MM `1,32,32;1,32,131072` 49.11 + MM `16,64,64;16,64,4096` 15.33 + TP `16,32,2,4096` 35.62 | **148.5** | 84.0 (57%) | 64.4 (43%) | 91.6 |
| 1024 | 4 | TP `4096,1024,2` 87.70 + MM `1,64,64;1,64,131072` 93.90 + MM `32,64,64;32,64,4096` 24.30 + TP `32,32,2,4096` 58.85 | **264.7** | 146.6 (55%) | 118.2 (45%) | 90.6 |

`blocks = 48`（Transpose，全部 AIV）/ `blocks = 22~24`（MM，cube）。
n=64 只有 **1** 个 matmul（DFT 实矩阵 128×128），n≥256 是 2 个。

### 6.3.2 管线占用率：Transpose 是 MTE2 主导（读 GM）

对全部 Transpose 任务取均值：

| n | `aiv_vec` | `aiv_scalar` | **`aiv_mte2`** | `aiv_mte3` | cube% |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.111 | 0.189 | **0.464** | 0.106 | 0 |
| 256 | 0.262 | 0.334 | **0.404** | 0.192 | 0 |
| 512 | 0.321 | 0.340 | **0.362** | 0.235 | 0 |
| 1024 | 0.373 | 0.359 | **0.321** | 0.251 | 0 |

→ **原生在失分点上是"搬"的墙，不是"算"的墙**；与 §4.2 我们的画像正好相反
（我们 Vector 78.6% / MTE2 5.5% / MTE3 4.3% → 矢量发射墙；三者同为 11 次采样的最不利单点，
均值 81.5% / 4.0% / 2.7% 见 §4.1）。

### 6.3.3 墙钟拆账：原生 host 侧占 20~63%（n 越大越被摊薄）

| n | 设备侧/迭代（msprof） | 墙钟·基线 `matrix_test_raw.md` | 墙钟·本次实测 mean / min | host ≈ |
|---:|---:|---:|---:|---:|
| 64 | 48.0 | 107.7 | 129.1 / 101.5 | 60 ~ 81 |
| 256 | 78.0 | 152.5 | 136.7 / 130.5 | 59 |
| 512 | 148.5 | 225.8 | 211.8 / 205.5 | 63 |
| 1024 | 264.7 | 343.1 | 330.5 / 323.7 | 66 |

host 占比（= host / 墙钟，按上表 mean 口径）：**63% / 43% / 30% / 20%**（n=64/256/512/1024）；
min 口径为 **53% / 40% / 28% / 18%** —— 即全区间 **20~63%**，大 n 被设备时间摊薄。

原生每迭代提交 **3~4 次 aclnn**；我们是**单次 launch**，η 模型 `launchUs = 21.26 µs`
→ 在 n=1024 上比原生省约 **45 µs host**。

### 6.3.4 对批折叠计划的三点修正

1. **cube 不是失分原因** → 不需要"用 cube 打回去"；计划里 cube 作为"最后手段"的定位保留，但优先级下调。
2. **原生是搬运墙** → 我们的杠杆仍是**减少矢量算子数**（批折叠 ÷D），因为我们的瓶颈在 Vector 发射，不在带宽。
3. **n=1024 的翻盘缓冲来自 host，不是设备侧**：
   - D=4 → η≈315 = host 21.3 + 设备 **294**，原生设备 265 → **设备侧仍输 11%**，靠 host 45 µs 翻盘
   - D=8 → η≈295 = host 21.3 + 设备 **274** → 设备侧只输 3%，墙钟 **1.16×**
   - **设备侧地板**：ops→0 时 η ≈ 21.3 + 86 × 2857.5 ns ≈ **267 µs**（elems 项不可再降）
     → n=1024 墙钟上限 ≈ 343.1 / 267 = **1.28×**；D=8 的 1.16× 已接近该地板

复现：
```bash
source scripts/env.sh

# 4 个原生失分点各采一份 profile（等价于 scripts/profile_test.sh --only nat 的参数化版）
for N in 64 256 512 1024; do
  "$AB_MSPROF" --output="$AB_WORK/p_nat$N" --task-time=on --aic-metrics=PipeUtilization \
    "$AB_PY" scripts/native_fft.py --n $N --b 4096 --reps 10
done
"$AB_PY" scripts/time_native.py --ns 64,256,512,1024 --bs 4096 --reps 10   # 墙钟对照
# 汇总：python3 scripts/sum_prof.py $AB_WORK/p_nat64 ... （或 scripts/profile_test.sh 自动出）
python3 scripts/sum_prof.py "$AB_WORK"/p_nat{64,256,512,1024} --per 13 --drop 13
```

---

## 6.4 A6：屏障份额量化 —— 屏障只占 ~1%，prologue 屏障**保留**（2026-10-03）

`msprof --aic-metrics=PipeUtilization` 打 `fft_check 1024 4096 3`，
把 `fft_radix2.cpp:123` 那次 prologue `PipeBarrier<PIPE_ALL>` 并进每组取数屏障前后各跑 3 次：

| 指标 | 并入前（3 次） | 并入后（3 次） |
|:---|---:|---:|
| Task Duration | 297.9 / 299.1 / 299.5 µs | 298.1 / 299.3 / 298.8 µs |
| `aiv_vec_ratio` | ≈0.820 | ≈0.825 |
| `aiv_scalar_ratio` | ≈0.168 | ≈0.168 |
| `aiv_mte2_ratio` / `aiv_mte3_ratio` | ≈0.09 / ≈0.08 | ≈0.09 / ≈0.08 |

**`vec + scalar ≈ 0.99` ⇒ 空闲 + 屏障份额 ≈1%。**
这不是"还有 1% 可省"的信号 —— 屏障本身要占的时间已经低于本轮的测量噪声
（298.0 → 298.1 µs 反向波动）。

**处置：prologue 屏障保留，改动已回退。** 理由写进 `src/ascendc/fft_radix2.cpp:118`：

> batch 很小（如 1）时大量 block 不进组循环、直接 return，prologue 屏障是它们**唯一**的
> UB 排空点。撤掉会让上一次 launch 在途的 UB 写跨任务污染下一次 launch。
> 「省 1 次 drain」与「跨 launch UB 竞争」相比，后者风险更高。

三处 `PipeBarrier<PIPE_ALL>` 现状：`:123` prologue（**必须留**）、
`:147` 每组 `DataCopy(plan)` 后、`:459` 每组 Gather 后 / 写回前。
内核级矢量管内的 `PipeBarrier` 早在 `docs/性能优化-C2b与K择优.md` §5 已去掉（实测≈0）。

> 与 §4.1 的关系：那份表是**更早的 11 次采样**（`prof_pipe`），`vec=0.815 / scalar=0.203`；
> 本轮 3 次采样是 `vec≈0.82 / scalar≈0.168`。两者的 `vec+scalar` 都≈1.0，
> 结论一致 —— **发射饱和，屏障不是瓶颈**。

**采集命令与 profile 目录**（`scripts/profile_test.sh` 没有这两份用例，须手工采）：
```bash
source scripts/env.sh
# 下面两条各连跑 3 次，即表中的「并入前 / 并入后」两组
"$AB_MSPROF" --output="$AB_WORK/p_a6_before" --task-time=on --aic-metrics=PipeUtilization \
        ./build/fft_check 1024 4096 3          # 并入前
"$AB_MSPROF" --output="$AB_WORK/p_a6_after"  --task-time=on --aic-metrics=PipeUtilization \
        ./build/fft_check 1024 4096 3          # 并入后（改动已回退，仅存档）
```
目录为 `$AB_WORK/p_a6_before`、`$AB_WORK/p_a6_after`
（默认 `$AB_WORK` = `<仓库>/.tmp`，可用 `AB_WORK=` 覆盖；本文写就时的历史目录在 `/tmp/op/`）。

---

## 7. 数据出处

| 数据 | 文件 / 取法 |
|:---|:---|
| 设备任务时长 | `device_0/sqlite/ascend_task.db` → `AscendTask.duration`（**ns**，`sum_prof.py` 已 /1000） |
| 管线占用率 | `mindstudio_profiler_output/op_summary_*.csv` → `aiv_vec_ratio` 等 |
| host API 耗时 | `mindstudio_profiler_output/api_statistic_*.csv` |
| op 聚合统计 | `mindstudio_profiler_output/op_statistic_*.csv` |
| **采集** | `scripts/profile_test.sh`（5 个用例，落 `results/profiles/<UTC>/`；rfft 用例当前因 `baseline_rfft` 参数不全失败，另有人修） |
| **汇总** | `scripts/sum_prof.py <dir> [--per N] [--drop N] [--csv]` |
| 原始 profile 目录 | 本文写就时在 `/tmp/op/prof_ours`、`prof_b1`、`prof_b64`、`prof_pipe`、`prof_nat`、`prof_nat2`；重跑落在 `results/profiles/` |
| 原生失分点 profile（§6.3） | `scripts/profile_test.sh --only nat` 只采 n=4096/B=4096（脚本无 `--n` 参数）；其余 n 用 §6.3 的手工循环，落 `$AB_WORK/p_nat{64,256,512,1024}` |
| 原生墙钟对照（§6.3.3） | `scripts/native_fft.py`（单形状）、`scripts/time_native.py`（多形状）、汇总 `scripts/sum_prof.py` |
| **A6 屏障前后（§6.4）** | `$AB_WORK/p_a6_before`、`$AB_WORK/p_a6_after` |

> 注：`--aic-metrics=Memory` 拿到的 `aiv_main_mem_read_bw ≈ 0.013 GB/s` 明显无效
> （本核实际 GM 流量 82 GB/s），**该组计数器在纯 AIV kernel 上不可用**，
> 内存结论请以 `mte2/mte3_ratio` 为准。
