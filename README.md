# Ascend-FFT

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Citation](https://img.shields.io/badge/CITATION-CFF-orange.svg)](CITATION.cff)
[![Hardware](https://img.shields.io/badge/SoC-Ascend910__9382%20%C2%B7%2048%20AIV-8A2BE2.svg)](config/ascend910_93_profile.json)
[![CANN](https://img.shields.io/badge/CANN-9.0.0-0B6BCB.svg)](docs/阶段0-1-发现与结果.md)
[![Result](https://img.shields.io/badge/results-49%3A0%20vs%20CANN%20native-brightgreen.svg)](docs/matrix_test_a7.md)

**Ascend-FFT 是一个跑在华为昇腾 NPU 上的复数 fp32 FFT 库。**
device kernel 用 AscendC 写，host 侧带一层「枚举 → 计费 → 实测回填」的选型框架。
在 Ascend910_9382（48 AIV）全网格 49 个 `(n, batch)` 点上，**device-only 对 CANN 原生复数 FFT 49 : 0**。
设计移植自 [**cuButterfly**](https://github.com/TruNcat3/cuButterfly)（BSD-3-Clause）。

**一句话总结。** 相较于 CANN 原生复数 FFT：**device-only 计算口径快 1.04× ~ 6.86×**
（几何均值 **3.01×**，全网格 **49 : 0**）；把 H2D / D2H 搬运也算进来的**端到端口径**
（三方主机缓冲统一 **pinned**）**46 / 49 个点更快、1.02× ~ 2.70×**（几何均值 **1.66×**，
小批量 `B ≤ 64` 一档 **28 / 28 全胜**、几何均值 2.01×），另外 3 个大 batch 点是
0.95× ~ 0.99× 的持平档 —— 两种口径的逐点数据与拆分原因见下文 `结果`。

> **English.** Ascend-FFT is a radix-2 DIT complex fp32 FFT library for Huawei Ascend NPUs,
> written in AscendC, with a host-side framework that enumerates a design space, prices each
> candidate with a cost model, and back-fills real measurements. Its object set (`H/G/A/P/L/F/Q`),
> `Context`/`Plan`/`Transform` API shape, batch-parallel decomposition and model-driven search are
> ported from [cuButterfly](https://github.com/TruNcat3/cuButterfly) (BSD-3-Clause). See
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance.

---

## 怎么读这份 README

按目的挑一条线，不用从头读到尾：

1. **只想知道它是什么** → [架构](#架构)：一张图讲清四层选型闭环与端到端数据流。
2. **想知道这些数字怎么来的** → [性能总阶段](#性能总阶段)：阶段 0 → 49 : 0 的每一次跳变。
3. **要复现结果** → [结果](#结果) 的图 1 / 图 3 / 图 6，再进
   [`docs/实验对比.md`](docs/实验对比.md)（7 张图的完整版 + 逐点详表）。
4. **要改 kernel 或模型** → [思路](#思路) 4 条，再进
   [`docs/性能优化-C2b与K择优.md`](docs/性能优化-C2b与K择优.md)（5 轮 A/B 全记录）。
5. **要从零跑起来** → [快速开始](#快速开始)：一条 `scripts/init.sh`。
6. **要全仓文档清单、术语表与「哪个脚本生成哪份文档」** → [`docs/README.md`](docs/README.md)。

---

## 架构

<p align="center">
  <img src="docs/figures/architecture.svg"
       alt="选型闭环四层（硬件事实 → 设计空间 → 成本模型 η → Plan/kernel）与端到端数据流（H2D → kfft_fwd → D2H）"
       width="100%">
</p>

图由 [`scripts/gen_arch_diagram.py`](scripts/gen_arch_diagram.py) 生成
（`python3 scripts/gen_arch_diagram.py`，输出字节稳定、不写时间戳）。

**怎么读这张图：**

- **上半是选型闭环**，四层各管一件事：① `config/*.json` 存探针实测出来的硬件事实；
  ② 设计空间把事实翻译成「可行 / 不可行 + 不可行的原因」的候选集合；
  ③ `estimate()` 在 kernel **还没编译**时给每个候选算一笔账（η），`rank()` 按估算排序取 `topK`；
  ④ 逐个 `prepare()` → `measure()` 实测并回填 ——
  **量过的候选永远排在只是估过的前面**（`Measured > Feasible > Unverified`）。
- **下半是端到端数据流**：`float32` 交错复数 `[batch][2n]` 进出，旋转因子与三张索引
  由 host 预生成、随 plan 一次上传，设备上只跑变换本身。
- **两个计时区不要混**：`device-only` 只框 kernel 的 launch + 同步；`端到端` 框住
  H2D + 变换 + D2H，三方主机缓冲统一 **pinned**（`AB_E2E_HOST`）。
- 图中每个数字都指得回命令：硬件事实 → [`scripts/hw_probe.sh`](scripts/hw_probe.sh)，
  候选与四态 → `ctx.enumerate()`（[`tests/test_framework.cpp`](tests/test_framework.cpp)），
  η → [`scripts/calib_eta.py`](scripts/calib_eta.py)，
  比分 → [`docs/matrix_test_a7.md`](docs/matrix_test_a7.md)。

---

## 为什么做这个

**Ascend 上缺一个能直接调的复数 FFT。** CANN 9.0.0 的公开头文件里没有复数→复数的
FFT C API（`include/` 全量扫描只有 `aclRfft1D` 与 `aclStft` 两个）。大家平时用的
`torch.fft.fft` 走的是 torch_npu 自己注册的 `_fft_c2c` NPU 算子，
**不在 CANN 的算子库条目里**。于是 host 侧 C++ 想做一次复数 FFT，要么绕进 Python/torch，
要么自己写 kernel。本仓库补上后者 —— 一个 AscendC 的 radix-2 DIT 复数 fp32 FFT，
外加一层 `Context` / `Plan` 的 C++ 框架。

**但比「能跑」更重要的是「知道为什么快、也知道什么时候不快」。** 所以这个仓库把三样东西
做成了可测量的：

- **硬件事实**——UB 上限、Level-0 `mask` 上限、`repeat` 上限、32B 对齐、有没有 SIMT，
  全部由探针实测得出，人工回填 `config/*.json`，不靠推断；
- **成本模型 η**——kernel 还没编译就能算出这一趟要多少 µs，选型闭环因此才跑得起来；
- **改动的证据链**——4 道门禁 + 49 点矩阵 + 交替 A/B，加上能从任意历史提交重建基线 `.o`
  的脚本，文档里每个数字都指向一条命令。

**边界也照写。** device-only 全胜，但端到端仍有 3 个大 batch 点跌到 1.0× 以下
（0.95×~0.99×，被 PCIe 搬运封顶；46/49，几何均值 1.66×）；
小 batch 输给 CPU 单核（NPU 要付固定的发射开销）；本 SoC 没有 SIMT，设计空间里
`local_exchange = shuffle` 直接判 `Infeasible`；Cube 探针的结论是「矢量发射墙，不是算力墙」。
这些都躺在各自文档里，不藏。

---

## 结果

全网格 49 点：`n ∈ {64…4096}` × `batch ∈ {1…4096}`，`--reps 20 --rounds 3`、逐点取 min-of-means。
**图编号沿用 [`docs/实验对比.md`](docs/实验对比.md)**（那里是全部 7 张图的完整版）。

![device-only speedup heatmap: 49/49 points faster than CANN native, geometric mean 3.01x](docs/figures/fig1_speedup_heatmap.png)

**图 1** —— 自研 ÷ CANN 原生（同卡、同变换、同口径），**>1 表示自研更快**。
颜色越绿倍率越高；整张表没有任何一格跌破 1.0×，几何均值 **3.01×**。
优势集中在**中小 batch**（`B≤64` 一档 **2.3~6.9×**，几何均值 4.1×），`B=4096` 一档收窄到 **1.0~1.6×** ——
原因见下面的图 3 与「思路」第 2 条。

| 指标 | 结果 |
|---|---|
| 正确性（双精度 CPU 参考，`maxRel ≤ 1e-4`） | **49 / 49 PASS**（worst 2.44e-7） |
| **vs CANN 原生复数 FFT**（同卡同变换） | **49 : 0**，几何均值 **3.01×** |
| 最好 / 最紧倍率 | **6.86×** @ `n=128,B=4` ｜ **1.04×** @ `n=1024,B=4096` |
| 头条点 `n=4096,B=4096` | 1,492.7 vs 2,444.7 µs = **1.64×**（674 GFLOP/s） |
| 端到端（H2D + 变换 + D2H，pinned 主机缓冲） | **46 / 49** 更快，几何均值 **1.66×**（`B≤64` **28 / 28**、2.01×） |
| η 成本模型偏差 | 平均 **7.7%**，43/49 落在 ±15% 内（带外 6 点见 [实验对比 图5](docs/实验对比.md#图5-成本模型-η-vs-实测)） |
| vs numpy / torch (CPU) | **36 / 49**、**38 / 49**（`B≥1024` 各 **14 / 14**） |
| vs 自研 v1（标量旋转因子版） | **49 / 49**，中位 **11.2×**、最好 **38.8×** |

逐点数值见 [`docs/matrix_test_a7.md`](docs/matrix_test_a7.md)；
六基线（numpy / torch / `aclRfft1D` / v1 / 原生 / 自研）同场见
[`docs/性能对比-标准库vs自研.md`](docs/性能对比-标准库vs自研.md)。

![speedup and latency vs batch, one line per n](docs/figures/fig3_speedup_curve.png)

**图 3** —— 同样的数据按 batch 展开。左：倍率随 batch 单调下滑，7 条线**全部在 1.0× 虚线之上**；
右：自研绝对延迟（对数轴）—— `B≤64` 这一档几乎不随 batch 变
（`n=1024` 只从 26.5 涨到 29.1 µs），这一段被固定开销主导；`B>64` 才开始真正随 batch 上升
（`n=4096` 从 53.0 走到 1492.7 µs）。
**读法：小 batch 赢在 launch 开销低，大 batch 赢在每元素算得少。**

![end-to-end vs device-only speedup, and share of time outside the kernel](docs/figures/fig6_end_to_end.png)

**图 6** —— 把数据搬运算进来之后的真实读数（三方主机缓冲统一 **pinned**，口径见
[§6.3](docs/实验对比.md#63-已知局限必须一起读)）。上半：灰柱 device-only（**49/49**，
这一轮几何均值 **3.17×**，与图 1 的 3.01× 是两个时段的两次测量），彩色柱端到端
（**46/49、几何均值 1.66×**，绿=更快、红=更慢）；下半是一张 `n × batch` 热力图，
标的是**内核之外的时间占比**（H2D + D2H + 同步），`B≥1024` 一档 **69~83%** ——
那一档比的不是 FFT，是 PCIe。端到端没做到 49/49 是**预期行为**：输掉的 3 个点是
0.95×~0.99× 的持平档，不是内核退化。

> 另外三张图放在这里太长，留给 [`docs/实验对比.md`](docs/实验对比.md)：
> [图2 延迟热力图](docs/实验对比.md#图2-延迟热力图) ·
> [图4 六基线同场](docs/实验对比.md#图4-六基线同场对比) ·
> [图7 自研 / torch / 裸 CANN 三路端到端](docs/实验对比.md#图7-三路端到端自研--torch--裸-cann)
> （后者的结论很有意思：`n≤256 且 B≤64` 的 12 个点上，**裸 CANN C API 103 µs、
> torch 原生 92 µs、自研 19 µs** —— 那 ~100 µs 的固定开销在 CANN 调用路径本身，不在 torch 层）。

---

## 性能总阶段

整条路线是**先测硬件 → 再让每条指令多干点活 → 最后让模型替你选**，
算法从头到尾都是 radix-2 DIT，变的是**每级发多少条指令、一批过几遍**。
下表按时间顺序列出每个阶段留下的头条读数（除标注外，都是 `n=4096, B=4096` 这一点，
reps 口径见各文档），最后一列是出处。

| 阶段 | 做了什么 | 头条读数 | 出处 |
|---|---|---|---|
| 阶段 0 | 环境体检 + 硬件探针：48 AIV、UB 196608 B、Level-0 `mask ≤ 64`、`repeat ≤ 255`、**无 SIMT** | 7 个探针 → 结论人工回填 `config/*.json` | [阶段0-1 §1](docs/阶段0-1-发现与结果.md) |
| 阶段 0.1 | 唯一公开的 C API `aclRfft1D` 打基线（**实→复、单边**） | 固定开销 ≈100 µs、workspace ≈2.16 GB | 同上 §2 |
| 阶段 1 | `kfft_fwd` 首版（radix-2 DIT）+ 双精度参考验收 | 15/15 PASS，`maxRel ≤ 1.6e-7` | 同上 §3 |
| 阶段 2 | 框架骨架：`H/G/A/P/L/F/Q` 对象 + 枚举 + 选型 + 实测回填 | `ctx.enumerate()` / `ctx.select()` 跑通 | 同上 §5 |
| 阶段 3 → 3++ | 8-plane 布局（前 3 级矢量化）；标量散射改 host 预生成 + `Gather` | 49 336 → 27 938（1.77×）→ 17 351 → **7 607 µs** | 同上 §6·§7 |
| 阶段 4 | K-plane 推广，`planeKFor(n)` 静态择优 | 7 607 → **3 765 µs**（2.02×） | 同上 §8 |
| 阶段 5 | 蝶形就地写回 + 屏障消融（当时把平面级 radix-4 **否掉**） | 3 758.7 → **3 293.3 µs**（−12.4%） | 同上 §9 |
| 阶段 6 | 设计空间真正生效：864 候选 → 四态枚举；η 带上 `F` 结构因子；选型闭环验收 | `n=4096`：816 Infeasible / **48 Feasible** | 同上 §11 |
| 第 1 批 C1+C2 | 平面级 `Axpy` 合并、planar Level-0 折组（记作后续批次的基线） | **~2013 µs** | [性能优化 §0](docs/性能优化-C2b与K择优.md) |
| 第 2~3 批 | C2b（复数乘 6→4）、**K 择优修正**、merge 放宽、MTE3⊗MTE2 重叠、**平面级 radix-4**（否极泰来） | 2013 → **1481 µs**（累计 −26.4%） | 同上 §0·§9 |
| 第 4~5 批 | 候选可行性逐条核查（多为否决）+ **批折叠 A3~A5**（一次 repeat 过 D 个 batch） | 比分 **44 : 5 → 49 : 0** | 同上 §10·§11 |
| **现在（A7）** | 权威 49 点矩阵 + 三方 pinned 端到端 | device-only **49 : 0、几何均值 3.01×**；端到端 **46 / 49、1.66×** | [matrix_test_a7](docs/matrix_test_a7.md) · [实验对比 §6](docs/实验对比.md#63-已知局限必须一起读) |

**读法**：前六阶段把「跑不通」变成「跑得快」（同一点 49 336 → 3 293 µs），
后面几批把「单点快」变成「全网格都不输」（44:5 → 49:0）；
端到端那一列是把 PCIe 也算进来之后的诚实读数（46/49，输的 3 个点是 0.95×~0.99× 持平档）。
每一步的取舍与被否掉的想法都在两份过程文档里，**没做的也写了为什么没做**。

---

## 思路

### 1. 先测硬件，再写代码

移植到一块新卡上，最容易犯的错是按产品文档的「标称值」写 kernel。这里反过来：
**所有上限都由探针实测，测出来的才进代码。**

`scripts/hw_probe.sh` 一次跑完 7 个探针，结论是：48 个矢量核、每个核**只有 1 个 AIV**
（`GetSubBlockNum()` 恒为 1，没有 sub-block 分工）、UB **196608 B**、
矢量访存 UB 偏移必须 **32B 对齐**、Level-0 `mask > 64` 会被**静默截断**（硬上限）、
`repeat ≤ 255`、**没有 SIMT**（`--enable-simt` 直接被 ccec 拒绝）。

后果是直接的：设计空间里 `local_exchange = shuffle` 这条轴在本卡上全部判 `Infeasible`
（只有 `shared` / `register` 可行）；
`foldDFor` 的几道闸（`kFoldCap=4` → `arRep = n/8 ≤ 255` → planar 段 `repeatTime ≤ 255`
→ 活跃组数 ≥ 并发核数）就是照着这些实测值写的。
换 SoC 时先跑这条探针，再读文档里的数字。

### 2. kernel：两段式布局，把 batch 折进指令

```
平面级   n 摆成 rows = n/K 行 × K 列的二维阵列，走 radix-2 / 融合 radix-4
         K 由 planeKFor(n) 静态择优
planar 级 逐级 2h 蝶形，group 维不发标量循环，直接折进 Level-0 repeat
批折叠   foldDFor 把最多 4 个连续 batch 拍进同一组 repeat，planar 段只走一遍 d 循环
```

一句话：**让一条矢量指令同时干更多事**。这是本仓库相对原生拉开差距的主因 ——
`n=64,B=4096` 这一格在批折叠之前是 **0.69×（输）**，批折叠之后变成 **1.49×（赢）**，
整张矩阵的比分也从 44:5 变成 49:0。

代价是 kernel 与 host 必须**同式**：`foldDFor` 的两条门槛在 `fft_radix2.cpp` 里
复刻了一份，`estimate()` 的 `#op`/`#elem` 也得对着 kernel 逐条重新数一遍
（`planeKFor` 倒是三方共用 `fft_k.hpp` 里的同一个定义）。两边只要对不上，
选出来的候选就会跑出慢一个量级的结果 —— 所以 `AB_PLANE_K` / `AB_FOLD_D`
两个环境变量被留下来做**强制复现**，A/B 时可以一键钉死这两项。

### 3. 模型：η 在 kernel 编译之前算账

`estimate()` 数出这一次 launch 要多少 `#op` 和 `#elem`，按

```
η = launchUs + iters × ( opNs · #op + elemNs · #elem )
```

计费。三个系数（`launchUs` = 16.109 µs、`opNs` = 17.077 ns、`elemNs` = 0.0502 ns）
由 `scripts/calib_eta.py` 在全网格上最小二乘标定，**按 `1/y` 加权**拟合相对误差
（否则会被 `n=4096,B=4096` 那种大点主导）。

![eta cost model vs measured, 49 points, mean absolute deviation 7.7%](docs/figures/fig5_eta_scatter.png)

**图 5** —— 49 个点全部贴在理想线上，平均 |偏差| **7.7%**，±15% 带内 **43/49**。
带外 6 个点里 5 个的单轮 `mean/min` ≥ 1.28，是宿主负载离群点；
只有 `n=1024,B=1024`（+16.2%）单轮很稳，是模型真实的高估点。

η 的用处**不是事后解释**，而是它能在 **kernel 还没编译**的时候就排出生命周期里的
launch 参数（`AB_FOLD_D` / `AB_PLANE_K`）—— 这是选型闭环能跑起来的前提。
实测回填再补一刀：`Plan::measure()` 真跑一次把结果写回候选，
`rank()` 按 `Measured > Feasible` 分层，估的和测的互相校正。

### 4. 流程：每个数字都能点回一条命令

```
探针  hw_probe.sh      → 实测结论人工回填 config/*.json
门禁  one_click_test.sh → test_limits(18) → test_framework(4) → stride_probe(23+7)
                          → fft_check(4) → 矩阵判据，任一不过 exit 1
对比  matrix_test.py    → 49 点正确性 + 实测 + η + vs 原生
消融  ab_test.py        → 逐点交替、每轮 reps 取 min、3 轮取最优（抵消共租户噪声）
存档  baseline_o.sh     → 从任意 git 提交重建当时的基线 .o，历史结论可复跑
```

机器本身也不理想：`nproc=1`、`loadavg` 常年 20 以上（实测 21~30 浮动）、NPU 上还有别的租户。
正因为如此，**所有 A/B 都必须交替运行 + 取 min**，否则轮间噪声会比改动本身还大 ——
这条口径贯穿全部实验文档。

---

## 快速开始

### 环境要求

| 项 | 要求 |
|---|---|
| SoC | `Ascend910_9382`（48 AIV / 196608 B UB），改 SoC 需同步 `AB_SOC` 与 `config/*.json` |
| CANN | 9.0.0（`ccec` + `libascendcl`），路径由 `AB_CANN` 指定、默认自动探测 |
| 编译器 | host `g++ -std=c++17` |
| Python | 任意带 `torch` + `torch_npu` 的 `python3`，解释器由 `AB_PY` 指定、默认自动探测 |

**所有路径都统一由 [`scripts/env.sh`](scripts/env.sh) 探测并导出**，不硬编码：

| 变量 | 含义 | 默认 |
|---|---|---|
| `AB_CANN` | CANN 工具包根 | `ASCEND_TOOLKIT_HOME` → 最新 `/usr/local/Ascend/cann-*` → `ascend-toolkit/latest` → `PATH` 里的 `ccec` |
| `AB_PY` | 带 torch_npu 的 python | `python3` → 逐个试 import（结果缓存到 `.ab_py`） |
| `AB_MSPROF` | msprof 路径 | `PATH` → `$AB_CANN/bin` → `$AB_CANN/tools/profiler/bin` |
| `AB_WORK` | 临时工作区（profile 等） | `<仓库>/.tmp` |
| `AB_SOC` | SoC 名 | `Ascend910_9382` |
| `AB_ROOT` / `AB_BUILD` | 仓库根 / `build/` | 自动 |

用法：`source scripts/env.sh`（所有脚本已自动 source），或直接
`AB_CANN=/opt/cann/8.0.0 AB_PY=/usr/bin/python3.10 scripts/build.sh all` 覆盖。
Python 侧同一套规则在 [`scripts/abenv.py`](scripts/abenv.py)。

**实验口径开关**（不参与路径探测；默认值就是本文全部数字所用的口径）：

| 变量 | 含义 | 默认 |
|---|---|---|
| `AB_E2E` | 打开端到端计时区（H2D + 变换 + D2H），值为重复次数 | 关 |
| `AB_E2E_HOST` | 端到端的主机缓冲：`pinned`（三方同口径）\| `pageable`（受限对照） | `pinned` |
| `AB_E2E_MODE` | `async`（带流 memcpy + 显式同步）\| `sync`（阻塞 memcpy）\| `xfer`（只搬不算，隔离纯传输带宽） | `async` |

端到端一律用 **pinned** 主机缓冲：pageable 会把「拷贝路径没选对」记进结果，
`device-only` 一列不受影响，详见 [实验对比 §6.3](docs/实验对比.md#63-已知局限必须一起读)。

### 三条命令

```bash
scripts/init.sh             # 从零：环境体检 → 编译 → 4 道门禁（最快，不跑矩阵）
scripts/init.sh --check     # 只体检，不编译不跑门禁
scripts/init.sh --quick     # 上面 + 9 点抽样矩阵（约 5 min）
scripts/init.sh --matrix    # 上面 + 49 点全网格（最慢）
scripts/one_click_test.sh   # 编译 → 4 道门禁 → 49 点矩阵 → 结论，结果落 results/<UTC>/
scripts/one_click_test.sh --no-matrix   # 只编译 + 4 道门禁（最快）
```

门禁顺序：`test_limits`(18 项边界) → `test_framework`(4 抽样点) →
`stride_probe`(23 PASS / 7 预期 FAIL) → `fft_check`(4 抽样点) →
矩阵判据（正确性 / 比值全 ≥1× / η 偏差）。失败则 `exit 1`。

### 手动构建与单点运行

```bash
./scripts/build.sh all stride      # kernel + host + 测试；target 见 scripts/build.sh 头部
make                                # 等价（根目录 Makefile），make help 列全部目标

./build/fft_check <n> <batch> [reps=10]                    # 正确性 + 实测
AB_FOLD_D=4 AB_PLANE_K=16 ./build/fft_check 1024 4096 30   # 强制批折叠系数 D / 平面级 K
AB_FFT_O=build/fft_radix2_v1.o ./build/fft_check 64 1 20   # 指定 kernel .o
```

---

## API 概览

C++ API（**不提供稳定 C ABI**），完整声明在 [`include/butterfly/`](include/butterfly/)：

```cpp
#include "butterfly/plan.hpp"

bfly::Context ctx;
ctx.init("config/ascend910_93_profile.json",   // H：硬件事实
         "config/butterfly_space.json",        // 设计空间轴（唯一真源）
         "build/fft_radix2.o");                // device kernel

// 选型循环：η 预排序 → 前 topK 个候选各构一个 plan → 实测 → 取最优
std::unique_ptr<bfly::Plan> plan = ctx.select(4096, 4096, /*topK=*/3);
// 想自己看盘：ctx.enumerate(4096, 4096) 返回全部候选（含 Infeasible 及原因）

plan->prepare(4096, 4096);             // 生成并上传旋转因子/索引（幂等）
plan->run(in, out, 4096, 4096);        // 一次 n×batch 复数 fp32 前向 FFT
bfly::Metric m;
plan->measure(4096, 4096, &m);         // 实测 µs，回填 Metric 并置 Measured
```

输入/输出布局：`float32` **交错复数** `[batch][2*n]`（`re, im, re, im, ...`）。

核心启发式（`include/butterfly/fft_k.hpp`）：

```cpp
uint32_t planeKFor(uint32_t n);                  // 平面级 K：只看 n，纯静态
uint32_t foldDFor(uint32_t n, uint32_t batch,
                  uint32_t nblk = 1);            // 批折叠 D：kFoldCap=4，nblk = 并发核数
```

---

## 文档与复现

**从零到能跑**：`scripts/init.sh`。**全仓实验清单**：`scripts/repro.sh --list` ——
每个实验都对应一份文档与一条命令，按文档反查用 `scripts/repro.sh --doc <文档名片段>`。

### 实验 ↔ 脚本 ↔ 文档

| 实验 | 命令 | 产物 | 文档 |
|---|---|---|---|
| 环境体检 | `scripts/init.sh --check` | 终端 | 本文 · [docs/README.md](docs/README.md) |
| 4 道门禁 + 编译 | `scripts/one_click_test.sh --no-matrix` | `results/<UTC>/` | [docs/实验对比.md](docs/实验对比.md) §7 |
| 49 点性能矩阵 | `scripts/repro.sh matrix` | [docs/matrix_test_a7.md](docs/matrix_test_a7.md) | 同左（本文图1 的数据源） |
| 批折叠前基线存档 | `scripts/repro.sh matrix-archive` | [docs/matrix_test_raw.md](docs/matrix_test_raw.md) | 同左（**勿覆盖**） |
| 六基线 49 点 | `scripts/repro.sh sixway` | [docs/性能对比-标准库vs自研.md](docs/性能对比-标准库vs自研.md) | 同左 |
| 端到端三路（自研 / torch / 裸 CANN，**三路均 pinned 主机缓冲**） | `scripts/repro.sh e2e` | `results/e2e.{md,json}` | [docs/实验对比.md](docs/实验对比.md) 图6·图7 · §6.2口径 |
| 出图（图1~7） | `scripts/repro.sh figures` | [docs/figures/](docs/figures/) | [docs/实验对比.md](docs/实验对比.md) |
| 出架构图 | `python3 scripts/gen_arch_diagram.py` | [docs/figures/architecture.svg](docs/figures/architecture.svg) | 本文 [架构](#架构) |
| 出对比文档 | `scripts/repro.sh doc` | [docs/实验对比.md](docs/实验对比.md) | 同左（生成物） |
| η 成本模型标定 | `scripts/repro.sh eta` | 打印 3 个系数，人工回填 `estimate()` | [docs/性能优化-C2b与K择优.md](docs/性能优化-C2b与K择优.md) §3·§11.5 |
| **A/B 消融** | `python3 scripts/ab_test.py --base <.o> --cand build/fft_radix2.o …` | 终端 / `--json` | [docs/性能优化-C2b与K择优.md](docs/性能优化-C2b与K择优.md) §7·§9·§11 |
| 从提交重建历史基线 `.o` | `scripts/baseline_o.sh <rev>` | `build/baseline_<rev>_*.o` | 同上 §8·§9.8 |
| **硬件能力探针** | `scripts/hw_probe.sh` | 终端 | [docs/阶段0-1-发现与结果.md](docs/阶段0-1-发现与结果.md) §1 · 优化文档 §10.2 |
| **msprof 采集 + 汇总** | `scripts/profile_test.sh` | `results/profiles/<UTC>/` | [docs/trace与profile诊断-小尺寸与大尺寸.md](docs/trace与profile诊断-小尺寸与大尺寸.md) §0·§7 |
| 汇总已有 profile | `python3 scripts/sum_prof.py <dir>` | 终端 | 同上 |
| CANN 原生 NPU 基线 | `scripts/repro.sh native` | 终端 | [docs/性能对比-标准库vs自研.md](docs/性能对比-标准库vs自研.md) |
| CPU 标准库基线 | `scripts/repro.sh stdlib` | 终端 | 同左 |
| 裸 CANN `aclRfft1D` | `scripts/repro.sh rfft` / `rfft-e2e` | 终端 | [docs/阶段0-1-发现与结果.md](docs/阶段0-1-发现与结果.md) §2 · 实验对比 图7 |
| Cube（矩阵单元）探针 | `scripts/hw_probe.sh --only cube` | 终端 | [docs/Cube张量化探针.md](docs/Cube张量化探针.md) |
| 传输带宽探针 | `scripts/hw_probe.sh --only bw` | 终端 | [docs/实验对比.md](docs/实验对比.md) §6.3 |
| 与公开 GPU 结果对照 | `scripts/repro.sh gpu-compare` | `results/gpu_compare.md` | [docs/矩阵测试与GPU绝对性能对比.md](docs/矩阵测试与GPU绝对性能对比.md) |

### 文档地图

| 分组 | 文档 | 内容 |
|---|---|---|
| **结果（先看）** | [`docs/实验对比.md`](docs/实验对比.md) | **图1~7 + 详表**：热力图、batch 缩放、六基线、η 散点、端到端、三路端到端 |
| | [`docs/matrix_test_a7.md`](docs/matrix_test_a7.md) | 当前权威矩阵（49:0，逐点） |
| | [`docs/性能对比-标准库vs自研.md`](docs/性能对比-标准库vs自研.md) | 六基线 49 点同场 |
| **过程** | [`docs/阶段0-1-发现与结果.md`](docs/阶段0-1-发现与结果.md) | 环境/硬件能力探测、`aclRfft1D` 基线、从 0 到可跑通 |
| | [`docs/性能优化-C2b与K择优.md`](docs/性能优化-C2b与K择优.md) | 5 轮 A/B 优化全记录（C2b、K 择优、radix-4、批折叠、η 标定） |
| **诊断** | [`docs/trace与profile诊断-小尺寸与大尺寸.md`](docs/trace与profile诊断-小尺寸与大尺寸.md) | `msprof` 诊断：小尺寸输 GPU、大尺寸输原生、自己卡在矢量发射墙 |
| **对照** | [`docs/矩阵测试与GPU绝对性能对比.md`](docs/矩阵测试与GPU绝对性能对比.md) | 与公开 GPU 工作的绝对性能对照（**正文为历史基线**，比分现状 49:0） |
| **探针** | [`docs/Cube张量化探针.md`](docs/Cube张量化探针.md) | fp32 Cube 可行性（结论：机会在搬运通路，不在 MAC） |
| **存档** | [`docs/matrix_test_raw.md`](docs/matrix_test_raw.md) | 批折叠前的 44:5 基线（**表不改**） |

更细的阅读顺序、术语速查与「哪份文档由哪个脚本生成」见 [`docs/README.md`](docs/README.md)。

---

## 设计来源

本项目的**设计来源是 [cuButterfly](https://github.com/TruNcat3/cuButterfly)** ——
一份 GPU 上的硬件映射时空并行 FFT / NTT / FWHT 库。移植关系如下：

| cuButterfly | 本仓库 | 说明 |
|---|---|---|
| `plan.hpp` 对象集 | `include/butterfly/objects.hpp` | `H`(硬件) `G`(生成) `A`(映射) `P`(分段) `L`(布局) `F`(融合) `Q`(质量) |
| `Context` / `Plan` / `Transform` | `include/butterfly/plan.hpp` | C++ API（**不提供稳定 C ABI**） |
| `Ub·Tb` 批并行循环 | `foldDFor(n, batch)` 批折叠 | 一次 Level-0 repeat 覆盖 D 个连续 batch |
| 旋转因子表 + `kBitReverseInput` | `Generator::genTwiddles` / `genBitReverse` / `genInterleave` | host 生成、随 plan 上传 |
| 设计空间 + 搜索 + 计分 | `DesignSpace` + `enumerate` + `estimate` + `rank` | `config/*.json` 为唯一真源 |
| `local_exchange = shuffle` | ⛔ 不可用 | 本 SoC 无 SIMT（`--enable-simt` 被 ccec 拒绝），只有 `shared`/`register` 可行 |

上游 README 里「What Is Frozen In v0.8」那张表也值得记住：cuButterfly 在 V100 上
对 cuFFT 是**部分 shape 打平、饱和大 batch 反而更慢**。这说明我们和它共享的是
**方法论**（硬件事实 → 设计空间 → 成本模型 → 实测回填），而不是可以白拿的数字。

**未复制任何上游源文件**：本仓库每个翻译单元都是面向 Ascend 的 AscendC / host C++ 重写。
上游仅作为设计出处引用，完整声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## 目录结构

```
ascend-fft/
├── include/butterfly/        公共头文件（API + 静态启发式）
│   ├── plan.hpp                Context / Plan / Transform
│   ├── objects.hpp             H G A P L F Q 设计空间对象
│   ├── enumerate.hpp           四态枚举 + 候选排序（Measured > Feasible > Unverified）
│   ├── fft_k.hpp               planeKFor / foldDFor
│   └── reference.hpp           双精度 FFT 参考 + maxRel 判据
├── src/
│   ├── ascendc/               device kernel（AscendC）
│   │   ├── fft_radix2.cpp       主 kernel（批折叠 + 平面级 radix-4 融合）
│   │   ├── fft_radix2_v1.cpp    v1 基线（标量旋转因子）
│   │   ├── fft_radix2_v2.cpp    v2（前 3 级 8-plane 布局）
│   │   ├── probe_hw.cpp / probe_simt.cpp
│   │   └── {stride,gather,cube}_probe.cpp
│   ├── framework/             host 框架（`estimate()` 计费、选型、索引生成、launch）
│   │   ├── butterfly.cpp        Context/Plan 实现 + `estimate()`/`rank()`/`Plan::measure()` 主体
│   │   └── reference.cpp        参考实现 + 测试向量
│   └── host/                  可执行入口（fft_check / stride_probe / 探针 / launch）
│       ├── baseline_rfft.cpp    裸 CANN `aclRfft1D` 三路对照基线
│       └── profile.cpp          msprof 用例入口
├── tests/                    test_limits(18) / test_framework
├── config/                   硬件 profile + 设计空间 JSON（唯一真源）
├── scripts/                  构建 / 初始化 / 实验运行器（详见下表）
├── docs/                     设计与实验文档（索引见 docs/README.md）
├── results/                  一键测试与 profile 输出（默认 git 忽略；`e2e.{json,md}` 作为存档跟踪）
├── CITATION.cff              GitHub「Cite this repository」引用元数据
├── LICENSE                   Apache-2.0
└── THIRD_PARTY_NOTICES.md    cuButterfly 出处与 BSD-3-Clause 文本
```

### `scripts/` 一览

| 脚本 | 职责 |
|---|---|
| [`env.sh`](scripts/env.sh) | **路径与环境的单一真源**（`AB_CANN`/`AB_PY`/`AB_MSPROF`/`AB_WORK` + 编译函数） |
| [`abenv.py`](scripts/abenv.py) | 上面这套规则的 Python 侧（同一份 `.ab_py` 缓存） |
| [`init.sh`](scripts/init.sh) | 环境体检 → 编译 → 门禁（从零跑通用这条） |
| [`build.sh`](scripts/build.sh) / [`Makefile`](Makefile) | 编译 target：kernel/check/test/limits/rfft/probe/simt/bw/stride/cube/all |
| [`one_click_test.sh`](scripts/one_click_test.sh) | 门禁 + 49 点矩阵一键，判据不过则 exit 1 |
| [`repro.sh`](scripts/repro.sh) | **实验 ↔ 文档 ↔ 脚本 注册表**：`--list` / `--doc` / `<名字>` / `all` |
| [`matrix_test.py`](scripts/matrix_test.py) | 49 点矩阵（正确性 + 实测 + η + vs 原生） |
| [`e2e_test.py`](scripts/e2e_test.py) | 端到端三路（自研 / torch / 裸 CANN） |
| [`gen_stdlib_doc.py`](scripts/gen_stdlib_doc.py) · [`bench_stdlib.py`](scripts/bench_stdlib.py) · [`bench_native_npu.py`](scripts/bench_native_npu.py) | 六基线同场与两份基线 |
| [`plot_results.py`](scripts/plot_results.py) · [`gen_compare_doc.py`](scripts/gen_compare_doc.py) | 出图 + 出 `docs/实验对比.md` |
| [`gen_arch_diagram.py`](scripts/gen_arch_diagram.py) | 出上面那张架构图（`docs/figures/architecture.svg`） |
| [`calib_eta.py`](scripts/calib_eta.py) | η 成本模型最小二乘标定 |
| [`ab_test.py`](scripts/ab_test.py) · [`baseline_o.sh`](scripts/baseline_o.sh) | 批量 A/B 消融 + 从提交重建历史基线 `.o` |
| [`hw_probe.sh`](scripts/hw_probe.sh) | 硬件能力探针串联（核数/子核/mask 上限/Gather/带宽/SIMT） |
| [`profile_test.sh`](scripts/profile_test.sh) · [`sum_prof.py`](scripts/sum_prof.py) · [`native_fft.py`](scripts/native_fft.py) · [`time_native.py`](scripts/time_native.py) | msprof 采集、汇总、原生用例与墙钟对照 |

---

## Citation

使用本软件或其实验数据请引用 [`CITATION.cff`](CITATION.cff)。BibTeX：

```bibtex
@software{wang_ascendfft_2026,
  author  = {Teng Wang},
  title   = {Ascend-FFT: Hardware-Mapped Complex FFT on Huawei Ascend NPUs},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/TruNcat3/ascend-fft}
}
```

本仓库的**设计来源**是 [cuButterfly](https://github.com/TruNcat3/cuButterfly)，
一并引用（其许可见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)）：

```bibtex
@software{wang_cubutterfly_2026,
  author  = {Teng Wang},
  title   = {cuButterfly: Hardware-Mapped Space-Time Parallelism for Butterfly Computations on GPUs},
  year    = {2026},
  version = {0.9.0},
  url     = {https://github.com/TruNcat3/cuButterfly}
}
```

Author: **Teng Wang**, High Efficient Intelligent Computing Lab, Suzhou
Institute for Advanced Research of USTC, Suzhou, China.

## License

本仓库以 [**Apache License 2.0**](LICENSE) 发布。

设计来源 **cuButterfly** 采用 **BSD 3-Clause**，版权与完整条款见
[**THIRD_PARTY_NOTICES.md**](THIRD_PARTY_NOTICES.md)。
