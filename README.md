# Ascend-FFT

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Citation](https://img.shields.io/badge/CITATION-CFF-orange.svg)](CITATION.cff)
[![Hardware](https://img.shields.io/badge/SoC-Ascend910__9382%20%C2%B7%2048%20AIV-8A2BE2.svg)](config/ascend910_93_profile.json)
[![CANN](https://img.shields.io/badge/CANN-9.0.0-0B6BCB.svg)](docs/阶段0-1-发现与结果.md)
[![Result](https://img.shields.io/badge/results-49%3A0%20vs%20CANN%20native-brightgreen.svg)](docs/matrix_test_a7.md)

**华为昇腾（Ascend）上的复数 fp32 FFT 库，AscendC 实现。**
设计移植自 [**cuButterfly**](https://github.com/TruNcat3/cuButterfly)（BSD-3-Clause），
在 Ascend910_9382（48 AIV）全网格 49 个 `(n, batch)` 点上 **全部快于 CANN 原生复数 FFT（49 : 0）**。

> **English.** Ascend-FFT is a radix-2 DIT complex fp32 FFT library for Huawei Ascend NPUs,
> written in AscendC. Its design-space object set (`H/G/A/P/L/F/Q`), `Context`/`Plan`/`Transform`
> API shape, batch-parallel decomposition and cost-model-driven search are ported from
> [cuButterfly](https://github.com/TruNcat3/cuButterfly) (BSD-3-Clause). See
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance.

---

## 结果

49 点全网格（`n ∈ {64…4096}` × `batch ∈ {1…4096}`），`--reps 20 --rounds 3`、逐点取 min-of-means：

| 指标 | 结果 |
|---|---|
| 正确性（对双精度 CPU 参考，`maxRel ≤ 1e-4`） | **49 / 49 PASS**（worst 2.44e-7） |
| **vs CANN 原生复数 FFT**（同为 NPU、同变换） | **49 : 0**（基线 44:5） |
| vs `aclRfft1D`（实→复，仅参照） | **49 / 49** |
| vs 自研 v1（标量旋转因子版） | **49 / 49**（中位提速 11.2×，最好 38.8×） |
| vs numpy / torch (CPU) | **36 / 49**、**38 / 49**；`B≥1024` 各 **14 / 14** |
| 最佳 / 最紧倍率 | **6.86×** @ `n=128/B=4` ｜ **1.04×** @ `n=1024/B=4096` |
| 头条点 `n=4096/B=4096` | 1,492.7 vs 2,444.7 µs = **1.64×**，吞吐 **674 GFLOP/s** |
| **端到端**（H2D + 变换 + D2H） | **37/49** 更快，几何均值 **1.58×**（大 batch 被 PCIe 封顶，[图6](docs/实验对比.md#图6-端到端测试)） |
| 裸 CANN C API 参照（`aclRfft1D`，实→复、仅参照） | `n≤256 且 B≤64` 的 12 个点 device：**裸 104 µs** ｜ torch 原生 **93 µs** ｜ 自研 **17 µs** → 那 ~100 µs 固定开销在 CANN 调用路径本身，不在 torch 层（[图7](docs/实验对比.md#图7-三路端到端自研--torch--裸-cann)） |
| η 成本模型偏差 | mean **7.7%**、max 26.6% |

完整逐点表格见 [`docs/matrix_test_a7.md`](docs/matrix_test_a7.md)，
六基线（numpy/torch/`aclRfft1D`/v1/原生/自研）版见
[`docs/性能对比-标准库vs自研.md`](docs/性能对比-标准库vs自研.md)。

### 摘录（自研 ÷ CANN 原生，>1 表示更快）

| n \ batch | 1 | 4 | 16 | 64 | 256 | 1024 | 4096 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4.29× | 5.48× | 4.90× | 4.06× | 3.78× | 2.70× | 1.49× |
| 128 | 4.77× | **6.86×** | 5.76× | 4.62× | 3.50× | 3.11× | 1.55× |
| 256 | 4.08× | 4.09× | 4.46× | 4.62× | 3.78× | 2.53× | 1.30× |
| 512 | 4.52× | 4.60× | 4.56× | 3.70× | 3.30× | 2.53× | 1.29× |
| 1024 | 3.62× | 4.67× | 3.85× | 3.36× | 2.57× | 1.55× | **1.04×** |
| 2048 | 4.30× | 4.10× | 3.95× | 2.91× | 2.09× | 1.58× | 1.38× |
| 4096 | 3.14× | 3.25× | 3.19× | 2.31× | 1.50× | 1.09× | 1.64× |

---

## 导航

**从零到能跑**：`scripts/init.sh`（环境体检 → 编译 → 4 道门禁）。
**全仓实验清单**：`scripts/repro.sh --list` —— 每个实验都对应一份文档与一条命令。

### 实验 ↔ 脚本 ↔ 文档

| 实验 | 命令 | 产物 | 文档 |
|---|---|---|---|
| 环境体检 | `scripts/init.sh --check` | 终端 | 本文 · [docs/README.md](docs/README.md) |
| 4 道门禁 + 编译 | `scripts/one_click_test.sh --no-matrix` | `results/<UTC>/` | [docs/实验对比.md](docs/实验对比.md) §7 |
| 49 点性能矩阵 | `scripts/repro.sh matrix` | [docs/matrix_test_a7.md](docs/matrix_test_a7.md) | 同左（本文摘录的来源） |
| 批折叠前基线存档 | `scripts/repro.sh matrix-archive` | [docs/matrix_test_raw.md](docs/matrix_test_raw.md) | 同左（**勿覆盖**） |
| 六基线 49 点 | `scripts/repro.sh sixway` | [docs/性能对比-标准库vs自研.md](docs/性能对比-标准库vs自研.md) | 同左 |
| 端到端三路（自研 / torch / 裸 CANN） | `scripts/repro.sh e2e` | `results/e2e.{md,json}` | [docs/实验对比.md](docs/实验对比.md) 图6·图7 |
| 出图（图1~7） | `scripts/repro.sh figures` | [docs/figures/](docs/figures/) | [docs/实验对比.md](docs/实验对比.md) |
| 出对比文档 | `scripts/repro.sh doc` | [docs/实验对比.md](docs/实验对比.md) | 同左（生成物） |
| η 成本模型标定 | `scripts/repro.sh eta` | 回写 `estimate()` | [docs/性能优化-C2b与K择优.md](docs/性能优化-C2b与K择优.md) §3·§11.5 |
| **A/B 消融** | `python3 scripts/ab_test.py --base <.o> --cand build/fft_radix2.o …` | 终端 / `--json` | [docs/性能优化-C2b与K择优.md](docs/性能优化-C2b与K择优.md) §7·§9·§11 |
| 从提交重建历史基线 `.o` | `scripts/baseline_o.sh <rev>` | `build/baseline_<rev>_*.o` | 同上 §7 |
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
| **诊断** | [`docs/trace与profile诊断-小尺寸与大尺寸.md`](docs/trace与profile诊断-小尺寸与大尺寸.md) | `msprof` 诊断、管线占用率、屏障份额 |
| **对照** | [`docs/矩阵测试与GPU绝对性能对比.md`](docs/矩阵测试与GPU绝对性能对比.md) | 与公开 GPU 工作的绝对性能对照 |
| **探针** | [`docs/Cube张量化探针.md`](docs/Cube张量化探针.md) | fp32 Cube 可行性 |
| **存档** | [`docs/matrix_test_raw.md`](docs/matrix_test_raw.md) | 批折叠前的 44:5 基线（**表不改**） |

更细的阅读顺序、术语速查与「哪份文档由哪个脚本生成」见 [`docs/README.md`](docs/README.md)。

---

## 快速开始

### 环境要求

| 项 | 要求 |
|---|---|
| SoC | `Ascend910_9382`（48 AIV / 196608 B UB），改 SoC 需同步 `AB_SOC` 与 `config/*.json` |
| CANN | 9.0.0（`ccec` + `libascendcl`），路径由 `AB_CANN` 指定、默认自动探测 |
| 编译器 | host `g++ -std=c++17` |
| Python | 任意带 `torch` + `torch_npu` 的 `python3`，解释器由 `AB_PY` 指定、默认自动探测 |

**所有路径都统一由 [`scripts/env.sh`](scripts/env.sh) 探测并导出**，不再硬编码：

| 变量 | 含义 | 默认 |
|---|---|---|
| `AB_CANN` | CANN 工具包根 | `ASCEND_TOOLKIT_HOME` → `PATH` 里的 `ccec` → 最新 `/usr/local/Ascend/cann-*` |
| `AB_PY` | 带 torch_npu 的 python | `python3` → 逐个试 import（结果缓存到 `.ab_py`） |
| `AB_MSPROF` | msprof 路径 | `$AB_CANN/bin/msprof` → `PATH` |
| `AB_WORK` | 临时工作区（profile 等） | `<仓库>/.tmp` |
| `AB_SOC` | SoC 名 | `Ascend910_9382` |
| `AB_ROOT` / `AB_BUILD` | 仓库根 / `build/` | 自动 |

用法：`source scripts/env.sh`（所有脚本已自动 source），或直接
`AB_CANN=/opt/cann/8.0.0 AB_PY=/usr/bin/python3.10 scripts/build.sh all` 覆盖。
Python 侧同一套规则在 [`scripts/abenv.py`](scripts/abenv.py)。

### 从零初始化

```bash
scripts/init.sh            # 环境体检 → 编译 → 4 道门禁
scripts/init.sh --check    # 只体检，不编译
scripts/init.sh --quick    # 体检 + 编译 + 门禁 + 12 点抽样矩阵（约 5 min）
```

### 一键测试（编译 → 4 道门禁 → 49 点性能矩阵 → 结论）

```bash
scripts/one_click_test.sh                 # 全量，结果落 results/<UTC 时间戳>/，失败 exit 1
scripts/one_click_test.sh --quick         # 12 点抽样，约 5 min
scripts/one_click_test.sh --rounds 5      # 每点 5 轮，进一步压噪声
scripts/one_click_test.sh --no-matrix     # 只编译 + 门禁
```

门禁：`test_limits`(18) → `test_framework`(4 抽样点) → `stride_probe`(23 PASS/7 预期 FAIL)
→ `fft_check`(4 抽样点) → 矩阵判据（正确性 / 比值全 ≥1× / η 偏差）。

### 手动构建

```bash
./scripts/build.sh all stride      # kernel + host + 测试；target 见 scripts/build.sh 头部
make                                # 等价（根目录 Makefile），make help 列全部目标
```

### 单点运行

```bash
./build/fft_check <n> <batch> [reps=10]                    # 正确性 + 实测
AB_FOLD_D=4 AB_PLANE_K=16 ./build/fft_check 1024 4096 30   # 强制批折叠系数 D / 平面级 K
AB_FFT_O=build/fft_radix2_v1.o ./build/fft_check 64 1 20   # 指定 kernel .o
```

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
| `local_exchange = shuffle/warp` | ⛔ 不可用 | 本 SoC 无 SIMT（`--enable-simt` 被 ccec 拒绝） |

**未复制任何上游源文件**：本仓库每个翻译单元都是面向 Ascend 的 AscendC / host C++ 重写。
上游仅作为设计出处引用，完整声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## API 概览

C++ API（**不提供稳定 C ABI**），完整声明在 [`include/butterfly/`](include/butterfly/)：

```cpp
#include "butterfly/plan.hpp"

bfly::Context ctx;
ctx.init("config/ascend910_93_profile.json",   // H：硬件事实
         "config/butterfly_space.json",        // 设计空间轴（唯一真源）
         "build/fft_radix2.o");                // device kernel

// 枚举设计空间 → 用 η 成本模型排序 → 取最优候选
bfly::Plan plan = ctx.createPlan(/*n=*/4096, /*batch=*/4096);

plan.prepare(4096, 4096);              // 生成并上传旋转因子/索引（幂等）
plan.run(in, out, 4096, 4096);         // 一次 n×batch 复数 fp32 前向 FFT
bfly::Metric m;
plan.measure(4096, 4096, &m);          // 实测 µs，回填 Metric 并置 Measured
```

输入/输出布局：`float32` **交错复数** `[batch][2*n]`（`re, im, re, im, ...`）。

核心启发式（`include/butterfly/fft_k.hpp`）：

```cpp
uint32_t planeKFor(uint32_t n);                  // 平面级 K：只看 n，纯静态
uint32_t foldDFor(uint32_t n, uint32_t batch,
                  uint32_t nblk = 48);           // 批折叠 D：kFoldCap=4
```

---

## 目录结构

```
ascend-fft/
├── include/butterfly/        公共头文件（API + 静态启发式）
│   ├── plan.hpp                Context / Plan / Transform
│   ├── objects.hpp             H G A P L F Q 设计空间对象
│   ├── enumerate.hpp           枚举 + η 成本模型 + 排序
│   ├── fft_k.hpp               planeKFor / foldDFor / op-elem 计数
│   └── reference.hpp           双精度 CPU 参考实现
├── src/
│   ├── ascendc/               device kernel（AscendC）
│   │   ├── fft_radix2.cpp       主 kernel（批折叠 + 平面级 radix-4 融合）
│   │   ├── fft_radix2_v1.cpp    v1 基线（标量旋转因子）
│   │   └── *_probe.cpp          硬件/Stride/Cube/Gather 探针
│   ├── framework/             host 框架（选型、计费、索引生成、launch）
│   └── host/                  可执行入口（fft_check / stride_probe / 基线 / launch）
├── tests/                    test_limits(18) / test_framework
├── config/                   硬件 profile + 设计空间 JSON（唯一真源）
├── scripts/                  构建 / 初始化 / 实验运行器（详见下表）
├── docs/                     设计与实验文档（索引见 docs/README.md）
├── results/                  一键测试与 profile 输出（git 忽略）
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
| [`calib_eta.py`](scripts/calib_eta.py) | η 成本模型最小二乘标定 |
| [`ab_test.py`](scripts/ab_test.py) · [`baseline_o.sh`](scripts/baseline_o.sh) | 批量 A/B 消融 + 从提交重建历史基线 `.o` |
| [`hw_probe.sh`](scripts/hw_probe.sh) | 硬件能力探针串联（核数/子核/mask 上限/Gather/带宽/SIMT） |
| [`profile_test.sh`](scripts/profile_test.sh) · [`sum_prof.py`](scripts/sum_prof.py) · [`native_fft.py`](scripts/native_fft.py) · [`time_native.py`](scripts/time_native.py) | msprof 采集、汇总、原生用例与墙钟对照 |

---

## 设计要点

1. **radix-2 DIT，两段式布局** —— 平面级（`rows = n/K` 行 × `K` 列，`K` 由 `planeKFor` 择优）
   走 radix-2/融合 radix-4；planar 级逐级 `2h` 蝶形，靠 Level-0 `repeat` 把 group 维折进指令。
2. **批折叠 A3~A5** —— `foldDFor` 把最多 **4** 个连续 batch 拍进同一组 Level-0 repeat，
   planar 段 `useL0` 下只走一遍 `d` 循环。这是大 batch 点相对原生拉开差距的主因
   （`n=64/B=4096` 由 0.69× 反超到 1.49×）。三道闸：`kFoldCap=4` → `repeat ≤ 255` → 48 核并发。
3. **η 成本模型 + 选型闭环** —— `estimate()` 数出 `#op` / `#elem`，按
   `η = launchUs + iters·(opNs·#op + elemNs·#elem)` 计费，49 点加权标定、独立验证 mean 偏差 7.7%。
4. **实测回填** —— `measure()` 真跑一次并把结果写回候选，`rank()` 按 `Measured > Feasible` 分层。
5. **硬件事实优先于推断** —— 所有上限（UB 196608 B、`mask ≤ 64`、`repeat ≤ 255`、
   32 B 对齐、无 SIMT、单 sub-block）都由 `*_probe` 实测得出，不是猜的。
   换 SoC 后先跑 `scripts/hw_probe.sh` 再读文档里的数字。

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
