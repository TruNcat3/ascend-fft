# 文档索引

本目录是 Ascend-FFT 的**设计与实验记录**。所有数字均为本机
（Ascend910_9382，48 AIV，CANN 9.0.0）实测，复现命令写在各自文档里。

> **每份文档头部都有「复现脚本」一行**；全仓的
> **实验 ↔ 文档 ↔ 脚本** 清单用 `scripts/repro.sh --list` 查，
> 按文档反查用 `scripts/repro.sh --doc <文档名片段>`。
> **先看**：[实验对比](实验对比.md)（图 + 详表）与
> [设计思路与演进](设计思路与演进.md)（4 条思路 + 时间线）；
> 入门、快速开始与 API 见 [README](../README.md)。

---

## 推荐阅读顺序

```text
⓪ 先看    实验对比.md                    图 + 详表（推荐先看，含端到端与三路端到端）
           设计思路与演进.md              4 条思路 + 阶段 0 → 49:0 时间线（与上一份并列）
①          阶段0-1-发现与结果.md          环境/硬件能力/原生基线/最早能跑通；§5~§11 覆盖阶段 2~6
②          性能优化-C2b与K择优.md          主线：5 轮 A/B 优化全过程（含批折叠、屏障、η）
③          trace与profile诊断-*.md         手段：msprof 诊断、管线占用率、对照 GPU 的假设评估
④          性能对比-标准库vs自研.md        结果：六基线 49 点同场对比（自动生成）
⑤          matrix_test_a7.md               结果：当前 49:0 逐点矩阵（权威）
⑥          实数变换-r2c与c2r.md            能力：r2c/c2r 半谱链路设计与结果（xflip、行距、98/98 网格）
⑦          矩阵测试与GPU绝对性能对比.md     展望：与网上公开 GPU 工作的绝对性能对照（正文=历史基线）
⑧          Cube张量化探针.md               探针：fp32 Cube（矩阵单元）可行性
⑨          matrix_test_raw.md              存档：批折叠前的 44:5 基线（表格一字不改）
```

图在 [`figures/`](figures/)，由 `scripts/plot_results.py` 从表格/JSON 生成，
正文 [`实验对比.md`](实验对比.md) 负责图注与口径说明。

---

## 看结果

| 文档 | 内容 | 复现脚本 | 状态 |
|---|---|---|---|
| [实验对比.md](实验对比.md) | **图 + 详表**：speedup 热力图、延迟热力图、batch 缩放、六基线柱状、η 散点、端到端、三路端到端（含裸 CANN C API）、**三类典型应用负载（§6.4）**；每张图的详细解读只在这一份 | `repro.sh figures` · `repro.sh doc` | **推荐先看**（生成物） |
| [matrix_test_a7.md](matrix_test_a7.md) | **当前权威矩阵**（49:0，`--rounds 3` min-of-means，η 均值 7.7%） | `repro.sh matrix` · `one_click_test.sh` · `calib_eta.py` | 结果（权威） |
| [性能对比-标准库vs自研.md](性能对比-标准库vs自研.md) | numpy / torch / `aclRfft1D` / 自研 v1 / CANN 原生 / 自研 六列 49 点 | `repro.sh sixway` · `bench_stdlib.py` · `bench_native_npu.py` | **自动重生成**（生成物） |
| [实数变换-r2c与c2r.md](实数变换-r2c与c2r.md) | r2c/c2r 半谱链路设计：xflip 符号翻转、`n+16` 行距、UB 守卫、行首屏障、98/98 网格与基准 | `repro.sh r2c-c2r` · `AB_DIR=r2c\|c2r ./build/fft_check` | 设计 + 结果 |

## 搞懂方法

| 文档 | 回答的问题 | 复现脚本 |
|---|---|---|
| [设计思路与演进.md](设计思路与演进.md) | 为什么这么设计？数字怎么从阶段 0 涨到 49 : 0？ | 叙述性文档，时间线「出处」列即复现入口 |
| [阶段0-1-发现与结果.md](阶段0-1-发现与结果.md) | 环境、硬件能力探测（probe）、`aclRfft1D` 基线、`kfft_fwd` 首版（§0~§4），以及 §5~§11 的阶段 2（框架骨架）～阶段 6（设计空间 / η / 选型闭环） | `init.sh --check` · `hw_probe.sh` · `build.sh rfft` · `matrix_test.py` · `ab_test.py` |
| [性能优化-C2b与K择优.md](性能优化-C2b与K择优.md) | C2b、K 择优、merge 放宽、MTE 重叠、平面级 radix-4、**§11 批折叠/屏障/η/现状** | **`ab_test.py`**（§7/§9/§11）· `baseline_o.sh` · `calib_eta.py` · `one_click_test.sh` |
| [trace与profile诊断-小尺寸与大尺寸.md](trace与profile诊断-小尺寸与大尺寸.md) | `msprof` 原生小 n profile、**§6.4 屏障份额量化**、对「学 cuButterfly 搜索策略」的评估 | **`profile_test.sh`**（采集）· **`sum_prof.py`**（汇总）· `native_fft.py` / `time_native.py` |
| [Cube张量化探针.md](Cube张量化探针.md) | `Mmad` 可编译但结果读不出的实测结论（机会在搬运通路，不在 MAC） | `hw_probe.sh --only cube` · `profile_test.sh` · `sum_prof.py` |

## 对照与存档

| 文档 | 说明 | 状态 |
|---|---|---|
| [矩阵测试与GPU绝对性能对比.md](矩阵测试与GPU绝对性能对比.md) | 与 cuButterfly / cuFFT 等公开 GPU 结果的绝对性能对照 | **正文为历史基线**，比分现状 49:0 |
| [matrix_test_raw.md](matrix_test_raw.md) | 批折叠 A3~A5 之前的 44:5 基线 | **存档，勿覆盖** |

---

## 复现

### 实验 ↔ 脚本 ↔ 文档

完整清单与每条的**原样命令**：

```bash
scripts/repro.sh --list            # 全部实验：名字 | 文档 | 说明（-v 加命令列）
scripts/repro.sh --doc <文档名片段>  # 按文档反查
scripts/repro.sh <实验名>          # 跑一个实验（如 repro.sh e2e）
```

| 实验 | 命令 | 产物 | 文档 |
|---|---|---|---|
| 环境体检 | `scripts/init.sh --check` | 终端 | README · 本文 |
| 4 道门禁 + 编译 | `scripts/one_click_test.sh --no-matrix` | `results/<UTC>/` | [实验对比.md](实验对比.md) §7 |
| 49 点性能矩阵 | `scripts/repro.sh matrix` | [matrix_test_a7.md](matrix_test_a7.md) | 同左（README 图1 的数据源） |
| 批折叠前基线存档 | `scripts/repro.sh matrix-archive` | [matrix_test_raw.md](matrix_test_raw.md) | 同左（**勿覆盖**） |
| 六基线 49 点 | `scripts/repro.sh sixway` | [性能对比-标准库vs自研.md](性能对比-标准库vs自研.md) | 同左 |
| 端到端三路（自研 / torch / 裸 CANN，**三路均 pinned 主机缓冲**） | `scripts/repro.sh e2e` | `results/e2e.{md,json}` | [实验对比.md](实验对比.md) 图6·图7 · §6.2口径 |
| 应用负载端到端（OFDM / 雷达 / DL 频域层，**应用形状输入**，三方逐位同式） | `scripts/repro.sh e2e-app` | `results/e2e_app.{md,json}` | [实验对比.md](实验对比.md) §6.4 |
| 出图（图1~8） | `scripts/repro.sh figures` | [figures/](figures/) | [实验对比.md](实验对比.md) |
| 出架构图 | `python3 scripts/gen_arch_diagram.py` | [figures/architecture.svg](figures/architecture.svg) | [README · 架构](../README.md#架构) |
| 出对比文档 | `scripts/repro.sh doc` | [实验对比.md](实验对比.md) | 同左（生成物） |
| η 成本模型标定 | `scripts/repro.sh eta` | 打印 3 个系数，人工回填 `estimate()` | [性能优化-C2b与K择优.md](性能优化-C2b与K择优.md) §3·§11.5 |
| **A/B 消融** | `python3 scripts/ab_test.py --base <.o> --cand build/fft_radix2.o …` | 终端 / `--json` | [性能优化-C2b与K择优.md](性能优化-C2b与K择优.md) §7·§9·§11 |
| 从提交重建历史基线 `.o` | `scripts/baseline_o.sh <rev>` | `build/baseline_<rev>_*.o` | 同上 §8·§9.8 |
| **硬件能力探针** | `scripts/hw_probe.sh` | 终端 | [阶段0-1-发现与结果.md](阶段0-1-发现与结果.md) §1 · 优化文档 §10.2 |
| **msprof 采集 + 汇总** | `scripts/profile_test.sh` | `results/profiles/<UTC>/` | [trace与profile诊断-小尺寸与大尺寸.md](trace与profile诊断-小尺寸与大尺寸.md) §0·§7 |
| 汇总已有 profile | `python3 scripts/sum_prof.py <dir>` | 终端 | 同上 |
| CANN 原生 NPU 基线 | `scripts/repro.sh native` | 终端 | [性能对比-标准库vs自研.md](性能对比-标准库vs自研.md) |
| CPU 标准库基线 | `scripts/repro.sh stdlib` | 终端 | 同左 |
| 裸 CANN `aclRfft1D` | `scripts/repro.sh rfft` / `rfft-e2e` | 终端 | [阶段0-1-发现与结果.md](阶段0-1-发现与结果.md) §2 · 实验对比 图7 |
| r2c/c2r 半谱变换基准 | `scripts/repro.sh r2c-c2r` | `results/r2c_c2r.json` | [README · 结果](../README.md#结果) · [实数变换-r2c与c2r.md](实数变换-r2c与c2r.md) |
| Cube（矩阵单元）探针 | `scripts/hw_probe.sh --only cube` | 终端 | [Cube张量化探针.md](Cube张量化探针.md) |
| 传输带宽探针 | `scripts/hw_probe.sh --only bw` | 终端 | [实验对比.md](实验对比.md) §6.3 |
| 与公开 GPU 结果对照 | `scripts/repro.sh gpu-compare` | `results/gpu_compare.md` | [矩阵测试与GPU绝对性能对比.md](矩阵测试与GPU绝对性能对比.md) |

> ⚠️ `matrix_test_raw.md` 是**基线存档**，请勿覆盖；新矩阵另存
> `docs/matrix_test.md` 或 `results/<时间戳>/matrix.md`。

### `scripts/` 一览

| 脚本 | 职责 |
|---|---|
| [`env.sh`](../scripts/env.sh) | **路径与环境的单一真源**（`AB_CANN`/`AB_PY`/`AB_MSPROF`/`AB_WORK` + 编译函数） |
| [`abenv.py`](../scripts/abenv.py) | 上面这套规则的 Python 侧（同一份 `.ab_py` 缓存） |
| [`init.sh`](../scripts/init.sh) | 环境体检 → 编译 → 门禁（从零跑通用这条） |
| [`build.sh`](../scripts/build.sh) / [`Makefile`](../Makefile) | 编译 target：kernel/check/test/limits/rfft/probe/simt/bw/stride/cube/all |
| [`one_click_test.sh`](../scripts/one_click_test.sh) | 门禁 + 49 点矩阵一键，判据不过则 exit 1 |
| [`repro.sh`](../scripts/repro.sh) | **实验 ↔ 文档 ↔ 脚本 注册表**：`--list` / `--doc` / `<名字>` / `all` |
| [`matrix_test.py`](../scripts/matrix_test.py) | 49 点矩阵（正确性 + 实测 + η + vs 原生） |
| [`e2e_test.py`](../scripts/e2e_test.py) | 端到端三路（自研 / torch / 裸 CANN） |
| [`gen_stdlib_doc.py`](../scripts/gen_stdlib_doc.py) · [`bench_stdlib.py`](../scripts/bench_stdlib.py) · [`bench_native_npu.py`](../scripts/bench_native_npu.py) | 六基线同场与两份基线 |
| [`bench_r2c_c2r.py`](../scripts/bench_r2c_c2r.py) | r2c/c2r device 基准（自研 vs torch vs `aclRfft1D`）→ `results/r2c_c2r.json` |
| [`plot_results.py`](../scripts/plot_results.py) · [`gen_compare_doc.py`](../scripts/gen_compare_doc.py) | 出图 + 出 [实验对比.md](实验对比.md) |
| [`gen_arch_diagram.py`](../scripts/gen_arch_diagram.py) | 出架构图（[figures/architecture.svg](figures/architecture.svg)） |
| [`calib_eta.py`](../scripts/calib_eta.py) | η 成本模型最小二乘标定 |
| [`ab_test.py`](../scripts/ab_test.py) · [`baseline_o.sh`](../scripts/baseline_o.sh) | 批量 A/B 消融 + 从提交重建历史基线 `.o` |
| [`hw_probe.sh`](../scripts/hw_probe.sh) | 硬件能力探针串联（核数/子核/mask 上限/Gather/带宽/SIMT） |
| [`profile_test.sh`](../scripts/profile_test.sh) · [`sum_prof.py`](../scripts/sum_prof.py) · [`native_fft.py`](../scripts/native_fft.py) · [`time_native.py`](../scripts/time_native.py) | msprof 采集、汇总、原生用例与墙钟对照 |

---

## 路径约定（不再硬编码）

| 变量 | 用途 | 由谁探测 |
|---|---|---|
| `$AB_CANN` | CANN 工具包根（`ccec`/`msprof`/`include`/`lib64`） | `scripts/env.sh` |
| `$AB_PY` | 带 `torch_npu` 的 python | `env.sh` 与 `scripts/abenv.py`（共用 `.ab_py` 缓存） |
| `$AB_MSPROF` | msprof 可执行 | `env.sh` |
| `$AB_WORK` | 临时工作区（profile、A/B 中间文件），默认 `<仓库>/.tmp` | `env.sh` |
| `$AB_SOC` | SoC 名 | `env.sh` |

手工 shell 里先 `source scripts/env.sh` 即可拿到全部变量；
覆盖写法 `AB_CANN=... AB_PY=... scripts/xxx.sh`。

---

## 术语速查

| 术语 | 含义 |
|---|---|
| **η** | 成本模型估时（µs），`estimate()` 在 kernel launch 前给出 |
| **D** | 批折叠系数 `foldDFor`，一个 Level-0 repeat 覆盖的连续 batch 数（≤ 4） |
| **K** | 平面级 radix-2 的平面列数 `planeKFor`，`rows = n / K` |
| **原生** | 本卡原生复数 fp32 前向（`torch.fft.fft` 走 torch_npu op-plugin；CANN 9.0.0 无复数 C2C C API，详见 [实验对比 图7](实验对比.md#图7-三路端到端自研--torch--裸-cann)） |
| **r2c / c2r** | 实数半谱变换：实数 ↔ `n/2+1` 半谱（numpy `rfft`/`irfft` 布局，c2r 含 `1/n`），链路复用 `kfft_fwd` + 短链后处理，详见 [实数变换-r2c与c2r](实数变换-r2c与c2r.md) |
| **裸 CANN** | `aclRfft1D` 直接调用（aclNN），实→复、单边，仅作参照 |
| **v1** | 本仓库 `fft_radix2_v1.o`（标量旋转因子基线） |
| **自研** | 本仓库 `fft_radix2.o`（当前主 kernel） |
| **pinned** | `aclrtMallocHost` / `pin_memory()` 申请的主机内存，H2D/D2H 不走主机侧 staging；**端到端三路的默认口径**（开关 `AB_E2E_HOST`，`pageable` 仅为受限对照，见 [实验对比 §6.3](实验对比.md#63-已知局限必须一起读) |
| **E2E** | 端到端口径：H2D → 变换 → D2H 全程计时；`device-only` 只算 kernel，两者不可混引（见 [实验对比 §6.2](实验对比.md#62-口径)） |
| **min-of-means** | `--rounds R` 轮各算均值、再取 R 轮最小值（双方同口径） |
| **A/B** | 交替运行基线与候选、每轮取 min（共租户噪声下唯一可比的口径） |
