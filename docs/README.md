# 文档索引

本目录是 Ascend-FFT 的**设计与实验记录**。所有数字均为本机
（Ascend910_9382，48 AIV，CANN 9.0.0）实测，复现命令写在各自文档里。

> **每份文档头部都有「复现脚本」一行**；全仓的
> **实验 ↔ 文档 ↔ 脚本** 清单用 `scripts/repro.sh --list` 查，
> 按文档反查用 `scripts/repro.sh --doc <文档名片段>`。

---

## 推荐阅读顺序

```text
实验对比.md                      ⓪ 图 + 详表（推荐先看，含端到端与三路端到端）
阶段0-1-发现与结果.md          ① 环境/硬件能力/原生基线/最早能跑通；§5~§11 覆盖阶段 2~6
性能优化-C2b与K择优.md          ② 主线：5 轮 A/B 优化全过程（含批折叠、屏障、η）
trace与profile诊断-*.md         ③ 手段：msprof 诊断、管线占用率、对照 GPU 的假设评估
性能对比-标准库vs自研.md        ④ 结果：六基线 49 点同场对比（自动生成）
matrix_test_a7.md               ⑤ 结果：当前 49:0 逐点矩阵（权威）
矩阵测试与GPU绝对性能对比.md     ⑥ 展望：与网上公开 GPU 工作的绝对性能对照（正文=历史基线）
Cube张量化探针.md               ⑦ 探针：fp32 Cube（矩阵单元）可行性
matrix_test_raw.md              ⑧ 存档：批折叠前的 44:5 基线（表格一字不改）
```

图在 [`figures/`](figures/)，由 `scripts/plot_results.py` 从上面的表格/JSON 生成，
正文 `实验对比.md` 负责图注与口径说明。

---

## 文档清单（含复现脚本）

| 文档 | 主题 | 复现脚本 | 状态 |
|---|---|---|---|
| [实验对比.md](实验对比.md) | **图 + 详表**：speedup 热力图、延迟热力图、batch 缩放、六基线柱状、η 散点、端到端、三路端到端（含裸 CANN C API）、**三类典型应用负载（§6.4）** | `repro.sh figures` · `repro.sh doc`（本文是生成物）；数据来自 `matrix_test.py` / `e2e_test.py` | **推荐先看** |
| [阶段0-1-发现与结果.md](阶段0-1-发现与结果.md) | 环境、硬件能力探测（probe）、`aclRfft1D` 基线、`kfft_fwd` 首版（§0~§4），以及 §5~§11 的阶段 2（框架骨架）～阶段 6（设计空间 / η / 选型闭环） | `init.sh --check` · `hw_probe.sh` · `build.sh rfft` · `matrix_test.py` · `ab_test.py` | 历史主线 |
| [性能优化-C2b与K择优.md](性能优化-C2b与K择优.md) | C2b、K 择优、merge 放宽、MTE 重叠、平面级 radix-4、**§11 批折叠/屏障/η/现状** | **`ab_test.py`**（§7/§9/§11）· `baseline_o.sh`（历史基线 `.o`）· `calib_eta.py` · `hw_probe.sh` · `one_click_test.sh` | 最新主线 |
| [trace与profile诊断-小尺寸与大尺寸.md](trace与profile诊断-小尺寸与大尺寸.md) | `msprof` 原生小 n profile、**§6.4 屏障份额量化**、对「学 cuButterfly 搜索策略」的评估 | **`profile_test.sh`**（采集）· **`sum_prof.py`**（汇总）· `native_fft.py` / `time_native.py` | 诊断 |
| [性能对比-标准库vs自研.md](性能对比-标准库vs自研.md) | numpy / torch / `aclRfft1D` / 自研 v1 / CANN 原生 / 自研 六列 49 点 | `gen_stdlib_doc.py`（本文是生成物）· `bench_stdlib.py` · `bench_native_npu.py` | **自动重生成** |
| [matrix_test_a7.md](matrix_test_a7.md) | **当前权威矩阵**（49:0，`--rounds 3` min-of-means，η 均值 7.7%） | `repro.sh matrix` · `one_click_test.sh` · `calib_eta.py` | 结果 |
| [matrix_test_raw.md](matrix_test_raw.md) | 批折叠 A3~A5 之前的 44:5 基线存档 | `repro.sh matrix-archive`（**勿覆盖**） | 存档 |
| [矩阵测试与GPU绝对性能对比.md](矩阵测试与GPU绝对性能对比.md) | 与 cuButterfly / cuFFT 等公开 GPU 结果的绝对性能对照（**正文为历史基线**，比分现状 49:0） | `repro.sh gpu-compare` · `hw_probe.sh --only bw` | 对外对照 |
| [Cube张量化探针.md](Cube张量化探针.md) | `Mmad` 可编译但结果读不出的实测结论 | `hw_probe.sh --only cube` · `profile_test.sh` · `sum_prof.py` | 探针 |

---

## 实验 ↔ 脚本 ↔ 文档（反查入口）

完整清单与每条的**原样命令**：

```bash
scripts/repro.sh --list            # 全部实验：名字 | 文档 | 说明
scripts/repro.sh --list -v         # 加上命令列
scripts/repro.sh --doc 优化        # 只看 docs/性能优化-* 涉及的实验
scripts/repro.sh <实验名>          # 跑一个实验（如 repro.sh e2e）
```

按**主题**汇总：

| 主题 | 实验（`repro.sh` 名字） | 产出文档 |
|---|---|---|
| 环境与门禁 | `init` `gate` | README · 实验对比 §7 |
| 性能矩阵 | `matrix` `matrix-archive` `sixway` `gpu-compare` | matrix_test_a7 · matrix_test_raw · 性能对比-标准库 · GPU 对照 |
| 端到端（三路均 **pinned** 主机缓冲） | `e2e` `e2e-app` `rfft-e2e` | 实验对比 图6 / 图7 · §6.2口径 · **§6.4 应用负载** |
| 图与文档 | `figures` `doc` | 实验对比 · figures/ |
| 模型与选型 | `eta` `ab` `baseline-o` | 性能优化 §3·§7·§9·§11 |
| 硬件与探针 | `hwprobe` `cube` `bwprobe` | 阶段0-1 §1 · 优化 §10.2 · Cube 探针 |
| Profile | `profile` `profile-sum` | trace与profile诊断 §0·§7 |
| 基线 | `native` `stdlib` `rfft` | 性能对比-标准库 · 阶段0-1 §2 |

---

## 一键数据 / 文档的入口

```bash
scripts/init.sh                            # 从零：体检 + 编译 + 门禁
scripts/one_click_test.sh                  # 门禁 + 49 点矩阵，结果落 results/<UTC>/
scripts/hw_probe.sh                        # 硬件能力探针（换 SoC 后先跑这个）
scripts/profile_test.sh                    # msprof 采集 + 汇总 -> results/profiles/
python3 scripts/e2e_test.py --reps 10 --rounds 3   # 端到端（H2D+变换+D2H，三路，pinned）
python3 scripts/e2e_test.py --app all --reps 10 --rounds 3   # 三类典型应用负载（OFDM/雷达/DL 频域层）
python3 scripts/plot_results.py                   # 出图 -> docs/figures/
python3 scripts/gen_compare_doc.py                # 出 docs/实验对比.md
python3 scripts/gen_stdlib_doc.py                 # 出 性能对比-标准库vs自研.md
python3 scripts/calib_eta.py                      # 重跑 η 标定（lstsq3，1/y 加权；打印 3 个系数，人工回填 src/framework/butterfly.cpp 的 estimate()）
python3 scripts/matrix_test.py --rounds 5 --out docs/matrix_test.md
python3 scripts/ab_test.py --base <.o> --cand build/fft_radix2.o --points 4096x4096
python3 scripts/sum_prof.py results/profiles/p_b4k    # 读已有 profile
scripts/baseline_o.sh <git-rev>             # 从提交重建历史基线 .o
```

> ⚠️ `matrix_test_raw.md` 是**基线存档**，请勿覆盖；新矩阵另存
> `docs/matrix_test.md` 或 `results/<时间戳>/matrix.md`。

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
| **裸 CANN** | `aclRfft1D` 直接调用（aclNN），实→复、单边，仅作参照 |
| **v1** | 本仓库 `fft_radix2_v1.o`（标量旋转因子基线） |
| **自研** | 本仓库 `fft_radix2.o`（当前主 kernel） |
| **pinned** | `aclrtMallocHost` / `pin_memory()` 申请的主机内存，H2D/D2H 不走主机侧 staging；**端到端三路的默认口径**（开关 `AB_E2E_HOST`，`pageable` 仅为受限对照，见 [实验对比 §6.3](实验对比.md#63-已知局限必须一起读) |
| **E2E** | 端到端口径：H2D → 变换 → D2H 全程计时；`device-only` 只算 kernel，两者不可混引（见 [实验对比 §6.2](实验对比.md#62-口径)） |
| **min-of-means** | `--rounds R` 轮各算均值、再取 R 轮最小值（双方同口径） |
| **A/B** | 交替运行基线与候选、每轮取 min（共租户噪声下唯一可比的口径） |
