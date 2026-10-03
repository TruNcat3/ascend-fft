# 文档索引

本目录是 Ascend-FFT 的**设计与实验记录**。所有数字均为本机
（Ascend910_9382，48 AIV，CANN 9.0.0）实测，复现命令写在各自文档里。

## 推荐阅读顺序

```text
实验对比.md                      ⓪ 图 + 详表（推荐先看，含端到端）
阶段0-1-发现与结果.md          ① 环境/硬件能力/原生基线/最早能跑通
性能优化-C2b与K择优.md          ② 主线：5 轮 A/B 优化全过程（含批折叠、屏障、η）
trace与profile诊断-*.md         ③ 手段：msprof 诊断、管线占用率、对照 GPU 的假设评估
性能对比-标准库vs自研.md        ④ 结果：六基线 49 点同场对比（自动生成）
matrix_test_a7.md               ⑤ 结果：当前 49:0 逐点矩阵（权威）
矩阵测试与GPU绝对性能对比.md     ⑥ 展望：与网上公开 GPU 工作的绝对性能对照
Cube张量化探针.md               ⑦ 探针：fp32 Cube（矩阵单元）可行性
matrix_test_raw.md              ⑧ 存档：批折叠前的 44:5 基线（表格一字不改）
```

图在 [`figures/`](figures/)，由 `scripts/plot_results.py` 从上面的表格/JSON 生成，
正文 `实验对比.md` 负责图注与口径说明。

## 文档清单

| 文档 | 主题 | 状态 |
|---|---|---|
| [实验对比.md](实验对比.md) | **图 + 详表**：speedup 热力图、延迟热力图、batch 缩放、六基线柱状、η 散点、端到端、三路端到端（含裸 CANN C API） | **推荐先看** |
| [阶段0-1-发现与结果.md](阶段0-1-发现与结果.md) | 环境、硬件能力探测（probe）、`aclRfft1D` 基线、`kfft_fwd` 首版、框架骨架、阶段 3~6 | 历史主线 |
| [性能优化-C2b与K择优.md](性能优化-C2b与K择优.md) | C2b、K 择优、merge 放宽、MTE 重叠、平面级 radix-4、**§11 批折叠/屏障/η/现状** | 最新主线 |
| [trace与profile诊断-小尺寸与大尺寸.md](trace与profile诊断-小尺寸与大尺寸.md) | `msprof` 原生小 n profile、**§6.4 屏障份额量化**、对「学 cuButterfly 搜索策略」的评估 | 诊断 |
| [性能对比-标准库vs自研.md](性能对比-标准库vs自研.md) | numpy / torch / `aclRfft1D` / 自研 v1 / CANN 原生 / 自研 六列 49 点 | **自动重生成** |
| [matrix_test_a7.md](matrix_test_a7.md) | **当前权威矩阵**（49:0，`--rounds 3` min-of-means，η 均值 7.7%） | 结果 |
| [matrix_test_raw.md](matrix_test_raw.md) | 批折叠 A3~A5 之前的 44:5 基线存档 | 存档（勿改表） |
| [矩阵测试与GPU绝对性能对比.md](矩阵测试与GPU绝对性能对比.md) | 与 cuButterfly / cuFFT 等公开 GPU 结果的绝对性能对照 | 对外对照 |
| [Cube张量化探针.md](Cube张量化探针.md) | `Mmad` 可编译但结果读不出的实测结论 | 探针 |

## 一键数据 / 文档的入口

```bash
scripts/one_click_test.sh                 # 门禁 + 49 点矩阵，结果落 results/<UTC>/
python3 scripts/e2e_test.py --reps 10 --rounds 3   # 端到端（H2D+变换+D2H）
python3 scripts/plot_results.py                   # 出图 -> docs/figures/
python3 scripts/gen_compare_doc.py                # 出 docs/实验对比.md
scripts/gen_stdlib_doc.py                 # 重新生成 性能对比-标准库vs自研.md
scripts/calib_eta.py --fit                 # 重跑 η 标定（lstsq3，1/y 加权）
python3 scripts/matrix_test.py --rounds 5 --out docs/matrix_test.md
```

> ⚠️ `matrix_test_raw.md` 是**基线存档**，请勿覆盖；新矩阵另存
> `docs/matrix_test.md` 或 `results/<时间戳>/matrix.md`。

## 术语速查

| 术语 | 含义 |
|---|---|
| **η** | 成本模型估时（µs），`estimate()` 在 kernel launch 前给出 |
| **D** | 批折叠系数 `foldDFor`，一个 Level-0 repeat 覆盖的连续 batch 数（≤ 4） |
| **K** | 平面级 radix-2 的平面列数 `planeKFor`，`rows = n / K` |
| **原生** | CANN `aclfft` 复数 fp32 前向（同一 NPU） |
| **v1** | 本仓库 `fft_radix2_v1.o`（标量旋转因子基线） |
| **自研** | 本仓库 `fft_radix2.o`（当前主 kernel） |
| **min-of-means** | `--rounds R` 轮各算均值、再取 R 轮最小值（双方同口径） |
