# Ascend-FFT

**华为昇腾（Ascend）上的复数 fp32 FFT 库，AscendC 实现。**
设计移植自 [**cuButterfly**](https://github.com/TruNcat3/cuButterfly)（BSD-3-Clause），
在 Ascend910_9382（48 AIV）全网格 49 个 `(n, batch)` 点上 **全部快于 CANN 原生复数 FFT（49 : 0）**。

> **English.** Ascend-FFT is a radix-2 DIT complex fp32 FFT library for Huawei Ascend NPUs,
> written in AscendC. Its design-space object set (`H/G/A/P/L/F/Q`), `Context`/`Plan`/`Transform`
> API shape, batch-parallel decomposition and cost-model-driven search are ported from
> [cuButterfly](https://github.com/TruNcat3/cuButterfly) (BSD-3-Clause). See
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance.

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

## 快速开始

### 环境要求

| 项 | 要求 |
|---|---|
| SoC | `Ascend910_9382`（48 AIV / 196608 B UB），改 SoC 需同步 `AB_SOC` 与 `config/*.json` |
| CANN | 9.0.0（`ccec` + `libascendcl`），路径默认 `/usr/local/Ascend/cann-9.0.0` |
| 编译器 | host `g++ -std=c++17` |
| Python | 3.11（`/usr/local/python3.11.15/bin/python3`），矩阵测试需 `torch_npu` |

环境统一由 [`scripts/env.sh`](scripts/env.sh) 提供（`AB_SOC` / `AB_INC` / `ab_ccec` / `ab_cxx`）。

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
./scripts/build.sh all stride      # kernel + host + 测试；target: kernel|check|test|limits|rfft|probe|bw|stride|cube|all
make                                # 等价于上面（根目录 Makefile）
```

### 单点运行

```bash
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
│   └── host/                  可执行入口（fft_check / stride_probe / 基线）
├── tests/                    test_limits(18) / test_framework
├── config/                   硬件 profile + 设计空间 JSON（唯一真源）
├── scripts/                  构建 / 一键测试 / 基准 / η 标定 / 文档生成
├── docs/                     设计与实验文档（索引见 docs/README.md）
├── results/                  一键测试输出（git 忽略）
├── LICENSE                   Apache-2.0
└── THIRD_PARTY_NOTICES.md    cuButterfly 出处与 BSD-3-Clause 文本
```

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

---

## 文档

索引见 [`docs/README.md`](docs/README.md)。主要文档：

| 文档 | 内容 |
|---|---|
| [`docs/阶段0-1-发现与结果.md`](docs/阶段0-1-发现与结果.md) | 环境/硬件能力探测、原生 `aclRfft1D` 基线、从 0 到可跑通的全过程 |
| [`docs/性能优化-C2b与K择优.md`](docs/性能优化-C2b与K择优.md) | 5 轮 A/B 优化全记录：C2b、K 择优、radix-4 融合、批折叠、屏障、η 标定 |
| [`docs/trace与profile诊断-小尺寸与大尺寸.md`](docs/trace与profile诊断-小尺寸与大尺寸.md) | `msprof` 诊断：小尺寸 vs GPU、大尺寸 vs 原生、管线占用率 |
| [`docs/性能对比-标准库vs自研.md`](docs/性能对比-标准库vs自研.md) | 六基线 49 点同场对比（numpy/torch/`aclRfft1D`/v1/原生/自研） |
| [`docs/matrix_test_a7.md`](docs/matrix_test_a7.md) | 当前 49:0 矩阵（逐点） |
| [`docs/matrix_test_raw.md`](docs/matrix_test_raw.md) | 批折叠前的 44:5 基线存档（表不改，用于对照） |
| [`docs/矩阵测试与GPU绝对性能对比.md`](docs/矩阵测试与GPU绝对性能对比.md) | 与网上公开 GPU 工作的绝对性能对照 |
| [`docs/Cube张量化探针.md`](docs/Cube张量化探针.md) | fp32 Cube（矩阵单元）可行性探针 |

---

## License

本仓库以 [**Apache License 2.0**](LICENSE) 发布。

设计来源 **cuButterfly** 采用 **BSD 3-Clause**，版权与完整条款见
[**THIRD_PARTY_NOTICES.md**](THIRD_PARTY_NOTICES.md)。
