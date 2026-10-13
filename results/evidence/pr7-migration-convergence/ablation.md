# A2 小型受控消融：naive → 混合空间-时间映射 → 硬件约束映射

> 协议：[`docs/benchmarks/pr7-migration-convergence.md`](../../../docs/benchmarks/pr7-migration-convergence.md) §A2。
> 提交 `6716e0f`，硬件 Ascend910_9382（48 AIV），每点 3×rounds 取 min-of-means（`scripts/ab_test.py`）。
> 消融级：**v1** = naive planar lowering（`fft_radix2_v1.o`，早期级标量）→ **v2** = 8-plane
> 混合空间-时间映射（`fft_radix2_v2.o`）→ **current** = K-plane 硬件约束映射 + 模型选型
> （`fft_radix2.o`）。r2c 链的阶梯通过 `AB_FFT_O` 替换内层 `kfft_fwd(n/2)` 实现，
> 后处理 `kfft_r2c_post` 三级相同（其耗时即为该点无法被映射消除的固定链开销）。

## 消融级对比（kernel 时间与 E2E）

| 点 | naive v1 | mixed v2 | hw current | v1→v2 | v2→cur | v1→cur |
|---|---:|---:|---:|---:|---:|---:|
| C2C 1024×4096（边界点） | 12,329 µs | 7,062 µs | 343 µs | 1.75× | 20.6× | 35.9× |
| C2C 4096×4096 | 49,293 µs | 27,804 µs | 1,514 µs | 1.77× | 18.4× | 32.6× |
| C2C 64×4096 | 836 µs | 666 µs | 92 µs | 1.26× | 7.13× | 9.1× |
| C2C 128×4 | 54.2 µs | 79.3 µs | 47.0 µs | 0.68× ⚠ | 1.14× | 1.15× |
| R2C 128×4096（最慢 R2C 大 batch） | 974 µs | 791 µs | 211 µs | 1.23× | 3.75× | 4.62× |

E2E（`AB_E2E=3`，H2D+变换+D2H，min-of-5）：

| 点 | naive v1 | mixed v2 | hw current | v1→cur |
|---|---:|---:|---:|---:|
| C2C 1024×4096 | 13,907 µs | 8,648 µs | 1,903 µs | 7.3× |
| C2C 4096×4096 | 55,138 µs | 33,653 µs | 7,358 µs | 7.5× |
| C2C 64×4096 | 1,021 µs | 833 µs | 284 µs | 3.6× |
| R2C 128×4096 | 1,146 µs | 966 µs | 450 µs | 2.5× |

固定 CANN baseline（同点、同口径）：

| 点 | CANN device | CANN E2E | 自研 device | 自研 E2E | device 比值 |
|---|---:|---:|---:|---:|---:|
| C2C 1024×4096 | 339.9 µs | 1,774.9 µs | 352.8 µs | 1,853.1 µs | 0.96× |
| C2C 4096×4096 | 见 `matrix.md`（n=4096 行） | — | 见 `matrix.md` | — | 1.55× |
| R2C 128×4096（aclRfft1D） | 118.7 µs | — | 236.0 µs | 450.2 µs | 0.50× |

## AIV/MTE 利用率（msprof，`profiles/`，sum_prof 汇总）

| 点·级 | kernel | vec | scalar | MTE2 | MTE3 |
|---|---|---:|---:|---:|---:|
| C2C 1024×4096 · v1 | kfft_fwd 12,278 µs | 0.101 | 0.903 | 0.001 | 0.001 |
| C2C 1024×4096 · v2 | kfft_fwd 7,016 µs | 0.218 | 0.790 | 0.002 | 0.002 |
| C2C 1024×4096 · cur | kfft_fwd 294 µs | **0.862** | 0.175 | 0.059 | 0.052 |
| R2C 128×4096 · v1 | fwd 797 µs + post 120 µs | 0.087 | 0.889 | 0.013 | 0.014 |
| R2C 128×4096 · v2 | fwd 609 µs + post 120 µs | 0.332 | 0.660 | 0.017 | 0.018 |
| R2C 128×4096 · cur | fwd 57 µs + post 120 µs | 0.665 / 0.518 | 0.471 / 0.288 | 0.066 / 0.087 | 0.061 / 0.286 |

## GM/UB 搬运量（payload 模型，非 profiler 实测）

模型：`launches × 2 × n × batch × 8` 字节（descriptor 契约的 payload GM RW，**不含**旋转因子/
索引等辅助事务；v2/current 的 Gather 索引在 batch 循环外一次性载入 UB，不随 batch 增长）。

| 点 | launches | GM payload | UB v1 | UB v2 | UB current |
|---|---:|---:|---:|---:|---:|
| C2C 1024×4096 | 1 | 64.0 MiB | 30.1 KiB | 44.1 KiB | 47.2 KiB (K=16) |
| C2C 4096×4096 | 1 | 256.0 MiB | 120.1 KiB | 176.1 KiB | 187.5 KiB (K=16) |
| C2C 64×4096 | 1 | 4.0 MiB | 2.0 KiB | 2.9 KiB | 3.5 KiB (K=8) |
| R2C 128×4096 | 2 | 16.0 MiB | 3.9 KiB | 5.6 KiB | 6.4 KiB (K=8) |

## 模型选择、有限搜索与选择损失

模型选择 = `bfly::planeKFor(n)`（算子调用数模型，K∈{8,16,32}）× 批折叠 D（静态规则，
只看 n，内核/宿主/框架三处同源）。有限搜索 = K×D 网格实测 min。

| 点 | 模型选择 | 模型档实测 | 搜索最优 | 选择损失 |
|---|---|---:|---|---:|
| C2C 1024×4096 | K=16, D=4 | 349.8 µs | K16×D4 349.8 µs | **≈0%** |
| C2C 4096×4096 | K=16, D=1 | 1515.7 µs | K16×D1（K32=1754 µs 更差） | **≈0%** |
| R2C 128×4096（inner n=64） | K=8（唯一合法档，rows=8） | 230.3 µs | K=8 | **0%** |

选择损失实测明细见 `ablation_search_1024x4096.csv`（K∈{8,16,32}×D∈{1,2,4,8}）。

## 配置选择原因与负点解释

- **为什么 v1→v2 是中等收益（1.2–1.8×）、v2→current 是主要收益（7–21×）**：
  v1 早期级（h=1,2,4）受 32B 对齐约束只能标量，scalar 占用率 0.79–0.90；
  v2 用 8-plane 把早期级搬进同一 tile 的不同 plane 从而矢量化（vec 0.22–0.33），
  但仍保留 planar↔交错 Gather 与逐级标量转换；current 进一步 K-plane 择优 +
  全 Gather 化索引（batch 循环外一次载入）+ radix-4 融合，vec 升至 0.66–0.86、
  scalar 降至 0.18–0.47。
- **C2C 1024×4096 仍 0.96× 的原因（已知负点）**：该点 payload 64 MiB、
  kernel 343 µs 已接近 CANN 340 µs；剩余差距来自 scalar 尾部（0.175）与
  MTE2/3 仅 0.06 级的搬运并行不足，加上固定启动/栅栏开销——映射选择本身
  已无损失（选择损失≈0），负点由微架构并行度上限而非映射造成。
- **R2C 128×4096 仍 0.50×（vs aclRfft1D）的原因**：内层 FFT 只有 n=64、
  kernel 57 µs，而 `kfft_r2c_post` 固定 ~120 µs 与 batch 无关的表/分支开销
  占链 68%；该固定链开销不随映射级变化（三级 post 都是 120 µs），
  属于"当前映射无法消除"的结构性负点。
- **128×4 上 v1→v2 回退（0.68×）**：n 太小、8-plane Gather 固定开销
  无法被本已标量化的 3 级摊薄；current 以更小 K 与合并路径恢复 1.15×。
  小点不是该方法的目标区间（发布矩阵中小点对 CANN 全部 ≥1×，见 `matrix.md`）。
- **128×4 的正确性**：三级 maxRel 均 ≤2.36e-07 ≪ 1e-4，全部 PASS。

## 产物

- `ablation_v1_v2.json` / `ablation_v2_current.json` —— `scripts/ab_test.py` 原始 A/B
- `ablation_r2c_ladder.txt` —— r2c 内层阶梯
- `ablation_e2e.txt` —— E2E 阶梯
- `ablation_prof_labeled.txt` / `ablation_prof.csv` / `profiles/` —— msprof AIV/MTE
- `ablation_search_1024x4096.csv` —— K×D 有限搜索
