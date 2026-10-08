# 当前结果

> **发布快照**：Ascend910_9382（48 AIV），CANN 9.0.0，fp32，前向变换。
> C2C 主网格使用 49 个 `(n, batch)` 点；完整协议见[基准方法](methodology.md)。

![Ascend-FFT performance overview](../figures/overview_performance.png)

## 结论速览

| 工作负载 | 严格基线 | 点数 | 几何均值 speedup | 正确性 |
|---|---|---:|---:|---:|
| C2C device-only | CANN 原生复数 FFT（torch_npu） | 49 | **3.01x** | 49/49 |
| C2C end-to-end | 同上，pinned H2D + FFT + D2H | 49 | **1.66x** | 49/49 |
| 应用 shape end-to-end | 同上，OFDM / 雷达 / DL 频域层 | 12 | **1.70x** | 12/12 |
| R2C device-only | `torch.fft.rfft` | 42 | **1.84x** | 42/42 |
| C2R device-only | `torch.fft.irfft` | 49 | **3.06x** | 49/49 |

C2C device-only 的 49 个点全部快于原生基线，范围为 `1.04x..6.86x`；端到端有
46/49 个点胜出，另外 3 点为 `0.95x..0.99x`。这说明 kernel 优势已经成立，但大 batch 的
传输占比会压缩端到端收益。

## 如何阅读结果

- **当前结论只来自已发布网格**，不会通过追加有利 shape 改写平均值。
- R2C 还可与裸 `aclRfft1D` 比较，几何均值为 `3.21x`，但该结果不与 C2C 汇总。
- CPU、GPU 和裸 CANN 的不同语义数据保留在生成附件中，不进入严格胜负统计。
- 成本模型在 49 点上的平均偏差为 7.7%，43/49 个点位于 +/-15% 内；模型用于候选预排序，
  最终选择仍由实测回填确认。

## 完整数值

- [C2C 49 点矩阵](../generated/matrix.md)
- [多基线明细](../generated/comparison.md)
- [端到端发布数据](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-cann9.0.0/e2e.json)
- [应用 shape 发布数据](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-cann9.0.0/e2e_app.json)
- [发布 manifest](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-cann9.0.0/manifest.json)
- [R2C/C2R 历史原始数据](https://github.com/TruNcat3/ascend-fft/blob/master/results/r2c_c2r.json)（旧格式缺少逐行门禁字段，未纳入新 snapshot）

图和表必须与对应发布快照一起引用；禁止从不同日期或不同协议的文件拼接结论。
