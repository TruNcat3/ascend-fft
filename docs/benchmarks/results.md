# 实验设计与结果汇总

> **当前发布快照**：`ascend910_9382-migration-20261013` · Ascend910_9382 ·
> CANN 9.0.0 · fp32。所有本页数值均来自同一不可变快照；旧快照只作为历史记录，
> 不与当前结果拼接。

Ascend-FFT 是 [cuButterfly](https://github.com/TruNcat3/cuButterfly) 混合空间-时间数据流方法
在 Ascend NPU 上的迁移实例。实验要回答的不是“一个固定 kernel 是否处处最优”，而是：在预先
声明的硬件、变换语义和 shape 范围内，硬件 profile、参数化候选与实测选型能否形成正确、可复现、
总体有收益的执行方案。

## 实验问题

当前报告按四个问题组织，避免把协议、过程记录和原始大表混在主结论中：

1. **C2C 主矩阵**：同卡、同语义的 device-only 路径是否获得总体收益？
2. **端到端与应用 shape**：加入 H2D/D2H 后收益保留多少，在代表性 shape 上是否成立？
3. **R2C/C2R**：同一计算流方法迁移到实数变换后，相对严格基线和次级对照表现如何？
4. **规模切片**：当前实现能否正确运行到 `N=65536`，其延迟如何随长度和 batch 变化？

统计合同、可比性等级和 correctness gate 见[测量协议](methodology.md)。速度比统一定义为
`基线耗时 / Ascend-FFT 耗时`，大于 1 表示 Ascend-FFT 更快。

## 结果总览

![五类同卡实验相对匹配 torch_npu FFT API 的几何均值 speedup 汇总](../figures/overview_performance.png)

主图只放入同一前端/后端家族的匹配对照：C2C 使用 `torch.fft.fft`，R2C/C2R 分别使用
`torch.fft.rfft/irfft`，均由 torch_npu 在同卡执行。过去文档把 C2C 柱写成“CANN native”，
但它实际是 torch_npu op-plugin，而非裸 CANN C API；这种混合命名会造成基线层级不同的错觉。

| 实验 | 比较基线 | 可比性 | 点数 | 几何均值 | 严格胜/平/负 | ±5% 胜/平/负 |
|---|---|---:|---:|---:|---:|---:|
| C2C device-only | `torch.fft.fft`（torch_npu） | A | 49 | **1.555x** | 47/0/2 | 46/2/1 |
| C2C end-to-end | `torch.fft.fft`（torch_npu），pinned H2D + FFT + D2H | A* | 49 | **1.442x** | 47/0/2 | 45/4/0 |
| 应用 shape end-to-end | `torch.fft.fft`（torch_npu），OFDM / Radar / DL 代理 shape | A* | 12 | **1.422x** | 12/0/0 | 12/0/0 |
| R2C device-only | `torch.fft.rfft`（torch_npu） | A | 49 | **1.180x** | 39/0/10 | 34/7/8 |
| C2R device-only | `torch.fft.irfft`（torch_npu） | A | 49 | **1.868x** | 47/0/2 | 45/2/2 |

`A*` 表示变换语义、设备、精度、shape、计时区间和 min-of-means 汇总一致，但当前归档中历史
runner ID `native/self`（其中 `native` 即 torch_npu 路径）的内层重复数分别为 20/10，且没有
交错执行；这两行保留为同语义对比，同时明确其
采样限制。

### 次级 ACL 对照

| 实验 | 比较基线 | 可比性 | 点数 | 几何均值 | 严格胜/平/负 | ±5% 胜/平/负 |
|---|---|---:|---:|---:|---:|---:|
| R2C device-only | 裸 CANN `aclRfft1D` | B | 42 | **1.858x** | 37/0/5 | 36/1/5 |

该行与 R2C/Torch 主实验的变换语义和设备一致，但来自另一套 min-of-means 采集与汇总协议，
因此不进入首页主图，也不和五项 A/A* 结果计算联合均值。它用于回答“相对公开 ACL R2C API
表现如何”，不是供首页逐柱横向排名的替代基线。

“严格胜负”中的“严格”只表示按速度比是否严格大于 1 计数，不表示该行必然属于 A 级基线；
±5% 工程持平区间把 `0.95x..1.05x` 视为工程持平。
二者回答不同问题，不能互换。几何均值保留全部有效点，包括负点。

## 可以得出的结论

- **迁移成立**：C2C、应用 shape、R2C 和 C2R 的同卡同语义集合均取得大于 1 的几何均值，
  说明混合数据流方法可以在 Ascend 的 AIV/UB/MTE/GM 约束下形成有效实现。
- **不是全面逐点领先**：C2C 仍有 2 个严格负点；R2C 相对 torch_npu 有 10 个严格负点，
  其中多个位于小长度、大 batch。报告保留这些点，不用删点或只报均值掩盖边界。
- **端到端收益低于 device-only 收益**：49 点同步 device-only 复测为 1.644x，而 host-to-host
  E2E 为 1.442x，说明传输和同步会稀释 kernel 收益。
- **应用代理 shape 有一致收益**：12 个预声明 shape 的 E2E 几何均值为 1.422x，12/12 严格领先；
  这不等价于完整通信、雷达或深度学习应用的端到端加速。
- **规模证据尚不含加速比**：`N=8192..65536` 的 12 点全部通过正确性，但没有同协议外部基线，
  因此只支持“可正确运行及其延迟趋势”，不支持“长尺寸领先”。

## 分项阅读

| 页面 | 主要内容 |
|---|---|
| [实验一：C2C](c2c.md) | 49 点 device-only 网格、负点与 shape 趋势 |
| [实验二：端到端与应用](end-to-end.md) | 49 点 E2E、搬运占比与 12 个应用代理 shape |
| [实验三：R2C/C2R](real-transforms.md) | torch_npu / ACL 基线、实数变换边界与负点 |
| [实验四：规模切片](scale.md) | `8192..65536` 正确性和延迟，不含外部 speedup |
| [原始数据与复现](reproducibility.md) | 命令、manifest、逐点 JSON/CSV 与哈希 |

## 结论边界

本轮结果适用于 Ascend910_9382、CANN 9.0.0、fp32 和各分项页面声明的语义与计时区间。
它证明了一个有总体收益的跨平台迁移实例，但不证明所有 Ascend SoC、所有长度、所有精度、
所有 batch 或所有专业 FFT 库上的全局最优。跨硬件必须重新运行 profile、模型校准与候选实测。
