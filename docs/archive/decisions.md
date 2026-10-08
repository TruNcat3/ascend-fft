# 研究决策归档

本页只保留影响当前设计的决策，不按每轮实验追加正文。当前解释从 [设计动机](../design/motivation.md) 开始；性能结论以 [实验结果](../benchmarks/results.md) 与 [测试方法](../benchmarks/methodology.md) 的协议和数据为准。

方法来源：[cuButterfly](https://github.com/TruNcat3/cuButterfly)。Ascend 迁移保留阶段/数据空间-时间映射，重新 lowering 到 AIV/UB/MTE；不能把它理解成复制 CUDA warp 实现。

## 重要决策与修正

| 决策 | 证据/原因 | 当前含义 | 历史记录 |
|---|---|---|---|
| 先探测硬件再限定候选 | UB、对齐、SIMT 与指令字段约束影响正确性 | 设备 profile 需重新验证；不是通用硬编码 | [阶段0-1](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/阶段0-1-发现与结果.md) |
| 用 K-plane 修复小距离访问 | 任意 UB 偏移不满足矢量对齐 | 布局是架构变量，不是无代价的格式转换 | [早期布局演进](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/阶段0-1-发现与结果.md) |
| host 生成索引、kernel Gather | 标量散射/索引成本较高 | 数据交换由合法局部路径实现 | [Gather 决策](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/阶段0-1-发现与结果.md) |
| K 模型随核心优化更新 | 旧指令计数曾选错参数 | 核心、布局、模型必须共同维护 | [K 择优](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/性能优化-C2b与K择优.md) |
| 局部 radix-4 后来采用 | 早期通用融合负收益，后续平面级实现降低发射 | 不能把某次实现结论外推成 radix 的永久结论 | [融合修正](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/性能优化-C2b与K择优.md) |
| 小长度采用 batch 折叠 | 减少重复发射，但受步长/repeat 与活跃核数限制 | 折叠不是越多越好，低 batch 需要收窄 | [批折叠](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/性能优化-C2b与K择优.md) |
| 保留必要 prologue/收尾屏障 | 删除排空曾产生间歇错误 | 正确性优先，不能据单次 PASS 删同步 | [屏障消融](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/性能优化-C2b与K择优.md) |
| Cube 能力判断被修正 | 纯 AIV 不可直达；独立 AIC 探针可运行 fp32 链路 | 基础可用，不代表生产 FFT split-core 已完成 | [Cube 探针](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/Cube张量化探针.md) |
| 小 batch 与大 batch 分别诊断 | 小批受启动/活跃核限制，大批暴露稳态发射成本 | 不把跨硬件绝对时延当同硬件实现收益 | [trace 诊断](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/trace与profile诊断-小尺寸与大尺寸.md) |
| r2c/c2r 复用复数核心 | 半谱预后处理与符号控制可独立实现 | 公共语义、设备 padding 与端到端成本分别说明 | [实数链](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/实数变换-r2c与c2r.md) |

## 怎样阅读历史数字

旧记录覆盖多次核心、模型与计时协议变更，同一形状的数字不能直接拼成当前比较表。原始轮次属于当时状态，保留用于解释决策，而不是作为最新 API 或安装指南。

完整历史正文固定到 [9dccfec](https://github.com/TruNcat3/ascend-fft/tree/9dccfec/docs)。其中 [演进时间线](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/设计思路与演进.md) 和 [GPU 绝对性能对比](https://github.com/TruNcat3/ascend-fft/blob/9dccfec/docs/矩阵测试与GPU绝对性能对比.md) 可供追溯；当前文档不继续复制长篇逐轮日志。
