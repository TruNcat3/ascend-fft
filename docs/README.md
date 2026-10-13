# 文档阅读顺序

在线入口是 [GitHub Pages 文档站](https://truncat3.github.io/ascend-fft/)，
仓库内对应[文档首页](index.md)。按阅读目的选择下面的一条路线，不必先读历史探索记录。

## 使用算子库

[安装](getting-started/installation.md) → [首次运行](getting-started/quickstart.md) →
[变换契约](user-guide/transforms.md) → [Plan 与选型](user-guide/plans.md) → [API](reference/api.md)。

遇到环境或运行问题时查[故障排查](reference/troubleshooting.md)；在选择输入前确认[支持范围](reference/support.md)。

## 理解迁移方法

[设计动机](design/motivation.md) → [架构与数据流](design/architecture.md) →
[计算核心与硬件映射](design/kernels.md) → [性能模型](design/performance-model.md)。
[实数变换设计](design/real-transforms.md)说明 R2C/C2R 如何复用计算流。

## 阅读实验报告

1. [实验设计与结果汇总](benchmarks/results.md)：先了解研究问题、覆盖范围和主要结论。
2. [测量协议](benchmarks/methodology.md)：确认基线、计时区间、统计方法和正确性门槛。
3. [C2C 性能](benchmarks/c2c.md)：比较长度与 batch 对设备侧性能的影响。
4. [端到端与应用](benchmarks/end-to-end.md)：区分计算收益、传输开销和应用代理工作负载。
5. [R2C/C2R](benchmarks/real-transforms.md)：查看实数变换的同语义基线比较。
6. [规模切片](benchmarks/scale.md)：检查更大尺寸的正确性和执行能力，不把无基线切片当作加速证明。
7. [原始数据与复现](benchmarks/reproducibility.md)：查完整数据、发布快照和重跑命令。

[相关工作与基线](benchmarks/related-work.md)解释库与文献的比较边界。
主报告按实验目的说明结果，逐点大表和历史过程不作为阅读前置条件。

## 开发与后续工作

[工程结构](development/repository-layout.md) → [新增设计点](development/adding-design-points.md) →
[测试策略](testing.md) → [验证与发布](development/validation-and-release.md)。
性能问题查[诊断指南](development/profiling.md)，尚未完成的工程与研究目标见[未来计划](roadmap.md)。

[历史归档](archive/decisions.md)用于追溯设计取舍；长 FFT 和 PR 7 的阶段计划保留在文档站的历史归档分组中，
不替代当前结果或支持范围。[项目与作者](about.md)记录 cuButterfly 来源、作者信息和引用方式。

本页只维护阅读路线；命令、配置、实验数值与术语以对应专题页为准。
