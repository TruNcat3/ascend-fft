# 术语

| 术语 | 本仓含义 |
|---|---|
| n / batch | 单次变换点数 / 等长度变换数量，不是float数组长度 |
| C2C / R2C / C2R | 复数到复数 / 实数到半谱 / 半谱到实数 |
| 半谱 | 频率0..n/2，用户行距n+2个float |
| AIV / AIC | 矢量核 / 矩阵核，具体能力由SoC和工具链决定 |
| UB / GM | 每核本地缓冲 / 全局设备内存；本目标UB不跨核共享 |
| Roofline / `I_L` | 层级 `L` 上由计算峰值与 `BW_L * I_L` 共同限定的吞吐上界；`I_L=W_P/Q_L`，必须说明流量层级 |
| payload 流量 / 实际事务 | 算法数组按物化边界计算的逻辑字节账本 / 受系数、padding、重复、事务粒度与缓存层级影响的硬件计数 |
| execution group `G` | 中间状态在组内驻留、组间完整物化的物理执行组；不等于逻辑分段数或 kernel 名称数 |
| K | 平面级列数，rows=n/K |
| D | 批折叠系数，使矢量组织覆盖多个连续batch |
| η | 模型预测微秒数，不是实测 |
| H/G/A/P/L/F/Q | 硬件/生成/映射/分解/布局/融合/质量对象 |
| Candidate / Plan | 配置描述 / 装载可执行资源的计划 |
| Infeasible | 已检查约束失败，保留原因 |
| Unverified / Feasible / Measured | 未验证 / 满足相应验证 / 已回填实测 |
| 历史字段 `native` | `torch.fft.fft` 经 torch_npu 执行，不是公开 CANN C2C C API；现行图表直接写具体 API，避免称谓混淆 |
| aclRfft1D | CANN实数到复数接口，不能无条件当作C2C基线 |
| v1 | 本仓标量旋转因子历史kernel，不是外部库 |
| device-only | 准备后的稳态变换/API 调用到同步，不含显式 H2D/D2H；可能含 dispatch 或必要 executor 构造，须查看具体协议 |
| E2E | H2D、变换、D2H；通常不含plan和输入生成 |
| pinned | 锁页主机内存，各基线须采用相同传输口径 |
| min-of-means | 多轮各取均值，再取最小轮均值 |
| A/B消融 | 相同协议交替基线与候选，隔离实现变化 |

调用契约见[变换](../user-guide/transforms.md)，计时定义见[实验协议](../benchmarks/methodology.md)。
