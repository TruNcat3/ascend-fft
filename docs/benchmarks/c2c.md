# C2C：同语义基线对比

## 实验问题

将 cuButterfly 的阶段/数据混合空间-时间映射迁移到 Ascend 后，在不同 FFT 长度和 batch
上能否获得实际收益？本实验比较完整前向 C2C 执行路径，而不是比较不同变换或跨硬件峰值。

## 配置与变量

| 项目 | 本次配置 |
|---|---|
| 硬件与软件 | Ascend910_9382，48 AIV，CANN 9.0.0 |
| 语义 | fp32，前向 1D C2C，稠密交错输入，自然序输出 |
| 长度 N | 64、128、256、512、1024、2048、4096 |
| Batch | 1、4、16、64、256、1024、4096 |
| 严格基线 | 同卡 `torch.fft.fft` 的 torch_npu / CANN 原生复数 FFT 路径 |
| 计时 | Device-only 同步执行；不含 Plan 构造、输入生成和 H2D/D2H |
| 重复与汇总 | 每点双方各 3 轮，每轮 20 次；取每轮均值的最小值 |
| 正确性 | 自研与原生分别对 CPU 双精度参考，所有轮次 PASS 且 `maxRel <= 1e-4` |

两侧逐轮按记录的固定 seed 交错运行。49 个语义点产生 294 条原始记录；本页使用
`ascend910_9382-migration-20261013` 快照，不与旧日期的性能值拼接。

## 长度与 batch 的收益分布

![C2C device-only speedup heatmap](../figures/fig1_speedup_heatmap.png)

速度比为 `基线耗时 / Ascend-FFT 耗时`，大于 1 表示 Ascend-FFT 更快。热力图展示全部
49 点，而不是只选择领先配置；大 batch 的接近持平和负点同样保留。

## 数值结论

| 指标 | 最新快照 |
|---|---:|
| 几何均值 speedup | **1.555x** |
| 严格胜 / 平 / 负 | **47 / 0 / 2** |
| ±5% 持平带的胜 / 平 / 负 | **46 / 2 / 1** |
| 最低 / 最高 speedup | **0.942x / 2.046x** |
| 正确性 | **294/294 轮 PASS** |

严格统计以 `speedup > 1`、`= 1`、`< 1` 分类；持平带统计将 `0.95..1.05` 视为近似
持平。它只是描述性阈值，不是统计显著性或误差等价性证明。

两个严格负点为 `N=1024,batch=4096`（0.942x）和 `N=4096,batch=1024`（0.999x）。
后一项在两位小数表中显示为 1.00x，但未舍入比值仍小于 1。总体结果支持迁移后的执行路径
在声明网格上具有普遍收益，不支持“每一点都超过基线”的说法。

## 限制与解释

二维映射、UB 内复用与 Ascend lowering 是收益的设计依据，但这张整体比较图不能独立量化
每一项的贡献；机制归因需要受控消融，不能把总体 speedup 全部记到单个设计因素上。
本次 device-only 计时仍包含所测调用路径的启动与同步，不能等价为纯向量指令吞吐。

本实验只覆盖一台 SoC、fp32、前向稠密 1D 二次幂 C2C。更长 FFT、更多精度、逆向、二维和
其他 SoC 不进入这份几何均值；它们也不能由当前结果外推为已验证能力。

## 原始数据与复现

- [49 点可读矩阵](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/matrix.md)
- [快照矩阵 CSV](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/matrix.csv)
- [294 条逐轮原始记录](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/trials.csv)
- [协议与运行来源](https://github.com/TruNcat3/ascend-fft/tree/master/results/published/ascend910_9382-migration-20261013)
- [复现实验入口](reproducibility.md)

下一项实验：[端到端与应用代理](end-to-end.md)。
