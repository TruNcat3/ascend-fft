# 实数变换：R2C 与 C2R

## 实验问题

同一架构映射思路能否推广到实数 FFT 的前后处理与布局转换？本实验将 R2C 和 C2R
分开比较，并为 R2C 区分 Torch 调用路径与裸 CANN API，避免把不同基线混成一个均值。

## 配置与变量

| 项目 | R2C | C2R |
|---|---|---|
| 语义 | fp32 实数输入，输出 N/2+1 个复数 | fp32 合法半谱输入，输出 N 个实数，包含 1/N 归一化 |
| 长度 N | 128..8192 的二次幂，7 个长度 | 64..4096 的二次幂，7 个长度 |
| Batch | 1、4、16、64、256、1024、4096 | 同左 |
| Torch 基线 | 同卡 `torch.fft.rfft` / torch_npu | 同卡 `torch.fft.irfft` / torch_npu |
| 次级对照（B 级） | 裸 CANN `aclRfft1D`，42 个同语义共同支持点，但统计协议不同 | 未纳入其他对照 |

硬件为 Ascend910_9382，软件为 CANN 9.0.0，计时为 device-only。每侧每点 3 个独立 trial，
每个 trial 丢弃 warmup 后取计时样本中位数，再取跨 trial 中位数；重复次数随总点数为
50、30 或 10，完整策略记录于快照协议。

自研侧每个 trial 保留正确性状态与误差；Torch 侧按 shape 做一次独立参考预检后计时，
**不是逐 trial 正确性检查**。ACL 基线来自同次运行的 `e2e.json`，其采集与汇总协议仍为
3轮 min-of-means；虽然设备、变换和形状匹配，统计规则并不与 Torch 的 median 完全相同，
因此 ACL 属于同语义但异统计协议的 B 级次级对照，两种 R2C 基线分开报告，ACL 行不与 A 级
Torch 行合并为一个总体均值。

## 全网格收益与负点

![R2C/C2R device-only speedup heatmap](../figures/fig10_real_speedup_heatmap.png)

速度比为 `各自基线耗时 / Ascend-FFT 耗时`。未支持或没有对应基线的格子不是零耗时，
也不纳入几何均值。R2C 的8192点没有对应 ACL 数据，不能将42点 ACL结果外推到49点。

## 数值结论

| 对比 | 配对点数 | 几何均值 | 严格胜 / 平 / 负 | ±5% 胜 / 平 / 负 |
|---|---:|---:|---:|---:|
| R2C vs Torch | 49 | **1.180x** | **39 / 0 / 10** | **34 / 7 / 8** |
| R2C vs ACL | 42 | **1.858x** | **37 / 0 / 5** | **36 / 1 / 5** |
| C2R vs Torch | 49 | **1.868x** | **47 / 0 / 2** | **45 / 2 / 2** |

严格统计以未舍入速度比与1比较；±5%统计将 `0.95..1.05` 视为近似持平，仅用于描述。
R2C/Torch最低值为0.440x，R2C/ACL最低值为0.505x，C2R/Torch最低值为0.823x。
总体均值支持实数变换迁移具有收益，但它没有隐藏这些仍需定向优化的边界点。

主网格与补充长度合并后共有644条 raw记录，其中630条有效、14条为显式 unsupported。
有效自研 trial 均通过 `maxRel <= 1e-4`；本次自研最大记录误差为 **3.122e-7**。
未支持项保留状态，而不是从完整研究记录删除。

## 限制与解释

R2C 的FFT与后处理成本、半谱布局和批折叠，C2R 的逆向重构与归一化，使它们不能直接与
C2C速度比混合统计。整体比较也不能单独证明融合writer、MTE流水或某个核心选项的贡献；
未通过稳定路径晋升门槛的候选仍属于研究前沿。

本次 Torch precheck 使用独立输入，只支撑约定的一次每shape基线验证；它不等价于每个计时
trial 都做参考校验。当前数字适用于本次支持长度、稠密1D布局和SoC，不保证其他精度、
非二次幂、stride或其他设备具有相同收益。

## 原始数据与复现

- [合并后的逐点结果](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/r2c_c2r.json)
- [644条原始 trial](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/r2c_c2r.raw_trials.csv)
- [补充长度与预检记录](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/r2c_c2r_suppl.json)
- [同次 ACL 来源](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/e2e.json)
- [实数变换设计与接口语义](../design/real-transforms.md)
- [复现实验入口](reproducibility.md)

下一项实验：[规模扩展切片](scale.md)。
