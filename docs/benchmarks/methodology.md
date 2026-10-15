# 基准方法

本页定义 Ascend-FFT 发布结果的测量合同。任何 headline 数字都必须满足这里的条件；
不满足同一数学变换/数值语义、数据类型、输入规模和计时区间的结果只能作为受限对照或背景
信息。比较不要求双方使用相同 radix 或局部实现；否则无法公平比较可替换的算法核心。

本合同服务于“迁移实例”的收敛目标：先固定一个可审计的实验范围，再在范围内完成候选搜索和
必要优化，争取每个晋升点相对严格基线有正收益。范围外的尺寸、SoC 或核心实现不参与当前版本
的最优性声明；范围内的回退点必须保留、解释并列入下一轮，而不能通过删点或改变统计口径隐藏。

## 可比性分级

| 等级 | 条件 | 用途 |
|---|---|---|
| A：严格可比 | 同一 Ascend 设备、同一变换语义、同一精度、同一 shape、同一计时区间和同一汇总规则 | 计算主结论的胜负、speedup 和几何均值 |
| B：受限可比 | 同一设备，但变换、搬运量或统计协议至少一项不同 | 单独报告并解释调用链；不与 A 级结果合并 |
| C：背景参考 | 不同硬件或引用第三方公开数据 | 说明量级，不合并进汇总图 |

当前 C2C 主基线是 `torch.fft.fft` 在 torch_npu 上调用的原生复数 FFT，属于 A 级。
`aclRfft1D` 相对 C2C 的变换语义不同；相对 R2C 虽然语义一致，当前快照却使用不同的采样与
汇总协议。因此两种场合都按 B 级单独报告，不与 A 级结果合并。GPU 结果属于 C 级。
结果表中的 `A*` 表示 A 级语义和计时区间成立、汇总量同为 min-of-means，但历史采集的内层
重复次数或执行顺序不完全对称；它必须附具体限制，不能简写成完全同协议。

## 测量对象

- C2C：复数 fp32、前向、交错布局 `[batch][2n]`。
- R2C：实数 fp32 输入，输出 `n/2+1` 个交错复数。
- C2R：半谱输入，输出实数并包含 `1/n` 归一化。
- 主网格：`n in {64,128,256,512,1024,2048,4096}`，
  `batch in {1,4,16,64,256,1024,4096}`。
- 应用代理：OFDM、Radar、DL frequency-domain layer 各 4 个预声明 shape。
- 规模切片：`n in {8192,16384,32768,65536}`，`batch in {1,16,256}`；该集合当前没有
  外部基线，只报告正确性和延迟。
- 正确性：相对双精度 CPU 参考，`maxRel <= 1e-4`。

## 两种计时区间

**Device-only** 测量准备完成后的稳态变换/API 调用，终点为 stream 同步；它排除显式 H2D/D2H、
输入生成和自研 Plan 的一次性构造、旋转因子/索引上传。该区间不是统一的“纯 kernel 时间”：
自研 C2C 路径主要包含 launch 与同步，Torch 路径包含框架 dispatch，裸 `aclRfft1D` 还必须在
每次调用中执行 `aclRfft1DGetWorkspaceSize` 并创建 executor。它比较的是各自可调用的稳态设备
变换路径，而不能直接解释为局部算术核心吞吐。

**End-to-end** 测量 `H2D -> transform -> D2H`。三方使用 pinned host buffer，进程启动、
框架初始化、plan 构造和输入生成仍在计时区间外。两种区间不能混合计算 speedup。

## 统计规则

分两套协议，逐点记录进各自 raw 行，不混算：

- **C2C device-only 矩阵**：每个点先 warmup，再进行 3 轮、每轮 20 次；每轮取均值，最终采用
  min-of-means。历史 raw runner ID `native/self` 中，`native` 明确表示
  `torch.fft.fft`/torch_npu 基线；二者使用记录在 summary 中的固定 seed 逐轮随机交错，避免
  始终先跑完某一实现造成单向温度、频率或负载漂移。
- **端到端与应用 shape（`20261013` 快照）**：两侧同样按 3 轮 min-of-means 汇总，但内层重复数
  不相同：torch_npu 基线为每轮 20 次，自研为每轮 10 次，网格实验中的裸 ACL 为每轮 10 次（应用 shape
  不运行 ACL）；执行顺序为先完成全部 native 轮次，再执行裸 ACL 和自研，未随机交错。该历史协议和执行顺序必须随结果披露，不能
  写成与 C2C matrix 完全相同；未来重采集应统一重复数并交错执行。

随机交错用于降低共享服务器上的瞬态干扰，但不会替代对温度、频率和系统负载的记录。
- **R2C/C2R 对比（P1-B 对称计时协议）**：自研与 torch 两侧采用 3 个独立 trial，
  每个 trial 先做一次丢弃的 warmup 执行，再做 `reps` 次计时样本；trial 值取该 trial 样本的
  **median**。Ascend-FFT 每个 trial 都执行 correctness gate，失败的 trial 不准入 timing；Torch
  基线在每个 shape 计时前做一次 NumPy 语义预检，而不是每个 trial 重复正确性计算。两侧的
  `maxAbs`、`maxRel`、status 和失败原因按其协议保留；失败、超时或不支持的行不从分母抹去。
- **规模切片**：3 个 trial；设备时间先在每个 trial 内取最小值，再跨 trial 取中位数，亦即
  *median of per-trial minima*。host wall time 逐 trial 保留。该实验无基线，不计算 speedup。

汇总 speedup 使用逐点比值的几何均值，同时报告最小值、最大值和覆盖点数。严格胜/平/负按
速度比与 1 的关系统计；工程持平另用 `0.95x..1.05x` 区间统计。两套计数必须分别标注。

## 必需元数据

发布结果必须带 `manifest.json`，至少记录：Git 提交与 dirty 状态、UTC 起止时间、SoC/硬件 ID
与 profile 哈希、CANN/Python 等软件版本、逐实验命令、协议（protocol）、原始产物哈希与派生产物
哈希、变换语义、精度、shape 网格、warmup/reps/rounds、计时区间、基线、选择策略和正确性阈值。
`publish_results.py --check` 会在任何产物哈希、图形输入、归档图像输出或 manifest commit 漂移时失败。

当前发布快照的详细数值见[结果汇总](results.md)，复现入口见[原始数据与复现](reproducibility.md)。
