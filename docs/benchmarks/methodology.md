# 基准方法

本页定义 Ascend-FFT 发布结果的测量合同。任何 headline 数字都必须满足这里的条件；
不满足同一算法、数据类型、输入规模和计时区间的结果只能作为背景信息。

## 可比性分级

| 等级 | 条件 | 用途 |
|---|---|---|
| A：严格可比 | 同一 Ascend 设备、同一变换语义、同一精度、同一 shape、同一计时区间 | 计算胜负、speedup 和几何均值 |
| B：受限可比 | 同一设备，但变换或搬运量不同 | 解释固定开销或调用链，不参与胜负统计 |
| C：背景参考 | 不同硬件或引用第三方公开数据 | 说明量级，不合并进汇总图 |

当前 C2C 主基线是 `torch.fft.fft` 在 torch_npu 上调用的原生复数 FFT，属于 A 级。
`aclRfft1D` 是实数到半谱变换，与 C2C 不同，只能作为 B 级参照。GPU 结果属于 C 级。

## 测量对象

- C2C：复数 fp32、前向、交错布局 `[batch][2n]`。
- R2C：实数 fp32 输入，输出 `n/2+1` 个交错复数。
- C2R：半谱输入，输出实数并包含 `1/n` 归一化。
- 主网格：`n in {64,128,256,512,1024,2048,4096}`，
  `batch in {1,4,16,64,256,1024,4096}`。
- 正确性：相对双精度 CPU 参考，`maxRel <= 1e-4`。

## 两种计时区间

**Device-only** 只测 kernel launch 到 stream 同步。Plan 构造、输入生成、旋转因子和索引上传
均不在计时区间内。

**End-to-end** 测量 `H2D -> transform -> D2H`。三方使用 pinned host buffer，进程启动、
框架初始化、plan 构造和输入生成仍在计时区间外。两种区间不能混合计算 speedup。

## 统计规则

分两套协议，逐点记录进各自 raw 行，不混算：

- **C2C 矩阵 / 端到端**：每个点执行若干 warmup，然后进行 `rounds` 轮测量；每轮取 `reps`
  次均值，最终采用 min-of-means。C2C 的 native/self runner 使用记录在 summary 中的固定 seed
  逐轮随机交错，避免始终先跑完某一实现造成单向温度、频率或负载漂移。该规则用于降低共享
  服务器上的瞬态干扰，但不会替代对温度、频率和系统负载的记录。
- **R2C/C2R 对比（P1-B 对称协议）**：自研与 torch 两侧同一口径——5 个独立 trial，
  每个 trial 先做一次丢弃的 warmup 执行，再做 `reps` 次计时样本；trial 值取该 trial 样本的
  **median**，逐 trial 校验（自研侧 `maxRel <= 1e-4` 且 PASS 才准入 timing，`maxAbs`/
  `maxRel`/status 逐行留存）；跨 trial 汇总只由有效 raw 行派生，报告 median、p10、p90 与 CV。
  失败、超时、不支持的行保留在 `raw_trials` 中，不从分母抹去。

汇总 speedup 使用逐点比值的几何均值，同时报告最小值、最大值、胜出点数和总点数。

## 必需元数据

发布结果必须带 `manifest.json`，至少记录：Git 提交与 dirty 状态、UTC 起止时间、SoC/硬件 ID
与 profile 哈希、CANN/Python 等软件版本、逐实验命令、协议（protocol）、原始产物哈希与派生产物
哈希、变换语义、精度、shape 网格、warmup/reps/rounds、计时区间、基线、选择策略和正确性阈值。
`publish_results.py --check` 会在任何产物哈希、图形输入、归档图像输出或 manifest commit 漂移时失败。

当前发布快照的详细数值见[结果](results.md)，复现入口见[复现实验](reproducibility.md)。
