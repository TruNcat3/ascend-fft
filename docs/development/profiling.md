# 性能剖析

## 硬件事实

```bash
scripts/hw_probe.sh
scripts/hw_probe.sh --only bw
scripts/hw_probe.sh --only cube
```

探针用于确认 AIV 数量、UB、对齐、mask/repeat 上限、SIMT 可用性和传输带宽。探针结论进入
hardware profile 前必须记录 SoC 与 CANN 版本，不能把单机观察直接当作所有 Ascend 的常量。

## msprof

```bash
scripts/profile_test.sh
python3 scripts/sum_prof.py results/profiles/<run-id>
```

诊断按 GM 搬运、MTE2/MTE3、vector、scalar、同步和 host 调用开销拆账。先确认计时区间和
任务数，再解释利用率；单独看一个百分比不能证明瓶颈。

## 用 profile 检验 Roofline 推导

[方法页](../design/motivation.md#roofline-insight)中的
`Q_GM,payload = 2*B*N*q*G` 是完整数组边界的简化逻辑 payload 账本，不是硬件计数器读数。诊断时按
下面顺序把“工作点向右移动”与“最终变快”分开验证：

| 问题 | 需要记录的证据 | 能得出的结论 |
|---|---|---|
| 是否真的减少物化边界？ | manifest 的 execution groups、边界读写语义、kernel/task 数 | 逻辑 payload 是否下降 |
| 实际 GM 事务是否同步下降？ | 同一计时区间的 GM/MTE bytes、事务或独立带宽探针 | padding、非合并访问和系数流是否抵消收益 |
| 减少 GM 后卡在哪里？ | Vector/Scalar 发射、MTE 活跃区间、同步/等待、任务时间 | 瓶颈是否移到 UB、issue、dependency 或 tail |
| 驻留是否破坏并发？ | UB 峰值、活跃 AIV/任务分布、每核工作组数 | 较少边界是否以容量或利用率下降为代价 |
| 流水是否真正重叠？ | 同一 trace 中 MTE 与 Vector 的时间区间和缓冲所有权 | 只能由时间重叠证明，不能由双缓冲字段名证明 |

比较两个候选时，先固定变换语义与计算核心，核对 `W_P` 没有因 radix、精度或额外重排而改变；
再比较 payload、实际事务和时间。若只有 task-time/event，最多能定位昂贵阶段，不能据此确认
GM-bound 或声称达到某个 roof。

## A/B 原则

候选和基线必须在同一进程策略、同一输入、同一 reps/rounds 下交替运行。历史 `.o` 用
`scripts/baseline_o.sh <rev>` 从对应提交重建，不使用来源不明的临时对象文件。
