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

## A/B 原则

候选和基线必须在同一进程策略、同一输入、同一 reps/rounds 下交替运行。历史 `.o` 用
`scripts/baseline_o.sh <rev>` 从对应提交重建，不使用来源不明的临时对象文件。
