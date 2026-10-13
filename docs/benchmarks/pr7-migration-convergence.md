# PR 7：迁移验收与后续优化收敛

> **已完成历史记录**：[PR 7](https://github.com/TruNcat3/ascend-fft/pull/7) 已于
> 2026-10-13 合并。本页保留当时的目标、测试设计和后续候选晋升门槛；下文“必须”“预期”
> 等措辞描述原计划，不表示 PR 7 仍未完成。当前发布证据统一来自
> `ascend910_9382-migration-20261013`。

最新阅读顺序为：[实验设计与结果汇总](results.md) → [C2C](c2c.md) →
[端到端与应用](end-to-end.md) → [R2C/C2R](real-transforms.md) →
[规模切片](scale.md) → [原始数据与复现](reproducibility.md)。
后续 B1/B2/B3 仍属于[未来研究任务](../roadmap.md)，不作为本次迁移发布的未完成项。

本计划承接已关闭的 [PR 6](https://github.com/TruNcat3/ascend-fft/pull/6)。PR 6 的稳定迁移路径已经有独立发布快照和完整矩阵支撑；其
`static`、`pipe` 和 `fused` 变体仍属于边界优化实验，不能作为当前迁移成果的必要前提。

## 目标和边界

PR 7 只做两件事：

1. 在最终发布提交上复验并固化“cuButterfly 方法已经迁移到 Ascend”的证据；
2. 为 PR 6 遗留的边界融合实验定义可复现的后续测试和晋升门槛。

本 PR 不把 Ascend-FFT 宣称为所有 SoC、精度、长度和 batch 的全局最优库，也不以一次性完成
所有 fused writer 或持久化流水为退出条件。方法来源、二维空间/时间映射和计算流与计算核心的
正交关系均来自 [cuButterfly](https://github.com/TruNcat3/cuButterfly)；本仓库只验证其在
Ascend AIV/UB/MTE/GM 上的具体 lowering 和实验结果。

## A. 迁移证明的必做实验

这些实验决定当前迁移实例是否可以发布，不依赖 PR 6 的实验性 kernel。

### A1. 发布提交复验

在待发布的 clean commit 上，用与快照相同的硬件、CANN、编译器、计时区间和正确性门禁重跑：

| 工作负载 | 最低覆盖 | 结果要求 |
|---|---|---|
| C2C device-only | 当前 49 点矩阵 | 每点正确，保留胜/平/负和 raw trials |
| C2C end-to-end | 当前 49 点矩阵 | H2D、执行、D2H 口径不变 |
| R2C/C2R | 当前各 49 点矩阵 | 每个 trial 先过 `maxRel <= 1e-4` 再计时 |
| 应用 shape | OFDM、雷达、DL 当前 12 点 | 保持同语义 CANN 基线 |

复验的目的是绑定发布提交和结果快照，不是重新挑选有利点。结果必须保存硬件身份、CANN
版本、命令、构建产物哈希、逐 trial 原始值和失败原因。

### A2. 小型受控消融

为了说明收益来自架构级计算流，而不是某个偶然的局部 kernel，选择 3--5 个代表点比较：

```text
naive Ascend lowering
    -> 阶段/数据混合空间-时间映射
    -> 硬件约束下的映射与布局选择
```

至少报告 kernel time、端到端时间、GM/UB 搬运量、AIV/MTE 利用率、正确性和配置选择原因。
消融只用于解释方法，不替代完整发布矩阵，也不要求每个局部消融都超过 CANN。

代表点必须包含当时 C2C `N=1024,batch=4096` 的 `0.96x` 边界点，以及当时发布快照中最慢的
R2C 大 batch 点。对每个点同时记录固定 CANN baseline、稳定路径、模型选择、有限搜索最优点
和选择损失。这样既能解释方法的平均收益，也能说明已知负点为什么没有被当前映射消除。

### A3. 规模压力切片

为验证方法不是只适用于当前小尺寸，按实现能力选择少量点：

```text
N = 8192, 16384, 32768, 65536
batch = 1, 16, 256
```

只测试已经有合法 lowering 和正确性实现的组合。尚不支持的组合必须报告为
`unsupported / requires multi-tile implementation`，不能用补零或改变变换长度冒充原始问题。
该切片是迁移潜力的增强证据，不阻塞当前稳定版本发布。

### A4. 证据生成和审计

所有摘要、图表和 README 数字必须从同一个 immutable manifest 生成。发布前应通过：

- correctness gate 和失败样本隔离；
- raw trial 去重和覆盖检查；
- baseline 语义、精度、设备和计时口径检查；
- 文档/图表 freshness 和链接检查；
- 发布 commit、硬件 profile、CANN、编译产物哈希绑定。

## B. PR 6 遗留的实验性前沿

以下内容迁移到未来计划，不阻塞迁移案例发布。

### B1. Static paired writer

目标是减少 R2C 后处理中的 GM 边界和非连续写回。[PR 6 Round-4 记录](https://github.com/TruNcat3/ascend-fft/blob/03fab4897a2cb577ef49c3ee6f1198a7a4d701ff/docs/design/r1-round3-summary.md#round-4-results-2026-10-12)
显示 vectorized
interleave 可以改善原型，但仍慢于稳定 separate path。

下一轮必须先完成：

- `N >= 4096` 多 chunk reverse 的局部反转和写回偏移；
- `N=128/512/4096/8192`、batch `1/47/64/1024/4096` 的逐点 correctness；
- 5 个以上独立 trial 的 raw timing；
- 与稳定 baseline 同计时区间的 post 和 end-to-end 对比。

只有在所有声明尺寸正确且 median 至少快于 incumbent 2% 时，才允许进入候选池；否则保留
为显式实验选项并自动回退。

### B2. Pipelined writer

目标是重叠 MTE2、向量计算和 MTE3。`PIPE_ALL` 只能证明安全，不能证明流水有效。必须显式
验证如下 ownership 顺序：

```text
compute-ready -> writer consumes -> store-done -> buffer reusable
```

测试至少覆盖 batch `1/47/64/1024/4096` 和多 chunk 长度，并记录 trace 中的重叠区间、等待
时间、UB buffer 地址和事件关系。任何 race、未定义数据或尾部错误都不得进入性能排名。

### B3. Fused epilogue

目标是复用向量化 `kfft_fwd` 的最后阶段，消除中间 GM 边界；当前标量 `simple_fft` 原型不能
代表最终 fused 架构。重新实现前必须修复：

- fused twiddle host 表的零填充和复制长度，禁止 host source overread；
- 与 separate path 一致的 R2C writer contract、Nyquist 和布局；
- 向量化 FFT 核心、批折叠和 UB 容量约束；
- 与稳定路径相同的 raw correctness/timing 证据。

## C. 统一验收门槛

| 门禁 | 要求 |
|---|---|
| 正确性 | 每个 trial 通过既有数值契约，失败不得进入 timing 汇总 |
| 选择闭包 | 每个候选集合显式包含 incumbent；独立复测若未达到晋升门槛则选择 incumbent |
| 稳定收益 | 候选与 incumbent 在相同执行顺序中配对测量，median 至少提升 2% 才晋升 |
| 重复性 | 至少 5 个独立 session 的配对 trial，报告 median、p10、p90、CV 和 raw 值；分位数只作描述，不当作显著性证明 |
| 多 chunk | 所有声明长度覆盖单 chunk 和多 chunk 边界 |
| 可追溯性 | 保存 requested/resolved mapping、kernel 名称、版本和选择原因 |
| 回归 | 完整 C2C/R2C/C2R/E2E 矩阵无 correctness 或 incumbent 回退 |

若有限预算内仍有点无法超过 `1.0x`，保留该点和 profile 证据，标为 `unresolved`，并说明
瓶颈属于计算核心、数据布局、GM/UB 搬运、同步还是基线特化。不得通过删点或扩大有利样本
改变结论。

## 退出条件

### 迁移案例发布

当 A1 和 A4 完成，A2 给出至少一组可解释的架构消融，当前发布快照即可作为
“cuButterfly 方法在 Ascend 上的可复现迁移实例”发布。A3 可作为增强结果，但不阻塞发布。

### 边界优化晋升

B1/B2/B3 各自满足 C 门禁后，才能进入模型和 dispatch 候选；没有通过的实现保持 opt-in，
并回退到稳定 baseline。它们是后续性能研究，不改变已经成立的迁移结论。

## 预期产物

- 新的 immutable migration-validation snapshot；
- A1 的完整矩阵、A2 的消融图和 A3 的规模压力表（若实现支持）；
- 每个 R1（real-transform boundary optimization）变体的 raw trials、trace/profile 和失败分类；
- 更新后的支持矩阵、性能摘要、路线图和复现命令；
- 明确区分“已验证迁移成果”和“未收敛优化前沿”的发布说明。
