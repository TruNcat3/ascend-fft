# 为什么采用空间-时间混合映射

Ascend-FFT 是 [cuButterfly](https://github.com/TruNcat3/cuButterfly) 的硬件映射空间-时间并行方法在 Ascend 上的迁移实例。迁移的是计算编排范式，不是 CUDA kernel：阶段和数据两个维度的空间展开、时间复用仍是设计变量，具体实现重新 lowering 到 AIV、UB 和 MTE。

本页的分析框架和结论来源于 cuButterfly：包括 Roofline 只是上界、阶段依赖造成的流水空洞、
纯数据并行和纯阶段展开各自的限制，以及用阶段/数据两个维度的空间-时间映射统一描述设计空间。
Ascend-FFT 的工作是把这些方法论迁移到 Ascend，并用 AIV、UB、MTE、GM 的实际约束重新完成
lowering、候选筛选和实验验证；下文的“本项目”不表示重新提出这些基础观点。

本文先推导设计动机，再解释当前实现。建议接着阅读 [架构与映射](architecture.md)、[计算核心](kernels.md) 和 [性能模型](performance-model.md)。历史试验不作为当前设计说明，见 [决策归档](../archive/decisions.md)。

## 0. 研究对象的边界：计算流与计算核心

本项目有意把两个经常被混在一起的问题拆开：

| 层次 | 要回答的问题 | 典型参数 | 本项目的角色 |
|---|---|---|---|
| 计算流 / 架构编排 | 哪些阶段和数据块同时存在、按什么顺序交接、在哪里重排和写回？ | `Us/Ts/Ud/Td`、分段、layout、residence、pipeline、boundary | 主要研究对象，跨 GPU/NPU 迁移时保持抽象不变 |
| 算法核心 / 局部算术 | 一个已准备好的局部块采用哪种蝶形或 DFT 公式？ | radix-2、radix-4、radix-8、矩阵形式小型 DFT、专用局部 FFT | 可替换的算法选项，按硬件和 lowering 合约选择 |
| 硬件 lowering | 所选流编排和算法核心如何落到实际数据通路？ | AIV Vector、AIC Cube、对齐、UB/L1/L0、MTE、所有权与同步 | 目标硬件相关的合法实现，不是第三个算法轴 |

前两个设计层次**基本正交但成本耦合**，第三层 lowering 判断组合能否在目标硬件上实现。
例如，把 radix-2 换成 radix-4 可能减少局部阶段数，
却可能增加寄存器/UB、旋转因子和尾部处理；把同一个 radix-2 放进更好的数据流，可能减少
GM 边界和启动等待。前者是核心变化，后者是流编排变化，不能把两者的收益混为一个贡献。
因此每次实验应至少保留一个相同核心的流编排对照，以及一个相同编排的核心对照。

cuButterfly 不把计算核心本身作为主贡献，原因不是核心优化不重要，而是核心优化高度依赖
具体 GPU/NPU 的指令、数据类型、存储层级和成熟库经验；它很容易退化为某张卡上的局部技巧。
相比之下，阶段/数据两个维度的空间与时间组织、依赖约束、片上复用和交接位置是更稳定的
架构问题。把核心作为可插拔选项，既能吸收已有高性能单元，也能验证计算流方法在不同核心
下是否仍然成立。Ascend-FFT 迁移的是这个编排范式，而不是把 CUDA 的某个 kernel 原样搬到 AIV。

可以用下面的乘积空间理解设计：

```text
架构映射空间 A = {分段, Us/Ts/Ud/Td, layout, residence, pipeline, boundary}
算法核心空间 P = {radix-2, radix-4, 矩阵形式小型 DFT, 专用局部 FFT, ...}

候选设计点 = A × P
             -- 先做 AIV Vector / AIC Cube / UB / 对齐 / 同步 lowering 合法性过滤
             -- 再按流量、发射、等待和计算成本排序
             -- 最后用同协议实测回填
```

所以“计算流优化”和“计算核心优化”不互斥：理想实现应在同一架构映射下选择更好的核心，
也应在同一核心下比较不同的数据流；只有这样才能说明收益来自哪一层。

<figure class="doc-diagram">
  <a href="../../figures/method_layers.svg">
    <img src="../../figures/method_layers.svg" loading="lazy"
         alt="从 FFT 依赖图、四因素映射、Ascend lowering 到实测 Plan 的分层方法总览">
  </a>
  <figcaption>
    图中方法分析、局部核心、lowering 和证据连续但可区分。方法分析框架归属于 cuButterfly；
    Ascend-FFT 负责平台 lowering、实现和验证。
    <span class="doc-diagram__links"><a href="../../figures/method_layers.svg">SVG</a> ·
    <a href="../../figures/method_layers.pdf">PDF</a></span>
  </figcaption>
</figure>

<a id="roofline-insight"></a>
## 1. Roofline 到底给出了什么设计启发

Roofline 在这里不是一句“FFT 受带宽限制”的经验判断，而是把第一个优化问题变成可核算的
流量问题。对某一存储层级 `L`，固定变换语义和算法核心 `P`，令：

本节的推导沿用 cuButterfly 的
[设计总览](https://github.com/TruNcat3/cuButterfly/blob/main/docs/design_overview.md)
与[硬件映射方法](https://github.com/TruNcat3/cuButterfly/blob/main/docs/hardware_mapping_methodology.md)；
下面的 Ascend 状态说明是本仓库的迁移实现边界。

```text
W_P = 该核心实际执行的算术工作
Q_L = 数据穿过存储层级 L 的字节数
I_L = W_P / Q_L

可达算术吞吐 <= min(P_peak, BW_L * I_L)
执行时间       >= max(W_P / P_peak, Q_L / BW_L)
```

因此，当带宽上界已经成为限制时，仅增加算术资源不能提高 `BW_L*I_L` 这条上界；尚未建立足够
访存并发的实现仍可能从并行度中获益。真正应先问的是：**为什么同一份有效计算反复穿过较慢的存储
层级，哪些依赖边可以在片上直接交接？** 这就是 Roofline 对计算流设计的第一条启发。

### 1.1 用物化边界建立 GM 流量账本

设 `N=2^K`，`B` 为 batch，`q` 为一个复数 payload 的字节数。若一个 transform 被拆成 `G`
个物理执行组，组内中间值驻留，但相邻组之间把完整数组写入 GM 再读回，则 transform 数组的
逻辑 payload 账本为：

```text
Q_GM,payload = compulsory input/output + materialized boundaries
             = 2 * B * N * q + 2 * (G - 1) * B * N * q
             = 2 * B * N * q * G
```

这里的 `G` 是**物理执行组数**，不是源码中的逻辑分段数，也不是 kernel 名字的数量。只有真正
写入并重新读取完整中间数组的边界才增加这一项；同一 kernel 内按时间执行多个阶段，只要状态
仍驻留在 UB，就不应误算为新的 GM group。

以 `N=16`、`B=1` 为例，四级 radix-2 依赖图的算法工作保持相同：

| 组织 | `G` | payload GM 流量 | fp32 complex (`q=8 B`) | 相对逐级物化的 `I_GM` |
|---|---:|---:|---:|---:|
| 每级形成一个执行组 | 4 | `128q` | `1024 B` | `1x` |
| 每两级在片上驻留 | 2 | `64q` | `512 B` | `2x` |
| 四级全部合法驻留 | 1 | `32q` | `256 B` | `4x` |

<figure class="doc-diagram">
  <a href="../../figures/roofline_insight.svg">
    <img src="../../figures/roofline_insight.svg" loading="lazy"
         alt="以十六点 FFT 对比每级写回、每两级驻留和全阶段驻留的 GM 流量，并用符号化 Roofline 说明理论上界变化及四因素的成本作用">
  </a>
  <figcaption>
    从左到右阅读：橙色边界越少，同一算术工作对应的 GM payload 越低，设计点的 Roofline
    上界投影向右移动；但 GM 上界提高后，瓶颈可能转移到 UB 交换、Vector 发射、依赖等待或
    容量。图中坐标不是 Ascend 实测值，`Q_GM,payload` 也只是简化的逻辑流量账本。
    <span class="doc-diagram__links"><a href="../../figures/roofline_insight.svg">SVG</a> ·
    <a href="../../figures/roofline_insight.pdf">PDF</a></span>
  </figcaption>
</figure>

实际 GM/HBM 计数不必等于 payload：旋转因子和索引流、padding、非合并访问、重复读取以及事务
粒度会放大请求，缓存行为又可能改变真正到达外部存储的流量层级。上表也没有把 4×4 分解的边界
twiddle 和布局指令伪装成 radix-2 蝶形 FLOP；
因此用符号 `W_P`，而不以“32 个逻辑蝶形”冒充完整执行工作。payload 账本负责提出候选，真实
事务和时间仍由 profile 验证。

### 1.2 从 Roofline 推导出的五条选择原则

1. **优先消除重复的高层往返。** `G` 每减半一次，在 `W_P` 不变的理想模型中，`I_GM` 加倍。
   这解释了为什么片上阶段驻留和边界融合值得优先考虑，而不是先复制更多算术单元。
2. **规模增大不等于强度必然上升。** 在相同映射和线性工作/payload 模型下，`B` 同时放大
   `W_P` 和 `Q_GM,payload`，所以增大 batch 本身不减少每元素物化边界，也不改变该模型的稳态
   强度；它主要摊销启动、填充和尾部并提高活跃核数。实际系数复用、缓存命中和事务放大仍可能
   随 batch 改变，必须实测。对 FFT 长度，固定
   核心的工作近似正比于 `B*N*K`，故 `I_GM` 近似正比于 `K/G`：新增阶段仍留在同一个 group
   时强度可以上升，若容量迫使 `G` 随 `K` 同步增长，逐级物化的强度仍近似不变。
3. **阶段时间复用本身并不低效。** `Ts` 表示同一物理服务依次承担多个阶段；只要反馈状态留在
   UB，时间折叠不会新增 GM 边界。真正昂贵的是驻留层级容不下或供给不上状态，导致 fold 之间
   spill 到 GM。
4. **纯空间展开不是免费吞吐。** 增加 `Ud` 暴露更多独立蝶形，但依赖跨越所有权边界后，会扩大
   交换和边界压力；增加 `Us` 可能减少反馈或物化边界，却同时增加 live state、twiddle 供应、
   阶段间链路和同步。两者需要在固定资源预算下共同选择。
5. **计算流与计算核心应分别优化。** 更好的核心改变 `W_P`、计算峰值和资源需求，却不会自动
   消除 `G`；更好的流编排减少边界，却不会自动提高局部算术效率。二者基本正交、成本耦合，
   应组成候选后再做 lowering 和实测。

混合展开还可以用固定空间预算看得更清楚。令 `C=Us*Ud`，在理想单位服务率下改变分解不会凭空
增加主体算术能力，却会重新分配数据边界和阶段内部链路：

```text
每服务步的单侧边界值数（入口或出口） = 2*Ud = 2*C/Us
每服务步的输入+输出总服务量           = 4*Ud = 4*C/Us
每服务步的阶段间值数 = 2*Ud*(Us-1) = 2*C*(1 - 1/Us)
```

这里把入口和出口分别记为单侧宽度；双侧总量才是 `4*Ud`，不能把单侧服务宽度与前面的完整
GM 读写账本混用。

增大 `Us` 会减小外部边界宽度，却增大内部交接和 live state；增大 `Ud` 则相反。这是逻辑服务量
而非 Ascend 实测周期，但它说明了为什么 Roofline 最终导向二维混合点，而不是“数据全展开”或
“阶段全展开”任一极端。

这五条原则导出后文的混合设计点：`Ud/Td` 决定数据并发与复用，`Us/Ts` 决定阶段服务的复制与
复用，`residence` 决定时间折叠是否保持在片上，`layout/boundary` 决定理论 payload 能否转化为
合并、低冲突的实际事务。Roofline 给出了优化方向，却没有预先指定某个 tile、radix 或分段数。

### 1.3 为什么还需要流水、依赖和分层存储分析

“减少 GM 流量”不等于“达到计算峰值”。单一 GM Roofline 没有直接描述 UB 带宽与容量、
Gather/Vector 指令发射、依赖等待、同步、布局交换、启动和尾部。一个设计点的上界投影向右移动后，可能只是
从 GM-bound 变成 UB exchange-bound、issue-bound、dependency-bound 或 capacity-bound。

因此本项目使用的是分层问题而不是单一标签：先为 GM 建立 payload 与实际事务账本，再检查 UB
能否容纳和持续供给状态，最后以任务时间和计数器定位发射、等待与重叠。只有实测确认的限制才写成
设备结论；“带宽受限”若只有时间分段而没有 GM/MTE 证据，仍只是待验证假设。具体诊断方法见
[性能剖析](../development/profiling.md)。

当前 Ascend 短 C2C 路径将单个 transform 的中间阶段保留在单 AIV 私有 UB，接近上图的 `G=1`
payload 组织；长 FFT 的设备边界虽已融合部分 twiddle/转置，仍有 GM 物化，不能写成全图片上驻留。
这说明 Roofline 已经转化为实际设计决策，但并不证明当前每个路径都达到对应硬件 roof。

## 2. 流水不足使可用资源不能转化为吞吐

蝶形不是彼此独立的逐点运算。阶段内存在互不依赖的蝶形，阶段之间却需要前一阶段的结果。若直接采用“加载 → 完整计算 → 写回”的串行组织，搬运与计算资源可能交替空闲；若每级单独执行，又可能引入重复搬运和启动。

流水设计的目标是让不同独立数据块处于不同处理步骤，复用已经建立的局部计算和布局组织。但只有负载之间满足依赖和缓冲所有权约束，才允许重叠。增加 batch 能摊销启动和填充/排空成本，不能摊销每个数据块重复支付的指令、布局交换或 GM 边界成本。

## 3. 为什么纯数据维度展开不够

同一阶段的独立蝶形可以分配给不同 lane 或核；跨阶段的配对距离变化，则使某个局部块的输出成为其他局部块的输入。将数据完全铺开并不能自动消除这些依赖，反而可能产生跨核交换、同步与重排。

batch 内不同 transform 是额外的独立负载，适合数据并行；但小 batch 不一定能填满硬件。当前 Ascend 实现的一条 transform 不跨多个 AIV 分割，B=1 的活跃核数便受到这个实现边界限制。它不是蝶形范式的理论限制，也不是增加矢量 lane 数即可解决的问题。

## 4. 为什么纯阶段维度展开也不够

将多个阶段分别实例化成处理单元，能够构造生产-消费流水；代价是同时保留更多中间值、索引、旋转因子和交接缓冲。FFT 配对覆盖的依赖范围随展开阶段增加而扩大，缓冲增长取决于分块、布局和重排策略，不能简单假定为一个固定线性系数。

片上容量有限时，过大的阶段展开会降低可并发工作组数量，甚至迫使边界重新落到 GM。纯阶段展开可能因此用更高驻留成本换来更低并发度。在线重排可以调整依赖的局部性，但也有自己的指令和搬运成本，需要与展开方式共同选择。

## 5. 二维空间-时间映射：把两种极端变成同一设计空间

概念设计点可写为：

```text
a = (Us, Ts, Ud, Td, layout, residence, pipeline, processing_unit)
```

| 维度 | 空间展开 | 时间复用 |
|---|---|---|
| 阶段 | Us：同时实例化的阶段/子图处理资源 | Ts：同一处理资源承担的阶段序列 |
| 数据 | Ud：同时处理的独立数据片/批 | Td：同一资源遍历的数据片/批 |

这些是方法层的概念，不是对当前 C++ 字段数值的重新定义。硬件资源决定哪些组合可行；依赖和布局决定交接是否局部；缓冲容量决定能否驻留；流水与负载量决定是否值得重叠。分块大小和计算核心都可变，不应把某个 K 或 radix 当作范式本身。

<figure class="doc-diagram">
  <a href="../../figures/fft_space_time_example.svg">
    <img src="../../figures/fft_space_time_example.svg" loading="lazy"
         alt="以八点 FFT 为例，分别说明数据空间 Ud、数据时间 Td、阶段空间 Us 和阶段时间 Ts 如何映射同一依赖图">
  </a>
  <figcaption>
    八点 FFT 只是帮助阅读参数含义的最小实例，不是固定设计点。当前生产路径已经实现数据空间/时间映射
    和单 AIV 的 UB 驻留阶段时间复用；跨 AIV 的阶段角色流水仍需真实传输与同步 lowering。
    <span class="doc-diagram__links"><a href="../../figures/fft_space_time_example.svg">SVG</a> ·
    <a href="../../figures/fft_space_time_example.pdf">PDF</a></span>
  </figcaption>
</figure>

<a id="hybrid-unfolding-example"></a>
### 一个完整的混合展开例子

上一张图分别解释四个因子；下面把它们组合到一个可逐项核对的教学点。取一个前向、无归一化的
`N=16=4×4` FFT。它有 `K=log2(N)=4` 个 radix-2 阶段，每个阶段有 `N/2=8` 个蝶形。选择：

```text
(Us, Ts, Ud, Td) = (2, 2, 4, 2)
Us * Ts = 2 * 2 = K
Ud * Td = 4 * 2 = N/2
```

这里没有把 batch 计入 `Ud/Td`；示例只讨论一个 transform 内的蝶形数据维度。batch 是额外的
独立工作维度，可以在硬件 lowering 时再分配给 AIV 或折入时间循环。

| 因子 | 本例中的具体含义 |
|---|---|
| `Us=2` | 一个折叠中有两个明确的阶段服务位置，负责 `s0→s1` 或 `s2→s3`；两者之间必须存在可观察的数据交接 |
| `Ts=2` | 同一组阶段位置先服务前两个阶段，完成边界重排后再复用为后两个阶段 |
| `Ud=4` | 同一时刻处理四个蝶形，即两个独立 FFT4 子图各自的两个蝶形 |
| `Td=2` | 先流过 `Q0/Q1`，再复用相同数据通路处理 `Q2/Q3`；第二个阶段折叠同理 |

<figure class="doc-diagram">
  <a href="../../figures/hybrid_unfolding_example.svg">
    <img src="../../figures/hybrid_unfolding_example.svg" loading="lazy"
         alt="十六点 FFT 的完整混合数据流例子：四乘四子图分解、Us=2 Ts=2 Ud=4 Td=2 的空间时间覆盖，以及两个阶段角色流水数据包、在线重排并复用到第二折叠的时间线">
  </a>
  <figcaption>
    阅读顺序从左到右：先形成四个同构 4 点子图，再用 `2×4` 个空间计算单元覆盖一个 tile，
    最后让两个数据包错峰通过两个阶段角色，并在消费者就绪后重排进入下一阶段折叠。该实例按
    <a href="https://github.com/TruNcat3/cuButterfly/blob/main/docs/hardware_mapping_methodology.md">cuButterfly 四因素方法</a>
    特例化为 FFT；SVG 中的数值用于解释映射，不是当前 Ascend 的部署参数或性能数据。
    <span class="doc-diagram__links"><a href="../../figures/hybrid_unfolding_example.svg">SVG</a> ·
    <a href="../../figures/hybrid_unfolding_example.pdf">PDF</a></span>
  </figcaption>
</figure>

数据依赖可以用具体索引检查。第一组子图为：

```text
WM = exp(-2*pi*i/M)
Qj = [x[j], x[j+4], x[j+8], x[j+12]],  j = 0..3
F[j,k2] = sum(n2=0..3) x[j+4*n2] * W4^(n2*k2)
```

`Q0/Q1` 和 `Q2/Q3` 形成两个时间数据包。每个 `Qj` 经一个包含两级的 4 点依赖子图后得到
`F[j,k2]`，边界融合
`W16^(j*k2)` twiddle 并按下一消费者重排为：

```text
Z[j,k2] = F[j,k2] * W16^(j*k2)
Rk2 = [Z[0,k2], Z[1,k2], Z[2,k2], Z[3,k2]]
X[k2+4*k1] = sum(j=0..3) Z[j,k2] * W4^(j*k1)
```

于是第二个阶段折叠可以复用同一类服务位置处理 `R0..R3`。关键点是：某个 `Rk2` 依赖多个
`Qj` 的对应输出，所以“在线重排”仍需满足消费者 readiness；它不是任意单个 `Qj` 完成后立即
启动全部后续计算，也不是零成本操作。

在忽略填充/排空、重排、同步和 bank conflict 时，该点有 `Us×Ud=8` 个空间蝶形单元和
`Ts×Td=4` 个理想 tile 步。这个计数只用于解释工作怎样分配，不能直接当作性能预测。特别是，
`Us=2` 要求两个阶段位置同时驻留并有真实交接；把两级串行执行或融合进一个 radix-4 核心，
不能自动记作阶段空间展开。

当前 Ascend 短 FFT 生产路径实现的是单 AIV 私有 UB 中的状态驻留、阶段时间循环、矢量数据并行
和 batch 映射；它没有实现图中跨 AIV 的 `Us=2` 生产者/消费者角色。后者仍是 `NOT LOWERED`
的方法候选。长 FFT 已实现的 twiddle/转置融合边界仍会在 GM 物化，也不能据此声称完整图已经
片上驻留。

这个分析适用于具有分层蝶形依赖的计算，但不意味着所有算子的数值语义、重排和核心成本相同，也不意味着任何硬件上混合流水必然优于成熟库。相关工作已有多种融合、分块和流水技术；本仓库的主张是把这些选择放入统一、可约束和可测量的映射空间，而不是宣称此前没有任何相关分析。

## 6. Ascend 映射：保留范式，重建物理实现

当前验证设备的 AIV 使用显式矢量指令、私有 UB 与 MTE 搬运，不具备本仓 CUDA 路径所依赖的 warp shuffle 语义。因此不能直接移植 warp 子图；要使用对齐的 K-plane 布局、Gather 索引与 Level-0 repeat 组织同一批蝶形。

当前短 FFT 生产路径把单个 transform 的阶段留在一个 AIV 的 UB 中，多个 AIV 分担 batch，单核再时间遍历剩余批次。对合适的小长度，多个 batch 被折入同一 repeat 组织，减少重复指令发射。它实现了片上阶段时间复用与数据展开/复用，但尚不是独立阶段角色同时驻留的跨核流水。详见 [架构页中的实现边界](architecture.md)。

超过单 AIV 容量时，前向长 FFT 验收路径以分段行 FFT 与 GM 交接延续这一映射方法。
宿主/设备边界、separate/fused twiddle-转置以及合法缓冲组织，是固定局部核心下的流编排
对照，而非新算术核心的收益。长链已有实现和归档不等于任意长度、独立阶段角色流水或
自动搜索已经完成；尤其 `pp2` 的 device-chain 候选与 pinned E2E 放行必须分开。
实验合同与限制见 [长 FFT 实验计划](../benchmarks/long-fft-plan.md)。

## 7. 核心选型与模型搜索负责找到硬件设计点

radix-2、局部 radix-4 或未来的矩阵形式小型 DFT 都是算法核心候选，不改变上述编排方法。
它们可分别尝试 AIV Vector 或 AIC Cube 等 lowering；只有满足对齐、缓冲、指令字段、数据布局、
所有权和同步约束的组合才是合法实现。

筛选顺序应是：硬件探针 → 可行性判断 → 参数化成本估计 → 正确性验证 → 有限实测 → 保存选择。当前模型根据实际发射的算子数和元素访问数估计成本，并使用实测校准系数；它不是硬件无关的性能定理。改变设备、核心或布局后需要重新校准，而不是复制旧卡参数。

**因此，设计目标不是展开得最多，而是在合法依赖、容量与数据通路约束下，让独立负载以较低的稳态单位成本持续运行。**
