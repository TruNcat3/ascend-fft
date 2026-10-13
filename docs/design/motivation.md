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

## 1. Roofline 是上界，不是充分的优化策略

长度为 N 的 radix-2 FFT 有 log2(N) 个阶段，每个阶段约有 N/2 个蝶形。增加算术强度、降低片外读写量，可以提高 Roofline 上界：如果每一级都读写 GM，中间流量随阶段数增长；若多个阶段的数据留在片上，GM 流量可以显著下降。

但“数据在片上”不等于“计算达到峰值”。Roofline 的带宽与算力上界没有直接描述矢量指令发射、依赖等待、同步、布局交换、启动和尾部开销。一个已经减少 GM 搬运的实现，仍可能受这些成本限制。因此要同时分析数据流量与持续处理吞吐，而不是仅以 FLOP/byte 判断设计优劣。

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
