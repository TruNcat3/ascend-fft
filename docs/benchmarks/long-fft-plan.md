# 长 FFT 实验计划

本计划回答一个具体问题：当单次变换超过单 AIV 的 UB 容量后，阶段/数据两维的空间与时间
映射能否继续保持正确性，并以可解释的搬运、同步和流水开销获得高吞吐。它不是扩大的短 FFT
排行榜，而是验证方法从单块驻留向跨块子图推广的实验合同。

**状态：G0/G1 后端已实现，证据待采集；G2..G4 待实现。** 公开 C2C 支持仍为 `N=64..4096`
（R2C 的 `N=8192` 不代表同长度 C2C 已进入发布口径）。前向长 C2C 的最小后端已在
`fft_check` 验收链路落地：G0 能力/预算门 + 四步 Cooley-Tukey 分段（宿主转置/twiddle/
重排，设备复用 `kfft_fwd` 行 FFT），envelope `N=8192..65536`，G1 验收抽样在
Ascend910_9382 全部通过（见下方「G0/G1 落地记录」）。E01..E08 证据采集、G2 可搜索
映射、G3 大长度与逆向/2D 在完成前不得执行进发布均值，`N=65536` 仍是核心验收长度，
不是固定的最优设计点；分段数、局部点数、核心、驻留与布局仍是搜索参数。机器可读实验清单见
[`config/long_fft_experiments.json`](https://github.com/TruNcat3/ascend-fft/blob/master/config/long_fft_experiments.json)，
由它生成的[预声明数据表](../generated/long-fft-tables.md)只包含待测条件和空结果合同；基础测试
层级见 [`config/test_matrix.json`](https://github.com/TruNcat3/ascend-fft/blob/master/config/test_matrix.json)。
现有支持范围和基础门禁见[支持矩阵](../reference/support.md)与[测试策略](../testing.md)。

## 实施门槛与共同合同

在 Ascend910_9382 上，FP32 C2C `N=65536` 的单输入为 512 KiB，输入加输出至少 1 MiB，
超过每个 AIV 约 192 KiB 的 UB。必须先实现合法的分段、子图交接和输出重排，不能通过调大
UB 估算或修改测试参数宣称支持。

| 门槛 | 必须完成的内容 | 放行条件 |
|---|---|---|
| G0 能力与预算 | 长度 envelope、偏移位宽、GM/UB/workspace 预算、失败原因 | 不可行组合在执行前拒绝 |
| G1 最小长后端 | 8192..65536 前向 C2C；显式分段、交接、自然序输出 | 基础数值、尾 batch 与重复执行全部通过 |
| G2 可搜索映射 | 参数化分段、局部核心、驻留、布局、fold 与角色配置 | 候选实际生效，非法组合可解释地排除 |
| G3 大长度推广 | 131072..1048576；预算与超时控制 | 全部声明支持点通过，不以补零替代原 DFT |
| G4 扩展语义 | 逆向/归一化、2D、stride 或其他精度 | 各自接口契约与正确性先完成，再加入比较 |

### G0/G1 落地记录（2026-10-08，Ascend910_9382）

- 描述符层（addendum step 2，PR #2 阶段 2）：`include/butterfly/descriptors.hpp` 按
  G/A/P/L/H 分层（`TransformSpec`/`ArchitectureMapping` 含四维展开 `Us,Ts,Ud,Td`+角色+
  驻留+边界归宿、`ProcessingUnitCapability`（首个单元 = 现有 `kfft_fwd` 行 FFT）、
  `LoweringResult`、`HardwareProfile` 与 `config/ascend910_93_profile.json` 同步）。
  长路径在任何分配/启动之前调用 `query_lowering()`；判定分两级——`abstract_feasible`
  （资源/合法性可行）与 `supported`（仓库中存在可执行 lowering，选择层只可选后者）。
  未被 runtime 消费的 `Us/Ts/Ud/Td/pipeline_buffers/role_stages/residence` 组合返回
  `abstract-feasible-but-not-lowered: ...`，绝不标记为可执行。可执行结果携带真实
  launch manifest（`LaunchRecord` 类型/stage/boundary），`visible_launches`、
  `materialized_gm_boundaries`、`host_assisted` 全部由 manifest 派生：host 边界
  = 2×`row_fft`，device 边界 = `transpose_in|row_fft|twiddle|transpose_boundary|
  row_fft|transpose_out` 共 6 次，两者的 GM 物化段边界均为 1（`AB_DESC=1` 打印
  launches/gm_boundaries/host_assisted/abstract/kinds）。合法性测试：
  UB 资源模型按物理内核分别描述（`row_fft_resource`/`transpose_resource`/
  `twiddle_resource`：峰值字节、AIV 核类、同步范围、形状约束），串行计划取各
  内核峰值的最大值而非相加；公式经 `include/butterfly/long_fft_ub.h` 与
  `src/ascendc/fft_long.cpp`/`fft_radix2.cpp` 共享（转置 3×128×32×8=98304 B、
  点乘 24n、行 FFT 46.5n+128），按 manifest 发射序逐一校验——UB<98304 的
  DeviceGM 降级被拒并指名 `kfft_lt_tr`，同行主链（仅行 FFT）可继续通过。
  合法性测试：`tests/test_descriptors.cpp` 自测表 + `tests/test_descriptors.py`
  （纯 g++ 编译、26 个用例、profile 同步、查询先于分配的源码顺序断言、
  manifest/UB 边界回归），`make desc` 一键运行。
  短 FFT 结果不变：验收采集器重跑 12/12 PASS、E2E transfers 契约不变。

- 实现：`src/host/fft_check.cpp` 长路径。G0 在执行前拒绝超 envelope（`N>65536`）、
  非法因子分解、`batch*2n` uint32 goff 越界与 `>40 GiB` GM 预算，并打印拒绝原因；
  G1 分段 `N=N1×N2`（两因子均在 `64..4096`，从 sqrt 附近取平衡拆分），行 FFT 复用
  现有 `kfft_fwd`，转置/twiddle/自然序重排在宿主完成，`us/call` 计入完整分段链
  （含宿主段与段间拷贝）；该口径必须在 E02/E05 采集时如实声明。
- 验收抽样：`N∈{8192,16384,32768,65536}` × `B∈{1,3,47}`（含尾 batch）全部 `PASS`
  （`maxRel≈1e-7 ≤ 1e-4`）；`N=65536/B=1` 连续 20 次重复执行通过；`random-seeded`、
  `high-dynamic-range`、`impulse`、`constant` 输入在 `N=65536` 全部通过；G0 拒绝路径
  （`N=131072`、`GM=48 GiB`、非二次幂）按预期带原因返回。复现：
  `./build/fft_check <N> <B> 3`（`AB_INPUT=` 切换输入模式）。
- 动态输入（P1-A）：plan 状态只保留输入无关数据，`hT` 每次执行由当前 `hIn` 重建，
  结果留在宿主 `hOut`；G1 全包络 12/12 点 A/B/A 文件序列（一次 plan，输入内容变化）
  无重建重执行、无 STALE-OUTPUT，E2E 每次执行恰 1 次逻辑输入 + 1 次输出
  （`boundary=2` 计段边界传输）。机器可读证据：
  `results/evidence/long-fft-acceptance/acceptance.json`（`scripts/collect_long_fft_evidence.py`）。
- 段边界 device 化（addendum step 3）：`AB_BOUNDARY=device` 时段边界全程在设备上
  物化，不经宿主内存——`src/ascendc/fft_long.cpp` 的 `kfft_lt_tr`（`DataCopyParams`
  列切片 strided 读 + UB 源 Gather 分块转置、dst 行段连续写回）完成转置入/段间转置/
  自然序写出，`kfft_lt_tw` 原地行连续点乘完成 `W_N^{j·k1}` 段边界；wT 表 plan 期上传
  （输入无关），每次执行仍从当前 `hIn` 上传（动态输入契约）。E2E 每次执行
  `in=1 out=1 boundary=0`（段边界不回宿主），同一网格 A/B/A 12/12 点 PASS，机器可读
  证据 `results/evidence/long-fft-device-boundary/acceptance.json`
  （`scripts/collect_long_fft_evidence.py --boundary device`）。默认（不设
  `AB_BOUNDARY`）保持宿主中介链，逐点 `boundary=2` 契约锁定、行为与已发布结果一致
  （宿主证据同采集器重跑 12/12）。`AB_DESC=1` 在 device 模式如实报告
  `boundary_home=DeviceGM`：`gm_boundaries=1`（两段间一个 GM 物化边界）+
  `host_assisted=0`。性能数字不手写：每个形状 5 次独立 trial 的原始样本归档在
  两份 `acceptance.json`（`manifest` 含 clean git sha、二进制 sha256、构建配方、
  CANN/SoC/driver 与环境），median/min/mean/CV 由
  `python3 scripts/summarize_long_fft_evidence.py` 复算生成
  [验收证据表](../generated/long-fft-evidence.md) 与
  `results/evidence/long-fft-summary.json`，`--check` 可校验漂移。
- 未完成：E01..E08 全量采集、G2 可搜索映射、G3（`131072..1048576`）、逆向/2D；
  这些完成前 `future-long-fft` 档案仍拒绝运行，也不进入发布均值。

#### 计时字段（`scopes:` 行）与同步边界

| 字段 | 含义 | 同步边界 | 适用 |
|---|---|---|---|
| `plan_setup` | 一次性准备（旋转因子/索引生成、分配、二进制加载）墙钟 | 首次执行 warmup 之前截断 | 全部 |
| `first_use` | 首次执行墙钟（含冷启动上传/内核首发射） | 单次 launch 的外墙钟 | 全部 |
| `host_end_to_end mean/min` | 每次执行的同步墙钟：宿主段、段间拷贝、传输与同步全含 | `launch()` 进入到流同步后返回 | 全部 |
| `h2d` | 逻辑输入 H2D 的事件跨度 | `evIn..ev0`：输入上传入队到首个计算内核之前 | 仅长链，否则 `NA` |
| `device_chain` | 链上 device 计算内核的事件跨度，**不含任何传输** | `ev0..ev1`：device 边界 = 3×转置 + 2×FFT + 1×点乘的连续跨度；host 边界 = pass1+pass2 两次 FFT 跨度之和（段间传输/宿主段只在墙钟里） | 全部 |
| `d2h` | 逻辑输出 D2H 的事件跨度 | `ev1..evOut`；host 边界为 pass2 出数那次（宿主重排在其后，属宿主段） | 仅长链，否则 `NA` |
| `reps` | 采样次数（事件跨度与墙钟取同一迭代集合的最小值/均值） | — | 全部 |

宿主边界与设备边界使用同名同语义字段，不适用的范围显式打印 `NA`；恒有
`host_end_to_end ≥ h2d + device_chain + d2h`（逐次成立，允许时钟域与调度噪声）。
字段名、NA 规则与预算不等式由 `scripts/scopes.py` + `tests/test_scopes.py` 解析测试锁定，
防止后续把传输重新计入 device kernel 时间。

每个实验固定 SoC、CANN、构建、FP32、前向 1D C2C、稠密交错输入和自然序输出；其他语义
使用独立结果集。设备计时、同步 host 端到端、Plan 创建和首次执行分别报告。
输入输出 placement 与 workspace 必须声明，基线匹配后才计算 speedup。

每点至少 5 个独立 trial，保存所有 trial、warmup/reps、原始错误及所选完整配置；新计划以
trial 中位数和离散度为主，不能与历史 min-of-means 直接合并。实现之间轮换或随机执行顺序，
记录温度、频率及其他负载。失败点保留 `unsupported/oom/timeout/incorrect`，不填零、不删除。

架构模型校准点和留出点在采集前分开。搜索的 incumbent 必须包含当前可执行的成熟路径，
但报告“实测搜索最佳”和“模型自动选择”两列：使用历史最快值直接代替模型部署不能证明模型有效。
搜索预算、prepare 时间、候选数量和被排除原因同样属于结果。

严格性能基线优先采用同设备、同语义的 CANN/torch_npu FFT；必须先验证其对应长度可用。
CPU FP64 只作为数值参考；cuFFT/cuButterfly 在不同硬件的数据只作方法背景，不计入 Ascend
胜负均值。缺失的外部基线标为 unavailable，不能换成语义不同的 RFFT。

## 目的与矩阵总览

| ID | 实验问题 | 自变量 | 主要输出 |
|---|---|---|---|
| E01 | 超过片上容量时在哪里切换实现？ | N、能力 envelope、局部块预算 | 容量与实现转折表 |
| E02 | 小 batch 的延迟由什么组成？ | N、B=1/4/16 | 延迟分解曲线 |
| E03 | 同总数据量下，长度的影响是什么？ | N，固定 P=N×B | 等点数吞吐曲线 |
| E04 | 稳态吞吐与硬件拐点在哪里？ | B、fold、执行 wave | 吞吐/尾部曲线 |
| E05 | 混合数据流的各机制贡献多少？ | 单项映射机制、固定形状/核心 | 配对消融图 |
| E06 | 长度和编排是否损害数值？ | N、输入模式、seed、方向 | 误差与 round-trip 表 |
| E07 | 是否适用于真实应用形状？ | 应用 N/B/轴布局 | FFT 代理与完整流水分列表 |
| E08 | 性能变化能否由资源行为解释？ | E03/E04/E05 的预声明诊断点 | 流量、等待与重叠证据 |

下表的空单元格表示尚未采集，不是零耗时。所有报告表应链接到原始 trial 和 manifest，
字段模板必须随结果填写，不得只公布胜出行。

## E01：容量边界与实现切换

- 自变量：`N=2048/4096/8192/16384/32768/65536/131072/262144/524288/1048576`。
- 固定条件：`B=1`；每个候选准确记录 UB 和 GM 需求，不预设必须两段或三段。
- 指标：支持状态、分段/执行组数、每块局部点数、UB 高水位、workspace、交接位置及正确性。
- 基线：当前 2048/4096 成熟单块路径、可行的物化分段路径、同尺寸外部 FFT。
- 完成判据：65536 的合法长路径通过 G1；其他点无静默越界；资源预算与实际分配一致。

| N | B | 实现/状态 | 分段/执行组 | 局部点数 | UB KiB/块 | workspace MiB | 边界类型 | 正确性/原因 |
|---:|---:|---|---|---:|---:|---:|---|---|
| 65536 | 1 | pending | | | | | | |

## E02：低 batch 延迟与固定开销

- 自变量：`N=8192..1048576` 的所有二次幂，`B=1/4/16`，共 24 个长形状。
- 固定条件：相同测量合同；分别测 Plan、首次调用、warm device-only、H2D/FFT/D2H。
- 指标：中位数 ms、trial CV、首次/稳态比、启动数量及端到端 speedup。
- 基线：同形状外部 FFT、物化分段路径、模型选型与有限实测搜索最佳。
- 完成判据：每个支持点都有所有计时区间；错误、超时和未支持点可见；不以摊销隐藏 Plan 成本。

| N | B | 实现/配置 ID | Plan ms | 首次 ms | device ms | H2D ms | D2H ms | E2E ms | CV % | 外部 speedup |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 65536 | 1 | pending | | | | | | | | |

## E03：固定总点数，隔离长度与总负载

- 自变量：`N=4096/8192/16384/32768/65536/131072/262144/524288/1048576`。
- 固定条件：分别固定 `P=N×B=2^20/2^24/2^26`，令 `B=P/N`；不改变精度或计时区间。
- 指标：points/s、transforms/s、device/E2E ms、每点 GM 字节、workspace、搜索与模型损失。
- 基线：同形状外部 FFT；4096 成熟路径仅是长度对照，不冒充长 FFT 基线。
- 完成判据：预声明的 27 个点全部有状态记录；能区分依赖深度/边界增加与总数据量增加。

`P=2^24` 的核心组如下，另外两个 P 按同规则生成：

| N | B | 总点数 | FP32 复数单输入 MiB |
|---:|---:|---:|---:|
| 4096 | 4096 | 16777216 | 128 |
| 8192 | 2048 | 16777216 | 128 |
| 16384 | 1024 | 16777216 | 128 |
| 32768 | 512 | 16777216 | 128 |
| 65536 | 256 | 16777216 | 128 |
| 131072 | 128 | 16777216 | 128 |
| 262144 | 64 | 16777216 | 128 |
| 524288 | 32 | 16777216 | 128 |
| 1048576 | 16 | 16777216 | 128 |

| P | N | B | 实现/配置 ID | device ms | Gpoints/s | GM bytes/point | workspace MiB | model/search | 外部 speedup |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 16777216 | 65536 | 256 | pending | | | | | | |

单输入大小不等于完整预算，必须另计输出、scratch、系数、host staging 和参考实现。

## E04：饱和吞吐与负载拐点

- 自变量：`N=16384/65536/262144`；`B=1/4/16/32/64/128/256/512/1024`，另加每个
  合法 fold D 对应 `C×D-1/C×D/C×D+1`，去重后执行；C 从硬件 profile 读取。
- 固定条件：内存预算与超时在运行前固定；每个点都重新模型选型，也保留固定配置扫描作解释。
- 指标：吞吐、wave 数、有效核利用率、尾部时间、GM 带宽、每点耗时及转折处所选配置。
- 基线：外部 FFT、固定成熟长路径、模型/搜索配置。
- 完成判据：饱和区至少有三个连续测量点；尾块和配置切换被解释，不能假定 batch 越大必然更快。

| N | B | fold | 配置 ID | waves | device ms | Gpoints/s | 尾部 % | GM GB/s | 状态 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 65536 | 256 | | pending | | | | | | |

## E05：架构消融，而不是更换核心的混合比较

- 自变量：逐项比较物化交接/驻留子图、线性/在线重排、无重叠/合法双缓冲、单角色/多角色、
  固定分段/模型分段、固定配置/自动选型。只有存在合法等价实现的项才执行。
- 固定条件：`N=16384/65536/262144`，`B=1/16/256`；配对项固定算术核心、局部点数和其他参数。
  核心选型效果另列实验，不能把更快核心的收益全部记为架构收益。
- 指标：配对 ms/speedup、GM 读写、UB、同步/等待、重叠与模型选型损失。
- 基线：同一个完整混合数据流配置与其单因素变体；另报各变体重新搜索的最优 envelope。
- 完成判据：每项有合法性说明和配对参数；不可实现项标 unavailable；收益能与流量/等待变化对齐。

| N/B | 消融项 | 共同核心/局部点数 | 对照配置 | 变体配置 | 对照 ms | 变体 ms | speedup | 资源变化/限制 |
|---|---|---|---|---|---:|---:|---:|---|
| 65536/256 | 驻留 vs 物化 | | pending | pending | | | | |

## E06：长度相关数值误差

- 自变量：本计划中的八个长长度 `N=8192..1048576`；`B=1/3/16`；零、冲激、常量、单频、Nyquist、随机、高动态范围与
  近抵消，随机固定 seed `0/1/2`。不同结构性配置也要接受相同门禁。
- 固定条件：CPU FP64 参考；小 batch 全行验证；大 batch 采样规则预声明且覆盖组边界与尾部。
- 指标：最大绝对误差、归一化相对误差、RMS 误差、校验行数；逆向实现后增加 round-trip。
- 基线：FP64 参考和同语义外部 FFT 的误差；零/近零结果采用明确的绝对误差规则。
- 完成判据：所有 trial 通过；容差在结果采集前冻结，不能随失败放宽；误差随 N 的趋势可解释。

| N | B | 模式/seed | 配置 ID | 校验行数 | max abs | normalized rel | RMS | round-trip | 阈值/状态 |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 65536 | 3 | random/0 | pending | | | | | | |

## E07：长 FFT 应用代理与完整流水分开

- 自变量：下列应用形状；通道/帧到 batch 的映射、padding 和被计时阶段必须显式说明。
- 固定条件：同输入输出语义；FFT-only 与窗口/转置/乘法/IFFT 的完整流水分别计时。
- 指标：FFT 和流水延迟、吞吐、workspace、误差；应用代理不宣称完整应用加速。
- 基线：同设备外部 FFT；完整流水只有两边实现相同算子链后才比较。
- 完成判据：每类至少覆盖小/中/大负载；超出能力的项标 pending，不拿补零 FFT 替代原长度 DFT。

| 应用 | 预声明形状 | 工程依赖 |
|---|---|---|
| 长卷积代理 | N=4096/8192/16384/65536/262144，B=1/16/64 | 前向 C2C 代理；完整卷积还需逆向和逐点乘法 |
| 科学频谱代理 | N=65536/262144/1048576，B=1/4/16 | 稠密一维频谱；不冒充多维应用 |

OFDM、STFT、range-Doppler 和 2D 频域代理仍由 `config/test_matrix.json` 跟踪，但不进入本轮
长 C2C 的 E07 表；它们分别依赖短 FFT、R2C、二维 API 或额外布局合同，混合统计会破坏实验语义。

| 应用 | shape/轴/通道 | 计时链 | 实现 | FFT ms | 流水 ms | workspace MiB | 误差 | 基线/状态 |
|---|---|---|---|---:|---:|---:|---:|---|
| 长卷积代理 | 65536/16 | forward FFT only | pending | | | | | |

## E08：资源、profile 与原因验证

- 自变量：预声明 `N=4096/16384/65536/262144/1048576`，`B=1/16/256`；追加 E04 中实际观察到的
  首个配置切换两侧，以及 E05 的合法配对。追加条件预声明，不只 profile 最好点。
- 固定条件：同一二进制/配置；非 profile 性能和 profiler replay 耗时分开，不计算跨口径 speedup。
- 指标：UB/GM 实际量、MTE 时间、计算时间、队列重叠、等待、尾部、启动数量、有效 AIV 数；
  仅填写设备工具实际支持的计数器，缺失计数器标 unavailable。
- 基线：物化路径、驻留路径与同形状外部 FFT trace；模型结构计数与实测资源量配对。
- 完成判据：65536 至少覆盖低 batch 和饱和 batch；每个归因有 trace/计数器或控制实验支撑。
  总耗时下降不能单独证明流水重叠，也不能要求不存在依赖的全局同步。

| N/B | 配置 ID | UB KiB | GM read/write MiB | launches | MTE us | compute us | wait us | overlap % | 证据/限制 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 65536/256 | pending | | | | | | | | |

## 汇总、图与结论边界

最终报告按目的输出，而不是堆叠完整 CSV：E01 容量/实现阶梯图，E02 延迟曲线，E03 等总点数
吞吐图，E04 batch 拐点曲线，E05 配对消融柱图，E06 误差随长度图，E07 应用分组图，E08
时间线与流量对照。每张图注明实验 ID、形状、计时区间、单位和数据来源。

汇总同时列有效/失败/未支持点数、几何均值、最差 speedup、胜/平/负点数与模型选型损失。
“持平”使用采集前声明的波动容差，不能把所有小于 1 的点改称持平。主结论覆盖预声明的完整
矩阵；应用、短 FFT、长 FFT、不同计时区间分组统计，不能通过加入短 FFT 有利点抬高长 FFT 均值。

首先交付 G1 + E01/E02/E06 的 65536 证据，再完成 E03/E04 和 E05/E08 机制验证，最后纳入
应用流水。结果发布仍走[复现与发布流程](reproducibility.md)，尚未采集的本页不改变当前
[发布结果](results.md)或库的支持声明。

修改实验点时只编辑 `config/long_fft_experiments.json`，随后运行：

```bash
python3 scripts/generate_long_fft_tables.py
python3 scripts/generate_long_fft_tables.py --check
```
