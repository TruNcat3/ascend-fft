# 测试策略与扩展矩阵

当前实验是短中长度、高 batch FFT 的基础证据，不是通用 FFT 所有应用的覆盖证明。本页将测试
拆为五层，并保留历史矩阵作为对照。机器可读清单是
[`config/test_matrix.json`](https://github.com/TruNcat3/ascend-fft/blob/master/config/test_matrix.json)；
清单描述测试目标，不会让尚未实现的 kernel 自动成为可用能力。

## 当前证据与边界

| 项目 | 当前范围 | 仍需补齐 |
|---|---|---|
| C2C | fp32、前向、稠密 1D，N=64..4096 的二次幂 | 逆向、长 FFT、更多精度 |
| R2C | fp32，N=128..8192 的二次幂 | 非二次幂、更多精度（P3） |
| C2R | fp32，N=64..4096 的二次幂，含 1/N | 非二次幂、更多精度（P3） |
| 主性能网格 | 7 个长度 × 7 个 batch，共 49 点 | 核数拐点、尾块、独立模型验证 |
| 应用代理 | OFDM、雷达、DL 共 12 个 shape 的性能，四类负载（含 STFT 帧代理）的正确性 | STFT 性能代理、长卷积、range-Doppler 与 2D |
| 用户布局 | 连续数组、host 指针、同步 | 任意 stride、device-pointer、用户 stream |

主网格最大点 `N=4096, B=4096` 有 16,777,216 个复数，输入约 128 MiB。因此总负载并不小，
但单次 FFT 长度较短。`4096×4096` 和 `1048576×16` 的总点数相同，却具有不同的阶段深度、
片上容量需求和边界依赖，不能互相替代。当前 C2C 无法执行后一种形状。

支持契约见[支持范围](reference/support.md)，已发布结果见[当前结果](benchmarks/results.md)。

## 五层测试与退出条件

历史 `current` 是冻结对照，不计作新的覆盖层；后续发布不得覆盖其原始快照。

| 层级 / 配置 ID | 目的 | 主要矩阵 | 退出条件 |
|---|---|---|---|
| 快速门禁 / `smoke` | 每次改动发现基本错误 | C2C N=64/256/1024/4096，B=1/3/47/48/49；实数上下界 | 所有数值与非法输入测试通过 |
| 回归 / `regression` | 夜间检查尾块、复用与硬件拐点 | 全支持长度，奇数 batch、48D±1、多个 seed | 每轮正确；生命周期与误差无回退 |
| 发布 / `publication` | 预声明完整性能证据 | 历史点、硬件边界、固定总点数、实数与应用代理 | 同语义基线、原始 trial、所有输赢点齐全 |
| 压力 / `stress` | 内存、长期执行与错误恢复 | 1 MiB..1 GiB 输入预算，10,000 次 Plan 复用 | 无泄漏、越界、挂起，预算与失败可解释 |
| 长 FFT 研究 / `future-long-fft` | 验证跨片上容量的架构推广 | N=8192..1048576，B=1/4/16；长卷积、2D | 先实现后采证据，不进入当前发布均值 |

`runnable=true` 只表示该层的 C2C shape 矩阵可由现有执行路径运行。`real`、
`numeric_patterns`、`applications` 和 `requirements` 是整层验收清单；只运行 C2C 不能将整层
标为完成。`scripts/run_test_profile.py` 会执行当前可用的 C2C、实数、数值和应用代理；`stress`
和 `future-long-fft` 明确禁止直接运行，分别依赖预算化 runner 和新后端。
长 FFT 的假设、变量、退出条件和空数据合同分别见[目的化实验计划](benchmarks/long-fft-plan.md)
与[预声明数据表](generated/long-fft-tables.md)；机器清单以 `config/long_fft_experiments.json` 为准。

```bash
python3 scripts/run_test_profile.py --list
python3 scripts/run_test_profile.py smoke --build
python3 scripts/run_test_profile.py regression --trials 3
python3 scripts/run_test_profile.py future-long-fft  # 明确拒绝，不能伪装成已支持
```

## 硬件拐点与总点数对照

Ascend910_9382 有 48 个 AIV。数据折叠 D=1/2/3/4 时，除常见幂次 batch 外，应固定检查
`48D-1, 48D, 48D+1`：

```text
47 / 48 / 49       95 / 96 / 97
143 / 144 / 145    191 / 192 / 193
```

这些点检验核分配、折叠切换、尾块和时间遍历，而不是假定性能一定在该处突变。其他 SoC 必须
从实际 profile 读取核数与合法 fold 后重新生成边界，不能直接复用 48。

发布矩阵另设 `N×B=2^20、2^24` 的对照，对每个 N 使用 `B=总点数/N` 并与主网格去重。
这将“每次 FFT 的长度”与“总数据量”分开；未来长后端加入 `2^26` 对照。内存预算必须包含
输入、输出、workspace、host staging 和参考实现，不能只用总点数估算一次输入。

## 数值与接口正确性

输入应覆盖全零、冲激、常量、单频复指数、Nyquist 交替序列、固定 seed 随机、高动态范围和
近抵消。性能输入分布主要影响数值证据，FFT 的运行形状仍由长度、batch、布局与选型决定。

- smoke 小 batch 检查所有行；大 batch 使用预声明覆盖规则，包含头部、尾部及分组边界，不能只看第一行。
- 所有 trial 均须通过，误差报告取所有 trial 的最大值，不可用“任一轮通过”或最小误差替代。
- 记录绝对误差、归一化相对误差与 RMS 误差，明确近零输出规则；长 FFT 的容差需重新验证。
- R2C/C2R 检查 DC、Nyquist、共轭对称性、合法半谱、1/N 归一化及 round-trip。
- 接口检查非法长度、零 batch、指针、偏移溢出、重复 prepare/run、形状变更、构造与销毁。
- NaN/Inf 行为需先定义契约，再增加相应测试；尚无异步接口时不宣称已通过 stream 并发测试。

## 应用矩阵

| 负载 | 本轮目标 | 说明 |
|---|---|---|
| OFDM | N=2048/4096，B=14/140 | 保留现有符号批次代理；不是完整通信链路 |
| 雷达 range FFT | N=1024/2048，B=64/256 | 保留现有代理；不等价于 range-Doppler 全流程 |
| DL 频域层 | N=1024/4096，B=32/128 | 保留历史对照，不代表长序列卷积 |
| STFT | R2C N=256..8192，B=1/8/64/256 | 连续帧代理正确性已在 `publication` 采集（24 例 ×3 trial）；性能与重叠窗口、stride 成本另计 |
| 长卷积 | N=4096/16384/65536/262144 | 超出支持范围的部分等待长 FFT 后端 |
| 2D / range-Doppler | 方形 256²..2048²，矩形 1024×128 等 | 等待 2D API，转置与 workspace 必须计入 |

应用代理只能说明所用 shape 的性能与数值，不得标为完整应用端到端加速。新增应用必须说明
输入布局、通道如何映射到 batch、方向、归一化和被计时的流水段。

## 发布与模型验证规则

1. 在测量前固定矩阵与排除规则；原始失败、超时、内存不足和落后点都保留。不得增加有利点后只报告几何均值。
2. 同语义匹配 precision、direction、normalization、layout、placement、batch 和计时区间。
3. 分别报告 device-only、同步 host end-to-end、Plan 创建、首次执行和稳态；原始 trial、
   中位数、离散度、最大误差与最终配置一起保存。
4. correctness 是硬门禁；性能回归使用明确阈值与同构建基线，而不是要求所有点永远超过专业库。
5. profile 校准集与留出验证集分开，报告模型选型相对有限实测搜索的损失及搜索成本。
6. trace 应覆盖多个长度和 batch 拐点，解释 MTE/计算重叠、UB 使用、GM 搬运、等待与尾部，
   不能仅根据总耗时归因。

## 实施优先级

- P0：统一正确性入口、所有 trial 门禁、边界 batch、数值模式和原始逐例记录已在目标
  Ascend910_9382 完成采集：五道门禁、`smoke` 65/65、`regression` 349×3=1047/1047、
  `publication` 253×3=759/759（含 48D±1 边界批次、`2^20`/`2^24` 定总点数对照、实数
  上下界、两种数值模式与 STFT 帧代理）；49 点性能矩阵以逐 trial 协议复测，自研与原生
  各 3 轮共 294/294 PASS。逐例记录见 `results/runs/<UTC>-<profile>/`（`cases.csv` +
  `summary.json`；性能矩阵另存 `matrix.md`/`matrix.csv`/`trials.csv`/`summary.json`，
  均不进版本库）。
- P1：独立模型验证和预算化压力 runner；R2C/C2R 新协议发布快照已完成（见
  [当前结果 · 实数变换](benchmarks/results.md)）。
- P2：分段/多 AIV 长 FFT；显式记录阶段交接、重排、GM 流量和同步后，再测长卷积代理。
- P3：逆向 C2C、2D/stride、其他精度和非二次幂。补零不得冒充原长度 DFT。

这些是待完成工作，不是新增支持声明。整体工程依赖与验收见[未来计划](roadmap.md)，
复现协议见[测量方法](benchmarks/methodology.md)和[验证与发布](development/validation-and-release.md)。
