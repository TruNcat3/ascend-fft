# 原始数据与复现

当前报告只使用不可变快照 `ascend910_9382-migration-20261013`。快照记录实验命令、源码提交、
硬件与工具链身份、硬件 profile、构建产物哈希、原始 trial 和派生产物哈希。查看
[快照目录](https://github.com/TruNcat3/ascend-fft/tree/master/results/published/ascend910_9382-migration-20261013)
或直接从下面的索引进入所需证据。

## 最短路径

```bash
scripts/init.sh --check
scripts/one_click_test.sh --no-matrix
scripts/repro.sh matrix
```

第一条检查 CANN、Python、SoC 和工具路径；第二条完成编译与正确性门禁；第三条运行
C2C 49 点矩阵，并在同一目录生成 `matrix.md`、`summary.json`、`trials.csv` 和
`protocol.json`。实验默认写入 `results/runs/<UTC>-<experiment>/`，不会直接覆盖发布文档。

## 实验注册表

```bash
scripts/repro.sh --list
scripts/repro.sh --list --verbose
scripts/repro.sh --doc benchmarks/results.md
```

常用实验包括 `matrix`、`sixway`、`e2e`、`e2e-app`、`r2c-c2r`、`scale`、`eta`、`ab`、
`hwprobe` 和 `profile`。慢速实验和历史基线不会被 `repro.sh all` 隐式覆盖。

## 当前快照索引

| 内容 | 汇总 | 逐轮原始数据 |
|---|---|---|
| 来源、协议、环境与哈希 | [manifest.json](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/manifest.json) | [protocol.json](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/protocol.json) |
| C2C 49 点 | [matrix.csv](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/matrix.csv) | [trials.csv](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/trials.csv) |
| C2C device/E2E | [e2e.json](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/e2e.json) | [e2e.raw_rounds.csv](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/e2e.raw_rounds.csv) |
| 应用代理 shape | [e2e_app.json](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/e2e_app.json) | [e2e_app.raw_rounds.csv](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/e2e_app.raw_rounds.csv) |
| R2C/C2R | [r2c_c2r.json](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/r2c_c2r.json) | [r2c_c2r.raw_trials.csv](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/r2c_c2r.raw_trials.csv) |
| 长度规模切片 | [scale_slice.csv](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/scale_slice.csv) | [scale_slice.raw_trials.csv](https://github.com/TruNcat3/ascend-fft/blob/master/results/published/ascend910_9382-migration-20261013/scale_slice.raw_trials.csv) |

`r2c_c2r.json` 已合并主网格和 `N=64/8192` 补充采集；补充采集的独立文件也保留在同一目录。
这些链接是审计入口，不要求读者先浏览历史 PR 或生成大表。

## 发布结果

一次运行只产生候选结果。确认硬件、协议和正确性后，再显式发布：

```bash
export AB_HARDWARE_ID=<stable-device-or-node-id>
python3 scripts/publish_results.py --run results/runs/<run-id> --snapshot <new-snapshot-id>
python3 scripts/publish_results.py --check --snapshot <new-snapshot-id>
```

发布步骤创建新的不可变 `results/published/<snapshot>/`，并更新 `docs/generated/` 和受控的图表输入；
已有快照默认拒绝覆盖，`--replace-existing` 仅用于明确审阅过的历史迁移。手写的设计分析与
结果解释不会被生成器整页覆盖。当前 schema v3 门禁要求干净工作树、完整 commit、非 unknown 的
SoC/硬件 ID/CANN/ccec，以及 profile 和构建产物哈希。旧 schema v2 快照只能以
`legacy-unverified` 状态保留，不能作为精确可复现的新发布。

## 重绘图表

```bash
scripts/repro.sh figures
# 等价于：python3 scripts/plot_results.py
```

绘图器默认从 `results/published/ascend910_9382-migration-20261013/` 读取 C2C、端到端、应用
shape、R2C/C2R 和规模切片，不重新测量。输出统一写入 `docs/figures/`，每张图按内容使用原生
宽高比，不再强制填入统一画布。若某类输入缺失，绘图器会明确跳过对应图，不从另一份历史快照
静默拼接数据。

## 结果检查

发布前至少确认：

1. manifest 中的提交、设备、CANN 和计时协议完整，源码工作区干净、硬件身份非 unknown；
2. 所有输出和误差均为有限值，满足对应精度的相对/绝对误差门禁；零参考也须检查绝对误差；
3. baseline 与候选的变换、精度、布局和计时区间一致；
4. 每个实现每个 shape 的 trial 数完整；
5. `publish_results.py --check` 和文档构建均通过。

文档发布还应运行：

```bash
mkdocs build --strict
python3 scripts/check_docs_assets.py site --site-prefix /ascend-fft/
```

更深入的 A/B、硬件探针和 msprof 流程见[验证与发布](../development/validation-and-release.md)
及[性能剖析](../development/profiling.md)。
