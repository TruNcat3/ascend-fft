# 复现实验

## 最短路径

```bash
scripts/init.sh --check
scripts/one_click_test.sh --no-matrix
scripts/repro.sh matrix
```

第一条检查 CANN、Python、SoC 和工具路径；第二条完成编译与正确性门禁；第三条运行
C2C 49 点矩阵。实验默认写入 `results/runs/<UTC>-<experiment>/`，不会直接覆盖发布文档。

## 实验注册表

```bash
scripts/repro.sh --list
scripts/repro.sh --list --verbose
scripts/repro.sh --doc benchmarks/results.md
```

常用实验包括 `matrix`、`sixway`、`e2e`、`e2e-app`、`r2c-c2r`、`eta`、`ab`、
`hwprobe` 和 `profile`。慢速实验和历史基线不会被 `repro.sh all` 隐式覆盖。

## 发布结果

一次运行只产生候选结果。确认硬件、协议和正确性后，再显式发布：

```bash
python3 scripts/publish_results.py --run results/runs/<run-id>
python3 scripts/publish_results.py --check
```

发布步骤更新 `results/published/`、`docs/generated/` 和受控的图表输入；手写的设计分析与
结果解释不会被生成器整页覆盖。

## 结果检查

发布前至少确认：

1. manifest 中的提交、设备、CANN 和计时协议完整；
2. 所有点满足 `maxRel <= 1e-4`；
3. baseline 与候选的变换、精度、布局和计时区间一致；
4. 每个实现每个 shape 的 trial 数完整；
5. `publish_results.py --check` 和文档构建均通过。

更深入的 A/B、硬件探针和 msprof 流程见[验证与发布](../development/validation-and-release.md)
及[性能剖析](../development/profiling.md)。
