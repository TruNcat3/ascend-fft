# 仓库结构

```text
config/              硬件事实与设计空间
include/butterfly/   Context / Plan / Candidate 公共 C++ 接口
src/ascendc/         AscendC kernel 与硬件探针
src/framework/       枚举、可行性、成本模型和 Plan 实现
src/host/            测试、基线与 profiling host 程序
scripts/             构建、实验、绘图和发布工具
tests/               无设备逻辑与框架门禁
docs/                用户、设计、实验与开发文档
results/             运行结果和显式发布快照
```

架构参数、计算核心和 lowering 约束应保持分层：设计空间描述“如何组织阶段、数据和驻留”；
AscendC kernel 描述“角色内部如何计算”；可行性检查决定某个核心能否映射到给定硬件组织。
