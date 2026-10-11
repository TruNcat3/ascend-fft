# 验证与发布

## 五道门禁

```bash
scripts/one_click_test.sh --no-matrix
```

顺序为 `test_limits`、`test_framework`、`stride_probe`、`fft_check` 和确定性数值输入 profile。
任一失败都必须终止，不能继续发布性能数字。外部库逐点胜负属于报告结果，不是 correctness
门禁；性能回归应绑定冻结的本库基线与明确容差。

## 完整验证

```bash
scripts/one_click_test.sh
scripts/repro.sh e2e
scripts/repro.sh r2c-c2r
```

发布前还需要构建 MkDocs、检查内部链接、验证生成物 freshness，并确认 README headline 与
发布 snapshot 一致。

## 发布纪律

- 当前结果和历史快照分开；历史文档不继续追加“当前状态”。
- headline 只来自固定协议和固定网格。
- 变换不同、硬件不同或计时区间不同的结果显式降级为参照。
- 图表、摘要和完整数据必须指向同一个 manifest。
- schema v3 发布必须来自干净提交，记录稳定硬件 ID、SoC、CANN/ccec、profile 哈希和构建产物哈希；
- 每个实验必须有显式的零退出码；raw trial 必须完整且无重复地覆盖 runner × round × shape；
  声明的图像同时归档到快照并绑定输出哈希，不能只绑定作图输入；
  `unknown`、dirty tree 或空构建哈希均拒绝发布。
- 历史快照若缺少上述证据，只能显式标记为 `legacy-unverified` 并列出限制，不得升级其证据等级。
- Citation、许可证和 cuButterfly 来源声明随 release 一起检查。
