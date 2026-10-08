# 验证与发布

## 四道门禁

```bash
scripts/one_click_test.sh --no-matrix
```

顺序为 `test_limits`、`test_framework`、`stride_probe`、`fft_check`。任一失败都必须终止，
不能继续发布性能数字。

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
- Citation、许可证和 cuButterfly 来源声明随 release 一起检查。
