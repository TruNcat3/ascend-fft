# 项目与作者

<p class="institution-mark">
  <a href="https://www.ustc.edu.cn/">
    <img src="assets/ustc-logo.png" alt="University of Science and Technology of China">
  </a>
</p>

Ascend-FFT 是 Teng Wang 独立维护的开源研究软件，研究工作开展于 High Efficient Intelligent
Computing Lab，并以中国科学技术大学苏州高等研究院为作者机构归属。项目将
[cuButterfly](https://github.com/TruNcat3/cuButterfly) 的混合空间-时间蝶形计算方法迁移到
Ascend NPU，并独立实现 AscendC kernel、硬件映射、性能模型与实测选型。

## 作者与联系

| 项目 | 信息 |
|---|---|
| 作者 / 维护者 | **Teng Wang** |
| 实验室 | High Efficient Intelligent Computing Lab |
| 机构 | [Suzhou Institute for Advanced Research, University of Science and Technology of China](https://sz.ustc.edu.cn/en/index.html) |
| 所在地 | Suzhou, China |
| 邮箱 | [wangt635@ustc.edu.cn](mailto:wangt635@ustc.edu.cn) |
| GitHub | [TruNcat3](https://github.com/TruNcat3) |

问题报告与可复现缺陷优先使用
[GitHub Issues](https://github.com/TruNcat3/ascend-fft/issues)；研究合作、引用与机构联系可使用上表邮箱。

## 机构归属声明

机构名称和标识用于说明作者的研究归属。Ascend-FFT 不是中国科学技术大学或中国科学技术大学
苏州高等研究院的官方软件发布，机构标识的出现不表示上述机构对本项目作认证、赞助或背书。

本页使用的中科大横向标识来自[中国科学技术大学官方网站](https://www.ustc.edu.cn/)。该标识不属于
Ascend-FFT 的 Apache-2.0 授权范围，项目使用者不得依据本仓库的软件许可证推定其可被再次授权。
来源和许可边界同时记录在 [`THIRD_PARTY_NOTICES.md`](https://github.com/TruNcat3/ascend-fft/blob/master/THIRD_PARTY_NOTICES.md)。

## 引用

使用软件、实验数据或移植实现时，请引用 Ascend-FFT；使用混合数据流设计方法时，还应引用其上游
方法仓库 cuButterfly。机器可读元数据见
[`CITATION.cff`](https://github.com/TruNcat3/ascend-fft/blob/master/CITATION.cff)。

```bibtex
@software{wang2026ascendfft,
  author       = {Teng Wang},
  title        = {Ascend-FFT: A Hardware-Mapped FFT Operator Library for Huawei Ascend NPUs},
  year         = {2026},
  url          = {https://github.com/TruNcat3/ascend-fft},
  organization = {High Efficient Intelligent Computing Lab,
                  Suzhou Institute for Advanced Research,
                  University of Science and Technology of China}
}
```

## 项目责任边界

- 发布结果只适用于对应 manifest 中记录的硬件、工具链、精度、shape 和计时协议。
- 华为、CANN 和 Ascend 的名称属于其各自权利人；相关工具链不随本仓库分发。
- 软件按 Apache-2.0 的“AS IS”条款提供；论文或产品使用者应独立完成适用性验证。

继续阅读：[设计动机](design/motivation.md) · [当前实验结果](benchmarks/results.md) ·
[未来计划](roadmap.md)。
