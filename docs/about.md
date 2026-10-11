# 项目与作者

<p class="institution-mark">
  <a href="https://www.ustc.edu.cn/">
    <img src="../assets/ustc-logo.png" alt="University of Science and Technology of China">
  </a>
</p>

Ascend-FFT 是 Teng Wang 独立维护的开源研究软件，研究工作开展于 High Efficient Intelligent
Computing Lab，并以中国科学技术大学苏州高等研究院为作者机构归属。项目将
[cuButterfly](https://github.com/TruNcat3/cuButterfly) 的混合空间-时间蝶形计算方法迁移到
Ascend NPU，并独立实现 AscendC kernel、硬件映射、性能模型与实测选型。

这里的迁移对象是**方法层的计算流编排**，不是某个 CUDA 计算核心的移植。我们优化的是
“数据如何流经局部计算”：阶段/数据的空间展开、时间复用、布局交接、片上驻留和流水；局部
“计算是什么”则作为可替换的 processing-unit 选项。radix-2、radix-4、Cube 或其他成熟 FFT
核心都可以在满足 Ascend 的对齐、UB、指令和同步 lowering contract 后接入。二者在概念上正交、
在物理资源上耦合，因此性能收益必须用相同流组织比较核心，或用相同核心比较流组织，不能把
局部核心改进自动归因于混合数据流方法。

本仓库关于 Roofline、依赖导致的流水问题、空间/时间展开边界、流与核心的正交分层及其详细
设计结论，来源于并应归属于 [cuButterfly](https://github.com/TruNcat3/cuButterfly)。Ascend-FFT
的独立贡献是将这些观点迁移到 Ascend 平台，完成 AIV/UB/MTE/GM 的具体 lowering、硬件 profile、
候选搜索、正确性门禁和性能证据归档。引用方法论时请同时引用 cuButterfly；引用 Ascend 平台实现
和实验时再引用 Ascend-FFT。

**方法范围**：

- 主研究对象：架构映射、计算流、参数搜索和合法 lowering contract；
- 可替换空间：局部 radix/FFT/Cube/向量计算核心；
- 平台贡献：AscendC、AIV/UB/MTE/GM 映射、硬件 profile 和可复核实验归档。

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
