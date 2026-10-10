# 计算核心与合法实现

计算核心是 [映射空间](architecture.md) 的一个选择项，而不是新的架构范式。这里“核心属于流映射
空间中的可替换选项、而不是方法本身”的分层观点来自
[cuButterfly](https://github.com/TruNcat3/cuButterfly)。当前生产路径使用 radix-2 FFT，并在
平面级融合相邻阶段为 radix-4；Cube 是已验证基础指令的研究选项，尚非生产 FFT backend。

## 为什么核心不是本项目唯一的优化对象

一个 FFT 实现至少有两个可独立变化的部分：

1. **局部计算核心**：在一个已经准备好的块中完成蝶形，例如 radix-2、radix-4、向量复乘或
   Cube DFT。它关注指令吞吐、复数乘法、数据类型、寄存器/UB 使用和局部数值误差。
2. **计算流组织**：决定块如何进入 UB、哪些阶段连续复用、多个数据块如何时间遍历、阶段边界
   是否落 GM、如何重排，以及生产者和消费者如何交接。它关注依赖、搬运、同步、填充/排空和
   稳态吞吐。

cuButterfly 的主贡献放在第二部分，是因为第一部分通常已有大量与硬件绑定的成熟技术；重新发明
 一个局部 radix 并不能说明跨算子、跨硬件的编排规律。Ascend-FFT 因此优先复用/实现合法的
 radix-2 与局部 radix-4，同时把核心留成可替换的 `ProcessingUnit`。未来可以接入更好的向量、
 Cube 或专用 FFT 核心，只要它满足同一 lowering contract。

这不是说核心不重要。核心和流组织是**正交选择、资源耦合**：同一个 radix-2 放入不同的 tile/
 boundary/pipeline，GM 和同步成本会不同；同一个流组织替换 radix-4 或 Cube，UB、对齐、尾部和
 指令成本也会不同。因此性能报告必须区分“固定核心比较流”和“固定流比较核心”，不能把联合
 搜索的结果直接写成单一层次的收益。

```text
候选 = 架构映射（分段/Us/Ts/Ud/Td/layout/residence/pipeline）
      × 计算核心（radix/vector/Cube/专用 FFT core）
      -- capability + lowering contract 过滤
      -- 模型预排序
      -- correctness 后实测回填
```

![Detailed Ascend data path and local processing unit](../figures/ascend_data_path_detail.svg)

这里的 processing unit 只是流框架中的局部算术位置；替换它不会自动改变阶段交接、数据复用或
GM 边界，反之亦然。

## 为什么同时使用 plane 与 planar

小配对距离的蝶形直接映射到 UB 时，会产生不满足 32 B 对齐的访问。K-plane 布局把这些配对转换成对齐矢量访问；完成局部阶段后再 Gather 到 planar 布局，后续阶段可以利用较大的连续配对距离。

K 增大并不总能减少成本：平面级指令数、后续组数、行切片和批折叠条件都会变化。当前 K 候选为 8/16/32，通过共享的 [fft_k.hpp](https://github.com/TruNcat3/ascend-fft/blob/master/include/butterfly/fft_k.hpp) 规则选择；host 索引、kernel 和模型必须使用同一规则。这个集合是当前实现 envelope，不是方法规定的固定尺寸。

## 已实现的核心与组织

| 选择 | 实现作用 | 合法性与成本 |
|---|---|---|
| radix-2 | planar 后续阶段与平面级残级 | 基础蝶形；独立 group 尽量折入 repeat |
| 平面级 radix-4 | 相邻两级局部融合 | 必须匹配布局与旋转因子；不能由一次负收益试验推断所有 radix-4 都无效 |
| 四指令复数乘路径 | 减少中间计算/发射 | 依赖实际指令与源操作数约束，不是改变 FFT 公式 |
| batch 折叠 D | 多批共享一组 repeat 组织 | 当前结构上限 4；步长与 repeat 受限；组数不足会降低活跃核数 |
| Gather 索引生成 | 位反转、转置、交错输出 | 索引由 host 生成并上传；不是每个数据点运行标量索引计算 |

`AB_PLANE_K` 与 `AB_FOLD_D` 是消融钩子，用于固定布局和折叠策略；用户使用路径应优先使用 Plan 的合法候选，而不是绕过约束强制参数。核心代码见 [fft_radix2.cpp](https://github.com/TruNcat3/ascend-fft/blob/master/src/ascendc/fft_radix2.cpp)。

## Cube：基础能力已证实，FFT 收益未证实

[cube_probe.cpp](https://github.com/TruNcat3/ascend-fft/blob/master/src/ascendc/cube_probe.cpp) 在独立 `__cube__` AIC kernel 中验证了 fp32 的 LoadData → Mmad → Fixpipe 链路。验证形状为 16×16×16；探针使用保守栅栏且没有双缓冲，其吞吐不能当作 Cube 峰值或未来 FFT 性能。

要将 Cube 作为 FFT 局部 DFT 核心，还需要设计 AIC/AIV 分工、NZ/ND 布局、复数矩阵分解、L1/L0/GM 交接、事件同步和输出聚合。仅把算术换为矩阵乘，不能消除重排和搬运成本。当前未验证更大 tiling 与 AIC/AIV 重叠，不应写成已完成的候选。

历史记录中“Cube 不可用”主要指纯 AIV kernel 的访问路径；后来独立 AIC 探针纠正了这一判断。“串行探针慢”也不等于硬件永远不适合 FFT。见 [决策归档](../archive/decisions.md)。

## 同步首先保证正确性

屏障消融只能移除已证明无依赖的屏障。当前保留缓冲首写/收尾所需的排空，防止在途操作污染复用缓冲；不能用一次正确结果证明删屏障安全。实数链的方向切换还需要保持旋转因子符号与 radix-4 交叉相位一致，见 [实数变换](real-transforms.md)。

当前成本估计如何反映这些选择：见 [性能模型与选型](performance-model.md)。
