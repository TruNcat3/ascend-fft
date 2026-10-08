# 计算核心与合法实现

计算核心是 [映射空间](architecture.md) 的一个选择项，而不是新的架构范式。当前生产路径使用 radix-2 FFT，并在平面级融合相邻阶段为 radix-4；Cube 是已验证基础指令的研究选项，尚非生产 FFT backend。

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
