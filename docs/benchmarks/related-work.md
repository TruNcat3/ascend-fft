# 相关工作与基线

## 方法来源

Ascend-FFT 是 [cuButterfly](https://github.com/TruNcat3/cuButterfly) 的跨平台迁移实例。
cuButterfly 的[跨平台迁移清单](https://github.com/TruNcat3/cuButterfly/blob/main/docs/platform_ports.md)
也反向记录本项目。使用本项目的混合数据流方法时，应同时引用 Ascend-FFT 与 cuButterfly；
软件引用元数据见仓库根目录的 `CITATION.cff`。
继承的是“阶段维度与数据维度同时进行空间/时间划分、硬件 profile 驱动可行性判断、成本模型预排序、
实测回填”的架构范式；CUDA kernel 没有直接复制。Ascend 版本重新映射到 AIV、UB、MTE 和 GM，
并重新实现计算核心、布局变换和同步。

## Ascend 基线

- [Huawei CANN](https://www.hiascend.com/software/cann)：运行时、编译器和原生算子环境。
- [torch_npu](https://github.com/Ascend/pytorch)：当前严格可比的复数 C2C FFT 调用路径。
- `aclRfft1D`：CANN 提供的实数 FFT C API，只用于 R2C 或调用固定开销参照。

## GPU 与通用高性能库

- [cuFFT](https://docs.nvidia.com/cuda/cufft/)：NVIDIA GPU FFT 基线。
- [rocFFT](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocfft)：AMD GPU FFT 库。
- [CUTLASS](https://github.com/NVIDIA/cutlass)：可组合 GPU 计算核心与 profiler 的代表。
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)：硬件感知算子和分硬件报告性能的代表。
- [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform)：专用蝶形变换库。

不同硬件上的公开数字用于说明量级和设计趋势，不能与 Ascend 同卡结果合并计算 speedup。
跨平台比较必须同时报告硬件、软件版本、变换语义、精度、shape 和计时区间。
