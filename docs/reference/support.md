# 支持范围

本页记录当前发布版本的已验证接口，不定义项目的最终目标边界；JSON里出现某个值不表示已有可执行kernel。

| 能力 | 当前支持 | 限制 |
|---|---|---|
| C2C | fp32，前向，1D，二次幂`64..4096` | 无公开逆向C2C |
| R2C | fp32，1D，二次幂`128..8192` | 需要`fft_real.o` |
| C2R | fp32，1D，二次幂`64..4096` | 合法半谱，包含`1/n` |
| Batch | 连续等长度 | 正整数，受uint32偏移与内存限制 |
| 用户布局 | 交错复数、实数稠密数组、稠密半谱 | 无任意stride/planar接口 |
| 执行 | 主机指针、同步、内部设备缓冲区 | 无device-pointer/用户stream API |
| 精度 | fp32 | 无公开fp16/bf16/fp64 |
| 维度/长度 | 1D二次幂 | 非二次幂、2D/多维尚未支持 |

主C2C性能矩阵覆盖`n={64,128,256,512,1024,2048,4096}`和`batch={1,4,16,64,256,1024,4096}`。不是所有合法batch的穷举，也不是所有硬件的性能保证。

当前发布实验使用 `ascend910_9382-migration-20261013`。研究后端另外完成了
`N=8192/16384/32768/65536`、`batch=1/16/256` 的 12 点规模切片；这是正确性与延迟证据，
不是公共 `makePlan()` 长度范围的扩展，也没有同尺寸专业库 speedup 结论。
实验范围与接口范围应分开阅读，详见[规模切片](../benchmarks/scale.md)。

## 硬件

已验证`Ascend910_9382`，48矢量核、每核192KiB UB，实验使用CANN 9.0.0。Context固定选择设备0，profile的`device_index`不改变执行设备。

`AB_SOC`只选择编译目标，不自动迁移成本系数和合法性约束。其他SoC须验证UB、矢量指令、对齐、repeat与计算通路，见[架构](../design/architecture.md)。

## 设计空间边界

- 当前 `makePlan()` 仅装载 canonical radix-2/shared-UB/交错连续布局候选；并同时检查固定 mapping、
  fusion、系数驻留、bit-reverse 和 F 参数。平面级 radix-4 融合不是完整 radix4/8 Plan 后端。
- 本目标没有SIMT shuffle；Cube探针不是公开FFT后端。
- 枚举保留不可行点供分析，成本模型不承诺整个理论空间的全局最优。

调用见[变换指南](../user-guide/transforms.md)。性能先读[实验设计与结果汇总](../benchmarks/results.md)，
再按[实数变换](../benchmarks/real-transforms.md)或[C2C](../benchmarks/c2c.md)查看同语义基线与负点；
原始来源见[数据与复现](../benchmarks/reproducibility.md)。
