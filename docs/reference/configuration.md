# 配置参考

环境、硬件profile和设计空间分别作用于构建、资源约束与候选枚举，不能用强制调参替代硬件验证。

## 环境变量

脚本由`scripts/env.sh`探测环境，Python侧为`scripts/abenv.py`。

| 变量 | 用途 |
|---|---|
| `AB_CANN` | 完整工具包根，含`bin/ccec`、`include`和`lib64` |
| `AB_PY` | 可import torch/torch_npu的解释器，自动选择缓存于`.ab_py` |
| `AB_SOC` | 编译SoC，默认`Ascend910_9382` |
| `AB_PROFILE` | 硬件 profile 路径；默认仓库内的 Ascend910_9382 profile |
| `AB_MSPROF` | 可选profiler路径 |
| `AB_ROOT/AB_BUILD/AB_WORK` | 环境辅助函数使用的仓库/构建/临时目录 |

部分脚本和可执行程序仍使用相对`build/`路径，不要假定覆盖`AB_BUILD`就能改变全部产物位置。推荐从仓库根运行。

## 检查与实验开关

| 变量 | 入口 | 含义 |
|---|---|---|
| `AB_DIR` | `fft_check` | `c2c`默认、`r2c`、`c2r` |
| `AB_FFT_O` | `fft_check`等基准 | 主kernel对象；Context通过`init`参数指定 |
| `AB_REAL_O` | 检查程序与Context | 实数kernel；Context默认查主kernel同目录 |
| `AB_PLANE_K/AB_FOLD_D` | `fft_check`研究对照 | 强制K/D，不是公开Plan setter |
| `AB_E2E` | `fft_check` | 额外端到端计时重复数 |
| `AB_E2E_HOST` | 端到端基准 | `pinned`默认或`pageable`对照 |
| `AB_E2E_MODE` | 端到端基准 | `async`默认、`sync`、`xfer`纯传输 |
| `AB_INPUT` | `fft_check` | `sin`默认、`ofdm/radar/dl`；C2R不采用此生成器 |

强制K/D用于消融，须遵守守卫，不保证更快；R2C作用于内层长度`n/2`。计时解释见[实验协议](../benchmarks/methodology.md)。

## 硬件profile

[`config/ascend910_93_profile.json`](https://github.com/TruNcat3/ascend-fft/blob/master/config/ascend910_93_profile.json)记录SoC、核数、UB、矢量宽度、L2及证据。loader 严格验证当前实现消费的必需字段、类型和正资源值；文件缺失或配置非法时 `Context::init()` fail closed。仅 `l2_bytes`、SIMT/子块和少量对齐能力字段具有明确的保守默认值。

当前profile保留历史`cann_version`和部分占位证据，不要用该字符串代替实际安装版本。能力约束还须核对 loader 已消费字段、设计空间与探针；未被 loader 消费的字段不是运行时参数。初始化会同时检查运行设备名、`AB_SOC` 编译目标与 profile 是否一致。换硬件先探测再审阅回填，当前不自动标定。

## 设计空间

[`config/butterfly_space.json`](https://github.com/TruNcat3/ascend-fft/blob/master/config/butterfly_space.json)以H/G/A/P/L/F/Q描述候选。主要枚举轴为：

| 字段 | 作用 |
|---|---|
| `A.ud_core` | 并行矢量核数 |
| `A.ts` | 时间组织，当前kernel要求1 |
| `A.local_exchange` | shared/register/shuffle与硬件合法性 |
| `P.radix/P.fusion_level` | 计算结构候选，不保证都有kernel |
| `P.coefficient_residency` | 系数驻留，当前kernel要求开启 |
| `L.in/L.out` | 布局候选，当前用户执行仅交错复数 |

其他字段包含说明性元数据，不能假定每条都有独立运行时效果。修改后重新初始化Context和选型；当前不提供热重载。合法候选还须经过`makePlan()`的实际支持检查。
