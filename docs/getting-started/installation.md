# 安装与环境检查

当前工程从源码构建，不提供预编译安装包。已验证环境是 Ascend910_9382 与 CANN 9.0.0；其他环境须先探测和验证，不能直接沿用性能参数。

## 前置条件

| 组件 | 用途 |
|---|---|
| Ascend驱动、固件与可访问的NPU | 设备执行；当前Context使用设备0 |
| CANN工具包，含`ccec`、头文件和`libascendcl` | kernel与host构建 |
| `g++`，支持C++17 | host框架与检查程序 |
| Python与匹配的`torch`、`torch_npu` | 环境检查、矩阵与原生基线 |
| `msprof`，可选 | profiling；缺失不影响普通执行 |

系统组件须按对应发行版的兼容要求安装。初始化脚本检查现有环境，不会安装驱动、CANN或Python依赖。

## 从零构建

```bash
git clone https://github.com/TruNcat3/ascend-fft.git
cd ascend-fft
scripts/init.sh --check
scripts/init.sh
```

`--check`只体检；默认初始化执行体检、编译和五道门禁，不跑性能全矩阵。成功输出`INIT OK`，失败返回非零退出码。门禁包括资源边界、框架、stride探针、FFT抽样和确定性数值输入；明确标记的预期stride失败属于边界测试。

显式选择工具包与解释器：

```bash
AB_CANN=/path/to/cann AB_PY=/path/to/python3 scripts/init.sh --check
AB_CANN=/path/to/cann AB_PY=/path/to/python3 scripts/init.sh
```

`scripts/env.sh`探测工具链并加载可用的`set_env.sh`。详细规则见[配置](../reference/configuration.md)。

## 构建产物

```bash
scripts/build.sh all stride
./build/fft_check 1024 16 10
```

主要产物是`build/fft_radix2.o`、`build/fft_real.o`、`build/fft_check`和框架测试。C2C需要主kernel对象，R2C/C2R还需要`fft_real.o`。这是源码工程，不是`pip install`型包。

抽样性能检查使用`scripts/init.sh --quick`，完整矩阵使用`--matrix`。性能与正确性判据不同，见[实验协议](../benchmarks/methodology.md)。

## 换SoC

设置匹配的`AB_SOC`编译目标，运行探针，审阅回填硬件配置，然后验证资源约束、正确性与性能。`init.sh`不自动构建新硬件的成本模型，也不自动回填profile。移植方法见[架构](../design/architecture.md)，失败处理见[故障排查](../reference/troubleshooting.md)。
