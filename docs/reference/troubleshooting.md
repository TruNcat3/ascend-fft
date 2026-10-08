# 故障排查

先从仓库根运行`scripts/init.sh --check`，解决环境问题后再检查kernel和形状。不要关闭同步或资源守卫掩盖错误。

| 现象 | 处理 |
|---|---|
| 找不到ccec/头文件 | 设置完整`AB_CANN`，检查工具包环境 |
| torch_npu import失败或NPU不可用 | 选择正确`AB_PY`，核对驱动/CANN/torch/torch_npu兼容 |
| 无msprof | 普通执行可继续，profiling须设置`AB_MSPROF` |
| Context初始化失败 | `-1`ACL、`-2`设备0、`-3`stream创建 |
| select返回空指针 | 保存log；检查长度、UB、布局、kernel装载及正确性原因 |
| 实数执行返回-1 | 检查主kernel同目录`fft_real.o`或设置`AB_REAL_O` |
| 执行返回-3 | 检查空闲内存，减小batch或排查共租户 |
| 执行返回-8 | 检查实数长度与`batch*2*n`上限 |
| kernel装载失败 | 核对对象的SoC/工具链，按当前环境重建 |
| 大误差 | 取消强制K/D与调试符号，核对主机布局、长度、半谱端点和容量 |

## 选型慢与性能波动

`select()`实测候选并执行CPU参考检查，放在初始化阶段，形状不变时复用Plan。记录n、batch、候选ID与环境后再比较。

共租户会引入抖动，最小值不是应用延迟保证。Plan的启动加同步、profiler设备任务和端到端拷贝不能混为一谈；按[实验协议](../benchmarks/methodology.md)比较。

## 实数链被跳过

`test_framework`可能对缺失kernel或不适用长度打印`skipped`。C2C成功不替代实数验证，运行适用尺寸：

```bash
AB_DIR=r2c ./build/fft_check 1024 16 10
AB_DIR=c2r ./build/fft_check 1024 16 10
```

## 提交问题

提供commit、SoC、驱动/CANN/Python/torch_npu版本，完整命令和环境覆盖项、返回码与log，n/batch/方向/布局，以及是否共租户、修改配置或通过默认门禁。

更深入诊断见[profiling](../development/profiling.md)，返回码见[API](api.md)。
