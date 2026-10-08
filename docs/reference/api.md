# C++ API参考

声明见[`include/butterfly/plan.hpp`](https://github.com/TruNcat3/ascend-fft/blob/master/include/butterfly/plan.hpp)，命名空间`bfly`。当前无稳定C ABI、已安装共享库接口或用户stream注入。

## Context

| 方法 | 行为与结果 |
|---|---|
| `init(profilePath, spacePath, kernelPath)` | 读配置、初始化ACL、选择设备0并创建stream；0成功 |
| `hardware()` / `space()` / `kernelPath()` | 查询当前描述或路径 |
| `enumerate(n, batch)` | 返回候选与不可行原因，不执行kernel |
| `select(n, batch, topK=3, log=nullptr)` | 模型排序、校验和实测；失败为空`unique_ptr` |
| `makePlan(candidate)` | 装载指定候选；不可实现或装载失败返回空指针 |

`init()`没有完整JSON schema校验：缺失字段或文件可能采用默认值。不能将成功返回当作profile全部可信的证明。每个Context建议只初始化一次。

## Plan

| 方法 | 行为 |
|---|---|
| `candidate()` | 当前候选和指标的只读引用 |
| `transforms()` | 动作描述，不是完整设备trace |
| `prepare(n, batch=UINT32_MAX)` | 上传C2C系数和索引；推荐实际batch |
| `run(in, out, n, batch)` | 前向C2C，主机交错复数缓冲区 |
| `runR2C(in, out, n, batch)` | 实数到半谱，输出每行`n+2`个float |
| `runC2R(in, out, n, batch)` | 半谱到实数，含`1/n` |
| `measure(n, batch, Metric*)` | C2C计时、校验及状态回填 |

执行方法同步完成并拷回结果。调用者保证缓冲区容量；接口不接收容量参数。空指针、容量不足和非法半谱不属于受支持调用。布局见[变换指南](../user-guide/transforms.md)。

## 执行返回码

以下只适用于`run/runR2C/runC2R`；`init/prepare/measure`不共享同一错误枚举。

| 代码 | 执行阶段 |
|---|---|
| `0` | 成功 |
| `-1` | 主kernel或实数kernel未装载 |
| `-2` | 系数/索引准备失败 |
| `-3` | 设备分配失败 |
| `-4` | H2D失败 |
| `-5` | 启动失败 |
| `-6` | stream同步失败 |
| `-7` | D2H失败 |
| `-8` | 实数长度守卫或uint32偏移上限 |

`init`的`-1/-2/-3`分别为ACL初始化、设备选择、stream创建失败。`select`失败优先保存log和候选原因。

## 生命周期

Plan析构释放设备资源和kernel对象，Context析构销毁stream；Context不调用`aclFinalize()`或重置设备。集成程序须协调进程级ACL生命周期，不能在Plan活跃时关闭ACL或销毁Context。选型成本见[Plan指南](../user-guide/plans.md)，诊断见[故障排查](troubleshooting.md)。
