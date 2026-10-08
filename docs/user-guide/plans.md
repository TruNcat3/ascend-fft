# Plan与选型

`Context`管理硬件描述、设计空间和内部stream；`Plan`管理kernel、系数、索引及设备缓冲区。前者确定候选，后者重复执行，不应每次调用都重新搜索。

## 生命周期

```text
初始化Context -> 为实际n、batch选型 -> 保留Plan反复执行
             -> 销毁Plan -> 销毁Context
```

Plan使用Context创建的stream，因此Context必须比其所有Plan活得更久。当前无线程安全保证，不要让多个线程同时操作同一Plan的准备状态和缓冲区。

## 自动选型

`ctx.select(n, batch, topK, &log)`枚举、模型排序、构建可执行候选、校验并实测，返回成功候选中最快的Plan。默认`topK=3`是目标成功测量数；不可装载候选会被跳过，因此不保证只检查三个条目。

`enumerate()`包含不可行点及原因，不等于所有条目都有kernel。当前`makePlan()`仅装载`pointSize==2`候选，主kernel内部的平面级radix-4融合不表示完整`P.radix=4`后端已实现。

```cpp
auto candidates = ctx.enumerate(n, batch);
for (const auto& c : candidates) {
    if (c.state == bfly::State::Infeasible)
        std::printf("%s: %s\n", c.id.c_str(), c.reason.c_str());
}
```

`makePlan(candidate)`不实测，适合明确候选的实验。调用者须检查可行性和正确性，不能绕过约束后宣称组合受支持。

## 准备与复用

`prepare(n, batch)`预生成并上传系数、索引，相关准备键不变时不重传；`run()`会自动准备。需要移出首次准备成本时传入实际batch，不要依赖省略参数时的折叠上界。

保持形状不变时复用Plan；改变长度、batch或硬件时重新选型。旧Plan可以重新准备，但旧候选不保证是新形状的最优点。当前无公开的持久化最佳配置或跨进程计划缓存。

## 计时

`measure()`在准备及初次拷贝后测量C2C启动加同步，10次取最小值，再对最多8个均匀采样batch进行CPU参考校验，回填Metric与Measured状态。

这不是`run()`的完整墙钟时间，后者还含拷贝及必要的分配/准备；也不是profiler的纯设备时长。发布对比须使用统一[实验协议](../benchmarks/methodology.md)。
