# 快速开始

本页从仓库根目录运行，假定已完成[安装](installation.md)。先检查[支持范围](../reference/support.md)，无需先读历史日志。

## 检查三种变换

```bash
./build/fft_check 1024 16 10
AB_DIR=r2c ./build/fft_check 1024 16 10
AB_DIR=c2r ./build/fft_check 1024 16 10
```

参数依次为单个变换长度、batch数量、计时重复数。程序生成输入、执行NPU变换并与CPU参考比较。`AB_DIR`控制检查程序，不是Plan的全局方向开关。

检查框架选型：

```bash
./build/test_framework config/ascend910_93_profile.json \
  config/butterfly_space.json build/fft_radix2.o 1024 16
```

输出包含枚举、选型、误差与`PASS`。如果出现`r2c skipped`或`c2r skipped`，相应能力没有经过本次检查。

## 最小C++调用

输入和输出均为**主机**`float`缓冲区，复数以实部/虚部交错存放。Plan内部完成分配、拷贝、启动和同步。

```cpp
#include "butterfly/plan.hpp"
#include <cstdio>
#include <vector>

int main() {
    constexpr uint32_t n = 1024, batch = 16;
    bfly::Context ctx;
    if (int rc = ctx.init("config/ascend910_93_profile.json",
                          "config/butterfly_space.json",
                          "build/fft_radix2.o")) {
        std::fprintf(stderr, "init failed: %d\n", rc);
        return 1;
    }
    std::string log;
    auto plan = ctx.select(n, batch, 3, &log);
    if (!plan) {
        std::fprintf(stderr, "%s", log.c_str());
        return 1;
    }
    std::vector<float> input(static_cast<size_t>(batch) * 2 * n, 0.f);
    std::vector<float> output(input.size());
    for (uint32_t b = 0; b < batch; ++b)
        input[static_cast<size_t>(b) * 2 * n] = 1.f;
    if (int rc = plan->run(input.data(), output.data(), n, batch)) {
        std::fprintf(stderr, "run failed: %d\n", rc);
        return 1;
    }
    return 0;
}
```

用户应用使用与工程一致的编译函数：

```bash
source scripts/env.sh
ab_cxx example.cpp src/framework/butterfly.cpp \
  src/framework/reference.cpp -o build/example
./build/example
```

`example.cpp`指用户应用源文件，仓库不内置该文件。`select()`会实测候选，应在初始化阶段调用，再复用Plan。继续阅读[Plan指南](../user-guide/plans.md)和[变换布局](../user-guide/transforms.md)。
