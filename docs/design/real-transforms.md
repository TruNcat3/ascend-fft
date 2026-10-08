# 实数 FFT：r2c 与 c2r

实数链复用已有复数蝶形核心，通过预处理/后处理实现半谱语义，而不是维护另一套 FFT 本体。公共接口见 [plan.hpp](https://github.com/TruNcat3/ascend-fft/blob/master/include/butterfly/plan.hpp)，实现见 [fft_real.cpp](https://github.com/TruNcat3/ascend-fft/blob/master/src/ascendc/fft_real.cpp)。

## 链路与语义

```text
r2c：N 个实数 → 视为 N/2 个交错复数 → N/2 点 FFT → 半谱拼接
c2r：半谱 → 共轭镜像满谱 → 正相位 N 点 FFT → 实部提取与 1/N 归一化
```

r2c 将偶/奇输入成对视为复数，利用共轭关系和旋转因子恢复 `N/2+1` 个频点。c2r 使用正相位旋转因子，同时反转局部 radix-4 的交叉相位；不能只改表而忽略核心内的硬编码相位。

| 契约 | 当前要求 |
|---|---|
| 数值类型 | fp32 实数 / 交错 fp32 复数 |
| 公共半谱布局 | 每批 `N/2+1` 个复数，即 `N+2` 个 float |
| 设备半谱行距 | `N+16` 个 float，保留对齐与搬运 slack；由框架转换 |
| c2r 归一化 | 输出包含 `1/N` |
| Nyquist 虚部 | 必须为零；不能假定与忽略该值的参考 API 完全等价 |
| r2c 已验证长度 | 2 的幂，128..8192 |
| c2r 已验证长度 | 2 的幂，64..4096 |

上述是当前实现 envelope，不是实数 FFT 算法的数学限制。缺少 `fft_real.o` 时不影响 c2c，但实数接口不可执行；方向切换必须刷新对应符号的旋转因子缓存。

## 为什么设备布局不等于公共布局

DataCopy/Gather 有长度和基址对齐要求，直接把稠密半谱作为所有设备行距会破坏行尾搬运。框架负责公共稠密布局与设备 padded 布局转换，kernel 使用设备行距。比较库性能时应匹配变换语义，并分别说明设备时间和包括拼装/搬运的端到端时间。

复用 UB 前保留必要的排空屏障，防止在途写与缓冲复用发生竞争。这个正确性约束不能因减少启动或缓冲成本而绕过。

## 验证与复现

```bash
scripts/repro.sh r2c-c2r
scripts/one_click_test.sh --no-matrix
AB_DIR=r2c ./build/fft_check 1024 64 30
AB_DIR=c2r ./build/fft_check 1024 64 30
```

验收使用双精度参考与仓库误差判据；历史原始结果见 [r2c_c2r.json](https://github.com/TruNcat3/ascend-fft/blob/master/results/r2c_c2r.json)。最新性能与基线解释集中在 [实验结果](../benchmarks/results.md)，历史实现细节见 [决策归档](../archive/decisions.md)。
