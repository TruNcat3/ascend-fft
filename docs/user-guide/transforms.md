# 变换与数据布局

所有公开变换为fp32一维连续等长度batch。以下大小是`float`元素数，不是字节数。

| 变换 | 输入，每行 | 输出，每行 | 定义 |
|---|---|---|---|
| `Plan::run`，C2C | `2*n`，交错复数 | `2*n`，交错复数 | 前向、负指数、不归一化 |
| `Plan::runR2C` | `n`，实数 | `n+2`，交错半谱 | 前向实数FFT、不归一化 |
| `Plan::runC2R` | `n+2`，交错半谱 | `n`，实数 | 逆实数FFT、含`1/n` |

三种调用接收主机指针，返回时结果已拷回主机。不要传入`aclrtMalloc`分配的设备指针。

## C2C

第`b`行、第`j`个复数存于：

```text
input[b * 2*n + 2*j]       实部
input[b * 2*n + 2*j + 1]   虚部
```

输出为自然频率顺序，`X[k] = sum_j x[j] exp(-2*pi*i*j*k/n)`，与`numpy.fft.fft`的未归一化前向定义一致。当前没有公开逆向C2C方法。

## R2C与C2R

半谱包含`k=0..n/2`共`n/2+1`个复数，行距为`n+2`个float：

```text
half[b * (n+2) + 2*k]       实部
half[b * (n+2) + 2*k + 1]   虚部
```

R2C对应`numpy.fft.rfft(x, n=n)`，C2R对应`numpy.fft.irfft(half, n=n)`。合法实数频谱的DC和Nyquist均为实数，调用者应将其虚部置零；当前测试明确约束Nyquist虚部为零，接口不验证半谱的全部合法性。

R2C使用内层`n/2`复数FFT，建议按`ctx.select(n/2, batch)`选型；执行仍传实数长度`n`。C2R内层为`n`，按`ctx.select(n, batch)`选型：

```cpp
auto r2c = ctx.select(n / 2, batch);
std::vector<float> real(static_cast<size_t>(n) * batch);
std::vector<float> half(static_cast<size_t>(n + 2) * batch);
if (!r2c || r2c->runR2C(real.data(), half.data(), n, batch) != 0)
    return 1;

auto c2r = ctx.select(n, batch);
std::vector<float> restored(real.size());
if (!c2r || c2r->runC2R(half.data(), restored.data(), n, batch) != 0)
    return 1;
```

`select()`校验和测量的是对应长度的C2C核心，不是整条实数链。整体性能使用专用[复现实验](../benchmarks/reproducibility.md)。

## 形状与内存

- 使用二次幂长度：C2C `64..4096`、R2C `128..8192`、C2R `64..4096`。
- batch应大于零，且`batch*2*n <= UINT32_MAX`；实际还受内存容量限制。
- 不支持多维、任意stride或planar用户布局。
- 使用独立输入输出缓冲区；公开接口未声明可保证的原地/别名语义。

内部半谱行距`n+16`不是用户契约，不必手动padding。解释见[实数变换设计](../design/real-transforms.md)，边界见[支持范围](../reference/support.md)。
