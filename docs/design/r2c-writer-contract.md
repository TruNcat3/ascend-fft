# R2C Writer Contract

## 1. 语义层契约 (Semantic Layer)

### 数学定义
给定实数输入 `x[0..n-1]`，R2C 输出半谱 `out[0..m]`（m=n/2），其中：
- `out[k] = Σ_{j=0}^{n-1} x[j] · e^{-2πijk/n}`，k=0..m
- `out[m-k] = conj(out[k])`，k=1..m-1（共轭对称）

### 内部计算流程
1. **打包**: `z[k] = x[2k] + i·x[2k+1]`，k=0..m-1（m=n/2 个复数）
2. **内层 FFT**: `Z = FFT_m(z)`（m 点复数 FFT）
3. **后处理**: 对 k=0..m/2：
   - `a = Z[k]`，`b = Z[(m-k) % m]`
   - `E = (a + conj(b)) / 2`
   - `O = (a - conj(b)) / (2i)`
   - `t = W_n^k · O`（W_n^k = e^{-2πik/n}）
   - `out[k] = E + t`
   - `out[m-k] = conj(E - t)`（k=m/2 自配对只写一次）

### 输出布局
- **有效长度**: `m+1` 个复数（bin 0..m）
- **物理行距**: `hsStride(n) = n + 16` 个 float（= m+8 个复数槽）
- **对齐要求**: 行距上取整到 8 复数槽（32B 对齐）
- **越界保护**: 有效槽 [0..m]，其余 7 槽给上取整兜底，不跨行

## 2. 执行身份层契约 (Execution Identity Layer)

### 内核入口
- **名称**: `kfft_r2c_post`
- **签名**: `(out*, z*, wrr*, wri*, n, batch)` = 40B
- **调用时机**: 在 `kfft_fwd(n/2)` 之后

### 参数验证
| 参数 | 约束 | 违反行为 |
|------|------|----------|
| `n` | 2 的幂，128 ≤ n ≤ 8192 | 拒绝启动 |
| `batch` | ≥ 1 | 拒绝启动 |
| `out` | 对齐到 4B | 拒绝启动 |
| `z` | 对齐到 4B | 拒绝启动 |

### Resolved Mapping
内核必须输出实际使用的 mapping：
- `n`: 实际变换长度
- `batch`: 实际批大小
- `chunk_size`: 实际分块大小（kChunk）
- `layout`: 输出布局（hsStride）

## 3. 资源层契约 (Resource Layer)

### UB 预算
```
z:      n × 4B          (输入行)
wrr:    twLen × 4B      (旋转因子实部)
wri:    twLen × 4B      (旋转因子虚部)
表:     (3×q1 + 3×q2 + 2×kChunk) × 4B
临时:   16 × kChunk × 4B
双缓冲: 4 × kChunk × 4B
```
总计 ≈ 137KB（n=4096 最坏）

### 合法性约束
- `DataCopy` 长度必须是 8 的倍数（32B）
- `Gather` dst/idx 必须 32B 对齐
- 分块切片必须在相对下标 ≡0 (mod 8) 处

## 4. 性能层契约 (Performance Layer)

### 当前瓶颈（r2c 128×4096）
- `kfft_fwd`: 54 µs (vec 66%)
- `kfft_r2c_post`: 120 µs (vec 48%, mte2 12%, mte3 33%)
- **总时间**: 174 µs，目标 < 86.5 µs

### 优化路径
1. **separate**: 当前实现（baseline）
2. **paired-post**: k 和 m-k 同时计算，减少 Gather 次数
3. **fused-epilogue**: 融合到 kfft_fwd，消除中间 GM 边界

## 5. 证据层契约 (Evidence Layer)

### 验收标准
- **正确性**: maxRel < 1e-4（与 numpy.fft.rfft 对比）
- **性能**: 相对 incumbent 至少提升 2%
- **稳定性**: ≥5 trials，median 为准

### 失败处理
- 任何 correctness 失败 → 候选不可晋升
- 任何 trial 失败 → 整次实验无效
- 无法解释的性能回退 → 回退 incumbent
