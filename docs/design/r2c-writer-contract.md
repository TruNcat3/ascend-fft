# R2C Writer Contract

## 1. 语义层契约 (Semantic Layer)

### 数学定义
给定实数输入 `x[0..n-1]`，R2C 输出半谱 `X[0..m]`（m=n/2）：
- **打包**: `z[k] = x[2k] + i·x[2k+1]`，k=0..m-1（m=n/2 个复数）
- **内层 FFT**: `Z = FFT_m(z)`（m 点复数 FFT）
- **后处理**: 对 k=0..m/2：
  - `a = Z[k]`，`b = Z[(m-k) % m]`
  - `E = (a + conj(b)) / 2`
  - `O = (a - conj(b)) / (2i)` = `((ai+bi)/2, (br-ar)/2)`
  - `t = W_n^k · O`（W_n^k = e^{-2πik/n}）
  - `X[k] = E + t`
  - `X[m-k] = conj(E - t)`（k=m/2 自配对只写一次；k=0 时 X[0] 实数）

### 重要：共轭关系修正

**X[m-k] 不是 X[k] 的共轭**。

完整实 FFT 的 Hermitian 关系是 `X[N-k] = conj(X[k])`，但 packed R2C post 中：
- `k` 与 `m-k` 是**共同消费 Z[k]/Z[m-k] 的计算 pair**
- 生成的是**两个独立半谱 bin**
- 每个 bin 需要独立的 E/O/t 蝶形计算

示例（n=128, m=64）：
- X[0] = DC（实数）
- X[1] 与 X[63]：独立计算，不共轭
- X[32]：midpoint（自配对）
- X[64] = Nyquist（实数）

### 输出布局
- **有效长度**: `m+1` 个复数（bin 0..m）
- **物理行距**: `hsStride(n) = n + 16` 个 float（= m+8 个复数槽）
- **对齐要求**: 行距上取整到 8 复数槽（32B 对齐）
- **越界保护**: 有效槽 [0..m]，其余 7 槽给上取整兜底，不跨行

## 2. 执行身份层契约 (Execution Identity Layer)

### Pair Ownership（核心所有权规则）

```text
owner(k) 唯一消费 Z[k] 和 Z[m-k]
```

- 每个 pair `(k, m-k)` 由**唯一 owner** 负责计算
- owner 产出 X[k]（forward）和 X[m-k]（reverse）两个独立 bin
- k=0 独占 DC；k=m/2 独占 midpoint（自配对只写一次）

### Endpoint Ownership

| 端点 | 位置 | 处理方式 |
|------|------|----------|
| DC | X[0] | k=0 owner 独占，实数输出 |
| Nyquist | X[m] | k=0 owner 独占，实数输出 |
| Midpoint | X[m/2] | k=m/2 owner 独占，自配对只写一次 |

### Burst Ownership

```text
一个 32B GM block 只能由一个 writer/core 发布
```

- 禁止多个 core 写同一 32B 区域
- 跨 core 的 32B block 边界必须对齐

### Buffer Lifetime

```text
两个输出 fragment 的 MTE3 完成后才允许复用 ping/pong UB
```

- compute-ready event：生产者标记数据就绪
- store-done event：消费者标记写入完成
- 只有 store-done 后才能复用 buffer

### 内核入口
- **名称**: `kfft_r2c_post`（separate）或 `kfft_r2c_fwd`（fused）
- **签名**: `(out*, z*, wrr*, wri*, n, batch)` = 40B
- **调用时机**: separate 在 `kfft_fwd(n/2)` 之后；fused 内含 FFT

### Resolved Mapping
内核必须输出实际使用的 mapping：
- `n`: 实际变换长度
- `batch`: 实际批大小
- `chunk_size`: 实际分块大小（kChunk）
- `layout`: 输出布局（hsStride）
- `pair_count`: pair 数量（m/2 + 1）
- `buffer_mode`: single/double

## 3. 资源层契约 (Resource Layer)

### UB 预算（n=128, m=64 为例）

| 缓冲区 | 大小 | 用途 |
|--------|------|------|
| Z 行缓存 | n×4B = 512B | 加载完整 Z 行 |
| Forward fragment | (m/2+1)×8B = 264B | X[k], k=0..m/2 |
| Reverse fragment | (m/2)×8B = 256B | X[m-k], k=1..m/2-1 |
| Twiddles | (m/2+1)×8B = 264B | W_n^k |
| 索引表 | (m/2+1)×4B × 3 | tA, tB, tW |
| **总计** | ~1.6KB | 远小于 192KB UB |

### 双缓冲模式

```text
ping: [Z 行 | forward | reverse | twiddles]
pong: [Z 行 | forward | reverse | twiddles]
```

- producer: 写 ping.Z，标记 compute-ready
- compute: 读 ping.Z，写 ping.forward/reverse，标记 store-done
- writer: 读 ping.forward/reverse，写 GM，标记 store-done
- 下一轮：等待 ping.store-done 后复用

## 4. 性能层契约 (Performance Layer)

### Static Paired Writer 结构

```text
FFT final state / post pair
  -> 唯一 pair owner 完成 E/O/t
  -> writer-aligned UB static layout
  -> UB 内交换、反转或转置
  -> 一个或两个连续 MTE3 fragment
  -> dense/padded half-spectrum contract
```

### 关键优化点

1. **消除 scattered writes**：reverse fragment 在 UB 内转为正向连续，用块 DataCopy
2. **128-bit 连续 I/O**：对齐到 float4（16B）边界
3. **独立 writer role**：producer 不直接写 GM，由 writer 线程负责
4. **静态布局**：XOR/transpose 在片上完成，输出自然序

### 性能目标

| Shape | Incumbent | 目标 | 提升 |
|-------|-----------|------|------|
| r2c 128×4096 | 210.1µs | <86.5µs | ≥2.4× |
| r2c 256×4096 | - | - | ≥2% |

## 5. 证据层契约 (Evidence Layer)

### 验证要求

每次测试必须检查：
1. DC、Nyquist、midpoint 和最后一个有效 bin
2. N+2 有效区、N+16 padding guard、相邻 batch 不串写
3. requested/resolved entry、D/K/core、layout、buffer 数和 binary hash
4. separate trace 必须看到 `kfft_fwd + kfft_r2c_post`
5. fused trace 必须证明中间边界和 post entry 确实消失

### 测试输入

第一批只测 N=128：
```text
batch = 1, 5, 47, 48, 49, 64, 1024, 4096
输入 = zero、impulse、constant、single-tone、Nyquist、random、high-dynamic-range
```

### 停止条件

1. static paired writer 先达到全正确
2. fused 方案必须由 trace 证明 launch 和物化 GM 边界减少
3. 相对 exact incumbent median 稳健提升至少 2%
4. 冻结实验点全部获得正加速、完整矩阵无回退后即收敛

---

## 附录 A：CPU Reference 实现

```python
import numpy as np

def r2c_post_reference(Z, n):
    """R2C 后处理 CPU reference
    
    Args:
        Z: m 点复数 FFT 输出 (m = n//2)
        n: 原始实数序列长度
    
    Returns:
        X: 半谱 (m+1 个复数)
    """
    m = n // 2
    X = np.zeros(m + 1, dtype=np.complex128)
    
    # 旋转因子表
    W = np.exp(-2j * np.pi * np.arange(m + 1) / n)
    
    for k in range(m // 2 + 1):
        a = Z[k]
        b = Z[(m - k) % m]
        
        # E = (a + conj(b)) / 2
        E = (a + np.conj(b)) / 2
        
        # O = (a - conj(b)) / (2i)
        O = (a - np.conj(b)) / (2j)
        
        # t = W * O
        t = W[k] * O
        
        # X[k] = E + t
        X[k] = E + t
        
        # X[m-k] = conj(E - t)，k>0 且 k<m/2
        if k > 0 and k < m // 2:
            X[m - k] = np.conj(E - t)
    
    # k=0: X[0] = E + t（实数）
    # k=m/2: X[m/2] = E + t（自配对，已写入）
    return X


def r2c_full_reference(x):
    """完整 R2C reference（打包 + FFT + 后处理）"""
    n = len(x)
    m = n // 2
    
    # 打包：z[k] = x[2k] + i*x[2k+1]
    z = x[0::2] + 1j * x[1::2]
    
    # m 点 FFT
    Z = np.fft.fft(z)
    
    # 后处理
    X = r2c_post_reference(Z, n)
    
    return X


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    for n in [128, 256, 512]:
        x = np.random.randn(n)
        X_ref = r2c_full_reference(x)
        X_np = np.fft.rfft(x)
        assert np.allclose(X_ref, X_np, atol=1e-10), f"n={n} failed"
        print(f"n={n}: PASS")
```
