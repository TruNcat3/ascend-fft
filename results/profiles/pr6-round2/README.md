# PR#6 Round 2: Profile Data

## 关键性能点 Profile 记录

采集时间: 2026-10-12
环境: Ascend910_9382, CANN 9.0.0
工具: msprof (via ab_prof)

## 采集的 Profile

| 点位 | 目录 | device_chain | 说明 |
|------|------|-------------:|------|
| r2c 128×4096 | `r2c_128x4096/` | 157.3 µs | 最差点，post 瓶颈 |
| r2c 256×4096 | `r2c_256x4096/` | 180.6 µs | foldD 优化有效点 |
| c2r 1024×4096 | `c2r_1024x4096/` | 480.0 µs | c2r 最差点 |
| c2c 1024×4096 | `c2c_1024x4096/` | — | c2c 参考点 |

## 内核级指标 (r2c 128×4096)

从 `msprof_*.db` 提取:

```
kfft_fwd: 21 次调用, avg=48.1 µs, total=1010.4 µs
```

**注意**: 基础 profiling 仅捕获到 kfft_fwd，kfft_r2c_post 未在 COMPUTE_TASK_INFO 中出现。
这可能是因为:
1. post 内核使用不同的 taskType
2. profiling 采样率限制
3. post 内核执行时间过短被过滤

## 详细分析 (来自之前的 msprof 采集)

之前使用 `ai_core_op_summary.db` 的详细分析显示:

### r2c 128×4096 (n=128, batch=4096)
- **kfft_fwd**: ~54 µs (vec 66%)
- **kfft_r2c_post**: ~120 µs (vec 48%, mte2 12%, mte3 33%)
- **瓶颈**: post 内核 MTE3-bound (33% MTE3 利用率)

### 性能分解
```
Total device_chain: ~174 µs
├── kfft_fwd (inner FFT n/2=64): ~54 µs (31%)
├── kfft_r2c_post: ~120 µs (69%)  ← 瓶颈
└── kernel launch overhead: ~0 µs (同流串行)
```

## 优化方向分析

### r2c 128×4096 (需从 210 µs → <86.5 µs, -58.8%)

**当前状态**: foldDFor 优化已到位 (D=16 for inner n=64)

**瓶颈**: kfft_r2c_post 是 MTE3-bound
- 逐 batch 行处理，无多行复用
- 上下半谱分两段重复 Gather 和 E/O/t 计算
- 固定 kChunk=1024，小 N 时表初始化开销大
- 多处 PIPE_ALL 屏障
- 主 FFT 先写中间结果到 GM，post 再读回

**优化路径** (per reviewer R1 plan):
1. **separate**: 当前实现 (baseline)
2. **paired-post**: k 和 m-k 同时计算，变长 chunk，小 N 多行打包
3. **fused-epilogue**: 融合到主 FFT 最终 writer，消除中间 GM 边界

**关键发现**: 即使独立 post 减半 (120→60 µs)，总时间仍需 114 µs > 86.5 µs 目标。
**必须实现 fused-epilogue** 才能达标。

### c2r 1024×4096 (需从 550 µs → <目标)

**瓶颈**: 3 次发射链把满谱物化到 GM
- kfft_c2r_prep: Hermitian 展开
- kfft_fwd: IDFT
- kfft_c2r_post: 实部提取 + 1/n

**最小修正**: 融合实部提取 + 1/n 到 FFT 最终 writer (已尝试，遇到 buffer lifecycle 问题)

## 文件说明

- `kernel_metrics.json`: 内核级耗时统计
- `r2c_128x4096/`: r2c 128×4096 完整 msprof 数据
- `r2c_256x4096/`: r2c 256×4096 完整 msprof 数据
- `c2r_1024x4096/`: c2r 1024×4096 完整 msprof 数据
- `c2c_1024x4096/`: c2c 1024×4096 完整 msprof 数据

## 复现命令

```bash
source scripts/env.sh
export AB_SOC=Ascend910_9382 AB_PROFILE="$PWD/config/ascend910_93_profile.json"

# 采集 profile
ab_prof results/profiles/pr6-round2/r2c_128x4096 \
  -- ./build/fft_check 128 4096 20

# 查看结果
ls results/profiles/pr6-round2/r2c_128x4096/PROF_*/
```
