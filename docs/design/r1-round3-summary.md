# R1 Progress Summary (Round 3)

## Overview

Implemented R1-A through R1-D per review guidance. R1-B achieves correctness; R1-C/D have issues.

## Performance Comparison (r2c 128×4096)

| Variant | Status | device_chain | maxRel | Notes |
|---------|--------|--------------|--------|-------|
| baseline (separate) | ✅ PASS | 165.5µs | 2.3e-07 | incumbent |
| static paired (R1-B) | ✅ PASS | 347.9µs | 2.3e-07 | correctness verified |
| pipelined (R1-C) | ❌ FAIL | 325.4µs | 1.9e+00 | race condition |
| fused (R1-D) | ❌ FAIL | 1370.6µs | 1.4e+00 | FFT impl WIP |

## Key Findings

### R1-B: Static Paired Writer (Correct but Slow)

**Correctness**: PASS for all batch sizes (1, 5, 47, 48, 49, 64, 1024, 4096)

**Performance**: 347.9µs vs baseline 165.5µs (2.1× slower)

**Root Cause**: Scalar loop for reverse fragment reversal
```cpp
// Current: O(m/2) scalar iterations
for (uint32_t i = 0; i < revLen / 2; i++) {
    // swap rev[2i] and rev[2j]
}
```

**Optimization Path**: Use vectorized reverse (Gather with reversed index table)

### R1-C: Pipelined Version (Race Condition)

**Issue**: Intermittent failures for batch >= 47

**Root Cause**: MTE3 writeback from iteration i races with MTE2 prefetch of iteration i+1 when sharing UB buffer resources

**Current Mitigation**: PIPE_ALL barrier serializes everything, defeating pipelining

**Next Step**: Need to understand UB buffer ownership and use local events

### R1-D: Fused Epilogue (FFT Implementation WIP)

**Issue**: simple_fft scalar implementation doesn't match optimized kfft_fwd

**Root Cause**: kfft_fwd uses radix-2/radix-4 with plane stages, batch folding, etc.

**Next Step**: Extract kfft_fwd core and reuse in fused kernel

## Profile Analysis

**Baseline Profile** (`results/profiles/pr6-round3/baseline_128x4096/`):
- kfft_fwd: 49.3µs per call (×11 calls)
- r2c_post: NOT captured in basic profile (taskType=27 shows 0µs)

**Static Profile** (`results/profiles/pr6-round3/static_128x4096/`):
- kfft_fwd: 49.2µs per call (×11 calls)
- r2c_post_static: NOT captured in basic profile

**Note**: Basic profiling only captures kfft_fwd. Need detailed `--aic-metrics` profiling to capture r2c_post breakdown.

## Files

- `src/ascendc/fft_real_static.cpp`: R1-B static paired writer
- `src/ascendc/fft_real_pipe.cpp`: R1-C pipelined (WIP)
- `src/ascendc/fft_r2c_fwd.cpp`: R1-D fused epilogue (WIP)
- `docs/design/r2c-writer-contract.md`: Corrected contract
- `docs/design/r1-tier-status.md`: R1 tier status

## Commits

- `c6c2e14`: R1-A contract + R1-B static paired writer
- `8ae6579`: R1-C pipelined (WIP)
- `ba9d37f`: R1-D fused epilogue (WIP)

## Round-4 Results (2026-10-12)

### R1-B: Static Paired Writer Ablation
| Variant | Median (µs) | Δ vs base | Correctness |
|---------|-------------|-----------|-------------|
| base | 336.2 | — | PASS |
| norev (skip reverse swap) | 279.6 | -17% | FAIL |
| nostore (vectorize interleave) | 264.8 | -21% | PASS |
| nobar (remove barriers) | 336.0 | -0.1% | PASS |
| nogather (scalar Gather) | 497.7 | +48% | PASS |
| combo (nostore+norev) | 207.8 | -38% | FAIL |

**Applied nostore optimization**: 336.2µs → 266.4µs (21% improvement)
**Conclusion**: Not promoted. Still 2.2× slower than baseline post (123µs).

### R1-C: Pipe Race Condition Fix
**Root cause**: `PipeBarrier<PIPE_MTE2>` after initial DataCopy was insufficient.
**Fix**: Restore `PipeBarrier<PIPE_ALL>` after initial z0 DataCopy.
**Result**: batch=1,5,47,48,49,64 all PASS. Median: 316.3µs.
**Conclusion**: Not promoted. 2.6× slower than baseline. PIPE_ALL overhead > pipelining benefit.

### R1-D: Fused Kernel Bottleneck
**Analysis**: `kfft_r2c_fwd` bottleneck is `simple_fft` (scalar O(n²)), not writer logic.
**Conclusion**: Not promoted. 1361µs (11× slower than baseline). Need vectorized FFT from `kfft_fwd` (out of PR scope).

### Convergence Status
All 14 targets still <1.0×. Worst: r2c 128×4096 = 0.43×.
Full test suite: 324 Python + 33 C++ PASS.
