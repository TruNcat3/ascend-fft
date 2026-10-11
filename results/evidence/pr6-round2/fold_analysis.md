# PR#6 Round 2: foldDFor Analysis & R0 Fixes

## R0 Blockers Fixed (commit 719850f)

### 1. test_limits.cpp golden updated
- (64,128): D=2 → D=1 (ceil(128/D)>=96 concurrency gate with nblk=48)
- (64,524288): D=4 → D=16 (n<=64 cap=16)
- K=32 override for len=64: explicit REJECT (legal=false), not silent clamp
- C++ gate: 33/33 pass via `scripts/build.sh limits`

### 2. bench_convergence.py rewritten with candidate closure
- incumbent always in candidate set
- `--candidate KEY=VAL` repeatable for env overrides
- C2C uses matrix_test.py min-of-means protocol
- R2C/C2R uses bench_r2c_c2r.py warmup+timed protocol
- AB_DIR set explicitly per invocation
- any missing measurement → exit 1
- records candidate_id, mapping(D,K), processing_unit, binary_sha256, selection_reason

## foldDFor Behavior (n-aware cap + 2*nblk concurrency)

| inner n | kFoldCap | 128x4096 D | 128x1024 D | 256x4096 D | 1024x4096 D |
|---------|----------|------------|------------|------------|-------------|
| 64      | 16       | 16         | 10         | —          | —           |
| 128     | 8        | —          | 8          | 8          | —           |
| 256     | 8        | —          | —          | 8          | —           |
| 512     | 4        | —          | —          | —          | 4           |

Concurrency gate: ceil(batch/D) >= 2*nblk (nblk=48 → minGroups=96)

## Performance Summary (median of 5 trials)

| point | baseline speedup | with fold cap | improvement |
|-------|-----------------:|--------------:|------------:|
| r2c 128x64 | 1.168 | 1.059-1.165 | ≥1.0x ✓ |
| r2c 256x4096 | 0.461 | 0.545-0.595 | +18-29% |
| r2c 512x4096 | 0.620 | 0.619-0.668 | +0-8% |
| c2r 1024x4096 | 0.835 | 0.835-0.907 | +0-5% |
| r2c 128x4096 | 0.428 | 0.401-0.428 | ≈ (post bottleneck) |

## Remaining Gap Analysis (r2c 128x4096)

Profile (msprof ai_core_op_summary.db):
- kfft_fwd: 54 µs (vec 66%)
- kfft_r2c_post: 120 µs (vec 48%, mte2 12%, mte3 33%)

Post kernel is MTE3-bound. Even halving post (→60 µs) + fwd (54 µs) = 114 µs,
still above the 86.5 µs target. Full GM-boundary elimination (fused-epilogue)
is required per reviewer's R1 plan.

## R1 Implementation Plan (not yet started)

1. `kfft_r2c_fwd`: fuse inner FFT final writer + r2c E/O/t into one launch
   - eliminates intermediate GM write/read of Z
   - eliminates separate post kernel launch
   - requires passing r2c twiddles (wrr, wri) to FFT kernel
2. Fallback: keep current separate path as `kfft_r2c_post` (incumbent)
3. Candidate closure: incumbent + fused, same protocol, >=2% promotion gate

## R1 Implementation Attempt (c2r fusion)

Attempted to implement c2r fused-epilogue by adding a mode flag (bit 17 of
packArg) to kfft_fwd that would do real extraction + 1/n normalization in
the final writer, eliminating the kfft_c2r_post launch.

**Issue encountered**: The fused kernel output matched the interleaved
complex output rather than the planar real parts. Despite:
- Adding PipeBarrier before reading plan
- Using plan/pr correctly (pr = plan alias)
- Matching the c2r_post kernel's int32->float conversion pattern

The output was still interleaved, suggesting the planar stages may not
leave data in the expected format in plan, or there's a subtle buffer
aliasing issue.

**Resolution**: Reverted kernel changes. The c2r fusion requires deeper
investigation of the kfft_fwd buffer lifecycle. This is a P2 optimization
and should be done as a separate task with dedicated debugging.

## Summary

- R0 blockers (test_limits golden + bench_convergence candidate closure): **FIXED**
- R1 r2c fused-epilogue: **NOT STARTED** (requires kernel fusion design)
- R1 c2r fused-epilogue: **ATTEMPTED, REVERTED** (buffer lifecycle issue)
- Remaining 13/14 points <1.0x: requires kernel-level changes per review plan

## Files Changed

- `tests/test_limits.cpp`: golden updated for n-aware foldDFor (33/33 pass)
- `scripts/bench_convergence.py`: full candidate-closure runner
- `tests/test_bench_convergence.py`: 6 tests for runner
- `results/evidence/pr6-round2/`: evidence + analysis
