# R1 Tier Status

## Overview

R1 requires three-tier ablation for R2C real-boundary candidates:
1. **separate**: Current baseline (`kfft_fwd` + `kfft_r2c_post`)
2. **paired-post**: Compute k and m-k together (cancelled)
3. **fused-epilogue**: Eliminate intermediate GM boundary (not started)

## Tier 1: separate (Baseline)

**Status**: ✅ PASSING

**Implementation**: 
- `kfft_fwd(n/2)` computes inner FFT, writes Z to GM
- `kfft_r2c_post(n)` reads Z, computes E/O/t, writes half-spectrum

**Performance** (r2c 128×4096):
- kfft_fwd: ~54µs
- kfft_r2c_post: ~120µs (MTE3 33%)
- Total: ~174µs (device_chain)
- Target: <86.5µs

**Correctness**: maxRel=1.062e-07 (PASS)

## Tier 2: paired-post (Cancelled)

**Status**: ❌ CANCELLED (correctness issues)

**Implementation attempt**:
- Kernel `kfft_r2c_post_paired` in `src/ascendc/fft_real.cpp`
- Computes k and m-k in single pass
- Saves Gather operations by sharing E/O computation

**Issues**:
1. Initial version: maxRel=5.278e-01 (interleaved output instead of planar real)
2. Fixed k=0 boundary (Z[m] out of bounds → Z[0]): maxRel=1.348e-02
3. Remaining issue: out[m-k] individual DataCopy writes have race conditions or incorrect layout

**Root cause**: The out[m-k] writes use scattered positions (m-k for k in chunk), requiring:
- Gather to interleave oR/oI into complex layout
- Individual DataCopy calls for each non-contiguous position
- Race condition risk between double-buffer swap and individual writes

**Decision**: Reverted to keep baseline working. Fused-epilogue (Tier 3) is more promising for the required 2× speedup.

## Tier 3: fused-epilogue (Not Started)

**Status**: 🔴 NOT STARTED

**Goal**: Eliminate intermediate GM boundary between kfft_fwd and kfft_r2c_post.

**Requirements**:
1. Modify kfft_fwd to optionally do r2c post-processing in final writer
2. Pass r2c twiddles (wrr, wri) to kfft_fwd as additional parameters
3. Compute E/O/t on planar data before interleaving
4. Write half-spectrum directly to GM

**Challenges**:
1. kfft_fwd uses 500+ lines with batch folding, plane stages
2. Final writer currently does planar→interleaved gather
3. Need to add r2c computation without exceeding UB capacity
4. Must maintain bit-exact equivalence with separate approach

**Prerequisites**:
- R2C writer contract (✅ `docs/design/r2c-writer-contract.md`)
- Baseline correctness tests (✅ 33/33 C++ tests)
- Profiling data (✅ `results/profiles/pr6-round2/`)

**Estimated effort**: High (3-5 days)

## Recommendation

1. **Immediate**: Commit baseline (separate) with documentation
2. **Short-term**: Debug paired kernel out[m-k] writes (1-2 days)
3. **Medium-term**: Implement fused-epilogue (3-5 days)

Given the review's requirement of 2× speedup (210µs → <86.5µs), **fused-epilogue is mandatory**. Even halving post time (120µs → 60µs) gives 54+60=114µs > 86.5µs target.

## Evidence

- **Baseline**: `results/evidence/pr6-round2/fold_analysis.md`
- **Profiles**: `results/profiles/pr6-round2/`
- **Contract**: `docs/design/r2c-writer-contract.md`
- **Tests**: 33/33 C++ + 325 Python (1 skip)
