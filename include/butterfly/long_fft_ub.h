/* Shared UB resource constants for the long-chain kernels (PR #2 stage 3).
 *
 * Included by BOTH the AscendC kernels (src/ascendc/fft_long.cpp) and the
 * host-side descriptor model (include/butterfly/descriptors.hpp) so the
 * resource formulas cannot drift away from the buffers the kernels actually
 * take.  Keep this header C-compatible: no <string>/<vector>, macros only.
 */
#pragma once

/* Transpose tile shape (kfft_lt_tr): default LT_H rows x LT_W cols per tile.
 * R2-A: these are the DEFAULTS of a candidate space, not fixed constants --
 * AB_LT_TILE=HxW selects a compiled entry; the candidate sets and every
 * constraint that trims them live in this header (single source shared by
 * the kernel static_asserts, the host entry selection and the descriptor
 * gate, so nothing can drift). */
#define AB_LT_H 128u
#define AB_LT_W 32u

/* R2-A candidate sets (review stage 2 initial list). */
#define AB_LT_H_CANDIDATES {64u, 128u, 256u}
#define AB_LT_W_CANDIDATES {16u, 32u, 64u}
#define AB_FUSE_K_CANDIDATES {128u, 256u, 512u}

/* AIV UB budget per core (192 KiB); every candidate must fit under it. */
#define AB_UB_TOTAL_BYTES 196608u

/* Per-tile bytes: tileN complex x 2 floats x 4B = 8*H*W. */
#define AB_LT_TILE_BYTES(H, W) \
  (8ull * (unsigned long long)(H) * (unsigned long long)(W))

/* kfft_lt_tr peak UB for tile HxW in the single fused-capable .o entry:
 * four tile buffers bIn+bOut+bIdx+bTw.  R2-A Round 3 (resident stripe
 * index): the three stripe index tables (4*K uint32 = 16*K bytes) move
 * out of bOut's lifetime into a dedicated read-only buffer built ONCE per
 * AIV at entry (no per-tile rebuild after the big transpose Gather wipes
 * bOut); the planar carve in bOut shrinks to 6*K floats (3K <= H*W).
 *   rebuild mode  (R=0): peak = 4*H*W*8,        carve 10K <= 2HW
 *   resident mode (R=1): peak = 4*H*W*8 + 16K,  carve  3K <= HW
 * Locked by test_limits: 4*128*32*8 = 131072 <= 192K (+4096 at K=256). */
#define AB_FUSED_UB_BYTES_HWK(H, W, K) (4ull * AB_LT_TILE_BYTES(H, W))
#define AB_RESIDENT_IDX_UB_BYTES(K) (16ull * (unsigned long long)(K))
#define AB_FUSED_UB_BYTES_HWKR(H, W, K, R)                                 \
  (AB_FUSED_UB_BYTES_HWK(H, W, K) +                                        \
   ((R) ? AB_RESIDENT_IDX_UB_BYTES(K) : 0ull))

/* Separate (no-bTw) peak for a future split entry: three tile buffers. */
#define AB_LT_SEPARATE_UB_BYTES_HW(H, W) (3ull * AB_LT_TILE_BYTES(H, W))

/* Back-compat default-shape aliases (attribution parser reads AB_LT_H/W). */
#define AB_TRANSPOSE_UB_BYTES \
  (3ull * AB_LT_TILE_BYTES(AB_LT_H, AB_LT_W))
#define AB_FUSED_UB_BYTES \
  AB_FUSED_UB_BYTES_HWK(AB_LT_H, AB_LT_W, AB_FUSE_STRIPE_K)

/* kfft_lt_tw peak UB: six buffers of (len/2)*8 bytes over half-row len/2
 * (bRow/bAr/bWp/bR/bEx/bIn) => 24 bytes per row element. */
#define AB_TWIDDLE_UB_BYTES(len) (24ull * (unsigned long long)(len))

/* Stripe length (complex elements) for the fused in-tile twiddle: the three
 * planar regions need 3*2*K floats and the three index tables 4*K uint32 in
 * bOut (2*tileN floats), i.e. 10*K <= 2*H*W must hold (locked by
 * test_limits: 10*512 <= 2*128*32). */
#define AB_FUSE_STRIPE_K 512u

/* R2-A single-source legality: candidate membership, 32B row-slice
 * alignment (W multiple of 4), Gather range (K <= H*W) and the stripe
 * carve in bOut -- rebuild mode carves idx tables there too (10K <= 2HW),
 * resident mode only the planar regions (3K <= HW).  Used by the host
 * entry selector, the descriptor gate and the kernel static_asserts. */
static inline int ab_lt_in_set(unsigned v, const unsigned* s, unsigned n) {
  unsigned i;
  for (i = 0; i < n; i++)
    if (s[i] == v) return 1;
  return 0;
}
static inline int ab_stripe_legal(unsigned h, unsigned w, unsigned k,
                                  unsigned resident) {
  static const unsigned hs[] = AB_LT_H_CANDIDATES;
  static const unsigned ws[] = AB_LT_W_CANDIDATES;
  static const unsigned ks[] = AB_FUSE_K_CANDIDATES;
  if (!ab_lt_in_set(h, hs, 3) || !ab_lt_in_set(w, ws, 3) ||
      !ab_lt_in_set(k, ks, 3))
    return 0;
  if ((w & 3u) != 0u) return 0;                                   /* 32B */
  if ((unsigned long long)k > (unsigned long long)h * w) return 0; /* Gather */
  if (resident) {
    if (3ull * k > (unsigned long long)h * w) return 0; /* planar carve */
  } else {
    if (10ull * k > 2ull * h * w) return 0;             /* planar + idx */
  }
  return 1;
}

/* kfft_fwd peak UB: exact InitBuffer sum of src/ascendc/fft_radix2.cpp as a
 * function of the stage length n, the batch fold D (arg byte 0, resolved by
 * bfly::foldDFor(len, launch_rows, 48) or the AB_FOLD_D override) and the
 * plane factor K (arg byte 1, bfly::planeKFor / AB_PLANE_K override):
 *   bPlan 8nD + bPlane 8nD + bIdxB 4n + bIdxT 4n + bIdxO 8nD      = 24nD + 8n
 *   bTwR/bTwI 2*(n+16)*4                                          = 8n + 128
 *   bTmp 4*tmpF, tmpF = 2*(n/2)*D + max(n/2, (n/K)*D) + 3*(n/K)*D
 *   total = 24nD + 8n + 8(n+16) + 4(nD + max(n/2, nD/K) + 3nD/K)
 * The host StagePlan::ubBytes, the descriptor row-fft gate and the kernel
 * accounting all derive from this one macro (tests/test_limits.cpp keeps the
 * D=1 golden 46n + 12n/K + 128).  Integer division is exact because K is a
 * power of two dividing n. */
#define AB_ROW_FFT_UB_BYTES(n, D, K)                                     \
  (24ull * (unsigned long long)(n) * (unsigned long long)(D) +           \
   8ull * (unsigned long long)(n) +                                      \
   8ull * ((unsigned long long)(n) + 16ull) +                            \
   4ull * ((unsigned long long)(n) * (unsigned long long)(D) +           \
           ((((unsigned long long)(n) >> 1) >                            \
             ((unsigned long long)(n) * (unsigned long long)(D) /        \
              (unsigned long long)(K)))                                  \
              ? ((unsigned long long)(n) >> 1)                           \
              : ((unsigned long long)(n) * (unsigned long long)(D) /     \
                 (unsigned long long)(K))) +                             \
           3ull * (unsigned long long)(n) * (unsigned long long)(D) /    \
             (unsigned long long)(K)))
