/* Shared UB resource constants for the long-chain kernels (PR #2 stage 3).
 *
 * Included by BOTH the AscendC kernels (src/ascendc/fft_long.cpp) and the
 * host-side descriptor model (include/butterfly/descriptors.hpp) so the
 * resource formulas cannot drift away from the buffers the kernels actually
 * take.  Keep this header C-compatible: no <string>/<vector>, macros only.
 */
#pragma once

/* Transpose tile shape (kfft_lt_tr): LT_H rows x LT_W cols per tile. */
#define AB_LT_H 128u
#define AB_LT_W 32u

/* kfft_lt_tr peak UB: bIn + bOut + bIdx, each LT_H*LT_W complex-sized
 * (tileN*2 floats or tileN*2 uint32) = 3 * LT_H * LT_W * 8 bytes. */
#define AB_TRANSPOSE_UB_BYTES (3ull * (unsigned long long)AB_LT_H * \
                               (unsigned long long)AB_LT_W * 8ull)

/* kfft_lt_tw peak UB: six buffers of (len/2)*8 bytes over half-row len/2
 * (bRow/bAr/bWp/bR/bEx/bIn) => 24 bytes per row element. */
#define AB_TWIDDLE_UB_BYTES(len) (24ull * (unsigned long long)(len))

/* Fused boundary chain (PR-B / R1): kfft_lt_tr gains the tw tile buffer bTw
 * (tileN*2 floats, loaded with the data tile's DataCopyParams) when it
 * consumes `tw`.  The stripe planarize/multiply machinery (planar data,
 * planar tw, product, three index tables) is carved out of bOut's lifetime,
 * so the peak is exactly four tile buffers:
 *   AB_FUSED_UB_BYTES = AB_TRANSPOSE_UB_BYTES + LT_H*LT_W*8 = 128 KiB <= 192.
 * Single .o entry: the compiler's static layout includes bTw on every path,
 * so the descriptor gates EVERY kfft_lt_tr launch at this peak (separate
 * chains pass tw=nullptr and skip the twiddle phase at runtime). */
#define AB_FUSED_UB_BYTES                                                 \
  (AB_TRANSPOSE_UB_BYTES +                                                \
   (unsigned long long)AB_LT_H * (unsigned long long)AB_LT_W * 8ull)

/* Stripe length (complex elements) for the fused in-tile twiddle: the three
 * planar regions need 3*2*K floats and the three index tables 4*K uint32 in
 * bOut (2*tileN floats), i.e. 10*K <= 2*LT_H*LT_W must hold (locked by
 * test_limits: 10*512*4 = 20480 <= 32768). */
#define AB_FUSE_STRIPE_K 512u

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
