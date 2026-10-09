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

/* kfft_fwd peak UB: 46.5n + 128 bytes, the accounting documented at the top
 * of src/ascendc/fft_radix2.cpp (plan 8n + plane 8n + idxB 4n + idxT 4n +
 * idxOut 8n + tw (4n+64)*2 + tmp 6.5n). */
#define AB_ROW_FFT_UB_BYTES(n) ((93ull * (unsigned long long)(n)) / 2ull + \
                                128ull)
