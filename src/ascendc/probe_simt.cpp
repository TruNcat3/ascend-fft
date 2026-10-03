// Phase 0.3 part B: SIMT model probe. Compiled with `ccec ... --enable-simt`.
// If SIMT is unavailable on this SoC the build fails here, which is itself the answer.
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

extern "C" __global__ __aicore__ __vector__ void kprobe_simt(__gm__ float* out) {
    int32_t nthread = Simt::GetThreadNum<0>();
    int32_t ithread = Simt::GetThreadIdx<0>();
    int32_t warpSz  = Simt::GetWarpSize();
    int64_t blk     = GetBlockIdx();
    int64_t nblk    = GetBlockNum();

    out[0] = (float)nblk;
    out[1] = (float)blk;
    out[2] = (float)nthread;
    out[3] = (float)ithread;
    out[4] = (float)warpSz;
    // WarpShflXorSync is the primitive cuButterfly's `local_exchange=shuffle` needs.
    float v = 1.0f;
    v = Simt::WarpShflXorSync<float>(v, 1);
    out[5] = v;
    out[7] = 42.0f;
}
