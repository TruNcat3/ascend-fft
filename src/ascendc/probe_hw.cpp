// Phase 0.3: does the SIMT programming model work on Ascend910_93?
// Two probes in one file:
//   A) plain AscendC kernel (control)   -> kprobe_hw
//   B) SIMT-assisted kernel             -> kprobe_simt   (compiled only with --enable-simt)
#include "kernel_operator.h"

using namespace AscendC;

// ---- A) hardware probe: report block count, index, and whether sub-blocks exist.
extern "C" __global__ __aicore__ __vector__ void kprobe_hw(__gm__ float* out) {
    int64_t blk = GetBlockIdx();
    int64_t nblk = GetBlockNum();
    if (nblk <= 0) nblk = 1;
    int64_t sub = GetSubBlockIdx();
    int64_t nsub = GetSubBlockNum();
    __gm__ float* o = out + blk * 16;
    o[0] = (float)nblk;
    o[1] = (float)blk;
    o[2] = (float)sub;
    o[3] = (float)nsub;
    o[4] = (float)AscendC::GetSubBlockNum();
    o[5] = 0.0f;
    o[6] = 0.0f;
    o[7] = 42.0f;
}
