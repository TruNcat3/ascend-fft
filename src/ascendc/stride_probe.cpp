// 探针：Level-0 矢量算子的 mask / repeatTime / *RepStride 语义（fp32）。
// 语义假设（待证）：repeat r 的第 j 个元素位于
//     dst[dstRep*8*r + j] = s[src0Rep*8*r + j] * s[src1Rep*8*r + j]
// 即 repStride 单位 = c0Count = 8 个 float（32B）。未被写的下标保持初值 == 下标。
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ __vector__ void kstride(
    __gm__ float* out, __gm__ float* in, uint32_t n, uint32_t mask, uint32_t repeat,
    uint32_t dstRep, uint32_t src0Rep, uint32_t src1Rep)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> bS, bD;
    pipe.InitBuffer(bS, 16384 * sizeof(float));
    pipe.InitBuffer(bD, 16384 * sizeof(float));
    LocalTensor<float> s = bS.Get<float>(), d = bD.Get<float>();
    GlobalTensor<float> gi, go;
    gi.SetGlobalBuffer(in, 16384);
    DataCopy(s, gi, 16384);
    DataCopy(d, gi, 16384);
    PipeBarrier<PIPE_ALL>();

    BinaryRepeatParams p(1, 1, 1, (uint8_t)dstRep, (uint8_t)src0Rep, (uint8_t)src1Rep);
    Mul(d, s, s, mask, (uint8_t)repeat, p);

    PipeBarrier<PIPE_ALL>();
    go.SetGlobalBuffer(out, 16384);
    DataCopy(go, d, 16384);
}
