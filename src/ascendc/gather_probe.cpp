// 探针：Gather(dst, src, srcOffset, srcBaseAddr, count) 的 srcOffset/srcBaseAddr 单位。
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
using namespace AscendC;

extern "C" __global__ __aicore__ __vector__ void kgather(__gm__ float* out, __gm__ float* src, uint32_t n)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> bSrc, bOut, bIdx;
    pipe.InitBuffer(bSrc, 256 * sizeof(float));
    pipe.InitBuffer(bOut, 256 * sizeof(float));
    pipe.InitBuffer(bIdx, 64 * sizeof(uint32_t));
    LocalTensor<float>   srcL = bSrc.Get<float>();
    LocalTensor<float>   outL = bOut.Get<float>();
    LocalTensor<uint32_t> idx = bIdx.Get<uint32_t>();

    GlobalTensor<float> gs;
    gs.SetGlobalBuffer(src, n);
    DataCopy(srcL, gs, n);
    for (uint32_t i = 0; i < 64; i++) idx.SetValue(i, i * 3u + 1u);
    for (uint32_t i = 0; i < 32; i++) idx.SetValue(32 + i, (i * 3u + 1u) * 4u);

    PipeBarrier<PIPE_ALL>();
    Gather(outL, srcL, idx, 0u, 32u);          // A: 偏移原值
    Gather(outL[32], srcL, idx[32], 0u, 32u);  // B: 偏移*4
    PipeBarrier<PIPE_ALL>();
    Gather(outL[64], srcL, idx, 16u, 8u);
    PipeBarrier<PIPE_ALL>();      // C: base=16

    GlobalTensor<float> go;
    go.SetGlobalBuffer(out, 96);
    DataCopy(go, outL, 96);
}
