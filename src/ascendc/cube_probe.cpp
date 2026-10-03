// Cube 探针：fp32 Mmad(LoadData GM->L0 + Mmad + Fixpipe L0C->GM)。
// 单 tile 形状 16x16x16（fp32 fractal = 16x8 = 512B），按 nTile 个矩阵顺序执行。
// mode: 0 = 只 LoadData；1 = +Mmad；2 = +Fixpipe（全链路）。
// 已验证参数: loadRep=2 srcStride=1 dstStride=16, A=[k/8][16m x 8k], B=[k/8][16n x 8k]。
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ __cube__ void kcube(
    __gm__ float* out, __gm__ float* aGm, __gm__ float* bGm,
    uint32_t mode, uint32_t loadRep, uint32_t srcStride, uint32_t dstStride, uint32_t nTile)
{
    TBuffAddr ta, tb, tc;
    LocalTensor<float> l0a, l0b, l0c;
    ta.logicPos = static_cast<uint8_t>(TPosition::A2);
    l0a.SetAddr(ta);
    l0a.InitBuffer(0, 4096);
    tb.logicPos = static_cast<uint8_t>(TPosition::B2);
    l0b.SetAddr(tb);
    l0b.InitBuffer(0, 4096);
    tc.logicPos = static_cast<uint8_t>(TPosition::CO1);
    l0c.SetAddr(tc);
    l0c.InitBuffer(0, 4096);

    GlobalTensor<float> ga, gb, go;
    ga.SetGlobalBuffer(aGm, 256);
    gb.SetGlobalBuffer(bGm, 256);
    go.SetGlobalBuffer(out, 256);

    LoadData2DParams lp(0, (uint8_t)loadRep, (uint16_t)srcStride, 0, 0, false, 0);
    MmadParams mp(16, 16, 16, 0, false, true);
    FixpipeParamsV220 fp((uint16_t)16, (uint16_t)16, (uint16_t)16, (uint32_t)dstStride, false);

    if (mode == 3) {
        LoadData(l0a, ga, lp);
        LoadData(l0b, gb, lp);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t t = 0; t < nTile; ++t) {
            Mmad(l0c, l0a, l0b, mp);
            PipeBarrier<PIPE_M>();
        }
        return;
    }

    for (uint32_t t = 0; t < nTile; ++t) {
        LoadData(l0a, ga[t * 256], lp);
        LoadData(l0b, gb[t * 256], lp);
        PipeBarrier<PIPE_ALL>();
        if (mode == 0) {
            continue;
        }
        Mmad(l0c, l0a, l0b, mp);
        PipeBarrier<PIPE_ALL>();
        if (mode == 1) {
            continue;
        }
        Fixpipe(go[t * 256], l0c, fp);
        PipeBarrier<PIPE_ALL>();
    }
}
