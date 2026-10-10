// 长链 device-materialized 边界内核（addendum §3 step 3，PR-B 融合）：
//   kfft_lt_tr   分块矩阵转置：src[b][nRows][nCols] -> dst[b][nCols][nRows]，
//                即 dst[c][i] = src[i][c]（转置入 / 段间转置 / 自然序写出共用）。
//                tw != nullptr（AB_LONG_BOUNDARY_IMPL=fused 的段边界发射）时先在
//                读入 tile 内做 twiddle 复乘 dst[c][i] = src[i][c]*wT[i][c]，把
//                kfft_lt_tw 的整张量 GM 往返折进转置读，发射数 6 -> 5。
//   kfft_lt_tw   段边界点乘：dIn[j][k1] *= W_N^{j·k1}（行连续，原地，separate 用）
// 边界形态：host 路径把 twiddle+转置合在宿主内存里做；device 路径默认拆成
// 点乘（全连续 GM 读写）+ 转置两步（separate），fused 模式合成一次转置发射；
// 段边界数据不回宿主（E2E boundary=0），wT 表 plan 期上传（输入无关、批共享、
// 与 src 同形、同 DataCopyParams）。
//
// GM 非连续访问用 DataCopyParams 分块搬运（CANN 9.0.0 单位：blockLen/srcStride
// 均为 32B datablock，gap 为「上一块尾->下一块头」，blockCount<=4095）：
//   每块 = src 行内列切片 [c0, c0+wCnt)（8*wCnt 字节），块间隔 = 行距 - 块长。
// 本 SoC 的 Gather 仅支持 UB 源（GlobalTensor 源编译报错，首版实测），
// 因此先整块搬进 UB，再用 UB 源 Gather 做重排（索引只依赖形状，启动时建一次）。
// 约束：nRows/nCols 为 >=64 的 2 的幂（阶段因子域），W=32 恒 4 对齐，
// blockLen=wCnt/4、srcGap=(nCols-wCnt)/4 均为整 datablock。
// 两内核同签名 (dst, src, tw, nRows, nCols, batch) = 36B（fft_real 同款约定，
// readArgSize 对整个 .o 只给一个值）。
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
// 分块常量与 UB 资源公式同源（PR #2 阶段 3 / PR-B）：描述符层 query_lowering
// 用 include/butterfly/long_fft_ub.h 的 AB_TRANSPOSE_UB_BYTES /
// AB_FUSED_UB_BYTES 校验本内核峰值，两处共享同一份宏，防止漂移。
#include "butterfly/long_fft_ub.h"
using namespace AscendC;

// ---- 分块转置（+ 可融合 twiddle）------------------------------------------
// tile 读入 [H][W]（行=i 列=c，交错复数），tw!=nullptr 时先做带内 twiddle
// 复乘（AB_FUSE_STRIPE_K 复数一条带），Gather 重排为 dst 行序写回。
// R2-A：内核体在 src/ascendc/fft_long_lt_tr.inc，按候选 (H,W) 逐个展开；
// TPipe 必须留在 __global__ 入口（ccec 9.0.0 约束，见 .inc 头注释）。
#define AB_KLTT_CAT2(a, b) a##b
#define AB_KLTT_CAT(a, b) AB_KLTT_CAT2(a, b)

// 默认 128x32（PR-B 行为不变）+ K=512 下裁剪后的融合合法候选。
// 裁剪：10K=5120 <= 2HW => HW>=2560；4*HW*8 <= 192 KiB => HW<=6144。
// (64,32)/(128,16) 在 K=512 非法（stripe K 轮解锁）；(128,64)/(256,32/
// 256,64) UB 超预算，不导出。
#define AB_LT_TR_SUFFIX
#define AB_LT_TR_H 128
#define AB_LT_TR_W 32
#define AB_LT_TR_K AB_FUSE_STRIPE_K
#include "fft_long_lt_tr.inc"
#undef AB_LT_TR_SUFFIX
#undef AB_LT_TR_H
#undef AB_LT_TR_W
#undef AB_LT_TR_K

#define AB_LT_TR_SUFFIX _t64x64
#define AB_LT_TR_H 64
#define AB_LT_TR_W 64
#define AB_LT_TR_K AB_FUSE_STRIPE_K
#include "fft_long_lt_tr.inc"
#undef AB_LT_TR_SUFFIX
#undef AB_LT_TR_H
#undef AB_LT_TR_W
#undef AB_LT_TR_K

#define AB_LT_TR_SUFFIX _t256x16
#define AB_LT_TR_H 256
#define AB_LT_TR_W 16
#define AB_LT_TR_K AB_FUSE_STRIPE_K
#include "fft_long_lt_tr.inc"
#undef AB_LT_TR_SUFFIX
#undef AB_LT_TR_H
#undef AB_LT_TR_W
#undef AB_LT_TR_K

// ---- 段边界点乘（原地、行连续） ------------------------------------------
// 行 r 属批 b 的第 j 行：dIn[(b*n2+j)][k1] *= wT[j][k1]（wT 与 j 对齐、批共享）。
// 每行拆两个半行块（n/2 复数）以压 UB 占用；半行内：交错读 -> 平面化 Gather
// (data/tw 各一) -> 平面复数乘 -> 交错 Gather -> 连续写回（与源同址，原地）。
extern "C" __global__ __aicore__ __vector__ void kfft_lt_tw(
    __gm__ float* dst, __gm__ float* src, __gm__ float* tw,
    uint32_t nRows, uint32_t nCols, uint32_t batch)
{
    TPipe pipe;
    const uint32_t m = nRows / 2u;                         // 半行长（复数）
    TBuf<TPosition::VECCALC> bRow, bAr, bWp, bR, bEx, bIn;
    pipe.InitBuffer(bRow, 2u * m * sizeof(float));         // 数据交错 / 结果交错
    pipe.InitBuffer(bAr,  2u * m * sizeof(float));         // 平面 [ar | ai]
    pipe.InitBuffer(bWp,  2u * m * sizeof(float));         // 平面 [wr | wi]
    pipe.InitBuffer(bR,  2u * m * sizeof(float));          // tw 交错 / 平面 [re | im]
    pipe.InitBuffer(bEx,  2u * m * sizeof(uint32_t));      // 交错 -> 平面
    pipe.InitBuffer(bIn,  2u * m * sizeof(uint32_t));      // 平面 -> 交错
    LocalTensor<float> row = bRow.Get<float>();
    LocalTensor<float> ar  = bAr.Get<float>();
    LocalTensor<float> wp  = bWp.Get<float>();
    LocalTensor<float> r   = bR.Get<float>();
    LocalTensor<uint32_t> ex = bEx.Get<uint32_t>();
    LocalTensor<uint32_t> it = bIn.Get<uint32_t>();
    for (uint32_t k = 0; k < m; k++) {
        ex.SetValue(k,      k * 8u);                       // -> planar re
        ex.SetValue(m + k,  k * 8u + 4u);                  // -> planar im
        it.SetValue(2u * k,      4u * k);                  // planar re -> 交错
        it.SetValue(2u * k + 1u, 4u * m + 4u * k);         // planar im -> 交错
    }
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk  = GetBlockIdx();
    const uint64_t chunks = (uint64_t)batch * nCols * 2u;  // 每行 2 半行块
    for (uint64_t cc = (uint64_t)blk; cc < chunks; cc += (uint64_t)nblk) {
        const uint64_t rr = cc >> 1;                       // 行号
        const uint32_t h  = (uint32_t)(cc & 1u);           // 半行号
        const uint32_t j  = (uint32_t)(rr % nCols);        // wT 行号（批共享）
        const uint64_t off = rr * nRows * 2u + (uint64_t)h * m * 2u;
        GlobalTensor<float> g, gt, gd;
        g.SetGlobalBuffer(src + off, 2u * m);
        gt.SetGlobalBuffer(tw + (uint64_t)j * nRows * 2u + (uint64_t)h * m * 2u, 2u * m);
        gd.SetGlobalBuffer(dst + off, 2u * m);
        DataCopy(row, g,  2u * m);                         // 数据（交错）
        DataCopy(r,   gt, 2u * m);                         // tw（交错，暂存 r）
        PipeBarrier<PIPE_ALL>();
        Gather(ar, row, ex, 0u, 2u * m);                   // ar <- [ar | ai]
        Gather(wp, r,   ex, 0u, 2u * m);                   // wp <- [wr | wi]
        // 复数乘（平面）：im 先算（wr/wi 为正），再取负 wi 算 re，最后恢复 wi。
        //   im = ar*wi + ai*wr    re = ar*wr + ai*(-wi)
        Mul(r[m], ar, wp[m], m);
        MulAddDst(r[m], ar[m], wp, m);
        Muls(wp[m], wp[m], -1.f, m);
        Mul(r, ar, wp, m);
        MulAddDst(r, ar[m], wp[m], m);
        Muls(wp[m], wp[m], -1.f, m);
        Gather(row, r, it, 0u, 2u * m);                    // row 复用：交错结果
        PipeBarrier<PIPE_ALL>();
        DataCopy(gd, row, 2u * m);
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}

// R2-A Round 2（stripe K 单因素，默认 128x32 tile）：K 只改条带机件的
// 循环粒度与 carve 尺寸（10K<=2HW 在默认 tile 恒满足），不改 tile 峰值。
// AB_LT_STRIPE_K=256|128 选择 _k{K} 入口；K=512 即默认入口。
#define AB_LT_TR_SUFFIX _k256
#define AB_LT_TR_H 128
#define AB_LT_TR_W 32
#define AB_LT_TR_K 256
#include "fft_long_lt_tr.inc"
#undef AB_LT_TR_SUFFIX
#undef AB_LT_TR_H
#undef AB_LT_TR_W
#undef AB_LT_TR_K

#define AB_LT_TR_SUFFIX _k128
#define AB_LT_TR_H 128
#define AB_LT_TR_W 32
#define AB_LT_TR_K 128
#include "fft_long_lt_tr.inc"
#undef AB_LT_TR_SUFFIX
#undef AB_LT_TR_H
#undef AB_LT_TR_W
#undef AB_LT_TR_K
