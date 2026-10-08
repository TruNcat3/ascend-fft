// 长链 device-materialized 边界内核（addendum §3 step 3）：
//   kfft_lt_tr   分块矩阵转置：src[b][nRows][nCols] -> dst[b][nCols][nRows]，
//                即 dst[c][i] = src[i][c]（转置入 / 段间转置 / 自然序写出共用）
//   kfft_lt_tw   段边界点乘：dIn[j][k1] *= W_N^{j·k1}（行连续，原地）
// 边界形态：host 路径把 twiddle+转置合在宿主内存里做；device 路径拆成
// 点乘（全连续 GM 读写）+ 转置（列切片 strided 读 + 行段连续写）两步，
// 段边界数据不回宿主（E2E boundary=0），wT 表 plan 期上传（输入无关）。
//
// GM 非连续访问用 DataCopyParams 分块搬运（CANN 9.0.0 单位：blockLen/srcStride
// 均为 32B datablock，gap 为「上一块尾->下一块头」，blockCount<=4095）：
//   每块 = src 行内列切片 [c0, c0+wCnt)（8*wCnt 字节），块间隔 = 行距 - 块长。
// 本 SoC 的 Gather 仅支持 UB 源（GlobalTensor 源编译报错，首版实测），
// 因此先整块搬进 UB，再用 UB 源 Gather 做重排（索引只依赖形状，启动时建一次）。
// 约束：nRows/nCols 为 >=64 的 2 的幂（阶段因子域），W=32 恒 4 对齐，
// blockLen=wCnt/4、srcGap=(nCols-wCnt)/4 均为整 datablock。
// 两内核同签名 (dst, src, tw, nRows, nCols, batch) = 36B（fft_real 同款约定，
// readArgSize 对整个 .o 只给一个值；kfft_lt_tr 的 tw 不用但必须占位）。
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
using namespace AscendC;

#define LT_H 128u   // 分块行数（src 行内切块），blockCount <= 4095
#define LT_W 32u    // 分块列数（dst 行内一段），须 4 的倍数（32B 对齐）

// ---- 分块转置 ------------------------------------------------------------
// tile 读入 [H][W]（行=i 列=c，交错复数），Gather 重排为 dst 行序 [W][H]，
// 再按 dst 行逐段连续写回。dst 行号按批线性展开，grid-stride 分块。
extern "C" __global__ __aicore__ __vector__ void kfft_lt_tr(
    __gm__ float* dst, __gm__ float* src, __gm__ float* tw,
    uint32_t nRows, uint32_t nCols, uint32_t batch)
{
    (void)tw;
    TPipe pipe;
    TBuf<TPosition::VECCALC> bIn, bOut, bIdx;
    const uint32_t tileN = LT_H * LT_W;                    // 复数 / tile
    pipe.InitBuffer(bIn,  tileN * 2u * sizeof(float));
    pipe.InitBuffer(bOut, tileN * 2u * sizeof(float));
    pipe.InitBuffer(bIdx, tileN * 2u * sizeof(uint32_t));
    LocalTensor<float> tIn  = bIn.Get<float>();
    LocalTensor<float> tOut = bOut.Get<float>();
    LocalTensor<uint32_t> idx = bIdx.Get<uint32_t>();
    // 重排索引：dst 行序 t=2*(w*H+i)+e <- tile 位置 (i*W+w)
    for (uint32_t i = 0; i < LT_H; i++)
        for (uint32_t w = 0; w < LT_W; w++) {
            uint32_t p = (i * LT_W + w) * 8u;
            idx.SetValue(2u * (w * LT_H + i),      p);
            idx.SetValue(2u * (w * LT_H + i) + 1u, p + 4u);
        }
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk  = GetBlockIdx();
    const uint32_t nC = nCols / LT_W;                      // dst 行 tile 数 / 批
    const uint32_t nI = (nRows + LT_H - 1u) / LT_H;        // src 行 tile 数
    const uint64_t tiles = (uint64_t)batch * nC * nI;
    const uint64_t span  = 2ull * (uint64_t)nRows * nCols; // float / 批（复数*2）
    for (uint64_t tt = (uint64_t)blk; tt < tiles; tt += (uint64_t)nblk) {
        const uint32_t wT0 = (uint32_t)(tt % nC) * LT_W;
        const uint32_t iT0 = (uint32_t)((tt / nC) % nI) * LT_H;
        const uint32_t b   = (uint32_t)(tt / ((uint64_t)nC * nI));
        const uint32_t wCnt = nCols - wT0 >= LT_W ? LT_W : nCols - wT0;
        const uint32_t rCnt = nRows - iT0 >= LT_H ? LT_H : nRows - iT0;
        GlobalTensor<float> gs, gd;
        // 块 r 起点 = base + r*(nCols*8 字节)；块长 wCnt*8，gap = (nCols-wCnt)*8
        gs.SetGlobalBuffer(src + (uint64_t)b * span +
                           ((uint64_t)iT0 * nCols + (uint64_t)wT0) * 2u,
                           (uint64_t)rCnt * nCols * 2u);
        DataCopy(tIn, gs,
                 DataCopyParams{(uint16_t)rCnt, (uint16_t)(wCnt / 4u),
                                (uint16_t)((nCols - wCnt) / 4u), 0});
        PipeBarrier<PIPE_ALL>();
        Gather(tOut, tIn, idx, 0u, 2u * tileN);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t w = 0; w < wCnt; w++) {
            gd.SetGlobalBuffer(dst + (uint64_t)b * span +
                               ((uint64_t)(wT0 + w) * nRows + iT0) * 2u, rCnt * 2u);
            DataCopy(gd, tOut[w * 2u * LT_H], rCnt * 2u);
        }
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}

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
