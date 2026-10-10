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

#define LT_H AB_LT_H   // 分块行数（src 行内切块），blockCount <= 4095
#define LT_W AB_LT_W   // 分块列数（dst 行内一段），须 4 的倍数（32B 对齐）

// ---- 分块转置（+ 可融合 twiddle）------------------------------------------
// tile 读入 [H][W]（行=i 列=c，交错复数），tw!=nullptr 时先做带内 twiddle
// 复乘（AB_FUSE_STRIPE_K 复数一条带：交错->平面 Gather data/tw 各两发 ->
// 平面复乘，与 kfft_lt_tw 同式 -> 交错 Gather 写回本带），再 Gather 重排为
// dst 行序 [W][H]，按 dst 行逐段连续写回。dst 行号按批线性展开，grid-stride 分块。
// 融合时 UB 只多 bTw 一块（AB_FUSED_UB_BYTES = 128 KiB，四块 tile 缓冲）：
// 平面/乘积区与三张索引表全部切自 bOut 的生命周期（转置阶段才写满 bOut），
// 索引只依赖 K 且按条带基址偏移，启动时建一次；tw 表与 src 同形同参、批共享
// （gt 不带 b*span），src 元素 (iT0+i, wT0+w) 即 wT 的 (j,k1) 坐标。
extern "C" __global__ __aicore__ __vector__ void kfft_lt_tr(
    __gm__ float* dst, __gm__ float* src, __gm__ float* tw,
    uint32_t nRows, uint32_t nCols, uint32_t batch)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> bIn, bOut, bIdx, bTw;
    const uint32_t tileN = LT_H * LT_W;                    // 复数 / tile
    pipe.InitBuffer(bIn,  tileN * 2u * sizeof(float));
    pipe.InitBuffer(bOut, tileN * 2u * sizeof(float));
    pipe.InitBuffer(bIdx, tileN * 2u * sizeof(uint32_t));
    pipe.InitBuffer(bTw,  tileN * 2u * sizeof(float));    // 融合 tw tile（峰值 AB_FUSED_UB_BYTES）
    LocalTensor<float> tIn  = bIn.Get<float>();
    LocalTensor<float> tOut = bOut.Get<float>();
    LocalTensor<float> tTw  = bTw.Get<float>();
    LocalTensor<uint32_t> idx = bIdx.Get<uint32_t>();
    // 重排索引：dst 行序 t=2*(w*H+i)+e <- tile 位置 (i*W+w)
    for (uint32_t i = 0; i < LT_H; i++)
        for (uint32_t w = 0; w < LT_W; w++) {
            uint32_t p = (i * LT_W + w) * 8u;
            idx.SetValue(2u * (w * LT_H + i),      p);
            idx.SetValue(2u * (w * LT_H + i) + 1u, p + 4u);
        }
    // 融合条带机件（tw!=nullptr 才建；bOut 里切出生命周期，float/uint32 等宽）：
    //   planarA [0,2K) 数据平面 | planarW [2K,4K) tw 平面 | prod [4K,6K) 乘积
    //   idxRe/idxIm 各 K、idxX 2K 个 uint32 收在 [6K,10K)（10K*4B = 20KB <= 32KB）
    const uint32_t K = AB_FUSE_STRIPE_K;
    LocalTensor<float> planA = tOut;
    LocalTensor<float> planW = tOut[2u * K];
    LocalTensor<float> prod  = tOut[4u * K];
    LocalTensor<uint32_t> uB    = bOut.Get<uint32_t>();
    LocalTensor<uint32_t> idxRe = uB[6u * K];
    LocalTensor<uint32_t> idxIm = uB[7u * K];
    LocalTensor<uint32_t> idxX  = uB[8u * K];
    if (tw) {
        for (uint32_t k = 0; k < K; k++) {
            idxRe.SetValue(k,      8u * k);              // 平面 re <- 交错字节
            idxIm.SetValue(k,      8u * k + 4u);         // 平面 im <- 交错字节
            idxX.SetValue(2u * k,      4u * k);          // 交错 re <- 平面 re
            idxX.SetValue(2u * k + 1u, 4u * K + 4u * k); // 交错 im <- 平面 im
        }
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
        if (tw) {
            // tw 切片与数据切片同坐标同参（wT 形状 [nRows][nCols]，批共享 -> 无 b*span）
            GlobalTensor<float> gt;
            gt.SetGlobalBuffer(tw + ((uint64_t)iT0 * nCols + (uint64_t)wT0) * 2u,
                               (uint64_t)rCnt * nCols * 2u);
            DataCopy(tTw, gt,
                     DataCopyParams{(uint16_t)rCnt, (uint16_t)(wCnt / 4u),
                                    (uint16_t)((nCols - wCnt) / 4u), 0});
        }
        PipeBarrier<PIPE_ALL>();
        if (tw) {
            // idx 表切自 bOut，而每 tile 末尾的大 Gather 会全量写 tOut（= bOut）：
            // grid-stride 第 2 个及以后的 tile 必须重建，否则平面 Gather 按被清掉的
            // 垃圾偏移读 UB -> 越界（AIV 507035）。首个 tile 用入口处建好的表。
            if (tt != (uint64_t)blk) {
                for (uint32_t k = 0; k < K; k++) {
                    idxRe.SetValue(k,      8u * k);
                    idxIm.SetValue(k,      8u * k + 4u);
                    idxX.SetValue(2u * k,      4u * k);
                    idxX.SetValue(2u * k + 1u, 4u * K + 4u * k);
                }
                PipeBarrier<PIPE_ALL>();
            }
            const uint32_t C = rCnt * wCnt;               // 打包复数 / tile
            for (uint32_t s = 0; s < C; s += K) {
                const uint32_t kl = (C - s >= K) ? K : (C - s);
                LocalTensor<float> ds = tIn[2u * s];
                LocalTensor<float> ts = tTw[2u * s];
                Gather(planA,    ds, idxRe, 0u, kl);
                Gather(planA[K], ds, idxIm, 0u, kl);
                Gather(planW,    ts, idxRe, 0u, kl);
                Gather(planW[K], ts, idxIm, 0u, kl);
                PipeBarrier<PIPE_ALL>();
                // 平面复乘（与 kfft_lt_tw 同式）：
                //   im = ar*wi + ai*wr    re = ar*wr + ai*(-wi)
                Mul(prod[K], planA,     planW[K], kl);
                MulAddDst(prod[K], planA[K], planW, kl);
                Muls(planW[K], planW[K], -1.f, kl);
                Mul(prod, planA, planW, kl);
                MulAddDst(prod, planA[K], planW[K], kl);
                Muls(planW[K], planW[K], -1.f, kl);
                PipeBarrier<PIPE_ALL>();
                Gather(ds, prod, idxX, 0u, 2u * kl);       // 写回本带（交错、已乘 tw）
                PipeBarrier<PIPE_ALL>();
            }
        }
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
