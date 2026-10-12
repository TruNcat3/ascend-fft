// R1-B ablation 变体：通过 -D 宏控制消除各组件，测量各组件对 304µs 基线的贡献。
//   ABL_NOREV   : 跳过 reverse 反转（省标量 swap 开销）
//   ABL_NOSTORE : 标量 SetValue 换矢量 Gather 交错（省标量 store 开销）
//   ABL_NOBAR   : 去掉 chunk 内 PipeBarrier（省屏障开销）
//   ABL_NOGATHER: Gather 换标量 GetValue（测量 Gather 节省）
#include "butterfly/fft_k.hpp"
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
using namespace AscendC;

static const uint32_t kChunk = 1024u;
BFLY_AICORE inline uint32_t hsStride(uint32_t n) { return n + 16u; }
BFLY_AICORE inline uint32_t ub32(uint32_t bytes) { return ((bytes + 31u) & ~31u); }

extern "C" __global__ __aicore__ __vector__ void kfft_r2c_post_static_abl(
    __gm__ float* out, __gm__ float* z, __gm__ float* wrrGm, __gm__ float* wriGm,
    uint32_t n, uint32_t batch)
{
    const uint32_t m      = n >> 1;
    const uint32_t mHalf  = m >> 1;
    const uint32_t strideF = hsStride(n);
    const uint32_t twLen  = ((mHalf + 1u) + 7u) & ~7u;

    TPipe pipe;
    TBuf<TPosition::VECCALC> bZ, bWr, bWi, bT, bTmp, bStg, bPair;
    pipe.InitBuffer(bZ, ub32(n * sizeof(float)));
    pipe.InitBuffer(bWr, ub32(twLen * sizeof(float)));
    pipe.InitBuffer(bWi, ub32(twLen * sizeof(float)));
    pipe.InitBuffer(bT, ub32(3u * twLen * sizeof(uint32_t)));
    pipe.InitBuffer(bTmp, ub32(16u * kChunk * sizeof(float)));
    const uint32_t fragLen = ((mHalf + 1u) + 3u) & ~3u;
    pipe.InitBuffer(bStg, ub32(4u * fragLen * sizeof(float)));
    // ABL_NOSTORE 用：交错表 tPair（dst[2i]=4i, dst[2i+1]=4*(kChunk+i)）
    pipe.InitBuffer(bPair, ub32(2u * kChunk * sizeof(uint32_t)));

    LocalTensor<float> zr    = bZ.Get<float>();
    LocalTensor<float> wrr   = bWr.Get<float>(), wri = bWi.Get<float>();
    LocalTensor<uint32_t> t0 = bT.Get<uint32_t>();
    LocalTensor<uint32_t> tA = t0,           tB = tA[twLen],       tW = tB[twLen];
    LocalTensor<float> tm    = bTmp.Get<float>();
    LocalTensor<float> ar = tm,        ai = ar[kChunk],   br = ai[kChunk],  bi = br[kChunk];
    LocalTensor<float> er = bi[kChunk], ei = er[kChunk],  ore = ei[kChunk], oim = ore[kChunk];
    LocalTensor<float> tre = oim[kChunk], tim = tre[kChunk];
    LocalTensor<float> wv0 = tim[kChunk], wv1 = wv0[kChunk];
    LocalTensor<float> p0  = wv1[kChunk], p1 = p0[kChunk];
    LocalTensor<float> oR  = p1[kChunk],  oI = oR[kChunk];
    LocalTensor<float> stg = bStg.Get<float>();
    LocalTensor<float> fwd = stg, rev = stg[2u * fragLen];
    LocalTensor<uint32_t> tPair = bPair.Get<uint32_t>();

    for (uint32_t k = 0; k <= mHalf; k++) {
        tA.SetValue(k, k * 8u);
        tB.SetValue(k, (k == 0u) ? 0u : ((m - k) * 8u));
        tW.SetValue(k, k * 4u);
    }
#ifdef ABL_NOSTORE
    for (uint32_t i = 0; i < kChunk; i++) {
        tPair.SetValue(2u * i,     4u * i);
        tPair.SetValue(2u * i + 1u, 4u * (kChunk + i));
    }
#endif
    GlobalTensor<float> gwrr, gwri;
    gwrr.SetGlobalBuffer(wrrGm, twLen);
    gwri.SetGlobalBuffer(wriGm, twLen);
    DataCopy(wrr, gwrr, twLen);
    DataCopy(wri, gwri, twLen);
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk  = GetBlockIdx();

    for (int64_t b = blk; b < (int64_t)batch; b += nblk) {
        PipeBarrier<PIPE_ALL>();
        const uint32_t rowOff = (uint32_t)b * strideF;
        GlobalTensor<float> gz, gout;
        gz.SetGlobalBuffer(z + (uint32_t)b * n, n);
        DataCopy(zr, gz, n);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t k = 0; k <= mHalf; k += kChunk) {
            const uint32_t len  = ((mHalf + 1u - k) < kChunk) ? (mHalf + 1u - k) : kChunk;
            const uint32_t lenO = (len + 3u) & ~3u;
            const uint32_t fwdOff = k;
            const uint32_t revBase = mHalf + 1u;

#ifdef ABL_NOGATHER
            // 标量读替代 Gather
            for (uint32_t i = 0; i < len; i++) {
                const uint32_t kk = k + i;
                const uint32_t bk = (kk == 0u) ? 0u : (m - kk);
                ar.SetValue(i, zr.GetValue(2u * kk));
                ai.SetValue(i, zr.GetValue(2u * kk + 1u));
                br.SetValue(i, zr.GetValue(2u * bk));
                bi.SetValue(i, zr.GetValue(2u * bk + 1u));
                wv0.SetValue(i, wrr.GetValue(kk));
                wv1.SetValue(i, wri.GetValue(kk));
            }
#else
            Gather(ar,  zr, tA[k], 0u, len);
            Gather(ai,  zr, tA[k], 4u, len);
            Gather(br,  zr, tB[k], 0u, len);
            Gather(bi,  zr, tB[k], 4u, len);
            Gather(wv0, wrr, tW[k], 0u, len);
            Gather(wv1, wri, tW[k], 0u, len);
#endif
#ifndef ABL_NOBAR
            PipeBarrier<PIPE_V>();
#endif

            Add(er, ar, br, len);  Muls(er, er, 0.5f, len);
            Sub(ei, ai, bi, len);  Muls(ei, ei, 0.5f, len);
            Add(ore, ai, bi, len); Muls(ore, ore, 0.5f, len);
            Sub(oim, br, ar, len); Muls(oim, oim, 0.5f, len);
            Mul(p0, wv0, ore, len); Mul(p1, wv1, oim, len); Sub(tre, p0, p1, len);
            Mul(p0, wv0, oim, len); Mul(p1, wv1, ore, len); Add(tim, p0, p1, len);

            Add(oR, er, tre, len);
            Add(oI, ei, tim, len);

#ifdef ABL_NOSTORE
            // 矢量 Gather 交错（替代标量 SetValue）
            Gather(fwd, oR, tPair, 0u, 2u * len);
#else
            for (uint32_t i = 0; i < len; i++) {
                fwd.SetValue(2u * i,     oR.GetValue(i));
                fwd.SetValue(2u * i + 1u, oI.GetValue(i));
            }
#endif

            Sub(oR, er, tre, len);
            Sub(oI, tim, ei, len);

#ifdef ABL_NOSTORE
            // reverse 也用矢量 Gather 交错（但需要先反转索引）
            // 简化：直接用标量写 reverse（保留，因为 reverse 有分支逻辑）
            for (uint32_t i = 0; i < len; i++) {
                const uint32_t kk = k + i;
                if (kk == 0u) {
                    rev.SetValue(0u,    oR.GetValue(i));
                    rev.SetValue(1u,    0.0f);
                }
                else if (kk != mHalf) {
                    rev.SetValue(2u * kk,     oR.GetValue(i));
                    rev.SetValue(2u * kk + 1u, oI.GetValue(i));
                }
            }
#else
            for (uint32_t i = 0; i < len; i++) {
                const uint32_t kk = k + i;
                if (kk == 0u) {
                    rev.SetValue(0u,    oR.GetValue(i));
                    rev.SetValue(1u,    0.0f);
                }
                else if (kk != mHalf) {
                    rev.SetValue(2u * kk,     oR.GetValue(i));
                    rev.SetValue(2u * kk + 1u, oI.GetValue(i));
                }
            }
#endif

#ifndef ABL_NOBAR
            PipeBarrier<PIPE_V>();
#endif

            gout.SetGlobalBuffer(out + rowOff + 2u * fwdOff, 2u * lenO);
            DataCopy(gout, fwd, 2u * lenO);

            if (mHalf >= 1) {
                const uint32_t revLen = mHalf;
#ifndef ABL_NOREV
                for (uint32_t i = 0; i < revLen / 2; i++) {
                    const uint32_t j = revLen - 1 - i;
                    float tmp0 = rev.GetValue(2u * i);
                    float tmp1 = rev.GetValue(2u * i + 1u);
                    rev.SetValue(2u * i,     rev.GetValue(2u * j));
                    rev.SetValue(2u * i + 1u, rev.GetValue(2u * j + 1u));
                    rev.SetValue(2u * j,     tmp0);
                    rev.SetValue(2u * j + 1u, tmp1);
                }
#endif
                const uint32_t revLenO = (revLen + 3u) & ~3u;
                const uint32_t revOff = revBase + k;
                gout.SetGlobalBuffer(out + rowOff + 2u * revOff, 2u * revLenO);
                DataCopy(gout, rev, 2u * revLenO);
            }
        }
    }
}
