// Phase 1: radix-2 decimation-in-time complex fp32 FFT, planar layout in UB.
//   in  : [batch][2*n] interleaved complex (re,im)
//   out : [batch][2*n] interleaved complex
//   twr/twi : twiddle table with one 8-aligned slot per stage.
//             stage s (h = 1<<s) occupies slot offset OFF(s) .. OFF(s)+align8(h),
//             entry j of the stage = exp(-i*pi*j/h), consumed as twr[OFF(s)+j].
// Input is consumed in bit-reversed order (matches cuButterfly kBitReverseInput).
//
// HARDWARE RULE (measured): every LocalTensor offset used by a vector op must be
// 32B aligned, i.e. a multiple of 8 floats.  Stages h = 1,2,4 put the upper half at
// an unaligned offset, so those three stages run scalar; h >= 8 runs vector.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ __vector__ void kfft_fwd(__gm__ float* out, __gm__ float* in,
    uint32_t n, uint32_t batch, __gm__ float* twrGm, __gm__ float* twiGm)
{
    uint32_t logn = 0;
    while ((1u << logn) < n) logn++;
    uint32_t twPad = n + 16;
    uint32_t hmax = n >> 1;

    TPipe pipe;
    TBuf<TPosition::VECCALC> bInter, bRe, bIm, bTwR, bTwI, bT0, bT1, bT2;
    pipe.InitBuffer(bInter, 2 * n * sizeof(float));
    pipe.InitBuffer(bRe, n * sizeof(float));
    pipe.InitBuffer(bIm, n * sizeof(float));
    pipe.InitBuffer(bTwR, twPad * sizeof(float));
    pipe.InitBuffer(bTwI, twPad * sizeof(float));
    pipe.InitBuffer(bT0, hmax * sizeof(float));
    pipe.InitBuffer(bT1, hmax * sizeof(float));
    pipe.InitBuffer(bT2, hmax * sizeof(float));

    LocalTensor<float> inter = bInter.Get<float>();
    LocalTensor<float> re = bRe.Get<float>();
    LocalTensor<float> im = bIm.Get<float>();
    LocalTensor<float> twr = bTwR.Get<float>();
    LocalTensor<float> twi = bTwI.Get<float>();
    LocalTensor<float> t0 = bT0.Get<float>();
    LocalTensor<float> t1 = bT1.Get<float>();
    LocalTensor<float> t2 = bT2.Get<float>();

    GlobalTensor<float> gtwr, gtwi;
    gtwr.SetGlobalBuffer(twrGm, twPad);
    gtwi.SetGlobalBuffer(twiGm, twPad);
    DataCopy(twr, gtwr, twPad);
    DataCopy(twi, gtwi, twPad);
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum();
    if (nblk <= 0) nblk = 1;
    int64_t blk = GetBlockIdx();

    for (int64_t b = blk; b < (int64_t)batch; b += nblk) {
        uint32_t goff = (uint32_t)b * 2u * n;
        GlobalTensor<float> gin, gout;
        gin.SetGlobalBuffer(in + goff, 2 * n);
        gout.SetGlobalBuffer(out + goff, 2 * n);
        DataCopy(inter, gin, 2 * n);
        PipeBarrier<PIPE_ALL>();

        // 增量式位反转计数器：均摊 ~2 次运算/元素，替代 logn 次/元素
        uint32_t j = 0;
        const uint32_t hiBit = 1u << (logn - 1);
        for (uint32_t i = 0; i < n; i++) {
#ifdef PHASE_NO_BITREV
            re.SetValue(i, inter.GetValue(2 * i));
            im.SetValue(i, inter.GetValue(2 * i + 1));
#else
            re.SetValue(j, inter.GetValue(2 * i));
            im.SetValue(j, inter.GetValue(2 * i + 1));
            uint32_t mask = hiBit;
            while (j & mask) { j ^= mask; mask >>= 1; }
            j ^= mask;
#endif
        }
        PipeBarrier<PIPE_ALL>();

#ifdef PHASE_NO_STAGES
        // 只做 进入+位反转+交错+写出，用于分段计时
        for (uint32_t i = 0; i < n; i++) {
            inter.SetValue(2 * i, re.GetValue(i));
            inter.SetValue(2 * i + 1, im.GetValue(i));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(gout, inter, 2 * n);
        PipeBarrier<PIPE_ALL>();
        return;
#endif
#ifdef PHASE_PRETOUCH
        Adds(re, re, 0.0f, n);
        Adds(im, im, 0.0f, n);
        Adds(twr, twr, 0.0f, 8);
        Adds(twi, twi, 0.0f, 8);
        PipeBarrier<PIPE_ALL>();
#endif
        // precompute the 8-aligned slot offset of every stage
        uint32_t off = 0;
        for (uint32_t s = 0; s < logn; s++) {
            uint32_t h = 1u << s;
#if defined(PHASE_VEC_ONLY) && defined(PHASE_MAX_STAGE)
            if (s > PHASE_MAX_STAGE) break;
#endif
            uint32_t slot = off;
            off += (h + 7u) & ~7u;
#if defined(PHASE_ONLY_S)
            if (s != PHASE_ONLY_S) continue;
#endif
#ifdef PHASE_BARRIER_ONLY
            PipeBarrier<PIPE_V>();
            break;
#endif
#ifdef PHASE_BARRIER_ALL
            PipeBarrier<PIPE_ALL>();
            break;
#endif
#ifdef PHASE_BARRIER_NONE
            break;
#endif
#ifdef PHASE_TINY_T0
            Adds(t0, re, 0.0f, 8); PipeBarrier<PIPE_V>(); break;
#endif
#ifdef PHASE_TINY_T1
            Adds(t1, re, 0.0f, 8); PipeBarrier<PIPE_V>(); break;
#endif
#ifdef PHASE_TINY_TW
            Adds(t0, twr, 0.0f, 8); PipeBarrier<PIPE_V>(); break;
#endif
#ifdef PHASE_TINY_ALL
            Adds(t0, re, 0.0f, 8); Adds(t1, im, 0.0f, 8);
            Adds(t2, twr, 0.0f, 8); PipeBarrier<PIPE_V>(); break;
#endif
#ifdef PHASE_MADV
            Mul(t0, re, twr, 8); PipeBarrier<PIPE_V>(); break;
#endif
#if defined(PHASE_VEC_ONLY)
            if (h < 8) { continue; }
#elif defined(PHASE_SCALAR_ONLY)
            if (h >= 8) { continue; }
#endif
            if (h < 8) {
                for (uint32_t base = 0; base < n; base += 2 * h) {
                    for (uint32_t j = 0; j < h; j++) {
                        float wr = twr.GetValue(slot + j);
                        float wi = twi.GetValue(slot + j);
                        float ar = re.GetValue(base + j), ai = im.GetValue(base + j);
                        float br = re.GetValue(base + h + j), bi = im.GetValue(base + h + j);
                        float tr = br * wr - bi * wi;
                        float ti = br * wi + bi * wr;
                        re.SetValue(base + j, ar + tr);
                        re.SetValue(base + h + j, ar - tr);
                        im.SetValue(base + j, ai + ti);
                        im.SetValue(base + h + j, ai - ti);
                    }
                }
            } else {
                LocalTensor<float> tr = twr[slot];
                LocalTensor<float> ti = twi[slot];
                for (uint32_t base = 0; base < n; base += 2 * h) {
                    LocalTensor<float> r0 = re[base];
                    LocalTensor<float> r1 = re[base + h];
                    LocalTensor<float> i0 = im[base];
                    LocalTensor<float> i1 = im[base + h];
                    Mul(t0, r1, tr, h);
                    Mul(t2, i1, ti, h);
                    Sub(t0, t0, t2, h);
                    Mul(t2, i1, tr, h);
                    Mul(t1, r1, ti, h);
                    Add(t1, t2, t1, h);
                    Sub(r1, r0, t0, h);
                    Add(r0, r0, t0, h);
                    Sub(i1, i0, t1, h);
                    Add(i0, i0, t1, h);
                }
            }
#ifdef PHASE_FORCE_ALL
            PipeBarrier<PIPE_ALL>();
#else
            if (h < 8) {
                PipeBarrier<PIPE_ALL>();   // scalar (S) -> vector/next stage
            } else {
                PipeBarrier<PIPE_V>();     // vector -> vector
            }
#endif
        }

        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < n; i++) {
            inter.SetValue(2 * i, re.GetValue(i));
            inter.SetValue(2 * i + 1, im.GetValue(i));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(gout, inter, 2 * n);
        PipeBarrier<PIPE_ALL>();
#ifdef PHASE_RET
        return;
#endif
    }
}
