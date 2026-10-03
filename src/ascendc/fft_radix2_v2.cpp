// Phase 3: radix-2 DIT complex fp32 FFT，前 3 级(h=1,2,4)改用 8-plane 布局矢量化。
//
// 关键约束(实测)：矢量算子的 LocalTensor offset 必须是 8 float(32B) 的倍数，
//   否则 AIV 抛 507035。planar 布局下 stage h 的高半区 offset = base+h 恒非 8 对齐
//   (h=1,2,4)，因此这三级只能标量 —— 实测占全 kernel 的 46.6%。
//
// 解法：把 下标 j 的前 3 位取出来当 plane、其余当 tile：
//     pos(j) = (j & 7) * (n/8) + (j >> 3)
//   于是 stage h∈{1,2,4} 的蝶形两端落在 **同一个 tile、不同 plane**：
//     h=1: (p,p+1) p∈{0,2,4,6}      w = 1
//     h=2: (0,2)(4,6) w[0]=1 ; (1,3)(5,7) w[1]=-i
//     h=4: (p,p+4)   p∈{0..3}       w = exp(-i*pi*p/4)
//   plane 基址 = p*(n/8)，n 为 64 的倍数 => 恒 8 对齐；且 twiddle 对每对 plane 是常数，
//   可用 Muls 标量乘完成复数乘。三个 stage 结束后用 Gather 按索引转回 planar，
//   后续 h>=8 沿用原 planar 矢量路径。
//
// UB: inter 2n + plan(pr,pi) 2n + plane(ar,ai) 2n + idx 4n(B) + tw 8n+128 + tmp 4*(n/2)*4
//   = 44n + 128 bytes  (n=4096 -> 176.1 KiB < 192 KiB)
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
using namespace AscendC;

extern "C" __global__ __aicore__ __vector__ void kfft_fwd(__gm__ float* out, __gm__ float* in,
    uint32_t n, uint32_t batch, __gm__ float* twrGm, __gm__ float* twiGm)
{
    uint32_t logn = 0;
    while ((1u << logn) < n) logn++;
    const uint32_t twPad = n + 16;
    const uint32_t hmax = n >> 1;
    const uint32_t tiles = n >> 3;      // n/8，plane 长度

    TPipe pipe;
    TBuf<TPosition::VECCALC> bInter, bPlan, bPlane, bIdx, bTwR, bTwI, bTmp;
    pipe.InitBuffer(bInter, 2 * n * sizeof(float));
    pipe.InitBuffer(bPlan,  2 * n * sizeof(float));
    pipe.InitBuffer(bPlane, 2 * n * sizeof(float));
    pipe.InitBuffer(bIdx,     n * sizeof(uint32_t));
    pipe.InitBuffer(bTwR,  twPad * sizeof(float));
    pipe.InitBuffer(bTwI,  twPad * sizeof(float));
    pipe.InitBuffer(bTmp,  4 * hmax * sizeof(float));

    LocalTensor<float> inter = bInter.Get<float>();
    LocalTensor<float> pr = bPlan.Get<float>(),   pi = pr[n];
    LocalTensor<float> ar = bPlane.Get<float>(),  ai = ar[n];
    LocalTensor<uint32_t> idx = bIdx.Get<uint32_t>();
    LocalTensor<float> twr = bTwR.Get<float>(),   twi = bTwI.Get<float>();
    LocalTensor<float> t0 = bTmp.Get<float>(), t1 = t0[hmax], t2 = t1[hmax], t3 = t2[hmax];

    GlobalTensor<float> gtwr, gtwi;
    gtwr.SetGlobalBuffer(twrGm, twPad);
    gtwi.SetGlobalBuffer(twiGm, twPad);
    DataCopy(twr, gtwr, twPad);
    DataCopy(twi, gtwi, twPad);

    // 转置索引：planar[k] = plane[(k&7)*tiles + (k>>3)]，单位字节(实测 srcOffset 为字节)
    for (uint32_t k = 0; k < n; k++)
        idx.SetValue(k, (((k & 7u) * tiles) + (k >> 3)) * 4u);
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk = GetBlockIdx();

    for (int64_t b = blk; b < (int64_t)batch; b += nblk) {
        uint32_t goff = (uint32_t)b * 2u * n;
        GlobalTensor<float> gin, gout;
        gin.SetGlobalBuffer(in + goff, 2 * n);
        gout.SetGlobalBuffer(out + goff, 2 * n);
        DataCopy(inter, gin, 2 * n);
        PipeBarrier<PIPE_ALL>();

        // 位反转散射直接写进 plane 布局（标量，代价与原实现相同）
        uint32_t j = 0;
        const uint32_t hiBit = 1u << (logn - 1);
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t pos = ((j & 7u) * tiles) + (j >> 3);
            ar.SetValue(pos, inter.GetValue(2 * i));
            ai.SetValue(pos, inter.GetValue(2 * i + 1));
            uint32_t mask = hiBit;
            while (j & mask) { j ^= mask; mask >>= 1; }
            j ^= mask;
        }
        PipeBarrier<PIPE_ALL>();

        // ---- stage h = 1,2,4：plane 布局上的矢量化 ----
        uint32_t slot = 0;
        for (uint32_t si = 0; si < 3; si++) {
            const uint32_t h = 1u << si;
            const uint32_t sl = slot;
            slot += (h + 7u) & ~7u;
            // plane 对：p = 2h*m + j, m<4/h, j<h  => 恰 4 对
            for (uint32_t m = 0; m < (4u / h); m++) {
                for (uint32_t j2 = 0; j2 < h; j2++) {
                    const uint32_t p = 2u * h * m + j2;
                    const uint32_t q = p + h;
                    const float wr = twr.GetValue(sl + j2);
                    const float wi = twi.GetValue(sl + j2);
                    LocalTensor<float> rp = ar[p * tiles], rq = ar[q * tiles];
                    LocalTensor<float> ip = ai[p * tiles], iq = ai[q * tiles];

                    if (wr != 1.0f || wi != 0.0f) {   // 复数乘 high *= w
                        Muls(t0, rq, wr, tiles);
                        Muls(t1, iq, wi, tiles);
                        Sub(t2, t0, t1, tiles);
                        Muls(t0, rq, wi, tiles);
                        Muls(t1, iq, wr, tiles);
                        Add(t3, t0, t1, tiles);
                    } else {
                        DataCopy(t2, rq, tiles);
                        DataCopy(t3, iq, tiles);
                    }
                    PipeBarrier<PIPE_V>();
                    // 蝶形 (re)
                    Add(t0, rp, t2, tiles);
                    Sub(t1, rp, t2, tiles);
                    Adds(rp, t0, 0.0f, tiles);
                    Adds(rq, t1, 0.0f, tiles);
                    // 蝶形 (im)
                    Add(t0, ip, t3, tiles);
                    Sub(t1, ip, t3, tiles);
                    Adds(ip, t0, 0.0f, tiles);
                    Adds(iq, t1, 0.0f, tiles);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        PipeBarrier<PIPE_ALL>();

        // ---- plane -> planar 转置（Gather，索引单位为字节，实测）----
        Gather(pr, ar, idx, 0u, n);
        Gather(pi, ai, idx, 0u, n);
        PipeBarrier<PIPE_ALL>();

        // ---- stage h >= 8：planar 矢量路径（h%8==0 => 高半区 8 对齐）----
        for (uint32_t s = 3; s < logn; s++) {
            const uint32_t h = 1u << s;
            const uint32_t groups = n / (2u * h);
            const uint32_t sl = slot;
            slot += (h + 7u) & ~7u;
            for (uint32_t g = 0; g < groups; g++) {
                const uint32_t base = g * 2u * h;
                LocalTensor<float> tr = twr[sl];
                LocalTensor<float> ti = twi[sl];
                LocalTensor<float> r0 = pr[base], r1 = pr[base + h];
                LocalTensor<float> i0 = pi[base], i1 = pi[base + h];
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
                PipeBarrier<PIPE_V>();
            }
        }
        PipeBarrier<PIPE_ALL>();

        // ---- planar -> 交错输出 ----
        for (uint32_t i = 0; i < n; i++) {
            inter.SetValue(2 * i,     pr.GetValue(i));
            inter.SetValue(2 * i + 1, pi.GetValue(i));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(gout, inter, 2 * n);
        PipeBarrier<PIPE_ALL>();
    }
}
