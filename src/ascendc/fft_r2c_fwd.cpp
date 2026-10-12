// ---------------------------------------------------------------- r2c 融合内核 (fused epilogue)
// R1-D: 同一 writer policy lower 到 fused epilogue
//
// 目标：同时取消：
//   1. C2C 中间结果的 GM 写回
//   2. post 对该中间结果的 GM 重读
//   3. 独立 post launch
//   4. 自然序中间表示
//
// 结构：
//   FFT core final state
//     -> R2C pair math policy (E/O/t 蝶形)
//     -> UB layout policy (forward/reverse fragments)
//     -> publication policy (两个连续 MTE3 DataCopy)
//
// 与 separate 的区别：
//   separate: kfft_fwd -> GM write Z -> kfft_r2c_post -> GM read Z -> GM write X
//   fused:    kfft_r2c_fwd (内含 FFT + r2c post) -> GM write X
//
// 数学（修正版，X[m-k] 不是 X[k] 的共轭）：
//   对 k=0..m/2，a=Z[k]，b=Z[(m-k)%m]
//   E = (a + conj(b)) / 2
//   O = (a - conj(b)) / (2i) = ((ai+bi)/2, (br-ar)/2)
//   t = W_n^k · O
//   X[k] = E + t
//   X[m-k] = conj(E - t)  （k=0 时 X[0] 实数；k=m/2 时自配对只写一次）
//
// 简化版：D=1（无批折叠），K=planeKFor(n)（默认 plane 长度）
// 这是 R1-D 的原型，证明 fused-epilogue 的正确性和性能潜力
#include "butterfly/fft_k.hpp"
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
using namespace AscendC;

// 每块处理的复数槽数（4 的倍数 => DataCopy 计数 2C 为 8 倍数）
static const uint32_t kChunk = 1024u;

// 半谱行距（r2c 输出 / c2r 输入），单位 float。
BFLY_AICORE inline uint32_t hsStride(uint32_t n) { return n + 16u; }

// UB 分区大小上取整到 32B
BFLY_AICORE inline uint32_t ub32(uint32_t bytes) {
    return ((bytes + 31u) & ~31u);
}

// 简化版 radix-2 FFT（单 plane，无批折叠）
// 输入：planar (pr, pi)，输出：planar (pr, pi)
// 这是 kfft_fwd 的简化版本，用于 R1-D 原型
BFLY_AICORE inline void simple_fft(
    LocalTensor<float> pr, LocalTensor<float> pi,
    LocalTensor<float> wr, LocalTensor<float> wi,
    uint32_t n)
{
    // 位反转置换
    for (uint32_t i = 0; i < n; i++) {
        uint32_t j = 0;
        for (uint32_t b = 0; b < 32; b++) {
            if (n <= (1u << b)) break;
            j = (j << 1) | ((i >> b) & 1u);
        }
        if (i < j) {
            float tr = pr.GetValue(i); pr.SetValue(i, pr.GetValue(j)); pr.SetValue(j, tr);
            float ti = pi.GetValue(i); pi.SetValue(i, pi.GetValue(j)); pi.SetValue(j, ti);
        }
    }

    // 蝶形计算
    for (uint32_t len = 2; len <= n; len <<= 1) {
        const uint32_t half = len >> 1;
        const uint32_t step = n / len;
        for (uint32_t i = 0; i < n; i += len) {
            for (uint32_t j = 0; j < half; j++) {
                const uint32_t k = j * step;
                const uint32_t idx0 = i + j;
                const uint32_t idx1 = i + j + half;

                const float xr = pr.GetValue(idx1);
                const float xi = pi.GetValue(idx1);
                const float wrr = wr.GetValue(k);
                const float wri = wi.GetValue(k);

                // twiddle * x
                const float tr = wrr * xr - wri * xi;
                const float ti = wrr * xi + wri * xr;

                const float ar = pr.GetValue(idx0);
                const float ai = pi.GetValue(idx0);

                pr.SetValue(idx0, ar + tr);
                pi.SetValue(idx0, ai + ti);
                pr.SetValue(idx1, ar - tr);
                pi.SetValue(idx1, ai - ti);
            }
        }
    }
}

// 融合内核签名匹配 setReal 的 40B 约定：out, in, ex0(twCombined), ex1(dummy), n, batch
// twCombined = [fftWr | fftWi | r2cWr | r2cWi]，各 pPad floats
extern "C" __global__ __aicore__ __vector__ void kfft_r2c_fwd(
    __gm__ float* out, __gm__ float* in,
    __gm__ float* twCombined, __gm__ float* dummy,
    uint32_t n, uint32_t batch)
{
    (void)dummy;
    const uint32_t m      = n >> 1;              // 内层 FFT 长度
    const uint32_t mHalf  = m >> 1;              // pair 数上限（k=0..m/2）
    const uint32_t strideF = hsStride(n);        // 半谱行距
    const uint32_t twPad  = ((m + 1u) + 15u) & ~15u;  // 旋转因子表 padding（与宿主 pPad 一致）
    const uint32_t r2cTwLen = ((mHalf + 1u) + 7u) & ~7u;

    TPipe pipe;
    TBuf<TPosition::VECCALC> bIn, bFftWr, bFftWi, bR2cWr, bR2cWi, bPlan, bStg;
    pipe.InitBuffer(bIn, ub32(2u * m * sizeof(float)));
    pipe.InitBuffer(bFftWr, ub32(twPad * sizeof(float)));
    pipe.InitBuffer(bFftWi, ub32(twPad * sizeof(float)));
    pipe.InitBuffer(bR2cWr, ub32(twPad * sizeof(float)));
    pipe.InitBuffer(bR2cWi, ub32(twPad * sizeof(float)));
    pipe.InitBuffer(bPlan, ub32(2u * m * sizeof(float)));
    const uint32_t fragLen = ((mHalf + 1u) + 3u) & ~3u;
    pipe.InitBuffer(bStg, ub32(4u * fragLen * sizeof(float)));

    LocalTensor<float> zin   = bIn.Get<float>();
    LocalTensor<float> fftWr = bFftWr.Get<float>(), fftWi = bFftWi.Get<float>();
    LocalTensor<float> r2cWr = bR2cWr.Get<float>(), r2cWi = bR2cWi.Get<float>();
    LocalTensor<float> plan  = bPlan.Get<float>();
    LocalTensor<float> pr = plan, pi = plan[m];
    LocalTensor<float> stg = bStg.Get<float>();
    LocalTensor<float> fwd = stg, rev = stg[2u * fragLen];

    // 加载合并的旋转因子表
    GlobalTensor<float> gTw;
    gTw.SetGlobalBuffer(twCombined, 4u * twPad);
    DataCopy(fftWr, gTw, twPad);
    DataCopy(fftWi, gTw[twPad], twPad);
    DataCopy(r2cWr, gTw[2u * twPad], twPad);
    DataCopy(r2cWi, gTw[3u * twPad], twPad);
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk  = GetBlockIdx();

    for (int64_t b = blk; b < (int64_t)batch; b += nblk) {
        PipeBarrier<PIPE_ALL>();      // 行首 drain
        const uint32_t rowOff = (uint32_t)b * strideF;
        GlobalTensor<float> gin, gout;
        gin.SetGlobalBuffer(in + (uint32_t)b * n, n);

        // ---- 1. 打包：z[k] = x[2k] + i*x[2k+1] ----
        DataCopy(zin, gin, n);
        PipeBarrier<PIPE_ALL>();
        // 分离实部和虚部到 planar
        for (uint32_t k = 0; k < m; k++) {
            pr.SetValue(k, zin.GetValue(2u * k));
            pi.SetValue(k, zin.GetValue(2u * k + 1u));
        }

        // ---- 2. FFT ----
        simple_fft(pr, pi, fftWr, fftWi, m);
        PipeBarrier<PIPE_ALL>();

        // ---- 3. R2C 后处理（pair math + layout + publication）----
        // 对 k=0..mHalf，计算 X[k] 和 X[m-k]
        for (uint32_t k = 0; k <= mHalf; k++) {
            // a = Z[k]
            const float ar = pr.GetValue(k);
            const float ai = pi.GetValue(k);
            // b = Z[(m-k) % m]
            const uint32_t bk = (k == 0u) ? 0u : (m - k);
            const float br = pr.GetValue(bk);
            const float bi = pi.GetValue(bk);
            // W_n^k
            const float wrr = r2cWr.GetValue(k);
            const float wri = r2cWi.GetValue(k);

            // E = (a + conj(b))/2
            const float er = (ar + br) * 0.5f;
            const float ei = (ai - bi) * 0.5f;
            // O = (a - conj(b))/(2i)：O.re=(ai+bi)/2，O.im=(br-ar)/2
            const float ore = (ai + bi) * 0.5f;
            const float oim = (br - ar) * 0.5f;
            // t = W * O
            const float tre = wrr * ore - wri * oim;
            const float tim = wrr * oim + wri * ore;

            // X[k] = E + t（forward fragment）
            fwd.SetValue(2u * k,     er + tre);
            fwd.SetValue(2u * k + 1u, ei + tim);

            // X[m-k] = conj(E - t)
            if (k == 0u) {
                // X[m] = conj(E - t) = (Re(E-t), -Im(E-t))
                rev.SetValue(0u,    er - tre);
                rev.SetValue(1u,    0.0f);  // Nyquist 实数
            }
            else if (k < mHalf) {
                // X[m-k] = conj(E - t) = (Re(E-t), -Im(E-t))
                rev.SetValue(2u * k,     er - tre);
                rev.SetValue(2u * k + 1u, -(tim - ei));  // -Im(E-t) = Im(t)-Im(E)
            }
        }
        PipeBarrier<PIPE_ALL>();

        // ---- 4. Publication：两个连续 MTE3 DataCopy ----
        // 1. Forward fragment: X[0..mHalf] -> out[0..mHalf]
        gout.SetGlobalBuffer(out + rowOff, 2u * fragLen);
        DataCopy(gout, fwd, 2u * fragLen);

        // 2. Reverse fragment: X[m], X[m-1], ..., X[mHalf+1] -> out[mHalf+1..m]
        //    UB 中存储顺序：rev[0]=X[m], rev[1]=X[m-1], ..., rev[mHalf-1]=X[mHalf+1]
        //    GM 中需要正向连续：out[mHalf+1], out[mHalf+2], ..., out[m]
        //    需要反转后写出
        if (mHalf >= 1) {
            // 反转 reverse fragment（交错复数布局）
            const uint32_t revLen = mHalf;
            for (uint32_t i = 0; i < revLen / 2; i++) {
                const uint32_t j = revLen - 1 - i;
                float tmp0 = rev.GetValue(2u * i);
                float tmp1 = rev.GetValue(2u * i + 1u);
                rev.SetValue(2u * i,     rev.GetValue(2u * j));
                rev.SetValue(2u * i + 1u, rev.GetValue(2u * j + 1u));
                rev.SetValue(2u * j,     tmp0);
                rev.SetValue(2u * j + 1u, tmp1);
            }
            // 写入 GM：out[mHalf+1..m]
            const uint32_t revLenO = (revLen + 3u) & ~3u;
            gout.SetGlobalBuffer(out + rowOff + 2u * (mHalf + 1u), 2u * revLenO);
            DataCopy(gout, rev, 2u * revLenO);
        }
    }
}
