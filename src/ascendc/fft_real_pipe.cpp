// ---------------------------------------------------------------- r2c 后处理 (pipelined static paired writer)
// R1-C: 恢复流水，缩小同步域
//
// 相比 R1-B 的改进：
//   1. Ping-pong 双缓冲：两组 UB 缓冲区交替使用
//   2. 缩小同步域：只在必要时使用 PipeBarrier，而非每次迭代
//   3. 计算与写回重叠：下一批的 MTE2 与当前批的 MTE3 并行
//
// 流水线结构：
//   iter i:   MTE2(Z_i) -> compute(X_i) -> MTE3(fwd_i, rev_i)
//   iter i+1: MTE2(Z_{i+1}) 与 MTE3(fwd_i, rev_i) 并行
//
// 关键设计（来自 cuButterfly 迁移）：
//   1. 唯一 pair owner 同时消费 Z[k] 和 Z[m-k]
//   2. 计算 X[k] (forward) 和 X[m-k] (reverse) 两个独立 bin
//   3. 分别放入 UB 的 forward fragment 和 reverse-physical fragment
//   4. 在 UB 中把 reverse fragment 转为正向连续地址
//   5. 用两个连续 MTE3 DataCopy 发布
//
// 数学（修正版，X[m-k] 不是 X[k] 的共轭）：
//   对 k=0..m/2，a=Z[k]，b=Z[(m-k)%m]
//   E = (a + conj(b)) / 2
//   O = (a - conj(b)) / (2i) = ((ai+bi)/2, (br-ar)/2)
//   t = W_n^k · O
//   X[k] = E + t
//   X[m-k] = conj(E - t)  （k=0 时 X[0] 实数；k=m/2 时自配对只写一次）
#include "butterfly/fft_k.hpp"
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
using namespace AscendC;

// 每块处理的复数槽数（4 的倍数 => DataCopy 计数 2C 为 8 倍数）
static const uint32_t kChunk = 1024u;

// 半谱行距（r2c 输出 / c2r 输入），单位 float。
BFLY_AICORE inline uint32_t hsStride(uint32_t n) { return n + 16u; }

// UB 分区大小上取整到 32B：TBuf 依次排布，奇数个 uint32 会让后面的分区起始地址
// 偏离 32B 对齐（实测 Gather dst 非 32B 对齐直接 507035）。统一把每个分区的
// float 数/uint32 数上取整到 8 的倍数。
BFLY_AICORE inline uint32_t ub32(uint32_t bytes) {
    return ((bytes + 31u) & ~31u);
}

extern "C" __global__ __aicore__ __vector__ void kfft_r2c_post_pipe(
    __gm__ float* out, __gm__ float* z, __gm__ float* wrrGm, __gm__ float* wriGm,
    uint32_t n, uint32_t batch)
{
    const uint32_t m      = n >> 1;              // 内层 FFT 长度
    const uint32_t mHalf  = m >> 1;              // pair 数上限（k=0..m/2）
    const uint32_t strideF = hsStride(n);        // 半谱行距
    const uint32_t twLen  = ((mHalf + 1u) + 7u) & ~7u;  // 旋转因子表按 8 倍数上取整

    TPipe pipe;
    // Ping-pong 双缓冲：两组 Z 行缓存交替使用
    TBuf<TPosition::VECCALC> bZ0, bZ1, bWr, bWi, bT, bTmp, bStg0, bStg1;
    pipe.InitBuffer(bZ0, ub32(n * sizeof(float)));
    pipe.InitBuffer(bZ1, ub32(n * sizeof(float)));
    pipe.InitBuffer(bWr, ub32(twLen * sizeof(float)));
    pipe.InitBuffer(bWi, ub32(twLen * sizeof(float)));
    // 表：tA(k) / tB(m-k) / tW(k) 各 twLen
    pipe.InitBuffer(bT, ub32(3u * twLen * sizeof(uint32_t)));
    pipe.InitBuffer(bTmp, ub32(16u * kChunk * sizeof(float)));
    // Forward/Reverse fragments（双缓冲）
    const uint32_t fragLen = ((mHalf + 1u) + 3u) & ~3u;  // 上取整到 4 复数槽
    pipe.InitBuffer(bStg0, ub32(4u * fragLen * sizeof(float)));
    pipe.InitBuffer(bStg1, ub32(4u * fragLen * sizeof(float)));

    LocalTensor<float> z0    = bZ0.Get<float>(), z1 = bZ1.Get<float>();
    LocalTensor<float> wrr   = bWr.Get<float>(), wri = bWi.Get<float>();
    LocalTensor<uint32_t> t0 = bT.Get<uint32_t>();
    LocalTensor<uint32_t> tA = t0,           tB = tA[twLen],       tW = tB[twLen];
    LocalTensor<float> tm    = bTmp.Get<float>();
    // 临时缓冲：ar, ai, br, bi, er, ei, ore, oim, tre, tim, wv0, wv1, p0, p1, oR, oI
    LocalTensor<float> ar = tm,        ai = ar[kChunk],   br = ai[kChunk],  bi = br[kChunk];
    LocalTensor<float> er = bi[kChunk], ei = er[kChunk],  ore = ei[kChunk], oim = ore[kChunk];
    LocalTensor<float> tre = oim[kChunk], tim = tre[kChunk];
    LocalTensor<float> wv0 = tim[kChunk], wv1 = wv0[kChunk];
    LocalTensor<float> p0  = wv1[kChunk], p1 = p0[kChunk];
    LocalTensor<float> oR  = p1[kChunk],  oI = oR[kChunk];
    // Forward/Reverse fragments（双缓冲）
    LocalTensor<float> stg0 = bStg0.Get<float>(), stg1 = bStg1.Get<float>();
    LocalTensor<float> fwd0 = stg0, rev0 = stg0[2u * fragLen];
    LocalTensor<float> fwd1 = stg1, rev1 = stg1[2u * fragLen];

    // ---- 表（批次循环外一次建好，全批共用）----
    // tA[k] = k 的字节偏移，tB[k] = (m-k) 的字节偏移，tW[k] = k 的 float 偏移
    // 注意：k=0 时 b=Z[0]（不是 Z[m]，因为 Z 只有 m 个元素）
    for (uint32_t k = 0; k <= mHalf; k++) {
        tA.SetValue(k, k * 8u);                    // Z[k] 的字节偏移（复数）
        tB.SetValue(k, (k == 0u) ? 0u : ((m - k) * 8u));  // Z[m-k] 的字节偏移（k=0 时用 Z[0]）
        tW.SetValue(k, k * 4u);                    // W[k] 的 float 偏移
    }
    GlobalTensor<float> gwrr, gwri;
    gwrr.SetGlobalBuffer(wrrGm, twLen);
    gwri.SetGlobalBuffer(wriGm, twLen);
    DataCopy(wrr, gwrr, twLen);
    DataCopy(wri, gwri, twLen);
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk  = GetBlockIdx();

    // 预取第一批数据到 ping 缓冲
    int64_t firstB = blk;
    if (firstB < (int64_t)batch) {
        GlobalTensor<float> gz;
        gz.SetGlobalBuffer(z + (uint32_t)firstB * n, n);
        DataCopy(z0, gz, n);
        PipeBarrier<PIPE_MTE2>();
    }

    // 流水线主循环：处理 batch[b] 时，预取 batch[b+nblk] 到另一个缓冲
    uint32_t iter = 0;  // 迭代计数器，用于 ping-pong 切换
    for (int64_t b = firstB; b < (int64_t)batch; b += nblk, iter++) {
        const uint32_t bufIdx = iter & 1;  // 0=ping, 1=pong
        LocalTensor<float> zr    = (bufIdx == 0) ? z0 : z1;
        LocalTensor<float> fwd   = (bufIdx == 0) ? fwd0 : fwd1;
        LocalTensor<float> rev   = (bufIdx == 0) ? rev0 : rev1;
        LocalTensor<float> zrNext = (bufIdx == 0) ? z1 : z0;

        const uint32_t rowOff = (uint32_t)b * strideF;

        // 等待上一批的 MTE3 完成（如果有的话），再预取下一批
        // 否则下一批的 MTE2 会与上一批的 MTE3 竞争同一 UB 缓冲
        if (iter > 0) {
            PipeBarrier<PIPE_ALL>();
        }

        // 预取下一批数据（与当前批的计算并行）
        // 注意：只预取到 zrNext，不覆盖当前批正在使用的 zr
        int64_t nextB = b + nblk;
        if (nextB < (int64_t)batch) {
            GlobalTensor<float> gzNext;
            gzNext.SetGlobalBuffer(z + (uint32_t)nextB * n, n);
            DataCopy(zrNext, gzNext, n);
            // 只等待 MTE2 完成，不等待 MTE3（流水线关键）
            PipeBarrier<PIPE_MTE2>();
        }

        GlobalTensor<float> gout;

        // ---- 一次遍历 k=0..mHalf，同时产出 X[k] 和 X[m-k] ----
        for (uint32_t k = 0; k <= mHalf; k += kChunk) {
            const uint32_t len  = ((mHalf + 1u - k) < kChunk) ? (mHalf + 1u - k) : kChunk;
            const uint32_t lenO = (len + 3u) & ~3u;   // DataCopy 上取整到 4 槽

            // Gather a = Z[k], b = Z[m-k]
            Gather(ar,  zr, tA[k], 0u, len);
            Gather(ai,  zr, tA[k], 4u, len);
            Gather(br,  zr, tB[k], 0u, len);
            Gather(bi,  zr, tB[k], 4u, len);
            // Gather twiddles W[k]
            Gather(wv0, wrr, tW[k], 0u, len);
            Gather(wv1, wri, tW[k], 0u, len);
            PipeBarrier<PIPE_V>();    // 等待 MTE2 完成，开始计算

            // E = (a + conj(b))/2
            Add(er, ar, br, len);  Muls(er, er, 0.5f, len);
            Sub(ei, ai, bi, len);  Muls(ei, ei, 0.5f, len);
            // O = (a - conj(b))/(2i)：O.re=(ai+bi)/2，O.im=(br-ar)/2
            Add(ore, ai, bi, len); Muls(ore, ore, 0.5f, len);
            Sub(oim, br, ar, len); Muls(oim, oim, 0.5f, len);
            // t = W * O
            Mul(p0, wv0, ore, len); Mul(p1, wv1, oim, len); Sub(tre, p0, p1, len);
            Mul(p0, wv0, oim, len); Mul(p1, wv1, ore, len); Add(tim, p0, p1, len);

            // X[k] = E + t（forward fragment，交错复数布局）
            Add(oR, er, tre, len);
            Add(oI, ei, tim, len);
            // 写入 forward fragment（交错复数）
            for (uint32_t i = 0; i < len; i++) {
                fwd.SetValue(2u * i,     oR.GetValue(i));
                fwd.SetValue(2u * i + 1u, oI.GetValue(i));
            }

            // X[m-k] = conj(E - t)（reverse fragment，交错复数布局）
            Sub(oR, er, tre, len);    // Re(E-t)
            Sub(oI, tim, ei, len);    // Im(conj(E-t))
            // 写入 reverse fragment：rev[kk] = X[m-kk]（kk=0..mHalf-1）
            for (uint32_t i = 0; i < len; i++) {
                const uint32_t kk = k + i;
                if (kk == 0) {
                    // k=0: X[m] = conj(E - t) = (Re(E-t), -Im(E-t))
                    // 写入 reverse 的开头（位置 0）
                    rev.SetValue(0u,    oR.GetValue(i));
                    rev.SetValue(1u,    0.0f);  // Nyquist 实数
                }
                else if (kk == mHalf) continue;  // k=m/2 自配对只写 forward
                else {
                    rev.SetValue(2u * kk,     oR.GetValue(i));
                    rev.SetValue(2u * kk + 1u, oI.GetValue(i));
                }
            }
            PipeBarrier<PIPE_V>();    // 等待计算完成，开始写回
        }

        // ---- 发布：两个连续 MTE3 DataCopy ----
        // 1. Forward fragment: X[0..mHalf] -> out[0..mHalf]
        gout.SetGlobalBuffer(out + rowOff, 2u * fragLen);
        DataCopy(gout, fwd, 2u * fragLen);

        // 2. Reverse fragment: X[m], X[m-1], ..., X[mHalf+1] -> out[mHalf+1..m]
        //    UB 中存储顺序：rev[0]=X[m], rev[1]=X[m-1], ..., rev[mHalf-1]=X[mHalf+1]
        //    GM 中需要正向连续：out[mHalf+1], out[mHalf+2], ..., out[m]
        //    需要反转后写出
        if (mHalf >= 1) {
            // 反转 reverse fragment（交错复数布局）
            // revLen = mHalf（包含 X[m] 和 X[m-1]..X[mHalf+1]）
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
        // 故意不加屏障：让本组的 MTE3 与下一组的 MTE2 重叠
        // （下一组的 MTE2 写 zrNext，与本组的 MTE3 读 fwd/rev 不冲突）
    }
}
