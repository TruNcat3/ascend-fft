// Phase 3++: radix-2 DIT complex fp32 FFT —— 8-plane 布局 + 全 Gather 化。
//
// 消融实测（n=4096, B=4096, 48 AIV，基线 17,347 us）：
//   planar->交错标量循环  9,790 us  56.4%   <- 本版用 Gather 消除
//   h>=8 planar 矢量级    6,436 us  37.1%
//   其余（转置/位反转/GM/启动）1,121 us   6.5%
//   位反转 Gather(2*n)      282 us  1.6%
// => 用 2n 项 Gather 索引把 planar->交错 也矢量化（dst[2j]=pr[j], dst[2j+1]=pi[j]），
//    pr/pi 同处一个 UB 缓冲，故一份索引即可同时寻址两段。
//
// 复数乘优化（本版 3 项，逐项实测，A/B 各 3×50 reps 取 min）：
//   C1  平面级：Muls+Axpy 合并，6 矢量算子 -> 4            3288 -> 2030.6 us 墙钟
//   C2  planar 合并路径用 BinaryRepeatParams 折 group 维
//   C2b planar 复数乘 6 -> 4 算子：
//       re = r1*tr - i1*ti = i1*(-ti) + r1*tr   -> Mul + MulAddDst
//       im = i1*tr + r1*ti                     -> Mul + MulAddDst
//       负号twiddle 一次性 Muls 到本级空闲的 t2 区（**不占额外 UB**），每级 1 次。
//       合并/非合并两路各单独 A/B：-0.9% + -0.9%，合计 2013 -> 1975 us（-1.9%）。
//   代价模型：device ≈ ops/batch × 86 × 12.1 ns + ~1167 us 下限
//       （下限 = scalar 16.8% ≈336us + mte2+mte3 ≈214us + 栅栏/gather）

//
// K-plane 布局（K = bfly::planeKFor(n)，在 {8,16,32} 中按算子调用数择优）：
//     pos(j) = (j & (K-1))*(n/K) + (j >> logK)
//   行基址 = a*(n/K) 且 n/K >= 8 => 恒 32B 对齐，因此**所有** stage h<K 的蝶形两端
//   落在同一列的不同行、基址天然对齐 => 可矢量化。
//   stage h 可平面化的条件是 2h | K，即 h <= K/2；每级恰好 K/2 对，每对一次长度 n/K 的算子。
// 算子调用量（微基准标定：每次矢量调用 ~13.3ns 固定开销，元素 ~0.023ns）：
//   平面级 h=1 为 5K、h>=2 为 7K/级，平面后 5n/h 次/级
//     n=4096: K=8 -> 5,262   K=16 -> 2,966   K=32 -> 2,326（择优）
//     n=256 : K=8 -> 462     （K=32 会涨到 1,126，故小 n 必须取小 K）
//   => 算子调用数是唯一有效杠杆（微基准：~11.6 ns/次 + 0.052 ns/元素）。
// 三张索引合成一个 4n 的 GM 张量（[idxB | idxT | idxOut]），全部在 batch 循环外一次
// DataCopy 进 UB（逐 batch 传 16KB 会多花 ~1us/batch，得不偿失）。
// 索引单位为字节：Gather 的 srcOffset/srcBaseAddr 实测均为字节。
//   idxB[k] = bitrev(invpos(k)) * 8,  invpos(k) = ((k%rows)<<logK)|(k/rows)
//   idxT[k] = pos(k) * 4              = ((k&(K-1))*rows + (k>>logK)) * 4
//   idxOut[2j] = 4*j   idxOut[2j+1] = 4*(n+j)
//
// UB: plan 8n + plane 8n + idxB 4n + idxT 4n + idxOut 8n + tw (4n+64)*2 + tmp 6.5n
//   = 46.5n + 128 bytes  (n=4096 -> 186.1 KiB < 192 KiB)
#include "butterfly/fft_k.hpp"
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
using namespace AscendC;

extern "C" __global__ __aicore__ __vector__ void kfft_fwd(__gm__ float* out, __gm__ float* in,
    uint32_t n, uint32_t batch, __gm__ float* twrGm, __gm__ float* twiGm,
    __gm__ uint32_t* idxGm, uint32_t foldD)
{
    uint32_t logn = 0;
    while ((1u << logn) < n) logn++;
    const uint32_t twPad = n + 16;
    const uint32_t hmax = n >> 1;
    // ---- 第 8 参打包：低 8 位 = foldD，高 8 位 = 平面 K（0 = 按 bfly::planeKFor 规则自选）----
    // 高 8 位是测试钩子（fft_check 的 AB_PLANE_K，也承载默认的 planeKFor(n)）——
    // 这样可以在不改 ABI 的前提下把 K 打到别处，验证 rows>64 的行切片 / K=32 等非默认分支。
    // 传 0 时内核自行按 bfly::planeKFor 规则取 K（与宿主一致）。
    // K 择优规则见 bfly::planeKFor（与宿主 fft_check 的索引生成、框架 Generator 共用）。
    const uint32_t K  = ((foldD >> 8) & 0xFFu) ? ((foldD >> 8) & 0xFFu) : bfly::planeKFor(n);
    // 第 16 位：旋转因子符号约定翻转（c2r 用 e^{+2πi·} 表）。
    // 融合 radix-4 的交叉旋转 ±i = W_{4h}^{h} 会随约定翻转成 ∓i，而 ±i 是写死在蝶形里的
    // （W、W²、W³ 都来自查找表、会自动共轭，唯独它不会）=> 翻转约定时把 s 取反即可，
    // y1/y3 的算子形态保持不变（q+i*(-s) = q-i*s）。
    const bool xflip = ((foldD >> 16) & 0x1u) != 0u;
    uint32_t logK = 0;
    while ((1u << logK) < K) logK++;
    const uint32_t rows = n / K;       // 每个 plane 的长度（= 列长）

    // ---- 批折叠：D 个连续 batch 拍进同一个 repeat（g = min(D, batch-b) 是本组实际批数）----
    // 三条不变量，保证 D=1 时与折叠前**逐字节等价**（尺寸、地址、算子全部退化成原样）：
    //   1. 分区布局不变：plan = [re_all|im_all]、plane = [re_all|im_all]，
    //      只是区域由 n 放大到 n*D；pr = plan、pi = plan + n*D（D=1 时 pi = pr[n]）
    //   2. idxB / idxT 两张索引表完全不改（偏移都在各自 [0,n) 区内，与 D 无关）
    //   3. Level-0 的 mask/repeat 在 g=1 时与 Level-2 count 模式语义等价
    const uint32_t D = (foldD & 0xFFu) ? (foldD & 0xFFu) : 1u;
    // 批步长（32B 单位，uint8 上限 255）：ar/ai 走 n、t* 走 rows；repeat=1 时该字段不被读，
    // 但仍然钳到 255，避免 n=2048 时 n/8=256 溢出成 0。
    const uint16_t arRep = (uint16_t)((n >> 3) > 255u ? 255u : (n >> 3));
    const uint16_t tRep  = (uint16_t)((rows >> 3) > 255u ? 255u : (rows >> 3));

    TPipe pipe;
    TBuf<TPosition::VECCALC> bPlan, bPlane, bIdxB, bIdxT, bIdxO, bTwR, bTwI, bTmp;
    pipe.InitBuffer(bPlan,  2 * n * D * sizeof(float));
    pipe.InitBuffer(bPlane, 2 * n * D * sizeof(float));
    pipe.InitBuffer(bIdxB,    n * sizeof(uint32_t));
    pipe.InitBuffer(bIdxT,    n * sizeof(uint32_t));
    pipe.InitBuffer(bIdxO, 2 * n * D * sizeof(uint32_t));
    pipe.InitBuffer(bTwR,  twPad * sizeof(float));
    pipe.InitBuffer(bTwI,  twPad * sizeof(float));
    // tmp 需求 = max(plane 段, planar 段)：
    //   plane 段：t0..t5 各 rows*D，但 t0/t1 的批基址是 hmax*D、t2..t5 基址是
    //             2hmax*D + t2sz 起顺排 3 个 rows*D  => 2hmax*D + t2sz + 3rows*D
    //   planar 段：t0/t1 各 hmax*D、t2 只需 hmax（-ti 复用，跨批共享） => 2hmax*D + hmax
    //             <= plane 段（因 t2sz >= hmax、3rows*D >= 0），故取 plane 段即可。
    // D=1（rows < hmax => t2sz = hmax）=> 3hmax + 3rows，与折叠前的 InitBuffer 同值。
    const uint32_t t2sz = (hmax > rows * D) ? hmax : (rows * D);
    const uint32_t tmpF = 2u * hmax * D + t2sz + 3u * rows * D;
    pipe.InitBuffer(bTmp, tmpF * sizeof(float));

    LocalTensor<float> plan = bPlan.Get<float>();
    LocalTensor<float> pr = plan, pi = pr[n * D];
    LocalTensor<float> ar = bPlane.Get<float>(), ai = ar[n * D];
    LocalTensor<uint32_t> idxB = bIdxB.Get<uint32_t>();
    LocalTensor<uint32_t> idxT = bIdxT.Get<uint32_t>();
    LocalTensor<uint32_t> idxO = bIdxO.Get<uint32_t>();
    LocalTensor<float> twr = bTwR.Get<float>(), twi = bTwI.Get<float>();
    LocalTensor<float> t0 = bTmp.Get<float>(), t1 = t0[hmax * D], t2 = t1[hmax * D];
    LocalTensor<float> t3 = t2[t2sz];
    LocalTensor<float> t4 = t3[rows * D], t5 = t4[rows * D];  // radix-4 融合需要 6 个临时区

    GlobalTensor<float> gtwr, gtwi;
    gtwr.SetGlobalBuffer(twrGm, twPad);
    gtwi.SetGlobalBuffer(twiGm, twPad);
    DataCopy(twr, gtwr, twPad);
    DataCopy(twi, gtwi, twPad);

    GlobalTensor<uint32_t> gidx;
    gidx.SetGlobalBuffer(idxGm, 2u * n + 2u * n * D);  // [idxB n | idxT n | idxO 2nD]
    DataCopy(idxB, gidx, n);                 // [0, n)
    DataCopy(idxT, gidx[n], n);              // [n, 2n)
    DataCopy(idxO, gidx[2 * n], 2u * n * D); // [2n, 2n+2nD)   D=1 -> 与旧 4n 表同
    // 【A6】这里必须留屏障，且**不能**并入下面每组的 DataCopy(plan) 屏障：
    //   ① 本轮实测并入后 Task Duration 298.0→298.1 µs、aiv_vec 0.825→0.825，收益 < 噪声；
    //   ② batch 很小（如 1）时大量 block 不进循环体、直接返回，本屏障是它们唯一的 UB
    //      排空点 —— 撤掉会让上一次 launch 在途的 UB 写跨任务污染下一次 launch。
    //      「省 1 次 drain」与「跨 launch UB 竞争」相比，后者风险更高，故保留。
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk = GetBlockIdx();

    // 把 D 个连续 batch 收成一组处理：组内共享一次 MTE2 装载、一次屏障、一次 MTE3 写回。
    // 尾组 g < D 走同一套指令（Level-0 在 g=1 时与 Level-2 count 等价）。
    //
    // 分组必须按「组号 gk 取模」分配到 block，**不能**按 b += D*nblk：
    //   后者在 batch 未铺满 nblk*D 时会重叠（batch=3,D=4：blk0 覆盖 0..2，blk1 又从 1 起）。
    //   gk = 0..ceil(batch/D)-1，组起点 b = gk*D，每组恰好覆盖 [b, b+D)，
    //   block blk 认领 gk ≡ blk (mod nblk) => 每个 batch 恰好一次。
    // D=1 时 gk*D = b、g=1，与折叠前的 `b = blk; b += nblk` 逐点相同。
    for (int64_t gk = blk;
         gk * (int64_t)D < (int64_t)batch;
         gk += nblk) {
        const int64_t b = gk * (int64_t)D;
        const int64_t rem = (int64_t)batch - b;
        const uint32_t g = (rem < (int64_t)D) ? (uint32_t)rem : D;
        uint32_t goff = (uint32_t)b * 2u * n;
        GlobalTensor<float> gin, gout;
        gin.SetGlobalBuffer(in + goff, 2u * n * g);
        gout.SetGlobalBuffer(out + goff, 2u * n * g);
        DataCopy(plan, gin, 2u * n * g);     // 组内 g 批的交错输入一次装完
        PipeBarrier<PIPE_ALL>();

        // 位反转 + 去交错 + 转 plane：ar[k]=re_in[bitrev(invpos(k))]；im 用 srcBaseAddr=4 字节
        // idxB 与 D 无关：批维只体现在目标地址 ar+d*n、源地址 plan+2n*d
        for (uint32_t d = 0; d < g; d++) {
            LocalTensor<float> pln = plan[2u * n * d];
            Gather(ar[d * n], pln, idxB, 0u, n);
            Gather(ai[d * n], pln, idxB, 4u, n);
        }
        // 矢量管内顺序执行：Gather(vgather) 与后面的平面级算子同在 vector 管，无需 PIPE_ALL

        // ---- stage 对 (h, 2h) 融合成 radix-4：K-plane 布局 ----
        //   记 W = slot_{2h}[j2] = e^{-i*pi*j2/(2h)}，则 slot_h[j2] = W^2（宿主表不变，
        //   仍按 si 升序消耗 align8(h)+align8(2h)），W^3 = W*W^2 在核内标量算。
        //   两阶段组合（已代数核对）：p=x0+W^2 x1, q=x0-W^2 x1, u2=W x2, u3=W^3 x3
        //       y0 = p+r,  y2 = p-r          (r = u2+u3)
        //       y1 = q-i*s, y3 = q+i*s        (s = u2-u3)
        //   i 是交叉旋转：实/虚分张量 => y1/y3 只需跨张量换位，0 算子。
        //   j2==0 => W=W^2=W^3=1 => 16 算子；否则 28 算子（3 复乘 12 + 蝶形 16）。
        uint32_t slot = 0;
        uint32_t si = 0;
        // plane 段两条路径：
        //   useL0 = g>1 且 n/8<=255（arRep=n/8 是 uint8 批步长，n>=2048 时溢出）——
        //           Level-0 mask=rows/repeat=g 把 g 批拍进同一 repeat。
        //           Level-0 单 repeat 的 mask 硬顶 64 元素，故 rows>64 时按 64 元素**行切片**
        //           逐片重放（与 planar 段的 h>64 切片同法）；rows<=64 时 nRowSlice=1，
        //           切片循环退化成一层 ⇒ **与不切片逐字节等价**。
        //   否则（g==1，或 n>=2048）走原 Level-2 count 路径 + 逐批 for 循环；
        //           g==1 时 d 恒为 0，所有偏移退化成原式 ⇒ **与折叠前逐字节等价**。
        //   host 的 bfly::foldDFor 与这里的两条判据同式（A5 把 rows 门槛换成 arRep 门槛）。
        const bool useL0 = (g > 1u) && ((n >> 3) <= 255u);
        const uint32_t nRowSlice = (rows + 63u) >> 6;   // rows<=64 => 1（零开销）
        // Level-0 批折叠的 repeat 参数（dstBlk/srcBlk 恒 1，rep 步长单位 32B）
        const UnaryRepeatParams uTA(1, 1, tRep, arRep);              // dst=t*  src=ar
        const BinaryRepeatParams bAAT(1, 1, 1, arRep, arRep, tRep);  // dst=ar  src0=ar  src1=t*
        const BinaryRepeatParams bATT(1, 1, 1, arRep, tRep, tRep);   // dst=ar  src0=t*  src1=t*
        const BinaryRepeatParams bAAA(1, 1, 1, arRep, arRep, arRep); // dst=src0=src1=ar
        const BinaryRepeatParams bTTA(1, 1, 1, tRep, arRep, arRep);  // dst=t*  src0=src1=ar
        const BinaryRepeatParams bTTT(1, 1, 1, tRep, tRep, tRep);    // dst=src0=src1=t*
        for (uint32_t d = 0; d < g; d++) {
        slot = 0;   // twiddle slot 消费与 batch 无关，逐批复位
        si = 0;
        const uint32_t bOff = d * n;     // ar/ai 的批偏移
        const uint32_t tOff = d * rows;  // t0..t5 的批偏移
        for (; si + 1u < logK; si += 2u) {
            const uint32_t h = 1u << si;
            const uint32_t slW2 = slot; slot += (h + 7u) & ~7u;        // slot_h   -> W^2
            const uint32_t slW  = slot; slot += (2u * h + 7u) & ~7u;   // slot_{2h}-> W
            for (uint32_t m = 0; m < (K / (4u * h)); m++) {
                for (uint32_t j2 = 0; j2 < h; j2++) {
                    const uint32_t b = 4u * h * m + j2;
                    if (useL0) {
                    LocalTensor<float> r0 = ar[b * rows],          i0 = ai[b * rows];
                    LocalTensor<float> r1 = ar[(b + h) * rows],    i1 = ai[(b + h) * rows];
                    LocalTensor<float> r2 = ar[(b + 2u * h) * rows], i2 = ai[(b + 2u * h) * rows];
                    LocalTensor<float> r3 = ar[(b + 3u * h) * rows], i3 = ai[(b + 3u * h) * rows];

                    if (j2 != 0) {
                        const float w2r = twr.GetValue(slW2 + j2), w2i = twi.GetValue(slW2 + j2);
                        const float wr  = twr.GetValue(slW  + j2), wi  = twi.GetValue(slW  + j2);
                        const float w3r = wr * w2r - wi * w2i;
                        const float w3i = wr * w2i + wi * w2r;
                        for (uint32_t slc = 0; slc < nRowSlice; slc++) {
                        const uint32_t off = slc << 6;
                        const uint32_t msk = ((rows - off) > 64u) ? 64u : (rows - off);
                        LocalTensor<float> a0 = r0[off], a1 = r1[off], a2 = r2[off], a3 = r3[off];
                        LocalTensor<float> c0 = i0[off], c1 = i1[off], c2 = i2[off], c3 = i3[off];
                        LocalTensor<float> u0 = t0[off], u1 = t1[off], u2 = t2[off];
                        LocalTensor<float> u3 = t3[off], u4 = t4[off], u5 = t5[off];
                        // u1 = W^2 * x1  (复乘 4 算子：Muls + Axpy)
                        Muls(u0, a1, w2r, msk, g, uTA); Axpy(u0, c1, -w2i, msk, g, uTA);
                        Muls(u1, a1, w2i, msk, g, uTA); Axpy(u1, c1,  w2r, msk, g, uTA);
                        // u2 = W * x2
                        Muls(u2, a2, wr,  msk, g, uTA); Axpy(u2, c2, -wi,  msk, g, uTA);
                        Muls(u3, a2, wi,  msk, g, uTA); Axpy(u3, c2,  wr,  msk, g, uTA);
                        // u3 = W^3 * x3
                        Muls(u4, a3, w3r, msk, g, uTA); Axpy(u4, c3, -w3i, msk, g, uTA);
                        Muls(u5, a3, w3i, msk, g, uTA); Axpy(u5, c3,  w3r, msk, g, uTA);
                        // q = x0-u1 -> x1 ; p = x0+u1 -> x0   (u0/u1 仍是 u1)
                        Sub(a1, a0, u0, msk, g, bAAT); Sub(c1, c0, u1, msk, g, bAAT);
                        Add(a0, a0, u0, msk, g, bAAT); Add(c0, c0, u1, msk, g, bAAT);
                        // s = u2-u3 -> u0/u1 ；xflip（+i 约定）时交叉旋转 ∓i => s 取反：
                        // 直接交换减数/被减数（Level-0 就地矢量算子在批折叠下不可靠，且省 2 算子）
                        if (!xflip) { Sub(u0, u2, u4, msk, g, bTTT); Sub(u1, u3, u5, msk, g, bTTT); }
                        else        { Sub(u0, u4, u2, msk, g, bTTT); Sub(u1, u5, u3, msk, g, bTTT); }
                        Add(u2, u2, u4, msk, g, bTTT); Add(u3, u3, u5, msk, g, bTTT);
                        // y2 = p-r -> x2 ; y0 = p+r -> x0
                        Sub(a2, a0, u2, msk, g, bAAT); Sub(c2, c0, u3, msk, g, bAAT);
                        Add(a0, a0, u2, msk, g, bAAT); Add(c0, c0, u3, msk, g, bAAT);
                        // y3 = q+i*s -> x3 ; y1 = q-i*s -> x1
                        Sub(a3, a1, u1, msk, g, bAAT); Add(c3, c1, u0, msk, g, bAAT);
                        Add(a1, a1, u1, msk, g, bAAT); Sub(c1, c1, u0, msk, g, bAAT);
                        }
                    } else {
                        for (uint32_t slc = 0; slc < nRowSlice; slc++) {
                        const uint32_t off = slc << 6;
                        const uint32_t msk = ((rows - off) > 64u) ? 64u : (rows - off);
                        LocalTensor<float> a0 = r0[off], a1 = r1[off], a2 = r2[off], a3 = r3[off];
                        LocalTensor<float> c0 = i0[off], c1 = i1[off], c2 = i2[off], c3 = i3[off];
                        LocalTensor<float> u0 = t0[off], u1 = t1[off], u2 = t2[off];
                        LocalTensor<float> u3 = t3[off], u4 = t4[off], u5 = t5[off];
                        Sub(u0, a0, a1, msk, g, bTTA); Sub(u1, c0, c1, msk, g, bTTA); // q
                        Add(a0, a0, a1, msk, g, bAAA); Add(c0, c0, c1, msk, g, bAAA); // p
                        if (!xflip) { Sub(u2, a2, a3, msk, g, bTTA); Sub(u3, c2, c3, msk, g, bTTA); } // s
                        else        { Sub(u2, a3, a2, msk, g, bTTA); Sub(u3, c3, c2, msk, g, bTTA); } // s 取反
                        Add(u4, a2, a3, msk, g, bTTA); Add(u5, c2, c3, msk, g, bTTA); // r
                        Sub(a2, a0, u4, msk, g, bAAT); Sub(c2, c0, u5, msk, g, bAAT); // y2
                        Add(a0, a0, u4, msk, g, bAAT); Add(c0, c0, u5, msk, g, bAAT); // y0
                        Sub(a3, u0, u3, msk, g, bATT); Add(c3, u1, u2, msk, g, bATT); // y3
                        Add(a1, u0, u3, msk, g, bATT); Sub(c1, u1, u2, msk, g, bATT); // y1
                        }
                    }
                    } else {  // Level-2 count 路径（g==1 或 n>=2048）
                    LocalTensor<float> r0 = ar[bOff + b * rows],       i0 = ai[bOff + b * rows];
                    LocalTensor<float> r1 = ar[bOff + (b + h) * rows], i1 = ai[bOff + (b + h) * rows];
                    LocalTensor<float> r2 = ar[bOff + (b + 2u * h) * rows], i2 = ai[bOff + (b + 2u * h) * rows];
                    LocalTensor<float> r3 = ar[bOff + (b + 3u * h) * rows], i3 = ai[bOff + (b + 3u * h) * rows];

                    if (j2 != 0) {
                        const float w2r = twr.GetValue(slW2 + j2), w2i = twi.GetValue(slW2 + j2);
                        const float wr  = twr.GetValue(slW  + j2), wi  = twi.GetValue(slW  + j2);
                        const float w3r = wr * w2r - wi * w2i;
                        const float w3i = wr * w2i + wi * w2r;
                        // u1 = W^2 * x1  (复乘 4 算子：Muls + Axpy)
                        Muls(t0[tOff], r1, w2r, rows); Axpy(t0[tOff], i1, -w2i, rows);
                        Muls(t1[tOff], r1, w2i, rows); Axpy(t1[tOff], i1,  w2r, rows);
                        // u2 = W * x2
                        Muls(t2[tOff], r2, wr,  rows); Axpy(t2[tOff], i2, -wi,  rows);
                        Muls(t3[tOff], r2, wi,  rows); Axpy(t3[tOff], i2,  wr,  rows);
                        // u3 = W^3 * x3
                        Muls(t4[tOff], r3, w3r, rows); Axpy(t4[tOff], i3, -w3i, rows);
                        Muls(t5[tOff], r3, w3i, rows); Axpy(t5[tOff], i3,  w3r, rows);
                        // q = x0-u1 -> x1 ; p = x0+u1 -> x0   (t0/t1 仍是 u1)
                        Sub(r1, r0, t0[tOff], rows); Sub(i1, i0, t1[tOff], rows);
                        Add(r0, r0, t0[tOff], rows); Add(i0, i0, t1[tOff], rows);
                        // s = u2-u3 -> t0/t1 ；xflip 交换减数/被减数取反 s；r = u2+u3 -> t2/t3
                        if (!xflip) { Sub(t0[tOff], t2[tOff], t4[tOff], rows); Sub(t1[tOff], t3[tOff], t5[tOff], rows); }
                        else        { Sub(t0[tOff], t4[tOff], t2[tOff], rows); Sub(t1[tOff], t5[tOff], t3[tOff], rows); }
                        Add(t2[tOff], t2[tOff], t4[tOff], rows); Add(t3[tOff], t3[tOff], t5[tOff], rows);
                        // y2 = p-r -> x2 ; y0 = p+r -> x0
                        Sub(r2, r0, t2[tOff], rows); Sub(i2, i0, t3[tOff], rows);
                        Add(r0, r0, t2[tOff], rows); Add(i0, i0, t3[tOff], rows);
                        // y3 = q+i*s -> x3 ; y1 = q-i*s -> x1
                        Sub(r3, r1, t1[tOff], rows); Add(i3, i1, t0[tOff], rows);
                        Add(r1, r1, t1[tOff], rows); Sub(i1, i1, t0[tOff], rows);
                    } else {
                        Sub(t0[tOff], r0, r1, rows); Sub(t1[tOff], i0, i1, rows);   // q -> t0/t1
                        Add(r0, r0, r1, rows); Add(i0, i0, i1, rows);               // p -> x0
                        if (!xflip) { Sub(t2[tOff], r2, r3, rows); Sub(t3[tOff], i2, i3, rows); } // s -> t2/t3
                        else        { Sub(t2[tOff], r3, r2, rows); Sub(t3[tOff], i3, i2, rows); } // s 取反
                        Add(t4[tOff], r2, r3, rows); Add(t5[tOff], i2, i3, rows);   // r -> t4/t5
                        Sub(r2, r0, t4[tOff], rows); Sub(i2, i0, t5[tOff], rows);   // y2
                        Add(r0, r0, t4[tOff], rows); Add(i0, i0, t5[tOff], rows);   // y0
                        Sub(r3, t0[tOff], t3[tOff], rows); Add(i3, t1[tOff], t2[tOff], rows);   // y3
                        Add(r1, t0[tOff], t3[tOff], rows); Sub(i1, t1[tOff], t2[tOff], rows);   // y1
                    }
                    }
                }
            }
        }
        // 残余奇数级（K=8 -> logK=3、K=32 -> logK=5）保留原 radix-2 路径
        if (si < logK) {
            const uint32_t h = 1u << si;
            const uint32_t sl = slot;
            slot += (h + 7u) & ~7u;
            for (uint32_t m = 0; m < (K / (2u * h)); m++) {
                for (uint32_t j2 = 0; j2 < h; j2++) {
                    const uint32_t p = 2u * h * m + j2;
                    const uint32_t q = p + h;
                    const float wr = twr.GetValue(sl + j2);
                    const float wi = twi.GetValue(sl + j2);
                    if (useL0) {
                    LocalTensor<float> rp = ar[p * rows], rq = ar[q * rows];
                    LocalTensor<float> ip = ai[p * rows], iq = ai[q * rows];

                    if (wr == 1.0f && wi == 0.0f) {
                        // 平凡复乘（w=1）：DataCopy 无 repeat 形态，逐批 g 次
                        // （g=1 时与原式同）。Level-2 count 形态没有 64 上限，
                        // 一次拷完整行 ⇒ 不需要按 64 切片。
                        for (uint32_t dd = 0; dd < g; dd++) {
                            DataCopy(t2[dd * rows], ar[dd * n + q * rows], rows);
                            DataCopy(t3[dd * rows], ai[dd * n + q * rows], rows);
                        }
                    }
                    // 矢量部分按 64 元素行切片（rows<=64 时 nRowSlice=1，退化成原式）
                    for (uint32_t slc = 0; slc < nRowSlice; slc++) {
                        const uint32_t off = slc << 6;
                        const uint32_t msk = ((rows - off) > 64u) ? 64u : (rows - off);
                        LocalTensor<float> ap = rp[off], aq = rq[off];
                        LocalTensor<float> bp = ip[off], bq = iq[off];
                        LocalTensor<float> c2 = t2[off], c3 = t3[off];
                        if (wr != 1.0f || wi != 0.0f) {   // 复数乘 high *= w：6 矢量算子 -> 4（Axpy 读改写合并）
                            Muls(c2, aq, wr, msk, g, uTA);
                            Axpy(c2, bq, -wi, msk, g, uTA);       // c2 = rq*wr - iq*wi
                            Muls(c3, aq, wi, msk, g, uTA);
                            Axpy(c3, bq, wr, msk, g, uTA);        // c3 = rq*wi + iq*wr
                        }
                        Sub(aq, ap, c2, msk, g, bAAT);
                        Add(ap, ap, c2, msk, g, bAAT);
                        Sub(bq, bp, c3, msk, g, bAAT);
                        Add(bp, bp, c3, msk, g, bAAT);
                    }
                    } else {
                    LocalTensor<float> rp = ar[bOff + p * rows], rq = ar[bOff + q * rows];
                    LocalTensor<float> ip = ai[bOff + p * rows], iq = ai[bOff + q * rows];

                    if (wr != 1.0f || wi != 0.0f) {   // 复数乘 high *= w：6 矢量算子 -> 4（Axpy 读改写合并）
                        Muls(t2[tOff], rq, wr, rows);
                        Axpy(t2[tOff], iq, -wi, rows);       // t2 = rq*wr - iq*wi
                        Muls(t3[tOff], rq, wi, rows);
                        Axpy(t3[tOff], iq, wr, rows);        // t3 = rq*wi + iq*wr
                    } else {
                        DataCopy(t2[tOff], rq, rows);
                        DataCopy(t3[tOff], iq, rows);
                    }
                    Sub(rq, rp, t2[tOff], rows);
                    Add(rp, rp, t2[tOff], rows);
                    Sub(iq, ip, t3[tOff], rows);
                    Add(ip, ip, t3[tOff], rows);
                    }
                }
            }
        }
        if (useL0) break;   // useL0 路径一次把 g 批全做完，退出逐批循环
        }   // for d

        // ---- plane -> planar 转置（Gather；此时 plan 内的输入已用尽，可就地覆写）----
        for (uint32_t d = 0; d < g; d++) {
            Gather(pr[d * n], ar[d * n], idxT, 0u, n);
            Gather(pi[d * n], ai[d * n], idxT, 0u, n);
        }

        // ---- stage h >= K：planar 矢量路径（h%8==0 => 高半区 8 对齐）----
        // 同一级内 groups 之间是固定等差步长（pr/pi 走 2h、twiddle 走 h、临时区走 h），
        // 因此可用 Level-0 的 repStride 把整个 group 维度折进 repeatTime，
        // 每级 10 次矢量调用而不是 10*groups 次。fp32 单 repeat = 64 元素（256B/4B），
        // 故 h>64 时按 64 元素切片、逐片重放 base。收益在 groups 很大的 h=32/64/128/256。
        //
        // 【批折叠 A3】把 repeat 由 groups 换成 `g*groups` 即可，pPV/pP 三个 stride 字段
        // **一个都不用改** —— 拍平下标 r = d*groups + gi 后：
        //   src0(pr/pi) group 步长 2h、批步长 groups*2h = n   => pr_d = plan + d*n ✓
        //   dst(t0/t1)  group 步长 h  、批步长 groups*h  = n/2 => t0_d = t0 + d*hmax ✓
        //   src1(tw/t2) 两维都跨组共享，src1Rep = 0           => 不变 ✓
        // t2(-ti) 是 Level-2 count、整组只跑 1 次（原来每批 1 次），顺带省 g-1 次。
        for (uint32_t s = logK; s < logn; s++) {
            const uint32_t h = 1u << s;
            const uint32_t groups = n / (2u * h);
            const uint32_t sl = slot;
            slot += (h + 7u) & ~7u;
            // 折组（把 groups 维折进 repeatTime）的成立条件：
            //   BinaryRepeatParams 的 *RepStride 是 uint8_t  =>  pSt=2h/8 <= 255 (h<=1020)、
            //   vSt=h/8 <= 255 (h<=2040)；repeat=g*groups <= 255；每片 msk<=64。
            //   g==1 时 `g*groups <= 255` 与 `groups <= 255` 同式、`nSlice <= groups || g>1`
            //   退化成 `nSlice <= groups` ⇒ **与折叠前逐字节同条件**。
            //   g>1 时放宽 nSlice 约束：切片路每组 (1+8*nSlice) 条指令服务 g 批，
            //   恒不劣于逐组循环。
            const uint32_t nSlice = (h + 63u) >> 6;
            const uint32_t rep = (uint32_t)g * groups;
            const bool merge = (h >= 8u) && (((2u * h) % 8u) == 0u) &&
                               ((2u * h) / 8u <= 255u) && (h / 8u <= 255u) &&
                               (rep <= 255u) && (nSlice <= groups || g > 1u);
            if (!merge) {
                LocalTensor<float> tr = twr[sl];
                LocalTensor<float> ti = twi[sl];
                Muls(t2, ti, -1.0f, h);          // t2 = -ti（跨批共享，整组 1 次）
                for (uint32_t dd = 0; dd < g; dd++) {
                    LocalTensor<float> prd = pr[dd * n], pid = pi[dd * n];
                    LocalTensor<float> t0d = t0[dd * hmax], t1d = t1[dd * hmax];
                    for (uint32_t gi = 0; gi < groups; gi++) {
                        const uint32_t base = gi * 2u * h;
                        LocalTensor<float> r0 = prd[base], r1 = prd[base + h];
                        LocalTensor<float> i0 = pid[base], i1 = pid[base + h];
                        // re = r1*tr - i1*ti = i1*(-ti) + r1*tr   (Mul + MulAddDst = 2 op)
                        Mul(t0d, i1, t2, h);
                        MulAddDst(t0d, r1, tr, h);
                        // im = i1*tr + r1*ti                     (Mul + MulAddDst = 2 op)
                        Mul(t1d, i1, tr, h);
                        MulAddDst(t1d, r1, ti, h);
                        Sub(r1, r0, t0d, h);
                        Add(r0, r0, t0d, h);
                        Sub(i1, i0, t1d, h);
                        Add(i0, i0, t1d, h);
                    }
                }
                continue;
            }
            const uint32_t pSt = (2u * h) >> 3;   // pr/pi 的 group 间步长（32B 块）
            const uint32_t vSt = h >> 3;          // t*/twiddle 的 group 间步长（32B 块）
            const BinaryRepeatParams pPV(1, 1, 1, vSt, pSt, 0);  // dst=t*, src0=r*, src1=tw*(与 g 无关)
            const BinaryRepeatParams pP(1, 1, 1, pSt, pSt, vSt);  // dst/src0=r*, src1=t*
            Muls(t2, twi[sl], -1.0f, h);          // vneg = -ti（t2 在合并路径中已无他用）
            for (uint32_t k = 0; k < nSlice; k++) {
                const uint32_t off = k << 6;
                const uint32_t msk = ((h - off) > 64u) ? 64u : (h - off);
                LocalTensor<float> tr = twr[sl + off];
                LocalTensor<float> ti = twi[sl + off];
                LocalTensor<float> r0 = pr[off], r1 = pr[h + off];
                LocalTensor<float> i0 = pi[off], i1 = pi[h + off];
                LocalTensor<float> a0 = t0[off], a1 = t1[off];
                Mul(a0, i1, t2[off], msk, rep, pPV);
                MulAddDst(a0, r1, tr, msk, rep, pPV);   // re
                Mul(a1, i1, tr, msk, rep, pPV);
                MulAddDst(a1, r1, ti, msk, rep, pPV);   // im
                Sub(r1, r0, a0, msk, rep, pP);
                Add(r0, r0, a0, msk, rep, pP);
                Sub(i1, i0, a1, msk, rep, pP);
                Add(i0, i0, a1, msk, rep, pP);
            }
        }

        // ---- planar -> 交错输出：单次 2n 项 Gather（替代 4n 次标量 SetValue，实测占 56%）----
        // 【批折叠 A3】idxO 按 D 重生成：idxO'[d*2n + 2j] = 4(d*n + j)（pr_d）、
        // idxO'[d*2n + 2j+1] = 4(nD + d*n + j)（pi_d）。D=1 退化成旧表 4j / 4(n+j)。
        // 逐 d Gather 到 ar 的不同段，一次屏障，再逐 d 写 GM（屏障数与折叠前按批均摊相同）。
        for (uint32_t d = 0; d < g; d++) {
            Gather(ar[2u * n * d], plan, idxO[2u * n * d], 0u, 2u * n);
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t d = 0; d < g; d++) {
            DataCopy(gout[2u * n * d], ar[2u * n * d], 2u * n);
        }
        // 故意不加屏障：让本组的 MTE3(读 ar 写 GM) 与下一组的 MTE2(GM->plan) 重叠。
        // ar 与 plan 是不同 UB，且下一组的矢量写 ar 会被下一轮 MTE2 之后的 PIPE_ALL 挡住
        // （PIPE_ALL 会把在途的 MTE3 一起排空），故无写读竞争。
    }
}
