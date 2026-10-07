// 实数变换扩展内核：r2c 后处理 / c2r Hermitian 展开 / c2r 实部提取。
//
// 组合方式（宿主 src/host/fft_check.cpp 按 AB_DIR 发射，三内核同签名 p,p,p,p,u,u = 40B）：
//   r2c : kfft_fwd(n/2) 复用实输入字节 -> kfft_r2c_post            （2 发射）
//   c2r : kfft_c2r_prep -> kfft_fwd(n) 旋转因子取反 -> kfft_c2r_post（3 发射）
//   c2r 的反号旋转因子让 kfft_fwd 直接算 G(X)=Σ X_k e^{+2πijk/n}=n·x（位反转/蝶形与符号无关，
//   只有旋转因子符号起作用），顺带免费得到一个逆复数 FFT 能力。
//
// ---- r2c 数学（x 长 n 实数，m=n/2，z[k]=x[2k]+i·x[2k+1]，Z=FFT_m(z)）----
//   对 k=0..m/2：a=Z[k]，b=Z[(m-k)%m]，E=(a+conj b)/2，O=(a-conj b)/(2i)，t=W_n^k·O，
//   则 out[k]=E+t、out[m-k]=conj(E-t)（k=m/2 自配对只写一次 out[m/2]=E+t）。
//   内核按「槽 s ∈ [0, m]」统一遍历：idx = (s<=kMax ? s : m-s)（kMax=n/4），
//   b = (idx==0 ? 0 : m-idx)，W 取 idx，两段分支只差最后一步（L: E+t；U: conj(E-t)）。
//   公式在 Python 里对 n∈{64..4096} 与 numpy.fft.rfft 逐点核对过（err ~3e-16）。
//
// ---- 布局约束（本 SoC 探针实测）----
//   DataCopy 长度**静默截断到 32B（8 个 float）倍数**，起始地址只需 4B 对齐 =>
//   半谱行距取 n+16 float（= m+8 个复数槽；有效槽 [0..m]，其余 7 槽给「上取整 ≤3 槽」
//   的越界写兜底，越界不会跨出行）。矢量算子/Gather 任意长度均忠实（4/12/36 已验证）。
//   Gather 的 dst/idx 视图必须 32B 对齐（+4B 视图直接 507035），src 只需 4B 对齐；
//   Scatter 在本 SoC 是 NOT_SUPPORT 空壳。故 UB 分区与表分区一律上取整到 8 元素倍数，
//   分块只在「相对表下标 ≡0 (mod 8)」处切片。
//
// ---- UB 预算（n=4096 最坏）----
//   r2c_post: z 16KB + wr/wi 2×4.1KB + 表 32.8KB + 临时 64KB + 交错双缓冲 16KB ≈ 137KB
//   prep/post: < 60KB
#include "butterfly/fft_k.hpp"
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
using namespace AscendC;

// 每块处理的复数槽数（4 的倍数 => DataCopy 计数 2C 为 8 倍数）
static const uint32_t kChunk = 1024u;

// 半谱行距（r2c 输出 / c2r 输入），单位 float。
// ccec 下 __global__ 调用的自由函数必须带 [aicore]（见 butterfly/fft_k.hpp 头注）。
BFLY_AICORE inline uint32_t hsStride(uint32_t n) { return n + 16u; }

// UB 分区大小上取整到 32B：TBuf 依次排布，奇数个 uint32 会让后面的分区起始地址
// 失去 32B 对齐 -> 矢量算子直接抛 507035（本 SoC 实测）。
BFLY_AICORE inline uint32_t ub32(uint32_t bytes) { return (bytes + 31u) & ~31u; }

// ---------------------------------------------------------------- r2c 后处理
// out  : [batch][hsStride(n)] 半谱（前 n/2+1 个复数有效）
// z    : [batch][n]           内层 FFT 输出（n/2 个复数，交错）
// wr/wi: W_n^k = e^{-2πik/n}，k ∈ [0, n/4]（宿主按 8 倍数上取整后上传）
extern "C" __global__ __aicore__ __vector__ void kfft_r2c_post(
    __gm__ float* out, __gm__ float* z, __gm__ float* wrrGm, __gm__ float* wriGm,
    uint32_t n, uint32_t batch)
{
    const uint32_t m      = n >> 1;              // 内层 FFT 长度
    const uint32_t kmax1  = (n >> 2) + 1u;       // L 段槽数 = kMax+1（含 kMax 自配对）
    const uint32_t ms1    = m + 1u;              // 有效槽数 = m+1（bin 0..m）
    const uint32_t strideF = hsStride(n);        // 半谱行距
    const uint32_t twLen  = (kmax1 + 7u) & ~7u;  // 旋转因子表按 8 倍数上取整
    const uint32_t q1 = (kmax1 + 7u) & ~7u;      // L 段表长（8 倍数 => 区基址 32B 对齐）
    const uint32_t q2 = ((ms1 - kmax1) + 7u) & ~7u;  // U 段表长

    TPipe pipe;
    TBuf<TPosition::VECCALC> bZ, bWr, bWi, bT, bTmp, bStg;
    pipe.InitBuffer(bZ, ub32(n * sizeof(float)));
    pipe.InitBuffer(bWr, ub32(twLen * sizeof(float)));
    pipe.InitBuffer(bWi, ub32(twLen * sizeof(float)));
    // 表：L 段 tA/tB/tW（各 q1）+ U 段 tA/tB/tW（各 q2）+ tPair（2*kChunk），全 8 倍数
    pipe.InitBuffer(bT, ub32((3u * q1 + 3u * q2 + 2u * kChunk) * sizeof(uint32_t)));
    pipe.InitBuffer(bTmp, ub32(16u * kChunk * sizeof(float)));
    pipe.InitBuffer(bStg, ub32(4u * kChunk * sizeof(float)));   // 交错输出双缓冲（各 2*kChunk）

    LocalTensor<float> zr    = bZ.Get<float>();
    LocalTensor<float> wrr   = bWr.Get<float>(), wri = bWi.Get<float>();
    LocalTensor<uint32_t> t0 = bT.Get<uint32_t>();
    LocalTensor<uint32_t> tAl = t0,           tBl = tAl[q1],        tWl = tBl[q1];
    LocalTensor<uint32_t> tAh = tWl[q1],      tBh = tAh[q2],        tWh = tBh[q2];
    LocalTensor<uint32_t> tPair = tWh[q2];
    LocalTensor<float> tm    = bTmp.Get<float>();
    LocalTensor<float> ar = tm,        ai = ar[kChunk],   br = ai[kChunk],  bi = br[kChunk];
    LocalTensor<float> er = bi[kChunk], ei = er[kChunk],  ore = ei[kChunk], oim = ore[kChunk];
    LocalTensor<float> tre = oim[kChunk], tim = tre[kChunk];
    LocalTensor<float> wv0 = tim[kChunk], wv1 = wv0[kChunk];
    LocalTensor<float> p0  = wv1[kChunk], p1 = p0[kChunk];
    LocalTensor<float> oR  = p1[kChunk],  oI = oR[kChunk];
    LocalTensor<float> stg0 = bStg.Get<float>(), stg1 = stg0[2u * kChunk];

    // ---- 表（批次循环外一次建好，全批共用）----
    // 分块相对下标恒为 0/kChunk/2kChunk…（≡0 mod 8），故分段独立存表保证切片 32B 对齐。
    for (uint32_t s = 0; s < kmax1; s++) {
        const uint32_t bb = (s == 0u) ? 0u : (m - s);
        tAl.SetValue(s, s * 8u);      // 复数槽 -> 字节偏移
        tBl.SetValue(s, bb * 8u);
        tWl.SetValue(s, s * 4u);      // float 槽 -> 字节偏移
    }
    for (uint32_t s = kmax1; s < ms1; s++) {
        const uint32_t idx = m - s;
        const uint32_t bb  = (idx == 0u) ? 0u : (m - idx);
        tAh.SetValue(s - kmax1, idx * 8u);
        tBh.SetValue(s - kmax1, bb * 8u);
        tWh.SetValue(s - kmax1, idx * 4u);
    }
    for (uint32_t i = 0; i < kChunk; i++) {   // 交错：dst[2i]=oR[i], dst[2i+1]=oI[i]
        tPair.SetValue(2u * i,     4u * i);
        tPair.SetValue(2u * i + 1u, 4u * (kChunk + i));
    }
    GlobalTensor<float> gwrr, gwri;
    gwrr.SetGlobalBuffer(wrrGm, twLen);
    gwri.SetGlobalBuffer(wriGm, twLen);
    DataCopy(wrr, gwrr, twLen);
    DataCopy(wri, gwri, twLen);
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk  = GetBlockIdx();

    for (int64_t b = blk; b < (int64_t)batch; b += nblk) {
        PipeBarrier<PIPE_ALL>();      // 行首 drain（见 prep 同款屏障）
        const uint32_t rowOff = (uint32_t)b * strideF;
        GlobalTensor<float> gz, gout;
        gz.SetGlobalBuffer(z + (uint32_t)b * n, n);
        DataCopy(zr, gz, n);            // 整行 Z 进 UB（n float，8 倍数）
        PipeBarrier<PIPE_ALL>();

        uint32_t ck = 0;                // 块序号（决定交错双缓冲）
        for (uint32_t seg = 0; seg < 2u; seg++) {
            const uint32_t lo   = seg ? kmax1 : 0u;
            const uint32_t hi   = seg ? ms1   : kmax1;
            const bool     kindL = (seg == 0u);
            LocalTensor<uint32_t> ta = seg ? tAh : tAl;
            LocalTensor<uint32_t> tb = seg ? tBh : tBl;
            LocalTensor<uint32_t> tw = seg ? tWh : tWl;
            for (uint32_t s = lo; s < hi; s += kChunk) {
                const uint32_t rel = s - lo;          // 相对表下标 ≡0 (mod 8)
                const uint32_t len  = ((hi - s) < kChunk) ? (hi - s) : kChunk;
                const uint32_t lenO = (len + 3u) & ~3u;   // DataCopy 上取整到 4 槽
                Gather(ar,  zr, ta[rel], 0u, len);
                Gather(ai,  zr, ta[rel], 4u, len);
                Gather(br,  zr, tb[rel], 0u, len);
                Gather(bi,  zr, tb[rel], 4u, len);
                Gather(wv0, wrr, tw[rel], 0u, len);
                Gather(wv1, wri, tw[rel], 0u, len);
                // E = (a + conj b)/2
                Add(er, ar, br, len);  Muls(er, er, 0.5f, len);
                Sub(ei, ai, bi, len);  Muls(ei, ei, 0.5f, len);
                // O = (a - conj b)/(2i)：O.re=(ai+bi)/2，O.im=(br-ar)/2
                Add(ore, ai, bi, len); Muls(ore, ore, 0.5f, len);
                Sub(oim, br, ar, len); Muls(oim, oim, 0.5f, len);
                // t = W * O
                Mul(p0, wv0, ore, len); Mul(p1, wv1, oim, len); Sub(tre, p0, p1, len);
                Mul(p0, wv0, oim, len); Mul(p1, wv1, ore, len); Add(tim, p0, p1, len);
                if (kindL) { Add(oR, er, tre, len); Add(oI, ei, tim, len); }
                else       { Sub(oR, er, tre, len); Sub(oI, tim, ei, len); }
                // 交错回复数布局（矢量管内顺序执行，与上面的算子之间不需屏障）
                LocalTensor<float> stg = (ck & 1u) ? stg1 : stg0;
                Gather(stg, oR, tPair, 0u, 2u * len);
                ck++;
                PipeBarrier<PIPE_ALL>();
                gout.SetGlobalBuffer(out + rowOff + 2u * s, 2u * lenO);
                DataCopy(gout, stg, 2u * lenO);
                // 故意不加屏障：交错双缓冲与下一块的算子读写不相交，
                // 而本行首块之前的 ALL 屏障（zr 装载后）负责跨行排空。
            }
        }
    }
}

// ---------------------------------------------------------- c2r Hermitian 展开
// out : [batch][2n]          满谱（交错复数）
// in  : [batch][hsStride(n)] 半谱（前 n/2+1 个复数有效）
extern "C" __global__ __aicore__ __vector__ void kfft_c2r_prep(
    __gm__ float* out, __gm__ float* in, __gm__ float* dummy0, __gm__ float* dummy1,
    uint32_t n, uint32_t batch)
{
    (void)dummy0; (void)dummy1;          // 与其余两内核保持同签名（40B）
    const uint32_t m      = n >> 1;
    const uint32_t inSt   = hsStride(n);
    const uint32_t mp1    = m + 1u;       // 计划域 k=0..m（含 Nyquist，供镜像首槽取值）
    const uint32_t rp     = (mp1 + 7u) & ~7u;  // [re | -im] 各区上取整到 8 元素（32B 对齐）

    TPipe pipe;
    TBuf<TPosition::VECCALC> bF, bP, bT;
    // 输入行直接落进 full[0..inSt)（inSt<=2n 恒成立），省掉独立 hbuf —— n=8192 时
    // 四缓冲合计 196736B > 192KB UB 会溢出，去掉 hBuf 后 163968B 富余 32KB。
    pipe.InitBuffer(bF, ub32(2u * n * sizeof(float)));
    pipe.InitBuffer(bP, ub32(2u * rp * sizeof(float)));            // [re(0..m) | -im(0..m)]
    pipe.InitBuffer(bT, ub32((2u * rp + n) * sizeof(uint32_t)));   // tRe(rp) + tIm(rp) + tV(n)

    LocalTensor<float> full = bF.Get<float>(), pm = bP.Get<float>();
    LocalTensor<uint32_t> tt = bT.Get<uint32_t>();
    LocalTensor<uint32_t> tRe = tt, tIm = tRe[rp], tV = tIm[rp];
    LocalTensor<float> pmIm = pm[rp];

    // tRe/tIm: full 上 bin k 实/虚部的字节偏移；tV: 满谱尾段 [n,2n) 每个 float 从 pm 取值的表
    for (uint32_t k = 0; k <= m; k++) {
        tRe.SetValue(k, 8u * k);
        tIm.SetValue(k, 8u * k + 4u);
    }
    // 位置 n+i：i=2t 属 bin m+t 实部，i=2t+1 属其虚部；X[m+t]=conj(X[m-t]) => 取 pm[m-t]
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t kk = m - (i >> 1);
        tV.SetValue(i, 4u * ((i & 1u) ? (rp + kk) : kk));
    }

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk  = GetBlockIdx();

    for (int64_t b = blk; b < (int64_t)batch; b += nblk) {
        // 行首 drain：full 是行复用的首写缓冲，必须先排空上一 launch 残留/上一行的在途 UB 写
        // （同 fft_radix2 的 A6 屏障），否则首行 DataCopy 与残留写 WAW 竞争 -> 偶发错值。
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<float> gin, gout;
        gin.SetGlobalBuffer(in + (uint32_t)b * inSt, inSt);
        gout.SetGlobalBuffer(out + (uint32_t)b * 2u * n, 2u * n);
        DataCopy(full, gin, inSt);               // 半谱行直接落位 full[0..inSt)（正序 bin 0..m + 行距尾巴）
        PipeBarrier<PIPE_ALL>();                 // b1: full 前段就绪（MTE 写完 -> 矢量读）
        Gather(pm,    full, tRe, 0u, mp1);       // 计划域 re（bin 0..m）
        Gather(pmIm,  full, tIm, 0u, mp1);       // im
        Muls(pmIm, pmIm, -1.0f, mp1);            // 镜像取共轭
        PipeBarrier<PIPE_ALL>();                 // b2: 满谱前段 MTE 写与矢量写分界
        // full+n 起点 n≡0(mod 8) => 32B 对齐，正好覆盖镜像区 [n,2n)（含 bin m 重写同值）
        Gather(full[n], pm, tV, 0u, n);
        PipeBarrier<PIPE_ALL>();                 // b3: 矢量写满谱 -> MTE 读满谱
        DataCopy(gout, full, 2u * n);             // 满谱行
        // 同一行内：full 被本行后续算子复用，跨行由行首/行尾的 ALL 屏障排空。
    }
}

// ---------------------------------------------------------- c2r 实部提取 + 归一
// out : [batch][n]   实数输出
// y   : [batch][2n]  kfft_fwd 反号旋转因子的输出（= n·x，虚部应≈0）
extern "C" __global__ __aicore__ __vector__ void kfft_c2r_post(
    __gm__ float* out, __gm__ float* y, __gm__ float* dummy0, __gm__ float* dummy1,
    uint32_t n, uint32_t batch)
{
    (void)dummy0; (void)dummy1;
    // aicore 标量里 (float)uint32 被 ccec 拒绝 => 走 vconv Cast（int32 -> fp32）
    TPipe pipe;
    TBuf<TPosition::VECCALC> bI, bF, bY, bR, bT;
    pipe.InitBuffer(bI, ub32(8 * sizeof(int32_t)));
    pipe.InitBuffer(bF, ub32(8 * sizeof(float)));
    pipe.InitBuffer(bY, ub32(2u * n * sizeof(float)));
    pipe.InitBuffer(bR, ub32(n * sizeof(float)));
    pipe.InitBuffer(bT, ub32(n * sizeof(uint32_t)));

    LocalTensor<int32_t> iv = bI.Get<int32_t>();
    LocalTensor<float>   fv = bF.Get<float>();
    for (uint32_t i = 0; i < 8u; i++) iv.SetValue(i, (int32_t)n);
    Cast(fv, iv, RoundMode::CAST_NONE, 8u);
    PipeBarrier<PIPE_ALL>();              // 标量读矢量写，先排空
    const float sc = 1.0f / fv.GetValue(0);
    LocalTensor<float> yb = bY.Get<float>(), re = bR.Get<float>();
    LocalTensor<uint32_t> tE = bT.Get<uint32_t>();
    for (uint32_t i = 0; i < n; i++) tE.SetValue(i, 8u * i);   // 每个复数的实部

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk  = GetBlockIdx();

    for (int64_t b = blk; b < (int64_t)batch; b += nblk) {
        PipeBarrier<PIPE_ALL>();      // 行首 drain（见 prep 同款屏障）
        GlobalTensor<float> gy, gout;
        gy.SetGlobalBuffer(y + (uint32_t)b * 2u * n, 2u * n);
        gout.SetGlobalBuffer(out + (uint32_t)b * n, n);
        DataCopy(yb, gy, 2u * n);
        PipeBarrier<PIPE_ALL>();
        Gather(re, yb, tE, 0u, n);        // 取偶数位（实部）
        Muls(re, re, sc, n);              // /n
        PipeBarrier<PIPE_ALL>();
        DataCopy(gout, re, n);
    }
}
