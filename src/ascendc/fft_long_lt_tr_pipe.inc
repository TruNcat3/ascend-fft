// R2-B 管线实验入口（固定在 R2-A 胜者配置：128x32 tile、K=256、resident
// 索引）。TPipe 必须留在 __global__ 入口，且 ccec 9.0.0 的 device lambda
// 内不可用 LocalTensor 成员（operator[] 等）——本文件全部直线代码。
//   AB_LT_TR_PP=0 (_nb)：串行 tile 循环不变，只把跨 pipe 依赖的 PIPE_ALL
//        收窄到真实依赖对（条带循环内部全为 vector pipe 同序操作，
//        删除逐 stripe 的 PIPE_ALL——rebuild 版每 tile ~3*(C/K) 次空转）。
//   AB_LT_TR_PP=1 (_pp)：输入乒乓（第二块 bIn，UB 132K+32K=164K<=192K）
//        + 软流水发射：MTE2(数据,t+1) 与 vector(t) 重叠，MTE3(t) 与
//        MTE2(t+1) 在飞；bTw/bOut 单份，用依赖屏障隔开。
// 依赖（AIV 三 pipe；DataCopy GM<->UB = MTE2/MTE3，Gather/Mul = vector）：
//   MTE2 -> vector : 本 tile 载入完成后才可读（MTE2,VECTOR 屏障）
//   vector -> MTE3 : tOut 写完才可按行写出（VECTOR,MTE3 屏障）
//   MTE3(t) -> vector(t+1) : 单 bOut 复用（VECTOR,MTE3 屏障，双向）
//   vector(t) -> MTE2(t+1) tw : 单 bTw 复用（vector(t) 完成后再发 tw）
//   vector(t) -> MTE2(t+1) data : 输入乒乓后无此依赖（双 bIn）
static_assert((AB_LT_TR_W & 3u) == 0u, "tile W must keep 32B alignment");
static_assert(AB_LT_TR_RESIDENT == 1, "pipe experiments assume resident idx");
static_assert(3ull * AB_LT_TR_K <=
                  (unsigned long long)AB_LT_TR_H * AB_LT_TR_W,
              "planar carve must fit bOut");

extern "C" __global__ __aicore__ __vector__ void
    AB_KLTT_CAT(kfft_lt_tr, AB_LT_TR_SUFFIX)(
        __gm__ float* dst, __gm__ float* src, __gm__ float* tw,
        uint32_t nRows, uint32_t nCols, uint32_t batch)
{
    TPipe pipe;
    const uint32_t tileN = AB_LT_TR_H * AB_LT_TR_W;
    TBuf<TPosition::VECCALC> bIn0, bOut, bIdx, bTw;
    pipe.InitBuffer(bIn0, tileN * 2u * sizeof(float));
#if AB_LT_TR_PP
    TBuf<TPosition::VECCALC> bIn1;
    pipe.InitBuffer(bIn1, tileN * 2u * sizeof(float));
#endif
    pipe.InitBuffer(bOut, tileN * 2u * sizeof(float));
    pipe.InitBuffer(bIdx, tileN * 2u * sizeof(uint32_t));
    pipe.InitBuffer(bTw,  tileN * 2u * sizeof(float));
    TBuf<TPosition::VECCALC> bSidx;
    pipe.InitBuffer(bSidx, AB_LT_TR_K * 4u * sizeof(uint32_t));
    LocalTensor<float> tIn0 = bIn0.Get<float>();
#if AB_LT_TR_PP
    LocalTensor<float> tIn1 = bIn1.Get<float>();
#endif
    LocalTensor<float> tOut = bOut.Get<float>();
    LocalTensor<float> tTw  = bTw.Get<float>();
    LocalTensor<uint32_t> idx = bIdx.Get<uint32_t>();
    for (uint32_t i = 0; i < AB_LT_TR_H; i++)
      for (uint32_t w = 0; w < AB_LT_TR_W; w++) {
        uint32_t p = (i * AB_LT_TR_W + w) * 8u;
        idx.SetValue(2u * (w * AB_LT_TR_H + i),      p);
        idx.SetValue(2u * (w * AB_LT_TR_H + i) + 1u, p + 4u);
      }
    const uint32_t K = AB_LT_TR_K;
    LocalTensor<float> planA = tOut;
    LocalTensor<float> planW = tOut[2u * K];
    LocalTensor<float> prod  = tOut[4u * K];
    LocalTensor<uint32_t> idxRe = bSidx.Get<uint32_t>();
    LocalTensor<uint32_t> idxIm = idxRe[K];
    LocalTensor<uint32_t> idxX  = idxRe[2u * K];
    if (tw) {
      for (uint32_t k = 0; k < K; k++) {
        idxRe.SetValue(k,      8u * k);
        idxIm.SetValue(k,      8u * k + 4u);
        idxX.SetValue(2u * k,      4u * k);
        idxX.SetValue(2u * k + 1u, 4u * K + 4u * k);
      }
    }
    PipeBarrier<PIPE_ALL>();

    int64_t nblk = GetBlockNum(); if (nblk <= 0) nblk = 1;
    int64_t blk  = GetBlockIdx();
    const uint32_t nC = nCols / AB_LT_TR_W;
    const uint32_t nI = (nRows + AB_LT_TR_H - 1u) / AB_LT_TR_H;
    const uint64_t tiles = (uint64_t)batch * nC * nI;
    const uint64_t span  = 2ull * (uint64_t)nRows * nCols;

#if !AB_LT_TR_PP
    // ---- _nb：串行循环 + 收窄屏障 ---------------------------------------
    for (uint64_t tt = (uint64_t)blk; tt < tiles; tt += (uint64_t)nblk) {
      uint32_t wT0 = (uint32_t)(tt % nC) * AB_LT_TR_W;
      uint32_t iT0 = (uint32_t)((tt / nC) % nI) * AB_LT_TR_H;
      uint32_t b   = (uint32_t)(tt / ((uint64_t)nC * nI));
      uint32_t wCnt = nCols - wT0 >= AB_LT_TR_W ? AB_LT_TR_W : nCols - wT0;
      uint32_t rCnt = nRows - iT0 >= AB_LT_TR_H ? AB_LT_TR_H : nRows - iT0;
      {
        GlobalTensor<float> gs;
        gs.SetGlobalBuffer(src + (uint64_t)b * span +
                           ((uint64_t)iT0 * nCols + (uint64_t)wT0) * 2u,
                           (uint64_t)rCnt * nCols * 2u);
        DataCopy(tIn0, gs,
                 DataCopyParams{(uint16_t)rCnt, (uint16_t)(wCnt / 4u),
                                (uint16_t)((nCols - wCnt) / 4u), 0});
        if (tw) {
          GlobalTensor<float> gt;
          gt.SetGlobalBuffer(tw + ((uint64_t)iT0 * nCols + (uint64_t)wT0) * 2u,
                             (uint64_t)rCnt * nCols * 2u);
          DataCopy(tTw, gt,
                   DataCopyParams{(uint16_t)rCnt, (uint16_t)(wCnt / 4u),
                                  (uint16_t)((nCols - wCnt) / 4u), 0});
        }
      }
      PipeBarrier<PIPE_ALL>(); // [S6] load fence ALL
      if (tw) {
        const uint32_t C = rCnt * wCnt;
        for (uint32_t s = 0; s < C; s += K) {
          const uint32_t kl = (C - s >= K) ? K : (C - s);
          LocalTensor<float> ds = tIn0[2u * s];
          LocalTensor<float> ts = tTw[2u * s];
          Gather(planA,    ds, idxRe, 0u, kl);
          Gather(planA[K], ds, idxIm, 0u, kl);
          Gather(planW,    ts, idxRe, 0u, kl);
          Gather(planW[K], ts, idxIm, 0u, kl);
          PipeBarrier<PIPE_V>();   // gather -> FMA（同 pipe 双侧：V 一次）
          Mul(prod[K], planA,     planW[K], kl);
          MulAddDst(prod[K], planA[K], planW, kl);
          Muls(planW[K], planW[K], -1.f, kl);
          Mul(prod, planA, planW, kl);
          MulAddDst(prod, planA[K], planW[K], kl);
          Muls(planW[K], planW[K], -1.f, kl);
          PipeBarrier<PIPE_V>();   // FMA -> 写回
          Gather(ds, prod, idxX, 0u, 2u * kl);
          PipeBarrier<PIPE_V>();   // 写回 -> 下一带 / 大 Gather
        }
      }
      Gather(tOut, tIn0, idx, 0u, 2u * tileN);
      PipeBarrier<PIPE_ALL>(); // [S6] gather->MTE3 ALL
      {
        GlobalTensor<float> gd;
        for (uint32_t w = 0; w < wCnt; w++) {
          gd.SetGlobalBuffer(dst + (uint64_t)b * span +
                             ((uint64_t)(wT0 + w) * nRows + iT0) * 2u,
                             rCnt * 2u);
          DataCopy(gd, tOut[w * 2u * AB_LT_TR_H], rCnt * 2u);
        }
      }
      // 下一 tile 的 MTE2 复用 tIn0/tTw/tOut：与本 tile 的 MTE3/vector 同步
      PipeBarrier<PIPE_MTE2>(); PipeBarrier<PIPE_MTE3>(); PipeBarrier<PIPE_V>();
    }
#else
    // ---- _pp：软流水 + 输入乒乓（奇偶=迭代序号 (tt-blk)/nblk：序曲 tile 恒进 in[0]）
    // 本栈（ccec 9.0.0/AIV）实测：跨 pipe 依赖只有 PIPE_ALL 可靠（单 pipe
    // fence 不提供跨 pipe UB 可见性）。因此重叠不靠窄屏障，靠异步发射顺序：
    //   1) vector(t) 发射（异步）+ MTE2(数据,t+1) 发射（双 bIn，无 WAR，重叠）
    //   2) PIPE_ALL：vector(t) 完 && 数据(t+1) 载完 -> 发 MTE3(t) 行写出
    //      + MTE2(tw,t+1) 发射（单 bTw 须等 vector(t)，但可与 MTE3(t) 重叠）
    //   3) PIPE_ALL：MTE3(t) 完（单 bOut WAR）&& tw(t+1) 载完 -> vector(t+1)
    uint32_t iT0, wT0, b, wCnt, rCnt;
    {
      uint64_t tt = (uint64_t)blk;
      wT0 = (uint32_t)(tt % nC) * AB_LT_TR_W;
      iT0 = (uint32_t)((tt / nC) % nI) * AB_LT_TR_H;
      b   = (uint32_t)(tt / ((uint64_t)nC * nI));
      wCnt = nCols - wT0 >= AB_LT_TR_W ? AB_LT_TR_W : nCols - wT0;
      rCnt = nRows - iT0 >= AB_LT_TR_H ? AB_LT_TR_H : nRows - iT0;
      // 序曲：tile(=blk) 数据载入进 in[0]，tw 单独发
      GlobalTensor<float> gs;
      gs.SetGlobalBuffer(src + (uint64_t)b * span +
                         ((uint64_t)iT0 * nCols + (uint64_t)wT0) * 2u,
                         (uint64_t)rCnt * nCols * 2u);
      DataCopy(tIn0, gs,
               DataCopyParams{(uint16_t)rCnt, (uint16_t)(wCnt / 4u),
                              (uint16_t)((nCols - wCnt) / 4u), 0});
      if (tw) {
        GlobalTensor<float> gt;
        gt.SetGlobalBuffer(tw + ((uint64_t)iT0 * nCols + (uint64_t)wT0) * 2u,
                           (uint64_t)rCnt * nCols * 2u);
        DataCopy(tTw, gt,
                 DataCopyParams{(uint16_t)rCnt, (uint16_t)(wCnt / 4u),
                                (uint16_t)((nCols - wCnt) / 4u), 0});
      }
      PipeBarrier<PIPE_ALL>();
    }
    for (uint64_t tt = (uint64_t)blk; tt < tiles; tt += (uint64_t)nblk) {
      LocalTensor<float> tIn = ((((tt - (uint64_t)blk) / (uint64_t)nblk) & 1u) == 0u) ? tIn0 : tIn1;
      if (tw) {
        const uint32_t C = rCnt * wCnt;
        for (uint32_t s = 0; s < C; s += K) {
          const uint32_t kl = (C - s >= K) ? K : (C - s);
          LocalTensor<float> ds = tIn[2u * s];
          LocalTensor<float> ts = tTw[2u * s];
          Gather(planA,    ds, idxRe, 0u, kl);
          Gather(planA[K], ds, idxIm, 0u, kl);
          Gather(planW,    ts, idxRe, 0u, kl);
          Gather(planW[K], ts, idxIm, 0u, kl);
          PipeBarrier<PIPE_V>();   // gather -> FMA（同 pipe：V 一次，已验证）
          Mul(prod[K], planA,     planW[K], kl);
          MulAddDst(prod[K], planA[K], planW, kl);
          Muls(planW[K], planW[K], -1.f, kl);
          Mul(prod, planA, planW, kl);
          MulAddDst(prod, planA[K], planW[K], kl);
          Muls(planW[K], planW[K], -1.f, kl);
          PipeBarrier<PIPE_V>();   // FMA -> 写回
          Gather(ds, prod, idxX, 0u, 2u * kl);
          PipeBarrier<PIPE_V>();   // 写回 -> 下一带 / 大 Gather
        }
      }
      Gather(tOut, tIn, idx, 0u, 2u * tileN);
      const uint64_t tn = tt + (uint64_t)nblk;
      uint32_t iTn = iT0, wTn = wT0, bn = b, wN = wCnt, rN = rCnt;
      if (tn < tiles) {
        wTn = (uint32_t)(tn % nC) * AB_LT_TR_W;
        iTn = (uint32_t)((tn / nC) % nI) * AB_LT_TR_H;
        bn  = (uint32_t)(tn / ((uint64_t)nC * nI));
        wN  = nCols - wTn >= AB_LT_TR_W ? AB_LT_TR_W : nCols - wTn;
        rN  = nRows - iTn >= AB_LT_TR_H ? AB_LT_TR_H : nRows - iTn;
      }
      if (tn < tiles) {      // 数据(t+1)：双 bIn，与 vector(t) 无依赖，重叠发射
        LocalTensor<float> tInN = ((((tn - (uint64_t)blk) / (uint64_t)nblk) & 1u) == 0u) ? tIn0 : tIn1;
        GlobalTensor<float> gsn;
        gsn.SetGlobalBuffer(src + (uint64_t)bn * span +
                            ((uint64_t)iTn * nCols + (uint64_t)wTn) * 2u,
                            (uint64_t)rN * nCols * 2u);
        DataCopy(tInN, gsn,
                 DataCopyParams{(uint16_t)rN, (uint16_t)(wN / 4u),
                                (uint16_t)((nCols - wN) / 4u), 0});
      }
      // vector(t) 完成（tOut 写完、tTw/tIn(t) 读完）&& 数据(t+1) 载完
      PipeBarrier<PIPE_ALL>();
      {
        GlobalTensor<float> gd;
        for (uint32_t w = 0; w < wCnt; w++) {
          gd.SetGlobalBuffer(dst + (uint64_t)b * span +
                             ((uint64_t)(wT0 + w) * nRows + iT0) * 2u,
                             rCnt * 2u);
          DataCopy(gd, tOut[w * 2u * AB_LT_TR_H], rCnt * 2u);
        }
      }
      if (tn < tiles && tw) { // tw(t+1)：单 bTw，等 vector(t) 后才可发；与 MTE3(t) 重叠
        GlobalTensor<float> gtn;
        gtn.SetGlobalBuffer(tw + ((uint64_t)iTn * nCols + (uint64_t)wTn) * 2u,
                            (uint64_t)rN * nCols * 2u);
        DataCopy(tTw, gtn,
                 DataCopyParams{(uint16_t)rN, (uint16_t)(wN / 4u),
                                (uint16_t)((nCols - wN) / 4u), 0});
      }
      // MTE3(t) 完（单 bOut WAR）&& tw(t+1) 载完 -> vector(t+1)
      PipeBarrier<PIPE_ALL>();
      iT0 = iTn; wT0 = wTn; b = bn; wCnt = wN; rCnt = rN;
    }
#endif
    PipeBarrier<PIPE_ALL>();
}
