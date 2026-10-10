// R2-A：kfft_lt_tr 的完整内核体，按 (AB_LT_TR_SUFFIX, AB_LT_TR_H,
// AB_LT_TR_W, AB_LT_TR_K, AB_LT_TR_RESIDENT) 逐候选在包含点展开（ccec
// 9.0.0 不接受把 TPipe/InitBuffer 移进非入口 __aicore__ 函数——实测 AIV
// 挂死，故用包含展开而非模板）。TPipe 必须留在 __global__ 入口。
// Round 3（AB_LT_TR_RESIDENT=1）：条带三张索引表迁入独立只读 buffer，
// 入口建一次，grid-stride 不再逐 tile 重建（bOut 末尾大 Gather 会清 bOut，
// 重建即本 bug 的 2048 次 scalar SetValue/tile）；bOut 只切平面三区。
#ifndef AB_LT_TR_RESIDENT
#define AB_LT_TR_RESIDENT 0
#endif
static_assert((AB_LT_TR_W & 3u) == 0u,
              "tile W must keep 32B row-slice alignment");
#if AB_LT_TR_RESIDENT
static_assert(3ull * AB_LT_TR_K <= (unsigned long long)AB_LT_TR_H * AB_LT_TR_W,
              "resident planar carve must fit the bOut lifetime");
#else
static_assert(10ull * AB_LT_TR_K <=
                  2ull * AB_LT_TR_H * AB_LT_TR_W,
              "rebuild stripe machinery must fit the bOut lifetime carve");
#endif

extern "C" __global__ __aicore__ __vector__ void
    AB_KLTT_CAT(kfft_lt_tr, AB_LT_TR_SUFFIX)(
        __gm__ float* dst, __gm__ float* src, __gm__ float* tw,
        uint32_t nRows, uint32_t nCols, uint32_t batch)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> bIn, bOut, bIdx, bTw;
    const uint32_t tileN = AB_LT_TR_H * AB_LT_TR_W;                    // 复数 / tile
    pipe.InitBuffer(bIn,  tileN * 2u * sizeof(float));
    pipe.InitBuffer(bOut, tileN * 2u * sizeof(float));
    pipe.InitBuffer(bIdx, tileN * 2u * sizeof(uint32_t));
    pipe.InitBuffer(bTw,  tileN * 2u * sizeof(float));  // 融合 tw tile
#if AB_LT_TR_RESIDENT
    TBuf<TPosition::VECCALC> bSidx;
    pipe.InitBuffer(bSidx, AB_LT_TR_K * 4u * sizeof(uint32_t)); // 4K 表
#endif
    LocalTensor<float> tIn  = bIn.Get<float>();
    LocalTensor<float> tOut = bOut.Get<float>();
    LocalTensor<float> tTw  = bTw.Get<float>();
    LocalTensor<uint32_t> idx = bIdx.Get<uint32_t>();
    // 重排索引：dst 行序 t=2*(w*H+i)+e <- tile 位置 (i*W+w)
    for (uint32_t i = 0; i < AB_LT_TR_H; i++)
      for (uint32_t w = 0; w < AB_LT_TR_W; w++) {
        uint32_t p = (i * AB_LT_TR_W + w) * 8u;
        idx.SetValue(2u * (w * AB_LT_TR_H + i),      p);
        idx.SetValue(2u * (w * AB_LT_TR_H + i) + 1u, p + 4u);
      }
    // 融合条带机件（tw!=nullptr 才建）：
    //   planarA [0,2K) 数据平面 | planarW [2K,4K) tw 平面 | prod [4K,6K) 乘积
    // rebuild 模式：idxRe/idxIm 各 K、idxX 2K 个 uint32 收在 bOut [6K,10K)
    //              （10K*4B <= 2HW*4B），每 tile 重建（末尾大 Gather 清 bOut）。
    // resident 模式：三表移入独立 bSidx（16K 字节），入口建一次，不重建。
    const uint32_t K = AB_LT_TR_K;
    LocalTensor<float> planA = tOut;
    LocalTensor<float> planW = tOut[2u * K];
    LocalTensor<float> prod  = tOut[4u * K];
#if AB_LT_TR_RESIDENT
    LocalTensor<uint32_t> idxRe = bSidx.Get<uint32_t>();
    LocalTensor<uint32_t> idxIm = idxRe[K];
    LocalTensor<uint32_t> idxX  = idxRe[2u * K];   // 4K 张表收满 bSidx
#else
    LocalTensor<uint32_t> uB    = bOut.Get<uint32_t>();
    LocalTensor<uint32_t> idxRe = uB[6u * K];
    LocalTensor<uint32_t> idxIm = uB[7u * K];
    LocalTensor<uint32_t> idxX  = uB[8u * K];
#endif
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
    const uint32_t nC = nCols / AB_LT_TR_W;                    // dst 行 tile 数 / 批
    const uint32_t nI = (nRows + AB_LT_TR_H - 1u) / AB_LT_TR_H;        // src 行 tile 数
    const uint64_t tiles = (uint64_t)batch * nC * nI;
    const uint64_t span  = 2ull * (uint64_t)nRows * nCols; // float / 批
    for (uint64_t tt = (uint64_t)blk; tt < tiles; tt += (uint64_t)nblk) {
      const uint32_t wT0 = (uint32_t)(tt % nC) * AB_LT_TR_W;
      const uint32_t iT0 = (uint32_t)((tt / nC) % nI) * AB_LT_TR_H;
      const uint32_t b   = (uint32_t)(tt / ((uint64_t)nC * nI));
      const uint32_t wCnt = nCols - wT0 >= AB_LT_TR_W ? AB_LT_TR_W : nCols - wT0;
      const uint32_t rCnt = nRows - iT0 >= AB_LT_TR_H ? AB_LT_TR_H : nRows - iT0;
      GlobalTensor<float> gs, gd;
      // 块 r 起点 = base + r*(nCols*8 字节)；块长 wCnt*8，gap = (nCols-wCnt)*8
      gs.SetGlobalBuffer(src + (uint64_t)b * span +
                         ((uint64_t)iT0 * nCols + (uint64_t)wT0) * 2u,
                         (uint64_t)rCnt * nCols * 2u);
      DataCopy(tIn, gs,
               DataCopyParams{(uint16_t)rCnt, (uint16_t)(wCnt / 4u),
                              (uint16_t)((nCols - wCnt) / 4u), 0});
      if (tw) {
        // tw 切片与数据切片同坐标同参（wT 形状 [nRows][nCols]，批共享）
        GlobalTensor<float> gt;
        gt.SetGlobalBuffer(tw + ((uint64_t)iT0 * nCols + (uint64_t)wT0) * 2u,
                           (uint64_t)rCnt * nCols * 2u);
        DataCopy(tTw, gt,
                 DataCopyParams{(uint16_t)rCnt, (uint16_t)(wCnt / 4u),
                                (uint16_t)((nCols - wCnt) / 4u), 0});
      }
      PipeBarrier<PIPE_ALL>();
      if (tw) {
#if !AB_LT_TR_RESIDENT
        // idx 表切自 bOut，而每 tile 末尾的大 Gather 会全量写 tOut（= bOut）：
        // grid-stride 第 2 个及以后的 tile 必须重建，否则平面 Gather 按被清掉的
        // 垃圾偏移读 UB -> 越界（AIV 507035）。首个 tile 用入口处建好的表。
        // resident 模式三表在独立 bSidx，永不被清 -> 整段删除。
        if (tt != (uint64_t)blk) {
          for (uint32_t k = 0; k < K; k++) {
            idxRe.SetValue(k,      8u * k);
            idxIm.SetValue(k,      8u * k + 4u);
            idxX.SetValue(2u * k,      4u * k);
            idxX.SetValue(2u * k + 1u, 4u * K + 4u * k);
          }
          PipeBarrier<PIPE_ALL>();
        }
#endif
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
          Gather(ds, prod, idxX, 0u, 2u * kl);        // 写回本带（交错、已乘 tw）
          PipeBarrier<PIPE_ALL>();
        }
      }
      Gather(tOut, tIn, idx, 0u, 2u * tileN);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t w = 0; w < wCnt; w++) {
        gd.SetGlobalBuffer(dst + (uint64_t)b * span +
                           ((uint64_t)(wT0 + w) * nRows + iT0) * 2u, rCnt * 2u);
        DataCopy(gd, tOut[w * 2u * AB_LT_TR_H], rCnt * 2u);
      }
      PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
