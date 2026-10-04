// Phase 0.1: native CANN aclRfft1D baseline (float32 real -> float32 interleaved complex).
// usage: baseline_rfft <n> <batch> <norm 0|1|2> <dump.bin> [--e2e] [--reps=K]
//   out shape per batch = 2*(n/2+1) floats (interleaved complex, onesided).
//
// --e2e : 裸 CANN C API 的端到端口径（H2D -> aclRfft1D -> D2H），输出一行 BARE，
//         与 scripts/e2e_test.py 里「自研 kfft_fwd」/「torch_npu」两路逐段对应。
//         计时段外剔除：进程启动(boot_us)、主机输入生成 + 显存分配 + 首次上传(setup_us)。
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/acl_rfft1d.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <complex>
#define CK(x) do{ auto _e=(x); if(_e!=ACL_SUCCESS){ printf("ERR %d at %d\n",(int)_e,__LINE__); return 1;} }while(0)
// 版：用于返回 double 的 lambda（错误统一返回 -1.0）
#define CKL(x) do{ auto _e=(x); if(_e!=ACL_SUCCESS){ printf("ERR %d at %d\n",(int)_e,__LINE__); return -1.0;} }while(0)

using clk = std::chrono::steady_clock;
static double since(const clk::time_point& t0){
    return std::chrono::duration<double,std::micro>(clk::now()-t0).count();
}

// 参考：与 numpy.fft.rfft 同义（前向不缩放）。实测 aclRfft1D norm=1 即 unscaled。
// 只对 row0 做直接求和（O(n*modes)），单进程一次，n<=4096 约几十 ms。
static double rfftMaxRel(const std::vector<float>& h, const std::vector<float>& ho,
                         int64_t n, int64_t modes){
    double maxAbs=0;
    std::vector<std::complex<double>> ref((size_t)modes);
    for(int64_t k=0;k<modes;k++){
        const double ang=-2.0*M_PI*(double)k/(double)n;
        std::complex<double> w(std::cos(ang), std::sin(ang));
        std::complex<double> acc(1.0,0.0), s(0.0,0.0);
        for(int64_t j=0;j<n;j++){ s += (double)h[(size_t)j]*acc; acc *= w; }
        ref[(size_t)k]=s;
        const double a=std::abs(s);
        if(a>maxAbs) maxAbs=a;
    }
    double maxErr=0;
    for(int64_t k=0;k<modes;k++){
        const std::complex<double> y((double)ho[(size_t)(k*2)], (double)ho[(size_t)(k*2+1)]);
        const double e=std::abs(y-ref[(size_t)k]);
        if(e>maxErr) maxErr=e;
    }
    return maxAbs>0 ? maxErr/maxAbs : maxErr;
}

int main(int argc, char** argv){
    if(argc<5){ printf("usage: %s <n> <batch> <norm> <dump.bin> [--e2e] [--reps=K]\n", argv[0]); return 2; }
    int64_t n = atoll(argv[1]), b = atoll(argv[2]), norm = atoll(argv[3]);
    const char* dump = argv[4];
    bool doE2E=false; int reps=50;
    for(int i=5;i<argc;i++){
        if(!strcmp(argv[i],"--e2e")) doE2E=true;
        else if(!strncmp(argv[i],"--reps=",7)) reps=atoi(argv[i]+7);
    }
    if(reps<=0) reps = doE2E?10:50;
    int64_t modes = n/2+1;
    int64_t outN = modes*2;   // out shape [b, modes, 2]

    const auto tBoot0 = clk::now();
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s=nullptr; CK(aclrtCreateStream(&s));
    const double bootUs = since(tBoot0);

    std::vector<float> h((size_t)(b*n));
    for(int64_t i=0;i<b*n;i++) h[i] = std::sin(0.017*i) + 0.3f*std::cos(0.005*i);
    std::vector<float> ho((size_t)(b*outN), -1.f);

    const auto tSetup0 = clk::now();
    void *dx=nullptr,*dy=nullptr;
    CK(aclrtMalloc(&dx, h.size()*4, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dy, ho.size()*4, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMemcpy(dx, h.size()*4, h.data(), h.size()*4, ACL_MEMCPY_HOST_TO_DEVICE));

    int64_t inDims[2]={b,n}, outDims[3]={b,modes,2};
    aclTensor* x = aclCreateTensor(inDims,2,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,inDims,2,dx);
    aclTensor* y = aclCreateTensor(outDims,3,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,outDims,3,dy);
    if(!x||!y){ printf("tensor create failed\n"); return 1; }
    const double setupUs = since(tSetup0);

    void *dws=nullptr; uint64_t wsCap=0;
    auto ensureWs=[&](uint64_t ws)->int{
        if(ws>wsCap){
            if(dws) aclrtFree(dws);
            dws=nullptr;
            if(aclrtMalloc(&dws, ws, ACL_MEM_MALLOC_NORMAL_ONLY)!=ACL_SUCCESS) return 1;
            wsCap=ws;
        }
        return 0;
    };

    // ================= 裸 CANN C API 端到端 =================
    // 进程级冷启动（模块装载 / 首次执行的图构建）会把 first_us 拉到几百 ms，
    // 与「该 shape 的 plan 冷构建」混在一起。同 bench_native_npu.py：先用一次
    // 极小变换把这部分消耗掉，让 first_us 只反映本 shape 的 plan 构建。
    // 计时区外，不计入 boot/setup/first。仅 --e2e 需要（默认路径行为保持不变）。
    if(doE2E){
        int64_t wn[2]={1,8}, wo[3]={1,5,2};
        aclTensor* wx = aclCreateTensor(wn,2,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,wn,2,dx);
        aclTensor* wy = aclCreateTensor(wo,3,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,wo,3,dy);
        if(wx&&wy){
            uint64_t ws=0; aclOpExecutor* ex=nullptr;
            if(aclRfft1DGetWorkspaceSize(wx,8,1,1,wy,&ws,&ex)==0 && ensureWs(ws)==0 &&
               aclRfft1D(dws, ws, ex, s)==0){
                aclrtSynchronizeStream(s);
            }
        }
        if(wx) aclDestroyTensor(wx);
        if(wy) aclDestroyTensor(wy);
    }

    if(doE2E){
        // 一轮 = H2D -> aclRfft1D -> D2H。与 bench_native_npu.py --e2e 的
        //   xd.copy_(x_cpu) -> torch.fft.fft(xd) -> yd.copy_(out) 逐段对应。
        // AB_E2E_HOST: pinned（默认，aclrtMallocHost）| pageable（受限口径，仅 A/B 对照）。
        //   pageable 带宽随 loadavg 摆 2~4×，会把「没用最优拷贝路径」记成这一路的账。
        const char* hostMode = getenv("AB_E2E_HOST");
        const bool wantPinned = !(hostMode && strcmp(hostMode,"pageable")==0);
        float *pIn=nullptr, *pOut=nullptr;
        if(wantPinned){
            void *a=nullptr, *b=nullptr;
            const size_t ni=h.size()*4, no=ho.size()*4;
            const int ei=aclrtMallocHost(&a, ni), eo=aclrtMallocHost(&b, no);
            if(ei==ACL_SUCCESS && eo==ACL_SUCCESS){
                pIn=(float*)a; pOut=(float*)b; memcpy(pIn, h.data(), ni);
            }else{
                if(ei==ACL_SUCCESS) aclrtFreeHost(a);
                if(eo==ACL_SUCCESS) aclrtFreeHost(b);
                printf("BARE host=pinned alloc failed (%d/%d) -> pageable\n", ei, eo);
            }
        }
        const float* src = pIn ? pIn : h.data();
        float*       dst = pOut ? pOut : ho.data();
        const char*  hostTag = pIn ? "pinned" : "pageable";
        auto e2eOnce=[&](){
            const auto a0 = clk::now();
            CKL(aclrtMemcpyAsync(dx, h.size()*4, src, h.size()*4, ACL_MEMCPY_HOST_TO_DEVICE, s));
            CKL(aclrtSynchronizeStream(s));
            // aclRfft1D 两段式约定：executor 不可复用，每轮必须重新 GetWorkspaceSize。
            uint64_t ws=0; aclOpExecutor* ex=nullptr;
            if(aclRfft1DGetWorkspaceSize(x,n,1,1,y,&ws,&ex)!=0 || ensureWs(ws)) return -1.0;
            if(aclRfft1D(dws, ws, ex, s)!=0) return -1.0;
            CKL(aclrtSynchronizeStream(s));
            CKL(aclrtMemcpyAsync(dst, ho.size()*4, dy, ho.size()*4, ACL_MEMCPY_DEVICE_TO_HOST, s));
            CKL(aclrtSynchronizeStream(s));
            return since(a0);
        };
        // 只算 kernel：GetWorkspaceSize + aclRfft1D + sync
        auto devOnce=[&](){
            const auto a0 = clk::now();
            uint64_t ws=0; aclOpExecutor* ex=nullptr;
            if(aclRfft1DGetWorkspaceSize(x,n,1,1,y,&ws,&ex)!=0 || ensureWs(ws)) return -1.0;
            if(aclRfft1D(dws, ws, ex, s)!=0) return -1.0;
            CKL(aclrtSynchronizeStream(s));
            return since(a0);
        };

        // 冷首调单列（该 shape 第一次调用：含 GetWorkspaceSize + workspace 分配 + 首次执行），
        // 与 NATIVE_E2E 的 first_us 同义；不计入 mean/min。
        const double firstE = e2eOnce();  if(firstE<0) return 1;
        e2eOnce();                        // warmup（同 bench()：不计入）
        double sumE=0, minE=0;
        for(int i=0;i<reps;i++){
            const double u=e2eOnce(); if(u<0) return 1;
            sumE+=u; if(i==0) minE=u; else if(u<minE) minE=u;
        }
        const double eMean=sumE/reps;

        devOnce();                        // warmup
        double sumD=0, minD=0;
        for(int i=0;i<reps;i++){
            const double u=devOnce(); if(u<0) return 1;
            sumD+=u; if(i==0) minD=u; else if(u<minD) minD=u;
        }
        const double dMean=sumD/reps;

        CK(aclrtMemcpy(ho.data(), ho.size()*4, dy, ho.size()*4, ACL_MEMCPY_DEVICE_TO_HOST));
        const double rel = rfftMaxRel(h, ho, n, modes);

        printf("BARE n=%lld b=%lld dev_us=%.1f dev_mean_us=%.1f e2e_us=%.1f e2e_mean_us=%.1f "
               "first_us=%.1f boot_us=%.1f setup_us=%.1f ws_mb=%.1f maxRel=%.3e %s host=%s\n",
               (long long)n,(long long)b, minD,dMean, minE,eMean, firstE, bootUs, setupUs,
               wsCap/1048576.0, rel, rel<=1e-4?"PASS":"FAIL", hostTag);
        fflush(stdout);
        if(pIn){ aclrtFreeHost(pIn); aclrtFreeHost(pOut); }
        FILE* f=fopen(dump,"wb");
        if(f){ fwrite(ho.data(),4,ho.size(),f); fclose(f); }
        return rel<=1e-4?0:1;
    }

    // ================= 默认路径（历史行为，输出不变）=================
    if(norm==0) printf("WARNING: norm=0 rejected by the op; forcing 1\n");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    aclnnStatus st = aclRfft1DGetWorkspaceSize(x,n,1,1,y,&ws,&ex);
    printf("GetWorkspaceSize status=%d workspace=%llu\n", (int)st, (unsigned long long)ws);
    if(st!=0) return 1;
    if(ws){ CK(aclrtMalloc(&dws, ws, ACL_MEM_MALLOC_NORMAL_ONLY)); }

    st = aclRfft1D(dws, ws, ex, s);
    if(st!=0){ printf("aclRfft1D status=%d\n",(int)st); return 1; }
    CK(aclrtSynchronizeStream(s));

    // 计时口径与 build/fft_check 一致：每次调用 launch + sync（逐次同步）。
    // 注意：aclRfft1D 的 executor 不能复用，必须每轮重新 GetWorkspaceSize，
    //       这属于该算子的调用约定，计入耗时。
    auto t0 = clk::now();
    for(int i=0;i<reps;i++){
        aclOpExecutor* e=nullptr;
        aclRfft1DGetWorkspaceSize(x,n,1,1,y,&ws,&e);
        if(aclRfft1D(dws, ws, e, s)!=0){ printf("aclRfft1D failed in loop\n"); return 1; }
        CK(aclrtSynchronizeStream(s));
    }
    double us = since(t0)/reps;
    printf("aclRfft1D n=%lld b=%lld : %.1f us/call  (%.1f ms per %d reps)\n",
           (long long)n,(long long)b,us,us*reps/1000.0,reps);

    CK(aclrtMemcpy(ho.data(), ho.size()*4, dy, ho.size()*4, ACL_MEMCPY_DEVICE_TO_HOST));
    FILE* f=fopen(dump,"wb"); if(f){ fwrite(ho.data(),4,ho.size(),f); fclose(f); }
    printf("out[0..7]:");
    for(int i=0;i<8 && i<outN;i++) printf(" %.6g", ho[i]);
    printf("\nwrote %s (%zu floats)\n", dump, ho.size());
    return 0;
}
