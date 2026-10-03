// Phase 0.1: native CANN aclRfft1D baseline (float32 real -> float32 interleaved complex).
// usage: baseline_rfft <n> <batch> <norm 0|1|2> <dump.bin>
//   out shape per batch = 2*(n/2+1) floats (interleaved complex, onesided).
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/acl_rfft1d.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#define CK(x) do{ auto _e=(x); if(_e!=ACL_SUCCESS){ printf("ERR %d at %d\n",(int)_e,__LINE__); return 1;} }while(0)

int main(int argc, char** argv){
    if(argc<5){ printf("usage: %s <n> <batch> <norm> <dump.bin>\n", argv[0]); return 2; }
    int64_t n = atoll(argv[1]), b = atoll(argv[2]), norm = atoll(argv[3]);
    const char* dump = argv[4];
    int64_t modes = n/2+1;
    int64_t outN = modes*2;   // out shape [b, modes, 2]

    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s=nullptr; CK(aclrtCreateStream(&s));

    std::vector<float> h((size_t)(b*n));
    for(int64_t i=0;i<b*n;i++) h[i] = std::sin(0.017*i) + 0.3f*std::cos(0.005*i);
    std::vector<float> ho((size_t)(b*outN), -1.f);

    void *dx=nullptr,*dy=nullptr,*dws=nullptr;
    CK(aclrtMalloc(&dx, h.size()*4, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dy, ho.size()*4, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMemcpy(dx, h.size()*4, h.data(), h.size()*4, ACL_MEMCPY_HOST_TO_DEVICE));

    int64_t inDims[2]={b,n}, outDims[3]={b,modes,2};
    aclTensor* x = aclCreateTensor(inDims,2,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,inDims,2,dx);
    aclTensor* y = aclCreateTensor(outDims,3,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,outDims,3,dy);
    if(!x||!y){ printf("tensor create failed\n"); return 1; }

    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    aclnnStatus st = aclRfft1DGetWorkspaceSize(x,n,1,1,y,&ws,&ex);
    printf("GetWorkspaceSize status=%d workspace=%llu\n", (int)st, (unsigned long long)ws);
    if(st!=0) return 1;
    if(ws){ CK(aclrtMalloc(&dws, ws, ACL_MEM_MALLOC_NORMAL_ONLY)); }

    if(norm==0) printf("WARNING: norm=0 rejected by the op; forcing 1\n");
    st = aclRfft1D(dws, ws, ex, s);
    if(st!=0){ printf("aclRfft1D status=%d\n",(int)st); return 1; }
    CK(aclrtSynchronizeStream(s));

    // 计时口径与 build/fft_check 一致：每次调用 launch + sync（逐次同步）。
    // 注意：aclRfft1D 的 executor 不能复用，必须每轮重新 GetWorkspaceSize，
    //       这属于该算子的调用约定，计入耗时。
    int reps = 50;
    auto t0 = std::chrono::steady_clock::now();
    for(int i=0;i<reps;i++){
        aclOpExecutor* e=nullptr;
        aclRfft1DGetWorkspaceSize(x,n,1,1,y,&ws,&e);
        if(aclRfft1D(dws, ws, e, s)!=0){ printf("aclRfft1D failed in loop\n"); return 1; }
        CK(aclrtSynchronizeStream(s));
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double,std::micro>(t1-t0).count()/reps;
    printf("aclRfft1D n=%lld b=%lld : %.1f us/call  (%.1f ms per %d reps)\n",
           (long long)n,(long long)b,us,us*reps/1000.0,reps);

    CK(aclrtMemcpy(ho.data(), ho.size()*4, dy, ho.size()*4, ACL_MEMCPY_DEVICE_TO_HOST));
    FILE* f=fopen(dump,"wb"); if(f){ fwrite(ho.data(),4,ho.size(),f); fclose(f); }
    printf("out[0..7]:");
    for(int i=0;i<8 && i<outN;i++) printf(" %.6g", ho[i]);
    printf("\nwrote %s (%zu floats)\n", dump, ho.size());
    return 0;
}
