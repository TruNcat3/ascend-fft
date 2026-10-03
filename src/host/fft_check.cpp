// Phase 1 acceptance harness: launch kfft_fwd and check against a double-precision CPU reference.
// usage: fft_check <n> <batch> [reps]
#include <acl/acl.h>
#include "butterfly/fft_k.hpp"
#include "butterfly/reference.hpp"
#include <algorithm>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#define CK(x) do{ auto _e=(x); if(_e!=ACL_SUCCESS){ printf("ERR %d at %d\n",(int)_e,__LINE__); return 1;} }while(0)

static uint32_t readArgSize(const char* path){
    FILE* f=fopen(path,"rb"); if(!f) return 0;
    fseek(f,0,SEEK_END); long flen=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<unsigned char> buf(flen);
    if(fread(buf.data(),1,flen,f)!=(size_t)flen){fclose(f);return 0;}
    fclose(f);
    if(flen<64 || *(uint32_t*)buf.data()!=0x464c457f) return 0;
    uint64_t shoff=*(uint64_t*)(buf.data()+40);
    uint16_t shentsize=*(uint16_t*)(buf.data()+58);
    uint16_t shnum=*(uint16_t*)(buf.data()+60);
    uint16_t shstrndx=*(uint16_t*)(buf.data()+62);
    if(!shoff||shstrndx>=shnum) return 0;
    auto sh=[&](int i){ return buf.data()+shoff+(uint64_t)i*shentsize; };
    uint64_t stroff=*(uint64_t*)(sh(shstrndx)+24);
    for(int i=0;i<shnum;i++){
        uint32_t nm=*(uint32_t*)(sh(i));
        if(stroff+nm>=(uint64_t)flen) continue;
        if(strcmp((const char*)(buf.data()+stroff+nm),"__CCE_KernelArgSize")==0){
            uint64_t off=*(uint64_t*)(sh(i)+24), sz=*(uint64_t*)(sh(i)+32);
            if(sz>=4&&off+4<=(uint64_t)flen) return *(uint32_t*)(buf.data()+off);
        }
    }
    return 0;
}

static void refFft(const float* in, std::complex<double>* y, uint32_t n){
    uint32_t logn=0; while((1u<<logn)<n) logn++;
    for(uint32_t i=0;i<n;i++){
        uint32_t j=0,t=i;
        for(uint32_t k=0;k<logn;k++){ j=(j<<1)|(t&1u); t>>=1; }
        y[j]=std::complex<double>(in[2*i], in[2*i+1]);
    }
    for(uint32_t s=0;s<logn;s++){
        uint32_t h=1u<<s;
        for(uint32_t base=0;base<n;base+=2*h){
            for(uint32_t j=0;j<h;j++){
                std::complex<double> w=std::exp(std::complex<double>(0.0,-2.0*M_PI*j/(2.0*h)));
                std::complex<double> a=y[base+j], b=y[base+h+j]*w;
                y[base+j]=a+b; y[base+h+j]=a-b;
            }
        }
    }
}

int main(int argc, char** argv){
    if(argc<3){ printf("usage: %s <n> <batch> [reps]\n", argv[0]); return 2; }
    const auto tInit0=std::chrono::steady_clock::now();
    uint32_t n=(uint32_t)atoi(argv[1]); uint32_t batch=(uint32_t)atoi(argv[2]);
    int reps = argc>3?atoi(argv[3]):10;
    if(n<8 || (n&(n-1))){ printf("n must be power of two >= 8\n"); return 2; }
    if(n<64){
        // 8-plane 布局：下面的索引生成是 if(n>=64) 才填，n<64 会拿着全 0 索引跑出垃圾
        printf("n must be >= 64 (8-plane layout: rows=n/K must be a multiple of 8)\n");
        return 2;
    }
    if((uint64_t)batch*2ull*(uint64_t)n > 0xFFFFFFFFull){
        // 内核 goff = b*2u*n 是 uint32（fft_radix2.cpp:97），b 可达 batch-1
        printf("batch*2*n exceeds uint32 goff (max batch for n=%u is %llu)\n",
               n, (unsigned long long)0xFFFFFFFFull/(2ull*(uint64_t)n));
        return 2;
    }
    uint32_t twPad=n+16;
    uint32_t elements=2u*n*batch;

    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s=nullptr; CK(aclrtCreateStream(&s));
    // boot = aclInit + SetDevice + CreateStream（进程级一次性）；plan = 之后的数据准备
    const auto tBoot1=std::chrono::steady_clock::now();

    std::vector<float> hIn(elements);
    for(uint32_t i=0;i<elements;i++)
        hIn[i]=(float)(std::sin(0.011*i)+0.25*std::cos(0.037*i));
    std::vector<float> hOut(elements,0.f);
    // plan 计时从这里起：跳过 hIn 测试输入的生成（host 上的 sin/cos 逐元素循环，
    // 大形状能到上百 ms，属于测试夹具而非 plan 准备）
    const auto tPlan0=std::chrono::steady_clock::now();

    // twiddles (planar, padded to twPad)
    std::vector<float> twr(twPad,0.f), twi(twPad,0.f);
    {
        uint32_t off=0;
        for(uint32_t h=1; h<=n/2; h<<=1){
            for(uint32_t j=0; j<h; j++){
                double ang = -M_PI*(double)j/(double)h;
                twr[off+j]=(float)std::cos(ang); twi[off+j]=(float)std::sin(ang);
            }
            off += (h+7u)&~7u;
        }
        if(off!=twPad) printf("WARN: twiddle slots %u != twPad %u\n", off, twPad);
    }

    // 三张索引合成一个 (4n + 2nD) 的 GM 张量：[idxB(n) | idxT(n) | idxOut(2nD)]，单位字节。
    // foldD：默认取 bfly::foldDFor(n, batch, 48)，与 launch 的 blocks=min(48,batch) 同口径，
    //   也与 src/framework/butterfly.cpp::Plan::prepare 同一规则（那边 nblk = cand.udCore）。
    //   AB_FOLD_D=<k> 可强制指定，仅供 A/B —— 核内 tmpF/idx 按传入值现算，任意 k>=1 都
    //   功能正确；k>1 而 rows=n/K>64 时核内 plane 段按 64 元素行切片（不变错，只是多一层循环）。
    uint32_t foldD = bfly::foldDFor(n, batch, 48u);
    if (const char* e = getenv("AB_FOLD_D")) {
        int v = atoi(e);
        if (v >= 1) foldD = (uint32_t)v;
    }
    const uint32_t D = foldD ? foldD : 1u;
    // AB_PLANE_K=<8|16|32> 覆盖平面 K（索引生成 + 打包进第 8 参高 8 位）。
    // 默认打包 planeKFor(n)，与内核不传高 8 位时按规则自选的结果相同；此钩子用来
    // 验证非默认分支：AB_PLANE_K=8 + AB_FOLD_D=4 @n=1024 => rows=128 => plane 段
    // 行切片（nRowSlice=2，A5 新增路径）。K 不能 < 8（planar 首级 h=K<8 时 32B 对齐
    // 失效，AIV 抛 507035），这正是 planeKFor 候选集从 8 起跳的原因。
    // 只对当前 8 参内核有效（遗留 v1/v2 不读第 8 参，索引会与核内 K 不一致）。
    uint32_t planeK = bfly::planeKFor(n);
    if (const char* e = getenv("AB_PLANE_K")) {
        int v = atoi(e);
        if (v == 8 || v == 16 || v == 32) planeK = (uint32_t)v;
    }
    const uint32_t packArg = foldD | (planeK << 8);
    const uint32_t idxN = 4u*n + 2u*n*D;
    std::vector<uint32_t> idxAll(idxN, 0u);
    if(n>=64){
        uint32_t K = planeK;
        uint32_t rows=n/K, logK=0; while((1u<<logK)<K) logK++;
        uint32_t logn=0; while((1u<<logn)<n) logn++;
        uint32_t mask=K-1u;
        for(uint32_t k=0;k<n;k++){
            uint32_t j=((k%rows)<<logK)|(k/rows), r=0, t=j;
            for(uint32_t b=0;b<logn;b++){ r=(r<<1)|(t&1u); t>>=1; }
            idxAll[k]   = 2u*r*4u;                                   // bitrev(invpos(k))
            idxAll[n+k] = (((k&mask)*rows)+(k>>logK))*4u;             // plane -> planar
        }
        for(uint32_t d=0;d<D;d++){                                    // planar -> 交错，逐批一段
            uint32_t base = 2u*n + d*2u*n, rd = d*n;
            for(uint32_t j=0;j<n;j++){
                idxAll[base+2u*j]     = 4u*(rd+j);
                idxAll[base+2u*j+1u]  = 4u*(D*n + rd+j);
            }
        }
    }
    void *dIn=nullptr,*dOut=nullptr,*dTwR=nullptr,*dTwI=nullptr,*dIdx=nullptr;
    CK(aclrtMalloc(&dIn,  elements*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dOut, elements*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dTwR, twPad*4u,     ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dTwI, twPad*4u,     ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dIdx, (size_t)idxN*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMemcpy(dIn, elements*4u, hIn.data(), elements*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dTwR, twPad*4u, twr.data(), twPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dTwI, twPad*4u, twi.data(), twPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dIdx, (size_t)idxN*4u, idxAll.data(), (size_t)idxN*4u, ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtBinHandle b=nullptr;
    CK(aclrtBinaryLoadFromFile(getenv("AB_FFT_O")?getenv("AB_FFT_O"):"build/fft_radix2.o", nullptr, &b));
    aclrtFuncHandle f=nullptr; CK(aclrtBinaryGetFunction(b,"kfft_fwd",&f));

    uint32_t blocks = batch<48u?batch:48u;
    // spec p p u u p p p u -> out,in,n,batch,twr,twi,idxGm(4n+2nD),foldD
    // 遗留内核 v1/v2 只有 7 参（argBytes<52）：AB_FFT_O 指向它们时不写第 8 参，
    // 否则 aclrtLaunchKernelWithHostArgs 会因实参块长度不符而拒收。
    const char* opath = getenv("AB_FFT_O")?getenv("AB_FFT_O"):"build/fft_radix2.o";
    uint32_t argBytes=readArgSize(opath);
    size_t used=56;
    if(argBytes==0){ printf("WARN: __CCE_KernelArgSize not found\n"); }
    else used=argBytes;
    const bool hasFold = (used>=52);
    std::vector<unsigned char> ab;
    ab.resize(used,0);
    uint64_t p[5]={(uint64_t)(uintptr_t)dOut,(uint64_t)(uintptr_t)dIn,
                   (uint64_t)(uintptr_t)dTwR,(uint64_t)(uintptr_t)dTwI,
                   (uint64_t)(uintptr_t)dIdx};
    memcpy(ab.data()+0, &p[0],8);
    memcpy(ab.data()+8, &p[1],8);
    memcpy(ab.data()+16,&n,4);
    memcpy(ab.data()+20,&batch,4);
    memcpy(ab.data()+24,&p[2],8);
    memcpy(ab.data()+32,&p[3],8);
    memcpy(ab.data()+40,&p[4],8);
    if(hasFold) memcpy(ab.data()+48,&packArg,4);   // 低 8 位 foldD | 高 8 位平面 K

    auto launch=[&](){
        aclError e=aclrtLaunchKernelWithHostArgs(f,blocks,s,nullptr,ab.data(),ab.size(),nullptr,0);
        if(e){ printf("launch=%d\n",(int)e); return false; }
        { aclError se=aclrtSynchronizeStream(s); if(se){ printf("sync=%d\n",(int)se); return false; } }
        return true;
    };

    // setup = 主机端准备（旋转因子/索引生成）+ 显存分配 + 首次上传 + 二进制加载，
    // 在冷启动 warmup 之前截断，供端到端报告一次性开销。
    const auto tSetup1=std::chrono::steady_clock::now();

    if(!launch()) return 1;

    // ---- 端到端口径（AB_E2E=1，默认关闭，不影响矩阵/门禁解析）----
    // 每次重复：H2D(输入) -> kfft_fwd -> D2H(输出)，与 bench_native_npu.py --e2e 的
    //   xd.copy_(x_cpu) -> torch.fft.fft(xd) -> y.cpu() 逐段对应（见 docs/实验对比.md）。
    // 计时边界含传输与同步，不含一次性 setup。
    if(const char* ee=getenv("AB_E2E")){
        int ereps = atoi(ee); if(ereps<=0) ereps = reps;
        // AB_E2E_MODE: async（默认，带流 memcpy + 显式同步）| sync（阻塞 aclrtMemcpy）
        //              | xfer（只做 H2D+D2H，不发射 kernel，用来隔离纯传输带宽）。
        const char* mode = getenv("AB_E2E_MODE");
        // async = H2D 后同步 + D2H 后同步（默认）
        // sync  = 阻塞 aclrtMemcpy（等价语义，两种实测带宽无稳定差异）
        // xfer  = 不发射 kernel，隔离纯 H2D+D2H 带宽（用于定位 E2E 中的传输占比）
        const bool useAsync = !(mode && strcmp(mode,"sync")==0);
        const bool xferOnly = (mode && strcmp(mode,"xfer")==0);
        double sumE=0, minE=0, firstE=0;
        const auto t0e=std::chrono::steady_clock::now();
        for(int i=0;i<ereps;i++){
            auto a0=std::chrono::steady_clock::now();
            if(useAsync){
                CK(aclrtMemcpyAsync(dIn, elements*4u, hIn.data(), elements*4u,
                                    ACL_MEMCPY_HOST_TO_DEVICE, s));
                CK(aclrtSynchronizeStream(s));
                if(!xferOnly && !launch()) return 1;
                CK(aclrtMemcpyAsync(hOut.data(), elements*4u, dOut, elements*4u,
                                    ACL_MEMCPY_DEVICE_TO_HOST, s));
                CK(aclrtSynchronizeStream(s));
            }else{
                CK(aclrtMemcpy(dIn, elements*4u, hIn.data(), elements*4u,
                               ACL_MEMCPY_HOST_TO_DEVICE));
                if(!xferOnly && !launch()) return 1;
                CK(aclrtMemcpy(hOut.data(), elements*4u, dOut, elements*4u,
                               ACL_MEMCPY_DEVICE_TO_HOST));
            }
            auto a1=std::chrono::steady_clock::now();
            double u=std::chrono::duration<double,std::micro>(a1-a0).count();
            sumE+=u; if(i==0){ firstE=u; minE=u; } else if(u<minE) minE=u;
        }
        double e=sumE/ereps;
        double bootUs=std::chrono::duration<double,std::micro>(tBoot1-tInit0).count();
        double planUs=std::chrono::duration<double,std::micro>(tSetup1-tPlan0).count();
        double spanUs=std::chrono::duration<double,std::micro>(t0e-tSetup1).count();
        printf("E2E n=%u batch=%u reps=%d e2e_us=%.1f e2e_min_us=%.1f first_us=%.1f "
               "boot_us=%.1f plan_us=%.1f warmup_us=%.1f mode=%s\n",
               n,batch,ereps,e,minE,firstE,bootUs,planUs,spanUs,useAsync?"async":"sync");
        fflush(stdout);
    }

    // 逐次 launch 计时：均值与最小值都报。均值与 §8.5 的 η 标定同口径，
    // 最小值与 bench_native_npu.py 的统计量同口径 —— 两边可以各自对齐比。
    double sumUs=0, minUs=0;
    for(int i=0;i<reps;i++){
        auto a0=std::chrono::steady_clock::now();
        if(!launch()) return 1;
        auto a1=std::chrono::steady_clock::now();
        double u=std::chrono::duration<double,std::micro>(a1-a0).count();
        sumUs+=u; if(i==0||u<minUs) minUs=u;
    }
    double us=sumUs/reps;

    CK(aclrtMemcpy(hOut.data(), elements*4u, dOut, elements*4u, ACL_MEMCPY_DEVICE_TO_HOST));

    // AB_DUMP=<prefix> 时导出输入/输出，供 scripts/bench_stdlib.py 与 torch/numpy 交叉验证
    if(const char* dp=getenv("AB_DUMP")){
        auto wr=[&](const std::string& suf, const std::vector<float>& v){
            std::string fn=std::string(dp)+suf; FILE* f=fopen(fn.c_str(),"wb");
            if(!f){ printf("dump open failed: %s\n", fn.c_str()); return; }
            fwrite(v.data(),4,v.size(),f); fclose(f); printf("dumped %s (%zu floats)\n", fn.c_str(), v.size());
        };
        wr(".in.bin", hIn); wr(".out.bin", hOut);
    }

    // 判据：与 bfly::maxRelScaled 完全同一口径（分母 = max|ref| 按 float 分量，
    // 口径统一说明见 include/butterfly/reference.hpp）。maxAbs/worst 仍按复数模打印，
    // 只作诊断、不参与 PASS/FAIL。
    double maxAbs=0, maxRel=0;
    uint32_t worst=0;
    std::vector<std::complex<double>> y(n);
    std::vector<float> refFlat(2*n);
    for(uint32_t bi=0;bi<batch;bi++){
        refFft(&hIn[bi*2*n], y.data(), n);
        for(uint32_t j=0;j<n;j++){
            refFlat[2*j]  =(float)y[j].real();
            refFlat[2*j+1]=(float)y[j].imag();
        }
        for(uint32_t j=0;j<n;j++){
            double gr=hOut[(bi*2*n)+2*j], gi=hOut[(bi*2*n)+2*j+1];
            double e=std::abs(std::complex<double>(gr,gi)-y[j]);
            if(e>maxAbs){ maxAbs=e; worst=bi*n+j; }
        }
        maxRel=std::max(maxRel, bfly::maxRelScaled(&hOut[bi*2*n], refFlat.data(), 2*n));
    }
    printf("n=%u batch=%u blocks=%u  maxAbs=%.3e maxRel=%.3e (worst idx %u)\n",
           n,batch,blocks,maxAbs,maxRel,worst);
    printf("kfft_fwd: %.1f us/call over %d reps (min %.1f us)\n", us, reps, minUs);
    printf("%s\n", maxRel<=1e-4?"PASS":"FAIL");
    return maxRel<=1e-4?0:1;
}
