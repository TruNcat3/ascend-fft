// Phase 1 acceptance harness: launch kfft_fwd（c2c）或实数变换链（r2c/c2r），
// 与双精度 CPU 参考逐点比对。
// usage: fft_check <n> <batch> [reps]
//
//   AB_DIR   = c2c（默认）| r2c | c2r   选择变换方向（内核见 src/ascendc/fft_real.cpp）
//              c2c : kfft_fwd(n)
//              r2c : kfft_fwd(n/2)（实输入零拷贝复用为交错复数）-> kfft_r2c_post
//              c2r : kfft_c2r_prep -> kfft_fwd(n)（旋转因子取反，直接算 n·IDFT）
//                    -> kfft_c2r_post
//              r2c 要求 n>=128（内层 FFT n/2 必须满足 8-plane 布局 rows=n/K>=8）
//   AB_FFT_O   kfft_fwd 的 .o（默认 build/fft_radix2.o）
//   AB_REAL_O  实数内核的 .o（默认 build/fft_real.o，仅 r2c/c2r 需要）
//   其余环境变量（AB_INPUT/AB_E2E/AB_DUMP/AB_FOLD_D/AB_PLANE_K/AB_FFT_O ...）语义不变，
//   其中 AB_FOLD_D/AB_PLANE_K 作用在「内层 FFT」的长度上（r2c 时为 n/2）。
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

// ---- 输入模式（AB_INPUT，默认 sin，保持既有门禁/矩阵口径不变）------------
// 三类典型应用共用同一套确定性 PRNG，公式与 scripts/bench_native_npu.py 的
// --input 实现逐位一致（xorshift32 三轮，seed = b*2654435761 + k*40503）：
//   ofdm  16-QAM 子载波（DC/保护带置零、每 12 个子载波一个导频）—— 多载波通信
//   radar 4 个目标的复指数距离回波（逐脉冲多普勒相位）            —— 雷达距离门
//   dl    每 8 点一块的复激活（幅度 + 4PSK 相位）                 —— 深度学习频域层
// 输出仍与下面的双精度 refFft 逐点比对，PASS/FAIL 口径不变。
static inline uint32_t appHash(uint32_t b, uint32_t k){
    uint32_t x = b*2654435761u + k*40503u;
    x ^= x<<13; x ^= x>>17; x ^= x<<5;
    x ^= x<<13; x ^= x>>17; x ^= x<<5;
    return x;
}

static bool isInputMode(const char* mode){
    static const char* modes[] = {
        "sin", "deterministic-sin-cos", "zero", "impulse", "constant", "tone",
        "single-tone", "alternating", "nyquist-alternating", "random",
        "random-seeded", "dynamic", "high-dynamic-range", "near-cancellation",
        "ofdm", "radar", "dl"
    };
    for(const char* candidate : modes)
        if(strcmp(mode, candidate)==0) return true;
    return false;
}

// 数值回归向量。每个模式都是确定性的，便于失败后原样复现。
static void genNumericalInput(std::vector<float>& hIn, uint32_t n, uint32_t batch,
                              const char* mode){
    const uint32_t seed = getenv("AB_SEED") ? (uint32_t)strtoul(getenv("AB_SEED"), nullptr, 0) : 0u;
    for(uint32_t b=0;b<batch;b++){
        const uint32_t toneBin = 1u + b%7u;
        for(uint32_t k=0;k<n;k++){
            float re=0.f, im=0.f;
            if(strcmp(mode,"impulse")==0){
                re = (k == b%n) ? 1.f : 0.f;
            }else if(strcmp(mode,"constant")==0){
                re = 0.25f; im = -0.5f;
            }else if(strcmp(mode,"tone")==0 || strcmp(mode,"single-tone")==0){
                const double phase = 2.0*M_PI*(double)toneBin*(double)k/(double)n;
                re = (float)std::cos(phase); im = (float)std::sin(phase);
            }else if(strcmp(mode,"alternating")==0 || strcmp(mode,"nyquist-alternating")==0){
                re = (k&1u) ? -1.f : 1.f;
            }else if(strcmp(mode,"random")==0 || strcmp(mode,"random-seeded")==0){
                const uint32_t x1=appHash(b^seed,k), x2=appHash(b^seed,k^0x9e37u);
                re = (float)((int32_t)(x1&0xffffu)-32768)*(1.f/32768.f);
                im = (float)((int32_t)(x2&0xffffu)-32768)*(1.f/32768.f);
            }else if(strcmp(mode,"dynamic")==0 || strcmp(mode,"high-dynamic-range")==0){
                static const float scale[4] = {1.f, 1e-2f, 1e-4f, 1e-6f};
                const float s = scale[k&3u];
                re = (k&1u) ? -s : s;
                im = (k&2u) ? 0.5f*s : -0.5f*s;
            }else if(strcmp(mode,"near-cancellation")==0){
                re = (k&1u) ? -1.f+1e-6f : 1.f;
                im = (k&1u) ? 0.5f-1e-6f : -0.5f;
            }
            hIn[2u*((size_t)b*n+k)]   = re;
            hIn[2u*((size_t)b*n+k)+1] = im;
        }
    }
}

static void genAppInput(std::vector<float>& hIn, uint32_t n, uint32_t batch,
                        const char* mode){
    static const float q[4] = {-3.f, -1.f, 1.f, 3.f};
    const bool ofdm  = strcmp(mode,"ofdm")==0;
    const bool radar = strcmp(mode,"radar")==0;
    // dl 是第三种（都不匹配时的分支）
    for(uint32_t b=0;b<batch;b++){
        for(uint32_t k=0;k<n;k++){
            uint32_t x = appHash(b,k);
            float re=0.f, im=0.f;
            if(ofdm){
                bool guard = (k==0u) || (k < n/16u) || (k >= n-n/16u);
                if(guard){ re=0.f; im=0.f; }
                else if((k%12u)==0u){                    // 导频：QPSK、单位幅度
                    re = (x&1u)?1.f:-1.f; im = (x&2u)?1.f:-1.f;
                }else{                                    // 16-QAM，功率归一 1/sqrt(10)
                    re = q[(x>>0)&3u]*0.31622776f; im = q[(x>>2)&3u]*0.31622776f;
                }
            }else if(radar){
                static const float fr[4] = {1.f/16, 1.f/5, 1.f/3, 0.62f};
                static const float am[4] = {1.f, 0.5f, 0.25f, 0.125f};
                static const float dop[4] = {0.f, 1.f/64, -1.f/128, 3.f/256};
                double sr=0, si=0;
                for(int t=0;t<4;t++){
                    double ph = 2.0*M_PI*(fr[t]*(double)k + dop[t]*(double)b)
                              + (t*0.7 + ((x>>8)&255u)/255.0*0.3);
                    sr += am[t]*std::cos(ph); si += am[t]*std::sin(ph);
                }
                re = 0.25f*(float)sr; im = 0.25f*(float)si;
            }else{                                        // dl：8 点一块的复激活
                uint32_t xb = appHash(b, k/8u);
                float mag = (float)((xb>>8)&255u)/255.0f;
                uint32_t phs = (xb>>16)&3u;
                static const float c[4] = {1.f, 0.f, -1.f, 0.f};
                static const float s[4] = {0.f, 1.f, 0.f, -1.f};
                re = mag*c[phs]; im = mag*s[phs];
            }
            hIn[2u*((size_t)b*n+k)]   = re;
            hIn[2u*((size_t)b*n+k)+1] = im;
        }
    }
}

// c2r 输入：任意半谱（确定性 hash，行距 n+16 float，Nyquist 虚部置 0）。
// 非 Hermitian 一致的中间 bin 也照收 —— prep 按镜像规则补全满谱，参考实现同式，
// 因此这是对展开逻辑本身的更强测试（不是「输入即合法 rfft 输出」的恒等检验）。
static void genHalfInput(std::vector<float>& h, uint32_t n, uint32_t batch){
    const uint32_t m = n >> 1, st = n + 16u;
    for(uint32_t b=0;b<batch;b++){
        for(uint32_t k=0;k<=m;k++){
            uint32_t x1 = appHash(b, k);
            uint32_t x2 = appHash(b, k ^ 0x5bd1u);
            float re = (float)((int32_t)(x1 & 0xFFFFu) - 32768) * (1.0f/32768.0f);
            float im = (k==m) ? 0.f
                     : (float)((int32_t)(x2 & 0xFFFFu) - 32768) * (1.0f/32768.0f);
            h[(size_t)b*st + 2u*k]     = re;
            h[(size_t)b*st + 2u*k + 1u] = im;
        }
    }
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
    if(argc<3){ printf("usage: %s <n> <batch> [reps]   (AB_DIR=c2c|r2c|c2r)\n", argv[0]); return 2; }
    const auto tInit0=std::chrono::steady_clock::now();
    uint32_t n=(uint32_t)atoi(argv[1]); uint32_t batch=(uint32_t)atoi(argv[2]);
    int reps = argc>3?atoi(argv[3]):10;
    if(n<8 || (n&(n-1))){ printf("n must be power of two >= 8\n"); return 2; }
    if(n<64){
        // 8-plane 布局：下面的索引生成是 if(n>=64) 才填，n<64 会拿着全 0 索引跑出垃圾
        printf("n must be >= 64 (8-plane layout: rows=n/K must be a multiple of 8)\n");
        return 2;
    }
    // ---- 变换方向（AB_DIR）----
    const char* dir = getenv("AB_DIR") ? getenv("AB_DIR") : "c2c";
    const bool isR2C = strcmp(dir,"r2c")==0;
    const bool isC2R = strcmp(dir,"c2r")==0;
    const bool isReal = isR2C || isC2R;
    if(!isReal && strcmp(dir,"c2c")!=0){
        printf("AB_DIR must be c2c | r2c | c2r (got %s)\n", dir); return 2;
    }
    if(isR2C && n<128){
        printf("r2c requires n >= 128 (inner FFT n/2 must satisfy rows=n/K >= 8)\n");
        return 2;
    }
    // UB 预算（见 fft_radix2.cpp InitBuffer）：fwd @n=8192 ≈389KB > 192KB 会 507035。
    // r2c 的 fwd 走半长 => n<=8192（inner<=4096）；c2r 的 fwd 走全长 => n<=4096。
    if(isC2R && n>4096){ printf("c2r requires n <= 4096 (fwd runs at full n, UB budget)\n"); return 2; }
    if(isR2C && n>8192){ printf("r2c requires n <= 8192 (inner n/2 <= 4096)\n"); return 2; }
    if(!isReal && n>4096){ printf("c2c requires n <= 4096 (UB budget)\n"); return 2; }
    const uint32_t innerN = isR2C ? (n>>1) : n;   // kfft_fwd 的长度（r2c 走半长）
    const uint32_t hsSt   = n + 16u;              // 半谱行距（float，见 fft_real.cpp）
    const uint32_t inSt   = isR2C ? n : (isC2R ? hsSt : 2u*n);
    const uint32_t outSt  = isR2C ? hsSt : (isC2R ? n : 2u*n);
    const size_t inElems  = (size_t)inSt * batch;
    const size_t outElems = (size_t)outSt * batch;
    const size_t elements = 2ull*(size_t)n*batch; // 复数张量（满谱 / 复 FFT 中间量）
    if((uint64_t)batch*2ull*(uint64_t)n > 0xFFFFFFFFull){
        // 内核 goff = b*2u*n 是 uint32（fft_radix2.cpp:97），b 可达 batch-1
        printf("batch*2*n exceeds uint32 goff (max batch for n=%u is %llu)\n",
               n, (unsigned long long)0xFFFFFFFFull/(2ull*(uint64_t)n));
        return 2;
    }
    const uint32_t twPad=innerN+16;

    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s=nullptr; CK(aclrtCreateStream(&s));
    // boot = aclInit + SetDevice + CreateStream（进程级一次性）；plan = 之后的数据准备
    const auto tBoot1=std::chrono::steady_clock::now();

    std::vector<float> hIn(inElems, 0.f);
    // AB_INPUT=sin（默认）| 数值回归模式 | ofdm | radar | dl。
    // c2r 固定用 genHalfInput（任意半谱），AB_INPUT 对它不生效。
    const char* inMode = getenv("AB_INPUT") ? getenv("AB_INPUT") : "sin";
    if(!isInputMode(inMode)){
        printf("unsupported AB_INPUT mode: %s\n", inMode);
        return 2;
    }
    if(isC2R){
        genHalfInput(hIn, n, batch);
    }else if(isR2C){
        if(strcmp(inMode,"sin")==0 || strcmp(inMode,"deterministic-sin-cos")==0){
            for(size_t i=0;i<inElems;i++)
                hIn[i]=(float)(std::sin(0.011*i)+0.25*std::cos(0.037*i));
        }else if(strcmp(inMode,"ofdm")==0 || strcmp(inMode,"radar")==0 ||
                 strcmp(inMode,"dl")==0){
            std::vector<float> cx(elements);           // 应用形状是复数的，取实部当实输入
            genAppInput(cx, n, batch, inMode);
            for(size_t i=0;i<inElems;i++) hIn[i]=cx[2*i];
        }else{
            std::vector<float> cx(elements);
            genNumericalInput(cx, n, batch, inMode);
            for(size_t i=0;i<inElems;i++) hIn[i]=cx[2*i];
        }
    }else{
        if(strcmp(inMode,"sin")==0 || strcmp(inMode,"deterministic-sin-cos")==0){
            for(size_t i=0;i<elements;i++)
                hIn[i]=(float)(std::sin(0.011*i)+0.25*std::cos(0.037*i));
        }else if(strcmp(inMode,"ofdm")==0 || strcmp(inMode,"radar")==0 ||
                 strcmp(inMode,"dl")==0){
            genAppInput(hIn, n, batch, inMode);
        }else{
            genNumericalInput(hIn, n, batch, inMode);
        }
    }
    std::vector<float> hOut(outElems,0.f);
    // AB_INPUT_FILE=<path>：整块覆盖 hIn（任意模式），供链路中间量交叉验证。
    if(const char* inf=getenv("AB_INPUT_FILE")){
        FILE* fp=fopen(inf,"rb");
        if(!fp || fread(hIn.data(),4,inElems,fp)!=inElems){
            printf("AB_INPUT_FILE read failed: %s\n", inf); return 2;
        }
        fclose(fp);
    }
    // plan 计时从这里起：跳过 hIn 测试输入的生成（host 上的 sin/cos 逐元素循环，
    // 大形状能到上百 ms，属于测试夹具而非 plan 准备）
    const auto tPlan0=std::chrono::steady_clock::now();

    // twiddles (planar, padded to twPad)，按内层 FFT 长度 innerN 生成。
    // c2r：符号取反（e^{+2πi·}）=> kfft_fwd 直接输出 G(X)=n·IDFT(X)（见 fft_real.cpp 头注）。
    // AB_SGN=+1|-1：强制覆盖旋转因子符号（默认 c2r 取反、其余正向），供调试。
    // sgn 还打包进 packArg 第 16 位（内核 xflip：交叉旋转 ±i 互换，见 fft_radix2.cpp）。
    double sgn = isC2R ? 1.0 : -1.0;
    if(const char* se=getenv("AB_SGN")) sgn = (atof(se)>0)?1.0:-1.0;
    std::vector<float> twr(twPad,0.f), twi(twPad,0.f);
    {
        uint32_t off=0;
        for(uint32_t h=1; h<=innerN/2; h<<=1){
            for(uint32_t j=0; j<h; j++){
                double ang = sgn*M_PI*(double)j/(double)h;
                twr[off+j]=(float)std::cos(ang); twi[off+j]=(float)std::sin(ang);
            }
            off += (h+7u)&~7u;
        }
        if(off!=twPad) printf("WARN: twiddle slots %u != twPad %u\n", off, twPad);
    }

    // r2c 后处理的旋转因子 W_n^k = e^{-2πik/n}，k ∈ [0, n/4]（8 倍数上取整）
    std::vector<float> pwr, pwi;
    uint32_t pPad=0;
    if(isR2C){
        const uint32_t pk1=(n>>2)+1u;
        pPad=(pk1+7u)&~7u;
        pwr.assign(pPad,0.f); pwi.assign(pPad,0.f);
        for(uint32_t k=0;k<pk1;k++){
            double ang=-2.0*M_PI*(double)k/(double)n;
            pwr[k]=(float)std::cos(ang); pwi[k]=(float)std::sin(ang);
        }
    }

    // 三张索引合成一个 (4n + 2nD) 的 GM 张量：[idxB(n) | idxT(n) | idxOut(2nD)]，单位字节。
    // foldD：默认取 bfly::foldDFor(n, batch, 48)，与 launch 的 blocks=min(48,batch) 同口径，
    //   也与 src/framework/butterfly.cpp::Plan::prepare 同一规则（那边 nblk = cand.udCore）。
    //   AB_FOLD_D=<k> 可强制指定，仅供 A/B —— 核内 tmpF/idx 按传入值现算，任意 k>=1 都
    //   功能正确；k>1 而 rows=n/K>64 时核内 plane 段按 64 元素行切片（不变错，只是多一层循环）。
    uint32_t foldD = bfly::foldDFor(innerN, batch, 48u);
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
    uint32_t planeK = bfly::planeKFor(innerN);
    if (const char* e = getenv("AB_PLANE_K")) {
        int v = atoi(e);
        if (v == 8 || v == 16 || v == 32) planeK = (uint32_t)v;
    }
    const uint32_t packArg = foldD | (planeK << 8) | ((sgn > 0.0) ? (1u << 16) : 0u);
    const uint32_t idxN = 4u*innerN + 2u*innerN*D;
    std::vector<uint32_t> idxAll(idxN, 0u);
    if(innerN>=64){
        uint32_t K = planeK;
        uint32_t rows=innerN/K, logK=0; while((1u<<logK)<K) logK++;
        uint32_t logn=0; while((1u<<logn)<innerN) logn++;
        uint32_t mask=K-1u;
        for(uint32_t k=0;k<innerN;k++){
            uint32_t j=((k%rows)<<logK)|(k/rows), r=0, t=j;
            for(uint32_t b=0;b<logn;b++){ r=(r<<1)|(t&1u); t>>=1; }
            idxAll[k]   = 2u*r*4u;                                        // bitrev(invpos(k))
            idxAll[innerN+k] = (((k&mask)*rows)+(k>>logK))*4u;             // plane -> planar
        }
        for(uint32_t d=0;d<D;d++){                                        // planar -> 交错，逐批一段
            uint32_t base = 2u*innerN + d*2u*innerN, rd = d*innerN;
            for(uint32_t j=0;j<innerN;j++){
                idxAll[base+2u*j]     = 4u*(rd+j);
                idxAll[base+2u*j+1u]  = 4u*(D*innerN + rd+j);
            }
        }
    }
    void *dIn=nullptr,*dOut=nullptr,*dA=nullptr,*dB=nullptr,*dTwR=nullptr,*dTwI=nullptr,
         *dIdx=nullptr,*dPW=nullptr;
    CK(aclrtMalloc(&dIn,  inElems*4u,  ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dOut, outElems*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    if(isR2C){                                    // Z：内层 FFT 输出（n float/行）
        CK(aclrtMalloc(&dA, inElems*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
        CK(aclrtMalloc(&dPW, (size_t)2u*pPad*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    }else if(isC2R){                              // A：满谱；B：kfft_fwd 反号输出
        CK(aclrtMalloc(&dA, elements*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
        CK(aclrtMalloc(&dB, elements*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    }
    CK(aclrtMalloc(&dTwR, twPad*4u,     ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dTwI, twPad*4u,     ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dIdx, (size_t)idxN*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMemcpy(dIn, inElems*4u, hIn.data(), inElems*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dTwR, twPad*4u, twr.data(), twPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dTwI, twPad*4u, twi.data(), twPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dIdx, (size_t)idxN*4u, idxAll.data(), (size_t)idxN*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    if(isR2C){
        CK(aclrtMemcpy(dPW, (size_t)pPad*4u, pwr.data(), (size_t)pPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy((char*)dPW + (size_t)pPad*4u, (size_t)pPad*4u,
                       pwi.data(), (size_t)pPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    aclrtBinHandle b=nullptr;
    const char* opath = getenv("AB_FFT_O")?getenv("AB_FFT_O"):"build/fft_radix2.o";
    CK(aclrtBinaryLoadFromFile(opath, nullptr, &b));
    aclrtFuncHandle f=nullptr; CK(aclrtBinaryGetFunction(b,"kfft_fwd",&f));

    // 实数内核（同一 .o 内三个内核同签名，argSize=40）
    aclrtBinHandle rb=nullptr;
    aclrtFuncHandle fPrep=nullptr, fPost=nullptr;
    if(isReal){
        const char* ropath = getenv("AB_REAL_O")?getenv("AB_REAL_O"):"build/fft_real.o";
        CK(aclrtBinaryLoadFromFile(ropath, nullptr, &rb));
        if(isR2C) CK(aclrtBinaryGetFunction(rb,"kfft_r2c_post",&fPost));
        else{
            CK(aclrtBinaryGetFunction(rb,"kfft_c2r_prep",&fPrep));
            CK(aclrtBinaryGetFunction(rb,"kfft_c2r_post",&fPost));
        }
    }

    uint32_t blocks = batch<48u?batch:48u;
    // spec p p u u p p p u -> out,in,n,batch,twr,twi,idxGm(4n+2nD),foldD
    // 遗留内核 v1/v2 只有 7 参（argBytes<52）：AB_FFT_O 指向它们时不写第 8 参，
    // 否则 aclrtLaunchKernelWithHostArgs 会因实参块长度不符而拒收。
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
    memcpy(ab.data()+16,&innerN,4);
    memcpy(ab.data()+20,&batch,4);
    memcpy(ab.data()+24,&p[2],8);
    memcpy(ab.data()+32,&p[3],8);
    memcpy(ab.data()+40,&p[4],8);
    if(hasFold) memcpy(ab.data()+48,&packArg,4);   // 低 8 位 foldD | 高 8 位平面 K

    // 实数内核实参：out, in, ex0, ex1, n, batch（=40B，fft_real.o 的 argSize）。
    // ex0/ex1：r2c_post 是旋转因子表 wrr/wri（dPW）；prep/post 不用，传合法指针过校验。
    std::vector<unsigned char> abR(40,0);
    const uint64_t dMy=(uint64_t)(uintptr_t)dIn;    // dummy 也给合法指针，避免空指针校验
    auto setReal=[&](void* dst, void* src, void* ex0=nullptr, void* ex1=nullptr){
        uint64_t q0=(uint64_t)(uintptr_t)dst, q1=(uint64_t)(uintptr_t)src;
        uint64_t q2=ex0?(uint64_t)(uintptr_t)ex0:dMy;
        uint64_t q3=ex1?(uint64_t)(uintptr_t)ex1:dMy;
        memcpy(abR.data()+0, &q0,8);
        memcpy(abR.data()+8, &q1,8);
        memcpy(abR.data()+16,&q2,8);
        memcpy(abR.data()+24,&q3,8);
        memcpy(abR.data()+32,&n,4);
        memcpy(abR.data()+36,&batch,4);
    };
    auto issueFwd=[&](void* dst, void* src)->aclError{
        uint64_t q0=(uint64_t)(uintptr_t)dst, q1=(uint64_t)(uintptr_t)src;
        memcpy(ab.data()+0,&q0,8);
        memcpy(ab.data()+8,&q1,8);
        return aclrtLaunchKernelWithHostArgs(f,blocks,s,nullptr,ab.data(),ab.size(),nullptr,0);
    };
    auto issueReal=[&](aclrtFuncHandle fh, void* dst, void* src,
                       void* ex0=nullptr, void* ex1=nullptr)->aclError{
        setReal(dst, src, ex0, ex1);
        return aclrtLaunchKernelWithHostArgs(fh,blocks,s,nullptr,abR.data(),abR.size(),nullptr,0);
    };
    // 一次 launch = 整条方向链（多内核在同一流上串行），末尾一次同步 => 计的是整链 device 时间
    auto launch=[&](){
        aclError e=ACL_SUCCESS;
        if(isR2C)      { e=issueFwd(dA,dIn);
                         if(!e) e=issueReal(fPost,dOut,dA, dPW,(char*)dPW+(size_t)pPad*4); }
        else if(isC2R) { e=issueReal(fPrep,dA,dIn);     if(!e) e=issueFwd(dB,dA);
                         if(!e) e=issueReal(fPost,dOut,dB); }
        else           { e=issueFwd(dOut,dIn); }
        if(e){ printf("launch=%d\n",(int)e); return false; }
        { aclError se=aclrtSynchronizeStream(s); if(se){ printf("sync=%d\n",(int)se); return false; } }
        return true;
    };

    // setup = 主机端准备（旋转因子/索引生成）+ 显存分配 + 首次上传 + 二进制加载，
    // 在冷启动 warmup 之前截断，供端到端报告一次性开销。
    const auto tSetup1=std::chrono::steady_clock::now();

    if(!launch()) return 1;

    // ---- 端到端口径（AB_E2E=1，默认关闭，不影响矩阵/门禁解析）----
    // 每次重复：H2D(输入) -> 变换链 -> D2H(输出)，与 bench_native_npu.py --e2e 的
    //   xd.copy_(x_cpu) -> torch.fft.fft(xd) -> yd.copy_(out) 逐段对应，
    //   三路主机缓冲同为 pinned（见 docs/实验对比.md §6.2/§6.3）。
    // 计时边界含传输与同步，不含一次性 setup。
    if(const char* ee=getenv("AB_E2E")){
        int ereps = atoi(ee); if(ereps<=0) ereps = reps;
        // AB_E2E_MODE: async（默认，带流 memcpy + 显式同步）| sync（阻塞 aclrtMemcpy）
        //              | xfer（只做 H2D+D2H，不发射 kernel，用来隔离纯传输带宽）。
        // AB_E2E_HOST: pinned（默认，aclrtMallocHost）| pageable（受限口径，仅 A/B 对照）。
        //              pageable 的 H2D/D2H 走主机侧 staging + 缺页，带宽随 loadavg 摆
        //              2~4×（最大点 E2E 7.2 ms -> 27.2 ms，见 docs/实验对比.md §6.3），
        //              会让端到端口径把「我们没用最优拷贝路径」记成 kernel 的账。
        const char* mode = getenv("AB_E2E_MODE");
        // async = H2D 后同步 + D2H 后同步（默认）
        // sync  = 阻塞 aclrtMemcpy（等价语义，两种实测带宽无稳定差异）
        // xfer  = 不发射 kernel，隔离纯 H2D+D2H 带宽（用于定位 E2E 中的传输占比）
        const bool useAsync = !(mode && strcmp(mode,"sync")==0);
        const bool xferOnly = (mode && strcmp(mode,"xfer")==0);
        const char* hostMode = getenv("AB_E2E_HOST");
        const bool wantPinned = !(hostMode && strcmp(hostMode,"pageable")==0);
        float* pIn = nullptr; float* pOut = nullptr;
        if(wantPinned){
            void *a=nullptr, *b=nullptr;
            const size_t nbi=inElems*4u, nbo=outElems*4u;
            const int ei=aclrtMallocHost(&a, nbi), eo=aclrtMallocHost(&b, nbo);
            if(ei==ACL_SUCCESS && eo==ACL_SUCCESS){
                pIn=(float*)a; pOut=(float*)b;
                memcpy(pIn, hIn.data(), nbi);
            }else{
                if(ei==ACL_SUCCESS) aclrtFreeHost(a);
                if(eo==ACL_SUCCESS) aclrtFreeHost(b);
                printf("E2E host=pinned alloc failed (%d/%d) -> pageable\n", ei, eo);
            }
        }
        const float* src = pIn ? pIn : hIn.data();
        float*       dst = pOut ? pOut : hOut.data();
        const char*  hostTag = pIn ? "pinned" : "pageable";
        double sumE=0, minE=0, firstE=0;
        const auto t0e=std::chrono::steady_clock::now();
        for(int i=0;i<ereps;i++){
            auto a0=std::chrono::steady_clock::now();
            if(useAsync){
                CK(aclrtMemcpyAsync(dIn, inElems*4u, src, inElems*4u,
                                    ACL_MEMCPY_HOST_TO_DEVICE, s));
                CK(aclrtSynchronizeStream(s));
                if(!xferOnly && !launch()) return 1;
                CK(aclrtMemcpyAsync(dst, outElems*4u, dOut, outElems*4u,
                                    ACL_MEMCPY_DEVICE_TO_HOST, s));
                CK(aclrtSynchronizeStream(s));
            }else{
                CK(aclrtMemcpy(dIn, inElems*4u, src, inElems*4u,
                               ACL_MEMCPY_HOST_TO_DEVICE));
                if(!xferOnly && !launch()) return 1;
                CK(aclrtMemcpy(dst, outElems*4u, dOut, outElems*4u,
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
               "boot_us=%.1f plan_us=%.1f warmup_us=%.1f mode=%s host=%s input=%s\n",
               n,batch,ereps,e,minE,firstE,bootUs,planUs,spanUs,
               useAsync?"async":"sync", hostTag, inMode);
        fflush(stdout);
        if(pIn){ aclrtFreeHost(pIn); aclrtFreeHost(pOut); }
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

    CK(aclrtMemcpy(hOut.data(), outElems*4u, dOut, outElems*4u, ACL_MEMCPY_DEVICE_TO_HOST));

    // AB_DUMP=<prefix> 时导出输入/输出，供 scripts/bench_stdlib.py 与 torch/numpy 交叉验证
    if(const char* dp=getenv("AB_DUMP")){
        auto wr=[&](const std::string& suf, const std::vector<float>& v){
            std::string fn=std::string(dp)+suf; FILE* f=fopen(fn.c_str(),"wb");
            if(!f){ printf("dump open failed: %s\n", fn.c_str()); return; }
            fwrite(v.data(),4,v.size(),f); fclose(f); printf("dumped %s (%zu floats)\n", fn.c_str(), v.size());
        };
        wr(".in.bin", hIn); wr(".out.bin", hOut);
    }
    // AB_DUMP2=<prefix>：导出链路中间量（r2c: a=内层FFT输出；c2r: a=prep 满谱, b=fwd 输出）
    if(const char* dp=getenv("AB_DUMP2")){
        auto wrv=[&](const std::string& suf, void* dev, size_t bytes){
            std::vector<unsigned char> tmp(bytes);
            if(aclrtMemcpy(tmp.data(),bytes,dev,bytes,ACL_MEMCPY_DEVICE_TO_HOST)){
                printf("dump2 d2h failed\n"); return;
            }
            std::string fn=std::string(dp)+suf; FILE* f=fopen(fn.c_str(),"wb");
            if(!f){ printf("dump2 open failed: %s\n", fn.c_str()); return; }
            fwrite(tmp.data(),1,bytes,f); fclose(f); printf("dumped %s (%zu bytes)\n", fn.c_str(), bytes);
        };
        if(isR2C)     wrv(".a.bin", dA, inElems*4u);
        else if(isC2R){ wrv(".a.bin", dA, elements*4u); wrv(".b.bin", dB, elements*4u); }
    }

    // 判据：与 bfly::maxRelScaled 完全同一口径（分母 = max|ref| 按 float 分量，
    // 口径统一说明见 include/butterfly/reference.hpp）。maxAbs/worst 仍按复数模打印，
    // 只作诊断、不参与 PASS/FAIL。
    double maxAbs=0, maxRel=0;
    uint32_t worst=0;
    if(!isReal){
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
    }else if(isR2C){
        // 参考：n 点复 FFT（虚部 0）取前 n/2+1 个 bin == numpy.fft.rfft
        const uint32_t m=n>>1;
        const uint32_t halfLen=2u*(m+1u);
        std::vector<std::complex<double>> y(n);
        std::vector<float> cin(2u*n), refFlat(halfLen);
        for(uint32_t bi=0;bi<batch;bi++){
            const float* row=&hIn[(size_t)bi*n];
            for(uint32_t j=0;j<n;j++){ cin[2*j]=row[j]; cin[2*j+1]=0.f; }
            refFft(cin.data(), y.data(), n);
            for(uint32_t j=0;j<=m;j++){
                refFlat[2*j]  =(float)y[j].real();
                refFlat[2*j+1]=(float)y[j].imag();
            }
            const float* got=&hOut[(size_t)bi*hsSt];
            for(uint32_t j=0;j<=m;j++){
                double gr=got[2*j], gi=got[2*j+1];
                double e=std::abs(std::complex<double>(gr,gi)-y[j]);
                if(e>maxAbs){ maxAbs=e; worst=bi*n+j; }
            }
            maxRel=std::max(maxRel, bfly::maxRelScaled(got, refFlat.data(), halfLen));
        }
    }else{
        // 参考：满谱按 prep 的镜像规则展开，G(X)/n = conj(F(conj X))/n（实部）
        const uint32_t m=n>>1;
        std::vector<std::complex<double>> X(n), y(n);
        std::vector<float> cin(2u*n), refFlat(n);
        for(uint32_t bi=0;bi<batch;bi++){
            const float* row=&hIn[(size_t)bi*hsSt];
            for(uint32_t k=0;k<=m;k++) X[k]=std::complex<double>(row[2*k], row[2*k+1]);
            for(uint32_t k=1;k<m;k++) X[n-k]=std::conj(X[k]);
            for(uint32_t j=0;j<n;j++){ cin[2*j]=(float)X[j].real(); cin[2*j+1]=(float)(-X[j].imag()); }
            refFft(cin.data(), y.data(), n);
            for(uint32_t j=0;j<n;j++) refFlat[j]=(float)(std::conj(y[j])/(double)n).real();
            const float* got=&hOut[(size_t)bi*n];
            for(uint32_t j=0;j<n;j++){
                double e=std::fabs((double)got[j]-(double)refFlat[j]);
                if(e>maxAbs){ maxAbs=e; worst=bi*n+j; }
            }
            maxRel=std::max(maxRel, bfly::maxRelScaled(got, refFlat.data(), n));
        }
    }
    printf("n=%u batch=%u blocks=%u  maxAbs=%.3e maxRel=%.3e (worst idx %u)\n",
           n,batch,blocks,maxAbs,maxRel,worst);
    const char* kname = isR2C ? "kfft_r2c" : (isC2R ? "kfft_c2r" : "kfft_fwd");
    printf("%s: %.1f us/call over %d reps (min %.1f us)\n", kname, us, reps, minUs);
    printf("%s\n", maxRel<=1e-4?"PASS":"FAIL");
    return maxRel<=1e-4?0:1;
}
