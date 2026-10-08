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
#include "butterfly/descriptors.hpp"
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
// c2c 输入选择（与 AB_INPUT 语义同口径）：sin 默认 / 应用形状 / 数值回归模式。
// 供 main 的 c2c 分支与 AB_INPUT_SEQ 序列换入共用 —— 序列模式保证换了输入但
// 不重建 plan（旋转因子/索引/工作区均输入无关）。
static void genC2CInput(std::vector<float>& h, uint32_t n, uint32_t batch, const char* mode){
    const size_t elements=2ull*(size_t)n*batch;
    if(strcmp(mode,"sin")==0 || strcmp(mode,"deterministic-sin-cos")==0){
        for(size_t i=0;i<elements;i++)
            h[i]=(float)(std::sin(0.011*i)+0.25*std::cos(0.037*i));
    }else if(strcmp(mode,"ofdm")==0 || strcmp(mode,"radar")==0 || strcmp(mode,"dl")==0){
        genAppInput(h, n, batch, mode);
    }else{
        genNumericalInput(h, n, batch, mode);
    }
}

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
    // ---- G0 能力门：c2c 长后端（四步 Cooley-Tukey，宿主分段）----
    // 放行条件（docs/benchmarks/long-fft-plan.md G1）：显式分成 N1*N2 两段，每段
    // 长度落在单 AIV 行 FFT 的 UB envelope [64,4096]；超出 G1 envelope 直接拒绝。
    uint32_t longN1=0, longN2=0;
    const bool isLong = !isReal && n>4096;
    if(isLong){
        if(n>65536){
            printf("c2c long backend envelope is 8192..65536 (G3 larger lengths pending)\n");
            return 2;
        }
        // 从 sqrt 附近找平衡的 2 幂因子：64 <= N1,N2 <= 4096
        uint32_t target=1;
        while((target<<1)*(target<<1)<=n) target<<=1;
        for(int side=0; side<2 && !longN1; side++){
            for(uint32_t a = side? (target>>1) : target; a>=64 && a<=4096;
                a = side? (a>>1) : (a<<1)){
                if(n%a) continue;
                uint32_t b=n/a;
                if(b>=64 && b<=4096){ longN1=a; longN2=b; break; }
            }
        }
        if(!longN1){
            printf("no legal two-factor split n=N1*N2 with 64 <= N1,N2 <= 4096\n");
            return 2;
        }
        // G0 GM 预算：c2c 长链每次占用 3 块复张量（dIn/dA/dOut）+ 每段行 twiddle/idx
        const uint64_t gmBudget=3ull*2ull*(uint64_t)n*(uint64_t)batch*4u;
        if(gmBudget>(40ull<<30)){
            printf("long gm budget %llu B exceeds 40 GiB guard (batch too large)\n",
                   (unsigned long long)gmBudget);
            return 2;
        }
        // Lowering 合约（addendum §2）：在任何分配/启动之前先查 (mapping, unit, hardware)
        // 合法性；不支持的元组返回显式 reason 并拒绝，绝不静默换核或换映射。
        {
            const auto& hw = butterfly::builtin_hardware_profile();
            const auto mapping = butterfly::default_long_mapping(longN1, longN2, hw);
            butterfly::TransformSpec spec{n, batch, butterfly::Precision::FP32,
                                          butterfly::Direction::C2C_FWD};
            const auto lowered = butterfly::query_lowering(spec, mapping,
                                       butterfly::local_fft_capability(), hw);
            if(!lowered.supported){
                printf("lowering rejected: %s\n", lowered.reason.c_str());
                return 2;
            }
            if(getenv("AB_DESC")){
                printf("lowering: groups=%zu launches=%d gm_boundaries=%d resident=%s "
                       "on_chip=%d host_assisted=%d\n",
                       lowered.execution_groups.size(), lowered.visible_launches,
                       lowered.materialized_gm_boundaries,
                       butterfly::residence_name(lowered.resident_subgraph),
                       (int)lowered.whole_transform_on_chip, (int)lowered.host_assisted);
            }
        }
        printf("long backend: n=%u split N1=%u N2=%u rows1=%u rows2=%u gm_bytes=%llu\n",
               n, longN1, longN2, longN2*batch, longN1*batch,
               (unsigned long long)gmBudget);
    }
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
        genC2CInput(hIn, n, batch, inMode);
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

    // twiddles (planar, padded)，按行 FFT 长度生成。c2r：符号取反（e^{+2πi·}）=>
    // kfft_fwd 直接输出 G(X)=n·IDFT(X)（见 fft_real.cpp 头注）。AB_SGN=+1|-1 强制覆盖
    // 旋转因子符号（默认 c2r 取反、其余正向），供调试。sgn 还打包进 packArg 第 16 位
    // （内核 xflip：交叉旋转 ±i 互换，见 fft_radix2.cpp）。
    // 长后端（isLong，G1）分两段：pass1=(N1, rows=N2·batch)、pass2=(N2, rows=N1·batch)；
    // 短路径单段 = (innerN, batch)。prepPass 产出该段的 twiddle/索引/实参打包。
    double sgn = isC2R ? 1.0 : -1.0;
    if(const char* se=getenv("AB_SGN")) sgn = (atof(se)>0)?1.0:-1.0;
    struct PassPrep {
        std::vector<float> twr, twi;
        std::vector<uint32_t> idx;
        uint32_t twPad=0, idxN=0, packArg=0;
    };
    auto prepPass=[&](uint32_t len, uint32_t rows)->PassPrep{
        PassPrep P;
        P.twPad = len + 16u;
        P.twr.assign(P.twPad, 0.f); P.twi.assign(P.twPad, 0.f);
        uint32_t off=0;
        for(uint32_t h=1; h<=len/2; h<<=1){
            for(uint32_t j=0; j<h; j++){
                double ang = sgn*M_PI*(double)j/(double)h;
                P.twr[off+j]=(float)std::cos(ang); P.twi[off+j]=(float)std::sin(ang);
            }
            off += (h+7u)&~7u;
        }
        if(off!=P.twPad) printf("WARN: twiddle slots %u != twPad %u\n", off, P.twPad);
        // 三张索引合成一个 (4n + 2nD) 的 GM 张量：[idxB(n) | idxT(n) | idxOut(2nD)]，单位字节。
        // foldD：默认取 bfly::foldDFor(len, rows, 48)，与 launch 的 blocks=min(48,rows)
        //   同口径，也与 src/framework/butterfly.cpp::Plan::prepare 同一规则（那边 nblk=cand.udCore）。
        //   AB_FOLD_D=<k> 可强制指定，仅供 A/B —— 核内 tmpF/idx 按传入值现算，任意 k>=1
        //   都功能正确；k>1 而 rows=len/K>64 时核内 plane 段按 64 元素行切片（不变错）。
        uint32_t foldD = bfly::foldDFor(len, rows, 48u);
        if (const char* e = getenv("AB_FOLD_D")) {
            int v = atoi(e);
            if (v >= 1) foldD = (uint32_t)v;
        }
        const uint32_t D = foldD ? foldD : 1u;
        // AB_PLANE_K=<8|16|32> 覆盖平面 K（索引生成 + 打包进第 8 参高 8 位）。K 不能 < 8
        // （planar 首级 h=K<8 时 32B 对齐失效，AIV 抛 507035），这正是 planeKFor 候选集从
        // 8 起跳的原因。只对当前 8 参内核有效（遗留 v1/v2 不读第 8 参）。
        uint32_t planeK = bfly::planeKFor(len);
        if (const char* e = getenv("AB_PLANE_K")) {
            int v = atoi(e);
            if (v == 8 || v == 16 || v == 32) planeK = (uint32_t)v;
        }
        P.packArg = foldD | (planeK << 8) | ((sgn > 0.0) ? (1u << 16) : 0u);
        P.idxN = 4u*len + 2u*len*D;
        P.idx.assign(P.idxN, 0u);
        if(len>=64){
            uint32_t K = planeK;
            uint32_t rws=len/K, logK=0; while((1u<<logK)<K) logK++;
            uint32_t logn=0; while((1u<<logn)<len) logn++;
            uint32_t mask=K-1u;
            for(uint32_t k=0;k<len;k++){
                uint32_t j=((k%rws)<<logK)|(k/rws), r=0, t=j;
                for(uint32_t bb=0;bb<logn;bb++){ r=(r<<1)|(t&1u); t>>=1; }
                P.idx[k]   = 2u*r*4u;                                     // bitrev(invpos(k))
                P.idx[len+k] = (((k&mask)*rws)+(k>>logK))*4u;             // plane -> planar
            }
            for(uint32_t d=0;d<D;d++){                                    // planar -> 交错，逐批一段
                uint32_t base = 2u*len + d*2u*len, rd = d*len;
                for(uint32_t j=0;j<len;j++){
                    P.idx[base+2u*j]     = 4u*(rd+j);
                    P.idx[base+2u*j+1u]  = 4u*(D*len + rd+j);
                }
            }
        }
        return P;
    };
    // 段参数：短路径单段；长后端两段（rows = 对向长度 × batch，两段的 rows·len 同为 n·batch）。
    const uint32_t gLen1 = isLong ? longN1 : innerN;
    const uint32_t gRows1 = isLong ? (longN2*batch) : batch;
    const uint32_t gLen2 = isLong ? longN2 : innerN;
    const uint32_t gRows2 = isLong ? (longN1*batch) : batch;
    PassPrep pp1 = prepPass(gLen1, gRows1);
    PassPrep pp2;  // 仅长后端第二段用

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

    // ---- 长后端 plan 状态：只留输入无关内容 ----
    // wT = W_N^{j·k1}（输入无关）；hT/hMid/hD 仅为 workspace，hT 每次执行都从
    // 当前输入重新转置（动态输入契约：plan 不得携带数据相关缓冲，见 PR 评论 P1-A）。
    std::vector<float> hT, hMid, hD, wT;
    if(isLong){
        hT.assign(inElems, 0.f);
        wT.assign(2ull*(size_t)n, 0.f);
        for(uint32_t j=0;j<longN2;j++)
            for(uint32_t k1=0;k1<longN1;k1++){
                double ang=-2.0*M_PI*(double)j*(double)k1/(double)n;
                size_t w=2ull*((size_t)j*longN1 + k1);
                wT[w]=(float)std::cos(ang); wT[w+1]=(float)std::sin(ang);
            }
        hMid.assign(inElems, 0.f);
        hD.assign(inElems, 0.f);
        pp2 = prepPass(gLen2, gRows2);
    }
    void *dIn=nullptr,*dOut=nullptr,*dA=nullptr,*dB=nullptr,*dTwR=nullptr,*dTwI=nullptr,
         *dIdx=nullptr,*dPW=nullptr,*dTw2R=nullptr,*dTw2I=nullptr,*dIdx2=nullptr;
    CK(aclrtMalloc(&dIn,  inElems*4u,  ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dOut, outElems*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    if(isR2C){                                    // Z：内层 FFT 输出（n float/行）
        CK(aclrtMalloc(&dA, inElems*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
        CK(aclrtMalloc(&dPW, (size_t)2u*pPad*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    }else if(isC2R){                              // A：满谱；B：kfft_fwd 反号输出
        CK(aclrtMalloc(&dA, elements*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
        CK(aclrtMalloc(&dB, elements*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    }else if(isLong){                             // pass1 输出（段间 D2H 交宿主 twiddle）
        CK(aclrtMalloc(&dA, inElems*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    }
    CK(aclrtMalloc(&dTwR, pp1.twPad*4u,     ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dTwI, pp1.twPad*4u,     ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dIdx, (size_t)pp1.idxN*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMemcpy(dIn, inElems*4u, hIn.data(), inElems*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dTwR, pp1.twPad*4u, pp1.twr.data(), pp1.twPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dTwI, pp1.twPad*4u, pp1.twi.data(), pp1.twPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dIdx, (size_t)pp1.idxN*4u, pp1.idx.data(),
                   (size_t)pp1.idxN*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    if(isLong){                                   // 第二段 twiddle/索引（仅长后端）
        CK(aclrtMalloc(&dTw2R, pp2.twPad*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
        CK(aclrtMalloc(&dTw2I, pp2.twPad*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
        CK(aclrtMalloc(&dIdx2, (size_t)pp2.idxN*4u, ACL_MEM_MALLOC_NORMAL_ONLY));
        CK(aclrtMemcpy(dTw2R, pp2.twPad*4u, pp2.twr.data(), pp2.twPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dTw2I, pp2.twPad*4u, pp2.twi.data(), pp2.twPad*4u, ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dIdx2, (size_t)pp2.idxN*4u, pp2.idx.data(),
                       (size_t)pp2.idxN*4u, ACL_MEMCPY_HOST_TO_DEVICE));
    }
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
    if(isLong){   // 打印口径：取两段较大 rows 的 block 数（发射时逐段取 min(48,rows)）
        uint32_t rmax = gRows1>gRows2?gRows1:gRows2;
        blocks = rmax<48u?rmax:48u;
    }
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
    memcpy(ab.data()+16,&gLen1,4);
    memcpy(ab.data()+20,&gRows1,4);
    memcpy(ab.data()+24,&p[2],8);
    memcpy(ab.data()+32,&p[3],8);
    memcpy(ab.data()+40,&p[4],8);
    if(hasFold) memcpy(ab.data()+48,&pp1.packArg,4); // 低 8 位 foldD | 高 8 位平面 K

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
    // 长后端逐段发射：实参逐段重写（len/rows/twiddle/idx/packArg），blocks=min(48,rows)。
    auto issuePass=[&](void* dst, void* src, uint32_t len, uint32_t rows,
                       void* twR, void* twI, void* idxGm, uint32_t pack)->aclError{
        uint64_t q0=(uint64_t)(uintptr_t)dst, q1=(uint64_t)(uintptr_t)src;
        uint64_t q2=(uint64_t)(uintptr_t)twR,  q3=(uint64_t)(uintptr_t)twI;
        uint64_t q4=(uint64_t)(uintptr_t)idxGm;
        memcpy(ab.data()+0, &q0,8);
        memcpy(ab.data()+8, &q1,8);
        memcpy(ab.data()+16,&len,4);
        memcpy(ab.data()+20,&rows,4);
        memcpy(ab.data()+24,&q2,8);
        memcpy(ab.data()+32,&q3,8);
        memcpy(ab.data()+40,&q4,8);
        if(hasFold) memcpy(ab.data()+48,&pack,4);
        uint32_t blk = rows<48u?rows:48u;
        return aclrtLaunchKernelWithHostArgs(f,blk,s,nullptr,ab.data(),ab.size(),nullptr,0);
    };
    // device_only 口径：每次 launch 用事件对量取各内核在 device 上的 span 之和
    // （不含宿主段、不含段间拷贝），由 aclrtEventElapsedTime 给出。
    aclrtEvent ev0=nullptr, ev1=nullptr;
    CK(aclrtCreateEvent(&ev0)); CK(aclrtCreateEvent(&ev1));
    double devSpanMs=-1;
    // 一次 launch = 整条方向链（多内核在同一流上串行），末尾一次同步 => 计的是整链 device 时间
    auto launch=[&](){
        aclError e=ACL_SUCCESS;
        devSpanMs=-1;
        if(isLong){
            // 四步 Cooley-Tukey（G1，见 docs/benchmarks/long-fft-plan.md）：
            //   宿主转置入 -> 行FFT(N1) -> 宿主 twiddle+转置 -> 行FFT(N2) -> 宿主重排出。
            // 每次执行都从当前 hIn 重新转置（动态输入契约，PR 评论 P1-A）；传输计数：
            //   逻辑输入 H2D x1、逻辑输出 D2H x1（pass2 出数）、段边界 x2。
            for(uint32_t b=0;b<batch;b++)
                for(uint32_t i=0;i<longN1;i++)
                    for(uint32_t j=0;j<longN2;j++){
                        size_t src=2ull*((size_t)b*n + (size_t)i*longN2 + j);
                        size_t dst=2ull*(((size_t)b*longN2 + j)*longN1 + i);
                        hT[dst]=hIn[src]; hT[dst+1]=hIn[src+1];
                    }
            float dm0=-1.f, dm1=-1.f;
            e=aclrtMemcpyAsync(dIn, inElems*4u, hT.data(), inElems*4u,
                               ACL_MEMCPY_HOST_TO_DEVICE, s);
            if(!e) e=aclrtRecordEvent(ev0, s);
            if(!e) e=issuePass(dA, dIn, longN1, longN2*batch,
                               dTwR, dTwI, dIdx, pp1.packArg);
            if(!e) e=aclrtRecordEvent(ev1, s);
            if(!e) e=aclrtMemcpyAsync(hMid.data(), inElems*4u, dA, inElems*4u,
                                      ACL_MEMCPY_DEVICE_TO_HOST, s);
            if(!e){ aclError se=aclrtSynchronizeStream(s); if(se) e=se; }
            if(!e) aclrtEventElapsedTime(&dm0, ev0, ev1);
            if(!e){ // 宿主段①：twiddle W_N^{j·k1} + 转置成 pass2 行布局
                for(uint32_t b=0;b<batch;b++)
                    for(uint32_t j=0;j<longN2;j++)
                        for(uint32_t k1=0;k1<longN1;k1++){
                            size_t si=2ull*(((size_t)b*longN2+j)*longN1 + k1);
                            size_t w =2ull*((size_t)j*longN1 + k1);
                            size_t di=2ull*(((size_t)b*longN1+k1)*longN2 + j);
                            double ar=hMid[si], ai=hMid[si+1];
                            double wr=wT[w], wi=wT[w+1];
                            hD[di]  =(float)(ar*wr-ai*wi);
                            hD[di+1]=(float)(ar*wi+ai*wr);
                        }
            }
            if(!e) e=aclrtMemcpyAsync(dIn, inElems*4u, hD.data(), inElems*4u,
                                      ACL_MEMCPY_HOST_TO_DEVICE, s);
            if(!e) e=aclrtRecordEvent(ev0, s);
            if(!e) e=issuePass(dOut, dIn, longN2, longN1*batch,
                               dTw2R, dTw2I, dIdx2, pp2.packArg);
            if(!e) e=aclrtRecordEvent(ev1, s);
            if(!e) e=aclrtMemcpyAsync(hMid.data(), inElems*4u, dOut, outElems*4u,
                                      ACL_MEMCPY_DEVICE_TO_HOST, s);
            if(!e){ aclError se=aclrtSynchronizeStream(s); if(se) e=se; }
            if(!e) aclrtEventElapsedTime(&dm1, ev0, ev1);
            if(!e){ // 宿主段②：E[k1][k2] -> X[k1+N1·k2] 自然序
                for(uint32_t b=0;b<batch;b++)
                    for(uint32_t k1=0;k1<longN1;k1++)
                        for(uint32_t k2=0;k2<longN2;k2++){
                            size_t si=2ull*(((size_t)b*longN1+k1)*longN2 + k2);
                            size_t k = (size_t)k1 + (size_t)longN1*k2;
                            size_t di=2ull*((size_t)b*n + k);
                            hOut[di]=hMid[si]; hOut[di+1]=hMid[si+1];
                        }
            }
            // 结果就地留在 hOut：pass2 的 D2H 就是唯一逻辑输出传输，不再回写 dOut
            if(e){ printf("long launch=%d\n",(int)e); return false; }
            devSpanMs = (dm0>=0.f && dm1>=0.f) ? (double)(dm0+dm1) : -1.0;  // ms
            return true;
        }
        float dm=-1.f;
        if(!e) e=aclrtRecordEvent(ev0, s);
        if(isR2C)      { e=issueFwd(dA,dIn);
                         if(!e) e=issueReal(fPost,dOut,dA, dPW,(char*)dPW+(size_t)pPad*4); }
        else if(isC2R) { e=issueReal(fPrep,dA,dIn);     if(!e) e=issueFwd(dB,dA);
                         if(!e) e=issueReal(fPost,dOut,dB); }
        else           { e=issueFwd(dOut,dIn); }
        if(!e) e=aclrtRecordEvent(ev1, s);
        if(e){ printf("launch=%d\n",(int)e); return false; }
        { aclError se=aclrtSynchronizeStream(s); if(se){ printf("sync=%d\n",(int)se); return false; } }
        aclrtEventElapsedTime(&dm, ev0, ev1);
        devSpanMs = dm>=0.f ? (double)dm : -1.0;   // ms
        return true;
    };

    // setup = 主机端准备（旋转因子/索引生成）+ 显存分配 + 首次上传 + 二进制加载，
    // 在冷启动 warmup 之前截断，供端到端报告一次性开销。
    const auto tSetup1=std::chrono::steady_clock::now();

    const auto tu0=std::chrono::steady_clock::now();
    if(!launch()) return 1;
    const double firstUseUs=std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-tu0).count();
    const double planUs=std::chrono::duration<double,std::micro>(tSetup1-tPlan0).count();

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
        if(wantPinned && !isLong){
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
        const char*  hostTag = isLong ? "chain" : (pIn ? "pinned" : "pageable");
        double sumE=0, minE=0, firstE=0;
        const auto t0e=std::chrono::steady_clock::now();
        for(int i=0;i<ereps;i++){
            auto a0=std::chrono::steady_clock::now();
            if(isLong && !xferOnly){
                // 长链自带唯一一次逻辑输入 H2D（当前输入转置后上传）与唯一一次
                // 逻辑输出 D2H（pass2 出数到宿主）；外层不再包裹任何传输。
                if(!launch()) return 1;
            }else if(useAsync){
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
        double spanUs=std::chrono::duration<double,std::micro>(t0e-tSetup1).count();
        printf("E2E n=%u batch=%u reps=%d e2e_us=%.1f e2e_min_us=%.1f first_us=%.1f "
               "boot_us=%.1f plan_us=%.1f warmup_us=%.1f mode=%s host=%s input=%s\n",
               n,batch,ereps,e,minE,firstE,bootUs,planUs,spanUs,
               useAsync?"async":"sync", hostTag, inMode);
        // 传输计数断言口径：每次执行恰 1 次逻辑输入 + 1 次逻辑输出；
        // 段边界传输（长链 pass1 出数 / pass2 入数）单列，不算逻辑传输。
        printf("E2E transfers: in=1 out=1 boundary=%d per_execution\n",
               (isLong && !xferOnly) ? 2 : 0);
        fflush(stdout);
        if(pIn){ aclrtFreeHost(pIn); aclrtFreeHost(pOut); }
    }

    // 逐次 launch 计时：均值与最小值都报。均值与 §8.5 的 η 标定同口径，
    // 最小值与 bench_native_npu.py 的统计量同口径 —— 两边可以各自对齐比。
    // device_only 取各次 launch 内核 span（事件对测量）的最小值。
    double sumUs=0, minUs=0, devMinUs=-1;
    std::vector<double> samples; samples.reserve(reps);
    for(int i=0;i<reps;i++){
        auto a0=std::chrono::steady_clock::now();
        if(!launch()) return 1;
        auto a1=std::chrono::steady_clock::now();
        double u=std::chrono::duration<double,std::micro>(a1-a0).count();
        sumUs+=u; if(i==0||u<minUs) minUs=u; samples.push_back(u);
        if(devSpanMs>=0 && (devMinUs<0 || devSpanMs*1000.0<devMinUs))
            devMinUs=devSpanMs*1000.0;
    }
    double us=sumUs/reps;
    double medUs=0;
    if(!samples.empty()){
        std::vector<double> sorted=samples;
        size_t m=sorted.size()/2;
        std::nth_element(sorted.begin(), sorted.begin()+m, sorted.end());
        medUs=sorted[m];
        if(sorted.size()%2==0){
            double lo=*std::max_element(sorted.begin(), sorted.begin()+m);
            medUs=(lo+medUs)/2.0;
        }
    }

    // 四个计时口径显式成行（PR 评论 P1-A）：plan 一次性准备、首次执行、
    // 同步 host 端到端（含宿主段与段间拷贝）、纯 device 内核 span。
    printf("scopes: plan_setup=%.1f us first_use=%.1f us host_end_to_end mean=%.1f "
           "min=%.1f us device_only=%.1f us reps=%d\n",
           planUs, firstUseUs, us, minUs, devMinUs, reps);

    if(!isLong)   // 长链的逻辑输出 D2H 已在 launch 内完成，结果就在 hOut
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
    // 只作诊断、不参与 PASS/FAIL。c2c 分支抽成闭包：主验证与 AB_INPUT_SEQ 换入
    // 后的逐次复核共用同一口径（P1-A：每次换入都对独立 FP64 参考复核）。
    auto verifyC2C=[&](double& maxAbs, double& maxRel, uint32_t& worst)->void{
        maxAbs=0; maxRel=0; worst=0;
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
    };
    double maxAbs=0, maxRel=0;
    uint32_t worst=0;
    if(!isReal) verifyC2C(maxAbs, maxRel, worst);
    else if(isR2C){
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
    // ---- AB_INPUT_SEQ=<tok>,<tok>[,...]：同一 plan 换入再执行（P1-A 动态输入契约）----
    // token = AB_INPUT 模式名（重新生成）或原始 float32 文件路径（大小须等于 inElems*4）。
    // 每项：换入 ->（短路径重传 dIn；长链每次执行自行转置当前输入）-> launch -> 取回
    // 输出 -> FP64 逐点复核；相邻不同输入的输出必须彼此不同（陈旧缓冲检测）。
    bool seqOk=true;
    if(const char* sq=getenv("AB_INPUT_SEQ")){
        if(isReal){
            printf("AB_INPUT_SEQ only supported for c2c (set AB_DIR=c2c)\n");
            return 2;
        }
        std::vector<std::string> toks;
        for(const char* p=sq; *p; ){
            const char* c=strchr(p, ',');
            size_t len = c ? (size_t)(c-p) : strlen(p);
            if(len) toks.emplace_back(p, len);
            if(!c) break;
            p=c+1;
        }
        if(toks.size()<2){
            printf("AB_INPUT_SEQ needs at least two inputs (A,B[,A])\n");
            return 2;
        }
        std::vector<float> prevIn, prevOut;
        for(size_t t=0;t<toks.size();t++){
            const std::string& tok=toks[t];
            if(isInputMode(tok.c_str())){
                genC2CInput(hIn, n, batch, tok.c_str());
            }else{
                FILE* fp=fopen(tok.c_str(),"rb");
                if(!fp || fread(hIn.data(),4,inElems,fp)!=inElems){
                    printf("AB_INPUT_SEQ[%zu] read failed: %s\n", t, tok.c_str());
                    return 2;
                }
                fclose(fp);
            }
            if(!isLong)
                CK(aclrtMemcpy(dIn, inElems*4u, hIn.data(), inElems*4u,
                               ACL_MEMCPY_HOST_TO_DEVICE));
            if(!launch()) return 1;
            if(!isLong)
                CK(aclrtMemcpy(hOut.data(), outElems*4u, dOut, outElems*4u,
                               ACL_MEMCPY_DEVICE_TO_HOST));
            double sa=0, sr=0; uint32_t sw=0;
            verifyC2C(sa, sr, sw);
            const bool ok = sr<=1e-4;
            bool stale=false;
            if(!prevIn.empty()){
                const bool inDiff = !std::equal(hIn.begin(), hIn.end(), prevIn.begin());
                const bool outDiff = !std::equal(hOut.begin(), hOut.end(), prevOut.begin());
                stale = inDiff && !outDiff;   // 输入变了而输出按位未变 => 复用了旧结果
            }
            printf("seq[%zu]=%.80s maxRel=%.3e %s%s\n", t, tok.c_str(), sr,
                   ok?"PASS":"FAIL", stale?" STALE-OUTPUT":"");
            if(!ok || stale) seqOk=false;
            prevIn=hIn; prevOut=hOut;
        }
        printf("seq: %zu inputs re-executed under one plan (no rebuild), stale-check on -> %s\n",
               toks.size(), seqOk?"PASS":"FAIL");
    }
    printf("n=%u batch=%u blocks=%u  maxAbs=%.3e maxRel=%.3e (worst idx %u)\n",
           n,batch,blocks,maxAbs,maxRel,worst);
    const char* kname = isR2C ? "kfft_r2c" : (isC2R ? "kfft_c2r" : "kfft_fwd");
    printf("%s: %.1f us/call over %d reps (min %.1f us) med %.1f us\n",
           kname, us, reps, minUs, medUs);
    printf("%s\n", (maxRel<=1e-4 && seqOk)?"PASS":"FAIL");
    return (maxRel<=1e-4 && seqOk)?0:1;
}
