// Level-0 矢量算子语义探针。用法: stride_probe
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
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

struct Case { const char* name; uint32_t mask, repeat, dstRep, src0Rep, src1Rep; };

int main(int argc, char** argv){
    const char* o = argc>1 ? argv[1] : "build/stride_probe.o";
    const uint32_t N = 16384;
    // 控制组 = AscendC Level-2 常用形态；其余 = fft_radix2 合并蝶形里实际用到的三类参数。
    std::vector<Case> cases = {
        {"ctrl_rep2_d8_s8",   64,  2,  8,  8,  8},
        {"h64_op1_t8_p16",    64, 32,  8, 16,  8},
        {"h64_op3_t8_t8",     64, 32,  8,  8,  8},
        {"h64_op7_p16_t8",    64, 32, 16, 16,  8},
        {"h32_op1_t4_p8",     32, 64,  4,  8,  4},
        {"h32_op7_p8_t4",     32, 64,  8,  8,  4},
        {"u16_t4_p8",         64,  4,  4,  8,  4},
        {"h64_op1_s1zero",    64, 32,  8, 16,  0},
        {"h32_op1_s1zero",    32, 64,  4,  8,  0},
        // ---- 单 repeat 的 fp32 元素数上限（planar 合并路径要 msk=h，当前按 64 切片）----
        {"m64_r1",            64,  1,  0,  0,  0},   // 控制组
        {"m100_r1",          100,  1,  0,  0,  0},   // count>64 且非 2 的幂
        {"m128_r1",          128,  1,  0,  0,  0},
        {"m256_r1",          256,  1,  0,  0,  0},
        {"m512_r1",          512,  1,  0,  0,  0},
        // 与 planar 合并路径逐字同参（h=128/256/512 的 vSt、pSt、src1Rep=0）
        {"pl128_g16",        128, 16, 16, 32,  0},
        {"pl256_g8",         256,  8, 32, 64,  0},
        {"pl512_g4",         512,  4, 64, 128, 0},
        // ==== 批折叠（Part A A2-2）====
        // plane 段：ar/ai/t* 统一走 [butterfly][batch][row] 布局，
        //   rows<=64 时 (batch) 直接折进 repeat，rows>64 时把 (batch, slice)
        //   拍平成 r = d*S+k、Δ=64 元素 => 三路 stride 同为 8 个 32B 块。
        {"plf_r8_d4",          8,  4,  1,  1,  1},   // n=64   rows=8
        {"plf_r16_d4",        16,  4,  2,  2,  2},   // n=128  rows=16
        {"plf_r32_d4",        32,  4,  4,  4,  4},   // n=256  rows=32
        {"plf_r64_d4",        64,  4,  8,  8,  8},   // n=512  rows=64
        {"plf_r128_d4",       64,  8,  8,  8,  8},   // n=1024 rows=128, D=4 -> rep=8
        {"plf_r128_d8",       64, 16,  8,  8,  8},   // n=1024 rows=128, D=8 -> rep=16
        {"plf_r8_d8",          8,  8,  1,  1,  1},   // n=64   D=8
        // planar 段 option F：r = d*groups+g、Δ=2h，
        //   dst(t*)=h/8、src0(pr)=2h/8、src1(tw)=0（twiddle 跨 group、跨 batch 共享）
        {"pff_h16_r128",      16,128,  2,   4,  0},  // n=1024 D=4 groups=32
        {"pff_h128_r16",      64, 16, 16,  32,  0},
        {"pff_h256_r8",       64,  8, 32,  64,  0},
        {"pff_h512_r4",       64,  4, 64, 128,  0},
        {"pff_h32_r128",      32,128,  4,   8,  0},  // n=1024 D=8 groups=16
        {"pff_rep255",        16,255,  2,   4,  0},  // repeat 的 uint8_t 上界
    };

    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s=nullptr; CK(aclrtCreateStream(&s));
    void *dIn=nullptr,*dOut=nullptr;
    CK(aclrtMalloc(&dIn,  N*4, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dOut, N*4, ACL_MEM_MALLOC_NORMAL_ONLY));
    std::vector<float> hin(N);
    for(uint32_t i=0;i<N;i++) hin[i]=float(i+1);
    CK(aclrtMemcpy(dIn, N*4, hin.data(), N*4, ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtBinHandle b=nullptr; CK(aclrtBinaryLoadFromFile(o,nullptr,&b));
    aclrtFuncHandle f=nullptr; CK(aclrtBinaryGetFunction(b,"kstride",&f));
    uint32_t argBytes=readArgSize(o);

    for(auto& c : cases){
        std::vector<float> zero(N,0.f);
        CK(aclrtMemcpy(dOut, N*4, zero.data(), N*4, ACL_MEMCPY_HOST_TO_DEVICE));
        unsigned char ab[64]; memset(ab,0,sizeof(ab));
        uint64_t pOut=(uint64_t)(uintptr_t)dOut, pIn=(uint64_t)(uintptr_t)dIn;
        memcpy(ab+0,&pOut,8); memcpy(ab+8,&pIn,8);
        uint32_t v[6]={N,c.mask,c.repeat,c.dstRep,c.src0Rep,c.src1Rep};
        memcpy(ab+16,v,24);
        size_t used = argBytes ? argBytes : 40;
        if(used>sizeof(ab)) used=sizeof(ab);
        aclError e=aclrtLaunchKernelWithHostArgs(f,1,s,nullptr,ab,used,nullptr,0);
        if(e){ printf("%-18s launch=%d\n",c.name,(int)e); continue; }
        CK(aclrtSynchronizeStream(s));
        std::vector<float> h(N);
        CK(aclrtMemcpy(h.data(), N*4, dOut, N*4, ACL_MEMCPY_DEVICE_TO_HOST));

        // 期望：未写位置仍 = i+1；写位置 dst[dstRep*8*r+j] = (a+1)*(b+1)
        std::vector<uint8_t> want(N,0);
        std::vector<float> exp(N,0.f);
        for(uint32_t r=0;r<c.repeat;r++){
            for(uint32_t j=0;j<c.mask;j++){
                uint32_t a=c.src0Rep*8*r+j, b=c.src1Rep*8*r+j, d=c.dstRep*8*r+j;
                if(a>=N||b>=N||d>=N) continue;
                want[d]=1; exp[d]=float((a+1)*(b+1));
            }
        }
        int badW=0, badU=0; uint32_t firstW=0, firstU=0;
        for(uint32_t i=0;i<N;i++){
            if(want[i]){ if(h[i]!=exp[i]){ if(!badW) firstW=i; badW++; } }
            else       { if(h[i]!=float(i+1)){ if(!badU) firstU=i; badU++; } }
        }
        printf("%-18s mask=%u rep=%u dRep=%u s0Rep=%u s1Rep=%u -> %s (badWritten=%d@%u badUntouched=%d@%u)\n",
               c.name,c.mask,c.repeat,c.dstRep,c.src0Rep,c.src1Rep,
               (badW||badU)?"FAIL":"PASS ", badW,firstW,badU,firstU);
    }
    return 0;
}
