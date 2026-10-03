// Cube(fp32 Mmad) 探针 host。
// 用法: cube_probe <mode> [srcStride] [dstStride] [loadRep] [layout] [nrep] [nTile] [reps]
//   mode   0=LoadData  1=+Mmad  2=+Fixpipe
//   layout A/B 的 NZ 布局候选 0..15（0 = 已验证正确的 a0b0）
//   nrep   每个 layout 重复次数（确定性检查）
//   nTile  单次 kernel 内的矩阵个数
//   reps   计时的 kernel 启动次数
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
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

static const int M=16, K=16, N=16;

static void packA(std::vector<float>& v, const std::vector<float>& A, int mode){
    std::fill(v.begin(), v.end(), 0.f);
    if(mode==0){ for(int kb=0;kb<K/8;kb++) for(int r=0;r<M;r++) for(int j=0;j<8;j++)
            v[kb*128 + r*8 + j]=A[r*K + kb*8+j]; }
    else if(mode==1){ for(int kb=0;kb<K/8;kb++) for(int r=0;r<M;r++) for(int j=0;j<8;j++)
            v[kb*128 + j*16 + r]=A[r*K + kb*8+j]; }
    else if(mode==2){ for(int r=0;r<M;r++) for(int c=0;c<K;c++) v[r*K+c]=A[r*K+c]; }
    else { for(int r=0;r<M;r++) for(int c=0;c<K;c++) v[c*M+r]=A[r*K+c]; }
}
static void packB(std::vector<float>& v, const std::vector<float>& B, int mode){
    std::fill(v.begin(), v.end(), 0.f);
    if(mode==0){ for(int kb=0;kb<K/8;kb++) for(int n=0;n<N;n++) for(int j=0;j<8;j++)
            v[kb*128 + n*8 + j]=B[(kb*8+j)*N + n]; }
    else if(mode==1){ for(int kb=0;kb<K/8;kb++) for(int n=0;n<N;n++) for(int j=0;j<8;j++)
            v[kb*128 + j*16 + n]=B[(kb*8+j)*N + n]; }
    else if(mode==2){ for(int k=0;k<K;k++) for(int n=0;n<N;n++) v[k*N+n]=B[k*N+n]; }
    else { for(int k=0;k<K;k++) for(int n=0;n<N;n++) v[n*K+k]=B[k*N+n]; }
}

int main(int argc, char** argv){
    const char* o = argc>1 ? argv[1] : "build/cube_probe.o";
    uint32_t mode     = argc>2 ? (uint32_t)atoi(argv[2]) : 2;
    uint32_t srcStride= argc>3 ? (uint32_t)atoi(argv[3]) : 1;
    uint32_t dstStride= argc>4 ? (uint32_t)atoi(argv[4]) : 16;
    uint32_t loadRep  = argc>5 ? (uint32_t)atoi(argv[5]) : 2;
    int      lay      = argc>6 ? atoi(argv[6]) : 0;
    int      nrep     = argc>7 ? atoi(argv[7]) : 3;
    uint32_t nTile    = argc>8 ? (uint32_t)atoi(argv[8]) : 1;
    int      reps     = argc>9 ? atoi(argv[9]) : 20;

    std::vector<float> A(M*K), B(K*N), ref(M*N, 0.f);
    for(int i=0;i<M*K;i++) A[i]=float((i*7)%13)-6.f;
    for(int i=0;i<K*N;i++) B[i]=float((i*5)%11)-5.f;
    for(int i=0;i<M;i++) for(int j=0;j<N;j++){ double s=0; for(int k=0;k<K;k++) s+=double(A[i*K+k])*double(B[k*N+j]); ref[i*N+j]=(float)s; }

    const size_t elems = (size_t)nTile*256;
    std::vector<float> pa(elems), pb(elems), C(elems, 0.f);
    std::vector<float> tileA(M*K), tileB(K*N);
    for(uint32_t t=0;t<nTile;t++){
        for(int i=0;i<M*K;i++) tileA[i]=float(((t*37+i)*7)%13)-6.f;
        for(int i=0;i<K*N;i++) tileB[i]=float(((t*53+i)*5)%11)-5.f;
        std::vector<float> sa(256), sb(256);
        packA(sa, tileA, lay/4); packB(sb, tileB, lay%4);
        memcpy(&pa[t*256], sa.data(), 256*4);
        memcpy(&pb[t*256], sb.data(), 256*4);
    }

    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s=nullptr; CK(aclrtCreateStream(&s));
    void *dA=nullptr,*dB=nullptr,*dC=nullptr;
    CK(aclrtMalloc(&dA, elems*4, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dB, elems*4, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dC, elems*4, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMemcpy(dA, elems*4, pa.data(), elems*4, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dB, elems*4, pb.data(), elems*4, ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<float> zero(elems,0.f);
    CK(aclrtMemcpy(dC, elems*4, zero.data(), elems*4, ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtBinHandle b=nullptr; CK(aclrtBinaryLoadFromFile(o,nullptr,&b));
    aclrtFuncHandle f=nullptr; CK(aclrtBinaryGetFunction(b,"kcube",&f));
    uint32_t argBytes=readArgSize(o);

    unsigned char ab[72]; memset(ab,0,sizeof(ab));
    uint64_t pC=(uint64_t)(uintptr_t)dC, pA=(uint64_t)(uintptr_t)dA, pB=(uint64_t)(uintptr_t)dB;
    memcpy(ab+0,&pC,8); memcpy(ab+8,&pA,8); memcpy(ab+16,&pB,8);
    uint32_t v[5]={mode, loadRep, srcStride, dstStride, nTile};
    memcpy(ab+24,v,20);
    size_t used = argBytes ? argBytes : 44;
    if(used>sizeof(ab)) used=sizeof(ab);

    int fail=0;
    for(int rep=0; rep<nrep; rep++){
        if(mode!=2 && mode!=3){ break; }
        CK(aclrtMemcpy(dC, elems*4, zero.data(), elems*4, ACL_MEMCPY_HOST_TO_DEVICE));
        aclError e=aclrtLaunchKernelWithHostArgs(f,1,s,nullptr,ab,used,nullptr,0);
        if(e){ printf("launch=%d\n",(int)e); return 1; }
        aclError se=aclrtSynchronizeStream(s);
        if(se){ printf("SYNC=%d\n",(int)se); return 1; }
        CK(aclrtMemcpy(C.data(), elems*4, dC, elems*4, ACL_MEMCPY_DEVICE_TO_HOST));
        double maxRel=0; int bad=0;
        for(int i=0;i<M*N;i++){
            double den=std::abs(double(ref[i]))>1e-6?std::abs(double(ref[i])):1.0;
            double r=std::abs(double(C[i])-double(ref[i]))/den;
            if(r>maxRel)maxRel=r; if(r>1e-3)bad++;
        }
        if(mode!=2){ continue; }
        if(bad){ fail=1; printf("rep%d lay=%d -> FAIL maxRel=%.3e bad=%d [c0=%g ref %g]\n",rep,lay,maxRel,bad,C[0],ref[0]); }
        else if(nrep<=3) printf("rep%d lay=%d -> PASS\n",rep,lay);
    }
    if(fail) return 2;

    // 计时
    for(int i=0;i<5;i++){ aclrtLaunchKernelWithHostArgs(f,1,s,nullptr,ab,used,nullptr,0); aclrtSynchronizeStream(s); }
    const int R=reps;
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<R;i++) aclrtLaunchKernelWithHostArgs(f,1,s,nullptr,ab,used,nullptr,0);
    CK(aclrtSynchronizeStream(s));
    auto t1=std::chrono::steady_clock::now();
    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/R;
    double macs=(double)nTile*16*16*16;
    printf("OK lay=%d mode=%u nTile=%u args=%zu -> %.2f us/call, %.1f GMAC/s, data %.2f MB/call, %.0f GB/s\n",
           lay, mode, nTile, used, us, macs/us/1000.0,
           elems*4*3/1e6, elems*4*3/(us*1e-6)/1e9);
    return 0;
}
