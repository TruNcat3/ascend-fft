#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
#include <dlfcn.h>
#include "acl/acl.h"
#define CK(x) do{ aclError e_=(x); if(e_){ printf("%s -> %d\n", #x,(int)e_); return 1;} }while(0)
int main(){
    const char* o = "build/gather_probe.o";
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s=nullptr; CK(aclrtCreateStream(&s));
    aclrtBinHandle b=nullptr; CK(aclrtBinaryLoadFromFile(o,nullptr,&b));
    aclrtFuncHandle f=nullptr; CK(aclrtBinaryGetFunction(b,"kgather",&f));
    const uint32_t n=64, N=96;
    std::vector<float> src(N), out(N, -1.f);
    for(uint32_t i=0;i<N;i++) src[i]=1000.f+i;
    void *dS=nullptr,*dO=nullptr;
    CK(aclrtMalloc(&dS,N*4,ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&dO,N*4,ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMemcpy(dS,N*4,src.data(),N*4,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemset(dO,N*4,0,N*4));
    unsigned char ab[40]; memset(ab,0,sizeof(ab));
    uint64_t p0=(uint64_t)(uintptr_t)dO, p1=(uint64_t)(uintptr_t)dS;
    memcpy(ab+0,&p0,8); memcpy(ab+8,&p1,8); memcpy(ab+16,&n,4);
    CK(aclrtLaunchKernelWithHostArgs(f,1,s,nullptr,ab,40,nullptr,0));
    CK(aclrtSynchronizeStream(s));
    CK(aclrtMemcpy(out.data(),N*4,dO,N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    printf("idx[i]=3i+1, expected src[idx]=\n  ");
    for(int i=0;i<6;i++) printf("%.0f ", src[3*i+1]);
    printf("\nA(offset 原值) : "); for(int i=0;i<6;i++) printf("%.0f ", out[i]);
    printf("\nB(offset*4)   : "); for(int i=0;i<6;i++) printf("%.0f ", out[32+i]);
    printf("\nC(base=16)    : "); for(int i=0;i<6;i++) printf("%.0f ", out[64+i]);
    printf("\n  -> 若 A==expected 则 offset 单位=元素；若 B==expected 则单位=字节；"
           "C==src[4] 说明 base 单位=字节\n");
    return 0;
}
