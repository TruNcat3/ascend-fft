// Generic AscendC .o launcher (Phase 0.4 migration of the validated harness).
// usage: launch <k.o> <kname> <argspec p|u> <check add1|set42|dump|none> [n] [blocks]
//   spec 'p' = 8B device ptr, 'u' = 4B uint32 (filled with n)
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#define CK(x) do{ auto _e=(x); if(_e!=ACL_SUCCESS){ printf("ERR %d at %d\n",(int)_e,__LINE__); return 1;} }while(0)

// Read __CCE_KernelArgSize (uint32) from the .o so argBytes matches the runtime's expectation.
static uint32_t readArgSize(const char* path){
    FILE* f=fopen(path,"rb"); if(!f) return 0;
    fseek(f,0,SEEK_END); long flen=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<unsigned char> buf(flen); if(fread(buf.data(),1,flen,f)!=(size_t)flen){fclose(f);return 0;}
    fclose(f);
    if(flen<64 || *(uint32_t*)buf.data()!=0x464c457f) return 0;          // ELF64 LE
    uint64_t shoff = *(uint64_t*)(buf.data()+40);
    uint16_t shentsize = *(uint16_t*)(buf.data()+58);
    uint16_t shnum = *(uint16_t*)(buf.data()+60);
    uint16_t shstrndx = *(uint16_t*)(buf.data()+62);
    if(!shoff || shstrndx>=shnum) return 0;
    auto sh=[&](int i){ return buf.data()+shoff+(uint64_t)i*shentsize; };
    uint64_t stroff = *(uint64_t*)(sh(shstrndx)+24);
    uint64_t strsz  = *(uint64_t*)(sh(shstrndx)+32);
    for(int i=0;i<shnum;i++){
        uint32_t nm = *(uint32_t*)(sh(i));
        if(stroff+nm+32 > (uint64_t)flen) continue;
        if(strcmp((const char*)(buf.data()+stroff+nm),"__CCE_KernelArgSize")==0){
            uint64_t off = *(uint64_t*)(sh(i)+24);
            uint64_t sz  = *(uint64_t*)(sh(i)+32);
            if(sz>=4 && off+4<=(uint64_t)flen) return *(uint32_t*)(buf.data()+off);
        }
    }
    return 0;
}


int main(int argc, char** argv){
    if(argc<5){
        printf("usage: %s <k.o> <kname> <argspec e.g. ppp|ppu|p> <check add1|set42|dump|none> [n] [blocks]\n", argv[0]);
        return 2;
    }
    const char* kfile=argv[1], *kname=argv[2];
    std::string spec=argv[3], check=argv[4];
    int n = argc>5?atoi(argv[5]):64;
    int blocks = argc>6?atoi(argv[6]):1;

    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s=nullptr; CK(aclrtCreateStream(&s));
    aclrtBinHandle b=nullptr; CK(aclrtBinaryLoadFromFile(kfile,nullptr,&b));
    aclrtFuncHandle f=nullptr; CK(aclrtBinaryGetFunction(b,kname,&f));

    const size_t nptr = (size_t)std::count(spec.begin(),spec.end(),'p');
    std::vector<float*> buf(nptr,nullptr);
    for(size_t i=0;i<nptr;i++) CK(aclrtMalloc((void**)&buf[i], n*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));

    for(size_t i=0;i<nptr;i++){
        std::vector<float> z(n,0.f);
        CK(aclrtMemcpy(buf[i], n*4, z.data(), n*4, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    if(nptr>=1 && check=="add1"){
        std::vector<float> h(n);
        for(int i=0;i<n;i++) h[i]=float(i);
        CK(aclrtMemcpy(buf[0], n*4, h.data(), n*4, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    size_t off=0; std::vector<size_t> offs;
    for(char c: spec){
        size_t a=(c=='p')?8:4; off=(off+a-1)/a*a;
        offs.push_back(off); off += a;
    }
    std::vector<unsigned char> ab(off,0);
    size_t pi=0;
    for(size_t i=0;i<spec.size();i++){
        if(spec[i]=='p'){ unsigned long long v=(unsigned long long)(uintptr_t)buf[pi++]; memcpy(ab.data()+offs[i],&v,8); }
        else { unsigned int v=(unsigned int)n; memcpy(ab.data()+offs[i],&v,4); }
    }

    uint32_t argBytes = readArgSize(kfile);
    size_t packed = ab.size();
    if(argBytes && ab.size()<argBytes) ab.resize(argBytes,0);
    if(argBytes && packed!=argBytes) printf("note: packed=%zu __CCE_KernelArgSize=%u\n", packed, argBytes);

    aclError e=aclrtLaunchKernelWithHostArgs(f,(uint32_t)blocks,s,nullptr,ab.data(),ab.size(),nullptr,0);
    if(e){ printf("launch=%d\n",(int)e); return 1; }
    e=aclrtSynchronizeStream(s);
    if(e){ printf("sync=%d\n",(int)e); return 1; }

    if(check=="none"){ printf("RAN args=%zu\n", ab.size()); return 0; }
    float* target = (check=="add1" && nptr>=2) ? buf[1] : buf[0];
    std::vector<float> h(n);
    CK(aclrtMemcpy(h.data(), n*4, target, n*4, ACL_MEMCPY_DEVICE_TO_HOST));
    if(check=="dump"){
        for(int i=0;i<n;i++) printf("%d %.9g\n", i, h[i]);
        return 0;
    }
    int bad=0;
    for(int i=0;i<n;i++){
        bool ok = (check=="add1") ? (h[i]==float(i)+1.f) : (check=="set42" ? (h[i]==(i==0?42.f:0.f)) : true);
        if(!ok){ if(bad<8) printf("[%d]=%g\n",i,h[i]); bad++; }
    }
    printf("%s (%d/%d bad)\n", bad?"FAIL":"PASS", bad, n);
    return bad?1:0;
}
