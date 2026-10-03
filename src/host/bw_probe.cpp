// HBM 有效带宽探针：纯 device→device memcpy，读写各 S 字节。
// 用法: bw_probe [S_MB] [reps]     默认 256 MB x 20
// 报告: gb/s = 2 * S * reps / t   （一次 D2D 计 1 读 + 1 写）
// 用途：给 docs 的"绝对性能对比"提供一个本卡 HBM 峰值参照，
//       避免拿别人的峰值规格当分母。
#include <acl/acl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do{ auto _e=(x); if(_e!=ACL_SUCCESS){ printf("ERR %d at %d\n",(int)_e,__LINE__); return 1;} }while(0)

int main(int argc, char** argv) {
    size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 256;
    int reps = argc > 2 ? atoi(argv[2]) : 20;
    size_t S = mb * 1024ull * 1024ull;
    if (S < 4096) { printf("S too small\n"); return 2; }

    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s = nullptr; CK(aclrtCreateStream(&s));

    void *a = nullptr, *b = nullptr;
    CK(aclrtMalloc(&a, S, ACL_MEM_MALLOC_NORMAL_ONLY));
    CK(aclrtMalloc(&b, S, ACL_MEM_MALLOC_NORMAL_ONLY));
    std::vector<char> h(4096, 0x5a);
    CK(aclrtMemcpy(a, 4096, h.data(), 4096, ACL_MEMCPY_HOST_TO_DEVICE));

    // warmup：把物理页提交掉，避免首次缺页污染计时
    for (int i = 0; i < 3; i++) {
        CK(aclrtMemcpy(b, S, a, S, ACL_MEMCPY_DEVICE_TO_DEVICE));
        CK(aclrtMemcpy(a, S, b, S, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    CK(aclrtSynchronizeStream(s));

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        CK(aclrtMemcpy(b, S, a, S, ACL_MEMCPY_DEVICE_TO_DEVICE));
        CK(aclrtMemcpy(a, S, b, S, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    CK(aclrtSynchronizeStream(s));
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    // 每次循环 = 2 次 D2D，每次 1 读 + 1 写
    double bytes = 2.0 * (double)reps * 2.0 * (double)S;
    double gbs = bytes / sec / 1e9;
    printf("bw_probe: S=%zu MB reps=%d  %.1f GB/s  (%.1f MB in %.3f s)\n",
           mb, reps, gbs, bytes / 1e6, sec);

    aclrtFree(a); aclrtFree(b);
    return 0;
}
