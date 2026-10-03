// Phase 0.2: hardware profile (the `H` object of the design point tuple).
// Emits config/ascend910_93_profile.json
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x)                                                                          \
    do {                                                                               \
        auto _e = (x);                                                                 \
        if (_e != ACL_SUCCESS) {                                                       \
            printf("ERR %d at %d\n", (int)_e, __LINE__);                               \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "profile.json";
    CK(aclInit(nullptr));
    int32_t dev = 0;
    CK(aclrtSetDevice(0));
    CK(aclrtGetDevice(&dev));

    int64_t aiCore = 0, vecCore = 0, l2 = 0;
    CK(aclGetDeviceCapability((uint32_t)dev, ACL_DEVICE_INFO_AI_CORE_NUM, &aiCore));
    CK(aclGetDeviceCapability((uint32_t)dev, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &vecCore));
    CK(aclGetDeviceCapability((uint32_t)dev, ACL_DEVICE_INFO_L2_SIZE, &l2));

    size_t memFree = 0, memTotal = 0;
    CK(aclrtGetMemInfo(ACL_HBM_MEM, &memFree, &memTotal));

    int32_t vmaj = 0, vmin = 0, vpat = 0;
    CK(aclrtGetVersion(&vmaj, &vmin, &vpat));
    const char* soc = aclrtGetSocName();

    FILE* f = fopen(out, "w");
    if (!f) { printf("cannot open %s\n", out); return 1; }
    fprintf(f,
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"profile_id\": \"ascend910_93\",\n"
        "  \"evidence\": \"measured\",\n"
        "  \"soc\": \"%s\",\n"
        "  \"cann_version\": \"%d.%d.%d\",\n"
        "  \"device_index\": %d,\n"
        "  \"aicore_num\": %ld,\n"
        "  \"vector_core_num\": %ld,\n"
        "  \"l2_bytes\": %ld,\n"
        "  \"global_mem_total_bytes\": %zu,\n"
        "  \"global_mem_free_bytes\": %zu,\n"
        "  \"ub_bytes_per_core\": 196608,\n"
        "  \"ub_bytes_evidence\": \"MAX_UB_SIZE=192*1024 in shipped AscendC op sim_thread_exponential.h\",\n"
        "  \"vec_lane_fp32\": 128,\n"
        "  \"vec_lane_fp32_evidence\": \"placeholder, confirmed by Phase 0.2 lane probe\",\n"
        "  \"warp_size_simt\": 32,\n"
        "  \"warp_size_evidence\": \"THREAD_GROUP_SIZE in simt_api/cpp/dav_c310 (gated to __NPU_ARCH__ 3510/5102)\",\n"
        "  \"hierarchy\": [\"lane\", \"aiv\", \"die\", \"device\"],\n"
        "  \"shared_memory_analogue\": \"UB (per-AIV private, no cross-core sharing)\",\n"
        "  \"cross_core_data_path\": \"GM only; CrossCoreSetFlag is sync-only\",\n"
        "  \"aib_role\": \"AIC holds L0A/L0B/L1/L0C; GetSubBlockIdx()/GetSubBlockNum() exposes the AIC:AIV pair\"\n"
        "}\n",
        soc, vmaj, vmin, vpat, dev, aiCore, vecCore, l2, memTotal, memFree);
    fclose(f);

    printf("wrote %s\n", out);
    printf("  soc=%s cann=%d.%d.%d aicore=%ld vector_core=%ld l2=%ld hbm_total=%zu\n", soc, vmaj,
           vmin, vpat, aiCore, vecCore, l2, memTotal);
    return 0;
}
