#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>

using aclError = int;
using aclrtStream = void*;
using aclrtFuncHandle = void*;
using aclrtBinHandle = void*;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 0;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 1;

namespace acl_test {
inline int allocations = 0;
inline int copies = 0;
inline int launches = 0;
inline int failAllocation = 0;
inline int failCopy = 0;
inline int streamsCreated = 0;
inline int streamsDestroyed = 0;
inline int binaryLoads = 0;
}
inline int aclInit(const char*) { return 0; }
inline int aclrtSetDevice(int) { return 0; }
inline int aclrtCreateStream(void** out) { ++acl_test::streamsCreated; *out = reinterpret_cast<void*>(1); return 0; }
inline int aclrtDestroyStream(void*) { ++acl_test::streamsDestroyed; return 0; }
inline int aclrtBinaryLoadFromFile(const char*, void*, void** out) {
    ++acl_test::binaryLoads;
    *out = reinterpret_cast<void*>(1); return 0;
}
inline int aclrtBinaryGetFunction(void*, const char*, void** out) {
    *out = reinterpret_cast<void*>(1); return 0;
}
inline int aclrtBinaryUnLoad(void*) { return 0; }
inline int aclrtMalloc(void** out, size_t bytes, int) {
    ++acl_test::allocations;
    if (acl_test::allocations == acl_test::failAllocation) { *out = nullptr; return 1; }
    *out = std::malloc(bytes);
    return *out ? 0 : 1;
}
inline int aclrtFree(void* pointer) { std::free(pointer); return 0; }
inline int aclrtMemcpy(void* dst, size_t capacity, const void* src, size_t bytes, int) {
    ++acl_test::copies;
    if (acl_test::copies == acl_test::failCopy || !dst || !src || capacity < bytes) return 1;
    std::memcpy(dst, src, bytes); return 0;
}
inline int aclrtLaunchKernelWithHostArgs(void*, unsigned, void*, void*,
                                       void*, size_t, void*, unsigned) {
    ++acl_test::launches; return 0;
}
inline int aclrtSynchronizeStream(void*) { return 0; }
