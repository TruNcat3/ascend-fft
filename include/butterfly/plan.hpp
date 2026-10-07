// Context / Plan / Transform —— 对 cuButterfly plan.hpp 的 API 镜像。
// 不提供稳定 C ABI；计划在 aclrtStream 上物化为一次 kernel launch。
#pragma once
#include <acl/acl_rt.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "butterfly/enumerate.hpp"
#include "butterfly/objects.hpp"

namespace bfly {

class Context;
class Plan;

// Transform：plan 的一次"物化动作"（生成旋转因子 / 上传索引 / launch kernel）
class Transform {
public:
    enum Kind : uint8_t { GenTwiddles, GenBitReverse, LaunchKernel, Measure } kind;
    std::string label;
};

// Plan：一次可执行的 FFT 计划
class Plan {
public:
    ~Plan();
    const Candidate& candidate() const { return cand_; }
    const std::vector<Transform>& transforms() const { return tf_; }

    // 把旋转因子/位反转索引写到 device（幂等，(n, 折叠系数 D) 不变则不重传）。
    // batch 决定批折叠系数 D（见 bfly::foldDFor），故必须一起传；省略时按 D 上界预生成。
    int prepare(uint32_t n, uint32_t batch = 0xFFFFFFFFu);
    // 执行一次 n×batch 的复数 fp32 前向 FFT
    int run(const float* in, float* out, uint32_t n, uint32_t batch);
    // r2c：in [batch][n] 实数；out [batch][n+2] = n/2+1 个交错复数（numpy.fft.rfft 稠密布局）。
    // 需要与 kernelPath 同目录的 fft_real.o（缺失返回 -1）；n 偶数且 128..8192。
    int runR2C(const float* in, float* out, uint32_t n, uint32_t batch);
    // c2r：in [batch][n+2] 半谱（Nyquist 虚部须为 0，与 fft_check::genHalfInput 同约定）；
    // out [batch][n] 实数（含 1/n，口径同 numpy.fft.irfft）。n 偶数且 64..4096。
    int runC2R(const float* in, float* out, uint32_t n, uint32_t batch);
    // 实测 µs（含同步），成功时回填 candidate 的 Metric 并把状态置 Measured
    int measure(uint32_t n, uint32_t batch, Metric* out);

private:
    friend class Context;
    explicit Plan(Candidate c);   // 定义在 butterfly.cpp（Impl 需完整类型）
    // prepare 的实现体：sign>0 时旋转因子取反（c2r 的 +i 约定，与内核 xflip 位配套）
    int prepareSign(uint32_t n, uint32_t batch, int sign);
    Candidate cand_;
    std::vector<Transform> tf_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    aclrtStream ctxStream_ = nullptr;   // 由 Context 注入
};

// Context：持有设备句柄、硬件描述、kernel 注册表、设计空间
class Context {
public:
    Context();
    ~Context();

    int init(const std::string& profilePath, const std::string& spacePath,
             const std::string& kernelPath);
    const Hardware& hardware() const { return hw_; }
    const DesignSpace& space() const { return sp_; }
    const std::string& kernelPath() const { return kernelPath_; }

    // 枚举全部候选（含 Infeasible 及原因）
    std::vector<Candidate> enumerate(uint32_t n, uint32_t batch) const;

    // 选型循环：η 预排序 → 前 K 个构 plan → 实测 → 取最优
    std::unique_ptr<Plan> select(uint32_t n, uint32_t batch, int topK = 3,
                                 std::string* log = nullptr);

    // 直接用给定候选建 plan
    std::unique_ptr<Plan> makePlan(const Candidate& c);

private:
    Hardware hw_;
    DesignSpace sp_;
    std::string kernelPath_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    aclrtStream ctxStream_ = nullptr;   // 由 Context 注入
};

}  // namespace bfly
