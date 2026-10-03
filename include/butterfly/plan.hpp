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
    // 实测 µs（含同步），成功时回填 candidate 的 Metric 并把状态置 Measured
    int measure(uint32_t n, uint32_t batch, Metric* out);

private:
    friend class Context;
    explicit Plan(Candidate c);   // 定义在 butterfly.cpp（Impl 需完整类型）
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
