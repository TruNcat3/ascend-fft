// 四态枚举器：Feasible / Infeasible / Unverified / Measured
// 对应 cuButterfly 枚举器的三态 + 本移植版追加的 Measured（实测回填）。
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "butterfly/objects.hpp"

namespace bfly {

enum class State : uint8_t {
    Infeasible = 0,   // 硬件/对齐/UB 直接不可行，不测
    Unverified = 1,   // 语法合法但没跑过，只给 η 排名
    Feasible   = 2,   // 已通过正确性验收，可参与选型
    Measured   = 3,   // 已回填实测 µs
};

const char* toString(State s);

// 一个候选设计点：对象组合 + 估算 + 状态
struct Candidate {
    std::string id;         // e.g. "r2_ud48_ts1_shared"
    Mapping a;
    StagePlan p;
    Layout l;
    Fusion f;
    Metric q;
    State state = State::Unverified;
    std::string reason;     // Infeasible 时给出原因
};

struct DesignSpace {
    std::vector<int> udCores;
    std::vector<int> radices;
    std::vector<std::string> localExchanges;
    std::vector<int> fusionStages;
    std::vector<std::string> layoutsIn;      // L.in 的取值
    std::vector<std::string> layoutsOut;     // L.out 的取值
    std::vector<int> tsValues;               // A.ts: 1=旋转因子/索引驻留 UB
    std::vector<int> coeffResidency;         // P.coefficient_residency: 1=驻留
    std::string raw;                         // 原始 JSON（留档）

    // 从 config/*.json 读取；读不到的轴回落到 applyDefaults() 的缺省值
    static DesignSpace load(const std::string& path);
    void applyDefaults();

    // R = radix^fusionLevel：本实现里"融合后蝶形的点数"
    static int pointSize(int radix, int fusionLevel);
};

// 枚举 + 合法性裁剪：只会产出 Infeasible / Unverified 两种初始状态
std::vector<Candidate> enumerate(const DesignSpace& sp, const Hardware& hw,
                                 uint32_t n, uint32_t batch);

// 只保 Feasible/Measured，按 eta 升序
std::vector<const Candidate*> rank(const std::vector<Candidate>& cs);

}  // namespace bfly
