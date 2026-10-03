// K-plane 择优：在 {8,16,32} 中选 K。目标函数 = 矢量算子调用次数；A5 起在批折叠可行
// 区间（n<=1024）先用 rows=n/K<=64 收窄候选，再取算子最少者（详见 planeKFor 注释）。
// 内核、宿主 fft_check、框架 Generator/estimate 三处必须用同一份规则。
#pragma once
#include <stdint.h>

// ccec 下从 __global__ 函数调用的自由函数必须带 [aicore]，否则报
// "reference to [host] function"；g++ 宿主编译时该宏为空。
#ifdef __CCE_AICORE__
#define BFLY_AICORE [aicore]
#else
#define BFLY_AICORE
#endif

namespace bfly {

// 约束：rows = n/K >= 8（矢量算子 32B 对齐规则，否则 AIV 抛 507035）=> K <= n/8。
// K 必须是 2 的幂，且 <= 32。
//
// 逐条数出的**当前**算子调用数（src/ascendc/fft_radix2.cpp，C1+C2+C2b 之后）：
//   平面级 si（h=1<<si, 共 K/2 对）：
//     j2==0 的对旋转因子恒为 (1,0)，走 2 次 DataCopy 分支 -> 6 算子；
//     其余对复数乘 4 + 蝶形 4 -> 8 算子。
//     j2==0 的对数 = K/(2h)，故该级 = (K/(2h))*6 + (K/2 - K/(2h))*8
//     h=1  => 3K；h>=4 => 接近 4K（旧公式写死 5K，是 C1 之前的值）
//   planar 级 h=1<<s（s 从 logK 到 logn-1）：
//     merge 条件与内核同式；merge 时 8 * ceil(h/64)（每片 4 复数乘 + 4 蝶形），
//     否则 8 * groups；再加每级 1 次 Muls(-ti)。旧公式写死 5n/h（C2 之前的值）。
//   每 batch 收尾：2 DataCopy + 5 Gather = 7。
//
// 目标函数 = 算子调用数（元素访存项与 K 几乎无关，实测同阶）。
// 实测（n=4096,B=4096，min/50reps，A/B 各三轮）：
//   n=512  K16 438 -> K8  297 us (1.48x)    n=1024 K16 558 -> K8  457 us (1.22x)
//   n=2048 K32 1153-> K8  752 us (1.53x)    n=4096 K32 1982-> K16 1677 us (1.18x)
//
// 【A5 批折叠改目标】上面那组是 **D=1** 的结论；折叠之后 plane 段在 rows>64 时必须按
// 64 元素行切片（算子数 x ceil(rows/64)），而 rows<=64 免切片、D 还能拉满。故：
//   * n <= 1024 是折叠可行区间（arRep=n/8<=255）=> 该区间内**只在 rows<=64 的 K 里**按
//     算子数取小（K >= ceil(n/64)）；
//   * n > 1024 恒 D=1 => 维持纯算子择优（n=2048 仍 K8、n=4096 仍 K16，实测 1.53x/1.18x
//     的结论不变）。
// n=1024 实测（B=4096，min/30reps）：K8+D3 352.6us -> K16+D4 308.8us，
// native 333.7us => 0.81x -> **1.07x**。n=512 的 kMin=8，规则退化成原样（K8+D4 154.7us）。
// 规则纯静态（只看 n），内核/宿主/框架三处永远一致，不需要多传 batch/nblk。
BFLY_AICORE inline uint32_t planeKFor(uint32_t n) {
    const uint32_t kNeed = (n + 63u) >> 6;   // 使 rows = n/K <= 64 的最小 K
    const uint32_t kMin  = (n <= 1024u && kNeed > 8u) ? kNeed : 8u;
    uint32_t best = kMin;
    uint64_t bestKey = ~0ull;
    for (uint32_t K = kMin; K <= 32; K <<= 1) {
        if (K * 8u > n) break;
        uint32_t logK = 0, logn = 0;
        while ((1u << logK) < K) logK++;
        while ((1u << logn) < n) logn++;
        uint64_t ops = 0;
        // 平面级：相邻两级 (h,2h) 融合为 radix-4。4h 点一块，块内 j2==0 全平凡(16 算子)，
        // 其余 28 算子(3 复乘 12 + 蝶形 16)。logK 为奇数时末级退回 radix-2。
        uint32_t si = 0;
        for (; si + 1u < logK; si += 2u) {
            const uint32_t h = 1u << si;
            const uint64_t blocks = (uint64_t)K / (4u * h);
            ops += blocks * (16ull + (uint64_t)(h - 1u) * 28ull);
        }
        if (si < logK) {
            const uint32_t h = 1u << si;
            const uint64_t pairs = (uint64_t)K / 2u;
            const uint64_t triv  = (uint64_t)K / (2u * h);   // j2==0 的对
            ops += triv * 6ull + (pairs - triv) * 8ull;
        }
        for (uint32_t s = logK; s < logn; s++) {
            const uint32_t h = 1u << s;
            const uint32_t groups = n / (2u * h);
            const uint32_t nSlice = (h + 63u) >> 6;
            // 与内核同式：stride 字段是 uint8（pSt=2h/8、vSt=h/8），且只有 nSlice<=groups 才折组
            const uint32_t merge = (h >= 8u) && (((2u * h) % 8u) == 0u) &&
                                   (((2u * h) / 8u) <= 255u) && ((h / 8u) <= 255u) &&
                                   (groups <= 255u) && (nSlice <= groups);
            ops += merge ? (uint64_t)8u * nSlice : (uint64_t)8u * groups;
            ops += 1ull;                       // 每级 1 次 Muls(-ti)
        }
        ops += 7ull;                           // 每 batch 的 DataCopy/Gather
        if (ops <= bestKey) { bestKey = ops; best = K; }   // 平局取大 K
    }
    return best;
}

BFLY_AICORE inline uint32_t planeLogKFor(uint32_t n) {
    const uint32_t K = planeKFor(n);
    uint32_t logK = 0;
    while ((1u << logK) < K) logK++;
    return logK;
}

// 批折叠系数 D：把 D 个连续 batch 拍进同一组 Level-0 repeat（A3/A4/A5）。
// 硬约束全部来自 stride_probe 实测（/tmp/op/stride_probe_a22.log）：
//   * Level-0 的 mask 必须 <= 64 —— mask>64 被**静默截断**成 64（探针 m100/m128/m256/m512
//     与 pl128/pl256/pl512 共 7 例只写了前 64 元素）。A5 起 plane 段改按 64 元素**行切片**
//     重放（见内核 nRowSlice），故 rows = n/K 不再是门槛；rows<=64 时切片退化成单片，
//     与不切片逐字节等价（n<=512 全部如此，回归零风险）。
//   * arRep = n/8 是 uint8 批步长 => n <= 2040 => 幂次里 n <= 1024；n >= 2048 时内核
//     useL0=false 走 Level-2 count 路径，折叠无收益 => 直接 return 1。
//   * repeatTime 是 uint8_t => repeat <= 255；rep=255 实测 PASS（pff_rep255）。
//   * src1Rep=0（三元共享源）PASS（pff_*、h64_op1_s1zero、h32_op1_s1zero）。
//   * 64 元素切片 + 任意 32B 对齐基址 PASS（pff_h16/h128/h256/h512、plf_r128_*）。
// 结构约束（与 batch 无关）：n <= 1024（arRep 门槛）且 D * groups_max <= 255，
//                          groups_max = n/(2K)（planar 最严的一级）——超限时**向下钳 D**，
//                          不再整只关掉折叠（n=1024 时 groups_max=64 => D<=3）。
//
// 并发约束（与 batch 有关，同 session A/B 实测）：
//   折叠把 ceil(batch/D) 个组摊到 min(nblk, batch) 个核上；组数一旦少于核数，
//   活跃核就掉下来、串行度反而上升。A/B（D=4 vs D=1，reps=30 取 min，n∈{64..512}）：
//     B=1024  1.42~1.95x   B=256 1.19~1.44x   B=64  0.75~1.05x   B=4/16 0.78~0.93x
//   故要求 ceil(batch/D) >= nblk，即 batch >= D*nblk（nblk=48 => D=4 需 B>=192）。
//   batch / nblk 传 0xFFFFFFFF / 1（默认）表示"不施加并发约束"，用于静态 UB 上界估算。
BFLY_AICORE inline uint32_t foldDFor(uint32_t n,
                                      uint32_t batch = 0xFFFFFFFFu,
                                      uint32_t nblk = 1u) {
    // A3 开关：1 = 关闭折叠（验证与折叠前实现逐字节等价）；过门后改 4。
    const uint32_t kFoldCap = 4;
    uint32_t D = kFoldCap;
    if (D <= 1u) return 1u;
    const uint32_t K = planeKFor(n);
    if (K == 0u || (n % K) != 0u) return 1u;
    const uint32_t rows = n / K;
    if (rows == 0u) return 1u;
    // arRep = n/8 必须 <= 255（否则内核把 stride 钳成 255 = 错误步长）=> n <= 1024。
    // 这条同时保证 nRowSlice 切片的批步长算术正确；n>=2048 时内核 useL0 也是 false。
    if ((n >> 3) > 255u) return 1u;
    uint32_t logK = 0;
    while ((1u << logK) < K) logK++;
    const uint32_t groupsMax = n >> (logK + 1u);   // planar 最大 groups = n/(2K)
    if (groupsMax == 0u) return 1u;
    // D * groupsMax 是 planar 段的 repeatTime，必须 <= 255 —— 超了**递减钳**（不是右移，
    // 右移会把 4 直接砍成 2、白丢一档；n=1024 groupsMax=64 => 4*64=256>255 => 钳到 3）。
    while (D > 1u && D * groupsMax > 255u) D--;
    // 并发：活跃组数 ceil(batch/D) 必须 >= nblk。batch >= 1，用 ((batch-1)/D+1) 做 ceil
    // —— 写成 (batch+D-1)/D 会在 batch=0xFFFFFFFF（"不施加约束"的哨兵值）时溢出回绕。
    if (nblk > 1u) {
        while (D > 1u && ((batch - 1u) / D + 1u) < nblk) D--;
    }
    return D;
}

}  // namespace bfly
