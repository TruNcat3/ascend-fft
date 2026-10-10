// Butterfly 设计空间对象：H(硬件) G(生成) A(映射) P(分段策略) L(布局) F(融合) Q(质量)
// 对应 cuButterfly 的 plan.hpp 对象集；本移植版用 C++ API（不提供稳定 C ABI）。
#pragma once
#include <cstdint>

#include "butterfly/fft_k.hpp"
#include <string>
#include <vector>

namespace bfly {

// ---------------------------------------------------------------- H 硬件对象
struct Hardware {
    std::string soc;
    std::string cannVersion;
    int aicoreNum = 0;          // AIC 个数
    int vectorCoreNum = 0;      // AIV 个数（Ud 的上界）
    int ubBytesPerCore = 0;     // 196608
    int vecLaneFp32 = 128;      // 矢量 lane
    int l2Bytes = 0;

    // ---- Phase 0.3 实测补充事实（不是推断值）
    bool simt = false;              // --enable-simt 在所有 soc 上被 ccec 拒绝
    int subBlockNum = 1;            // GetSubBlockNum()==1，无子核分工
    int ubOffsetAlignFloats = 8;    // 矢量算子 UB 偏移必须 32B 对齐，否则 507035
    int minBlockOutputBytes = 64;   // 单 block 输出 < 64B 会跨核丢行
    bool matrixUnitUsable = false;  // Mmad 可编译但结果不可读出（见 docs/阶段0-1）

    // Throws std::exception for unreadable, malformed or invalid configuration.
    static Hardware load(const std::string& profilePath);
    // local_exchange 合法性：shuffle/warp 需要 SIMT，本机不可行
    bool supportsShuffleExchange() const { return simt; }
};

// ------------------------------------------------------------- G 生成对象
// 旋转因子表 + 位反转索引：两者都由 host 生成、随 plan 上传（"generation" 语义）
struct Generator {
    enum Kind : uint8_t { Radix2Dit = 0 } kind = Radix2Dit;
    uint32_t n = 0;

    // 每个 stage 一个 align8(h) 的槽；槽内第 j 项 = exp(-i*pi*j/h)
    uint32_t twiddlePad() const { return n + 16; }
    void genTwiddles(std::vector<float>& twr, std::vector<float>& twi) const;
    // 位反转索引（Gather 用）：dst[j] 的源交错下标 = 2*rev(j)
    void genBitReverse(std::vector<uint32_t>& dst) const;
    // K-plane 布局（K = min(32, n/8)）用的 plane 长度与 log2K
    uint32_t planeK() const { return planeKFor(n); }   // 与内核/fft_check 共用同一择优规则
    // 位反转 + 去交错 -> K-plane 布局的 Gather 字节偏移：
    //   dst[k] = bitrev(invpos(k)) * 8，invpos(k) = ((k%rows)<<logK)|(k/rows)
    // im 分量复用同一张表，Gather 时把 srcBaseAddr 设为 4 字节。
    void genBitReversePlane(std::vector<uint32_t>& dst) const;
    // plane -> planar 转置的 Gather 字节偏移：dst[k] = pos(k)*4
    //   pos(j) = (j&(K-1))*rows + (j>>logK)
    void genTransposePlane(std::vector<uint32_t>& dst) const;
    // planar -> 交错输出的 Gather 字节偏移（长度 2n*D）：
    //   批 d 的段写到 dst[d*2n .. d*2n+2n)，D=1 时与旧 2n 表逐字节相同：
    //   dst[d*2n+2j]   = 4*(d*n + j)          （pr_d[j] = plan + d*n + j）
    //   dst[d*2n+2j+1] = 4*(n*D + d*n + j)    （pi_d[j] = plan + n*D + d*n + j）
    void genInterleave(std::vector<uint32_t>& dst, uint32_t D = 1) const;
};

// ---------------------------------------------------------------- A 映射对象
struct Mapping {
    int udCore = 48;      // 每次 launch 的 block 数
    int td = 1;           // batch 分块维度（D=1 表示不做 Nd 分块）
    int tb = 1;           // 单核内 batch 循环粒度
    int ub = 1;           // 每 block 认领的 batch 数（"Ub"）
    int ts = 1;           // stage 是否折叠进 UB（1=全驻留）
    int us = 1;           // 单核内 stage 串行度
    int level = 0;        // 0 = 单卡，1 = 跨卡（当前不可用）
    std::string localExchange = "shared";  // shared | register | shuffle(不可行)

    bool validate(const Hardware& hw, std::string* why) const;
};

// ---------------------------------------------------------------- P 分段策略
struct StagePlan {
    int radix = 2;            // 2 | 4 | 8
    bool coefficientResidency = true;  // 旋转因子全驻留 UB
    int fusionLevel = 1;      // 融合的 stage 数（radix-2 => 1）
    bool bitReverseInput = true;       // 输入位反转（DIT）

    // UB 占用估算（字节）
    size_t ubBytes(uint32_t n) const;
    bool fits(uint32_t n, const Hardware& hw) const { return ubBytes(n) <= (size_t)hw.ubBytesPerCore; }
};

// ---------------------------------------------------------------- L 布局
struct Layout {
    enum Pack : uint8_t { Interleaved = 0, Planar = 1 };
    Pack in = Interleaved;
    Pack out = Interleaved;
    bool contiguous = true;
};

// ---------------------------------------------------------------- F 融合
struct Fusion {
    int butterflyCount = 1;   // 融合的蝴蝶个数
    int stageCount = 1;       // 融合的 stage 个数
    bool dftMatrix = false;   // 是否改用稠密 DFT 矩阵（小 n 走 matmul 路径）
};

// ---------------------------------------------------------------- Q 质量/选型
struct Metric {
    double etaUs = 0.0;       // 解析估算耗时 (µs)
    double measuredUs = 0.0;  // 实测耗时 (µs)
    double maxRelErr = 0.0;   // 实测相对误差
    bool measuredOk = false;
};

// 估算器：用 Phase 0/1 标定的标量/矢量单价，给设计点排序（选型前的过滤器）
Metric estimate(const Hardware& hw, const Mapping& a, const StagePlan& p,
                uint32_t n, uint32_t batch);

}  // namespace bfly
