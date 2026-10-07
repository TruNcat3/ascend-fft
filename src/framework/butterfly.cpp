#include "butterfly/plan.hpp"
#include "butterfly/reference.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>

namespace bfly {

// ---------------------------------------------------------------- 极简 JSON 取值
// 只为读取我们自己产出的扁平配置；键重复时取最后一个。
namespace {
bool findValue(const std::string& text, const std::string& key, std::string* val) {
    std::string pat = "\"" + key + "\"";
    size_t pos = text.rfind(pat);
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos + pat.size());
    if (pos == std::string::npos) return false;
    pos = text.find_first_not_of(" \t\r\n", pos + 1);
    if (pos == std::string::npos) return false;
    if (text[pos] == '"') {
        size_t e = text.find('"', pos + 1);
        if (e == std::string::npos) return false;
        *val = text.substr(pos + 1, e - pos - 1);
        return true;
    }
    size_t e = pos;
    while (e < text.size() && (isdigit((unsigned char)text[e]) || text[e] == '-' || text[e] == '+' ||
                               text[e] == '.' || text[e] == 'e' || text[e] == 'E'))
        e++;
    *val = text.substr(pos, e - pos);
    return !val->empty();
}
std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
int asInt(const std::string& text, const std::string& key, int dflt) {
    std::string v;
    return findValue(text, key, &v) ? atoi(v.c_str()) : dflt;
}
double asDbl(const std::string& text, const std::string& key, double dflt) {
    std::string v;
    return findValue(text, key, &v) ? atof(v.c_str()) : dflt;
}
// 读 "key": [ v1, v2, ... ] —— 设计空间里的轴都是这种形式。
// 返回其中的标量 token（字符串去引号）。找不到 / 空数组返回 false。
bool findArray(const std::string& text, const std::string& key,
               std::vector<std::string>* out) {
    std::string pat = "\"" + key + "\"";
    size_t pos = text.rfind(pat);
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos + pat.size());
    if (pos == std::string::npos) return false;
    pos = text.find('[', pos);
    if (pos == std::string::npos) return false;
    out->clear();
    size_t i = pos + 1;
    while (i < text.size() && text[i] != ']') {
        i = text.find_first_not_of(" \t\r\n,", i);
        if (i == std::string::npos || text[i] == ']') break;
        if (text[i] == '"') {
            size_t e = text.find('"', i + 1);
            if (e == std::string::npos) return false;
            out->push_back(text.substr(i + 1, e - i - 1));
            i = e + 1;
        } else {
            size_t e = i;
            while (e < text.size() && text[e] != ',' && text[e] != ']' &&
                   !isspace((unsigned char)text[e]))
                e++;
            out->push_back(text.substr(i, e - i));
            i = e;
        }
    }
    return !out->empty();
}
void toArray(const std::string& text, const char* key, std::vector<int>* out) {
    std::vector<std::string> toks;
    if (!findArray(text, key, &toks)) return;
    out->clear();
    for (const auto& t : toks) {
        if (t == "true")  { out->push_back(1); continue; }   // coefficient_residency 用布尔字面量
        if (t == "false") { out->push_back(0); continue; }
        out->push_back(atoi(t.c_str()));
    }
}
void toStringArray(const std::string& text, const char* key, std::vector<std::string>* out) {
    std::vector<std::string> toks;
    if (findArray(text, key, &toks)) *out = std::move(toks);
}
}  // namespace

// ================================================================== H 对象
Hardware Hardware::load(const std::string& profilePath) {
    Hardware hw;
    std::string t = slurp(profilePath);
    std::string v;
    if (findValue(t, "soc", &v)) hw.soc = v;
    if (findValue(t, "cann_version", &v)) hw.cannVersion = v;
    hw.aicoreNum = asInt(t, "aicore_num", 24);
    hw.vectorCoreNum = asInt(t, "vector_core_num", 48);
    hw.ubBytesPerCore = asInt(t, "ub_bytes_per_core", 196608);
    hw.vecLaneFp32 = asInt(t, "vec_lane_fp32", 128);
    hw.l2Bytes = asInt(t, "l2_bytes", 0);
    // Phase 0.3 实测事实（与 profile 的 simt/subblock 保持一致）
    hw.simt = asInt(t, "simt", 0) != 0;
    hw.subBlockNum = asInt(t, "sub_block_num", 1);
    hw.ubOffsetAlignFloats = asInt(t, "ub_offset_align_floats", 8);
    hw.minBlockOutputBytes = asInt(t, "min_block_output_bytes", 64);
    hw.matrixUnitUsable = asInt(t, "matrix_unit_usable", 0) != 0;
    return hw;
}

// ================================================================== G 对象
void Generator::genTwiddles(std::vector<float>& twr, std::vector<float>& twi) const {
    const uint32_t pad = twiddlePad();
    twr.assign(pad, 0.f);
    twi.assign(pad, 0.f);
    uint32_t off = 0;
    for (uint32_t h = 1; h <= n / 2; h <<= 1) {
        for (uint32_t j = 0; j < h; j++) {
            const double ang = -M_PI * (double)j / (double)h;
            twr[off + j] = (float)std::cos(ang);
            twi[off + j] = (float)std::sin(ang);
        }
        off += (h + 7u) & ~7u;
    }
}

void Generator::genBitReverse(std::vector<uint32_t>& dst) const {
    dst.resize(n);
    uint32_t logn = 0;
    for (uint32_t t = n; t > 1; t >>= 1) logn++;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t r = 0, t = i;
        for (uint32_t k = 0; k < logn; k++) {
            r = (r << 1) | (t & 1u);
            t >>= 1;
        }
        dst[i] = 2u * r;   // 交错布局：re[j] 来自 inter[2*rev(j)]
    }
}

// 8-plane 布局的位反转索引（字节）：ar[k] 应取输入元素 bitrev(invpos(k))
void Generator::genBitReversePlane(std::vector<uint32_t>& dst) const {
    dst.resize(n);
    uint32_t logn = 0;
    for (uint32_t t = n; t > 1; t >>= 1) logn++;
    const uint32_t K = planeK(), rows = n / K;
    uint32_t logK = 0; for (uint32_t t = K; t > 1; t >>= 1) logK++;
    for (uint32_t k = 0; k < n; k++) {
        // pos(j) = (j&(K-1))*rows + (j>>logK) 的逆：j = ((k%rows)<<logK) | (k/rows)
        uint32_t j = ((k % rows) << logK) | (k / rows);
        uint32_t r = 0, t = j;
        for (uint32_t b = 0; b < logn; b++) { r = (r << 1) | (t & 1u); t >>= 1; }
        dst[k] = 2u * r * 4u;   // 交错布局中元素 r 的字节偏移 = 8*r
    }
}

// plane -> planar 转置索引（字节）：planar[k] 来自 plane[pos(k)]
void Generator::genTransposePlane(std::vector<uint32_t>& dst) const {
    dst.resize(n);
    const uint32_t K = planeK(), rows = n / K;
    uint32_t logK = 0; for (uint32_t t = K; t > 1; t >>= 1) logK++;
    const uint32_t mask = K - 1u;
    for (uint32_t k = 0; k < n; k++)
        dst[k] = (((k & mask) * rows) + (k >> logK)) * 4u;
}

// planar -> 交错输出的 Gather 索引（长度 2n*D，字节）
// 批折叠：批 d 的输出段独占 dst[d*2n .. d*2n+2n)，两个分量分别指向 pr_d / pi_d。
// D=1 => d=0 => 4*j / 4*(n+j)，与折叠前的表逐字节相同。
void Generator::genInterleave(std::vector<uint32_t>& dst, uint32_t D) const {
    if (D == 0u) D = 1u;
    dst.resize((size_t)2 * n * D);
    const uint32_t piBase = D * n;          // pi = plan + n*D
    for (uint32_t d = 0; d < D; d++) {
        const uint32_t b = d * 2u * n;
        const uint32_t rd = d * n;
        for (uint32_t j = 0; j < n; j++) {
            dst[b + 2u * j]      = 4u * (rd + j);             // pr_d[j]
            dst[b + 2u * j + 1u] = 4u * (piBase + rd + j);    // pi_d[j]
        }
    }
}

// ================================================================== A 对象
bool Mapping::validate(const Hardware& hw, std::string* why) const {
    auto fail = [&](const char* m) {
        if (why) *why = m;
        return false;
    };
    if (udCore <= 0) return fail("ud_core must be > 0");
    if (udCore > hw.vectorCoreNum) return fail("ud_core exceeds vector_core_num");
    if (level != 0) return fail("level!=0 (multi-card) not implemented");
    if (localExchange == "shuffle" && !hw.supportsShuffleExchange())
        return fail("local_exchange=shuffle requires SIMT; not available on this SoC");
    if (localExchange != "shared" && localExchange != "register" && localExchange != "shuffle")
        return fail("unknown local_exchange");
    if (ts != 0 && ts != 1) return fail("ts must be 0 or 1");
    return true;
}

// ================================================================== P 对象
// 与 src/ascendc/fft_radix2.cpp（8-plane 布局）的 InitBuffer 清单逐条一致（含批折叠 D）：
//   bPlan 2nD + bPlane 2nD + bIdxO 2nD = 6nD（字节 ×4）  + bIdxB n + bIdxT n = 2n
//   bTwR(n+16) + bTwI(n+16)                             -> 8n + 128 字节
//   bTmp tmpF，tmpF = 2*hmax*D + max(hmax, rows*D) + 3*rows*D
//   => 4 * (6nD + 4n + 32 + tmpF) 字节
//   D=1（t2sz=hmax, hmax=n/2, rows=n/K）=> 46n + 12n/K + 128，与旧公式同值。
// plane 布局要求 tiles=n/8 是 8 的倍数 => n >= 64。
// 注意：旧公式 44n + 128 少算了 idx 4n 字节与 rows 项，n=4096 上虚报 15 KB 余量。
size_t StagePlan::ubBytes(uint32_t n) const {
    const size_t f = sizeof(float);
    // 静态可行性检查用 D 的**结构上界**（foldDFor 默认 batch/nblk 不施加并发约束），
    // 对所有 batch 都保守成立；实际运行时 D 由 Plan::prepare 按 (n,batch) 取更小值。
    const size_t D    = bfly::foldDFor(n);
    const size_t K    = planeKFor(n);
    const size_t hmax = n >> 1;
    const size_t rows = n / K;
    // 与内核 InitBuffer 同式：tmpF = 2*hmax*D + max(hmax, rows*D) + 3*rows*D
    const size_t t2sz = (hmax > rows * D) ? hmax : (rows * D);
    const size_t tmpF = 2ull * hmax * D + t2sz + 3ull * rows * D;
    const size_t plan  = 2ull * n * D * f;
    const size_t plane = 2ull * n * D * f;
    const size_t idx   = 2ull * n * D * sizeof(uint32_t)   // idxO
                       + 2ull * n * sizeof(uint32_t);      // idxB + idxT
    const size_t tw    = 2ull * (n + 16) * f;
    const size_t temps = tmpF * f;
    return plan + plane + idx + tw + temps;
}

// ================================================================== η 估算器
// 结构化计数 + 最小二乘标定。标定集：n ∈ {128,1024,4096} × B 扫描
// (1,48,96,192,384,768,1536,4096)，reps=20，48 AIV；B=1 点含额外启动开销，剔除。
// 留一交叉验证：平均误差 4.6%，最大 12.3%。
//
// 成本 = launchUs + ceil(B/udCore) * (opNs * 算子调用数 + elemNs * 元素访存数)
//
// 算子/元素按 src/ascendc/fft_radix2.cpp 逐条数出（Phase 3b：蝶形就地写回后）：
//   K-plane 平面级（h=1..K/2，K=planeKFor(n)）：每级 K/2 对蝶形，长度 rows=n/K；
//     h=1 旋转因子恒为 1 => 6 算子/对 => 3K
//     h>=2 复乘 6 算子 + 就地蝶形 4 算子 = 10 算子/对 => 5K(logK-1)
//   planar 级（h=K..n/2）：每组 10 算子 × n/(2h) 组 => 5n/h 次调用
//   收尾：5 次 Gather + 1 次 DataCopy = 6 次调用
//   元素：平面级 3n+5n(logK-1)，planar 级 5n(logn-logK)，收尾 8n
//        合计 = 5n*logn + 6n（与 logK 无关）
//
// 等效蝶形点数 R = radix^fusionLevel 的结构因子（只作用于蝶形部分，收尾不动）：
//   R 点分组、矢量长度 h，算子数 g(R) = 6(R-1) + 2R·log2R
//     （6(R-1) = 每组 R-1 次复乘；2R·log2R = log2R 层 radix-2 蝶形网络的实数加减）
//   R=2 => g=10（与实测内核逐条数出的一致），R=4 => g=34，R=8 => g=90
//   planar 区调用数 ∝ g/(R-1)、元素数 ∝ g/(R·log2R)；plane 区按同一比例缩放
//   （plane 区的 32B 对齐规则会随 R 重推，见 docs §11.3 —— 这是一个结构估算，不是实测）
Metric estimate(const Hardware& hw, const Mapping& a, const StagePlan& p,
                uint32_t n, uint32_t batch) {
    Metric m;
    if (n == 0 || batch == 0) return m;
    (void)hw;
    // scripts/calib_eta.py 最小二乘标定（A7 重标定：7 n × 7 B = 49 点，reps=20、
    // 每点取 3 轮 mean 的最小值以剔掉本机负载尖峰；**按 1/y 加权**拟合相对误差；
    // 模型见该脚本头部注释）：
    //   fit: launchUs=16.109  opNs=17.077  elemNs=0.05016
    //   residual mean=6.9% max=27.5%；LOO mean=7.3% max=30.2%
    // 与上一版（21.260 / 13.252 / 0.0597，旧模型**不含 D**）的差别就是本版建模了批折叠：
    //   旧式把 feature 当成「每 batch」，折叠点把 #op 少算 D 倍 ⇒
    //   折叠点 η 高估 +67%~+104%（n=64/4096 最坏 +90.7%，现在 -1.7%）。
    // 残差 27% 那几点全是测量噪声：同一 (n,B) 重测 mean 本身摆 10~30%（宿主负载 20~30），
    // 例如 n=64/B=16 曾测到 14.0 µs，比 launchUs 还低。
    const double launchUs = 16.109406;  // 一次 launch + 回同步
    const double opNs      = 17.07683;  // 一次矢量算子调用
    const double elemNs    = 0.05016;   // 一次元素访存

    uint32_t logn = 0; while ((1u << logn) < n) logn++;
    const uint32_t K = planeKFor(n);
    uint32_t logK = 0; while ((1u << logK) < K) logK++;
    const uint32_t rows = n / K;
    // 批折叠：D 个连续 batch 拍进同一组 Level-0 repeat（A3~A5），与内核/fft_check 同源
    const uint32_t D = foldDFor(n, batch, (uint32_t)a.udCore);
    const bool useL0 = (D > 1u) && ((n >> 3) <= 255u);

    //  R=2 的精确计数（与 include/butterfly/fft_k.hpp 的 planeKFor 同式）：
    //  平面级：(h,2h) 融合成 radix-4 —— 4h 点一块，j2==0 全平凡 16 算子、其余 28 算子；
    //           logK 为奇数时末级退回 radix-2（triv 6、非 triv 8）
    //  planar 级：merge 时 8*ceil(h/64)，否则 8*D*groups；每级再加 1 次 Muls(-ti)
    double planeOps = 0.0, planeElems = 0.0;
    uint32_t si = 0;
    for (; si + 1u < logK; si += 2u) {
        const uint32_t h = 1u << si;
        const double blocks = (double)K / (double)(4u * h);
        const double o = blocks * (16.0 + (double)(h - 1u) * 28.0);
        planeOps += o;
        planeElems += o * (double)rows;
    }
    if (si < logK) {
        const uint32_t h = 1u << si;
        const double pairs = 0.5 * (double)K;
        const double triv  = (double)K / (double)(2u * h);
        const double o = triv * 6.0 + (pairs - triv) * 8.0;
        planeOps += o;
        planeElems += o * (double)rows;
    }
    double planarOps = 0.0, planarElems = 0.0;
    for (uint32_t s = logK; s < logn; s++) {
        const uint32_t h = 1u << s;
        const uint32_t groupsH = n / (2u * h);
        const uint32_t nSlice = (h + 63u) >> 6;
        // 与内核同式：repeat = D*groupsH，且 g>1 时放宽 nSlice 约束（fft_radix2.cpp:400）
        const uint32_t rep = (uint32_t)D * groupsH;
        const uint32_t merge = (h >= 8u) && (((2u * h) % 8u) == 0u) &&
                               (((2u * h) / 8u) <= 255u) && ((h / 8u) <= 255u) &&
                               (rep <= 255u) && (nSlice <= groupsH || D > 1u);
        if (merge) {
            planarOps += 8.0 * (double)nSlice + 1.0;      // Level-0 一次盖 D 批，与 D 无关
        } else {
            planarOps += 8.0 * (double)D * (double)groupsH + 1.0;   // 逐 d × 逐 group
        }
        // 元素数与 merge 与否无关（同一批数据，只是指令打包方式不同）：
        //   每级 8 个算子各覆盖 h*groups*D = n*D/2 => 4nD，另加跨批共享的 Muls(-ti) 的 h
        planarElems += 4.0 * (double)n * (double)D + (double)h;
    }

    // ---- R != 2 的结构缩放（分母 = R=2 时的基准：8 算子/对、4n 元素/级）----
    //  旧式 opF = (g/(R-1))/8 把复数乘按 6 个实数算术计、蝶形却按矢量调用计，
    //  单位不一致（R=4 得 1.1667）；正确比值（融合后覆盖 log2R 级的调用数/元素数）与
    //  elF 同源：R=4 -> 0.875（28 调用对 32、元素 4h 对 4h 略减）。故 opF = elF。
    const int R = DesignSpace::pointSize(p.radix, p.fusionLevel);
    if (R != 2 && R >= 4 && (R & (R - 1)) == 0) {
        int l2 = 0; for (int t = R; t > 1; t >>= 1) l2++;   // = log2(R)
        const double g = 4.0 * (double)(R - 1) + 2.0 * (double)R * (double)l2;
        const double elF = (g / ((double)R * l2)) / 4.0;
        const double opF = elF;
        planeOps *= opF;   planarOps *= opF;
        planeElems *= elF; planarElems *= elF;
    }

    // plane 段：Level-0（useL0）靠 `if (useL0) break;` 只走一遍 d 循环 ⇒ 算子数与 D 无关、
    //           元素数 ×D；否则走 Level-2 + 逐批 for ⇒ 算子/元素都 ×D（D=1 时两者都 ×1）。
    if (!useL0) planeOps *= (double)D;
    planeElems *= (double)D;
    // 每 group 收尾：1 次组装载 DataCopy + 6 次逐 d Gather/DataCopy
    //   （twr/twi/idx 的一次性 GM 预取已折进 launchUs）
    const double ops = planeOps + planarOps + 1.0 + 6.0 * (double)D;
    const double elems = planeElems + planarElems + 10.0 * (double)n * (double)D;

    const uint32_t groups = (batch + D - 1u) / D;      // ceil(B/D)
    const int blocks = std::max(1, std::min(a.udCore, (int)groups));
    const double iters = std::ceil((double)groups / (double)blocks);  // 墙钟取最慢核
    m.etaUs = launchUs + iters * (ops * opNs + elems * elemNs) / 1000.0;
    return m;
}

// ================================================================== 设计空间
// config/*.json 是设计空间的唯一真源；数组轴在这里真正生效（此前只存了 raw 留档）。
DesignSpace DesignSpace::load(const std::string& path) {
    DesignSpace sp;
    sp.raw = slurp(path);
    toArray(sp.raw, "ud_core", &sp.udCores);
    toArray(sp.raw, "radix", &sp.radices);
    toArray(sp.raw, "fusion_level", &sp.fusionStages);
    toArray(sp.raw, "ts", &sp.tsValues);
    toArray(sp.raw, "coefficient_residency", &sp.coeffResidency);
    toStringArray(sp.raw, "local_exchange", &sp.localExchanges);
    toStringArray(sp.raw, "in", &sp.layoutsIn);
    toStringArray(sp.raw, "out", &sp.layoutsOut);
    sp.applyDefaults();
    return sp;
}

void DesignSpace::applyDefaults() {
    if (udCores.empty()) udCores = {1, 8, 24, 48};
    if (radices.empty()) radices = {2, 4, 8};
    if (localExchanges.empty()) localExchanges = {"shared", "register"};
    if (fusionStages.empty()) fusionStages = {1};
    if (layoutsIn.empty()) layoutsIn = {"interleaved"};
    if (layoutsOut.empty()) layoutsOut = {"interleaved"};
    if (tsValues.empty()) tsValues = {1};
    if (coeffResidency.empty()) coeffResidency = {1};
}

// R = radix^fusionLevel —— 融合后蝶形的点数（fusionLevel=1 时 R 就是 radix）。
int DesignSpace::pointSize(int radix, int fusionLevel) {
    long long R = 1;
    int exp = std::max(1, fusionLevel);
    long long base = std::max(2, radix);
    for (int i = 0; i < exp && R <= (1LL << 24); i++) R *= base;
    return (int)R;
}

// ================================================================== 枚举
const char* toString(State s) {
    switch (s) {
        case State::Infeasible: return "Infeasible";
        case State::Unverified: return "Unverified";
        case State::Feasible: return "Feasible";
        case State::Measured: return "Measured";
    }
    return "?";
}

std::vector<Candidate> enumerate(const DesignSpace& sp, const Hardware& hw,
                                 uint32_t n, uint32_t batch) {
    std::vector<Candidate> out;
    const uint32_t K = planeKFor(n);
    const uint32_t planeW = K;               // plane 区宽度
    const uint32_t planarW = (K ? n / K : 0); // planar 区每行长度
    const uint32_t widthCap = std::min(planeW, planarW ? planarW : planeW);

    auto layoutOf = [](const std::string& s) {
        return (s == "planar") ? Layout::Planar : Layout::Interleaved;
    };

    for (int ud : sp.udCores)
      for (int radix : sp.radices)
        for (int fs : sp.fusionStages)
          for (int ts : sp.tsValues)
            for (int cr : sp.coeffResidency)
              for (const std::string& lx : sp.localExchanges)
                for (const std::string& li : sp.layoutsIn)
                  for (const std::string& lo : sp.layoutsOut) {
            Candidate c;
            c.a.udCore = ud;
            c.a.ts = ts;
            c.a.localExchange = lx;
            c.p.radix = radix;
            c.p.fusionLevel = fs;
            c.p.coefficientResidency = (cr != 0);
            c.p.bitReverseInput = true;
            c.l.in = layoutOf(li);
            c.l.out = layoutOf(lo);
            c.f.butterflyCount = 1;
            c.f.stageCount = fs;

            const int R = DesignSpace::pointSize(radix, fs);
            char id[192];
            snprintf(id, sizeof(id), "r%d_ud%d_ts%d_%s_f%d_R%d", radix, ud, ts,
                     lx.c_str(), fs, R);
            c.id = id;
            if (c.l.in == Layout::Planar) c.id += "_in" + li;
            if (c.l.out == Layout::Planar) c.id += "_out" + lo;
            if (!c.p.coefficientResidency) c.id += "_noRes";

            std::string why;
            if (!c.a.validate(hw, &why)) {
                c.state = State::Infeasible;
                c.reason = why;
            } else if (!c.p.coefficientResidency) {
                c.state = State::Infeasible;
                c.reason = "kernel requires coefficient residency in UB "
                           "(coefficient_residency=false would reload twiddles per batch)";
            } else if (ts != 1) {
                c.state = State::Infeasible;
                c.reason = "kernel assumes ts=1 (twiddles + indices resident across the "
                           "batch loop); ts=0 needs a per-batch GM reload";
            } else if (n < 8 || (n & (n - 1)) != 0) {
                // 位反转索引：宿主 Generator::genBitReverse 用 floor-logn（本文件 :141,:155），
                // 内核 fft_radix2.cpp:51-52 用 ceil-logn —— 非 2 幂时两者错位，必须在框架侧拒掉
                // （fft_check.cpp:64 有同款判定，此处是 enumerate 的对应门槛）。
                c.state = State::Infeasible;
                c.reason = "n must be a power of two (host and kernel logn must agree)";
            } else if (n < 64 || planarW % 8 != 0) {
                c.state = State::Infeasible;
                char b[160];
                snprintf(b, sizeof(b),
                         "plane layout needs rows=n/K a multiple of 8 "
                         "(rows=%u, K=%u, n=%u)",
                         (unsigned)planarW, (unsigned)K, n);
                c.reason = b;
            } else if (!c.p.fits(n, hw)) {
                c.state = State::Infeasible;
                char b[160];
                snprintf(b, sizeof(b), "UB %.1f KiB > %d KiB (radix=%d, n=%u)",
                         c.p.ubBytes(n) / 1024.0, hw.ubBytesPerCore / 1024, radix, n);
                c.reason = b;
            } else if (R < 2 || (R & (R - 1)) != 0) {
                c.state = State::Infeasible;
                c.reason = "butterfly point size R=radix^fusionLevel is not a power of two";
            } else if (n % (uint32_t)R != 0) {
                c.state = State::Infeasible;
                char b[96];
                snprintf(b, sizeof(b), "R=%d does not divide n=%u", R, n);
                c.reason = b;
            } else if ((uint32_t)R > widthCap) {
                // 蝶形组要同时落进 plane 区（宽 K）和 planar 区（行 n/K）
                c.state = State::Infeasible;
                char b[160];
                snprintf(b, sizeof(b),
                         "R=%d exceeds usable butterfly width min(K=%u, n/K=%u)=%u",
                         R, planeW, planarW, widthCap);
                c.reason = b;
            } else if ((uint64_t)batch * 2ull * (uint64_t)n > 0xFFFFFFFFull) {
                // 内核 goff = b*2u*n 是 uint32（fft_radix2.cpp:97），b 可达 batch-1
                c.state = State::Infeasible;
                char b[160];
                snprintf(b, sizeof(b),
                         "batch*2*n = %llu exceeds uint32 goff in kernel "
                         "(max batch for n=%u is %llu)",
                         (unsigned long long)batch * 2ull * n, n,
                         (unsigned long long)0xFFFFFFFFull / (2ull * n));
                c.reason = b;
            } else if (c.l.in != Layout::Interleaved || c.l.out != Layout::Interleaved) {
                c.state = State::Infeasible;
                c.reason = "kernel consumes interleaved input and produces interleaved "
                           "output (planar layout would need an extra transpose pass)";
            } else {
                c.state = State::Unverified;
            }
            c.q = estimate(hw, c.a, c.p, n, batch);
            out.push_back(std::move(c));
                  }
    return out;
}

std::vector<const Candidate*> rank(const std::vector<Candidate>& cs) {
    std::vector<const Candidate*> v;
    for (const auto& c : cs)
        if (c.state != State::Infeasible) v.push_back(&c);
    std::sort(v.begin(), v.end(), [](const Candidate* x, const Candidate* y) {
        if (x->state != y->state) return x->state > y->state;  // Measured > Feasible > Unverified
        return x->q.etaUs < y->q.etaUs;
    });
    return v;
}

// ================================================================== Plan
struct Plan::Impl {
    aclrtFuncHandle fn = nullptr;
    aclrtBinHandle bin = nullptr;
    // r2c/c2r 链（fft_real.o，argBytes=40）：三内核同签名 (out,in,ex0,ex1,n,batch)
    aclrtBinHandle binReal = nullptr;
    aclrtFuncHandle fnPrep = nullptr, fnR2cPost = nullptr, fnC2rPost = nullptr;
    void* devOut = nullptr;
    void* devIn = nullptr;
    void* devTwr = nullptr;
    void* devTwi = nullptr;
    void* devOff = nullptr;
    void* devIdx = nullptr;      // 4n + 2nD 的索引张量：[idxB(n) | idxT(n) | idxOut(2nD)]
    void* devA = nullptr;        // 链路中间量：r2c=内层FFT输出，c2r=prep 满谱
    void* devB = nullptr;        // c2r：fwd 反号输出（post 的输入）
    void* devSt = nullptr;       // 半谱行距 n+16 的设备缓冲（r2c 出 / c2r 入）
    void* devPw = nullptr;       // r2c_post 的 W_n^k 表：[re(pPad) | im(pPad)]
    size_t devInCap = 0, devOutCap = 0, devACap = 0, devBCap = 0, devStCap = 0;
    size_t devPwCap = 0;
    uint32_t pwPad = 0;          // 当前 devPw 的表长（8 倍数）
    uint32_t argBytes = 0;
    uint32_t argBytesReal = 0;   // fft_real.o 的 __CCE_KernelArgSize（应为 40）
    bool prepared = false;
    bool preparedReal = false;   // r2c/r2c_post 表（按全长度 n）是否就绪
    int preparedSign = -1;       // devTwr/twi 的符号：-1 正向 / +1 c2r 反号
    uint32_t preparedN = 0;
    uint32_t preparedD = 0;   // 生成 idxO 时用的折叠系数；D 变了必须重生成
    uint32_t preparedPwN = 0;
};

Plan::Plan(Candidate c) : cand_(std::move(c)) {}

Plan::~Plan() {
    if (!impl_) return;
    if (impl_->devOut) aclrtFree(impl_->devOut);
    if (impl_->devIn) aclrtFree(impl_->devIn);
    if (impl_->devTwr) aclrtFree(impl_->devTwr);
    if (impl_->devTwi) aclrtFree(impl_->devTwi);
    if (impl_->devOff) aclrtFree(impl_->devOff);
    if (impl_->devIdx) aclrtFree(impl_->devIdx);
    if (impl_->devA) aclrtFree(impl_->devA);
    if (impl_->devB) aclrtFree(impl_->devB);
    if (impl_->devSt) aclrtFree(impl_->devSt);
    if (impl_->devPw) aclrtFree(impl_->devPw);
    if (impl_->bin) aclrtBinaryUnLoad(impl_->bin);
    if (impl_->binReal) aclrtBinaryUnLoad(impl_->binReal);
}

// 读 __CCE_KernelArgSize（与 src/host/launch.cpp 一致的 ELF 解析）
static uint32_t readArgSize(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() < 64 || *(uint32_t*)buf.data() != 0x464c457f) return 0;
    const uint64_t shoff = *(uint64_t*)(buf.data() + 40);
    const uint16_t shentsize = *(uint16_t*)(buf.data() + 58);
    const uint16_t shnum = *(uint16_t*)(buf.data() + 60);
    const uint16_t shstrndx = *(uint16_t*)(buf.data() + 62);
    if (!shoff || shstrndx >= shnum) return 0;
    auto sh = [&](int i) { return buf.data() + shoff + (uint64_t)i * shentsize; };
    const uint64_t stroff = *(uint64_t*)(sh(shstrndx) + 24);
    for (int i = 0; i < shnum; i++) {
        const uint32_t nm = *(uint32_t*)(sh(i));
        if (stroff + nm + 32 > (uint64_t)buf.size()) continue;
        if (strcmp((const char*)(buf.data() + stroff + nm), "__CCE_KernelArgSize") == 0) {
            const uint64_t off = *(uint64_t*)(sh(i) + 24);
            const uint64_t sz = *(uint64_t*)(sh(i) + 32);
            if (sz >= 4 && off + 4 <= (uint64_t)buf.size()) return *(uint32_t*)(buf.data() + off);
        }
    }
    return 0;
}

// 旋转因子/位反转索引：每次 (n, D) 变化时重新生成并上传
int Plan::prepare(uint32_t n, uint32_t batch) { return prepareSign(n, batch, -1); }

int Plan::prepareSign(uint32_t n, uint32_t batch, int sign) {
    // 折叠系数同时由 n、batch、launch 核数决定（foldDFor 的并发约束），故一并做缓存键。
    const uint32_t nblk = (cand_.a.udCore > 0) ? (uint32_t)cand_.a.udCore : 48u;
    const uint32_t D = bfly::foldDFor(n, batch, nblk);
    if (impl_->prepared && impl_->preparedN == n && impl_->preparedD == D &&
        impl_->preparedSign == sign) return 0;
    Generator g;
    g.n = n;
    std::vector<float> twr, twi;
    g.genTwiddles(twr, twi);
    // c2r：旋转因子取反 => kfft_fwd 直接输出 n·IDFT（+i 约定，内核侧 xflip 位配套翻蝶形）
    if (sign > 0) for (float& v : twi) v = -v;
    std::vector<uint32_t> offB, offT, offO;
    g.genBitReversePlane(offB);   // 位反转+去交错 -> plane 的字节偏移
    g.genTransposePlane(offT);    // plane -> planar 转置的字节偏移
    g.genInterleave(offO, D);     // planar -> 交错输出的字节偏移（长度 2n*D）
    std::vector<uint32_t> idxAll;
    idxAll.reserve(offB.size() + offT.size() + offO.size());
    idxAll.insert(idxAll.end(), offB.begin(), offB.end());
    idxAll.insert(idxAll.end(), offT.begin(), offT.end());
    idxAll.insert(idxAll.end(), offO.begin(), offO.end());   // = 4n + 2nD

    auto dev = [&](void** p, size_t bytes) {
        if (*p) { aclrtFree(*p); *p = nullptr; }
        return aclrtMalloc(p, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    };
    if (dev(&impl_->devTwr, twr.size() * 4)) return -1;
    if (dev(&impl_->devTwi, twi.size() * 4)) return -1;
    if (aclrtMemcpy(impl_->devTwr, twr.size() * 4, twr.data(), twr.size() * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE)) return -1;
    if (aclrtMemcpy(impl_->devTwi, twi.size() * 4, twi.data(), twi.size() * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE)) return -1;
    if (dev(&impl_->devIdx, idxAll.size() * 4)) return -1;
    if (aclrtMemcpy(impl_->devIdx, idxAll.size() * 4, idxAll.data(), idxAll.size() * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE)) return -1;
    impl_->prepared = true;
    impl_->preparedN = n;
    impl_->preparedD = D;
    impl_->preparedSign = sign;
    tf_.clear();
    tf_.push_back({Transform::GenTwiddles, "gen twiddles (align8 per-stage slots)"});
    tf_.push_back({Transform::GenBitReverse, "gen bitrev/transpose byte offsets (plane layout)"});
    tf_.push_back({Transform::LaunchKernel, "kfft_fwd on aclrtStream"});
    return 0;
}

// 按 .o 的 __CCE_KernelArgSize 模板打包 kfft_fwd 实参：
//   0 out | 8 in | 16 n | 20 batch | 24 twr | 32 twi | 40 idxGm(4n+2nD)
//   48 第 8 参 = 低 8 位 foldD | 高 8 位平面 K | 第 16 位 xflip（c2r 的 +i 约定）
static void buildArgs(std::vector<unsigned char>& ab, size_t argBytes,
                      void* out, void* in, uint32_t n, uint32_t batch, void* twr, void* twi,
                      void* idx, uint32_t foldD, bool xflip = false) {
    ab.assign(argBytes ? argBytes : 56, 0);
    auto putPtr = [&](size_t off, void* p) {
        unsigned long long v = (unsigned long long)(uintptr_t)p;
        if (off + 8 <= ab.size()) memcpy(ab.data() + off, &v, 8);
    };
    auto putU32 = [&](size_t off, uint32_t v) {
        if (off + 4 <= ab.size()) memcpy(ab.data() + off, &v, 4);
    };
    putPtr(0, out); putPtr(8, in);
    putU32(16, n);  putU32(20, batch);
    putPtr(24, twr); putPtr(32, twi);
    putPtr(40, idx);
    putU32(48, (foldD & 0xFFu) | (planeKFor(n) << 8) | (xflip ? (1u << 16) : 0u));
}

// 设备缓冲按需扩容（旧路径只在首次 malloc；同一 plan 换 shape 复用会越界）
static int ensureBuf(void** p, size_t* cap, size_t bytes) {
    if (*cap >= bytes) return 0;
    if (*p) { aclrtFree(*p); *p = nullptr; }
    *cap = 0;
    if (aclrtMalloc(p, bytes, ACL_MEM_MALLOC_NORMAL_ONLY)) return -1;
    *cap = bytes;
    return 0;
}

// r2c_post 的旋转因子 W_n^k = e^{-2πik/n}，k ∈ [0, n/4]（槽上取整到 8，同 fft_check）
static void genR2CTwiddles(uint32_t n, std::vector<float>& pwr,
                           std::vector<float>& pwi, uint32_t& pPad) {
    const uint32_t pk1 = (n >> 2) + 1u;
    pPad = (pk1 + 7u) & ~7u;
    pwr.assign(pPad, 0.f);
    pwi.assign(pPad, 0.f);
    for (uint32_t k = 0; k < pk1; k++) {
        const double ang = -2.0 * M_PI * (double)k / (double)n;
        pwr[k] = (float)std::cos(ang);
        pwi[k] = (float)std::sin(ang);
    }
}

// 实数内核实参：out, in, ex0, ex1, n, batch（fft_real.o argBytes=40）。
// ex0/ex1：r2c_post 是 W_n^k 表实/虚半区；prep/post 不用，传合法指针过空指针校验。
static void buildRealArgs(std::vector<unsigned char>& ab, size_t argBytes,
                          void* dst, void* src, void* ex0, void* ex1,
                          uint32_t n, uint32_t batch) {
    ab.assign(argBytes ? argBytes : 40, 0);
    auto putPtr = [&](size_t off, void* p) {
        unsigned long long v = (unsigned long long)(uintptr_t)p;
        if (off + 8 <= ab.size()) memcpy(ab.data() + off, &v, 8);
    };
    auto putU32 = [&](size_t off, uint32_t v) {
        if (off + 4 <= ab.size()) memcpy(ab.data() + off, &v, 4);
    };
    putPtr(0, dst); putPtr(8, src);
    putPtr(16, ex0); putPtr(24, ex1);
    putU32(32, n); putU32(36, batch);
}

int Plan::run(const float* in, float* out, uint32_t n, uint32_t batch) {
    if (!impl_ || !impl_->fn) return -1;
    if (prepare(n, batch)) return -2;
    // 内核 fft_radix2.cpp:97 `uint32_t goff = (uint32_t)b * 2u * n`，b 最大 batch-1。
    // batch*2*n 超 32 位即错位写；enumerate 已把这类点判 Infeasible，这里是兜底。
    if ((uint64_t)batch * 2ull * (uint64_t)n > 0xFFFFFFFFull) return -8;

    const size_t nFloats = (size_t)n * 2 * batch;
    if (ensureBuf(&impl_->devIn, &impl_->devInCap, nFloats * 4)) return -3;
    if (ensureBuf(&impl_->devOut, &impl_->devOutCap, nFloats * 4)) return -3;
    if (aclrtMemcpy(impl_->devIn, nFloats * 4, in, nFloats * 4, ACL_MEMCPY_HOST_TO_DEVICE)) return -4;

    std::vector<unsigned char> ab;
    buildArgs(ab, impl_->argBytes, impl_->devOut, impl_->devIn, n, batch,
              impl_->devTwr, impl_->devTwi, impl_->devIdx,
              bfly::foldDFor(n, batch,
                             (cand_.a.udCore > 0) ? (uint32_t)cand_.a.udCore : 48u));

    const int blocks = std::max(1, std::min(cand_.a.udCore, (int)batch));
    aclError e = aclrtLaunchKernelWithHostArgs(impl_->fn, (uint32_t)blocks, ctxStream_, nullptr,
                                               ab.data(), ab.size(), nullptr, 0);
    if (e) return -5;
    if (aclrtSynchronizeStream(ctxStream_)) return -6;
    if (aclrtMemcpy(out, nFloats * 4, impl_->devOut, nFloats * 4, ACL_MEMCPY_DEVICE_TO_HOST)) return -7;
    return 0;
}

// r2c：kfft_fwd(n/2)（实输入 n float/行零拷贝复用为交错复数）-> kfft_r2c_post。
// 链与 src/host/fft_check.cpp（AB_DIR=r2c）同口径；输出为稠密半谱（行首 n+2 float）。
int Plan::runR2C(const float* in, float* out, uint32_t n, uint32_t batch) {
    if (!impl_ || !impl_->fn || !impl_->fnR2cPost) return -1;   // fft_real.o 未加载
    // UB 预算同 fft_check 的 launch 守卫：内层 fwd 在 n/2<=4096、post 本身 <=192KB
    if (n < 128 || n > 8192 || (n & 1u)) return -8;
    if ((uint64_t)batch * 2ull * (uint64_t)n > 0xFFFFFFFFull) return -8;
    const uint32_t innerN = n >> 1;
    // fwd 的旋转因子/索引按内层长度 n/2 生成（sign -1 正向）
    if (prepareSign(innerN, batch, -1)) return -2;
    // W_n^k 表按全长度 n 键控
    if (!impl_->preparedReal || impl_->preparedPwN != n) {
        std::vector<float> pwr, pwi;
        uint32_t pPad = 0;
        genR2CTwiddles(n, pwr, pwi, pPad);
        if (ensureBuf(&impl_->devPw, &impl_->devPwCap, (size_t)2 * pPad * 4)) return -3;
        if (aclrtMemcpy(impl_->devPw, pPad * 4, pwr.data(), pPad * 4,
                        ACL_MEMCPY_HOST_TO_DEVICE)) return -4;
        if (aclrtMemcpy((char*)impl_->devPw + pPad * 4, pPad * 4, pwi.data(), pPad * 4,
                        ACL_MEMCPY_HOST_TO_DEVICE)) return -4;
        impl_->pwPad = pPad;
        impl_->preparedReal = true;
        impl_->preparedPwN = n;
    }
    const uint32_t nblk = (cand_.a.udCore > 0) ? (uint32_t)cand_.a.udCore : 48u;
    const size_t inF = (size_t)n * batch;                 // 实数输入
    const size_t hsSt = (size_t)n + 16u;                  // 半谱行距（内核 hsStride(n)）
    const size_t stF = hsSt * batch;
    if (ensureBuf(&impl_->devIn, &impl_->devInCap, (size_t)n * 2u * batch * 4u)) return -3;
    if (ensureBuf(&impl_->devA, &impl_->devACap, (size_t)n * 2u * batch * 4u)) return -3;
    if (ensureBuf(&impl_->devSt, &impl_->devStCap, stF * 4)) return -3;
    if (aclrtMemcpy(impl_->devIn, inF * 4, in, inF * 4, ACL_MEMCPY_HOST_TO_DEVICE)) return -4;

    const uint32_t blocks = (uint32_t)std::max(1, std::min(cand_.a.udCore, (int)batch));
    std::vector<unsigned char> ab, ar;
    buildArgs(ab, impl_->argBytes, impl_->devA, impl_->devIn, innerN, batch,
              impl_->devTwr, impl_->devTwi, impl_->devIdx,
              bfly::foldDFor(innerN, batch, nblk), false);
    if (aclrtLaunchKernelWithHostArgs(impl_->fn, blocks, ctxStream_, nullptr,
                                      ab.data(), ab.size(), nullptr, 0)) return -5;
    buildRealArgs(ar, impl_->argBytesReal, impl_->devSt, impl_->devA,
                  impl_->devPw, (char*)impl_->devPw + impl_->pwPad * 4, n, batch);
    if (aclrtLaunchKernelWithHostArgs(impl_->fnR2cPost, blocks, ctxStream_, nullptr,
                                      ar.data(), ar.size(), nullptr, 0)) return -5;
    if (aclrtSynchronizeStream(ctxStream_)) return -6;

    std::vector<float> st(stF);
    if (aclrtMemcpy(st.data(), stF * 4, impl_->devSt, stF * 4,
                    ACL_MEMCPY_DEVICE_TO_HOST)) return -7;
    for (uint32_t b = 0; b < batch; b++)   // 行距 n+16 -> 稠密 n+2
        memcpy(out + (size_t)b * (n + 2u), st.data() + (size_t)b * hsSt, (size_t)(n + 2u) * 4u);
    return 0;
}

// c2r：kfft_c2r_prep（半谱镜像 -> 满谱）-> kfft_fwd(n, 旋转因子取反 + xflip) ->
// kfft_c2r_post（偶 bin 提取 * 1/n）。输出含 1/n，口径同 numpy.fft.irfft。
int Plan::runC2R(const float* in, float* out, uint32_t n, uint32_t batch) {
    if (!impl_ || !impl_->fn || !impl_->fnPrep || !impl_->fnC2rPost) return -1;
    if (n < 64 || n > 4096 || (n & 1u)) return -8;          // UB 预算同 fft_check 守卫
    if ((uint64_t)batch * 2ull * (uint64_t)n > 0xFFFFFFFFull) return -8;
    if (prepareSign(n, batch, 1)) return -2;                 // +i 约定（twiddle 取反）
    const uint32_t nblk = (cand_.a.udCore > 0) ? (uint32_t)cand_.a.udCore : 48u;
    const size_t hsSt = (size_t)n + 16u;
    const size_t stF = hsSt * batch;
    const size_t fullF = (size_t)n * 2u * batch;
    if (ensureBuf(&impl_->devSt, &impl_->devStCap, stF * 4)) return -3;
    if (ensureBuf(&impl_->devA, &impl_->devACap, fullF * 4)) return -3;
    if (ensureBuf(&impl_->devB, &impl_->devBCap, fullF * 4)) return -3;
    if (ensureBuf(&impl_->devOut, &impl_->devOutCap, fullF * 4)) return -3;
    // 稠密半谱 (n+2)/行 -> 设备行距 n+16（slack 归零；prep 的镜像会整段覆盖 [n,2n)）
    std::vector<float> st(stF, 0.f);
    for (uint32_t b = 0; b < batch; b++)
        memcpy(st.data() + (size_t)b * hsSt, in + (size_t)b * (n + 2u), (size_t)(n + 2u) * 4u);
    if (aclrtMemcpy(impl_->devSt, stF * 4, st.data(), stF * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE)) return -4;

    const uint32_t blocks = (uint32_t)std::max(1, std::min(cand_.a.udCore, (int)batch));
    std::vector<unsigned char> ab, ar;
    buildRealArgs(ar, impl_->argBytesReal, impl_->devA, impl_->devSt,
                  impl_->devSt, impl_->devSt, n, batch);
    if (aclrtLaunchKernelWithHostArgs(impl_->fnPrep, blocks, ctxStream_, nullptr,
                                      ar.data(), ar.size(), nullptr, 0)) return -5;
    buildArgs(ab, impl_->argBytes, impl_->devB, impl_->devA, n, batch,
              impl_->devTwr, impl_->devTwi, impl_->devIdx,
              bfly::foldDFor(n, batch, nblk), true);
    if (aclrtLaunchKernelWithHostArgs(impl_->fn, blocks, ctxStream_, nullptr,
                                      ab.data(), ab.size(), nullptr, 0)) return -5;
    buildRealArgs(ar, impl_->argBytesReal, impl_->devOut, impl_->devB,
                  impl_->devSt, impl_->devSt, n, batch);
    if (aclrtLaunchKernelWithHostArgs(impl_->fnC2rPost, blocks, ctxStream_, nullptr,
                                      ar.data(), ar.size(), nullptr, 0)) return -5;
    if (aclrtSynchronizeStream(ctxStream_)) return -6;
    if (aclrtMemcpy(out, (size_t)n * batch * 4, impl_->devOut, (size_t)n * batch * 4,
                    ACL_MEMCPY_DEVICE_TO_HOST)) return -7;
    return 0;
}

int Plan::measure(uint32_t n, uint32_t batch, Metric* m) {
    if (!m) return -1;
    const size_t nFloats = (size_t)n * 2 * batch;
    std::vector<float> in(nFloats), o1(nFloats), o2(nFloats);
    for (size_t i = 0; i < nFloats; i++) in[i] = (float)((i * 2654435761u) % 1000) / 1000.f - 0.5f;

    // 1) 首跑：准备 twiddles + 上传输入 + 正确性由 tests 侧交叉验证
    if (run(in.data(), o1.data(), n, batch)) return -2;

    // 2) 只计 kernel。H2D/D2H 每次 2*33.5 MB（n=4096,B=4096）可页内存拷贝
    //    实测 ~16.7 ms，若计入会把 49 ms 的 kernel 测成 66 ms。故拷贝只做一次。
    std::vector<unsigned char> ab;
    buildArgs(ab, impl_->argBytes, impl_->devOut, impl_->devIn, n, batch,
              impl_->devTwr, impl_->devTwi, impl_->devIdx,
              bfly::foldDFor(n, batch,
                             (cand_.a.udCore > 0) ? (uint32_t)cand_.a.udCore : 48u));
    const uint32_t blocks = (uint32_t)std::max(1, std::min(cand_.a.udCore, (int)batch));
    auto launch = [&]() -> int {
        if (aclrtLaunchKernelWithHostArgs(impl_->fn, blocks, ctxStream_, nullptr,
                                          ab.data(), ab.size(), nullptr, 0)) return -3;
        if (aclrtSynchronizeStream(ctxStream_)) return -3;
        return 0;
    };
    if (launch()) return -4;                       // warmup
    // 计时：逐次 launch 取最小值（10 次）。用均值会把首次/偶发的页缺、同步抖动算进去，
    // 而选型循环要比较的是 "同一配置的最好情况" —— 早期 reps=5 的均值曾把 register
    // 和 shared 两个完全等价的候选测出 119.8 vs 99.2 µs 的假差异。
    const int reps = 10;
    double bestUs = 0.0;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        if (launch()) return -5;
        auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (r == 0 || us < bestUs) bestUs = us;
    }
    m->measuredUs = bestUs;

    if (aclrtMemcpy(o2.data(), nFloats * 4, impl_->devOut, nFloats * 4,
                    ACL_MEMCPY_DEVICE_TO_HOST)) return -6;   // 结果留作自洽检查

    m->etaUs = cand_.q.etaUs;

    // 3) 正确性验收：与双精度 CPU 参考比对（跨全 batch 均匀采样 8 个下标，
    //    含首/中/尾，判据同 config 里的 Q.acceptance.max_rel_err = 1e-4）。
    //    采样规则由 bfly::sampleBatches 统一（test_framework 同用），不要再写"前 4 个"。
    //    Unverified -> （通过）Feasible -> （计时回填）Measured；不通过则停在 Unverified。
    uint32_t checkIdx[8];
    const uint32_t checkB = sampleBatches(batch, 8, checkIdx, 8);
    std::vector<float> ref(2 * n);
    double maxRel = 0.0;
    for (uint32_t i = 0; i < checkB; i++) {
        const uint32_t b = checkIdx[i];
        const float* got = o2.data() + (size_t)b * 2 * n;
        refFftF32(in.data() + (size_t)b * 2 * n, ref.data(), n);
        maxRel = std::max(maxRel, maxRelScaled(got, ref.data(), 2 * n));
    }
    m->maxRelErr = maxRel;
    if (maxRel > 1e-4) {
        m->measuredOk = false;
        cand_.q = *m;
        cand_.state = State::Unverified;
        cand_.reason = "correctness check failed (maxRel > 1e-4)";
        return -7;
    }
    m->measuredOk = true;
    cand_.state = State::Feasible;      // 正确性验收通过
    cand_.q = *m;
    cand_.reason.clear();
    cand_.state = State::Measured;      // 实测 µs 已回填
    return 0;
}

// ================================================================== Context
struct Context::Impl {
    aclrtStream stream = nullptr;
    bool initialized = false;
};

Context::Context() : impl_(new Impl) {}

Context::~Context() {
    if (impl_->stream) aclrtDestroyStream(impl_->stream);
}

int Context::init(const std::string& profilePath, const std::string& spacePath,
                  const std::string& kernelPath) {
    hw_ = Hardware::load(profilePath);
    sp_ = DesignSpace::load(spacePath);
    kernelPath_ = kernelPath;
    if (aclInit(nullptr) != ACL_SUCCESS) return -1;
    if (aclrtSetDevice(0) != ACL_SUCCESS) return -2;
    if (aclrtCreateStream(&impl_->stream) != ACL_SUCCESS) return -3;
    impl_->initialized = true;
    return 0;
}

std::vector<Candidate> Context::enumerate(uint32_t n, uint32_t batch) const {
    return bfly::enumerate(sp_, hw_, n, batch);
}

std::unique_ptr<Plan> Context::makePlan(const Candidate& c) {
    // 本移植版只实现了 R=2 的 kfft_fwd；其它 R 的 η 是结构估算，没有 kernel 可载入。
    if (DesignSpace::pointSize(c.p.radix, c.p.fusionLevel) != 2) return nullptr;
    auto p = std::unique_ptr<Plan>(new Plan(c));
    p->impl_.reset(new Plan::Impl());
    p->impl_->argBytes = readArgSize(kernelPath_);
    p->ctxStream_ = impl_->stream;
    if (aclrtBinaryLoadFromFile(kernelPath_.c_str(), nullptr, &p->impl_->bin) != ACL_SUCCESS)
        return nullptr;
    if (aclrtBinaryGetFunction(p->impl_->bin, "kfft_fwd", &p->impl_->fn) != ACL_SUCCESS)
        return nullptr;
    // r2c/c2r 链（fft_real.o）：与 kernelPath 同目录，AB_REAL_O 可覆盖；缺失不影响 c2c
    const char* rop = getenv("AB_REAL_O");
    std::string rpath;
    if (rop) {
        rpath = rop;
    } else {
        size_t s = kernelPath_.find_last_of("/\\");
        rpath = (s == std::string::npos ? std::string() : kernelPath_.substr(0, s + 1))
                + "fft_real.o";
    }
    if (aclrtBinaryLoadFromFile(rpath.c_str(), nullptr, &p->impl_->binReal) == ACL_SUCCESS) {
        aclrtBinaryGetFunction(p->impl_->binReal, "kfft_c2r_prep", &p->impl_->fnPrep);
        aclrtBinaryGetFunction(p->impl_->binReal, "kfft_r2c_post", &p->impl_->fnR2cPost);
        aclrtBinaryGetFunction(p->impl_->binReal, "kfft_c2r_post", &p->impl_->fnC2rPost);
        p->impl_->argBytesReal = readArgSize(rpath);
    }
    p->tf_.push_back({Transform::GenTwiddles, "gen twiddles (align8 per-stage slots)"});
    p->tf_.push_back({Transform::GenBitReverse, "gen bitrev/transpose byte offsets (plane layout)"});
    p->tf_.push_back({Transform::LaunchKernel, "kfft_fwd on aclrtStream"});
    return p;
}

std::unique_ptr<Plan> Context::select(uint32_t n, uint32_t batch, int topK, std::string* log) {
    auto all = enumerate(n, batch);
    auto order = rank(all);
    const int k = std::max(1, topK);
    char line[448];

    if (log) {
        log->clear();
        int infeasible = 0;
        for (const auto& c : all) if (c.state == State::Infeasible) infeasible++;
        snprintf(line, sizeof(line),
                 "enumerated=%zu infeasible=%d feasible=%zu (n=%u batch=%u)\n",
                 all.size(), infeasible, order.size(), n, batch);
        *log += line;
        if (infeasible) {                       // 归并原因，便于核对设计空间的可行性判据
            std::vector<std::pair<int, std::string>> rs;
            for (const auto& c : all)
                if (c.state == State::Infeasible) {
                    bool hit = false;
                    for (auto& r : rs) if (r.second == c.reason) { r.first++; hit = true; break; }
                    if (!hit) rs.push_back({1, c.reason});
                }
            std::sort(rs.begin(), rs.end(), [](auto& x, auto& y) { return x.first > y.first; });
            for (size_t i = 0; i < rs.size() && i < 5; i++) {
                snprintf(line, sizeof(line), "  infeasible x%-4d %s\n", rs[i].first, rs[i].second.c_str());
                *log += line;
            }
        }
        snprintf(line, sizeof(line), "  eta rank (top %d):\n", k);
        *log += line;
        for (size_t i = 0; i < order.size() && i < (size_t)k; i++) {
            snprintf(line, sizeof(line), "    %2zu. %-44s eta=%7.1f us  [%s]\n",
                     i + 1, order[i]->id.c_str(), order[i]->q.etaUs, toString(order[i]->state));
            *log += line;
        }
        snprintf(line, sizeof(line), "  measured selection:\n");
        *log += line;
    }
    if (order.empty()) return nullptr;

    // ---- 实测选型循环：逐个构 plan -> 正确性验收 + 计时 -> 回填 -> 取实测最优 ----
    // 此前只构 topK 里的最后一个（还取错了索引），且从不实测，η 相同的候选只能靠顺序撞。
    // 现在：以「累计实测成功 k 个」为退出条件，而不是「看了前 k 个」——否则 top-k 被
    // 未实现的 R!=2 候选占满时会一个都测不到（设计空间里 R=4 的 η 可能低于 R=2）。
    std::unique_ptr<Plan> best;
    double bestUs = 0.0;
    int measured = 0;
    size_t examined = 0;
    for (size_t i = 0; i < order.size() && measured < k; i++) {
        examined++;
        const Candidate& c = *order[i];
        const int R = DesignSpace::pointSize(c.p.radix, c.p.fusionLevel);
        if (R != 2) {
            if (log) {
                snprintf(line, sizeof(line),
                         "    skip   %-44s eta=%7.1f us  no kernel for R=%d (only R=2 implemented)\n",
                         c.id.c_str(), c.q.etaUs, R);
                *log += line;
            }
            continue;
        }
        auto p = makePlan(c);
        if (!p) {
            if (log) {
                snprintf(line, sizeof(line), "    fail   %-44s eta=%7.1f us  makePlan failed\n",
                         c.id.c_str(), c.q.etaUs);
                *log += line;
            }
            continue;
        }
        Metric m;
        int rc = p->measure(n, batch, &m);
        if (rc) {
            if (log) {
                snprintf(line, sizeof(line),
                         "    reject %-44s eta=%7.1f us  rc=%d maxRel=%.2e  [%s]\n",
                         c.id.c_str(), c.q.etaUs, rc, m.maxRelErr, toString(p->candidate().state));
                *log += line;
            }
            continue;
        }
        measured++;
        if (log) {
            snprintf(line, sizeof(line),
                     "    accept %-44s eta=%7.1f us  measured=%7.1f us  maxRel=%.1e  [%s]\n",
                     p->candidate().id.c_str(), m.etaUs, m.measuredUs, m.maxRelErr,
                     toString(p->candidate().state));
            *log += line;
        }
        if (!best || m.measuredUs < bestUs) { bestUs = m.measuredUs; best = std::move(p); }
    }
    if (log) {
        if (best)
            snprintf(line, sizeof(line),
                     "  selected: %s  (%.1f us, %d accepted after %zu examined, quota %d)\n",
                     best->candidate().id.c_str(), bestUs, measured, examined, k);
        else
            snprintf(line, sizeof(line),
                     "  selected: none (%d accepted after %zu examined, quota %d)\n",
                     measured, examined, k);
        *log += line;
    }
    return best;
}

}  // namespace bfly
