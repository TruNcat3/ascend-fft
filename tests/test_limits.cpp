// 边界与负例测试（Part B）：
//   1) n 下界 / 非 2 幂 / UB 上界 —— 必须全部 Infeasible 且 reason 命中
//   2) 非典型 batch —— 必须能跑通且 maxRel < 1e-4
//   3) goff 32 位溢出边界 —— batch=524288 拒绝、524287 放行（只枚举，不分配）
//   4) StagePlan::ubBytes 的黄金值（防止回退到旧的 44n+128）
// 与 tests/test_framework.cpp 共用同一 Context / 同一套 maxRelScaled 口径。
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "butterfly/plan.hpp"
#include "butterfly/reference.hpp"
#include "butterfly/descriptors.hpp"

using bfly::Context;
using bfly::State;

static int g_fail = 0;
static int g_pass = 0;

static void ok(const char* label) { g_pass++; printf("ok   %s\n", label); }
static void bad(const char* label, const char* fmt, ...) {
    g_fail++;
    printf("FAIL %s: ", label);
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n");
}

// 全部候选必须 Infeasible，且至少有一个 reason 命中 key
static void expectInfeasible(Context& ctx, uint32_t n, uint32_t batch,
                             const char* key, const char* label) {
    auto cs = ctx.enumerate(n, batch);
    if (cs.empty()) { bad(label, "enumerate 返回空"); return; }
    for (const auto& c : cs)
        if (c.state != State::Infeasible) {
            bad(label, "n=%u B=%u 候选 %s 状态=%s（应 Infeasible）",
                n, batch, c.id.c_str(), bfly::toString(c.state));
            return;
        }
    std::string sample = cs.front().reason;
    for (const auto& c : cs)
        if (c.reason.find(key) != std::string::npos) { sample = c.reason; break; }
    if (sample.find(key) == std::string::npos) {
        bad(label, "n=%u B=%u 没有 reason 含 \"%s\"（样本: %s）", n, batch, key, sample.c_str());
        return;
    }
    ok(label);
}

// 至少有一个候选不是 Infeasible（放行）
static void expectLive(Context& ctx, uint32_t n, uint32_t batch, const char* label) {
    auto cs = ctx.enumerate(n, batch);
    for (const auto& c : cs)
        if (c.state != State::Infeasible) { ok(label); return; }
    bad(label, "n=%u B=%u 全部候选 Infeasible（应至少放行一个）: %s",
        n, batch, cs.empty() ? "" : cs.front().reason.c_str());
}

// 真跑一个 n×batch 并与双精度参考比对
static void runCheck(Context& ctx, uint32_t n, uint32_t batch, const char* label,
                     bool allBatches = false) {
    auto plan = ctx.select(n, batch, 3, nullptr);
    if (!plan) { bad(label, "n=%u B=%u 无可行候选", n, batch); return; }
    const size_t nf = (size_t)n * 2 * batch;
    std::vector<float> in(nf), out(nf), ref(2 * n);
    for (size_t i = 0; i < nf; i++) in[i] = bfly::patternAt(i);
    if (int rc = plan->run(in.data(), out.data(), n, batch)) {
        bad(label, "n=%u B=%u run rc=%d", n, batch, rc);
        return;
    }
    if (!std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); })) {
        bad(label, "n=%u B=%u non-finite output", n, batch);
        return;
    }
    double maxRel = 0;
    uint32_t checkIdx[8];
    const uint32_t checkB = allBatches ? batch : bfly::sampleBatches(batch, 8, checkIdx, 8);
    for (uint32_t i = 0; i < checkB; i++) {
        const uint32_t b = allBatches ? i : checkIdx[i];
        bfly::refFftF32(in.data() + (size_t)b * 2 * n, ref.data(), n);
        const double error = bfly::maxRelScaled(
            out.data() + (size_t)b * 2 * n, ref.data(), 2 * n);
        if (!std::isfinite(error)) {
            bad(label, "n=%u B=%u row=%u non-finite error", n, batch, b);
            return;
        }
        maxRel = std::max(maxRel, error);
    }
    if (!std::isfinite(maxRel) || maxRel > 1e-4) { bad(label, "n=%u B=%u maxRel=%.3e > 1e-4", n, batch, maxRel); return; }
    printf("     n=%u B=%u maxRel=%.3e\n", n, batch, maxRel);
    ok(label);
}

int main(int argc, char** argv) {
    const char* profile = argc > 1 ? argv[1] : "config/ascend910_93_profile.json";
    const char* space   = argc > 2 ? argv[2] : "config/butterfly_space.json";
    const char* kernel  = argc > 3 ? argv[3] : "build/fft_radix2.o";

    Context ctx;
    if (int rc = ctx.init(profile, space, kernel)) {
        printf("context init failed rc=%d\n", rc);
        return 1;
    }
    const bfly::Hardware& hw = ctx.hardware();
    const uint32_t ub = (uint32_t)hw.ubBytesPerCore;
    printf("ub=%u\n", ub);

    // ---------- 4) ubBytes 黄金值（B1-2）----------
    {
        bfly::StagePlan p;   // 默认 coefficientResidency/radix 与内核一致
        const uint32_t probeN = 4096;
        const size_t got = p.ubBytes(probeN);
        const uint32_t K = bfly::planeKFor(probeN);
        const size_t expect = 46ull * probeN + 12ull * probeN / K + 128ull;
        if (got != expect)
            bad("ubBytes formula", "ubBytes(4096)=%zu != 46n+12n/K+128=%zu (K=%u)",
                got, expect, K);
        else if (got <= 44ull * probeN + 128ull)
            bad("ubBytes fix active", "ubBytes(4096)=%zu 仍是旧的 44n+128=%zu",
                got, 44ull * probeN + 128ull);
        else if (got > ub)
            bad("ubBytes fits", "ubBytes(4096)=%zu > ub=%u（应仍可跑 n=4096）", got, ub);
        else {
            printf("     ubBytes(4096)=%zu (K=%u)  margin=%zu bytes\n", got, K, ub - got);
            ok("ubBytes formula / fix active / fits");
        }
        // 旧公式虚报的余量必须已经消失
        const size_t oldMargin = ub - (44ull * probeN + 128ull);
        const size_t newMargin = ub - got;
        if (newMargin >= oldMargin)
            bad("ubBytes margin honest", "新余量 %zu >= 旧虚报余量 %zu（修正未生效）",
                newMargin, oldMargin);
        else
            ok("ubBytes margin honest");
        // R0.1 单一来源：ubBytes 必须逐字节等于共享宏（同一结构 D/K），
        // 否则 StagePlan 与 long_fft_ub.h 的公式又漂移了。
        {
            const uint32_t sD = bfly::foldDFor(probeN);
            const size_t viaMacro = (size_t)AB_ROW_FFT_UB_BYTES(probeN, sD, K);
            if (viaMacro != got)
                bad("ubBytes single source",
                    "ubBytes(4096)=%zu != AB_ROW_FFT_UB_BYTES=%zu (D=%u)",
                    got, viaMacro, sD);
            else
                ok("ubBytes single source (macro)");
        }
        // rows 感知 (D,K) 解析 —— descriptor 门禁与 launch 的同一入口：
        {
            using butterfly::resolve_row_fft;
            auto r1 = resolve_row_fft(64, 128, 0, 0);      // 8192x1 pass1
            auto r2 = resolve_row_fft(128, 64, 0, 0);      // 8192x1 pass2
            auto r3 = resolve_row_fft(64, 524288, 0, 0);   // 8192x4096 pass1
            auto r4 = resolve_row_fft(4096, 4096, 0, 0);   // n > 2048
            auto ro = resolve_row_fft(64, 128, 4, 32);     // override + K clamp
            const size_t ub1 = (size_t)AB_ROW_FFT_UB_BYTES(64, r1.fold_d, r1.plane_k);
            const size_t ub2 = (size_t)AB_ROW_FFT_UB_BYTES(128, r2.fold_d, r2.plane_k);
            const size_t ub3 = (size_t)AB_ROW_FFT_UB_BYTES(64, r3.fold_d, r3.plane_k);
            if (r1.fold_d != 2 || r1.plane_k != 8 || ub1 != 5056ull)
                bad("resolve_row_fft 8192x1 pass1",
                    "D=%u K=%u ub=%zu != 2/8/5056", r1.fold_d, r1.plane_k, ub1);
            else if (r2.fold_d != 1 || r2.plane_k != 8 || ub2 != 6208ull)
                bad("resolve_row_fft 8192x1 pass2",
                    "D=%u K=%u ub=%zu != 1/8/6208", r2.fold_d, r2.plane_k, ub2);
            else if (r3.fold_d != 4 || ub3 != 8832ull)
                bad("resolve_row_fft 8192x4096 pass1",
                    "D=%u ub=%zu != 4/8832", r3.fold_d, ub3);
            else if (r4.fold_d != 1 || r4.plane_k != 16)
                bad("resolve_row_fft n=4096",
                    "D=%u K=%u != 1/16", r4.fold_d, r4.plane_k);
            else if (ro.fold_d != 4 || ro.plane_k != 8)
                bad("resolve_row_fft overrides",
                    "forced D=4 kept=%d, K=32 illegal for n=64 must clamp to 8, got %u",
                    ro.fold_d == 4, ro.plane_k);
            else
                ok("resolve_row_fft rows-aware (D,K) goldens + K clamp");
        }
    }

    // ---------- 1) n 下界 / 非 2 幂 / UB 上界（B1-1）----------
    expectInfeasible(ctx, 8,     64, "rows",        "n=8 below floor");
    expectInfeasible(ctx, 16,    64, "rows",        "n=16 below floor");
    expectInfeasible(ctx, 32,    64, "rows",        "n=32 below floor");
    expectInfeasible(ctx, 72,    64, "power of two", "n=72 not power of two");
    expectInfeasible(ctx, 100,   64, "power of two", "n=100 not power of two");
    expectInfeasible(ctx, 192,   64, "power of two", "n=192 not power of two");
    expectInfeasible(ctx, 8192,  64, "UB",           "n=8192 over UB");

    // n=8192 的余量必须是负的（新公式下 380,032 > 196,608）
    {
        auto cs = ctx.enumerate(8192, 64);
        for (const auto& c : cs) {
            if (c.p.ubBytes(8192) <= ub) {
                bad("n=8192 over UB", "ubBytes(8192)=%zu <= ub=%u", c.p.ubBytes(8192), ub);
                break;
            }
        }
        ok("n=8192 over UB (numeric)");
    }

    // ---------- 3) goff 32 位溢出（B1-5）----------
    // batch*2*n <= 0xFFFFFFFF 放行；n=4096 -> 2n=8192 -> 上限 batch=524287
    {
        const uint32_t n = 4096;
        expectLive(ctx, n, 524287, "goff boundary below (B=524287)");
        expectInfeasible(ctx, n, 524288, "goff", "goff boundary at (B=524288)");
        expectInfeasible(ctx, n, 600000, "goff", "goff boundary above (B=600000)");
    }

    // ---------- 2) 非典型 batch（真跑）----------
    runCheck(ctx, 1024, 2,    "batch=2");
    runCheck(ctx, 1024, 3,    "batch=3");
    runCheck(ctx, 1024, 100,  "batch=100");
    runCheck(ctx, 1024, 1000, "batch=1000");
    runCheck(ctx, 64,   3,    "batch=3 @ n=64");

    // 48 AIV with fold D=1..4: exercise complete waves and both tails.
    // N=1024 keeps full-row reference checks inexpensive while permitting folding.
    for (uint32_t batch : {47u, 48u, 49u, 95u, 96u, 97u,
                           143u, 144u, 145u, 191u, 192u, 193u}) {
        const std::string label = "AIV/fold boundary batch=" + std::to_string(batch);
        runCheck(ctx, 1024, batch, label.c_str(), true);
    }

    printf("\n%d passed, %d failed  ->  %s\n", g_pass, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
