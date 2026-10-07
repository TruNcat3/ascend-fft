// Phase 2 冒烟测试：加载 H/设计空间 -> 四态枚举 -> η 预排序 -> 选型 ->
// 构建 Plan -> 真跑 kernel -> 与双精度 CPU 参考比对 -> 回填 Measured。
// 参考实现与比对判据统一走 src/framework/reference.cpp（框架内部验收也用同一份）。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "butterfly/plan.hpp"
#include "butterfly/reference.hpp"

using bfly::Context;
using bfly::State;

int main(int argc, char** argv) {
    const char* profile = argc > 1 ? argv[1] : "config/ascend910_93_profile.json";
    const char* space   = argc > 2 ? argv[2] : "config/butterfly_space.json";
    const char* kernel  = argc > 3 ? argv[3] : "build/fft_radix2.o";
    const uint32_t n = argc > 4 ? (uint32_t)atoi(argv[4]) : 4096;
    const uint32_t batch = argc > 5 ? (uint32_t)atoi(argv[5]) : 64;

    Context ctx;
    if (int rc = ctx.init(profile, space, kernel)) {
        printf("context init failed rc=%d\n", rc);
        return 1;
    }
    const auto& hw = ctx.hardware();
    printf("H: soc=%s cann=%s aicore=%d vector_core=%d ub=%d lane=%d simt=%d subblock=%d "
           "align=%dF minOutLine=%dB matrix_unit=%d\n",
           hw.soc.c_str(), hw.cannVersion.c_str(), hw.aicoreNum, hw.vectorCoreNum,
           hw.ubBytesPerCore, hw.vecLaneFp32, (int)hw.simt, hw.subBlockNum,
           hw.ubOffsetAlignFloats, hw.minBlockOutputBytes, (int)hw.matrixUnitUsable);

    std::string log;
    auto plan = ctx.select(n, batch, 3, &log);
    printf("%s", log.c_str());
    if (!plan) { printf("no feasible candidate\n"); return 1; }
    printf("selected: %s\n", plan->candidate().id.c_str());
    for (const auto& t : plan->transforms())
        printf("  transform [%d] %s\n", (int)t.kind, t.label.c_str());

    // 真跑 + 与双精度参考比对
    const size_t nf = (size_t)n * 2 * batch;
    std::vector<float> in(nf), out(nf), ref(2 * n);   // ref: 单 batch 参考
    for (size_t i = 0; i < nf; i++) in[i] = (float)((i * 2654435761u) % 1000) / 1000.f - 0.5f;
    if (int rc = plan->run(in.data(), out.data(), n, batch)) {
        printf("run failed rc=%d\n", rc);
        return 1;
    }
    double maxRel = 0;
    uint32_t checkIdx[8];
    const uint32_t checkB = bfly::sampleBatches(batch, 8, checkIdx, 8);
    for (uint32_t i = 0; i < checkB; i++) {
        const uint32_t b = checkIdx[i];
        bfly::refFftF32(in.data() + (size_t)b * 2 * n, ref.data(), n);
        maxRel = std::max(maxRel,
                          bfly::maxRelScaled(out.data() + (size_t)b * 2 * n, ref.data(), 2 * n));
    }
    bfly::Metric m;
    if (int rc = plan->measure(n, batch, &m)) { printf("measure failed rc=%d\n", rc); return 1; }
    printf("maxRel=%.3e (accept 1e-4)  eta=%.1f us  measured=%.1f us  state=%s\n",
           maxRel, m.etaUs, m.measuredUs, bfly::toString(plan->candidate().state));
    if (maxRel > 1e-4) { printf("FAIL\n"); return 1; }

    // r2c/c2r 冒烟：fft_real.o 缺失（rc=-1）或 n 越界时跳过；判据同 1e-4
    bool realOk = true;
    if (n >= 128 && n <= 8192 && (n & 1u) == 0) {
        std::vector<float> rin((size_t)n * batch), rout((size_t)(n + 2) * batch), rref(n + 2);
        for (size_t i = 0; i < rin.size(); i++) rin[i] = bfly::patternAt(i);
        if (int rc = plan->runR2C(rin.data(), rout.data(), n, batch)) {
            if (rc == -1) printf("r2c skipped (fft_real.o not loaded)\n");
            else { printf("r2c run failed rc=%d\n", rc); realOk = false; }
        } else {
            double rel = 0;
            const uint32_t checkB = bfly::sampleBatches(batch, 8, checkIdx, 8);
            for (uint32_t i = 0; i < checkB; i++) {
                const uint32_t b = checkIdx[i];
                bfly::refR2CF32(rin.data() + (size_t)b * n, rref.data(), n);
                rel = std::max(rel, bfly::maxRelScaled(rout.data() + (size_t)b * (n + 2),
                                                       rref.data(), n + 2));
            }
            printf("r2c: maxRel=%.3e\n", rel);
            if (rel > 1e-4) realOk = false;
        }
    } else {
        printf("r2c skipped (n=%u outside 128..8192)\n", n);
    }
    if (n >= 64 && n <= 4096 && (n & 1u) == 0) {
        std::vector<float> hin((size_t)(n + 2) * batch), hout((size_t)n * batch), rref(n);
        // c2r 输入 = 任意合法稠密半谱（逐元素 pattern）；唯一约定：Nyquist 虚部为 0
        for (size_t i = 0; i < hin.size(); i++) hin[i] = bfly::patternAt(i);
        for (uint32_t b = 0; b < batch; b++) hin[(size_t)b * (n + 2) + n + 1] = 0.f;
        if (int rc = plan->runC2R(hin.data(), hout.data(), n, batch)) {
            if (rc == -1) printf("c2r skipped (fft_real.o not loaded)\n");
            else { printf("c2r run failed rc=%d\n", rc); realOk = false; }
        } else {
            double rel = 0;
            const uint32_t checkB = bfly::sampleBatches(batch, 8, checkIdx, 8);
            for (uint32_t i = 0; i < checkB; i++) {
                const uint32_t b = checkIdx[i];
                bfly::refC2RF32(hin.data() + (size_t)b * (n + 2), rref.data(), n);
                rel = std::max(rel, bfly::maxRelScaled(hout.data() + (size_t)b * n,
                                                       rref.data(), n));
            }
            printf("c2r: maxRel=%.3e\n", rel);
            if (rel > 1e-4) realOk = false;
        }
    } else {
        printf("c2r skipped (n=%u outside 64..4096)\n", n);
    }
    if (!realOk) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
