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
    printf("PASS\n");
    return 0;
}
