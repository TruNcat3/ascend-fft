// 双精度 FFT 参考实现 + 与 src/host/fft_check.cpp 一致的相对误差判据。
// 供 Plan::measure 做验收、tests/test_framework.cpp 做交叉验证共用。
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace bfly {

// 原地无关：in 长 2n，out 长 2n（交错 [re0,im0,re1,im1,...]）。
// radix-2 DIT，先位反转再蝶形，旋转因子 exp(-2*pi*i*j/n)。
void refFftF32(const float* in, float* out, uint32_t n);

// r2c 参考：in 长 n 实数，out 长 n+2（前 n/2+1 个交错复数）== numpy.fft.rfft。
void refR2CF32(const float* in, float* out, uint32_t n);

// c2r 参考：in 长 n+2 稠密半谱（Nyquist 虚部须为 0），out 长 n 实数（含 1/n）
// == numpy.fft.irfft 的 n 点口径。镜像规则与 fft_check/内核 prep 一致。
void refC2RF32(const float* in, float* out, uint32_t n);

// 混合绝对/相对误差：max |got-ref| / max(1, max |ref|)，按 float 分量取模。
// 非有限输入返回 infinity；全零或小幅参考结果使用绝对误差，不可直接视为通过。
// 三处验收必须用同一个口径，否则三套测试报的不是同一个指标：
//   - src/framework/butterfly.cpp  Plan::measure
//   - tests/test_framework.cpp
//   - src/host/fft_check.cpp（旧版用复数模 max|y[j]| 作分母，已并到此处）
// 实现放头文件做 inline，因为 fft_check 是单文件链接、不带 src/framework/reference.cpp。
inline double maxRelScaled(const float* got, const float* ref, size_t len) {
    if (len && (!got || !ref)) return std::numeric_limits<double>::infinity();
    double scale = 0, maxError = 0;
    for (size_t i = 0; i < len; i++) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]))
            return std::numeric_limits<double>::infinity();
        scale = std::max(scale, std::fabs((double)ref[i]));
        maxError = std::max(maxError, std::fabs((double)got[i] - (double)ref[i]));
    }
    // Below unit scale, the shared 1e-4 gate is an absolute tolerance.
    return maxError / std::max(1.0, scale);
}

// 确定性输入图案（与 tests/test_framework.cpp / fft_check 一致）
float patternAt(size_t i);

// 跨全 batch 均匀采样至多 want 个下标（含 0 与 batch-1）。三处验收共用，
// 以免 Plan::measure 只查前 4 个、test_framework 只查前 8 个而漏掉尾部 batch 的错误。
inline uint32_t sampleBatches(uint32_t batch, uint32_t want,
                              uint32_t* out, uint32_t maxOut) {
    if (want > maxOut) want = maxOut;
    if (batch == 0 || want == 0) return 0;
    if (batch <= want) {
        for (uint32_t i = 0; i < batch; i++) out[i] = i;
        return batch;
    }
    if (want == 1) { out[0] = 0; return 1; }
    for (uint32_t i = 0; i < want; i++)
        out[i] = (uint32_t)((uint64_t)i * (uint64_t)(batch - 1) / (uint64_t)(want - 1));
    return want;
}

}  // namespace bfly
