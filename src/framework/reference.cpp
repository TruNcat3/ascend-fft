#include "butterfly/reference.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace bfly {

float patternAt(size_t i) {
    return (float)((i * 2654435761u) % 1000) / 1000.f - 0.5f;
}

void refFftF32(const float* in, float* out, uint32_t n) {
    if (!n) return;
    std::vector<std::complex<double>> a(n);
    for (uint32_t i = 0; i < n; i++) a[i] = {(double)in[2 * i], (double)in[2 * i + 1]};

    uint32_t logn = 0;
    for (uint32_t t = n; t > 1; t >>= 1) logn++;
    std::vector<uint32_t> rev(n, 0);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t r = 0, t = i;
        for (uint32_t k = 0; k < logn; k++) { r = (r << 1) | (t & 1u); t >>= 1; }
        rev[r] = i;
    }
    {   // 必须先拷贝：就地置换会覆盖源
        auto orig = a;
        for (uint32_t i = 0; i < n; i++) a[i] = orig[rev[i]];
    }

    std::vector<std::complex<double>> w(n / 2);
    for (uint32_t i = 0; i < n / 2; i++)
        w[i] = {std::cos(-2 * M_PI * i / n), std::sin(-2 * M_PI * i / n)};

    for (uint32_t len = 2; len <= n; len <<= 1) {
        uint32_t half = len >> 1, step = n / len;
        for (uint32_t base = 0; base < n; base += len)
            for (uint32_t j = 0; j < half; j++) {
                auto u = a[base + j];
                auto v = a[base + j + half] * w[j * step];
                a[base + j] = u + v;
                a[base + j + half] = u - v;
            }
    }
    for (uint32_t i = 0; i < n; i++) {
        out[2 * i] = (float)a[i].real();
        out[2 * i + 1] = (float)a[i].imag();
    }
}

// maxRelScaled 挪到 include/butterfly/reference.hpp 做 inline：
// src/host/fft_check.cpp 是单文件链接（不带本文件），需要共用同一个口径。

}  // namespace bfly
