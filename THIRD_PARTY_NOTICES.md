# Third-Party Notices

## cuButterfly

The design of this project — the butterfly/layout design-space object set
(`H/G/A/P/L/F/Q`), the `Context` / `Plan` / `Transform` API shape, batch-parallel
loop decomposition, twiddle-table + bit-reverse index generation, and the
space-search + cost-model strategy — is a port of:

- **Repository**: <https://github.com/TruNcat3/cuButterfly>
- **Upstream description**: *Hardware-mapped space-time parallelism for FFT, NTT,
  FWHT, and butterfly computations on GPUs*
- **Upstream license**: BSD 3-Clause

No upstream CUDA/C++ source file is copied verbatim into this repository; every
translation unit here is an AscendC / host C++ rewrite targeting the Ascend
`Ascend910_9382` SoC. Upstream is referenced for design provenance only.

```
BSD 3-Clause License

Copyright (c) 2026 Teng Wang, High Efficient Intelligent Computing Lab,
Suzhou Institute for Advanced Research of USTC
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Huawei CANN / Ascend toolkit

This repository links against `libascendcl` from the Huawei CANN toolkit and
compiles device code with the CANN `ccec` compiler. Those components are **not**
redistributed here and are covered by their own license terms.
