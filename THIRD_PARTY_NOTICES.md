# Third-Party Notices

## cuButterfly

The design of this project — the butterfly/layout design-space object set
(`H/G/A/P/L/F/Q`), the `Context` / `Plan` / `Transform` API shape, batch-parallel
loop decomposition, twiddle-table + bit-reverse index generation, and the
space-search + cost-model strategy — is a port of:

- **Repository**: <https://github.com/TruNcat3/cuButterfly>
- **Upstream description**: *Hardware-mapped space-time parallelism for FFT, NTT,
  FWHT, and butterfly computations on GPUs*
- **Upstream license**: Research-Only Non-Commercial License (RONC-1.0); see
  the upstream repository's `LICENSE` for the authoritative terms.

No upstream CUDA/C++ source file is copied verbatim into this repository; every
translation unit here is an AscendC / host C++ rewrite targeting the Ascend
`Ascend910_9382` SoC. Upstream is referenced for design provenance only.

No upstream source is copied verbatim, so this repository does not redistribute
the upstream license text. The upstream repository is cited for method
provenance only.

## Huawei CANN / Ascend toolkit

This repository links against `libascendcl` from the Huawei CANN toolkit and
compiles device code with the CANN `ccec` compiler. Those components are **not**
redistributed here and are covered by their own license terms.

## University of Science and Technology of China identifier

`docs/assets/ustc-logo.png` is an institutional identifier obtained from the
[official USTC website](https://www.ustc.edu.cn/images/zkdlogo.png). It is used
only to state the maintainer's research affiliation. The USTC name and mark are
not granted under this repository's RONC-1.0 software license and remain the property
of their respective owner. Their appearance does not imply that USTC or the
Suzhou Institute for Advanced Research officially releases, certifies, sponsors,
or endorses Ascend-FFT.
