# 长 FFT 同尺寸基线：torch_npu vs 自研 device 边界链

本页由 `python3 scripts/bench_long_baseline.py --check` 从归档 `results/evidence/long-fft-baseline/baseline.json` 渲染校验；请勿手改。

- 基线：`torch.fft.fft`（torch_npu op-plugin，复->复）同网格同语义，device-only 取 `bench_native_npu.py` 的 min，E2E 取其 pinned 口径 min。
- 自研：device 边界链 5-trial 归档的 `device_chain` / E2E 中位数（`segments:` 六段之和即 chain）。自研长链 E2E 当前从 `std::vector` 发起（`fft_check.cpp` 的 pinned 分配仅覆盖短路径），传输为 pageable 口径；E2E 列的差距包含此项，与 device-only 列要分开解读。
- `speedup = native_min / ours_median`，>1 表示自研更快；误差门槛 `maxRel ≤ 0.0001`。

| N | B | native dev min (us) | ours chain median (us) | dev speedup | native E2E min (us) | ours E2E median (us) | E2E speedup | ours maxRel | status |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 111.9 | 158.0 | 0.71x | 182.8 | 348.1 | 0.53x | 5.12e-08 | PASS |
| 8192 | 3 | 143.1 | 164.7 | 0.87x | 220.8 | 360.6 | 0.61x | 1.16e-07 | PASS |
| 8192 | 47 | 172.9 | 384.2 | 0.45x | 445.6 | 2304.2 | 0.19x | 1.99e-07 | PASS |
| 16384 | 1 | 119.7 | 160.9 | 0.74x | 191.1 | 360.8 | 0.53x | 2.27e-07 | PASS |
| 16384 | 3 | 181.0 | 171.1 | 1.06x | 283.9 | 615.4 | 0.46x | 2.27e-07 | PASS |
| 16384 | 47 | 227.5 | 457.6 | 0.50x | 570.2 | 3087.7 | 0.18x | 2.27e-07 | PASS |
| 32768 | 1 | 119.6 | 170.8 | 0.70x | 217.2 | 517.4 | 0.42x | 3.65e-08 | PASS |
| 32768 | 3 | 174.8 | 186.6 | 0.94x | 297.2 | 686.6 | 0.43x | 9.73e-08 | PASS |
| 32768 | 47 | 330.1 | 897.2 | 0.37x | 924.4 | 7972.7 | 0.12x | 1.31e-07 | PASS |
| 65536 | 1 | 143.9 | 194.2 | 0.74x | 253.3 | 639.0 | 0.40x | 8.04e-08 | PASS |
| 65536 | 3 | 231.8 | 209.9 | 1.10x | 384.0 | 730.8 | 0.53x | 8.04e-08 | PASS |
| 65536 | 47 | 715.9 | 1832.9 | 0.39x | 1818.6 | 13674.9 | 0.13x | 2.27e-07 | PASS |

数据源：`results/evidence/long-fft-baseline/baseline.json`（基线，git `6a1b3d5c`，torch 2.10.0+cpu / torch_npu 2.10.0，CANN 9.0.0）与 `results/evidence/long-fft-device-boundary/acceptance.json`（自研，git `50a5328d`，binary `bd0e9de55521`）。
