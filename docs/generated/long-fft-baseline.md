# 长 FFT 同尺寸基线：torch_npu vs 自研 device 边界链

本页由 `python3 scripts/bench_long_baseline.py --check` 从归档 `results/evidence/long-fft-baseline/baseline.json` 渲染校验；请勿手改。

- 基线：`torch.fft.fft`（torch_npu op-plugin，复->复）同网格同语义，device-only 取 `bench_native_npu.py` 的 min，E2E 取其 pinned 口径 min。
- 自研：device 边界链 5-trial 归档的 `device_chain` / E2E 中位数（`segments:` 六段之和即 chain）。自研长链 E2E 当前从 `std::vector` 发起（`fft_check.cpp` 的 pinned 分配仅覆盖短路径），传输为 pageable 口径；E2E 列的差距包含此项，与 device-only 列要分开解读。
- `speedup = native_min / ours_median`，>1 表示自研更快；误差门槛 `maxRel ≤ 0.0001`。

| N | B | native dev min (us) | ours chain median (us) | dev speedup | native E2E min (us) | ours E2E median (us) | E2E speedup | ours maxRel | status |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 92.6 | 167.8 | 0.55x | 155.3 | 372.7 | 0.42x | 5.12e-08 | PASS |
| 8192 | 3 | 126.6 | 175.8 | 0.72x | 192.3 | 402.8 | 0.48x | 1.16e-07 | PASS |
| 8192 | 47 | 154.1 | 379.9 | 0.41x | 349.8 | 1474.2 | 0.24x | 1.99e-07 | PASS |
| 16384 | 1 | 100.7 | 157.2 | 0.64x | 169.4 | 339.5 | 0.50x | 2.27e-07 | PASS |
| 16384 | 3 | 165.3 | 167.3 | 0.99x | 251.1 | 433.7 | 0.58x | 2.27e-07 | PASS |
| 16384 | 47 | 208.0 | 453.7 | 0.46x | 517.7 | 2628.4 | 0.20x | 2.27e-07 | PASS |
| 32768 | 1 | 114.5 | 179.9 | 0.64x | 204.2 | 442.5 | 0.46x | 3.65e-08 | PASS |
| 32768 | 3 | 167.1 | 190.0 | 0.88x | 273.6 | 601.1 | 0.46x | 9.73e-08 | PASS |
| 32768 | 47 | 306.7 | 856.5 | 0.36x | 870.6 | 3792.8 | 0.23x | 1.31e-07 | PASS |
| 65536 | 1 | 139.1 | 178.6 | 0.78x | 241.8 | 543.6 | 0.45x | 8.04e-08 | PASS |
| 65536 | 3 | 216.7 | 205.0 | 1.06x | 359.2 | 620.0 | 0.58x | 8.04e-08 | PASS |
| 65536 | 47 | 693.5 | 1587.0 | 0.44x | 1770.4 | 5615.5 | 0.32x | 2.27e-07 | PASS |

数据源：`results/evidence/long-fft-baseline/baseline.json`（基线，git `b186c34b`，torch 2.10.0+cpu / torch_npu 2.10.0，CANN 9.0.0）与 `results/evidence/long-fft-device-boundary/acceptance.json`（自研，git `c79a6ebb`，binary `a2186949a9fd`）。
