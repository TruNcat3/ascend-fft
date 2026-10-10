# 长 FFT 同尺寸基线：torch_npu vs 自研 device 边界链

本页由 `python3 scripts/bench_long_baseline.py --check` 从归档 `results/evidence/long-fft-baseline/baseline.json` 渲染校验；请勿手改。

- 基线：`torch.fft.fft`（torch_npu op-plugin，复->复）同网格同语义，device-only 取 `bench_native_npu.py` 的 min，E2E 取其 pinned 口径 min。
- 自研：device 边界链 5-trial 归档的 `device_chain` / E2E 中位数（`segments:` 六段之和即 chain）。自研长链 E2E 当前从 `std::vector` 发起（`fft_check.cpp` 的 pinned 分配仅覆盖短路径），传输为 pageable 口径；E2E 列的差距包含此项，与 device-only 列要分开解读。
- `speedup = native_min / ours_median`，>1 表示自研更快；误差门槛 `maxRel ≤ 0.0001`。

| N | B | native dev min (us) | ours chain median (us) | dev speedup | native E2E min (us) | ours E2E median (us) | E2E speedup | ours maxRel | status |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 98.6 | 159.5 | 0.62x | 163.2 | 329.0 | 0.50x | 5.12e-08 | PASS |
| 8192 | 3 | 125.9 | 166.3 | 0.76x | 197.7 | 347.6 | 0.57x | 1.16e-07 | PASS |
| 8192 | 47 | 155.3 | 385.1 | 0.40x | 361.1 | 2526.4 | 0.14x | 1.99e-07 | PASS |
| 16384 | 1 | 99.8 | 160.2 | 0.62x | 168.3 | 353.2 | 0.48x | 2.27e-07 | PASS |
| 16384 | 3 | 168.8 | 183.5 | 0.92x | 260.0 | 531.0 | 0.49x | 2.27e-07 | PASS |
| 16384 | 47 | 220.8 | 458.3 | 0.48x | 652.1 | 4378.6 | 0.15x | 2.27e-07 | PASS |
| 32768 | 1 | 116.0 | 179.0 | 0.65x | 214.4 | 447.9 | 0.48x | 3.65e-08 | PASS |
| 32768 | 3 | 168.4 | 191.2 | 0.88x | 283.1 | 490.3 | 0.58x | 9.73e-08 | PASS |
| 32768 | 47 | 309.6 | 850.9 | 0.36x | 886.7 | 8389.6 | 0.11x | 1.31e-07 | PASS |
| 65536 | 1 | 158.4 | 175.0 | 0.91x | 260.8 | 424.1 | 0.61x | 8.04e-08 | PASS |
| 65536 | 3 | 222.7 | 203.6 | 1.09x | 368.6 | 621.1 | 0.59x | 8.04e-08 | PASS |
| 65536 | 47 | 693.6 | 1596.8 | 0.43x | 1757.0 | 16206.6 | 0.11x | 2.27e-07 | PASS |

数据源：`results/evidence/long-fft-baseline/baseline.json`（基线，git `f56b960b`，torch 2.10.0+cpu / torch_npu 2.10.0，CANN 9.0.0）与 `results/evidence/long-fft-device-boundary/acceptance.json`（自研，git `f537a9ec`，binary `90f82de0ff8e`）。
