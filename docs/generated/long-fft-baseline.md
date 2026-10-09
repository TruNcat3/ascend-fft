# 长 FFT 同尺寸基线：torch_npu vs 自研 device 边界链

本页由 `python3 scripts/bench_long_baseline.py --check` 从归档 `results/evidence/long-fft-baseline/baseline.json` 渲染校验；请勿手改。

- 基线：`torch.fft.fft`（torch_npu op-plugin，复->复）同网格同语义，device-only 取 `bench_native_npu.py` 的 min，E2E 取其 pinned 口径 min。
- 自研：device 边界链 5-trial 归档的 `device_chain` / E2E 中位数（`segments:` 六段之和即 chain）。自研长链 E2E 当前从 `std::vector` 发起（`fft_check.cpp` 的 pinned 分配仅覆盖短路径），传输为 pageable 口径；E2E 列的差距包含此项，与 device-only 列要分开解读。
- `speedup = native_min / ours_median`，>1 表示自研更快；误差门槛 `maxRel ≤ 0.0001`。

| N | B | native dev min (us) | ours chain median (us) | dev speedup | native E2E min (us) | ours E2E median (us) | E2E speedup | ours maxRel | status |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 93.2 | 163.7 | 0.57x | 158.9 | 315.4 | 0.50x | 5.12e-08 | PASS |
| 8192 | 3 | 117.0 | 160.8 | 0.73x | 185.6 | 347.1 | 0.54x | 1.16e-07 | PASS |
| 8192 | 47 | 153.2 | 378.9 | 0.40x | 350.3 | 2377.7 | 0.15x | 1.99e-07 | PASS |
| 16384 | 1 | 105.5 | 154.3 | 0.68x | 173.1 | 352.3 | 0.49x | 2.27e-07 | PASS |
| 16384 | 3 | 154.7 | 172.9 | 0.90x | 244.8 | 443.2 | 0.55x | 2.27e-07 | PASS |
| 16384 | 47 | 206.1 | 453.6 | 0.45x | 522.0 | 4435.6 | 0.12x | 2.27e-07 | PASS |
| 32768 | 1 | 113.7 | 162.3 | 0.70x | 202.1 | 404.3 | 0.50x | 3.65e-08 | PASS |
| 32768 | 3 | 161.4 | 179.7 | 0.90x | 264.8 | 446.0 | 0.59x | 9.73e-08 | PASS |
| 32768 | 47 | 294.4 | 853.8 | 0.34x | 861.7 | 7943.9 | 0.11x | 1.31e-07 | PASS |
| 65536 | 1 | 147.2 | 173.2 | 0.85x | 235.7 | 450.6 | 0.52x | 8.04e-08 | PASS |
| 65536 | 3 | 210.6 | 204.7 | 1.03x | 351.7 | 739.6 | 0.48x | 8.04e-08 | PASS |
| 65536 | 47 | 672.6 | 1586.9 | 0.42x | 1733.4 | 15443.8 | 0.11x | 2.27e-07 | PASS |

数据源：`results/evidence/long-fft-baseline/baseline.json`（基线，git `f7d6e890`，torch 2.10.0+cpu / torch_npu 2.10.0，CANN 9.0.0）与 `results/evidence/long-fft-device-boundary/acceptance.json`（自研，git `0f9170b9`，binary `f987fe341a10`）。
