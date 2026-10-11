# 长 FFT 验收证据（宿主/设备段边界对照）

本页由 `python3 scripts/summarize_long_fft_evidence.py` 从两份`acceptance.json` 原始样本生成，`--check` 可复核漂移；请勿手改。

- 门槛 `maxRel ≤ 0.0001`，全网格最坏 `2.98e-07`（A/B/A 序列与整体输出口径的并集）。
- 协议：每个形状 5 次独立 trial（每次进程内 5 reps），原始样本全部归档于各 `acceptance.json` 的 `trials.raw`；下表为 trial 统计。
- 传输契约：宿主链每次执行 `in=1 out=1 boundary=2`，设备链`in=1 out=1 boundary=0`（段边界不回宿主）。

| N | B | host e2e median (us) | device e2e median (us) | median speedup | worst maxRel (host) | worst maxRel (device) |
|---|---|---|---|---|---|---|
| 8192 | 1 | 399.2 | 348.1 | 1.147x | 1.50e-07 | 1.50e-07 |
| 8192 | 3 | 428.4 | 360.6 | 1.188x | 1.68e-07 | 1.68e-07 |
| 8192 | 47 | 6231.2 | 2304.2 | 2.704x | 2.38e-07 | 2.38e-07 |
| 16384 | 1 | 506.4 | 360.8 | 1.404x | 2.27e-07 | 2.27e-07 |
| 16384 | 3 | 859.3 | 615.4 | 1.396x | 2.27e-07 | 2.27e-07 |
| 16384 | 47 | 12943.6 | 3087.7 | 4.192x | 2.98e-07 | 2.98e-07 |
| 32768 | 1 | 644.3 | 517.4 | 1.245x | 9.92e-08 | 1.49e-07 |
| 32768 | 3 | 1480.1 | 686.6 | 2.156x | 2.04e-07 | 2.04e-07 |
| 32768 | 47 | 24181.1 | 7972.7 | 3.033x | 2.98e-07 | 2.98e-07 |
| 65536 | 1 | 1195.0 | 639.0 | 1.87x | 1.27e-07 | 1.27e-07 |
| 65536 | 3 | 1952.0 | 730.8 | 2.671x | 1.41e-07 | 1.93e-07 |
| 65536 | 47 | 45223.4 | 13674.9 | 3.307x | 2.98e-07 | 2.98e-07 |

数据源：`results/evidence/long-fft-acceptance/`（host）与 `results/evidence/long-fft-device-boundary/`（device），汇总见 `results/evidence/long-fft-summary.json`。
