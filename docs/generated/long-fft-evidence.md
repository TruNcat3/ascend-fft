# 长 FFT 验收证据（宿主/设备段边界对照）

本页由 `python3 scripts/summarize_long_fft_evidence.py` 从两份`acceptance.json` 原始样本生成，`--check` 可复核漂移；请勿手改。

- 门槛 `maxRel ≤ 0.0001`，全网格最坏 `2.98e-07`（A/B/A 序列与整体输出口径的并集）。
- 协议：每个形状 5 次独立 trial（每次进程内 5 reps），原始样本全部归档于各 `acceptance.json` 的 `trials.raw`；下表为 trial 统计。
- 传输契约：宿主链每次执行 `in=1 out=1 boundary=2`，设备链`in=1 out=1 boundary=0`（段边界不回宿主）。

| N | B | host e2e median (us) | device e2e median (us) | median speedup | worst maxRel (host) | worst maxRel (device) |
|---|---|---|---|---|---|---|
| 8192 | 1 | 295.6 | 329.0 | 0.898x | 1.50e-07 | 1.50e-07 |
| 8192 | 3 | 429.1 | 347.6 | 1.234x | 1.68e-07 | 1.68e-07 |
| 8192 | 47 | 6032.0 | 2526.4 | 2.388x | 2.38e-07 | 2.38e-07 |
| 16384 | 1 | 366.7 | 353.2 | 1.038x | 2.27e-07 | 2.27e-07 |
| 16384 | 3 | 865.5 | 531.0 | 1.63x | 2.27e-07 | 2.27e-07 |
| 16384 | 47 | 12627.7 | 4378.6 | 2.884x | 2.98e-07 | 2.98e-07 |
| 32768 | 1 | 655.9 | 447.9 | 1.464x | 9.92e-08 | 1.49e-07 |
| 32768 | 3 | 1099.9 | 490.3 | 2.243x | 2.04e-07 | 2.04e-07 |
| 32768 | 47 | 25423.3 | 8389.6 | 3.03x | 2.98e-07 | 2.98e-07 |
| 65536 | 1 | 875.9 | 424.1 | 2.065x | 1.27e-07 | 1.27e-07 |
| 65536 | 3 | 2215.5 | 621.1 | 3.567x | 1.41e-07 | 1.93e-07 |
| 65536 | 47 | 49218.6 | 16206.6 | 3.037x | 2.98e-07 | 2.98e-07 |

数据源：`results/evidence/long-fft-acceptance/`（host）与 `results/evidence/long-fft-device-boundary/`（device），汇总见 `results/evidence/long-fft-summary.json`。
