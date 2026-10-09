# 长 FFT 验收证据（宿主/设备段边界对照）

本页由 `python3 scripts/summarize_long_fft_evidence.py` 从两份`acceptance.json` 原始样本生成，`--check` 可复核漂移；请勿手改。

- 门槛 `maxRel ≤ 0.0001`，全网格最坏 `2.98e-07`（A/B/A 序列与整体输出口径的并集）。
- 协议：每个形状 5 次独立 trial（每次进程内 5 reps），原始样本全部归档于各 `acceptance.json` 的 `trials.raw`；下表为 trial 统计。
- 传输契约：宿主链每次执行 `in=1 out=1 boundary=2`，设备链`in=1 out=1 boundary=0`（段边界不回宿主）。

| N | B | host e2e median (us) | device e2e median (us) | median speedup | worst maxRel (host) | worst maxRel (device) |
|---|---|---|---|---|---|---|
| 8192 | 1 | 290.6 | 372.7 | 0.78x | 1.50e-07 | 1.50e-07 |
| 8192 | 3 | 407.9 | 402.8 | 1.013x | 1.68e-07 | 1.68e-07 |
| 8192 | 47 | 3225.0 | 1474.2 | 2.188x | 2.38e-07 | 2.38e-07 |
| 16384 | 1 | 395.9 | 339.5 | 1.166x | 2.27e-07 | 2.27e-07 |
| 16384 | 3 | 705.8 | 433.7 | 1.627x | 2.27e-07 | 2.27e-07 |
| 16384 | 47 | 13111.4 | 2628.4 | 4.988x | 2.98e-07 | 2.98e-07 |
| 32768 | 1 | 707.2 | 442.5 | 1.598x | 9.92e-08 | 1.49e-07 |
| 32768 | 3 | 1086.0 | 601.1 | 1.807x | 2.04e-07 | 2.04e-07 |
| 32768 | 47 | 24096.1 | 3792.8 | 6.353x | 2.98e-07 | 2.98e-07 |
| 65536 | 1 | 1012.0 | 543.6 | 1.862x | 1.27e-07 | 1.27e-07 |
| 65536 | 3 | 2196.2 | 620.0 | 3.542x | 1.41e-07 | 1.93e-07 |
| 65536 | 47 | 45205.4 | 5615.5 | 8.05x | 2.98e-07 | 2.98e-07 |

数据源：`results/evidence/long-fft-acceptance/`（host）与 `results/evidence/long-fft-device-boundary/`（device），汇总见 `results/evidence/long-fft-summary.json`。
