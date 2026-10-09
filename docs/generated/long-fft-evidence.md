# 长 FFT 验收证据（宿主/设备段边界对照）

本页由 `python3 scripts/summarize_long_fft_evidence.py` 从两份`acceptance.json` 原始样本生成，`--check` 可复核漂移；请勿手改。

- 门槛 `maxRel ≤ 0.0001`，全网格最坏 `2.98e-07`（A/B/A 序列与整体输出口径的并集）。
- 协议：每个形状 5 次独立 trial（每次进程内 5 reps），原始样本全部归档于各 `acceptance.json` 的 `trials.raw`；下表为 trial 统计。
- 传输契约：宿主链每次执行 `in=1 out=1 boundary=2`，设备链`in=1 out=1 boundary=0`（段边界不回宿主）。

| N | B | host e2e median (us) | device e2e median (us) | median speedup | worst maxRel (host) | worst maxRel (device) |
|---|---|---|---|---|---|---|
| 8192 | 1 | 302.8 | 326.6 | 0.927x | 1.02e-07 | 5.12e-08 |
| 8192 | 3 | 407.6 | 354.7 | 1.149x | 1.16e-07 | 1.16e-07 |
| 8192 | 47 | 4200.2 | 2279.9 | 1.842x | 1.99e-07 | 1.99e-07 |
| 16384 | 1 | 421.3 | 336.8 | 1.251x | 2.27e-07 | 2.27e-07 |
| 16384 | 3 | 917.0 | 539.8 | 1.699x | 2.27e-07 | 2.27e-07 |
| 16384 | 47 | 9509.8 | 4213.8 | 2.257x | 2.27e-07 | 2.27e-07 |
| 32768 | 1 | 735.3 | 481.7 | 1.526x | 4.14e-08 | 3.65e-08 |
| 32768 | 3 | 1343.9 | 658.0 | 2.042x | 9.73e-08 | 9.73e-08 |
| 32768 | 47 | 17752.9 | 5798.8 | 3.061x | 1.31e-07 | 1.31e-07 |
| 65536 | 1 | 1020.9 | 529.9 | 1.927x | 8.04e-08 | 8.04e-08 |
| 65536 | 3 | 1969.2 | 600.5 | 3.279x | 1.41e-07 | 8.04e-08 |
| 65536 | 47 | 44173.5 | 12933.5 | 3.415x | 2.27e-07 | 2.27e-07 |

数据源：`results/evidence/long-fft-acceptance/`（host）与 `results/evidence/long-fft-device-boundary/`（device），汇总见 `results/evidence/long-fft-summary.json`。
