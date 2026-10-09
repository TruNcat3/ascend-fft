# 长 FFT 验收证据（宿主/设备段边界对照）

本页由 `python3 scripts/summarize_long_fft_evidence.py` 从两份`acceptance.json` 原始样本生成，`--check` 可复核漂移；请勿手改。

- 门槛 `maxRel ≤ 0.0001`，全网格最坏 `2.98e-07`（A/B/A 序列与整体输出口径的并集）。
- 协议：每个形状 5 次独立 trial（每次进程内 5 reps），原始样本全部归档于各 `acceptance.json` 的 `trials.raw`；下表为 trial 统计。
- 传输契约：宿主链每次执行 `in=1 out=1 boundary=2`，设备链`in=1 out=1 boundary=0`（段边界不回宿主）。

| N | B | host e2e median (us) | device e2e median (us) | median speedup | worst maxRel (host) | worst maxRel (device) |
|---|---|---|---|---|---|---|
| 8192 | 1 | 293.6 | 315.4 | 0.931x | 1.02e-07 | 5.12e-08 |
| 8192 | 3 | 447.0 | 347.1 | 1.288x | 1.16e-07 | 1.16e-07 |
| 8192 | 47 | 6012.8 | 2377.7 | 2.529x | 1.99e-07 | 1.99e-07 |
| 16384 | 1 | 353.2 | 352.3 | 1.003x | 2.27e-07 | 2.27e-07 |
| 16384 | 3 | 814.5 | 443.2 | 1.838x | 2.27e-07 | 2.27e-07 |
| 16384 | 47 | 13174.6 | 4435.6 | 2.97x | 2.27e-07 | 2.27e-07 |
| 32768 | 1 | 669.8 | 404.3 | 1.657x | 4.14e-08 | 3.65e-08 |
| 32768 | 3 | 1351.9 | 446.0 | 3.031x | 9.73e-08 | 9.73e-08 |
| 32768 | 47 | 24562.1 | 7943.9 | 3.092x | 1.31e-07 | 1.31e-07 |
| 65536 | 1 | 815.3 | 450.6 | 1.809x | 8.04e-08 | 8.04e-08 |
| 65536 | 3 | 2181.5 | 739.6 | 2.95x | 1.41e-07 | 8.04e-08 |
| 65536 | 47 | 44407.6 | 15443.8 | 2.875x | 2.27e-07 | 2.27e-07 |

数据源：`results/evidence/long-fft-acceptance/`（host）与 `results/evidence/long-fft-device-boundary/`（device），汇总见 `results/evidence/long-fft-summary.json`。
