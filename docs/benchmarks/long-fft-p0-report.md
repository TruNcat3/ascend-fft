# P0 报告：长 FFT device 链六段归因与同尺寸 torch_npu 基线

> PR #2 性能评论「阶段 2 · P0」的执行报告（第 1 轮）。本页由 `python3 scripts/gen_p0_report.py` 从仓库归档直读生成，复算入口见文末「复现」；不引用任何未归档读数。

## 1. 范围与方法

| 项 | 口径 |
|---|---|
| 六段事件计时 | `fft_check.cpp` 在 device 边界六次内核发射间插入事件对，`segments:` 行与产生 `device_chain` 最小值的同一次 launch 绑定，`sum(segments) == device_chain`（telescope 恒等）；host/短路径 `segments: NA`。`tests/test_scopes.py` + `tests/test_collect_evidence.py` 锁定 |
| 自研协议 | 每形状 5 独立 trial × 5 reps；host `boundary=2`、device `boundary=0`；同一 binary `sha256=a2186949a9fd` |
| 基线协议 | `torch.fft.fft`（torch_npu 2.10.0 op-plugin，复→复）同网格，device-only 与 E2E 各 20 reps 取 min；E2E 为 pinned 口径 |
| 逐 kernel 校验 | msprof `--task-time`（`profile_test.sh --only lfft8k1,lfft16k47,lfft32k47,lfft65k47`），提取值归档于 `results/evidence/long-fft-p0/msprof-task-time.json`，原始 profile 在 `results/profiles/20261009T041621Z/`（本地，不入库） |

归档绑定（全部清洁提交）：device `git=c79a6ebb`、host `git=519e211d`、baseline `git=b186c34b`。

## 2. 六段归因（全网格，5-trial 中位数，µs）

| N | B | transpose-in | FFT1 | twiddle | transpose-boundary | FFT2 | transpose-out | device_chain | chain CV% | E2E | E2E CV% | h2d | d2h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 53.9 | 9.0 | 6.4 | 44.2 | 7.6 | 44.0 | 167.8 | 4.84 | 372.7 | 5.97 | 65.5 | 63.0 |
| 8192 | 3 | 55.0 | 10.3 | 10.4 | 44.4 | 6.7 | 44.4 | 175.8 | 3.86 | 402.8 | 5.52 | 73.5 | 85.2 |
| 8192 | 47 | 52.7 | 74.8 | 108.4 | 46.9 | 46.8 | 50.1 | 379.9 | 1.33 | 1474.2 | 29.05 | 425.0 | 452.1 |
| 16384 | 1 | 46.0 | 8.1 | 7.2 | 43.9 | 7.9 | 44.1 | 157.2 | 5.74 | 339.5 | 8.64 | 64.0 | 62.1 |
| 16384 | 3 | 47.4 | 9.2 | 11.5 | 44.8 | 9.4 | 44.8 | 167.3 | 3.90 | 433.7 | 16.35 | 90.1 | 93.3 |
| 16384 | 47 | 54.6 | 89.5 | 117.3 | 51.3 | 89.6 | 51.5 | 453.7 | 1.59 | 2628.4 | 28.88 | 987.1 | 903.6 |
| 32768 | 1 | 58.3 | 9.4 | 9.3 | 45.1 | 10.0 | 44.9 | 179.9 | 3.78 | 442.5 | 5.32 | 83.6 | 101.5 |
| 32768 | 3 | 54.5 | 14.7 | 18.6 | 44.8 | 11.6 | 45.1 | 190.0 | 5.85 | 601.1 | 13.78 | 130.4 | 167.5 |
| 32768 | 47 | 94.8 | 179.1 | 258.5 | 94.9 | 129.9 | 97.9 | 856.5 | 0.63 | 3792.8 | 42.28 | 1340.7 | 1142.2 |
| 65536 | 1 | 54.2 | 11.6 | 10.8 | 45.1 | 11.6 | 45.1 | 178.6 | 3.51 | 543.6 | 11.25 | 109.6 | 132.8 |
| 65536 | 3 | 55.6 | 19.2 | 21.5 | 45.2 | 19.2 | 45.0 | 205.0 | 2.29 | 620.0 | 19.75 | 161.7 | 173.6 |
| 65536 | 47 | 234.8 | 264.5 | 335.6 | 237.2 | 264.9 | 247.5 | 1587.0 | 0.22 | 5615.5 | 5.81 | 1797.2 | 1793.9 |

数据源：`results/evidence/long-fft-device-boundary/acceptance.json` 的 `points[].trials.raw[].segments/scopes`。

## 3. 优先形状的段占比与解读

| 形状 | transpose-in | FFT1 | twiddle | transpose-boundary | FFT2 | transpose-out |
|---|---|---|---|---|---|---|
| 8192x1 | 32.1% | 5.4% | 3.8% | 26.3% | 4.5% | 26.2% |
| 16384x47 | 12.0% | 19.7% | 25.9% | 11.3% | 19.7% | 11.4% |
| 32768x47 | 11.1% | 20.9% | 30.2% | 11.1% | 15.2% | 11.4% |
| 65536x47 | 14.8% | 16.7% | 21.1% | 14.9% | 16.7% | 15.6% |

- **`8192×1`（0.78× 回退点）**：三次转置合计占 chain 的 84.6%，FFT 仅 9.9%、twiddle 3.8% —— 六次发射 + 三次全张量转置的固定成本无法由 batch=1 摊销，与评论「固定成本摊不销」的判断一致；优先级应放在减少转置成本而非 FFT 核心。
- **batch=47 形状：独立 twiddle 是最大单段**（16384×47 25.9%、32768×47 30.2%、65536×47 21.1%），超过任一段 FFT —— 它对中间态做一次完整 GM 读 + 写外加两次 Gather 重排，是 P1 融合（twiddle+transpose-boundary）的首选目标。
- `65536×47` 六段均衡（各 14.8%–21.1%），与带宽受限假设一致（待 MTE/GM/Vector 计数器验证，task-time/event 只能定位昂贵段、不能证明瓶颈类型）；融合后可省去中间态一整次 GM 写+读。

## 4. msprof task-time 交叉验证（µs/launch，4 优先形状）

| 形状 | kfft_lt_tr（3 次合计） | 事件三转置合计 | kfft_fwd（2 次合计） | 事件两 FFT 合计 | kfft_lt_tw | 事件 twiddle |
|---|---|---|---|---|---|---|
| 8192x1 | 128.7 | 142.1 | 16.0 | 16.6 | 5.8 | 6.4 |
| 16384x47 | 155.4 | 157.4 | 178.5 | 179.1 | 116.4 | 117.3 |
| 32768x47 | 280.2 | 287.6 | 310.2 | 309.0 | 259.8 | 258.5 |
| 65536x47 | 719.7 | 719.5 | 527.9 | 529.4 | 336.3 | 335.6 |

msprof 列数据源：`results/evidence/long-fft-p0/msprof-task-time.json`（`sum_prof.py` 从本地 profile 提取的均值口径）；事件列为 device 归档六段中位数之和。

两套独立测量（host 侧事件 vs msprof device task-time）在 b=47 形状逐 op 吻合 2.6% 以内；`8192×1` 的三转置项差 9.4%（profiler 汇入了 E2E 发射，事件只取 chain 最小值的单次 launch，小形状 launch 间散布大）。六段口径可以放心用于后续候选排名；profiler 继续用于解释原因（PipeUtilization 等），不替代真实 event 计时。

## 5. E2E 分解与传输口径

| 形状 | host E2E | device E2E | host/device | device h2d+d2h | 占 device E2E |
|---|---|---|---|---|---|
| 8192x1 | 290.6 | 372.7 | 0.78× | 128.5 | 34% |
| 8192x3 | 407.9 | 402.8 | 1.01× | 158.7 | 39% |
| 8192x47 | 3225.0 | 1474.2 | 2.19× | 877.1 | 59% |
| 16384x1 | 395.9 | 339.5 | 1.17× | 126.1 | 37% |
| 16384x3 | 705.8 | 433.7 | 1.63× | 183.4 | 42% |
| 16384x47 | 13111.4 | 2628.4 | 4.99× | 1890.7 | 72% |
| 32768x1 | 707.2 | 442.5 | 1.60× | 185.1 | 42% |
| 32768x3 | 1086.0 | 601.1 | 1.81× | 297.9 | 50% |
| 32768x47 | 24096.1 | 3792.8 | 6.35× | 2482.9 | 65% |
| 65536x1 | 1012.0 | 543.6 | 1.86× | 242.4 | 45% |
| 65536x3 | 2196.2 | 620.0 | 3.54× | 335.3 | 54% |
| 65536x47 | 45205.4 | 5615.5 | 8.05× | 3591.1 | 64% |

数据源：`results/evidence/long-fft-acceptance/acceptance.json`（host 链 E2E 中位数）与 `results/evidence/long-fft-device-boundary/acceptance.json`（device 链 E2E/h2d/d2h 中位数）。

- 设备段边界链在 11/12 个形状上降低 E2E（最高 8.05×），`8192×1` 例外回退（0.78×，第 3 节固定成本归因）。
- **自研长链 E2E 的 34%–72% 是 H2D+D2H**（b=47 达 59%–72%），且当前长路径从 `std::vector` 发起（`fft_check.cpp` 的 pinned 分配仅覆盖短路径），为 pageable 口径；下一节基线的 E2E 差距必须先扣除此项再谈 kernel。
- host 链的 E2E 主体是宿主标量段（四步转置/twiddle/重排的 CPU 循环），不是 kernel 慢，比较口径不同。

## 6. 波动性

- device `device_chain` CV：**0.22%–5.85%**（b=47 形状 0.22%–1.59%，b≤3 形状 2.29%–5.85%，短链单次 launch 的时钟分辨率噪声占比更高）。
- device E2E CV：**5.32%–42.28%**（`32768×47` 42.28%、`8192×47` 29.05%、`16384×47` 28.88%），显著高于同形状 chain CV —— 波动来自传输、host dispatch 与环境，不是 FFT 算术核心；评论的判断成立。
- 下一步按评论要求做交错运行 + 环境记录（温度/频率/共租户）后再定是否可接受。

## 7. 同尺寸 torch_npu 基线

完整 12 行对照见生成页 [长 FFT 同尺寸基线](../generated/long-fft-baseline.md)（`results/evidence/long-fft-baseline/baseline.json`，12/12 native PASS，torch 2.10.0 / torch_npu 2.10.0 / CANN 9.0.0，native 20 reps min）。要点：

- **device-only：`native_min / ours_median` = 0.358×–1.057×**（>1 表示自研更快：仅 `65536×3` 为 1.06×，其余 native 领先）。native 是单次融合变换不物化本实现的三转置段边界，形态不同；3.4× 的 device-vs-host 结论**不能**外推为对外部库优势。
- **E2E：同一比值 0.197×–0.579×**，b=47 最差（0.197×–0.315×）——其中含自研 pageable 传输口径 vs native 的差距（第 5 节），与 kernel 差距必须分列。
- 两口径、两协议均已归档，后续任何候选以同表复测对比。

## 8. 结论 → P1 优先级

1. **融合 twiddle + transpose-boundary**（`fft_long.cpp`，`kfft_lt_tr` 签名已带 `tw`）：目标消掉最大单段（21%–30%）+ 中间态一整次 GM 写读；验收 = 六段计时 + GM 字节 + A/B/A，六内核路径保留 incumbent。
2. **转置成本**（`8192×1` 占 84.6%、b=47 合计 34%–45%）：tile/LT_H×LT_W 参数化、双缓冲（当前 3×32KB UB，`long_fft_ub.h` 模型已就位）、窄作用域 pipe barrier。
3. **长链 pinned 传输**：修正 E2E 口径，使与 native 的 E2E 对比同口径。
4. **与 native 的形态差距**：P1 融合做完后重新对照；不达标候选不进默认配置。
5. **波动治理**：交错运行 + 环境记录（评论退出条件）。

评论「下一轮退出条件」现状：① 8192×1 仍慢于 host（未满足，已有归因）；② 六段时间可解释 chain（**满足**，§2/§4）；③ 高 CV 未治理（待环境记录）；④ twiddle 融合未做（P1 首项）；⑤ torch_npu 同语义矩阵（**满足**，12/12 §7）。

## 9. 复现

```bash
bash scripts/build.sh
python3 scripts/collect_long_fft_evidence.py              # host, 含 segments
python3 scripts/collect_long_fft_evidence.py --boundary device
python3 scripts/collect_long_fft_evidence.py --verify results/evidence/long-fft-acceptance/acceptance.json
python3 scripts/bench_long_baseline.py                    # torch_npu 基线
python3 scripts/bench_long_baseline.py --check            # md ↔ JSON
python3 scripts/summarize_long_fft_evidence.py --check
python3 scripts/gen_p0_report.py --check                  # 本页 ↔ 归档
scripts/profile_test.sh --only lfft8k1,lfft16k47,lfft32k47,lfft65k47
python3 scripts/sum_prof.py --csv results/profiles/<dir>/lfft*  # §4 出处
python3 -m unittest discover -s tests -q
```
