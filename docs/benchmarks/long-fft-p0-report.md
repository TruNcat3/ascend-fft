# P0 报告：长 FFT device 链六段归因与同尺寸 torch_npu 基线

> PR #2 性能评论「阶段 2 · P0」的执行报告（第 1 轮）。本页由 `python3 scripts/gen_p0_report.py` 从仓库归档直读生成，复算入口见文末「复现」；不引用任何未归档读数。

## 1. 范围与方法

| 项 | 口径 |
|---|---|
| 六段事件计时 | `fft_check.cpp` 在 device 边界内核发射间插入事件对，`segments:` 行与产生 `device_chain` 最小值的同一次 launch 绑定，`sum(segments) == device_chain`（telescope 恒等）；host/短路径 `segments: NA`。段数随 impl 分支：separate 六段（twiddle 与段边界转置两次发射）、fused 五段（twiddle 并入段边界转置），device 链另打 `boundary_impl:` 自报。`tests/test_scopes.py` + `tests/test_collect_evidence.py` 锁定 |
| 自研协议 | 每形状 5 独立 trial × 5 reps；host `boundary=2`、device `boundary=0`；同一 binary `sha256=90f82de0ff8e` |
| 基线协议 | `torch.fft.fft`（torch_npu 2.10.0 op-plugin，复→复）同网格，device-only 与 E2E 各 20 reps 取 min；E2E 为 pinned 口径 |
| 逐 kernel 校验 | msprof `--task-time`（`profile_test.sh --only lfft8k1,lfft16k47,lfft32k47,lfft65k47`），提取值归档于 `results/evidence/long-fft-p0/msprof-task-time.json`，原始 profile 在 `results/profiles/20261009T041621Z/`（本地，不入库） |

归档绑定（全部清洁提交）：device `git=f537a9ec`、host `git=54bfd868`、baseline `git=b186c34b`、fused `git=be60293b`。

## 2. 六段归因（separate impl，全网格，5-trial 中位数，µs）

| N | B | transpose-in | FFT1 | twiddle | transpose-boundary | FFT2 | transpose-out | device_chain | chain CV% | E2E | E2E CV% | h2d | d2h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 46.5 | 10.2 | 6.3 | 45.1 | 7.5 | 44.3 | 159.5 | 7.86 | 329.0 | 9.39 | 58.0 | 55.1 |
| 8192 | 3 | 46.5 | 9.3 | 11.8 | 44.7 | 8.4 | 44.8 | 166.3 | 3.22 | 347.6 | 5.30 | 56.1 | 61.2 |
| 8192 | 47 | 52.8 | 74.0 | 109.5 | 48.6 | 49.2 | 50.6 | 385.1 | 1.69 | 2526.4 | 3.31 | 952.8 | 944.9 |
| 16384 | 1 | 45.9 | 7.9 | 7.3 | 45.0 | 9.7 | 44.4 | 160.2 | 3.81 | 353.2 | 13.04 | 54.9 | 56.0 |
| 16384 | 3 | 59.9 | 9.4 | 12.7 | 45.2 | 11.0 | 45.0 | 183.5 | 2.99 | 531.0 | 13.58 | 86.6 | 92.1 |
| 16384 | 47 | 54.6 | 89.4 | 118.1 | 53.0 | 91.3 | 51.9 | 458.3 | 0.10 | 4378.6 | 9.74 | 1657.7 | 1777.0 |
| 32768 | 1 | 59.3 | 9.3 | 10.3 | 45.0 | 11.1 | 45.1 | 179.0 | 3.00 | 447.9 | 4.57 | 84.6 | 101.3 |
| 32768 | 3 | 53.9 | 14.5 | 19.2 | 45.1 | 14.1 | 45.4 | 191.2 | 3.06 | 490.3 | 13.95 | 106.7 | 109.6 |
| 32768 | 47 | 87.1 | 178.8 | 258.4 | 95.9 | 131.8 | 98.1 | 850.9 | 0.12 | 8389.6 | 5.34 | 3335.3 | 3347.4 |
| 65536 | 1 | 47.0 | 11.4 | 11.0 | 45.4 | 13.6 | 45.4 | 175.0 | 1.61 | 424.1 | 14.21 | 81.6 | 88.0 |
| 65536 | 3 | 47.1 | 19.1 | 23.2 | 47.0 | 21.4 | 45.4 | 203.6 | 2.46 | 621.1 | 15.17 | 157.5 | 170.8 |
| 65536 | 47 | 238.5 | 264.0 | 336.6 | 239.2 | 263.8 | 250.0 | 1596.8 | 0.30 | 16206.6 | 18.28 | 6544.3 | 6703.5 |

数据源：`results/evidence/long-fft-device-boundary/acceptance.json` 的 `points[].trials.raw[].segments/scopes`（`AB_LONG_BOUNDARY_IMPL=separate`，六发射六段；fused 五段对照见 §9）。

## 3. 优先形状的段占比与解读

| 形状 | transpose-in | FFT1 | twiddle | transpose-boundary | FFT2 | transpose-out |
|---|---|---|---|---|---|---|
| 8192x1 | 29.2% | 6.4% | 3.9% | 28.3% | 4.7% | 27.8% |
| 16384x47 | 11.9% | 19.5% | 25.8% | 11.6% | 19.9% | 11.3% |
| 32768x47 | 10.2% | 21.0% | 30.4% | 11.3% | 15.5% | 11.5% |
| 65536x47 | 14.9% | 16.5% | 21.1% | 15.0% | 16.5% | 15.7% |

- **`8192×1`（0.90× 回退点）**：三次转置合计占 chain 的 85.3%，FFT 仅 11.1%、twiddle 3.9% —— 六次发射 + 三次全张量转置的固定成本无法由 batch=1 摊销，与评论「固定成本摊不销」的判断一致；优先级应放在减少转置成本而非 FFT 核心。
- **batch=47 形状：独立 twiddle 是最大单段**（16384×47 25.8%、32768×47 30.4%、65536×47 21.1%），超过任一段 FFT —— 它对中间态做一次完整 GM 读 + 写外加两次 Gather 重排，是 P1 融合（twiddle+transpose-boundary）的首选目标。
- `65536×47` 六段均衡（各 14.9%–21.1%），与带宽受限假设一致（待 MTE/GM/Vector 计数器验证，task-time/event 只能定位昂贵段、不能证明瓶颈类型）；融合后可省去中间态一整次 GM 写+读。

## 4. msprof task-time 交叉验证（µs/launch，4 优先形状）

| 形状 | kfft_lt_tr（3 次合计） | 事件三转置合计 | kfft_fwd（2 次合计） | 事件两 FFT 合计 | kfft_lt_tw | 事件 twiddle |
|---|---|---|---|---|---|---|
| 8192x1 | 128.7 | 135.9 | 16.0 | 17.7 | 5.8 | 6.3 |
| 16384x47 | 155.4 | 159.5 | 178.5 | 180.7 | 116.4 | 118.1 |
| 32768x47 | 280.2 | 281.1 | 310.2 | 310.6 | 259.8 | 258.4 |
| 65536x47 | 719.7 | 727.7 | 527.9 | 527.8 | 336.3 | 336.6 |

msprof 列数据源：`results/evidence/long-fft-p0/msprof-task-time.json`（`sum_prof.py` 从本地 profile 提取的均值口径）；事件列为 device 归档六段中位数之和。

两套独立测量（host 侧事件 vs msprof device task-time）在 b=47 形状逐 op 吻合 2.6% 以内；`8192×1` 的三转置项差 5.3%（profiler 汇入了 E2E 发射，事件只取 chain 最小值的单次 launch，小形状 launch 间散布大）。六段口径可以放心用于后续候选排名；profiler 继续用于解释原因（PipeUtilization 等），不替代真实 event 计时。

## 5. E2E 分解与传输口径

| 形状 | host E2E | device E2E | host/device | device h2d+d2h | 占 device E2E |
|---|---|---|---|---|---|
| 8192x1 | 295.6 | 329.0 | 0.90× | 113.1 | 34% |
| 8192x3 | 429.1 | 347.6 | 1.23× | 117.3 | 34% |
| 8192x47 | 6032.0 | 2526.4 | 2.39× | 1897.7 | 75% |
| 16384x1 | 366.7 | 353.2 | 1.04× | 110.9 | 31% |
| 16384x3 | 865.5 | 531.0 | 1.63× | 178.7 | 34% |
| 16384x47 | 12627.7 | 4378.6 | 2.88× | 3434.7 | 78% |
| 32768x1 | 655.9 | 447.9 | 1.46× | 185.9 | 42% |
| 32768x3 | 1099.9 | 490.3 | 2.24× | 216.3 | 44% |
| 32768x47 | 25423.3 | 8389.6 | 3.03× | 6682.7 | 80% |
| 65536x1 | 875.9 | 424.1 | 2.07× | 169.6 | 40% |
| 65536x3 | 2215.5 | 621.1 | 3.57× | 328.3 | 53% |
| 65536x47 | 49218.6 | 16206.6 | 3.04× | 13247.8 | 82% |

数据源：`results/evidence/long-fft-acceptance/acceptance.json`（host 链 E2E 中位数）与 `results/evidence/long-fft-device-boundary/acceptance.json`（device 链 E2E/h2d/d2h 中位数）。

- 设备段边界链在 11/12 个形状上降低 E2E（最高 3.57×），`8192×1` 例外回退（0.90×，第 3 节固定成本归因）。
- **自研长链 E2E 的 31%–82% 是 H2D+D2H**（b=47 达 75%–82%），且当前长路径从 `std::vector` 发起（`fft_check.cpp` 的 pinned 分配仅覆盖短路径），为 pageable 口径；下一节基线的 E2E 差距必须先扣除此项再谈 kernel。
- host 链的 E2E 主体是宿主标量段（四步转置/twiddle/重排的 CPU 循环），不是 kernel 慢，比较口径不同。

## 6. 波动性

- device `device_chain` CV：**0.10%–7.86%**（b=47 形状 0.10%–1.69%，b≤3 形状 1.61%–7.86%，短链单次 launch 的时钟分辨率噪声占比更高）。
- device E2E CV：**3.31%–18.28%**（`65536×47` 18.28%、`65536×3` 15.17%、`65536×1` 14.21%），显著高于同形状 chain CV —— 波动来自传输、host dispatch 与环境，不是 FFT 算术核心；评论的判断成立。
- 下一步按评论要求做交错运行 + 环境记录（温度/频率/共租户）后再定是否可接受。

## 7. 同尺寸 torch_npu 基线

完整 12 行对照见生成页 [长 FFT 同尺寸基线](../generated/long-fft-baseline.md)（`results/evidence/long-fft-baseline/baseline.json`，12/12 native PASS，torch 2.10.0 / torch_npu 2.10.0 / CANN 9.0.0，native 20 reps min）。要点：

- **device-only：`native_min / ours_median` = 0.358×–1.057×**（>1 表示自研更快：仅 `65536×3` 为 1.06×，其余 native 领先）。native 是单次融合变换不物化本实现的三转置段边界，形态不同；3.4× 的 device-vs-host 结论**不能**外推为对外部库优势。
- **E2E：同一比值 0.197×–0.579×**，b=47 最差（0.197×–0.315×）——其中含自研 pageable 传输口径 vs native 的差距（第 5 节），与 kernel 差距必须分列。
- 两口径、两协议均已归档，后续任何候选以同表复测对比。

## 8. 结论 → P1 优先级

1. **融合 twiddle + transpose-boundary**（`fft_long.cpp`，`kfft_lt_tr` 签名已带 `tw`）：目标消掉最大单段（21%–30%）+ 中间态一整次 GM 写读；验收 = 六段计时 + GM 字节 + A/B/A，六内核路径保留 incumbent（PR-B 已落地 separate/fused 双路径，对照见 §9）。
2. **转置成本**（`8192×1` 占 85.3%、b=47 合计 33%–46%）：tile/LT_H×LT_W 参数化、双缓冲（当前 3×32KB UB，`long_fft_ub.h` 模型已就位）、窄作用域 pipe barrier。
3. **长链 pinned 传输**：修正 E2E 口径，使与 native 的 E2E 对比同口径。
4. **与 native 的形态差距**：P1 融合做完后重新对照；不达标候选不进默认配置。
5. **波动治理**：交错运行 + 环境记录（评论退出条件）。

评论「下一轮退出条件」现状：① 8192×1 仍慢于 host（未满足，已有归因）；② 六段时间可解释 chain（**满足**，§2/§4）；③ 高 CV 未治理（待环境记录）；④ twiddle 融合已做（PR-B separate/fused 双路径，§9）；⑤ torch_npu 同语义矩阵（**满足**，12/12 §7）。

## 9. PR-B R1：twiddle 并入边界转置（separate vs fused 对照）

`AB_LONG_BOUNDARY_IMPL` 双路径：separate = 六次发射六段（twiddle 独立成段），fused = 五次发射五段（`kfft_lt_tr` 在段边界转置的读入 tile 内复乘 twiddle，`tw` 参数直接消费）。归档 `results/evidence/long-fft-device-boundary-fused/acceptance.json`：`boundary=device-fused`、`impl=fused`，每 trial 自报 `boundary_impl: fused` 且段数契约由 collect 逐 trial 校验（`segments` 五段 telescope 进 `device_chain`）。链与 E2E 均为 5-trial 中位数：

| 形状 | separate chain | fused chain | Δ chain | separate E2E | fused E2E | Δ E2E |
|---|---|---|---|---|---|---|
| 8192x1 | 159.5 | 160.5 | +0.6% | 329.0 | 317.7 | -3.4% |
| 8192x3 | 166.3 | 171.5 | +3.1% | 347.6 | 378.8 | +9.0% |
| 8192x47 | 385.1 | 295.9 | -23.2% | 2526.4 | 2309.9 | -8.6% |
| 16384x1 | 160.2 | 159.6 | -0.4% | 353.2 | 325.6 | -7.8% |
| 16384x3 | 183.5 | 175.5 | -4.4% | 531.0 | 499.2 | -6.0% |
| 16384x47 | 458.3 | 381.7 | -16.7% | 4378.6 | 4079.3 | -6.8% |
| 32768x1 | 179.0 | 167.5 | -6.4% | 447.9 | 372.1 | -16.9% |
| 32768x3 | 191.2 | 181.5 | -5.1% | 490.3 | 523.5 | +6.8% |
| 32768x47 | 850.9 | 657.5 | -22.7% | 8389.6 | 7462.3 | -11.1% |
| 65536x1 | 175.0 | 179.3 | +2.5% | 424.1 | 518.4 | +22.2% |
| 65536x3 | 203.6 | 192.6 | -5.4% | 621.1 | 545.5 | -12.2% |
| 65536x47 | 1596.8 | 1595.1 | -0.1% | 16206.6 | 14638.3 | -9.7% |

- fused 在 9/12 个形状上缩短 device_chain（Δ 中位数 -4.7%，最差 +3.1%）；E2E 因 H2D+D2H 占比被稀释（ΔE2E 中位数 -7.3%）。
- 本表是全网格初测（非交错序）；晋升门槛（≥4/5 配对不慢且中位数改善 ≥5%，任一点回退 >3% 则保留 separate fallback）以 4 优先点 `8192×1 / 16384×47 / 32768×47 / 65536×47` 的 separate/fused 交错 ≥5 trial 归因为准。

## 10. 复现

```bash
bash scripts/build.sh
python3 scripts/collect_long_fft_evidence.py              # host, 含 segments
python3 scripts/collect_long_fft_evidence.py --boundary device
python3 scripts/collect_long_fft_evidence.py --boundary device-fused # PR-B §9
python3 scripts/collect_long_fft_evidence.py --verify results/evidence/long-fft-acceptance/acceptance.json
python3 scripts/bench_long_baseline.py                    # torch_npu 基线
python3 scripts/bench_long_baseline.py --check            # md ↔ JSON
python3 scripts/summarize_long_fft_evidence.py --check
python3 scripts/gen_p0_report.py --check                  # 本页 ↔ 归档
scripts/profile_test.sh --only lfft8k1,lfft16k47,lfft32k47,lfft65k47
python3 scripts/sum_prof.py --csv results/profiles/<dir>/lfft*  # §4 出处
python3 -m unittest discover -s tests -q
```
