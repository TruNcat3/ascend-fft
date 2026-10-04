#!/usr/bin/env bash
# 实验 ↔ 文档 ↔ 脚本 的注册表与运行器（全仓唯一的「怎么复现」入口）。
#
#   scripts/repro.sh --list            # 列出全部实验：名字 | 文档 | 命令
#   scripts/repro.sh <name> [参数...]  # 跑某个实验
#   scripts/repro.sh --doc <文件>      # 列出某份文档涉及的实验
#   scripts/repro.sh all               # 串行跑全部实验（很慢，逐个挑更实际）
#
# 退出码：0 = 全部成功；1 = 有失败；2 = 用法错误。
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

reg() { # reg <名字> <文档> <说明> <命令...>
  NAMES+=("$1"); DOCS+=("$2"); DESC+=("$3"); shift 3
  CMDS+=("$*")
}
NAMES=(); DOCS=(); DESC=(); CMDS=()

# ---- 实验注册表（文档列即 README/docs 的导航来源）-------------------------
reg init        "README.md · docs/README.md"                    "环境体检（不编译不跑门禁）" \
    "scripts/init.sh --check"
reg gate        "README.md · docs/实验对比.md"                  "4 道门禁 + 编译（不含性能矩阵）" \
    "scripts/one_click_test.sh --no-matrix"
reg matrix      "docs/matrix_test_a7.md"                        "49 点性能矩阵（权威结果）" \
    "python3 scripts/matrix_test.py --reps 20 --rounds 3 --out docs/matrix_test_a7.md"
reg matrix-archive "docs/matrix_test_raw.md"                    "批折叠前的 44:5 基线存档（勿覆盖）" \
    "python3 scripts/matrix_test.py --reps 20 --out docs/matrix_test_raw.md"
reg sixway      "docs/性能对比-标准库vs自研.md"                 "六基线 49 点同场（numpy/torch/rfft/v1/原生/自研）" \
    "python3 scripts/gen_stdlib_doc.py"
reg e2e         "docs/实验对比.md · 图6/图7"                    "端到端（H2D+变换+D2H）三路对比" \
    "python3 scripts/e2e_test.py --reps 10 --rounds 3"
reg figures     "docs/实验对比.md"                              "出 7 张图 -> docs/figures/" \
    "python3 scripts/plot_results.py"
reg doc         "docs/实验对比.md"                              "出图 + 详表文档（本文档是生成物）" \
    "python3 scripts/gen_compare_doc.py"
reg eta         "docs/性能优化-C2b与K择优.md §3/§11.5"          "η 成本模型最小二乘标定" \
    "python3 scripts/calib_eta.py --fit"
reg ab          "docs/性能优化-C2b与K择优.md §7/§9/§11"         "批量 A/B 消融（默认 自研v1 vs 自研）" \
    "python3 scripts/ab_test.py --base build/fft_radix2_v1.o --cand build/fft_radix2.o --points 4096x4096,64x4096,128x4 --reps 50 --rounds 3"
reg baseline-o  "docs/性能优化-C2b与K择优.md §7"                "从任意提交重新构建基线 kernel .o" \
    "scripts/baseline_o.sh HEAD"
reg hwprobe     "docs/阶段0-1-发现与结果.md §1 · 性能优化 §10.2" "硬件能力探针（核数/子核/mask 上限/Gather/带宽/SIMT）" \
    "scripts/hw_probe.sh"
reg profile     "docs/trace与profile诊断-小尺寸与大尺寸.md §0"  "msprof 采集 + 汇总（5 个用例）" \
    "scripts/profile_test.sh"
reg profile-sum "docs/trace与profile诊断-小尺寸与大尺寸.md §7"  "汇总已有 profile 目录（参数=目录）" \
    "python3 scripts/sum_prof.py results/profiles/p_b4k"
reg native      "docs/性能对比-标准库vs自研.md"                 "CANN 原生复数 FFT 基线（含 maxRel）" \
    "python3 scripts/bench_native_npu.py --ns 64,256,1024,4096 --bs 1,64,4096 --reps 20"
reg stdlib      "docs/性能对比-标准库vs自研.md"                 "CPU 标准库（numpy/torch）基线" \
    "python3 scripts/bench_stdlib.py perf --ns 64,256,1024,4096 --bs 1,64,4096 --reps 20"
reg rfft        "docs/阶段0-1-发现与结果.md §2"                 "裸 CANN aclRfft1D 基线（device-only）" \
    "./build/baseline_rfft 4096 4096 50"
reg rfft-e2e    "docs/实验对比.md · 图7"                        "裸 CANN aclRfft1D 端到端口径" \
    "./build/baseline_rfft 4096 4096 20 --e2e"
reg cube        "docs/Cube张量化探针.md"                        "fp32 Cube（矩阵单元）探针" \
    "./build/cube_probe build/cube_probe.o 3 1 16 2 0 1 65536 5"
reg bwprobe     "docs/实验对比.md §6.3"                         "传输带宽探针（H2D/D2H/GM）" \
    "./build/bw_probe 256 20"
reg gpu-compare "docs/矩阵测试与GPU绝对性能对比.md"             "与公开 GPU 工作的绝对性能对照矩阵" \
    "python3 scripts/matrix_test.py --reps 30 --out results/gpu_compare.md"

# ---------------------------------------------------------------------------

list() {
  printf '  %-16s %-46s %s\n' "实验" "文档" "说明"
  printf '  %-16s %-46s %s\n' "----" "----" "----"
  local i
  for i in "${!NAMES[@]}"; do
    printf '  %-16s %-46s %s\n' "${NAMES[$i]}" "${DOCS[$i]}" "${DESC[$i]}"
  done
  echo
  echo "  用 scripts/repro.sh <实验名> 运行；命令见 --list --verbose"
}

list_doc() {
  local pat="$1" i hit=0
  for i in "${!NAMES[@]}"; do
    case "${DOCS[$i]}" in
      *"$pat"*) printf '  %-16s %s\n' "${NAMES[$i]}" "${CMDS[$i]}"; hit=1 ;;
    esac
  done
  [ "$hit" = 1 ] || { echo "没有实验匹配文档 '$pat'"; return 2; }
}

run_one() {
  local name="$1"; shift
  local i
  for i in "${!NAMES[@]}"; do
    if [ "${NAMES[$i]}" = "$name" ]; then
      local cmd="${CMDS[$i]}"
      [ $# -gt 0 ] && cmd="$cmd $(printf '%q ' "$@")"
      printf '\n\033[1m== repro: %s ==\033[0m\n%s\n$ %s\n\n' \
        "$name" "${DESC[$i]}" "$cmd"
      eval "$cmd"
      return $?
    fi
  done
  echo "未知实验：$name（scripts/repro.sh --list 看清单）" >&2
  return 2
}

case "${1:-}" in
  ""|-h|--help) sed -n '2,13p' "$0"; exit 2 ;;
  --list)
    shift
    list
    if [ "${1:-}" = "--verbose" ] || [ "${1:-}" = "-v" ]; then
      echo; printf '  %-16s %s\n' "实验" "命令"
      for i in "${!NAMES[@]}"; do printf '  %-16s %s\n' "${NAMES[$i]}" "${CMDS[$i]}"; done
    fi
    exit 0 ;;
  --doc) list_doc "${2:?--doc 需要文档名片段}"; exit $? ;;
  all)
    fail=0
    # 慢实验与存档实验不进 all，避免误覆盖 matrix_test_raw.md
    for n in init gate hwprobe matrix sixway e2e figures doc eta rfft stdlib; do
      run_one "$n" || { echo "  [ FAIL ] $n"; fail=1; }
    done
    [ $fail -eq 0 ] && printf '\n\033[32mREPRO ALL OK\033[0m\n' || printf '\n\033[31mREPRO FAILED\033[0m\n'
    exit $fail ;;
  *) run_one "$1" "${@:2}"; exit $? ;;
esac
