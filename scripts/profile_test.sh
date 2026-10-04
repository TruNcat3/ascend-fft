#!/usr/bin/env bash
# msprof 一键采集 + 汇总（docs/trace与profile诊断-*.md §0 的可执行版）。
#
#   scripts/profile_test.sh                     # 全部用例，落 results/profiles/<UTC>/
#   scripts/profile_test.sh --list              # 只列用例
#   scripts/profile_test.sh --only ours_b4k,nat # 只跑指定用例
#   scripts/profile_test.sh --out DIR           # 指定输出目录
#   scripts/profile_test.sh --no-sum            # 只采集，不汇总
#
# 用例（名字 = 输出子目录名）：
#   ours_b1   自研 kernel, n=4096 B=1      （小尺寸诊断）
#   ours_b64  自研 kernel, n=4096 B=64
#   ours_b4k  自研 kernel, n=4096 B=4096   + PipeUtilization
#   nat       原生 torch.fft.fft, n=4096 B=4096 + PipeUtilization
#   rfft      裸 CANN aclRfft1D, n=4096 B=1024
#
# 采集不到 msprof 时直接报错退出（profile 实验不可用，其余实验不受影响）。
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

LIST_ONLY=0; NO_SUM=0; OUT=""; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list)  LIST_ONLY=1 ;;
    --no-sum) NO_SUM=1 ;;
    --out)   OUT="$2"; shift ;;
    --only)  ONLY="$2"; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

ALL=(ours_b1 ours_b64 ours_b4k nat rfft)

case_cmd() { # case_cmd <name> -> 设置命令数组 CMD_ARR 与 msprof 额外参数 PIPE_ARGS
  case "$1" in
    ours_b1)  CMD_ARR=(./build/fft_check 4096 1 10)        PIPE_ARGS=() ;;
    ours_b64) CMD_ARR=(./build/fft_check 4096 64 10)       PIPE_ARGS=() ;;
    ours_b4k) CMD_ARR=(./build/fft_check 4096 4096 10)     PIPE_ARGS=(--aic-metrics=PipeUtilization) ;;
    nat)      CMD_ARR=("$AB_PY" scripts/native_fft.py --n 4096 --b 4096 --reps 10)
              PIPE_ARGS=(--aic-metrics=PipeUtilization) ;;
    rfft)     CMD_ARR=(./build/baseline_rfft 4096 1024 10) PIPE_ARGS=() ;;
    *)        echo "unknown case: $1" >&2; return 2 ;;
  esac
}

if [ "$LIST_ONLY" -eq 1 ]; then
  for k in "${ALL[@]}"; do
    case_cmd "$k" || exit 2
    printf '  %-10s %s\n' "$k" "${CMD_ARR[*]}"
  done
  exit 0
fi

if [ -z "$AB_MSPROF" ] || [ ! -x "$AB_MSPROF" ]; then
  echo "profile_test.sh: 未找到 msprof（AB_CANN=$AB_CANN）。" >&2
  echo "  装 CANN profiler 后重试，或在 env.sh 里用 AB_MSPROF=... 指定。" >&2
  exit 3
fi

SEL=("${ALL[@]}")
if [ -n "$ONLY" ]; then IFS=',' read -r -a SEL <<< "$ONLY"; fi
[ -n "$OUT" ] || OUT="results/profiles/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
echo "msprof = $AB_MSPROF"
echo "输出目录 = $OUT"
echo

# 门禁类可执行必须先编译
if [ ! -x build/fft_check ]; then ./scripts/build.sh check >/dev/null || exit 1; fi
if [ ! -x build/baseline_rfft ]; then ./scripts/build.sh rfft >/dev/null || exit 1; fi
if [ ! -x build/cube_probe ] && printf '%s\n' "${SEL[@]}" | grep -qx cube; then
  ./scripts/build.sh cube >/dev/null || exit 1
fi

fail=0
for k in "${SEL[@]}"; do
  case_cmd "$k" || exit 2
  dir="$OUT/$k"
  printf '\033[1m-- 采集 %s --\033[0m\n' "$k"
  if ab_prof "$dir" --task-time=on ${PIPE_ARGS[@]+"${PIPE_ARGS[@]}"} -- "${CMD_ARR[@]}"; then
    echo "   -> $dir"
  else
    echo "   [ FAIL ] $k"; fail=1
  fi
done

if [ "$NO_SUM" -eq 0 ]; then
  printf '\n\033[1m== 汇总 ==\033[0m\n'
  args=()
  for k in "${SEL[@]}"; do [ -d "$OUT/$k" ] && args+=("$OUT/$k"); done
  if [ ${#args[@]} -gt 0 ]; then
    python3 scripts/sum_prof.py "${args[@]}" || fail=1
    echo "汇总已打到 stdout；原始 profile 在 $OUT/"
  fi
fi

if [ $fail -eq 0 ]; then printf '\n\033[32mPROFILE OK\033[0m\n'; exit 0
else printf '\n\033[31mPROFILE FAILED\033[0m\n'; exit 1; fi
