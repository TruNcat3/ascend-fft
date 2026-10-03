#!/usr/bin/env bash
# 一键：编译 → 门禁 → 49 点性能矩阵（自研 kfft_fwd vs CANN 原生复数 FFT）→ 结论
#
# 用法:
#   scripts/one_click_test.sh                 # 完整：build + 门禁 + 49 点矩阵
#   scripts/one_click_test.sh --quick         # 只 build + 门禁 + 4 点抽样矩阵
#   scripts/one_click_test.sh --no-matrix     # 只 build + 门禁（最快）
#   scripts/one_click_test.sh --rounds 5      # 矩阵每点测几遍、逐点取 min（默认 3）
#   scripts/one_click_test.sh --reps 20       # 每遍内部 reps（默认 20）
#   scripts/one_click_test.sh --out DIR       # 结果目录（默认 results/<UTC 时间戳>）
#
# 退出码：0 = 全部门禁 + 矩阵判据通过；1 = 有失败项。
set -uo pipefail
cd "$(dirname "$0")/.."

ROUND_ARG=3; REPS_ARG=20; DO_MATRIX=1; QUICK=0; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick)    QUICK=1 ;;
    --no-matrix) DO_MATRIX=0 ;;
    --rounds)   ROUND_ARG="$2"; shift ;;
    --reps)     REPS_ARG="$2"; shift ;;
    --out)      OUT="$2"; shift ;;
    -h|--help)  sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$OUT" ] || OUT="results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
LOG="$OUT/one_click.log"
exec > >(tee "$LOG") 2>&1

fail=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
PASS() { printf '  \033[32m[ PASS ]\033[0m %s\n' "$*"; }
FAIL() { printf '  \033[31m[ FAIL ]\033[0m %s\n' "$*"; fail=1; }
EXPECT() { # EXPECT <实际> <期望> <说明>  —— 字符串相等即通过
  if [ "$1" = "$2" ]; then PASS "$3 ($1)"; else FAIL "$3: got '$1' want '$2'"; fi
}

step "0/5 编译（kernel + check + framework + limits + stride_probe）"
./scripts/build.sh all stride || FAIL "build"

step "1/5 门禁 A：test_limits（18 项边界/上限）"
L=$(./build/test_limits 2>&1 | tail -1)
EXPECT "$(echo "$L" | grep -oE '[0-9]+ passed, [0-9]+ failed')" "18 passed, 0 failed" "test_limits"

step "2/5 门禁 B：test_framework（选型闭环 + 真实验收，4 个抽样点）"
FW_FAIL=0
for p in "64 4096" "1024 4096" "2048 192" "4096 1"; do
  set -- $p
  s=$(./build/test_framework config/ascend910_93_profile.json \
      config/butterfly_space.json build/fft_radix2.o "$1" "$2" 2>&1)
  if echo "$s" | grep -q "^PASS$" && echo "$s" | grep -q "selected:"; then
    printf '  \033[32m[ PASS ]\033[0m n=%-5s b=%-5s %s\n' "$1" "$2" \
      "$(echo "$s" | grep -oE 'eta=[0-9.]+ us  measured=[0-9.]+ us' | tail -1)"
  else
    FAIL "test_framework n=$1 b=$2"; FW_FAIL=1
  fi
done

step "3/5 门禁 C：stride_probe（Level-0 mask/repeat 上限，23 PASS + 7 个预期 FAIL）"
SP=$OUT/stride_probe.log
./build/stride_probe > "$SP" 2>&1
NP=$(grep -c -- "-> PASS" "$SP"); NF=$(grep -c -- "-> FAIL" "$SP")
EXPECT "$NP" "23" "stride_probe PASS 数"
EXPECT "$NF" "7"  "stride_probe 预期 FAIL 数（mask>64 属 Level-0 硬上限）"

step "4/5 门禁 D：fft_check 抽样（正确性 maxRel ≤ 1e-4）"
export AB_FFT_O="${AB_FFT_O:-build/fft_radix2.o}"
SPOT_FAIL=0
for p in "64 4096" "1024 4096" "4096 4096" "2048 144"; do
  set -- $p
  s=$(./build/fft_check "$1" "$2" 5 2>&1)
  if echo "$s" | grep -q "^PASS$"; then
    printf '  \033[32m[ PASS ]\033[0m n=%-5s b=%-5s %s\n' "$1" "$2" \
      "$(echo "$s" | grep -oE 'maxRel=[0-9.eE+-]+' | head -1)"
  else
    FAIL "fft_check n=$1 b=$2"; SPOT_FAIL=1
  fi
done

SCORE="-"; ETA_OVER="-"; ETA_MEAN="-"; ETA_MAX="-"; CORRECT="-"; ROWS="-"; BOLD="-"; TOTAL="-"
if [ "$DO_MATRIX" -eq 1 ]; then
  step "5/5 性能矩阵（自研 vs CANN 原生）"
  NS=64,128,256,512,1024,2048,4096
  BS=1,4,16,64,256,1024,4096
  if [ "$QUICK" -eq 1 ]; then NS=64,1024,4096; BS=1,64,4096; fi
  /usr/local/python3.11.15/bin/python3 scripts/matrix_test.py \
      --ns "$NS" --bs "$BS" --reps "$REPS_ARG" --rounds "$ROUND_ARG" \
      --out "$OUT/matrix.md" > /dev/null
  M="$OUT/matrix.md"
  SCORE=$(grep -oE '\*\*正确性：[0-9]+/[0-9]+' "$M" | grep -oE '[0-9]+/[0-9]+')
  CORRECT="${SCORE%%/*}"; ROWS="${SCORE##*/}"
  CORRECT_STR="$SCORE"
  BOLD=$(grep -cE '^\| *[0-9]+ *\| *[0-9]+ *\|.*\*\*[0-9.]+×\*\*' "$M" || true)
  TOTAL=$(grep -cE '^\| *[0-9]+ *\| *[0-9]+ *\|' "$M" || true)
  SCORE="$BOLD/$TOTAL"
  [ "$CORRECT" = "$ROWS" ] || FAIL "矩阵正确性 $CORRECT_STR（应 $ROWS/$ROWS）"
  [ "$BOLD" = "$TOTAL" ]   || FAIL "矩阵比值 $SCORE（应全 ≥1×，自研不慢于原生）"
  ETA_OVER=$(/usr/local/python3.11.15/bin/python3 - "$M" <<'PY'
import re,sys
pat=re.compile(r"\| (\d+) \| (\d+) \| ([\d,.]+) \| ([\d,.]+) \| ([\d,.]+) \| ([\d,.]+) \| (?:\*\*([\d.]+)×\*\*|([\d.]+)×) \| ([\d,.]+) \| ([+-][\d.]+)% \|")
f=lambda s: float(s.replace(',',''))
# 组号：1 n | 2 b | 3 自研mean | 4 自研min | 5 原生mean | 6 原生min
#      7/8 比值 | 9 η | 10 η/实测−1   ← group(10) 才是偏差
d=[abs(f(m.group(10))) for l in open(sys.argv[1]) if (m:=pat.match(l))]
if not d: print("NA"); raise SystemExit
over=sum(1 for x in d if x>15)
print(f"{over} {sum(d)/len(d):.1f} {max(d):.1f}")
PY
)
  set -- $ETA_OVER
  ETA_OVER="${1:-NA}"; ETA_MEAN="${2:-NA}"; ETA_MAX="${3:-NA}"
  case "$ETA_OVER" in
    ''|*[!0-9]*) FAIL "η 偏差统计解析失败（ETA_OVER='$ETA_OVER'）" ;;
    *)
      # 闸门：≥41/49 点 |η/实测−1| ≤15%（实测含宿主负载噪声 mean/min 摆 1.0~1.7×，
      # 且 test_framework 选中的 udCore 偶尔非 48，η 与 fft_check 的 48 块实测口径略有差异）
      if [ "$QUICK" -eq 0 ]; then
        [ "$ETA_OVER" -le 8 ] || FAIL "η 偏差 >15% 的点有 $ETA_OVER 个（阈值 ≤8）"
        [ "${ETA_MEAN%.*}" -le 10 ] || FAIL "η 平均偏差 ${ETA_MEAN}% > 10%"
      fi ;;
  esac
else
  step "5/5 性能矩阵 —— 已按 --no-matrix 跳过"
fi

step "结论"
printf '  %-26s %s\n' "门禁 A test_limits"    "18 项边界"
printf '  %-26s %s\n' "门禁 B test_framework" "4 点抽样"
printf '  %-26s %s\n' "门禁 C stride_probe"    "23 PASS / 7 预期 FAIL"
printf '  %-26s %s\n' "门禁 D fft_check 抽样"  "4 点"
if [ "$DO_MATRIX" -eq 1 ]; then
  printf '  %-26s %s\n' "正确性"                "$CORRECT_STR"
  printf '  %-26s %s\n' "自研 ≥ 原生 的点"      "$SCORE"
  printf '  %-26s %s\n' "|η/实测−1|>15% 的点"   "$ETA_OVER / $ROWS（均值 ${ETA_MEAN}%、最大 ${ETA_MAX}%）"
fi
printf '  %-26s %s\n' "结果目录" "$OUT"
if [ $fail -eq 0 ]; then
  printf '\n\033[32mALL PASS\033[0m\n'; exit 0
else
  printf '\n\033[31mFAILED\033[0m\n'; exit 1
fi
