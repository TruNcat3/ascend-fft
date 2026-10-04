#!/usr/bin/env bash
# 从任意 git 提交重新构建基线 kernel .o（解决文档里 `/tmp/op/*.o` 那种不可复现的基线）。
#
#   scripts/baseline_o.sh HEAD~5                      # -> build/baseline_HEAD_5_fft_radix2.o
#   scripts/baseline_o.sh <rev> [kernel名] [输出名]    # kernel 默认 fft_radix2
#   scripts/baseline_o.sh --list                      # 列出已构建的历史基线
#
# 产物可用于 A/B：
#   python3 scripts/ab_test.py --base build/baseline_<name>_fft_radix2.o \
#                               --cand build/fft_radix2.o --points ...
#
# 做法：把该提交的树导出到临时目录（不动当前工作区、不切分支），在其中编译，
# 再把 .o 拷回 build/。历史提交若没有 scripts/build.sh 会直接报错。
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

if [ "${1:-}" = "--list" ]; then
  ls -1 build/baseline_*_*.o 2>/dev/null | sed 's/^/  /' || echo "  （还没有历史基线）"
  exit 0
fi

REV="${1:-}"; KERN="${2:-fft_radix2}"; OUTNAME="${3:-}"
[ -n "$REV" ] || { sed -n '2,13p' "$0"; exit 2; }

git rev-parse --verify --quiet "$REV^{commit}" >/dev/null \
  || { echo "baseline_o.sh: 找不到提交 '$REV'" >&2; exit 2; }

SHA=$(git rev-parse --short "$REV")
[ -n "$OUTNAME" ] || OUTNAME=$REV
OUTNAME=$(printf '%s' "$OUTNAME" | tr -c 'A-Za-z0-9._-' '_')
OUT="$AB_BUILD/baseline_${OUTNAME}_${KERN}.o"

if [ -f "$OUT" ] && [ -z "${FORCE:-}" ]; then
  echo "已有 $OUT（设 FORCE=1 重建）"
  echo "$OUT"; exit 0
fi

TMP="$AB_WORK/baseline_$SHA"
rm -rf "$TMP"; mkdir -p "$TMP"
if ! git archive "$REV" | tar -x -C "$TMP"; then
  echo "baseline_o.sh: 导出 $REV 失败" >&2; rm -rf "$TMP"; exit 1
fi

if [ ! -x "$TMP/scripts/build.sh" ] || [ ! -f "$TMP/scripts/env.sh" ]; then
  echo "baseline_o.sh: $REV 里没有 scripts/{build,env}.sh，该提交无法用本流程编译。" >&2
  echo "  改用当时的编译命令（见该提交的 docs/）或挑一个更近的提交。" >&2
  rm -rf "$TMP"; exit 1
fi

echo "== 在 $REV 上编译 $KERN =="
if (cd "$TMP" && FORCE=1 scripts/build.sh kernel >/dev/null 2>&1); then
  :
else
  echo "baseline_o.sh: $TMP 下编译失败（输出：）" >&2
  (cd "$TMP" && scripts/build.sh kernel) 2>&1 | tail -20 >&2
  rm -rf "$TMP"; exit 1
fi

if [ ! -f "$TMP/build/$KERN.o" ]; then
  echo "baseline_o.sh: 编译产物 $KERN.o 不在预期位置（该提交的 kernel 名可能不同）" >&2
  echo "  可用：$(cd "$TMP/build" 2>/dev/null && ls *.o 2>/dev/null | tr '\n' ' ')" >&2
  rm -rf "$TMP"; exit 1
fi

cp "$TMP/build/$KERN.o" "$OUT"
rm -rf "$TMP"
echo "-> $OUT  （提交 $SHA）"
echo "$OUT"
