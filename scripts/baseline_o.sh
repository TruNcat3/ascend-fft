#!/usr/bin/env bash
# 从任意 git 提交重新构建基线 kernel .o（解决文档里 `/tmp/op/*.o` 那种不可复现的基线）。
#
#   scripts/baseline_o.sh HEAD~5                      # -> build/baseline_<commit>_<build-hash>_fft_radix2.o
#   scripts/baseline_o.sh <rev> [kernel名] [输出名]    # kernel 默认 fft_radix2
#   scripts/baseline_o.sh --list                      # 列出已构建的历史基线
#
# 产物可用于 A/B：
#   python3 scripts/ab_test.py --base build/baseline_<name>_fft_radix2.o \
#                               --cand build/fft_radix2.o --points ...
#
# 做法：把该提交的树导出到临时目录（不动当前工作区、不切分支），在其中编译，
# 再把 .o 拷回 build/。历史提交若没有 scripts/build.sh 会直接报错。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

if [ "${1:-}" = "--list" ]; then
  ls -1 "$AB_BUILD"/baseline_*_*.o 2>/dev/null | sed 's/^/  /' || echo "  （还没有历史基线）"
  exit 0
fi

REV="${1:-}"; KERN="${2:-fft_radix2}"; OUTNAME="${3:-}"
[ -n "$REV" ] || { sed -n '2,13p' "$0"; exit 2; }

git rev-parse --verify --quiet "$REV^{commit}" >/dev/null \
  || { echo "baseline_o.sh: 找不到提交 '$REV'" >&2; exit 2; }

SHA=$(git rev-parse --verify "$REV^{commit}")
OUTNAME=$(printf '%s' "$OUTNAME" | tr -c 'A-Za-z0-9._-' '_')
CCEC=$(command -v ccec) || { echo "baseline_o.sh: ccec 不可用，不能验证或构建历史基线" >&2; exit 1; }
CCEC=$(readlink -f "$CCEC")
IDENTITY=$(mktemp "$AB_WORK/baseline_identity_XXXXXX")
TMP=""
cleanup() { rm -f "$IDENTITY"; if [ -n "$TMP" ]; then rm -rf "$TMP"; fi; }
trap cleanup EXIT

# The archived build scripts supply the flags; hash their commit plus the
# current toolchain and environment they inherit, not a moving revision name.
{
  printf 'cache_schema=2\nsource_commit=%s\nkernel=%q\n' "$SHA" "$KERN"
  printf 'ccec_path=%q\nccec_sha256=%s\n' "$CCEC" "$(sha256sum "$CCEC" | cut -d ' ' -f1)"
  printf 'ccec_version=%q\n' "$("$CCEC" --version 2>&1 || true)"
  for variable in AB_SOC AB_CANN AB_INC AB_ENV_SOURCED PATH LD_LIBRARY_PATH \
      CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH CFLAGS CXXFLAGS \
      CPPFLAGS LDFLAGS ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH; do
    printf '%s=%q\n' "$variable" "${!variable-}"
  done
  for metadata in "$AB_CANN/version.cfg" "$AB_CANN/version.info" \
      "$AB_CANN/ascend_toolkit_install.info" "$AB_CANN/set_env.sh"; do
    if [ -f "$metadata" ]; then
      printf 'toolkit_file=%q sha256=%s\n' "$metadata" "$(sha256sum "$metadata" | cut -d ' ' -f1)"
    fi
  done
} >"$IDENTITY"
BUILD_HASH=$(sha256sum "$IDENTITY" | cut -c1-16)
OUT="$AB_BUILD/baseline_${OUTNAME:+${OUTNAME}_}${SHA}_${BUILD_HASH}_${KERN}.o"

if [ -f "$OUT" ] && [ -f "$OUT.manifest" ] && cmp -s "$IDENTITY" "$OUT.manifest" && [ -z "${FORCE:-}" ]; then
  echo "已有 $OUT（设 FORCE=1 重建）"
  echo "$OUT"; exit 0
fi

TMP=$(mktemp -d "$AB_WORK/baseline_${SHA}_XXXXXX")
if ! git archive "$SHA" | tar -x -C "$TMP"; then
  echo "baseline_o.sh: 导出 $REV 失败" >&2; rm -rf "$TMP"; exit 1
fi

if [ ! -x "$TMP/scripts/build.sh" ] || [ ! -f "$TMP/scripts/env.sh" ]; then
  echo "baseline_o.sh: $REV 里没有 scripts/{build,env}.sh，该提交无法用本流程编译。" >&2
  echo "  改用当时的编译命令（见该提交的 docs/）或挑一个更近的提交。" >&2
  rm -rf "$TMP"; exit 1
fi

echo "== 在 $REV 上编译 $KERN =="
if (cd "$TMP" && AB_ROOT="$TMP" AB_BUILD="$TMP/build" AB_WORK="$TMP/.tmp" \
    FORCE=1 scripts/build.sh kernel) >"$TMP/build.log" 2>&1; then
  :
else
  echo "baseline_o.sh: $TMP 下编译失败（输出：）" >&2
  tail -20 "$TMP/build.log" >&2
  rm -rf "$TMP"; exit 1
fi

if [ ! -f "$TMP/build/$KERN.o" ]; then
  echo "baseline_o.sh: 编译产物 $KERN.o 不在预期位置（该提交的 kernel 名可能不同）" >&2
  echo "  可用：$(cd "$TMP/build" 2>/dev/null && ls *.o 2>/dev/null | tr '\n' ' ')" >&2
  rm -rf "$TMP"; exit 1
fi

rm -f "$OUT.manifest"
cp "$TMP/build/$KERN.o" "$OUT"
cp "$IDENTITY" "$OUT.manifest"
rm -rf "$TMP"
echo "-> $OUT  （提交 $SHA）"
echo "$OUT"
