#!/usr/bin/env bash
# 初始化：环境体检 → 编译 → 4 道门禁 → 打印下一步。克隆仓库后跑这一条即可。
#
#   scripts/init.sh             # 体检 + 编译 + 门禁（不跑性能矩阵，最快）
#   scripts/init.sh --check     # 只体检，不编译不跑门禁
#   scripts/init.sh --quick     # 体检 + 编译 + 门禁 + 9 点抽样矩阵（约 5 min）
#   scripts/init.sh --matrix    # 体检 + 编译 + 门禁 + 49 点全网格（最慢）
#
# 退出码：0 = 体检与门禁全过；1 = 有失败项。
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

ONLY_CHECK=0; QUICK=0; MATRIX=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check)  ONLY_CHECK=1 ;;
    --quick)  QUICK=1 ;;
    --matrix) MATRIX=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

fail=0; warn=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
PASS() { printf '  \033[32m[ PASS ]\033[0m %s\n' "$*"; }
WARN() { printf '  \033[33m[ WARN ]\033[0m %s\n' "$*"; warn=$((warn+1)); }
FAIL() { printf '  \033[31m [ FAIL ]\033[0m %s\n' "$*"; fail=1; }

step "1/4 环境体检"

# --- CANN ---
if [ -x "$AB_CANN/bin/ccec" ]; then
  PASS "CANN 工具包  $AB_CANN"
else
  FAIL "未找到 ccec：AB_CANN=$AB_CANN（设 AB_CANN=... 或 source set_env.sh）"
fi
if command -v ccec >/dev/null 2>&1; then
  PASS "ccec 在 PATH  $(command -v ccec)"
else
  FAIL "ccec 不在 PATH"
fi

# --- host 编译器 ---
if command -v g++ >/dev/null 2>&1; then
  PASS "host 编译器  $(g++ --version | head -1)"
else
  FAIL "缺 g++"
fi

# --- python + torch_npu ---
if "$AB_PY" -c 'import torch, torch_npu' >/dev/null 2>&1; then
  PASS "python       $AB_PY  （torch $($AB_PY -c 'import torch;print(torch.__version__)')）"
else
  FAIL "python=$AB_PY 缺 torch/torch_npu（矩阵测试与原生基线都要用）"
fi

# --- NPU 设备 ---
NPU_OK=$("$AB_PY" - <<'PY' 2>/dev/null || true
import torch
print("yes" if torch.npu.is_available() else "no")
PY
)
if [ "$NPU_OK" = "yes" ]; then
  SOC=$("$AB_PY" -c 'import torch_npu;print(torch_npu.npu.get_device_name(0))' 2>/dev/null || echo "?")
  PASS "NPU 设备      npu:0  $SOC"
else
  FAIL "torch.npu.is_available() = False（驱动或 torch_npu 不匹配）"
fi

# --- msprof（可选）---
if [ -n "$AB_MSPROF" ] && [ -x "$AB_MSPROF" ]; then
  PASS "msprof       $AB_MSPROF"
else
  WARN "未找到 msprof —— profile 类实验（scripts/profile_test.sh）不可用，其余不受影响"
fi

# --- 设计空间 / profile JSON ---
for f in config/ascend910_93_profile.json config/butterfly_space.json; do
  if [ -f "$f" ]; then PASS "配置          $f"
  else FAIL "缺配置 $f"; fi
done

if [ "$ONLY_CHECK" -eq 1 ]; then
  step "体检结果"
  [ "$fail" -eq 0 ] && printf '\033[32mENV OK\033[0m\n' || printf '\033[31mENV BROKEN\033[0m\n'
  [ "$fail" -eq 0 ] && exit 0 || exit 1
fi

# ---- 编译 + 门禁 ---------------------------------------------------------
if [ "$fail" -ne 0 ]; then
  step "环境有 FAIL 项，跳过编译与门禁"
  printf '\033[31mFAILED\033[0m\n'; exit 1
fi

step "2/4 + 3/4 编译 → 门禁"
ONE=()
[ "$QUICK" -eq 1 ]   && ONE+=(--quick)
[ "$MATRIX" -eq 1 ]  && ONE=()
[ "$QUICK" -eq 0 ] && [ "$MATRIX" -eq 0 ] && ONE+=(--no-matrix)
scripts/one_click_test.sh "${ONE[@]+"${ONE[@]}"}" || fail=1

step "4/4 下一步"
cat <<'EOF'
  复现全部实验       scripts/repro.sh --list      # 看清单
                       scripts/repro.sh all        # 跑全部非慢速实验（--list -v 看清单，慢速项需单独指定）
  单个实验           python3 scripts/e2e_test.py --reps 10 --rounds 3
  出图 / 出文档      python3 scripts/plot_results.py && python3 scripts/gen_compare_doc.py
  文档导航           见 README.md「文档与复现」与 docs/README.md
EOF

if [ "$fail" -eq 0 ]; then
  printf '\n\033[32mINIT OK\033[0m（警告 %d 项）\n' "$warn"; exit 0
else
  printf '\n\033[31mINIT FAILED\033[0m\n'; exit 1
fi
