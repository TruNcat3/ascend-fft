#!/usr/bin/env bash
# 硬件能力探针一键串联（docs/阶段0-1-发现与结果.md §1、docs/性能优化-C2b与K择优.md §10.2 的可执行版）。
# 所有结论都是实测出来的，不是猜的 —— 换 SoC 后先跑这一条，再读文档里的数字。
#
#   scripts/hw_probe.sh              # 全部探针
#   scripts/hw_probe.sh --list       # 只列探针
#   scripts/hw_probe.sh --only a,b   # 只跑指定探针
#
# 探针（名字 = 输出小标题）：
#   limits   test_limits —— 30 项 ABI/边界门禁
#   hw       probe_hw via launch —— 核数 / 子核 / 架构号 / 标量 GM 写出可见性
#   stride   stride_probe —— Level-0 mask/repeat 硬上限（23 PASS + 7 预期 FAIL）
#   gather   gather_probe —— Gather 索引与偏移单位
#   bw       bw_probe —— H2D/D2H 与 GM 带宽
#   cube     cube_probe —— fp32 矩阵单元可行性（需先 scripts/build.sh cube）
#   simt     probe_simt 编译 —— 预期**失败**（本 SoC 无 SIMT）
#
# 退出码：0 = 全部符合预期；1 = 有不符预期的项。
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

LIST_ONLY=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST_ONLY=1 ;;
    --only) ONLY="$2"; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

ALL=(limits hw stride gather bw cube simt)

if [ "$LIST_ONLY" -eq 1 ]; then
  for k in "${ALL[@]}"; do printf '  %-8s %s\n' "$k" "$(grep -m1 "^#   $k " "$0" | sed 's/^   [a-z]* //')"; done
  exit 0
fi

SEL=("${ALL[@]}")
[ -n "$ONLY" ] && IFS=',' read -r -a SEL <<< "$ONLY"

fail=0; warn=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
PASS() { printf '  \033[32m[ PASS ]\033[0m %s\n' "$*"; }
WARN() { printf '  \033[33m[ WARN ]\033[0m %s\n' "$*"; warn=$((warn+1)); }
FAIL() { printf '  \033[31m [ FAIL ]\033[0m %s\n' "$*"; fail=1; }

need() { # need <bin> <build.sh target...>
  [ -x "build/$1" ] || ./scripts/build.sh "${@:2}" >/dev/null || { FAIL "编译 $1 失败"; return 1; }
}

for k in "${SEL[@]}"; do
  case "$k" in
  limits)
    step "limits —— 30 项 ABI/边界门禁（tests/test_limits.cpp）"
    need test_limits limits || continue
    L=$(./build/test_limits 2>&1 | tail -1)
    echo "  $L"
    case "$L" in *"30 passed, 0 failed"*) PASS "test_limits" ;; *) FAIL "test_limits: $L" ;; esac
    ;;

  hw)
    step "hw —— 核数 / 子核 / 架构号 / 标量 GM 写出（probe_hw + launch）"
    need launch probe || continue
    D=$(./build/launch build/probe_hw.o kprobe_hw p dump 768 48 2>&1)
    if [ -z "$D" ]; then FAIL "launch probe_hw 无输出"; continue; fi
    # 每核 16 个 float 一条记录：[0]=nblk [1]=blk [2]=sub [3]=nsub [4]=nsub' [7]=42
    R=$(printf '%s\n' "$D" | awk -v nblk=48 '
      {v[NR-1]=$2}
      END{
        okn=0; okb=0; oks=0; okn2=0; okm=0; seen=0;
        for(b=0;b<48;b++){o=b*16;
          if(v[o]==nblk) okn++;
          if(v[o+1]==b) okb++;
          if(v[o+2]==0) oks++;
          if(v[o+3]==1) okn2++;
          if(v[o+7]==42) okm++;
          seen++;
        }
        printf "nblk=%d blk=%d sub=%d nsub=%d magic=%d /%d",okn,okb,oks,okn2,okm,seen}')
    echo "  GetBlockNum=48 → $R"
    if echo "$R" | grep -q "nblk=48 blk=48 sub=48 nsub=48 magic=48 /48"; then
      PASS "48 核一一对应，GetSubBlockNum()=1（**无 sub-block**），记录未丢"
    else
      FAIL "probe_hw 记录不符（$R）"
    fi
    ;;

  stride)
    step "stride —— Level-0 mask/repeat 硬上限（23 PASS + 7 预期 FAIL）"
    need stride_probe stride || continue
    TMP="$AB_WORK/stride_probe.log"
    ./build/stride_probe > "$TMP" 2>&1
    NP=$(grep -c -- "-> PASS" "$TMP"); NF=$(grep -c -- "-> FAIL" "$TMP")
    echo "  末 5 行："; tail -5 "$TMP" | sed 's/^/    /'
    if [ "$NP" = "23" ] && [ "$NF" = "7" ]; then
      PASS "23 PASS / 7 预期 FAIL（mask>64 属 Level-0 硬上限）"
    else
      FAIL "stride_probe: $NP PASS / $NF FAIL（期望 23 / 7）—— 换 SoC 后此判据要复核"
    fi
    ;;

  gather)
    step "gather —— Gather 索引与偏移单位（src/host/gather_probe.cpp）"
    need gather_probe probe || continue
    G=$(./build/gather_probe 2>&1); rc=$?
    echo "$G" | sed 's/^/  /'
    if [ "$rc" -eq 0 ] && printf '%s\n' "$G" | grep -Fq 'Gather byte-offset/base validation: PASS (0 mismatches)'; then
      PASS "Gather 索引可用，索引 offset 单位 = **字节**（B 行 == 期望值）"
    else
      FAIL "gather_probe 数据验证失败或旧二进制缺少验证结果（rc=$rc；重建 scripts/build.sh probe）"
    fi
    ;;

  bw)
    step "bw —— 传输带宽（src/host/bw_probe.cpp）"
    need bw_probe bw || continue
    ./build/bw_probe 256 20 2>&1 | sed 's/^/  /'
    PASS "bw_probe 运行完成（数值用于 docs/实验对比.md §6.3 主机缓冲口径的量级参照）"
    ;;

  cube)
    step "cube —— fp32 矩阵单元（docs/Cube张量化探针.md）"
    if [ ! -x build/cube_probe ]; then
      WARN "cube_probe 未编译，跳过（scripts/build.sh cube）"
      continue
    fi
    # 正确性：a0b0 布局连跑 nrep=5 次。src/host/cube_probe.cpp 只在 nrep<=3 时才打印
    # `rep%d lay=%d -> PASS`，nrep=5 时全部通过只输出末尾 `OK lay=...` 行、
    # 有错则逐行 `-> FAIL` 并以退出码 2 结束 —— 所以这里按「退出码 0 且有 OK lay= 行」判定，
    # 不再找 `PASS`/`15/15`（那两个串在 nrep=5 的输出里根本不会出现）。
    C=$(./build/cube_probe build/cube_probe.o 2 1 16 2 0 5 2>&1); rc=$?
    echo "$C" | tail -3 | sed 's/^/  /'
    if [ "$rc" -eq 0 ] && printf '%s\n' "$C" | grep -q "OK lay="; then
      PASS "cube_probe 正确性（a0b0，nrep=5 全对 → OK lay= 行）"
    else
      WARN "cube_probe 正确性未拿到 OK 行（rc=$rc）：$(printf '%s\n' "$C" | tail -1)"
    fi
    ;;

  simt)
    step "simt —— probe_simt 编译（预期失败：本 SoC 无 SIMT）"
    O=$(./scripts/build.sh simt 2>&1); rc=$?
    printf '%s\n' "$O" | tail -5 | sed 's/^/  /'
    if [ "$rc" -ne 0 ]; then
      FAIL "probe_simt 工具链/编译失败，不能据此判断硬件能力（rc=$rc）"
    elif printf '%s\n' "$O" | grep -q "SIMT_PROBE_SUPPORTED"; then
      WARN "probe_simt 编译通过 —— docs/阶段0-1-发现与结果.md §1.1「SIMT 不可用」需复核"
    elif printf '%s\n' "$O" | grep -q "SIMT_PROBE_UNSUPPORTED"; then
      PASS "probe_simt 编译失败（预期：本 SoC 无 SIMT，见 docs/阶段0-1 §1.1）"
    else
      FAIL "probe_simt 未返回明确能力结果"
    fi
    ;;

  *) echo "unknown probe: $k" >&2; exit 2 ;;
  esac
done

step "结论"
printf '  %-24s %s\n' "探针" "${SEL[*]}"
printf '  %-24s %s\n' "警告" "$warn"
if [ $fail -eq 0 ]; then printf '\n\033[32mHW PROBE OK\033[0m\n'; exit 0
else printf '\n\033[31mHW PROBE FAILED\033[0m\n'; exit 1; fi
