#!/usr/bin/env bash
# 单一真源：本仓库所有脚本的环境变量、工具路径与构建函数都在这里探测与导出。
#
#   source scripts/env.sh        # 子脚本均已自动 source，手工 shell 里执行这一句即可
#   scripts/init.sh              # 从零初始化：环境体检 + 编译 + 门禁
#
# 全部路径都可用同名环境变量覆盖，例如：
#   AB_CANN=/opt/cann/8.0.0 AB_PY=/usr/bin/python3.10 scripts/build.sh all
#
# 导出的变量
#   AB_ROOT      仓库根
#   AB_BUILD     构建产物目录（默认 $AB_ROOT/build）
#   AB_WORK      临时工作区（profile / 中间文件，默认 $AB_ROOT/.tmp）
#   AB_SOC       SoC 名（默认 Ascend910_9382）
#   AB_CANN      CANN 工具包根（= ASCEND_TOOLKIT_HOME）
#   AB_PY        带 torch_npu 的 python3（PYTHON 为其历史别名）
#   AB_MSPROF    msprof 路径（空串 = 未装，profile 类脚本会降级提示）
#   AB_INC       ccec 需要的 -I 串（头文件按相对路径互相包含）
#
# 提供的函数
#   ab_ccec <src...> -o <out.o> [额外 flags...]      设备端（AscendC）编译
#   ab_cxx  <src...> -o <bin>  [额外 flags...]      host 端编译/链接
#   ab_py   <script.py> [args...]                   用探测到的 python 跑
#   ab_prof <out_dir> [msprof flags...] -- <cmd...> msprof 采集一个 profile

# ---- 仓库根 / 目录 -------------------------------------------------------
export AB_ROOT="${AB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export AB_BUILD="${AB_BUILD:-$AB_ROOT/build}"
export AB_WORK="${AB_WORK:-$AB_ROOT/.tmp}"
export AB_SOC="${AB_SOC:-Ascend910_9382}"
export AB_PROFILE="${AB_PROFILE:-$AB_ROOT/config/ascend910_93_profile.json}"
mkdir -p "$AB_BUILD" "$AB_WORK"

# ---- CANN 工具包 ---------------------------------------------------------
# 候选必须真的有 include/ 与 lib64/（/usr/local/Ascend/ascend-toolkit 只有 set_env.sh，
# 不能直接用作 AB_CANN）。优先级：显式 AB_CANN > ASCEND_TOOLKIT_HOME > PATH 里的 ccec
# > 最新 cann-* > ascend-toolkit/latest。
_ab_cann_ok() { [ -n "$1" ] && [ -d "$1/include" ] && [ -d "$1/lib64" ]; }
if [ -z "${AB_CANN:-}" ] || ! _ab_cann_ok "$AB_CANN"; then
  AB_CANN=""
  for _c in "${ASCEND_TOOLKIT_HOME:-}" /usr/local/Ascend/cann-* \
            /usr/local/Ascend/ascend-toolkit/latest; do
    if _ab_cann_ok "$_c"; then AB_CANN=$_c; break; fi
  done
  if [ -z "$AB_CANN" ]; then
    _ccec=$(command -v ccec 2>/dev/null || true)
    if [ -n "$_ccec" ]; then
      _c=$(cd "$(dirname "$(readlink -f "$_ccec")")/.." && pwd)
      _ab_cann_ok "$_c" && AB_CANN=$_c
    fi
    unset _ccec _c
  fi
  unset _c
fi
export AB_CANN

# set_env.sh 会前置 PATH / LD_LIBRARY_PATH / PYTHONPATH 并导出 ASCEND_TOOLKIT_HOME
if [ -f "$AB_CANN/set_env.sh" ] && [ -z "${AB_ENV_SOURCED:-}" ]; then
  AB_ENV_SOURCED=1; export AB_ENV_SOURCED
  # shellcheck disable=SC1090
  . "$AB_CANN/set_env.sh" >/dev/null 2>&1 || true
fi
export ASCEND_TOOLKIT_HOME="$AB_CANN"

# ---- python（要能 import torch + torch_npu） -----------------------------
# 结果缓存到 .ab_py，避免每次 source 都做一次 import 探测（约 1s）。
_ab_py_ok() { [ -n "$1" ] && "$1" -c 'import torch,torch_npu' >/dev/null 2>&1; }
if [ -z "${AB_PY:-}" ]; then
  AB_PY=$(cat "$AB_ROOT/.ab_py" 2>/dev/null || true)
  if [ -z "$AB_PY" ] || ! _ab_py_ok "$AB_PY"; then
    AB_PY=""
    for _c in python3 "${PYTHON:-}" python3.11 python3.10 python3.12 \
              /usr/local/python*/bin/python3; do
      if _ab_py_ok "$_c"; then AB_PY=$_c; break; fi
    done
    if [ -n "$AB_PY" ] && command -v "$AB_PY" >/dev/null 2>&1; then
      printf '%s' "$(command -v "$AB_PY")" > "$AB_ROOT/.ab_py" 2>/dev/null || true
      AB_PY=$(command -v "$AB_PY")
    fi
    unset _c
  fi
fi
[ -n "$AB_PY" ] || AB_PY=python3
export AB_PY
export PYTHON="$AB_PY"          # 历史别名，老脚本里 $PYTHON 仍可用

# ---- msprof --------------------------------------------------------------
if [ -z "${AB_MSPROF:-}" ]; then
  AB_MSPROF=$(command -v msprof 2>/dev/null || true)
  [ -n "$AB_MSPROF" ] || AB_MSPROF=$(ls "$AB_CANN"/bin/msprof 2>/dev/null || true)
  [ -n "$AB_MSPROF" ] || AB_MSPROF=$(ls "$AB_CANN"/tools/profiler/bin/msprof 2>/dev/null || true)
fi
export AB_MSPROF

# ---- ccec 的头文件 -I 串 -------------------------------------------------
if [ -z "${AB_INC:-}" ]; then
  _asc=$(ls -d "$AB_CANN"/aarch64-linux/asc "$AB_CANN"/tools/ccec_compiler/targets/*/asc 2>/dev/null | head -1 || true)
  if [ -n "$_asc" ]; then
    AB_INC=$(find "$_asc" -type d | sed 's/^/-I/' | tr '\n' ' ')
    export AB_INC
  fi
  unset _asc
fi

# ---- 构建函数 ------------------------------------------------------------
ab_ccec() { # ab_ccec <src...> -o <out.o> [extra flags...]
  ccec -c --cce-aicore-only --npu-soc="$AB_SOC" --asc-aicore-lang -O2 -std=c++17 \
       -I"$AB_ROOT/include" ${AB_INC:-} "$@"
}

ab_cxx() { # ab_cxx <src...> -o <bin> [extra flags...]
  g++ -O2 -std=c++17 -I"$AB_CANN/include" -I"$AB_ROOT/include" "$@" \
      -L"$AB_CANN/lib64" -lascendcl \
      -Wl,-rpath,"$AB_CANN/lib64" -lm
}

ab_py() { # ab_py <script.py> [args...]
  "$AB_PY" "$@"
}

ab_prof() { # ab_prof <out_dir> [msprof flags...] -- <cmd...>
  local out="$1"; shift
  local -a pre=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do pre+=("$1"); shift; done
  if [ "$#" -gt 0 ]; then shift; fi

  if [ -z "${AB_MSPROF:-}" ]; then
    echo "ab_prof: 未找到 msprof（AB_CANN=$AB_CANN），跳过采集、直接执行命令" >&2
    "$@"
    return $?
  fi
  rm -rf "$out"
  "$AB_MSPROF" --output="$out" ${pre[@]+"${pre[@]}"} "$@"
}
