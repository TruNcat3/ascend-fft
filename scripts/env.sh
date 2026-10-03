#!/usr/bin/env bash
# Shared environment for the Ascend butterfly port.
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
export ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}
export AB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AB_BUILD="$AB_ROOT/build"
export AB_SOC=${AB_SOC:-Ascend910_9382}
export PYTHON=${PYTHON:-/usr/local/python3.11.15/bin/python3}
mkdir -p "$AB_BUILD"

# ccec needs the full asc include tree as -I entries (headers include by relative path).
if [ -z "${AB_INC:-}" ]; then
  AB_INC=$(find "$ASCEND_TOOLKIT_HOME/aarch64-linux/asc" -type d | sed 's/^/-I/' | tr '\n' ' ')
  export AB_INC
fi

ab_ccec() { # ab_ccec <src...> -o <out.o> [extra flags...]
  ccec -c --cce-aicore-only --npu-soc="$AB_SOC" --asc-aicore-lang -O2 -std=c++17 \
       -I"$AB_ROOT/include" $AB_INC "$@"
}

ab_cxx() { # ab_cxx <src...> -o <bin> [extra flags...]
  g++ -O2 -std=c++17 -I"$ASCEND_TOOLKIT_HOME/include" -I"$AB_ROOT/include" "$@" \
      -L"$ASCEND_TOOLKIT_HOME/lib64" -lascendcl \
      -Wl,-rpath,"$ASCEND_TOOLKIT_HOME/lib64" -lm
}
