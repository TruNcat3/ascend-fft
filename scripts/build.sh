#!/usr/bin/env bash
# 全量编译。用法: source scripts/env.sh && scripts/build.sh [target...]
# target: kernel | check | test | limits | rfft | probe | bw | stride | cube | all   (默认 all)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p build

KERN=(fft_radix2 fft_radix2_v1 fft_radix2_v2)
HOST=(fft_check test_framework test_limits baseline_rfft)
PROBE=(probe_hw probe_simt gather_probe)

do_kernel() { # kernel [.cpp...]（.cpp 同名生成 build/*.o）
  local s; for s in "$@"; do
    echo "  ccec  src/ascendc/$s.cpp"
    ab_ccec "src/ascendc/$s.cpp" -o "build/$s.o"
  done
}
do_host() {
  echo "  cxx   tests/$1 / src/..."
}

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then targets=(all); fi

for t in "${targets[@]}"; do
  case "$t" in
    kernel) do_kernel "${KERN[@]}" ;;
    check)  echo "  cxx   src/host/fft_check.cpp"
            ab_cxx src/host/fft_check.cpp -o build/fft_check ;;
    test)   echo "  cxx   tests/test_framework.cpp"
            ab_cxx tests/test_framework.cpp src/framework/butterfly.cpp src/framework/reference.cpp -o build/test_framework ;;
    limits) echo "  cxx   tests/test_limits.cpp"
            ab_cxx tests/test_limits.cpp src/framework/butterfly.cpp src/framework/reference.cpp -o build/test_limits ;;
    rfft)   echo "  cxx   src/host/baseline_rfft.cpp"
            ab_cxx src/host/baseline_rfft.cpp -o build/baseline_rfft -lopapi -lnnopbase ;;
    probe)  do_kernel "${PROBE[@]}"
            echo "  cxx   src/host/probe_hw.cpp"
            ab_cxx src/host/probe_hw.cpp -o build/probe_hw ;;
    bw)     echo "  cxx   src/host/bw_probe.cpp"
            ab_cxx src/host/bw_probe.cpp -o build/bw_probe ;;
    stride) echo "  ccec  src/ascendc/stride_probe.cpp"
            ab_ccec src/ascendc/stride_probe.cpp -o build/stride_probe.o
            echo "  cxx   src/host/stride_probe.cpp"
            ab_cxx src/host/stride_probe.cpp -o build/stride_probe ;;
    cube)   echo "  ccec  src/ascendc/cube_probe.cpp"
            ab_ccec src/ascendc/cube_probe.cpp -o build/cube_probe.o
            echo "  cxx   src/host/cube_probe.cpp"
            ab_cxx src/host/cube_probe.cpp -o build/cube_probe ;;
    all)
      do_kernel "${KERN[@]}"
      echo "  cxx   src/host/fft_check.cpp"
      ab_cxx src/host/fft_check.cpp -o build/fft_check
      echo "  cxx   tests/test_framework.cpp"
      ab_cxx tests/test_framework.cpp src/framework/butterfly.cpp src/framework/reference.cpp -o build/test_framework
      echo "  cxx   tests/test_limits.cpp"
      ab_cxx tests/test_limits.cpp src/framework/butterfly.cpp src/framework/reference.cpp -o build/test_limits
      echo "  cxx   src/host/baseline_rfft.cpp"
      ab_cxx src/host/baseline_rfft.cpp -o build/baseline_rfft -lopapi -lnnopbase
      ;;
    *) echo "unknown target: $t" >&2; exit 2 ;;
  esac
done
echo "build done -> build/"
