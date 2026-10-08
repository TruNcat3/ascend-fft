#!/usr/bin/env bash
# 全量编译。用法: source scripts/env.sh && scripts/build.sh [target...]
# target: kernel | check | test | limits | rfft | probe | simt | bw | stride | cube | all   (默认 all)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p build

KERN=(fft_radix2 fft_radix2_v1 fft_radix2_v2 fft_real)
HOST=(fft_check test_framework test_limits baseline_rfft)
# 本 SoC 可编译的探针 kernel（probe_simt 单独放 simt 目标：无 SIMT，预期编译失败）
PROBE=(probe_hw gather_probe)

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
    desc)   echo "  cxx   tests/test_descriptors.cpp"
            g++ -O2 -std=c++17 -Wall -I"$AB_ROOT/include" \
                tests/test_descriptors.cpp -o build/test_descriptors
            build/test_descriptors ;;
    test)   echo "  cxx   tests/test_framework.cpp"
            ab_cxx tests/test_framework.cpp src/framework/butterfly.cpp src/framework/reference.cpp -o build/test_framework ;;
    limits) echo "  cxx   tests/test_limits.cpp"
            ab_cxx tests/test_limits.cpp src/framework/butterfly.cpp src/framework/reference.cpp -o build/test_limits ;;
    rfft)   echo "  cxx   src/host/baseline_rfft.cpp"
            ab_cxx src/host/baseline_rfft.cpp -o build/baseline_rfft -lopapi -lnnopbase ;;
    probe)  do_kernel "${PROBE[@]}"
            echo "  cxx   src/host/launch.cpp"
            ab_cxx src/host/launch.cpp -o build/launch ;;
    simt)   echo "  ccec  src/ascendc/probe_simt.cpp"
            if ab_ccec src/ascendc/probe_simt.cpp -o build/probe_simt.o; then
              echo "  [ warn ] probe_simt 编译通过 —— 本 SoC 可能支持 SIMT，"
              echo "           docs/阶段0-1-发现与结果.md §1.1 的「SIMT 不可用」需复核"
            else
              echo "  [ ok  ] probe_simt 编译失败（预期）：本 SoC 无 SIMT"
              echo "           见 docs/阶段0-1-发现与结果.md §1.1"
            fi ;;
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
      echo "  cxx   tests/test_descriptors.cpp"
      g++ -O2 -std=c++17 -Wall -I"$AB_ROOT/include" \
          tests/test_descriptors.cpp -o build/test_descriptors
      build/test_descriptors >/dev/null
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
