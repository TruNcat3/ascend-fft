# Ascend-FFT —— 便捷入口，实际构建逻辑在 scripts/build.sh
SHELL   := /bin/bash
AB      := ./scripts/build.sh
.PHONY: all kernel check test limits rfft probe bw stride cube quick matrix clean help

all:        ## 编译 kernel + 全部 host 目标 + Stride/Cube 探针
	$(AB) all stride
kernel:     ## 只编译 device kernel（.o）
	$(AB) kernel
check:      ## fft_check 正确性/计时器
	$(AB) check
test:       ## test_framework + test_limits
	$(AB) test
limits:     ## test_limits（18 项硬件/ABI 门禁）
	$(AB) limits
rfft:       ## aclRfft1D 基线
	$(AB) rfft
probe:      ## probe_hw / probe_simt
	$(AB) probe
bw:         ## 带宽探针
	$(AB) bw
stride:     ## Stride 探针
	$(AB) stride
cube:       ## Cube（矩阵单元）探针
	$(AB) cube
quick:      ## 一键快速自检（编译 + 12 点抽样矩阵）
	scripts/one_click_test.sh --quick
matrix:     ## 一键完整自检（编译 + 4 道门禁 + 49 点矩阵）
	scripts/one_click_test.sh
clean:      ## 删除 build/ 产物
	rm -rf build
help:       ## 列出所有目标
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'
