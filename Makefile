# Ascend-FFT —— 便捷入口，实际构建逻辑在 scripts/build.sh
SHELL   := /bin/bash
AB      := ./scripts/build.sh
.PHONY: all kernel check test limits desc rfft probe simt bw stride cube \
        init quick matrix hwprobe profile repro clean help

all:        ## 编译 kernel + 全部 host 目标 + Stride 探针
	$(AB) all stride
kernel:     ## 只编译 device kernel（.o）
	$(AB) kernel
check:      ## fft_check 正确性/计时器
	$(AB) check
desc:       ## 架构描述符合法性自测（addendum §2，纯 g++）
	$(AB) desc
test:       ## test_framework
	$(AB) test
limits:     ## test_limits（18 项硬件/ABI 门禁）
	$(AB) limits
rfft:       ## aclRfft1D 基线（--e2e 为裸 CANN 端到端口径）
	$(AB) rfft
probe:      ## 硬件探针 kernel + launch 启动器（probe_hw / gather_probe）
	$(AB) probe
simt:       ## probe_simt 编译（本 SoC 预期失败，用来验证「无 SIMT」结论）
	$(AB) simt
bw:         ## 带宽探针
	$(AB) bw
stride:     ## Stride 探针
	$(AB) stride
cube:       ## Cube（矩阵单元）探针
	$(AB) cube

init:       ## 环境体检（不编译不跑门禁）
	scripts/init.sh --check
hwprobe:    ## 硬件能力探针一键串联（核数/子核/mask 上限/Gather/带宽/SIMT）
	scripts/hw_probe.sh
profile:    ## msprof 采集 + 汇总
	scripts/profile_test.sh
repro:      ## 列出全部实验 ↔ 文档 ↔ 脚本
	scripts/repro.sh --list
quick:      ## 一键快速自检（编译 + 4 道门禁 + 9 点抽样矩阵）
	scripts/one_click_test.sh --quick
matrix:     ## 一键完整自检（编译 + 4 道门禁 + 49 点矩阵）
	scripts/one_click_test.sh

clean:      ## 删除 build/ 产物
	rm -rf build
help:       ## 列出所有目标
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'
