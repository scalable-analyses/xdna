srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

ifeq ($(devicename), npu)
xdna   ?= XDNA1
kernel ?= tensor_kernel_32x32x32_bf16_bf16_fp32
else ifeq (${devicename}, npu2)
xdna   ?= XDNA2
kernel ?=tensor_kernel_64x96x64_bfp16_bfp16_fp32
endif

mlir_file         ?= src/${xdna}.mlir
xclbin_target     ?= build/final_${xdna}.xclbin
insts_target      ?= build/insts_${xdna}.bin
executable_target ?= build/driver_kernel.bin

KERNEL_CC=${PEANO_INSTALL_DIR}/bin/clang++
ifeq (${devicename}, npu)
KERNEL_CFLAGS = --target=aie2-none-unknown-elf
else ifeq (${devicename}, npu2)
KERNEL_CFLAGS = --target=aie2p-none-unknown-elf
endif

.PHONY: run
run: ${executable_target} ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./${executable_target} ${xdna}

${executable_target}: ${srcdir}/src/driver_kernel.cpp
	rm -rf _build
	mkdir -p _build
	mkdir -p ${@D}
	cd _build && cmake -E env CXXFLAGS="-std=c++23" \
		cmake ${srcdir}/ -DTARGET_NAME=driver_kernel -Dsubdir=${subdir}
	cd _build && cmake --build . --config Release
	cp _build/driver_kernel $@

${xclbin_target}: ${mlir_file} ${kernel:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py \
				--alloc-scheme=basic-sequential --no-compile-host \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=${insts_target:build/%=%} \
				--aie-generate-xclbin --xclbin-name=${@F} \
				$(<:%=../%)

build/${kernel}.o: ${kernel:%=src/%.s}
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} -c ../$< -o ${@F}

.PHONY: clean
clean:
	rm -rf build _build
