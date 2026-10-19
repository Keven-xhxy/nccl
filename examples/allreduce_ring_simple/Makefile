.DEFAULT_GOAL := build
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
ARCH ?= sm_80
MINIMAL ?= 0
ifneq ($(MINIMAL),0)
ifneq ($(MINIMAL),1)
$(error MINIMAL must be 0 or 1)
endif
endif
MODE_FLAGS := $(if $(filter 1,$(MINIMAL)),-DNCCL_MINIMAL_RING_SIMPLE=1,)
NVCCFLAGS ?= -std=c++17 -O2 -lineinfo --extended-lambda
RANKS ?= 2
RANK_MATRIX ?= 1 2 3 4
COUNT ?= 4097
TIMEOUT ?= 120
BUILD ?= $(abspath ../../build-allreduce-ring-simple$(if $(filter 1,$(MINIMAL)),-minimal,))
CONFIG := $(BUILD)/.build-config
OBJECT := $(BUILD)/allreduce.o
BINARY := $(BUILD)/allreduce_ring_simple
NCCL_ROOT := ../..
include $(NCCL_ROOT)/makefiles/version.mk
NCCL_VERSION := $(shell printf '%d%02d%02d' $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH))
PUBLIC_HEADER := $(BUILD)/include/nccl.h
INCLUDES := -I$(BUILD)/include -I$(NCCL_ROOT)/src/include -I$(NCCL_ROOT)/src/include/plugin \
            -I$(NCCL_ROOT)/src/device -I$(CUDA_HOME)/include/cccl

.PHONY: build run test test-matrix inspect clean
build: $(BINARY)

.PHONY: FORCE
# Rebuild if an explicit BUILD is reused with a different mode or toolchain options.
$(CONFIG): FORCE
	@mkdir -p $(@D)
	@printf '%s\n' '$(MINIMAL)' '$(ARCH)' '$(NVCC)' '$(NVCCFLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(PUBLIC_HEADER): $(NCCL_ROOT)/src/nccl.h.in $(NCCL_ROOT)/makefiles/version.mk
	@mkdir -p $(@D)
	sed -e 's/$${nccl:Major}/$(NCCL_MAJOR)/g' -e 's/$${nccl:Minor}/$(NCCL_MINOR)/g' \
	  -e 's/$${nccl:Patch}/$(NCCL_PATCH)/g' -e 's/$${nccl:Suffix}/$(NCCL_SUFFIX)/g' \
	  -e 's/$${nccl:Version}/$(NCCL_VERSION)/g' $< > $@

$(OBJECT): allreduce.cu Makefile $(PUBLIC_HEADER) $(CONFIG)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(MODE_FLAGS) $(INCLUDES) -arch=$(ARCH) \
	  -c -rdc=false -MMD -MF $(@:.o=.d) $< -o $@

# Ordinary host link; no nvcc device-link stage and no libnccl dependency.
$(BINARY): $(OBJECT)
	$(CXX) $< -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -o $@

run: build
	timeout $(TIMEOUT) $(BINARY) --ranks $(RANKS) --count $(COUNT)

test: build
	timeout $(TIMEOUT) $(BINARY) --ranks $(RANKS) --self-test

test-matrix: build
	@status=0; skipped=0; for ranks in $(RANK_MATRIX); do \
	  timeout $(TIMEOUT) $(BINARY) --ranks $$ranks --self-test; result=$$?; \
	  if test $$result -eq 77; then skipped=1; \
	  elif test $$result -ne 0; then status=1; fi; \
	done; if test $$status -ne 0; then exit $$status; \
	elif test $$skipped -ne 0; then echo 'SKIP: GPU matrix incomplete'; exit 77; fi

inspect: build
	@grep -q 'all_reduce.h' $(OBJECT:.o=.d)
	@grep -q 'prims_simple.h' $(OBJECT:.o=.d)
	@! grep -E 'prims_ll(128)?\.h|/generate\.py' $(OBJECT:.o=.d)
	@$(NVCC) $(NVCCFLAGS) $(MODE_FLAGS) $(INCLUDES) -arch=$(ARCH) -E allreduce.cu > $(BUILD)/allreduce.ii
	@grep -q 'void runRing' $(BUILD)/allreduce.ii
	@! grep -E 'ncclDevFuncTable|ncclDevKernelForFunc|ncclDevFuncRowToId|ncclKernelMain|struct ProtoLL|runTreeSplit|runTreeUpDown|struct RunWorkColl' $(BUILD)/allreduce.ii
	@! nm -a $(OBJECT) | grep -E 'ncclDevFunc|ncclDevKernelForFunc|__cudaRegisterLinkedBinary'
	@test "$$($(CUDA_HOME)/bin/cuobjdump --dump-elf-symbols $(OBJECT) | grep -c STO_ENTRY)" -eq 1
	@$(CUDA_HOME)/bin/cuobjdump --dump-elf-symbols $(OBJECT) | grep allreduce_ring_simple
ifeq ($(MINIMAL),1)
	@! grep -E 'network/unpack/unpack\.h' $(OBJECT:.o=.d)
	@$(CUDA_HOME)/bin/cuobjdump --dump-ptx $(OBJECT) > $(BUILD)/allreduce.ptx
	@! grep -E 'multimem\.|network/unpack/unpack\.h' $(BUILD)/allreduce.ptx
	@! grep -E 'ncclNetDevice(Unpack|IncrementHead|SaveHead)' $(BUILD)/allreduce.ii
endif
	@echo 'PASS: original runRing + Simple primitives, one named kernel, no dispatch tables or RDC'

-include $(OBJECT:.o=.d)

clean:
	rm -rf $(BUILD)
