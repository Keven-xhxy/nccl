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
OPS := broadcast reduce allgather reducescatter sendrecv alltoall gather scatter
OP ?= broadcast
RANKS ?= 2
RANK_MATRIX ?= 1 2 3 4
COUNT ?= 4097
ROOT ?= 0
ARGS ?=
TIMEOUT ?= 120
BUILD ?= $(abspath ../../build-collectives-simple$(if $(filter 1,$(MINIMAL)),-minimal,))
CONFIG := $(BUILD)/.build-config
NCCL_ROOT := ../..
include $(NCCL_ROOT)/makefiles/version.mk
NCCL_VERSION := $(shell printf '%d%02d%02d' $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH))
PUBLIC_HEADER := $(BUILD)/include/nccl.h
INCLUDES := -I$(BUILD)/include -I$(NCCL_ROOT)/src/include -I$(NCCL_ROOT)/src/include/plugin \
            -I$(NCCL_ROOT)/src/device -I$(CUDA_HOME)/include/cccl
OBJECTS := $(addprefix $(BUILD)/,$(addsuffix .o,$(OPS)))
BINARIES := $(addprefix $(BUILD)/,$(OPS))

ifeq ($(filter $(OP),$(OPS)),)
$(error OP must be one of: $(OPS))
endif

.PHONY: build all run test test-all test-matrix inspect inspect-all clean
# Build just the chosen operation; "all" builds eight independent programs.
build: $(BUILD)/$(OP)
all: $(BINARIES)

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

$(OBJECTS): $(BUILD)/%.o: collectives.cu Makefile $(PUBLIC_HEADER) $(CONFIG)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(MODE_FLAGS) $(INCLUDES) -arch=$(ARCH) \
	  -DEXAMPLE_$(shell echo $* | tr '[:lower:]' '[:upper:]')=1 \
	  -c -rdc=false -MMD -MF $(@:.o=.d) $< -o $@

$(BINARIES): $(BUILD)/%: $(BUILD)/%.o
	$(CXX) $< -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -o $@

run: build
	timeout $(TIMEOUT) $(BUILD)/$(OP) --ranks $(RANKS) --count $(COUNT) --root $(ROOT) $(ARGS)

test: build
	timeout $(TIMEOUT) $(BUILD)/$(OP) --ranks $(RANKS) --self-test $(ARGS)

# Preserve SKIP (77), continue through every program, and report real failures.
test-all: all
	@status=0; skipped=0; for op in $(OPS); do \
	  timeout $(TIMEOUT) $(BUILD)/$$op --ranks $(RANKS) --self-test; result=$$?; \
	  if test $$result -eq 77; then skipped=1; \
	  elif test $$result -ne 0; then status=1; fi; \
	done; if test $$status -ne 0; then exit $$status; \
	elif test $$skipped -ne 0; then echo 'SKIP: GPU suite incomplete'; exit 77; fi

test-matrix: all
	@status=0; skipped=0; for ranks in $(RANK_MATRIX); do for op in $(OPS); do \
	  timeout $(TIMEOUT) $(BUILD)/$$op --ranks $$ranks --self-test; result=$$?; \
	  if test $$result -eq 77; then skipped=1; \
	  elif test $$result -ne 0; then status=1; fi; \
	done; done; if test $$status -ne 0; then exit $$status; \
	elif test $$skipped -ne 0; then echo 'SKIP: GPU matrix incomplete'; exit 77; fi

inspect-all: all
	@set -e; for op in $(OPS); do $(MAKE) --no-print-directory inspect OP=$$op; done

# These checks run without a GPU. Preprocessing uses the same selection as compilation.
inspect: build
	@grep -q 'prims_simple.h' $(BUILD)/$(OP).d
	@! grep -E 'prims_ll(128)?\.h|/generate\.py' $(BUILD)/$(OP).d
	@$(NVCC) $(NVCCFLAGS) $(MODE_FLAGS) $(INCLUDES) -arch=$(ARCH) \
	  -DEXAMPLE_$(shell echo $(OP) | tr '[:lower:]' '[:upper:]')=1 \
	  -E collectives.cu > $(BUILD)/$(OP).ii
	@! grep -E 'ncclDevFuncTable|ncclDevKernelForFunc|ncclDevFuncRowToId|ncclKernelMain|struct ProtoLL|struct RunWorkColl|struct RunWorkBatch' $(BUILD)/$(OP).ii
	@! nm -a $(BUILD)/$(OP).o | grep -E 'ncclDevFunc|ncclDevKernelForFunc|__cudaRegisterLinkedBinary'
	@test "$$($(CUDA_HOME)/bin/cuobjdump --dump-elf-symbols $(BUILD)/$(OP).o | grep -c STO_ENTRY)" -eq 1
	@$(CUDA_HOME)/bin/cuobjdump --dump-elf-symbols $(BUILD)/$(OP).o | grep 'STO_ENTRY.*$(OP)_'
	@set -e; case $(OP) in \
	  broadcast|reduce) grep -q 'void runRing' $(BUILD)/$(OP).ii; grep -q '$(OP).h' $(BUILD)/$(OP).d ;; \
	  allgather) grep -q 'void runRing' $(BUILD)/$(OP).ii; grep -q 'all_gather.h' $(BUILD)/$(OP).d ;; \
	  reducescatter) grep -q 'void runRing' $(BUILD)/$(OP).ii; grep -q 'reduce_scatter.h' $(BUILD)/$(OP).d ;; \
	  *) grep -q 'void runSend' $(BUILD)/$(OP).ii; grep -q 'void runRecv' $(BUILD)/$(OP).ii; grep -q 'sendrecv.h' $(BUILD)/$(OP).d ;; \
	esac
ifeq ($(MINIMAL),1)
	@! grep -E 'network/unpack/unpack\.h' $(BUILD)/$(OP).d
	@$(CUDA_HOME)/bin/cuobjdump --dump-ptx $(BUILD)/$(OP).o > $(BUILD)/$(OP).ptx
	@! grep -E 'multimem\.|network/unpack/unpack\.h' $(BUILD)/$(OP).ptx
	@! grep -E 'ncclNetDevice(Unpack|IncrementHead|SaveHead)' $(BUILD)/$(OP).ii
endif
	@echo 'PASS: $(OP), original Simple helpers, one named kernel, no dispatch tables or RDC'

-include $(OBJECTS:.o=.d)

clean:
	rm -rf $(BUILD)
