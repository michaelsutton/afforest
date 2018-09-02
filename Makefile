# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -O3 -Wall
PAR_FLAG = -fopenmp

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

KERNELS = bc bfs cc cc_afforest pr sssp tc
KERNELS_CUDA = cc_cuda
SUITE = $(KERNELS) converter

.PHONY: all
all: $(SUITE)

.PHONY: cuda
cuda: $(KERNELS_CUDA)

%_cuda: device/%.cu src/*.h
	nvcc -arch sm_60 -O3 -std=c++11 -Isrc -Ideps/cub -Xcompiler -fopenmp -Xcompiler -Wall $< -o $@

% : src/%.cc src/*.h
	$(CXX) $(CXX_FLAGS) $< -o $@

# Testing
include test/test.mk

# Benchmark Automation
include benchmark/bench.mk


.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*
