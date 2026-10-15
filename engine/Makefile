# GuardCAN Week-2 HPC engine — build + benchmark.
#
#   make                 build the three binaries into build/
#   make export          (re)export models/ae_weights.npz + ae_reference.npz from the trained model
#   make verify          GPU engine vs the Python model on the held-out test split -> $(OUT)/verify.json
#   make bench           full measured benchmark suite -> $(OUT)/*.json
#   make nsys            Nsight Systems timeline of the stream pipeline -> $(OUT)/*.nsys-rep + CSV stats
#   make plot            charts + results tables from the JSON -> $(OUT)/
#   make all-results     bench + nsys + plot
#
# Variables: ARCH=native (or sm_75 for a T4, sm_89 for an RTX 40xx), NVCC=nvcc, PYTHON=python3,
#            OUT=../bench/hpc, MODELS=../models, DATA=data/synthetic (relative to the repo root).

NVCC   ?= nvcc
CXX    ?= g++
ARCH   ?= native
PYTHON ?= python3
NSYS   ?= nsys
OUT    ?= ../bench/hpc
MODELS ?= ../models
DATA   ?= data/synthetic

NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -Wall
CPUFLAGS  ?= -O3 -march=native -ffp-contract=fast -std=c++17 -Wall
BUILD     := build
HEADERS   := npz.h ae_model.h bench_util.h ae_kernel.cuh gpu_monitor.h
BINS      := $(BUILD)/infer_streams $(BUILD)/mem_strategy_bench $(BUILD)/cpu_baseline
MODEL_ARGS = --weights $(MODELS)/ae_weights.npz --ref $(MODELS)/ae_reference.npz

.PHONY: all export verify bench nsys plot all-results clean

all: $(BINS)

$(BUILD)/%: %.cu $(HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< -ldl -lpthread

$(BUILD)/cpu_baseline: cpu_baseline.cpp npz.h ae_model.h bench_util.h | $(BUILD)
	$(CXX) $(CPUFLAGS) -DCPU_FLAGS='"$(CXX) $(CPUFLAGS)"' -o $@ $<

$(BUILD) $(OUT):
	mkdir -p $@

export:
	cd .. && $(PYTHON) -m model.export_weights --models models --data $(DATA)

verify: $(BUILD)/infer_streams | $(OUT)
	$(BUILD)/infer_streams $(MODEL_ARGS) --verify-only --json $(OUT)/verify.json

bench: $(BINS) | $(OUT)
	$(BUILD)/cpu_baseline $(MODEL_ARGS) --json $(OUT)/cpu_baseline.json
	$(BUILD)/infer_streams $(MODEL_ARGS) --json $(OUT)/infer_streams.json
	$(BUILD)/mem_strategy_bench $(MODEL_ARGS) --json $(OUT)/mem_strategy.json

# Short, NVTX-annotated run of both issue orders (S=4, B=4096) so the timeline stays small.
nsys: $(BUILD)/infer_streams | $(OUT)
	$(NSYS) profile -t cuda,nvtx --force-overwrite true -o $(OUT)/infer_streams \
		$(BUILD)/infer_streams $(MODEL_ARGS) --profile --json $(OUT)/nsys_run.json
	$(NSYS) stats --force-export true --report cuda_gpu_trace,nvtx_pushpop_trace,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
		--format csv --output $(OUT)/nsys $(OUT)/infer_streams.nsys-rep
	rm -f $(OUT)/infer_streams.sqlite

plot:
	cd .. && $(PYTHON) -m engine.plot_results --dir $(patsubst ../%,%,$(OUT))

all-results: bench nsys plot

clean:
	rm -rf $(BUILD)
