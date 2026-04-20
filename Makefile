# --- toy_llm_cuda Makefile ---------------------------------------------------
#
# Targets:
#   make              default: build everything (toy_llm + all tests)
#   make tokenizer    CPU-only: compile the BPE tokenizer CLI (no CUDA needed)
#   make tests        build all tests (CUDA required for *.cu tests)
#   make clean
#
# Variables:
#   SM=80             target compute capability (default sm_80)
#   DEBUG=1           -g -O0
#   NVCC=nvcc
#   CXX=g++

SM      ?= 80
NVCC    ?= nvcc
CXX     ?= g++

# CUDA include / lib paths. Auto-detected from nvcc if present; overridable.
CUDA_HOME ?= $(shell dirname $(shell dirname $(shell which $(NVCC) 2>/dev/null)) 2>/dev/null)
ifeq ($(strip $(CUDA_HOME)),)
  CUDA_HOME := /usr/local/cuda
endif
CUDA_INC  := -I$(CUDA_HOME)/include
CUDA_LIB  := -L$(CUDA_HOME)/lib64 -lcudart

INC     := -Iinclude $(CUDA_INC)

CXXFLAGS := -std=c++17 -Wall -Wextra -O2 $(INC)
NVCCFLAGS := -std=c++17 -O2 -arch=sm_$(SM) -Iinclude \
             -Xcompiler -Wall,-Wextra

ifeq ($(DEBUG),1)
  CXXFLAGS  := -std=c++17 -Wall -Wextra -O0 -g $(INC)
  NVCCFLAGS := -std=c++17 -O0 -g -G -arch=sm_$(SM) -Iinclude \
               -Xcompiler -Wall,-Wextra
endif

BUILD := build

# --- sources -----------------------------------------------------------------

TOKENIZER_CPP := src/tokenizer.cpp src/bpe_train.cpp
MAIN_CPP      := src/main.cpp
CUDA_SRC      := src/tensor.cu src/embedding.cu src/matmul.cu \
                 src/rmsnorm.cu src/softmax.cu src/attention.cu \
                 src/ffn.cu src/model.cu

TOKENIZER_OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(TOKENIZER_CPP))
MAIN_OBJ      := $(patsubst src/%.cpp,$(BUILD)/%.o,$(MAIN_CPP))
CUDA_OBJ      := $(patsubst src/%.cu,$(BUILD)/%.o,$(CUDA_SRC))

# --- binaries ----------------------------------------------------------------

BIN              := toy_llm
BIN_TOKENIZER    := toy_tokenizer

# Test binaries
TEST_BINS := \
    tests/test_tokenizer   \
    tests/test_bpe_train   \
    tests/test_matmul      \
    tests/test_rmsnorm     \
    tests/test_attention   \
    tests/test_model_smoke

# --- default target ----------------------------------------------------------

.PHONY: all tokenizer tests clean

all: $(BIN) tests

tokenizer: $(BIN_TOKENIZER)

tests: $(TEST_BINS)

# --- build dir ---------------------------------------------------------------

$(BUILD):
	@mkdir -p $(BUILD)

# --- compile rules -----------------------------------------------------------

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --- link the main binary (needs CUDA runtime) ------------------------------

$(BIN): $(MAIN_OBJ) $(TOKENIZER_OBJ) $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# --- link the CPU-only tokenizer CLI ----------------------------------------
#
# We reuse main.cpp but only the BPE-related subcommands actually run; the
# 'generate' subcommand references model.cuh symbols, so we can't link CPU-only
# without CUDA. Provide a pure-CPU helper instead that exposes train/encode/decode.

$(BIN_TOKENIZER): $(TOKENIZER_OBJ) $(BUILD)/tokenizer_cli.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/tokenizer_cli.o: src/tokenizer_cli.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- tests -------------------------------------------------------------------

tests/test_tokenizer: tests/test_tokenizer.cpp $(TOKENIZER_OBJ) | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

tests/test_bpe_train: tests/test_bpe_train.cpp $(TOKENIZER_OBJ) | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

tests/test_matmul: tests/test_matmul.cu $(BUILD)/matmul.o | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

tests/test_rmsnorm: tests/test_rmsnorm.cu $(BUILD)/rmsnorm.o | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

tests/test_attention: tests/test_attention.cu $(BUILD)/attention.o | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

tests/test_model_smoke: tests/test_model_smoke.cu $(TOKENIZER_OBJ) $(CUDA_OBJ) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# --- clean -------------------------------------------------------------------

clean:
	rm -rf $(BUILD) $(BIN) $(BIN_TOKENIZER) $(TEST_BINS)
