# BitMamba C11 Inference Engine

CC ?= gcc
CFLAGS_BASE = -std=gnu11 -O3 -Wall -Wextra -Isrc
LDFLAGS = -lm -lpthread

# Detect ISA family and OS
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

ifneq (,$(filter x86_64 amd64,$(UNAME_M)))
  ISA_FAMILY = x86
else ifneq (,$(filter aarch64 arm64,$(UNAME_M)))
  ISA_FAMILY = arm
else
  ISA_FAMILY = generic
endif

TARGET = bitmamba

# Common sources (always compiled, no arch flags)
COMMON_SRCS = \
    src/utils.c \
    src/quantization.c \
    src/dispatch.c \
    src/threadpool.c \
    src/kernel.c \
    src/block.c \
    src/tokenizer.c \
    src/model.c \
    src/main.c
COMMON_OBJS = $(COMMON_SRCS:.c=.o)

# Architecture-specific objects (compiled with per-file flags)
ARCH_OBJS =
ifeq ($(ISA_FAMILY),x86)
  ARCH_OBJS += src/arch/avx2.o
endif
ifeq ($(ISA_FAMILY),arm)
  ARCH_OBJS += src/arch/neon.o
endif

# Metal GPU acceleration (Apple Silicon only)
METAL_OBJS =
ifeq ($(UNAME_S),Darwin)
ifeq ($(ISA_FAMILY),arm)
  METAL_SUPPORT = 1
  CFLAGS_BASE += -DHAS_METAL
  METAL_LDFLAGS = -framework Metal -framework Foundation
  METAL_OBJS = src/metal/bm_metal.o
endif
endif

# Cache line size override: make BM_CACHE_LINE=64 (default: auto-detect in threadpool.h)
ifdef BM_CACHE_LINE
  CFLAGS_BASE += -DBM_CACHE_LINE=$(BM_CACHE_LINE)
endif

CFLAGS = $(CFLAGS_BASE)
LDFLAGS += $(METAL_LDFLAGS)

ALL_OBJS = $(COMMON_OBJS) $(ARCH_OBJS) $(METAL_OBJS)

# Download URLs
MODEL_URL = https://huggingface.co/Zhayr1/BitMamba-2-1B/resolve/main/bitmamba_cpp/bitmamba_1b.bin
VOCAB_URL = https://huggingface.co/gpt2/raw/main/vocab.json

.PHONY: all clean distclean check check-raw check-sampling check-tokenizer download

all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Common sources: base flags only
src/%.o: src/%.c src/bitmamba.h src/dispatch.h src/threadpool.h
	$(CC) $(CFLAGS) -c -o $@ $<

# AVX2 kernel: compiled with -mavx2 -mfma
src/arch/avx2.o: src/arch/avx2.c src/bitmamba.h src/dispatch.h src/threadpool.h
	$(CC) $(CFLAGS) -mavx2 -mfma -c -o $@ $<

# NEON kernel: compiled with -mcpu=native
src/arch/neon.o: src/arch/neon.c src/bitmamba.h src/dispatch.h src/threadpool.h
	$(CC) $(CFLAGS) -mcpu=native -c -o $@ $<

# Metal shader source header (auto-generated)
src/metal/bm_shaders_source.h: src/metal/bm_shaders.metal
	xxd -i $< > $@

# Metal GPU acceleration (pure C via ObjC runtime API)
src/metal/bm_metal.o: src/metal/bm_metal.c src/metal/bm_metal.h src/metal/bm_shaders_source.h src/bitmamba.h src/dispatch.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Helper: extract token IDs between banners, normalize whitespace
# Usage: $(call extract_raw, ./bitmamba args...)
extract_raw = $$($(1) 2>/dev/null \
    | awk '/^=== End/{f=0} f; /^=== Generated Token IDs ===$$/{f=1}' | xargs)

# Helper: extract generated text between banners, strip leading blank lines
extract_text = $$($(1) 2>/dev/null \
    | awk '/^=== End/{f=0} f && NF; /^=== Generated Text ===$$/{f=1}')

# Test 1: "Hello, I am" (15496 11 314 716) → 50 tokens, exercises long-range coherence
EXPECT_RAW1 = 257 3710 290 314 761 1037 351 616 16237 13 \
    1680 345 1037 502 351 340 30 198 17250 11 314 761 1037 351 616 16237 13 \
    314 761 1037 351 340 13 1680 345 1037 502 351 340 30 198 17250 11 314 \
    761 1037 351 616 16237 13

# Test 2: "What is the speed of light?" (464 318 262 3139 286 5765 30) → 20 tokens
EXPECT_RAW2 = 198 91 39315 15886 50 1648 844 15886 39315 286 15886 39315 \
    286 15886 39315 286 15886 39315 286 15886

# Test 3: long prefill (64 input tokens) → 5 output tokens, stresses batched matmul
PROMPT_LONG = 15496 11 314 716 257 3710 290 314 761 1037 351 616 16237 13 \
    1680 345 1037 502 351 340 30 198 17250 11 314 761 1037 351 616 16237 13 \
    314 761 1037 351 340 13 1680 345 1037 502 351 340 30 198 17250 11 314 \
    761 1037 351 616 16237 13 257 3710 290 314 761 1037 351 616 16237 13
EXPECT_RAW3 = 314 761 1037 351 340

# Test 4: single-token prompt (no prefill, exercises pure autoregressive path)
EXPECT_RAW4 = 11 616 1438 318 4422 11 290 314 716 257

# Test 5: deterministic sampling (--seed 42, temp=0.8 top_k=40 top_p=0.9 min_p=0.05)
# Exercises: histogram top_k, heap select, softmax, top_p sort, CDF sampling
# EOS (token 0) terminates at 24 tokens
EXPECT_SAMP1 = 257 31516 290 1312 423 645 2126 703 284 923 616 898 3052 13 \
    1680 345 1037 502 503 30 198 17250 612 0

# Test 6: sampling with top_k=0 top_p=1.0 (no-sort fallback path in sampler)
EXPECT_SAMP2 = 262 6123 286 11707 6644 10501 11 257 1664 326 29786 287 \
    13359 262 2324 290 11540 286 40705 8251

# Test 7: tokenizer round-trip, "Where is Tokyo?" → 20 tokens
EXPECT_TOK = Tokyo is the capital city of Japan and the largest city in the country. It is located

check: check-raw check-sampling check-tokenizer
	@echo "All checks passed."

check-raw: $(TARGET) bitmamba_1b.bin
	@echo "--- Raw check 1: Hello, I am (50 tokens) ---"
	@actual=$(call extract_raw, ./$(TARGET) bitmamba_1b.bin "15496 11 314 716" raw 0.0 1.0 0.0 1.0 0 50); \
	if [ "$$actual" = "$(EXPECT_RAW1)" ]; then \
	    echo "PASS"; \
	else \
	    echo "FAIL"; \
	    echo "  expected: $(EXPECT_RAW1)"; \
	    echo "  actual:   $$actual"; \
	    exit 1; \
	fi
	@echo "--- Raw check 2: What is the speed of light? (20 tokens) ---"
	@actual=$(call extract_raw, ./$(TARGET) bitmamba_1b.bin "464 318 262 3139 286 5765 30" raw 0.0 1.0 0.0 1.0 0 20); \
	if [ "$$actual" = "$(EXPECT_RAW2)" ]; then \
	    echo "PASS"; \
	else \
	    echo "FAIL"; \
	    echo "  expected: $(EXPECT_RAW2)"; \
	    echo "  actual:   $$actual"; \
	    exit 1; \
	fi
	@echo "--- Raw check 3: long prefill (64 input, 5 output) ---"
	@actual=$(call extract_raw, ./$(TARGET) bitmamba_1b.bin "$(PROMPT_LONG)" raw 0.0 1.0 0.0 1.0 0 5); \
	if [ "$$actual" = "$(EXPECT_RAW3)" ]; then \
	    echo "PASS"; \
	else \
	    echo "FAIL"; \
	    echo "  expected: $(EXPECT_RAW3)"; \
	    echo "  actual:   $$actual"; \
	    exit 1; \
	fi
	@echo "--- Raw check 4: single-token prompt (10 tokens) ---"
	@actual=$(call extract_raw, ./$(TARGET) bitmamba_1b.bin "15496" raw 0.0 1.0 0.0 1.0 0 10); \
	if [ "$$actual" = "$(EXPECT_RAW4)" ]; then \
	    echo "PASS"; \
	else \
	    echo "FAIL"; \
	    echo "  expected: $(EXPECT_RAW4)"; \
	    echo "  actual:   $$actual"; \
	    exit 1; \
	fi

check-sampling: $(TARGET) bitmamba_1b.bin
	@echo "--- Sampling check 1: seed=42 temp=0.8 top_k=40 (24 tokens, EOS-terminated) ---"
	@actual=$(call extract_raw, ./$(TARGET) --seed 42 bitmamba_1b.bin "15496 11 314 716" raw 0.8 1.15 0.05 0.9 40 30); \
	if [ "$$actual" = "$(EXPECT_SAMP1)" ]; then \
	    echo "PASS"; \
	else \
	    echo "FAIL"; \
	    echo "  expected: $(EXPECT_SAMP1)"; \
	    echo "  actual:   $$actual"; \
	    exit 1; \
	fi
	@echo "--- Sampling check 2: seed=42 top_k=0 top_p=1.0 (no-sort path, 20 tokens) ---"
	@actual=$(call extract_raw, ./$(TARGET) --seed 42 bitmamba_1b.bin "15496 11 314 716" raw 0.8 1.0 0.0 1.0 0 20); \
	if [ "$$actual" = "$(EXPECT_SAMP2)" ]; then \
	    echo "PASS"; \
	else \
	    echo "FAIL"; \
	    echo "  expected: $(EXPECT_SAMP2)"; \
	    echo "  actual:   $$actual"; \
	    exit 1; \
	fi

check-tokenizer: $(TARGET) bitmamba_1b.bin tokenizer.bin
	@echo "--- Tokenizer check: Where is Tokyo? (20 tokens) ---"
	@actual=$(call extract_text, ./$(TARGET) bitmamba_1b.bin "Where is Tokyo?" tokenizer 0.0 1.0 0.0 1.0 0 20); \
	if [ "$$actual" = "$(EXPECT_TOK)" ]; then \
	    echo "PASS"; \
	else \
	    echo "FAIL"; \
	    echo "  expected: $(EXPECT_TOK)"; \
	    echo "  actual:   $$actual"; \
	    exit 1; \
	fi

# Download model weights and build tokenizer
download: bitmamba_1b.bin tokenizer.bin

bitmamba_1b.bin:
	curl -L -o $@ $(MODEL_URL)

vocab.json:
	curl -L -o $@ $(VOCAB_URL)

tokenizer.bin: vocab.json scripts/export-tokenizer.py
	python3 scripts/export-tokenizer.py --vocab $< --output $@

clean:
	rm -f $(TARGET) src/*.o src/arch/*.o src/metal/*.o src/metal/bm_shaders_source.h

distclean: clean
	rm -f bitmamba_1b.bin vocab.json tokenizer.bin
