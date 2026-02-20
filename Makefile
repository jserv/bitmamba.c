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

.PHONY: all clean distclean check download

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

check: $(TARGET) bitmamba_1b.bin tokenizer.bin
	./$(TARGET) bitmamba_1b.bin "Where is Tokyo?" tokenizer

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
