# BitMamba.c

Portable C11 inference engine for
[BitMamba-2](https://github.com/Zhayr1/BitMamba-2) models (255M and 1B parameters).
Weights are BitNet 1.58-bit ternary values `{-1, 0, +1}` packed 4 per byte,
reducing the linear projection matmuls to additions and subtractions.
Non-ternary components (SSM recurrence, convolution, norms) still use float32.

## Why BitMamba

Edge AI inference is memory-bandwidth-bound. On a Raspberry Pi 5
(Cortex-A76, ~18 GB/s LPDDR4X), a 4-bit 1.5B Transformer already
saturates the memory bus at ~30-40 tok/s. Scaling to 3.8B drops to
~5 tok/s because the CPU stalls waiting on DRAM.

BitMamba-2 attacks both bottlenecks simultaneously:

1. Mamba-2 SSM: $O(1)$ per-token state update versus $O(N)$ attention decode.
   No KV cache, constant memory regardless of context length. The
   Structured State Space Duality (SSD) framework enables optimized
   matrix kernels during training while preserving linear-time recurrence
   at inference.
2. 1.58-bit ternary weights {-1, 0, 1}: a 1B model fits in ~250 MB
   (versus ~600 MB at 4-bit). Weights are trained from scratch via
   Quantization-Aware Training (QAT), not post-training compressed. The
   zero value is critical: it grants explicit feature filtering that
   strict binary {-1, 1} models lack, and reduces weight-activation
   multiplication to addition and subtraction.
3. Hybrid precision sanctuary: over 90% of parameters (dense projections)
   are ternary; the delicate SSM recurrence variables (z, B, C, dt) and
   the 1D convolution remain in full precision (bfloat16 during training,
   float32 in bitmamba.c), preventing the catastrophic
   forgetting that plagues uniformly quantized recurrent networks.

The result: BitMamba-2 1B achieves 52.86 tok/s on an Intel Core
i3-12100F (entry-level, DDR4, no NPU), in the same range as a 4-bit
Llama 3.2 1B benchmarked at ~50.7 tok/s on an AMD Ryzen AI 9 HX 375
(CPU-side AVX-512; NPU offload via vendor backends may differ).
The trend suggests algorithmic efficiency can narrow the gap with
hardware acceleration.

On zero-shot reasoning benchmarks, the 1B ternary model scores 63.3% on
ARC-Easy (versus 43.9% for the larger 1.3B binary Bi-Mamba) and defeats
Llama 3.2 1B on BoolQ (62.35% versus 50.24%), proving that linear-time
SSMs do not sacrifice reasoning under aggressive quantization.

This project provides a from-scratch C11 inference engine with T-MAC
lookup-table kernels, Metal GPU dispatch on Apple Silicon, and runtime
ISA detection, with no dependencies beyond a C compiler and POSIX.

## Features

- C11 with POSIX dependencies (mmap, clock_gettime), no external libraries
- Runtime kernel dispatch: scalar baseline, AVX2 (x86_64), NEON (AArch64), Metal GPU (Apple Silicon)
- mmap-based model loading (zero-copy, OS pages in weights on demand)
- T-MAC lookup-table kernel for ternary bitlinear
- Built-in GPT-2 byte-pair tokenizer
- Persistent spin-wait thread pool with `--threads N` (auto-detects P-cores on Apple Silicon)
- Per-stage profiling (`--profile` flag)
- GPU offload for batch prefill (`--gpu` flag, Apple Metal)

## Performance

Reference numbers from the bitmamba.cpp C++ engine on commodity CPUs
(no NPU, no GPU):

| Model | Parameters | Precision | Peak RAM | Speed (i3-12100F) |
|-------|------------|-----------|----------|--------------------|
| Small | 255M       | 1.58-bit  | 252 MB   | 146.16 tok/s       |
| Medium | 1B        | 1.58-bit  | 621 MB   | 52.86 tok/s        |

Average human reading speed is ~5-7 tok/s. The 1B model runs at roughly
10x real-time conversational pace on entry-level hardware.

The T-MAC LUT kernel replaces weight-activation multiplies with table
lookups: ternary weights index precomputed partial sums in L1/L2 cache,
keeping the critical path off the memory bus.

## Building and Running

Requires a C compiler (C11 or later), POSIX headers (`mmap`, `clock_gettime`),
and Python 3 for tokenizer generation (`make download`).
ISA-specific kernels (AVX2, NEON, Metal) are auto-detected at build time.

```shell
make              # Build ./bitmamba
make clean        # Remove build artifacts
```

### Downloading Weights

```shell
make download
```

This downloads the BitMamba-2 1B model weights (`bitmamba_1b.bin`),
fetches the GPT-2 vocabulary, and generates `tokenizer.bin`.
The model and tokenizer binaries are required before running inference in tokenizer mode.

For the 255M model, download manually:

```shell
curl -L -o bitmamba_255m.bin \
    https://huggingface.co/Zhayr1/BitMamba-2-0.25B/resolve/main/bitmamba_cpp/bitmamba_255m.bin
```

### Running Inference

```shell
./bitmamba [--profile] [--gpu] [--threads N] <model.bin> <input> <mode> \
    [temp] [penalty] [min_p] [top_p] [top_k] [max_tokens] [output_mode]
```

Parameters:
- `model.bin` - path to model weights
- `input` - input text (tokenizer mode) or space-separated token IDs
  (raw mode)
- `mode` - `tokenizer` (text in/out) or `raw` (token IDs)
- `temp` - temperature (default: 0.8)
- `penalty` - repetition penalty (default: 1.15)
- `min_p` - Min-P sampling threshold (default: 0.05)
- `top_p` - Top-P / nucleus sampling (default: 0.90)
- `top_k` - Top-K sampling (default: 40)
- `max_tokens` - maximum tokens to generate (default: 400)
- `output_mode` - `bench` (default, shows stats) or `clean` (output
  only)

### Examples

```shell
# Text input/output
./bitmamba bitmamba_1b.bin "Hello, I am" tokenizer 0.7 1.1

# Raw token IDs (greedy)
./bitmamba bitmamba_1b.bin "15496 11 314 716" raw 0.0

# Per-stage profiling
./bitmamba --profile bitmamba_1b.bin "Hello, I am" tokenizer

# Metal GPU (Apple Silicon, auto-fallback to CPU for short sequences)
./bitmamba --gpu bitmamba_1b.bin "15496 11 314 716" raw 0.0
```

## Documentation

The [docs/](docs/) directory contains detailed technical documentation:

- [Background](docs/background.md): Edge AI memory wall, Transformer
  bottlenecks, and state space model foundations
- [Architecture](docs/architecture.md): Ternary quantization, BitLinear
  mechanics, hybrid precision strategy, and comparative landscape
- [Performance](docs/performance.md): Scaling validation, CPU inference
  economics, T-MAC optimization, and energy efficiency

## Project Structure

```
src/
  bitmamba.h         Shared types, constants, API declarations
  dispatch.h/.c      Runtime kernel dispatch table
  main.c             CLI entry point, generation loop
  model.c            mmap model loading, forward pass, sampling
  kernel.c           Scalar T-MAC bitlinear, RMS norm, activations
  block.c            Mamba-2 block (SSM + conv1d) forward pass
  quantization.c     Weight unpacking LUTs, T-MAC mask tables
  tokenizer.c        GPT-2 byte-pair tokenizer (encode/decode)
  threadpool.h/.c    Persistent spin-wait thread pool (fork-join)
  utils.c            Memory, timing, PRNG, profiling utilities
  arch/avx2.c        AVX2-optimized kernels (x86_64)
  arch/neon.c        NEON-optimized kernels (AArch64)
  metal/             Metal GPU compute shaders (Apple Silicon)
Makefile             Build system (auto ISA detection)
scripts/
  export-bin.py      JAX checkpoint to .bin converter
  export-tokenizer.py  GPT-2 tokenizer export (stdlib only)
  generate-golden.py   Golden vector generation
```

## Model Format

Binary format with magic `0x42495432` ("BIT2").
The header contains magic, vocab_size, d_model, n_layers, and n_heads,
followed by tensors.
Float32 tensors store type, ndim, shape, and data.
BitNet tensors store type, rows, cols, scale, and packed 2-bit data.

## Edge Deployment Notes

For Raspberry Pi 5 (Cortex-A76) and similar ARMv8.2-A targets:
- The NEON kernel (`src/arch/neon.c`) uses the signed dot-product extension
  (SDOT via `vdotq_s32`) available on Cortex-A76+, with a `vmull_s8`
  fallback for older ARMv8-A cores. The build auto-detects via `-mcpu=native`.
- Ternary weights for the 1B model pack to ~250 MB (1B x 2 bits / 8).
  The `.bin` file is larger because embedding and norm tensors are float32,
  but total runtime footprint stays well under 1 GB.
- The T-MAC LUT kernel indexes precomputed partial sums in L1 cache instead
  of multiplying by {-1, 0, +1}, sidestepping the lack of native ternary
  instructions. Reference T-MAC benchmarks on RPi 5 achieve 11 tok/s for
  a 3B model.
- Energy consumption for ternary inference can be as low as 0.028 J/token,
  up to 70% less than 4-bit Transformer inference. This is critical for
  battery-operated and thermally constrained deployments.
- For comparison, a 4-bit Qwen-2.5-1.5B on the same hardware achieves
  ~40 tok/s but requires ~1.1 GB. The 1B BitMamba-2 uses ~621 MB peak
  RAM (roughly half) with ternary weight storage at ~1/4 the bits per
  parameter.

## Acknowledgments

This project draws from [bitmamba.cpp](https://github.com/Zhayr1/bitmamba.cpp) by
[Jesus Salazar](https://github.com/Zhayr1), a C++ inference implementation using AVX2 intrinsics and OpenMP threading.
bitmamba.c rewrites the engine in plain C11 with POSIX-only dependencies,
eliminating the C++ runtime and OpenMP to reduce system utilization and widen portability to bare-metal and
resource-constrained targets.

## License

`bitmamba.c` is freely redistributable under the MIT License.
