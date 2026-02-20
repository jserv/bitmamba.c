# Performance: Scaling Laws, Inference Economics, and Energy

## Training and Scaling

### Dataset Curriculum

The training data was curated to counterbalance the information bottleneck of
1.58-bit weights:

- FineWeb-Edu (60%): high-quality educational content for linguistic coherence
  and reasoning.
- Cosmopedia (20%): synthetic textbooks providing structured encyclopedic
  knowledge.
- The Stack-Dedup (20%): deduplicated Python source code for algorithmic
  sequencing and logical structure.

### Compute Allocation

| Variant | Tokens   | Batch Strategy         | Throughput       |
|---------|----------|------------------------|------------------|
| 255M    | 400B     | Manual gradient accum.  | ~380K tok/s      |
| 1B      | 150B     | Data-parallel (8 cores) | ~99K tok/s       |

Both trained on Google Cloud TPU v6e-8.

### Ternary Scaling Validation

Despite seeing less than half the training tokens (150B vs. 400B), the 1B model
established overwhelming superiority over the 255M baseline:

- WikiText-2 perplexity: 51.69 (255M) -> 29.62 (1B). Breaking the
  30-perplexity threshold with only 150B tokens confirms that expanding
  parameter count provides sufficient combinatoric pathways for ternary weights
  to overcome quantization noise.
- Cross-entropy loss stabilized at ~2.656 without catastrophic divergence
  spikes, validating the RMSNorm activation stabilization.

### The MoE Routing Failure (Ablation)

Sparse Upcycling was attempted: a 4x255M Mixture-of-Experts structure (~1B
parameters) initialized from the 255M dense checkpoint and trained for 50B
additional tokens.

| Benchmark         | 255M Dense* | 1B Sparse MoE | 1B Dense |
|-------------------|-------------|----------------|----------|
| ARC-Easy          | 55.51%      | 56.10%         | 63.30%   |
| PIQA              | 64.42%      | 66.87%         | 68.77%   |
| HellaSwag         | 35.22%      | 39.20%         | 45.59%   |
| BoolQ             | 42.54%      | 59.30%         | 62.35%   |
| WikiText-2 (PPL)  | 51.69       | 70.00          | 29.62    |

*255M Dense scores are from an intermediate ablation checkpoint and differ
from the primary evaluation in [architecture.md](architecture.md) (e.g.,
BoolQ 42.54% here vs. 59.30% there). Lower perplexity is better.

The MoE model gained modestly on pattern completion (+2.4% PIQA, +4.0%
HellaSwag) but suffered severe language modeling regression: WikiText-2
perplexity degraded from 51.69 to 70.00, indicating that the router's
token-routing instability destroyed overall fluency.

Root cause: MoE routing relies on a sensitive softmax distribution to direct
tokens to expert layers. Ternary quantization noise disrupts the router's
ability to establish sharp decision boundaries in embedding space, causing
chronic token misrouting and chaotic state transitions.

Implication: dense scaling of ternary SSMs is robust. Unlocking MoE efficiency
for ternary models requires novel routing mechanisms, likely maintaining the
router network entirely in high-precision bfloat16.

## CPU Inference Economics

### Algorithmic Efficiency vs. Hardware Acceleration

The critical comparison: Meta's Llama 3.2 1B quantized to 4-bit Q4_K_M GGUF,
running on an AMD Ryzen AI 9 HX 375 (CPU-side AVX-512; the chip also has a
50-TOPS NPU, but standard llama.cpp uses CPU unless a vendor NPU backend is
configured), achieves ~50.7 tok/s. Standard llama.cpp CPU-only inference on
high-end desktops (i7-14700HX, DDR5-5600) yields 11-30 tok/s for small models.

BitMamba-2 1B achieves 52.86 tok/s on an aging i3-12100F with DDR4, no NPU.

While the hardware differs across these data points, the trend suggests that
algorithmic efficiency, by eliminating quadratic attention cost via SSM and
circumventing floating-point matmul via ternary weights, can narrow the gap
with dedicated AI hardware acceleration. If confirmed under controlled
conditions, this would broaden access to generative AI on legacy and low-cost
hardware.

### T-MAC Lookup Table Optimization

The CPU bottleneck for 1.58-bit models is memory bandwidth, not compute. Cores
frequently idle waiting for weights from DRAM. T-MAC addresses this with
lookup-table-based mixed-precision GEMM (mpGEMM):

Because ternary weights belong to a restricted set $\{-1, 0, 1\}$, all possible
products against INT8 activations can be precomputed and stored in L1/L2 cache.
During inference, the CPU executes bit-wise lookups instead of multiplication
instructions.

T-MAC results on reference hardware:
- 3B BitNet b1.58 on Apple M2 Ultra: 30 tok/s (1 core), 71 tok/s (8 cores).
- Raspberry Pi 5: 11 tok/s with LUT optimization.

The bitmamba.c implementation uses T-MAC LUT kernels for its ternary bitlinear
operations, with runtime dispatch to scalar, AVX2, or NEON variants.

## Energy Efficiency

Three compounding mechanisms reduce power consumption:

1. Zero KV cache: the Mamba-2 SSM maintains a fixed-size hidden state.
   Memory footprint stays constant regardless of context length, avoiding
   power-intensive memory swapping that throttles Transformers on edge devices.

2. Ultra-low data transport: reducing projection weights from 16 bits to
   1.58 bits cuts memory bus traffic by a factor of ~10 per inference cycle.

3. Multiplier-less arithmetic: replacing floating-point multiply instructions
   with add/subtract operations lowers energy per token. Parallel
   BitNet b1.58 benchmarks show energy consumption as low as 0.028 J/token,
   representing up to 70% reduction versus 4-bit Transformer inference via
   llama.cpp.

For battery-operated, thermally constrained edge devices (agricultural
robotics, drone telemetry, IoT sensor networks), the energy-to-performance
ratio is the ultimate feasibility arbiter. BitMamba-2's ability to minimize data
transport while eliminating complex arithmetic makes it a strong candidate for
offline-first intelligence systems.
