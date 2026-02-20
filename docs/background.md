# Background: Edge AI and State Space Models

## The Memory Wall

Edge AI inference is memory-bandwidth-bound. On commodity hardware, the
bottleneck is not arithmetic throughput but the rate at which weight data can be
moved from DRAM into the processor's register file. A 4-bit-quantized 1.5B
Transformer on a Raspberry Pi 5 (Cortex-A76, ~18 GB/s LPDDR4X) already
saturates the memory bus at ~30-40 tok/s. Scaling to 3.8B parameters drops
throughput to ~5 tok/s because the CPU stalls waiting on DRAM.

Cloud-centric deployment sidesteps this with high-bandwidth memory (HBM) and
dedicated NPUs, but introduces recurring compute cost, network-dependent
latency, energy overhead, and privacy exposure. For autonomous vehicles,
embedded robotics, agricultural drones, and IoT sensor networks, the mandate is
"offline-first" intelligence: deterministic execution on standard CPUs and ARM
SoCs without continuous connectivity or specialized silicon.

Standard post-training quantization (PTQ) techniques, which compress FP16
weights to INT4 via GPTQ or AWQ, reduce file size but typically still require
higher-precision accumulation during the forward pass. Below 4 bits they
frequently trigger catastrophic degradation in reasoning quality. The industry
therefore needs architectures designed from the ground up for low-bit regimes.

## The Transformer Bottleneck

The self-attention mechanism computes pairwise relevance scores across every
token in a sequence, yielding $O(N^2)$ complexity for full-sequence processing
(training and prefill). During autoregressive decode, each new token attends to
all prior tokens via the KV cache, making per-token cost $O(N)$ in context
length. As the sequence grows, the KV cache inflates proportionally, consuming
RAM and forcing memory swaps that destroy inference speed on constrained
devices.

## Structured State Space Models

State Space Models (SSMs) treat sequence modeling as a discretized
continuous-time differential equation. The original Mamba architecture
introduced a selective scan mechanism that dynamically filters information,
remembering critical context while forgetting irrelevant data, achieving $O(N)$
complexity.

The key property: SSMs compress all historical sequence data into a fixed-size
hidden state vector. Each new token updates this state in constant time and
constant memory, regardless of context length. No KV cache. No memory growth.

## Mamba-2 and Structured State Space Duality

BitMamba-2 builds on the Mamba-2 framework, which introduced Structured State
Space Duality (SSD), a theoretical bridge between structured SSMs and causal
linear attention. The practical implications:

- Training phase: SSD enables use of optimized matrix multiplication kernels
  (analogous to FlashAttention), achieving hardware utilization comparable to
  Transformers.
- Inference phase: the model reverts to linear-time, constant-memory recurrent
  state updates.
- State dimension: Mamba-2 permits state sizes of $N=64$ to $N=256$ (versus
  $N=16$ in Mamba-1), significantly increasing expressiveness for long-range
  dependencies.
- Stability: additional normalization layers within Mamba-2 improve training
  stability at scale, which proves vital when introducing ternary quantization
  noise.
