# Architecture: Ternary Quantization and Hybrid Precision

## From Binary to 1.58-bit Ternary

The original BitNet proved that LLMs could be trained with strict 1-bit binary
weights $\{-1, 1\}$. However, binary models consistently underperformed
full-precision counterparts. BitNet b1.58 introduced ternary quantization:
weights assume three discrete values $\{-1, 0, 1\}$, requiring
$\log_2(3) \approx 1.58$ bits per weight.

The inclusion of zero is not merely a resolution increase. It fundamentally
alters the network's processing capability. In a binary system, every neural
connection is forced to either add or subtract. The zero value grants explicit
feature filtering: the model can organically prune noisy or irrelevant
connections, reducing the interference that plagues ultra-compressed models.

## BitLinear Layer Mechanics

In BitMamba-2, standard linear projections are replaced with BitLinear layers.
For a weight matrix $W$ of shape $(d_{in}, d_{out})$:

1. Compute the absmean scaling factor:

$$\gamma = \frac{1}{d_{in} \times d_{out}} \sum |W_{ij}| + \epsilon$$

2. Quantize to ternary:

$$W_{ternary} = \text{Clip}(\text{Round}(W / \gamma),\ -1,\ 1)$$

3. During backpropagation, the Straight-Through Estimator (STE) bypasses the
   non-differentiable rounding, allowing gradients to flow to high-precision
   master weights maintained for optimizer updates.

For input activations, BitMamba-2 applies RMSNorm to bound variance, then
dynamically quantizes to INT8 using absolute maximum scaling:

$$s_x = 127 \;/\; \max(|x|)$$

$$x_{quantized} = \text{Clip}(\text{Round}(x \cdot s_x),\ -127,\ 127)$$

## Multiply-Free Arithmetic

Because weights are strictly $\{-1, 0, 1\}$, weight-activation multiplication
reduces to addition and subtraction in the ternary projection layers:

- Weight = 1: add the activation value.
- Weight = -1: subtract the activation value.
- Weight = 0: skip entirely.

This dramatically reduces latency and thermal footprint, both critical for edge
deployment.

## The Hybrid Precision Strategy

Fusing extreme quantization with a recurrent SSM is delicate. Transformers
distribute redundancy across dozens of parallel attention heads and can absorb
quantization noise. SSMs, by contrast, rely on sequential propagation of a
single hidden state. Heavy quantization on recurrent variables destroys
long-range dependency tracking.

BitMamba-2 solves this with a deliberate hybrid precision framework:

- Ternary (1.58-bit): the dense projection matrices (input/output linear
  layers), which constitute over 90% of total parameters and dominate the
  memory footprint. These are compressed via BitLinear.
- High-precision: the 1D convolutional layer, the recurrent state variables
  (z, B, C), and the step size (dt). These form a precision sanctuary that
  shields the model's chronological memory and logical routing from
  quantization noise. The original training uses bfloat16 for these
  components; bitmamba.c promotes them to float32.

The output of ternary matrix multiplications is dequantized to float32
immediately before entering the state space operations. This compromise enables
scaling to 1 billion parameters while preserving complex reasoning.

## Model Specifications

| Specification     | BitMamba-2 Small | BitMamba-2 Medium |
|-------------------|------------------|-------------------|
| Total Parameters  | 255 Million      | 1 Billion         |
| Layers            | 24               | 32                |
| SSM Heads         | 16               | 32                |
| Model Dimension   | 1024             | 2048              |

Both variants were implemented in JAX/Flax and trained on Google Cloud TPU v6e-8
infrastructure.

## Comparative Landscape

### Bi-Mamba (1-bit Binary)

Bi-Mamba reduces Mamba weights to strict binary $\{-1, 1\}$ across 780M, 1.3B, and
2.7B configurations. It relies on autoregressive knowledge distillation from a
full-precision teacher, involving supervised fine-tuning and Direct Preference
Optimization (DPO). The strict binary constraint creates a performance ceiling:
without the zero value, Bi-Mamba cannot perform explicit feature filtering,
introducing chronic interference in tasks requiring selective forgetting. The
costly distillation pipeline also makes training domain-specific variants
prohibitive.

### Slender-Mamba (Head-to-Toe Ternary)

Slender-Mamba pushes 1.58-bit quantization beyond linear layers to encompass
embedding and output projection layers. This achieves ~90% reduction in
parameter-bits (versus ~48% for standard BitNet). However, forcing the
embedding layer into ternary values severely distorts geometric relationships
between token representations. At 170M parameters, Slender-Mamba scores only
38.60% on ARC-Easy and 55.60% on PIQA, demonstrating that shielding
embeddings, heads, and internal SSM recurrences from quantization is necessary
for semantic integrity.

### Zero-Shot Benchmark Comparison

| Architecture          | Size | Quantization          | ARC-Easy | PIQA   | HellaSwag | BoolQ  |
|-----------------------|------|-----------------------|----------|--------|-----------|--------|
| Slender-Mamba         | 170M | 1.58-bit (full)       | 38.60%   | 55.60% | 27.00%    | 58.50% |
| BitMamba-2 Baseline   | 255M | 1.58-bit (hybrid)     | 55.51%   | 64.42% | 35.22%    | 59.30% |
| Bi-Mamba              | 1.3B | 1.00-bit (binary)     | 43.90%   | 69.20% | 43.10%    | 62.00% |
| BitMamba-2 Scaled     | 1B   | 1.58-bit (hybrid)     | 63.30%   | 68.77% | 45.59%    | 62.35% |
| Llama 3.2 (4-bit)*    | 1B   | 4-bit (PTQ)           | 64.52%   | 75.46% | 48.56%    | 50.24% |

*Llama 3.2 1B metrics reflect community evaluations of 4-bit quantized
versions; prompting methodology may introduce minor variance.

Key findings:

1. BitMamba-2 1B decisively outperforms the larger 1.3B Bi-Mamba on ARC-Easy
   (63.30% vs. 43.90%) and HellaSwag (45.59% vs. 43.10%), validating ternary
   over binary.
2. BitMamba-2 1B trails Llama 3.2 1B by narrow margins on most benchmarks
   while defeating it on BoolQ (62.35% vs. 50.24%), proving linear-time SSMs
   do not inherently sacrifice reasoning under aggressive quantization-aware
   training.
