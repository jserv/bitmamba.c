#!/usr/bin/env python3

"""
Generate golden vectors for BitMamba correctness testing.

Dependencies: numpy, msgpack (no PyTorch required).

Usage:
    pip install numpy msgpack
    python3 scripts/generate-golden.py --ckpt_path <checkpoint> --output golden.bin
    python3 scripts/generate-golden.py --ckpt_path <checkpoint> --tokens "15496 11 314 716"

Output format (binary):
    - Magic: 0x474F4C44 ("GOLD")
    - Version: 1
    - Seed: uint64
    - d_model: int32
    - n_layers: int32
    - n_tokens: int32 (total tokens processed)
    - vocab_size: int32
    - prompt_len: int32
    - prompt_tokens: int32[prompt_len]
    - For each token processed:
        - layer_outputs: float32[n_layers][d_model] (hidden state after each layer)
        - final_hidden: float32[d_model] (after final norm)
        - logits: float32[vocab_size]
"""

import argparse
import os
import re
import struct
import sys

import msgpack
import numpy as np

MAGIC = 0x474F4C44  # "GOLD"
VERSION = 1

CONFIG_1B = {"vocab_size": 50257, "d_model": 2048, "n_layers": 32, "n_heads": 32}
CONFIG_255M = {"vocab_size": 50257, "d_model": 1024, "n_heads": 16, "n_layers": 24}

# Known prompt encodings (avoids transformers dependency)
KNOWN_PROMPTS = {
    "Hello, I am": [15496, 11, 314, 716],
    "The": [464],
}


# ============= NumPy math primitives =============


def rms_norm(x, weight, eps=1e-6):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


def silu(x):
    return x / (1.0 + np.exp(np.clip(-x, -88.0, 88.0)))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -88.0, 88.0))))


def bitlinear_forward(x, weight, norm_weight):
    """BitLinear: RMS norm -> quantize activations -> quantize weights -> matmul."""
    x = x.astype(np.float32)
    x_norm = rms_norm(x, norm_weight)

    abs_max = np.max(np.abs(x_norm), axis=-1, keepdims=True)
    abs_max = np.maximum(abs_max, 1e-5)
    scale_x = 127.0 / abs_max
    x_q = np.clip(np.round(x_norm * scale_x), -128, 127) / scale_x

    mean_abs = np.mean(np.abs(weight))
    if mean_abs == 0:
        scale_w = 1.0
    else:
        scale_w = 1.0 / mean_abs
    w_q = np.clip(np.round(weight * scale_w), -1.0, 1.0) / scale_w

    return x_q @ w_q.T


# ============= Mamba-2 block step =============


def mamba_block_step(x, w, conv_state, ssm_state):
    """Single-token Mamba-2 block forward step in pure numpy.

    Args:
        x: [d_model] input hidden state
        w: dict of layer weights (keys without layer prefix)
        conv_state: [d_inner, d_conv-1]
        ssm_state: [n_heads, head_dim]
    Returns:
        output, new_conv_state, new_ssm_state
    """
    d_inner = w["D"].shape[0]
    n_heads = w["dt_bias"].shape[0]
    head_dim = d_inner // n_heads

    # in_proj (BitLinear)
    proj = bitlinear_forward(
        x.reshape(1, -1), w["in_proj.weight"], w["in_proj.norm.weight"]
    ).flatten()

    # Split: z, x_in, B, C, dt
    off = 0
    z = proj[off : off + d_inner]
    off += d_inner
    x_in = proj[off : off + d_inner]
    off += d_inner
    B_val = proj[off : off + n_heads]
    off += n_heads
    C_val = proj[off : off + n_heads]
    off += n_heads
    dt = proj[off : off + n_heads]

    # Depthwise conv1d
    x_col = x_in.reshape(d_inner, 1)
    new_conv_state = np.concatenate([conv_state[:, 1:], x_col], axis=1)
    conv_window = np.concatenate([conv_state, x_col], axis=1)
    conv_w = w["conv1d.weight"].reshape(d_inner, -1)
    x_conv = np.sum(conv_window * conv_w, axis=1) + w["conv1d.bias"]

    x_t = silu(x_conv)

    # SSM step
    x_r = x_t.reshape(n_heads, head_dim)
    dt = softplus(dt + w["dt_bias"]).reshape(n_heads, 1)
    A = -np.exp(w["A_log"]).reshape(n_heads, 1)
    decay = np.exp(A * dt)
    u_ssm = x_r * B_val.reshape(n_heads, 1) * dt
    h_new = ssm_state * decay + u_ssm
    y = (h_new * C_val.reshape(n_heads, 1)).reshape(d_inner)

    # Combine and output
    y = y + x_t * w["D"]
    y = y * silu(z)

    out = bitlinear_forward(
        y.reshape(1, -1), w["out_proj.weight"], w["out_proj.norm.weight"]
    ).flatten()

    return out, new_conv_state, h_new


# ============= Checkpoint loading =============


def jax_to_numpy(param_name, jax_array):
    arr = np.array(jax_array, dtype=np.float32)
    if "kernel" in param_name and arr.ndim == 2:
        return arr.T
    if "conv1d" in param_name and "kernel" in param_name and arr.ndim == 3:
        return arr.transpose(2, 1, 0)
    return arr


def load_checkpoint(path):
    """Load JAX msgpack checkpoint into flat {name: ndarray} dict."""
    print(f"Loading JAX msgpack: {path} ...")
    with open(path, "rb") as f:
        state_dict = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    if "params" in state_dict:
        state_dict = state_dict["params"]

    print("Converting weights...")
    weights = {}
    layer_pat = re.compile(r".*BitMambaBlock_(\d+)$")

    for key, val in state_dict.items():
        if "Embed_0" in key:
            weights["embed.weight"] = jax_to_numpy("embed", val["embedding"])
        elif "RMSNorm_0" in key:
            weights["norm_f.weight"] = jax_to_numpy("norm", val["scale"])
        elif key == "BitLinear_0":
            weights["lm_head.weight"] = jax_to_numpy("kernel", val["kernel"])
            if "RMSNorm_0" in val:
                weights["lm_head.norm.weight"] = jax_to_numpy(
                    "norm", val["RMSNorm_0"]["scale"]
                )

        m = layer_pat.match(key)
        if m:
            base = f"layers.{m.group(1)}."
            if "BitLinear_0" in val:
                weights[base + "in_proj.weight"] = jax_to_numpy(
                    "kernel", val["BitLinear_0"]["kernel"]
                )
                if "RMSNorm_0" in val["BitLinear_0"]:
                    weights[base + "in_proj.norm.weight"] = jax_to_numpy(
                        "norm", val["BitLinear_0"]["RMSNorm_0"]["scale"]
                    )
            if "Conv_0" in val:
                if "bias" in val["Conv_0"]:
                    weights[base + "conv1d.bias"] = jax_to_numpy(
                        "bias", val["Conv_0"]["bias"]
                    )
                weights[base + "conv1d.weight"] = jax_to_numpy(
                    "conv1d.kernel", val["Conv_0"]["kernel"]
                )
            if "BitLinear_1" in val:
                weights[base + "out_proj.weight"] = jax_to_numpy(
                    "kernel", val["BitLinear_1"]["kernel"]
                )
                if "RMSNorm_0" in val["BitLinear_1"]:
                    weights[base + "out_proj.norm.weight"] = jax_to_numpy(
                        "norm", val["BitLinear_1"]["RMSNorm_0"]["scale"]
                    )
            if "dt_bias" in val:
                weights[base + "dt_bias"] = jax_to_numpy("p", val["dt_bias"])
            if "A_log" in val:
                weights[base + "A_log"] = jax_to_numpy("p", val["A_log"])
            if "D" in val:
                weights[base + "D"] = jax_to_numpy("p", val["D"])

    print(f"Loaded {len(weights)} tensors.")
    return weights


# ============= Model forward =============


def model_step(token_id, weights, config, conv_states, ssm_states):
    """Single-token forward pass. Returns per-layer outputs, final hidden, logits."""
    n_layers = config["n_layers"]
    d_model = config["d_model"]
    d_inner = d_model * 2
    n_heads = config["n_heads"]
    head_dim = d_inner // n_heads

    x = weights["embed.weight"][token_id].copy()

    layer_outputs = []
    new_conv = []
    new_ssm = []

    for i in range(n_layers):
        prefix = f"layers.{i}."
        plen = len(prefix)
        lw = {k[plen:]: v for k, v in weights.items() if k.startswith(prefix)}

        residual = x.copy()
        out, nc, ns = mamba_block_step(x, lw, conv_states[i], ssm_states[i])
        x = residual + out

        layer_outputs.append(x.copy())
        new_conv.append(nc)
        new_ssm.append(ns)

    final_hidden = rms_norm(x, weights["norm_f.weight"])
    logits = bitlinear_forward(
        final_hidden.reshape(1, -1),
        weights["lm_head.weight"],
        weights["lm_head.norm.weight"],
    ).flatten()

    return layer_outputs, final_hidden, logits, new_conv, new_ssm


def init_states(config):
    n_layers = config["n_layers"]
    d_inner = config["d_model"] * 2
    n_heads = config["n_heads"]
    head_dim = d_inner // n_heads
    d_conv = 4
    conv = [np.zeros((d_inner, d_conv - 1), dtype=np.float32) for _ in range(n_layers)]
    ssm = [np.zeros((n_heads, head_dim), dtype=np.float32) for _ in range(n_layers)]
    return conv, ssm


# ============= Binary output =============


def write_golden_file(
    path, seed, d_model, n_layers, vocab_size, prompt_tokens, token_vectors
):
    n_tokens = len(token_vectors)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", seed))
        f.write(struct.pack("<i", d_model))
        f.write(struct.pack("<i", n_layers))
        f.write(struct.pack("<i", n_tokens))
        f.write(struct.pack("<i", vocab_size))
        f.write(struct.pack("<i", len(prompt_tokens)))
        for tok in prompt_tokens:
            f.write(struct.pack("<i", tok))
        for vec in token_vectors:
            for lo in vec["layer_outputs"]:
                f.write(lo.astype(np.float32).tobytes())
            f.write(vec["final_hidden"].astype(np.float32).tobytes())
            f.write(vec["logits"].astype(np.float32).tobytes())

    print(f"Written {n_tokens} token vectors to {path}")
    print(f"  d_model={d_model}, n_layers={n_layers}, vocab_size={vocab_size}")
    print(f"  File size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")


# ============= Main =============


def main():
    parser = argparse.ArgumentParser(description="Generate golden vectors for BitMamba")
    parser.add_argument("--version", type=str, choices=["1b", "250m"], default="1b")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="golden.bin")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, I am",
        help="Text prompt (must be in known prompts list)",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Space-separated token IDs (bypasses prompt lookup)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=5)
    args = parser.parse_args()

    np.random.seed(args.seed)

    config = CONFIG_1B if args.version == "1b" else CONFIG_255M
    weights = load_checkpoint(args.ckpt_path)

    if args.tokens:
        prompt_tokens = [int(t) for t in args.tokens.split()]
    elif args.prompt in KNOWN_PROMPTS:
        prompt_tokens = KNOWN_PROMPTS[args.prompt]
    else:
        print(f"Error: unknown prompt '{args.prompt}'.", file=sys.stderr)
        print(f"Known prompts: {list(KNOWN_PROMPTS.keys())}", file=sys.stderr)
        print("Use --tokens to pass raw token IDs.", file=sys.stderr)
        sys.exit(1)

    print(f"Seed: {args.seed}")
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

    conv_states, ssm_states = init_states(config)
    token_vectors = []
    all_tokens = list(prompt_tokens)

    print("Processing prompt tokens...")
    for i, tok in enumerate(prompt_tokens):
        layer_outs, final_h, logits, conv_states, ssm_states = model_step(
            tok, weights, config, conv_states, ssm_states
        )
        token_vectors.append(
            {
                "layer_outputs": layer_outs,
                "final_hidden": final_h,
                "logits": logits,
            }
        )
        if i == len(prompt_tokens) - 1:
            next_token = int(np.argmax(logits))

    print(f"Generating {args.max_tokens} tokens...")
    current_token = next_token
    for _ in range(args.max_tokens):
        all_tokens.append(current_token)
        layer_outs, final_h, logits, conv_states, ssm_states = model_step(
            current_token, weights, config, conv_states, ssm_states
        )
        token_vectors.append(
            {
                "layer_outputs": layer_outs,
                "final_hidden": final_h,
                "logits": logits,
            }
        )
        current_token = int(np.argmax(logits))

    print(f"All tokens ({len(all_tokens)}): {all_tokens}")

    write_golden_file(
        args.output,
        seed=args.seed,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        vocab_size=config["vocab_size"],
        prompt_tokens=prompt_tokens,
        token_vectors=token_vectors,
    )
    print("\nGolden vector generation complete!")


if __name__ == "__main__":
    main()
