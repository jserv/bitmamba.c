#!/usr/bin/env python3

"""
Export BitMamba model from JAX checkpoint to packed binary format.

Dependencies: numpy, msgpack (no PyTorch required).

Usage:
    pip install numpy msgpack
    python3 scripts/export-bin.py \
        --version 1b --ckpt_path <checkpoint.msgpack> --output_name bitmamba_1b.bin
"""

import argparse
import os
import re
import struct
import sys

import msgpack
import numpy as np

CONFIG_1B = {"vocab_size": 50257, "d_model": 2048, "n_layers": 32, "n_heads": 32}
CONFIG_255M = {"vocab_size": 50257, "d_model": 1024, "n_heads": 16, "n_layers": 24}


def jax_to_numpy(param_name, jax_array):
    """Convert JAX array to numpy with layout transposition."""
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


def pack_ternary_weights(data, name):
    """Pack ternary weights (-1, 0, 1) into 2-bit format, 4 values per byte."""
    rows, cols = data.shape
    if cols % 4 != 0:
        raise ValueError(f"Fatal: {name} cols ({cols}) must be divisible by 4.")
    flat = (data.flatten() + 1).astype(np.uint8)
    r = flat.reshape(-1, 4)
    packed = r[:, 0] | (r[:, 1] << 2) | (r[:, 2] << 4) | (r[:, 3] << 6)
    return packed.astype(np.uint8)


def export_packed(weights, config, output_path):
    """Write binary model file from weight dict."""
    keys = ["embed.weight"]
    for i in range(config["n_layers"]):
        b = f"layers.{i}."
        keys.extend(
            [
                b + "in_proj.norm.weight",
                b + "in_proj.weight",
                b + "conv1d.weight",
                b + "conv1d.bias",
                b + "dt_bias",
                b + "A_log",
                b + "D",
                b + "out_proj.norm.weight",
                b + "out_proj.weight",
            ]
        )
    keys.extend(["norm_f.weight", "lm_head.norm.weight", "lm_head.weight"])

    print(f"Exporting packed (2-bit) to {output_path}...")
    total_params = 0
    compressed_bytes = 0
    n_keys = len(keys)

    with open(output_path, "wb") as f:
        f.write(
            struct.pack(
                "iiiii",
                0x42495432,
                config["vocab_size"],
                config["d_model"],
                config["n_layers"],
                config["n_heads"],
            )
        )

        for idx, name in enumerate(keys):
            if name not in weights:
                print(f"\nFatal: {name} not found in checkpoint.")
                print(
                    "C loader reads tensors sequentially; skipping "
                    "would corrupt the binary stream."
                )
                sys.exit(1)

            data = weights[name]
            if np.isnan(data).any():
                print(f"\nCRITICAL: NaNs in {name}. Model is corrupt.")
                sys.exit(1)

            is_bitlinear = (
                "weight" in name
                and ("in_proj" in name or "out_proj" in name or "lm_head" in name)
                and "norm" not in name
                and "embed" not in name
            )

            if is_bitlinear:
                mean_abs = np.mean(np.abs(data))
                if mean_abs == 0:
                    print(f"\nWarning: Dead layer (all zeros): {name}")
                    scale = 1.0
                else:
                    scale = 1.0 / mean_abs

                quant = np.clip(np.round(data * scale), -1, 1).astype(np.int8)
                if np.all(quant == 0):
                    print(f"\nWarning: {name} collapsed to zeros.")

                packed = pack_ternary_weights(quant, name)
                f.write(struct.pack("i", 2))
                f.write(struct.pack("ii", data.shape[0], data.shape[1]))
                f.write(struct.pack("f", float(scale)))
                f.write(packed.tobytes())
                compressed_bytes += packed.nbytes
            else:
                f.write(struct.pack("i", 0))
                f.write(struct.pack("i", len(data.shape)))
                for d in data.shape:
                    f.write(struct.pack("i", d))
                f.write(data.tobytes())
                compressed_bytes += data.nbytes

            total_params += data.size
            print(f"  [{idx + 1}/{n_keys}] {name}", end="\r")

    print(f"\nExport completed.")
    print(f"  Final size: {compressed_bytes / 1024 / 1024:.2f} MB")
    print(f"  Total parameters: {total_params / 1e9:.2f} B")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export BitMamba model to binary format."
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["1b", "250m"],
        required=True,
        help="Model version",
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to JAX checkpoint (msgpack)"
    )
    parser.add_argument(
        "--output_name", type=str, required=True, help="Output binary filename"
    )
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        print(f"Not found: {args.ckpt_path}")
        sys.exit(1)

    config = CONFIG_1B if args.version == "1b" else CONFIG_255M
    weights = load_checkpoint(args.ckpt_path)
    export_packed(weights, config, args.output_name)
    print(f"\nDone. Use: ./bitmamba {args.output_name}")
