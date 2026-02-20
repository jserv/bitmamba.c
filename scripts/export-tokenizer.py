#!/usr/bin/env python3

"""
Export GPT-2 vocabulary to binary format for C inference.

Downloads vocab.json from HuggingFace (a static ~1MB file) and converts it
to tokenizer.bin. No external dependencies -- stdlib only.

Usage:
    python3 scripts/export-tokenizer.py [--vocab vocab.json] [--output tokenizer.bin]

If vocab.json is not present locally, it is fetched automatically from:
    https://huggingface.co/gpt2/raw/main/vocab.json
"""

import json
import os
import struct
import sys
import tempfile
from urllib.request import urlretrieve

VOCAB_URL = "https://huggingface.co/gpt2/raw/main/vocab.json"
VOCAB_SIZE = 50257


def download_vocab(path):
    """Download vocab.json atomically (temp file + rename)."""
    print(f"Downloading vocab.json from {VOCAB_URL} ...")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    os.close(fd)
    try:
        urlretrieve(VOCAB_URL, tmp)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
    print(f"Saved to {path}")


def export_vocab_to_bin(vocab_path="vocab.json", output_path="tokenizer.bin"):
    if not os.path.exists(vocab_path):
        download_vocab(vocab_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        try:
            vocab = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: corrupt {vocab_path}: {e}", file=sys.stderr)
            print("Delete it and re-run to re-download.", file=sys.stderr)
            sys.exit(1)

    if len(vocab) != VOCAB_SIZE:
        print(f"Error: expected {VOCAB_SIZE} tokens, got {len(vocab)}", file=sys.stderr)
        sys.exit(1)

    # Invert: token_string -> id  =>  id -> token_string
    id_to_token = [None] * VOCAB_SIZE
    for token_str, token_id in vocab.items():
        if 0 <= token_id < VOCAB_SIZE:
            id_to_token[token_id] = token_str

    missing = [i for i, t in enumerate(id_to_token) if t is None]
    if missing:
        print(
            f"Error: missing token IDs: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Exporting {VOCAB_SIZE} tokens to '{output_path}' ...")

    with open(output_path, "wb") as f:
        for token_str in id_to_token:
            token_bytes = token_str.encode("utf-8")
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)

    print(f"Done. Tokenizer saved to '{output_path}'.")


if __name__ == "__main__":
    vocab = "vocab.json"
    output = "tokenizer.bin"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--vocab" and i + 1 < len(args):
            vocab = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        else:
            print(
                f"Usage: {sys.argv[0]} [--vocab vocab.json] "
                f"[--output tokenizer.bin]",
                file=sys.stderr,
            )
            sys.exit(1)

    export_vocab_to_bin(vocab, output)
