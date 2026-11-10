"""
Tokenizer for hypergraph-lm.

Reads all .txt files from data/corpora/
Tokenizes using tiktoken cl100k_base
Writes flat uint16 numpy array to data/corpus.bin

uint16 supports up to 65535 token ids.
cl100k_base vocab is ~100K so we use uint32 to be safe.
"""

import os
import numpy as np
import tiktoken
from pathlib import Path
from tqdm import tqdm

CORPORA_DIR = Path(__file__).parent / "corpora"
OUTPUT_BIN  = Path(__file__).parent / "corpus.bin"
ENCODING    = "cl100k_base"

# Read in chunks to avoid RAM overload
CHUNK_LINES = 10_000


def main():
    enc = tiktoken.get_encoding(ENCODING)

    # Collect all txt files
    txt_files = sorted(CORPORA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {CORPORA_DIR}")
        return

    print("=" * 50)
    print("Hypergraph-LM Tokenizer")
    print("=" * 50)
    print(f"Encoding:  {ENCODING}")
    print(f"Vocab size: {enc.n_vocab}")
    print(f"Files found: {[f.name for f in txt_files]}")
    print(f"Output: {OUTPUT_BIN}")
    print()

    total_tokens = 0

    with open(OUTPUT_BIN, "wb") as out_f:
        for txt_file in txt_files:
            print(f"Processing {txt_file.name} ...")

            file_size = txt_file.stat().st_size
            file_tokens = 0

            with open(txt_file, "r", encoding="utf-8", errors="ignore") as in_f:
                chunk = []
                pbar = tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=txt_file.name
                )

                for line in in_f:
                    pbar.update(len(line.encode("utf-8")))
                    chunk.append(line)

                    if len(chunk) >= CHUNK_LINES:
                        text = "".join(chunk)
                        tokens = enc.encode_ordinary(text)
                        arr = np.array(tokens, dtype=np.uint32)
                        arr.tofile(out_f)
                        file_tokens += len(tokens)
                        total_tokens += len(tokens)
                        chunk = []

                # Flush remaining
                if chunk:
                    text = "".join(chunk)
                    tokens = enc.encode_ordinary(text)
                    arr = np.array(tokens, dtype=np.uint32)
                    arr.tofile(out_f)
                    file_tokens += len(tokens)
                    total_tokens += len(tokens)

                pbar.close()

            print(f"  {txt_file.name}: {file_tokens:,} tokens")

    bin_size_gb = OUTPUT_BIN.stat().st_size / 1e9
    print()
    print("=" * 50)
    print(f"Total tokens:    {total_tokens:,}")
    print(f"Output size:     {bin_size_gb:.2f} GB")
    print(f"Output path:     {OUTPUT_BIN}")
    print()

    # Sanity check — read back first 10 tokens
    test = np.fromfile(OUTPUT_BIN, dtype=np.uint32, count=10)
    decoded = enc.decode(test.tolist())
    print(f"First 10 tokens decode to: {repr(decoded)}")
    print("=" * 50)


if __name__ == "__main__":
    main()