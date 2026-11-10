"""
Corpus downloader for hypergraph-lm.

Downloads and extracts text from:
1. ShareGPT
2. LMSYS Chat 1M
3. OpenAssistant (oasst2)
4. UltraChat

All text is written to data/corpora/corpus.txt
One conversation per block, turns separated by newlines.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# Output paths
CORPORA_DIR = Path(__file__).parent / "corpora"
OUTPUT_FILE = CORPORA_DIR / "corpus.txt"
SEPARATOR = "\n"

CORPORA_DIR.mkdir(parents=True, exist_ok=True)


def write_conversation(f, turns: list[str]):
    """Write a list of text turns as one conversation block."""
    block = SEPARATOR.join(t.strip() for t in turns if t and t.strip())
    if block:
        f.write(block + "\n\n")


def download_sharegpt(f):
    """ShareGPT — human/assistant conversations."""
    print("\n[1/4] Downloading ShareGPT...")
    from datasets import load_dataset

    try:
        ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            data_files="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
            split="train",
        )
        count = 0
        for row in tqdm(ds, desc="ShareGPT"):
            turns = []
            for conv in row.get("conversations", []):
                value = conv.get("value", "")
                if value:
                    turns.append(value)
            if turns:
                write_conversation(f, turns)
                count += 1
        print(f"  ShareGPT: {count} conversations written")
    except Exception as e:
        print(f"  ShareGPT failed: {e}")


def download_lmsys(f):
    """LMSYS Chat 1M — diverse multi-turn conversations."""
    print("\n[2/4] Downloading LMSYS Chat 1M...")
    from datasets import load_dataset

    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
        count = 0
        for row in tqdm(ds, desc="LMSYS"):
            turns = []
            for conv in row.get("conversation", []):
                content = conv.get("content", "")
                if content:
                    turns.append(content)
            if turns:
                write_conversation(f, turns)
                count += 1
        print(f"  LMSYS: {count} conversations written")
    except Exception as e:
        print(f"  LMSYS failed: {e}")


def download_oasst(f):
    """OpenAssistant oasst2 — high quality reasoning conversations."""
    print("\n[3/4] Downloading OpenAssistant oasst2...")
    from datasets import load_dataset

    try:
        ds = load_dataset("OpenAssistant/oasst2", split="train")

        # oasst2 is a tree structure — extract all messages as flat text
        count = 0
        for row in tqdm(ds, desc="OASST2"):
            text = row.get("text", "")
            if text and text.strip():
                f.write(text.strip() + "\n\n")
                count += 1
        print(f"  OASST2: {count} messages written")
    except Exception as e:
        print(f"  OASST2 failed: {e}")


def download_ultrachat(f):
    """UltraChat — long multi-turn dialogues."""
    print("\n[4/4] Downloading UltraChat...")
    from datasets import load_dataset

    try:
        ds = load_dataset("stingning/ultrachat", split="train")
        count = 0
        for row in tqdm(ds, desc="UltraChat"):
            turns = row.get("data", [])
            if turns:
                write_conversation(f, turns)
                count += 1
        print(f"  UltraChat: {count} conversations written")
    except Exception as e:
        print(f"  UltraChat failed: {e}")


def main():
    print("=" * 50)
    print("Hypergraph-LM Corpus Downloader")
    print("=" * 50)
    print(f"Output: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        download_sharegpt(f)
        download_lmsys(f)
        download_oasst(f)
        download_ultrachat(f)

    size_gb = OUTPUT_FILE.stat().st_size / 1e9
    print(f"\nDone. corpus.txt size: {size_gb:.2f} GB")
    print(f"Location: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()