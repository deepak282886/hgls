"""
Data Pipeline — Simple English Wikipedia
Phase 1 of training plan.

Run locally (not in sandbox — requires network access to dumps.wikimedia.org).

Usage:
    python data_pipeline.py --output_dir ./data

Outputs:
    data/sentences.txt   — one sentence per line, document order preserved
    data/pairs.jsonl     — consecutive sentence pairs for reward training
    data/stats.json      — corpus statistics
"""

import re
import json
import bz2
import argparse
import time
from pathlib import Path
from urllib.request import urlretrieve

try:
    import mwparserfromhell
except ImportError:
    raise SystemExit("Run: pip install mwparserfromhell")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DUMP_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
DUMP_FILE = "simplewiki-latest-pages-articles.xml.bz2"

# Categories to keep — science domains that produce causal chains
TARGET_CATEGORIES = {
    "weather", "atmosphere", "climate", "meteorology",
    "biology", "cell", "cells", "organism", "organisms", "plant", "animal",
    "physics", "force", "motion", "gravity", "energy", "light", "matter",
    "chemistry", "molecule", "atom", "element",
    "earth", "geology", "ocean", "water cycle",
    "ecology", "ecosystem", "evolution",
    "science", "nature",
}

# Sentence filters
MIN_WORDS = 5
MAX_WORDS = 40
MIN_SENTENCES_PER_ARTICLE = 3


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────

def download_dump(output_path: str):
    if Path(output_path).exists():
        print(f"Dump already exists: {output_path}")
        return

    print(f"Downloading Simple English Wikipedia dump (~250MB)...")
    print(f"Source: {DUMP_URL}")

    def progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        if count % 500 == 0:
            print(f"  {pct}%", end="\r", flush=True)

    urlretrieve(DUMP_URL, output_path, reporthook=progress)
    print(f"\nDownloaded: {output_path}")


# ─────────────────────────────────────────────
# XML PARSER
# ─────────────────────────────────────────────

def iter_articles(dump_path: str):
    """
    Stream articles from the bz2 XML dump without loading it all into memory.
    Yields (title, wikitext) tuples.
    """
    import xml.etree.ElementTree as ET

    ns = "http://www.mediawiki.org/xml/DTD/mediawiki"

    with bz2.open(dump_path, "rb") as f:
        title = None
        in_text = False
        text_buf = []

        for event, elem in ET.iterparse(f, events=("start", "end")):
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            if event == "start":
                if tag == "title":
                    title = None
                elif tag == "text":
                    in_text = True
                    text_buf = []

            elif event == "end":
                if tag == "title":
                    title = elem.text or ""
                elif tag == "text" and in_text:
                    text_buf.append(elem.text or "")
                    in_text = False
                    yield title, "".join(text_buf)
                    elem.clear()


# ─────────────────────────────────────────────
# WIKITEXT CLEANER
# ─────────────────────────────────────────────

def clean_wikitext(wikitext: str) -> str:
    """Strip wikitext markup and return plain text."""
    try:
        parsed = mwparserfromhell.parse(wikitext)
        text = parsed.strip_code()
    except Exception:
        text = wikitext

    # remove leftover markup
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'={2,}.*?={2,}', '', text)  # headers
    text = re.sub(r'\[\d+\]', '', text)          # citations
    text = re.sub(r"'{2,}", '', text)             # bold/italic
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_science_article(title: str, wikitext: str) -> bool:
    """Check if article falls in our target domains."""
    title_lower = title.lower()
    text_lower = wikitext[:2000].lower()

    for kw in TARGET_CATEGORIES:
        if kw in title_lower:
            return True

    # category tags in wikitext
    cats = re.findall(r'\[\[Category:(.*?)\]\]', wikitext, re.IGNORECASE)
    for cat in cats:
        for kw in TARGET_CATEGORIES:
            if kw in cat.lower():
                return True

    return False


# ─────────────────────────────────────────────
# SENTENCE SPLITTER & CLEANER
# ─────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, same logic as Segmenter._sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip().lower() for s in sentences if s.strip()]


def is_valid_sentence(sentence: str) -> bool:
    """Filter out junk sentences."""
    words = sentence.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    # skip sentences that are mostly non-alpha (tables, equations)
    alpha = sum(1 for c in sentence if c.isalpha())
    if alpha / max(len(sentence), 1) < 0.6:
        return False
    return True


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(dump_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sentences_path = out / "sentences.txt"
    pairs_path = out / "pairs.jsonl"
    stats_path = out / "stats.json"

    total_articles = 0
    kept_articles = 0
    total_sentences = 0
    total_pairs = 0

    seen_sentences = set()

    t0 = time.time()

    with open(sentences_path, "w") as sf, open(pairs_path, "w") as pf:
        for title, wikitext in iter_articles(dump_path):
            total_articles += 1
            if total_articles % 10000 == 0:
                print(f"  {total_articles} articles scanned, "
                      f"{kept_articles} kept, {total_sentences} sentences, "
                      f"{time.time()-t0:.0f}s elapsed")

            # redirect pages and disambiguation — skip
            if "#REDIRECT" in wikitext[:50] or "disambiguation" in title.lower():
                continue

            if not is_science_article(title, wikitext):
                continue

            text = clean_wikitext(wikitext)
            if not text:
                continue

            sentences = split_sentences(text)
            valid = [s for s in sentences if is_valid_sentence(s)]

            # deduplicate within article and globally
            unique = []
            for s in valid:
                if s not in seen_sentences:
                    seen_sentences.add(s)
                    unique.append(s)

            if len(unique) < MIN_SENTENCES_PER_ARTICLE:
                continue

            kept_articles += 1

            # write sentences preserving document order
            for s in unique:
                sf.write(s + "\n")
                total_sentences += 1

            # write consecutive pairs (same article only — cross-article pairs are noise)
            for i in range(len(unique) - 1):
                pair = {
                    "input": unique[i],
                    "target": unique[i + 1],
                    "article": title,
                    "pair_idx": i
                }
                pf.write(json.dumps(pair) + "\n")
                total_pairs += 1

    stats = {
        "total_articles_scanned": total_articles,
        "articles_kept": kept_articles,
        "total_sentences": total_sentences,
        "total_pairs": total_pairs,
        "unique_sentences": len(seen_sentences),
        "elapsed_seconds": round(time.time() - t0, 1)
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Pipeline Complete ===")
    print(json.dumps(stats, indent=2))
    return stats


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data",
                        help="Directory to write sentences.txt, pairs.jsonl, stats.json")
    parser.add_argument("--dump", default=DUMP_FILE,
                        help="Path to already-downloaded .xml.bz2 dump file")
    parser.add_argument("--download", action="store_true",
                        help="Download the dump before processing")
    args = parser.parse_args()

    if args.download:
        download_dump(args.dump)

    if not Path(args.dump).exists():
        print(f"Dump not found: {args.dump}")
        print("Run with --download to fetch it, or provide path with --dump")
        raise SystemExit(1)

    run_pipeline(args.dump, args.output_dir)