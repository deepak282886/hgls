"""
atoms.py

Defines the atomic units of the architecture.
Everything in the graph is built from these primitives upward.

Hierarchy:
    Level 0 — characters (letters, digits, punctuation)
    Level 1 — words (composites of characters)
    Level 2 — sentences (composites of words)
    Level 3 — paragraphs (composites of sentences)
    Level 4 — documents (composites of paragraphs)

Each level has:
    - A defined atomic unit
    - An embedding dimension reflecting its complexity
    - A merging threshold — minimum mutual information to form a composite
    - A reward scale — normalization anchor for that level
"""

from enum import IntEnum
import re


# ─────────────────────────────────────────────
# HIERARCHY LEVELS
# ─────────────────────────────────────────────

class Level(IntEnum):
    CHARACTER = 0
    WORD      = 1
    SENTENCE  = 2
    PARAGRAPH = 3
    DOCUMENT  = 4


# ─────────────────────────────────────────────
# EMBEDDING DIMENSIONS PER LEVEL
#
# Larger levels represent more complex composites
# and require more dimensions to encode their structure.
# Kept small for CPU efficiency on laptop hardware.
# ─────────────────────────────────────────────

EMBEDDING_DIM = {
    Level.CHARACTER : 16,
    Level.WORD      : 64,
    Level.SENTENCE  : 128,
    Level.PARAGRAPH : 256,
    Level.DOCUMENT  : 512,
}


# ─────────────────────────────────────────────
# MUTUAL INFORMATION MERGE THRESHOLD PER LEVEL
#
# Two nodes at level N merge into a composite at level N+1
# when their mutual information exceeds this threshold.
# Higher threshold = stricter merging = more selective composites.
# ─────────────────────────────────────────────

MERGE_THRESHOLD = {
    Level.CHARACTER : 0.15,   # letters merge readily into words
    Level.WORD      : 0.10,   # words merge into sentences with moderate signal
    Level.SENTENCE  : 0.08,   # sentences merge into paragraphs more selectively
    Level.PARAGRAPH : 0.06,   # paragraphs merge into documents most selectively
}


# ─────────────────────────────────────────────
# REWARD SCALE PER LEVEL
#
# Raw rewards are normalized within each level.
# This anchor defines the expected reward magnitude
# at each level — prevents higher levels from
# dominating simply by accumulating more signal.
# ─────────────────────────────────────────────

REWARD_SCALE = {
    Level.CHARACTER : 1.0,
    Level.WORD      : 2.0,
    Level.SENTENCE  : 4.0,
    Level.PARAGRAPH : 8.0,
    Level.DOCUMENT  : 16.0,
}


# ─────────────────────────────────────────────
# ATOMIC UNIT: CHARACTER
#
# The irreducible primitive of text.
# Every other structure is a configuration of characters.
# ─────────────────────────────────────────────

# Full character vocabulary — printable ASCII
# The graph learns which configurations are meaningful
# from the statistics of the data. Nothing is
# hand-labeled as a letter vs punctuation vs digit.
CHAR_VOCAB = (
    [chr(i) for i in range(32, 127)]  # printable ASCII
    + ['\n', '\t']                     # structural whitespace
)

CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE  = len(CHAR_VOCAB)

UNKNOWN_CHAR_ID = VOCAB_SIZE  # for characters outside vocabulary


def char_to_id(c: str) -> int:
    """Map a character to its vocabulary ID."""
    return CHAR_TO_ID.get(c, UNKNOWN_CHAR_ID)


def id_to_char(i: int) -> str:
    """Map a vocabulary ID back to its character."""
    return ID_TO_CHAR.get(i, '?')


# ─────────────────────────────────────────────
# TOKENIZATION FUNCTIONS
#
# Each function decomposes a unit into its
# constituent atomic units at the level below.
# No external tokenizer. No pretrained vocabulary.
# The graph learns what matters from co-occurrence.
# ─────────────────────────────────────────────

def text_to_characters(text: str) -> list[str]:
    """
    Decompose raw text into Level 0 character atoms.
    This is the entry point for all text ingestion.
    """
    return list(text)


def text_to_words(text: str) -> list[str]:
    """
    Decompose text into Level 1 word atoms.
    Preserves punctuation as separate tokens —
    punctuation carries causal signal (question marks,
    periods, commas all affect meaning of transitions).
    """
    # Split on whitespace but keep punctuation attached
    # to their words as the graph will learn their role
    tokens = re.findall(r"\S+", text)
    return tokens


def text_to_sentences(text: str) -> list[str]:
    """
    Decompose text into Level 2 sentence atoms.
    Sentence boundaries are strong causal transition points —
    each sentence is a complete state in the causal chain.
    """
    # Split on sentence-ending punctuation followed by whitespace
    # Keep the punctuation with the sentence — it is part of the state
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def text_to_paragraphs(text: str) -> list[str]:
    """
    Decompose text into Level 3 paragraph atoms.
    Paragraphs are high-level causal units —
    a paragraph develops one idea, then transitions.
    """
    raw = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in raw if p.strip()]


# ─────────────────────────────────────────────
# LEVEL DETECTION
#
# Given a raw text unit, infer its natural level.
# Used during ingestion when the level is not
# explicitly known.
# ─────────────────────────────────────────────

def infer_level(text: str) -> Level:
    """
    Infer the hierarchy level of a text unit
    from its structure. Heuristic but effective —
    the graph will refine this through training.
    """
    stripped = text.strip()

    if len(stripped) == 1:
        return Level.CHARACTER

    if ' ' not in stripped and '\n' not in stripped:
        return Level.WORD

    if '\n\n' in stripped:
        return Level.PARAGRAPH

    if '\n' in stripped:
        return Level.PARAGRAPH

    # Ends with sentence punctuation and is short
    if stripped[-1] in '.!?' and len(stripped.split()) < 60:
        return Level.SENTENCE

    return Level.SENTENCE


# ─────────────────────────────────────────────
# LEVEL DECOMPOSITION DISPATCH
#
# Given a text unit and its level, return
# the constituent units at the level below.
# This drives the bottom-up hierarchy construction.
# ─────────────────────────────────────────────

def decompose(text: str, level: Level) -> list[str]:
    """
    Decompose a text unit at the given level
    into its constituent units at level - 1.
    """
    if level == Level.DOCUMENT:
        return text_to_paragraphs(text)

    if level == Level.PARAGRAPH:
        return text_to_sentences(text)

    if level == Level.SENTENCE:
        return text_to_words(text)

    if level == Level.WORD:
        return text_to_characters(text)

    if level == Level.CHARACTER:
        # Characters are atomic — no further decomposition
        return [text]

    raise ValueError(f"Unknown level: {level}")


# ─────────────────────────────────────────────
# LEVEL METADATA
# ─────────────────────────────────────────────

LEVEL_NAMES = {
    Level.CHARACTER : "character",
    Level.WORD      : "word",
    Level.SENTENCE  : "sentence",
    Level.PARAGRAPH : "paragraph",
    Level.DOCUMENT  : "document",
}

def level_name(level: Level) -> str:
    return LEVEL_NAMES.get(level, "unknown")


if __name__ == "__main__":
    # Smoke test
    sample = (
        "Detective Maria arrived at the warehouse.\n\n"
        "The building smelled of rust. She found footprints."
    )

    print("=== atoms.py smoke test ===\n")
    print(f"Vocab size: {VOCAB_SIZE} characters\n")

    print("Characters (first 10):")
    for c in text_to_characters(sample)[:10]:
        print(f"  {repr(c)} → id {char_to_id(c)}")

    print("\nWords:")
    for w in text_to_words(sample):
        print(f"  {repr(w)}")

    print("\nSentences:")
    for s in text_to_sentences(sample):
        print(f"  {repr(s)}")

    print("\nParagraphs:")
    for p in text_to_paragraphs(sample):
        print(f"  {repr(p)}")

    print("\nLevel inference:")
    tests = ["a", "hello", "She found footprints.", sample]
    for t in tests:
        print(f"  {repr(t[:30])} → {level_name(infer_level(t))}")

    print("\nDecompose sentence → words:")
    sent = "She found footprints."
    for w in decompose(sent, Level.SENTENCE):
        print(f"  {repr(w)}")