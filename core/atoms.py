"""
atoms.py

Defines the atomic units of the architecture.
Everything in the graph is built from these primitives upward.

Hierarchy:
    Level 0 — CHARACTER  : single characters — the irreducible primitive
    Level 1 — WORD       : composites of characters — the complete lexicon
    Level 2 — SENTENCE   : composites of words — complete thoughts
    Level 3 — AXIOM      : single or multi-sentence foundational truths
                           the load-bearing nodes from which everything
                           else can be derived by sequencing

Why this hierarchy:

    CHARACTER and WORD are finite and complete.
    Ingest every character that exists. Ingest every word in the
    dictionary. These two layers are not sampled — they are exhaustive.
    This means the graph has complete coverage at the primitive level
    before any reasoning begins.

    SENTENCE is generated — frontier model produces sentences
    systematically across every domain. Not exhaustive but broad.
    Each sentence is a complete thought that connects word nodes
    through co-occurrence.

    AXIOM is the foundational layer — the load-bearing truths of
    every domain. Mathematics, logic, computation, language, physical
    reality. Axioms are what MCTS sequences through to derive answers.
    A high-V axiom node is one from which many conclusions are reachable.

Why PARAGRAPH and DOCUMENT were removed:

    Paragraphs and documents were intermediate containers — useful for
    ingesting raw text but not meaningful as reasoning units. They had
    no foundational role. AXIOM replaces PARAGRAPH as Level 3 with a
    fundamentally different purpose: not a container of sentences but
    a primitive of reasoning.

    DOCUMENT level is gone entirely. The graph is not a document store.
    It is a reasoning engine. Documents decompose into sentences and
    axioms during ingestion — they don't persist as nodes.
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
    AXIOM     = 3


# ─────────────────────────────────────────────
# EMBEDDING DIMENSIONS PER LEVEL
#
# CHARACTER: 16 dims — enough for positional and
#   categorical distinction of 97 printable chars.
#
# WORD: 64 dims — enough for morphological and
#   distributional features of ~200k dictionary words.
#
# SENTENCE: 128 dims — word hash features give
#   genuine lexical signal in 128 dims.
#
# AXIOM: 256 dims — axioms are dense with technical
#   vocabulary and logical structure. Need more dims
#   to distinguish axioms from different domains.
# ─────────────────────────────────────────────

EMBEDDING_DIM = {
    Level.CHARACTER : 16,
    Level.WORD      : 64,
    Level.SENTENCE  : 128,
    Level.AXIOM     : 256,
}


# ─────────────────────────────────────────────
# MUTUAL INFORMATION MERGE THRESHOLD PER LEVEL
#
# Two nodes at level N merge into a composite at level N+1
# when their mutual information exceeds this threshold.
# ─────────────────────────────────────────────

MERGE_THRESHOLD = {
    Level.CHARACTER : 0.15,   # characters merge readily into words
    Level.WORD      : 0.10,   # words merge into sentences with moderate signal
    Level.SENTENCE  : 0.05,   # sentences merge into axioms very selectively
                              # — only the most load-bearing combinations
}


# ─────────────────────────────────────────────
# REWARD SCALE PER LEVEL
#
# Raw rewards are normalized within each level.
# Axioms get the highest reward scale — they are
# the most valuable nodes in the graph.
# ─────────────────────────────────────────────

REWARD_SCALE = {
    Level.CHARACTER : 1.0,
    Level.WORD      : 2.0,
    Level.SENTENCE  : 4.0,
    Level.AXIOM     : 8.0,
}


# ─────────────────────────────────────────────
# AXIOM TYPES
#
# Axioms are tagged by their logical role.
# This is stored as metadata on axiom nodes —
# not as separate node types, just a label.
# ─────────────────────────────────────────────

class AxiomType:
    DEFINITION  = "definition"   # establishes what something IS
    RULE        = "rule"         # establishes what can be DONE
    CONSTRAINT  = "constraint"   # establishes what CANNOT be done
    RELATION    = "relation"     # establishes how things RELATE
    PROCEDURE   = "procedure"    # establishes HOW to do something


# ─────────────────────────────────────────────
# CHARACTER VOCABULARY
#
# The complete set of characters the graph knows.
# Finite and exhaustive — all printable ASCII plus
# structural whitespace. Nothing outside this set
# enters the graph at the character level.
# ─────────────────────────────────────────────

CHAR_VOCAB = (
    [chr(i) for i in range(32, 127)]   # printable ASCII (95 chars)
    + ['\n', '\t']                      # structural whitespace
)

CHAR_TO_ID  = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR  = {i: c for i, c in enumerate(CHAR_VOCAB)}
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
# ─────────────────────────────────────────────

def text_to_characters(text: str) -> list[str]:
    """
    Decompose text into Level 0 character atoms.
    Returns every individual character including whitespace.
    Only printable ASCII and structural whitespace enter the graph.
    """
    return [c for c in text if c in CHAR_TO_ID]


def text_to_words(text: str) -> list[str]:
    """
    Decompose text into Level 1 word atoms.
    Splits on whitespace. Punctuation attached to words
    is kept — the graph learns its role from co-occurrence.
    """
    return re.findall(r"\S+", text)


def text_to_sentences(text: str) -> list[str]:
    """
    Decompose text into Level 2 sentence atoms.
    Sentence boundaries are strong causal transition points.
    Splits on sentence-ending punctuation followed by whitespace.
    """
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def text_to_axioms(text: str) -> list[str]:
    """
    Decompose text into Level 3 axiom atoms.

    Axioms are delimited by:
        - Double newlines (explicit separation)
        - Numbered list items (1. 2. 3.)
        - Bullet points (• - *)
        - Semicolons joining independent clauses

    Single axioms stay whole. Multi-sentence axioms
    where all sentences form one foundational truth
    stay whole.

    Used when ingesting structured axiom corpora
    from frontier model extraction.
    """
    # Try numbered list first
    numbered = re.split(r'\n\s*\d+\.\s+', text.strip())
    if len(numbered) > 2:
        return [a.strip() for a in numbered if a.strip() and len(a.strip()) > 10]

    # Try bullet points
    bulleted = re.split(r'\n\s*[•\-\*]\s+', text.strip())
    if len(bulleted) > 2:
        return [a.strip() for a in bulleted if a.strip() and len(a.strip()) > 10]

    # Try double newline separation
    paragraphed = re.split(r'\n\s*\n', text.strip())
    if len(paragraphed) > 1:
        return [a.strip() for a in paragraphed if a.strip() and len(a.strip()) > 10]

    # Single axiom — return whole
    stripped = text.strip()
    if stripped and len(stripped) > 10:
        return [stripped]

    return []


# ─────────────────────────────────────────────
# LEVEL DETECTION
# ─────────────────────────────────────────────

def infer_level(text: str) -> Level:
    """
    Infer the hierarchy level of a text unit from structure.

    CHARACTER : exactly one character
    WORD      : no whitespace, no newlines
    AXIOM     : multi-sentence or explicitly structured (numbered,
                bulleted) foundational statements
    SENTENCE  : everything else
    """
    stripped = text.strip()

    if len(stripped) == 1:
        return Level.CHARACTER

    if ' ' not in stripped and '\n' not in stripped:
        return Level.WORD

    # Multi-sentence text with structure → AXIOM
    sentences = text_to_sentences(stripped)
    has_structure = (
        bool(re.search(r'\n\s*\d+\.', stripped)) or
        bool(re.search(r'\n\s*[•\-\*]', stripped)) or
        len(sentences) > 2
    )
    if has_structure:
        return Level.AXIOM

    return Level.SENTENCE


# ─────────────────────────────────────────────
# LEVEL DECOMPOSITION DISPATCH
# ─────────────────────────────────────────────

def decompose(text: str, level: Level) -> list[str]:
    """
    Decompose a text unit at the given level
    into its constituent units at level - 1.

    AXIOM     → SENTENCE  (axiom broken into component sentences)
    SENTENCE  → WORD      (sentence broken into words)
    WORD      → CHARACTER (word broken into characters)
    CHARACTER → CHARACTER (atomic — no further decomposition)
    """
    if level == Level.AXIOM:
        sentences = text_to_sentences(text)
        # Filter out numbering artifacts ("1.", "2.", "a)", etc.)
        # and very short fragments that aren't real sentences
        return [
            s for s in sentences
            if len(s.split()) >= 3
            and not re.match(r'^\d+\.?\)?$', s.strip())
            and not re.match(r'^[a-z]\.?\)?$', s.strip())
        ]

    if level == Level.SENTENCE:
        return text_to_words(text)

    if level == Level.WORD:
        return text_to_characters(text)

    if level == Level.CHARACTER:
        return [text]

    raise ValueError(f"Unknown level: {level}")


# ─────────────────────────────────────────────
# LEVEL METADATA
# ─────────────────────────────────────────────

LEVEL_NAMES = {
    Level.CHARACTER : "character",
    Level.WORD      : "word",
    Level.SENTENCE  : "sentence",
    Level.AXIOM     : "axiom",
}

LEVEL_FROM_NAME = {v: k for k, v in LEVEL_NAMES.items()}


def level_name(level: Level) -> str:
    return LEVEL_NAMES.get(level, "unknown")


def level_from_name(name: str) -> Level:
    """Parse a level name string into a Level enum."""
    name = name.lower().strip()
    if name not in LEVEL_FROM_NAME:
        raise ValueError(
            f"Unknown level name: '{name}'. "
            f"Valid: {list(LEVEL_FROM_NAME.keys())}"
        )
    return LEVEL_FROM_NAME[name]


if __name__ == "__main__":
    print("=== atoms.py smoke test ===\n")
    print(f"Vocab size: {VOCAB_SIZE} characters")
    print(f"Levels: {[l.name for l in Level]}\n")

    # Character layer
    print("Characters in 'Hello!':")
    for c in text_to_characters("Hello!"):
        print(f"  {repr(c)} → id {char_to_id(c)}")

    # Word layer
    print("\nWords in 'The derivative of x is 1.':")
    for w in text_to_words("The derivative of x is 1."):
        print(f"  {repr(w)}")

    # Sentence layer
    print("\nSentences in sample text:")
    sample = "For all real x, d/dx(x^2) = 2x. This follows from the power rule."
    for s in text_to_sentences(sample):
        print(f"  {repr(s)}")

    # Axiom layer
    print("\nAxioms from structured text:")
    axiom_text = """
1. For any function x^n, the derivative is n·x^(n-1).
2. The derivative of a sum equals the sum of derivatives.
3. The derivative of a constant is zero.
""".strip()
    for a in text_to_axioms(axiom_text):
        print(f"  {repr(a[:70])}")

    # Level inference
    print("\nLevel inference:")
    tests = [
        "a",
        "derivative",
        "The derivative of x squared is 2x.",
        axiom_text,
    ]
    for t in tests:
        print(f"  {repr(t[:40]):45s} → {level_name(infer_level(t))}")

    # Decomposition
    print("\nDecompose AXIOM → SENTENCE:")
    for s in decompose(axiom_text, Level.AXIOM):
        print(f"  {repr(s[:60])}")