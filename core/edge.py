"""
edge.py

Defines the Edge — the directed connection between two nodes.

An edge is not just a link. It has two roles:

    1. Reward signal — how coherent and causally valid is
       the transition from source to target. Signed scalar.
       Positive = valid transition. Negative = incoherent jump.

    2. Transition function — given the source node's raw signal,
       generate a candidate for what the target should look like.
       This is what makes the graph generative, not just retrieval.

No neural network. The transition function and reward are computed
directly from the raw signal structure at each level.

Edge types:
    SEQUENTIAL  — natural next unit (char→char, word→word, etc.)
    HIERARCHICAL — parent↔child across levels
    ASSOCIATIVE  — learned co-occurrence link (same level, non-adjacent)
    CONTRASTIVE  — explicitly negative example (reward always < 0)
"""

import numpy as np
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from core.atoms import Level, REWARD_SCALE


# ─────────────────────────────────────────────
# EDGE TYPE
# ─────────────────────────────────────────────

class EdgeType(IntEnum):
    SEQUENTIAL   = 0   # natural sequence: A follows B in raw signal
    HIERARCHICAL = 1   # parent contains child across levels
    ASSOCIATIVE  = 2   # learned co-occurrence, same level
    CONTRASTIVE  = 3   # explicitly invalid transition, reward forced negative


# ─────────────────────────────────────────────
# EDGE DATACLASS
# ─────────────────────────────────────────────

@dataclass
class Edge:
    # Identity
    edge_id    : int
    source_id  : int
    target_id  : int
    edge_type  : EdgeType

    # Reward
    reward      : float = 0.0       # raw signed reward
    norm_reward : float = 0.0       # normalized (set by normalizer.py)

    # MCTS statistics — updated during rollouts
    visit_count  : int   = 0
    total_reward : float = 0.0

    # Level of the source node — needed for reward computation
    source_level : Level = Level.SENTENCE


    def mean_reward(self) -> float:
        if self.visit_count == 0:
            return self.reward
        return self.total_reward / self.visit_count


    def to_dict(self) -> dict:
        return {
            "edge_id"     : self.edge_id,
            "source_id"   : self.source_id,
            "target_id"   : self.target_id,
            "edge_type"   : int(self.edge_type),
            "reward"      : self.reward,
            "norm_reward" : self.norm_reward,
            "visit_count" : self.visit_count,
            "total_reward": self.total_reward,
            "source_level": int(self.source_level),
        }


    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(
            edge_id      = d["edge_id"],
            source_id    = d["source_id"],
            target_id    = d["target_id"],
            edge_type    = EdgeType(d["edge_type"]),
            reward       = d["reward"],
            norm_reward  = d["norm_reward"],
            visit_count  = d["visit_count"],
            total_reward = d["total_reward"],
            source_level = Level(d["source_level"]),
        )


    def __repr__(self) -> str:
        return (
            f"Edge(id={self.edge_id}, "
            f"{self.source_id}→{self.target_id}, "
            f"type={self.edge_type.name}, "
            f"r={self.reward:+.3f}, "
            f"visits={self.visit_count})"
        )


# ─────────────────────────────────────────────
# REWARD COMPUTATION
#
# Signed reward for transitioning from source_text
# to target_text at a given level.
#
# Three components combined:
#   coherence — shared signal between source and target
#   novelty   — new signal introduced by target
#   validity  — structural correctness of the transition
#
# All computed from raw signal. No labels. No neural network.
# ─────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Word-level tokenization for overlap computation."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def _char_set(text: str) -> set:
    """Character-level tokenization."""
    return set(text.lower())


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity — intersection over union."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _novelty(source_text: str, target_text: str, level: Level) -> float:
    """
    How much new signal does target introduce beyond source?
    Returns [0, 1] — 0 = pure repetition, 1 = entirely new.
    """
    if level <= Level.WORD:
        src = _char_set(source_text)
        tgt = _char_set(target_text)
    else:
        src = _tokenize(source_text)
        tgt = _tokenize(target_text)

    if not tgt:
        return 0.0

    new_signal = tgt - src
    return len(new_signal) / len(tgt)


def _coherence(source_text: str, target_text: str, level: Level) -> float:
    """
    How much shared signal exists between source and target?
    Returns [0, 1] — 0 = nothing shared, 1 = identical.
    """
    if level <= Level.WORD:
        return _jaccard(_char_set(source_text), _char_set(target_text))
    else:
        return _jaccard(_tokenize(source_text), _tokenize(target_text))


def _structural_validity(source_text: str, target_text: str, level: Level) -> float:
    """
    Is the transition structurally valid at this level?
    Returns value in [-1, 1].

    Checks level-appropriate structural properties:
        CHARACTER: target is a valid character
        WORD:      target looks like a real word
        SENTENCE:  target is a complete sentence
        PARAGRAPH: target is a coherent paragraph
    """
    if not target_text or not target_text.strip():
        return -1.0

    if level == Level.CHARACTER:
        # Valid character transition — single printable character
        return 1.0 if len(target_text) == 1 and target_text.isprintable() else -1.0

    if level == Level.WORD:
        # Valid word — no spaces, not empty, reasonable length
        stripped = target_text.strip()
        if not stripped or ' ' in stripped:
            return -0.5
        if len(stripped) > 30:   # suspiciously long
            return -0.3
        return 1.0 if stripped.replace("'", "").replace("-", "").isalpha() else 0.3

    if level == Level.SENTENCE:
        # Valid sentence — ends with punctuation, has multiple words
        stripped = target_text.strip()
        words = stripped.split()
        if len(words) < 2:
            return -0.5
        if stripped[-1] not in '.!?:':
            return 0.2          # not ideal but not invalid
        if len(words) > 80:
            return -0.2         # suspiciously long
        return 1.0

    if level == Level.PARAGRAPH:
        # Valid paragraph — multiple sentences, substantial content
        stripped = target_text.strip()
        sentence_count = len(re.split(r'[.!?]+', stripped))
        if sentence_count < 2:
            return 0.3
        if len(stripped) < 30:
            return -0.3
        return 1.0

    if level == Level.DOCUMENT:
        # Valid document — multiple paragraphs
        paragraphs = [p for p in target_text.split('\n\n') if p.strip()]
        return 1.0 if len(paragraphs) >= 2 else 0.5

    return 0.0


def compute_edge_reward(
    source_text : str,
    target_text : str,
    level       : Level,
    edge_type   : EdgeType = EdgeType.SEQUENTIAL,
) -> float:
    """
    Compute signed edge reward for transitioning from
    source_text to target_text at the given level.

    Formula:
        reward = w_coh * coherence
               + w_nov * novelty
               + w_val * structural_validity
               - penalty_for_extremes

    Scaled to REWARD_SCALE[level] so rewards are comparable
    across levels after level-wise normalization.

    Returns a signed float. Positive = valid transition.
    Negative = incoherent or structurally invalid transition.
    """
    if edge_type == EdgeType.CONTRASTIVE:
        # Explicitly invalid — forced negative
        return -REWARD_SCALE[level]

    if edge_type == EdgeType.HIERARCHICAL:
        # Parent-child edge — always positive, strength by overlap
        coh = _coherence(source_text, target_text, level)
        return REWARD_SCALE[level] * (0.5 + 0.5 * coh)

    # SEQUENTIAL and ASSOCIATIVE — compute from signal
    coh = _coherence(source_text, target_text, level)
    nov = _novelty(source_text, target_text, level)
    val = _structural_validity(source_text, target_text, level)

    # Level-specific weights
    # Higher levels weight coherence more (context matters more)
    # Lower levels weight novelty more (new characters/words matter)
    if level == Level.CHARACTER:
        w_coh, w_nov, w_val = 0.2, 0.5, 0.3
    elif level == Level.WORD:
        w_coh, w_nov, w_val = 0.3, 0.4, 0.3
    elif level == Level.SENTENCE:
        w_coh, w_nov, w_val = 0.4, 0.3, 0.3
    elif level == Level.PARAGRAPH:
        w_coh, w_nov, w_val = 0.5, 0.2, 0.3
    else:  # DOCUMENT
        w_coh, w_nov, w_val = 0.5, 0.2, 0.3

    # Raw score in [-1, 1]
    # coherence: [0,1] → [-0.5, +0.5] (some coherence needed, too much = repetition)
    # novelty:   [0,1] → [-0.5, +0.5] (some novelty needed, too much = incoherent)
    # validity:  [-1,1] → direct contribution

    coh_score = (coh - 0.3) * 2.0    # penalize both extremes
    nov_score = (nov - 0.4) * 2.0    # penalize both extremes
    val_score = val

    raw = w_coh * coh_score + w_nov * nov_score + w_val * val_score

    # Scale to level reward scale
    scaled = raw * REWARD_SCALE[level]

    return round(float(np.clip(scaled, -REWARD_SCALE[level] * 2, REWARD_SCALE[level] * 2)), 4)


# ─────────────────────────────────────────────
# TRANSITION FUNCTION
#
# Given source text, generate a candidate for
# what the next node's text should look like.
#
# This is the generative capacity of the edge —
# not retrieval, actual generation from signal structure.
#
# At this stage: heuristic generation from signal.
# Captures the statistical regularities learned
# from co-occurrence without neural parameters.
# ─────────────────────────────────────────────

def transition(source_text: str, level: Level,
               context: Optional[list[str]] = None) -> list[str]:
    """
    Generate candidate next states from source_text at the given level.
    Returns a list of candidate strings — multiple candidates for beam search.

    context: optional list of preceding texts at same level,
             for richer transition signal.

    These are structural heuristic candidates — the graph's
    prior over what comes next based purely on signal structure.
    In a full system, higher-quality candidates come from
    traversal of existing connected nodes.
    """
    candidates = []
    ctx = context or []

    if level == Level.CHARACTER:
        # Character transitions — structural patterns
        c = source_text[-1] if source_text else ' '
        if c.isalpha():
            # After a letter: likely another letter or space
            candidates = [' ', 'e', 't', 'a', 's', 'n']
        elif c == ' ':
            # After space: likely a letter starting a word
            candidates = ['t', 'a', 'i', 's', 'o', 'T']
        elif c in '.!?':
            # After sentence end: space then capital
            candidates = [' ']
        else:
            candidates = [' ', 'a', 'e']

    elif level == Level.WORD:
        # Word transitions — use source word structure as signal
        words_seen = set()
        for t in ctx:
            words_seen.update(re.findall(r'\b\w+\b', t.lower()))

        src_lower = source_text.lower().strip('.,!?;:')

        # Heuristic: function words that commonly follow content words
        if src_lower in ('the', 'a', 'an', 'this', 'that', 'these', 'those'):
            candidates = ['is', 'was', 'will', 'has', 'had', 'can']
        elif source_text[-1] in '.!?':
            candidates = ['The', 'She', 'He', 'It', 'They', 'This']
        elif source_text[-1] == ',':
            candidates = ['and', 'but', 'or', 'which', 'where', 'when']
        else:
            candidates = ['the', 'and', 'to', 'of', 'in', 'a']

    elif level == Level.SENTENCE:
        # Sentence transitions — use source sentence structure
        src_words = _tokenize(source_text)

        # Generate continuations that share some context
        # but introduce new information
        # Heuristic templates based on source structure
        candidates_raw = []

        if '?' in source_text:
            # After a question: expect an answer or elaboration
            candidates_raw = [
                "The answer became clear as she examined the evidence.",
                "It remained uncertain, but the clues pointed in one direction.",
                "She did not yet know, but she was close to finding out.",
            ]
        elif source_text.strip()[-1] == '.':
            # After declarative: continuation, consequence, or contrast
            candidates_raw = [
                "This changed everything she thought she knew.",
                "The implications were not immediately obvious.",
                "She paused, considering what this meant.",
            ]
        else:
            candidates_raw = [
                "The situation continued to develop.",
                "More information was needed.",
                "She moved forward carefully.",
            ]

        # Filter candidates by reward signal — keep only coherent ones
        for cand in candidates_raw:
            r = compute_edge_reward(source_text, cand, level)
            if r > 0:
                candidates.append(cand)

        if not candidates:
            candidates = candidates_raw[:2]

    elif level == Level.PARAGRAPH:
        # Paragraph transitions — high-level causal continuation
        if ctx:
            prev = ctx[-1]
            prev_words = _tokenize(prev)
            src_words  = _tokenize(source_text)
            shared = prev_words & src_words
        else:
            shared = set()

        candidates = [
            "The evidence pointed toward a conclusion that had not been considered.",
            "This development shifted the direction of the investigation entirely.",
        ]

    elif level == Level.DOCUMENT:
        candidates = [
            "The following section examines the implications in greater detail.",
            "A new perspective emerged from the preceding analysis.",
        ]

    return candidates


# ─────────────────────────────────────────────
# EDGE FACTORY
# ─────────────────────────────────────────────

def make_edge(
    edge_id     : int,
    source_id   : int,
    target_id   : int,
    source_text : str,
    target_text : str,
    level       : Level,
    edge_type   : EdgeType = EdgeType.SEQUENTIAL,
) -> Edge:
    """
    Create a new edge with computed reward.
    Single entry point for edge creation.
    """
    reward = compute_edge_reward(source_text, target_text, level, edge_type)
    return Edge(
        edge_id      = edge_id,
        source_id    = source_id,
        target_id    = target_id,
        edge_type    = edge_type,
        reward       = reward,
        source_level = level,
    )


if __name__ == "__main__":
    print("=== edge.py smoke test ===\n")

    test_pairs = [
        ("She found footprints.",
         "The footprints were fresh.",
         Level.SENTENCE, EdgeType.SEQUENTIAL),

        ("She found footprints.",
         "Bananas grow in tropical climates.",
         Level.SENTENCE, EdgeType.SEQUENTIAL),

        ("She found footprints.",
         "The footprints were fresh.",
         Level.SENTENCE, EdgeType.CONTRASTIVE),

        ("detective", "maria", Level.WORD, EdgeType.SEQUENTIAL),
        ("detective", "xqzwj", Level.WORD, EdgeType.SEQUENTIAL),

        ("a", "b", Level.CHARACTER, EdgeType.SEQUENTIAL),
        (".", " ", Level.CHARACTER, EdgeType.SEQUENTIAL),

        ("In the corner she discovered a locked metal box.",
         "The box had three combination locks each with four digits.",
         Level.SENTENCE, EdgeType.SEQUENTIAL),
    ]

    for i, (src, tgt, level, etype) in enumerate(test_pairs):
        e = make_edge(i, 100+i, 200+i, src, tgt, level, etype)
        verdict = "POSITIVE ✓" if e.reward > 0 else "NEGATIVE ✗"
        print(f"[{i}] {verdict}  r={e.reward:+.4f}  {level.name}  {etype.name}")
        print(f"  src: {repr(src[:50])}")
        print(f"  tgt: {repr(tgt[:50])}")
        print()

    print("Transition candidates from sentence:")
    src = "She found footprints leading to the back room."
    cands = transition(src, Level.SENTENCE)
    for c in cands:
        r = compute_edge_reward(src, c, Level.SENTENCE)
        print(f"  r={r:+.4f}  {repr(c)}")