"""
atoms.py

The absolute foundation. Three parameters. Core types. Nothing else.

Three parameters control everything:
    BASE_DECAY              — how fast unused structure fades
    COMPRESSION_THRESHOLD   — how much recurrence before a pattern becomes a node
    BOUNDARY_SENSITIVITY    — how novel something must be to cross the membrane

Everything else in the system is derived from these three and from the data.

Node states:
    provisional  — seen but not yet stable. in the provisional buffer.
    stable       — well supported by observation. core graph.
    hypothetical — generated top-down by projection. not yet observed.
    decaying     — was stable, losing support. fading out.

Level:
    Not pre-decided. Emerges from compression.
    Level 0 = raw input tokens (primitives).
    Level N = patterns of level N-1 nodes.
    The system discovers how many levels it needs.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# THREE PARAMETERS — the only knobs
# ─────────────────────────────────────────────

BASE_DECAY            = 0.001   # per tick weight loss rate
COMPRESSION_THRESHOLD = 0.7     # PMI threshold for compression
BOUNDARY_SENSITIVITY  = 0.3     # novelty threshold for crossing membrane


# ─────────────────────────────────────────────
# NODE STATE
# ─────────────────────────────────────────────

class State(str, Enum):
    PROVISIONAL  = "provisional"
    STABLE       = "stable"
    HYPOTHETICAL = "hypothetical"
    DECAYING     = "decaying"


# ─────────────────────────────────────────────
# DECAY RATE — derived from level
#
# Higher level = slower decay.
# Lower level  = faster decay.
# Relearning a character costs nothing.
# Relearning a schema costs a lot.
# ─────────────────────────────────────────────

def decay_rate(level: int) -> float:
    """
    Derive decay rate from level.
    Level 0 (primitives) decay fastest — they are constantly reinforced.
    Higher levels decay slower — they are harder to relearn.
    """
    return BASE_DECAY / (level + 1)


# ─────────────────────────────────────────────
# ACTIVATION BOOST — how much a visit restores weight
# ─────────────────────────────────────────────

ACTIVATION_BOOST = 0.1   # weight added on each activation


# ─────────────────────────────────────────────
# WEIGHT THRESHOLDS
# ─────────────────────────────────────────────

STABLE_THRESHOLD   = 0.5    # below this, stable → decaying
DISSOLVE_THRESHOLD = 0.01   # below this, node dissolves
ACTIVATION_MIN     = 0.05   # below this, node invisible to traversal


# ─────────────────────────────────────────────
# THINKING CONSTANTS
# ─────────────────────────────────────────────

MAX_ROLLOUT_DEPTH    = 20    # maximum steps in a single rollout
MIN_CONFIDENCE       = 0.05  # stop rollout when confidence drops below this
PARALLEL_ROLLOUTS    = 4     # number of simultaneous rollouts
CONTEXT_WINDOW_SIZE  = 8     # how many recent nodes to keep in working memory


# ─────────────────────────────────────────────
# COMPRESSION CONSTANTS
# ─────────────────────────────────────────────

MIN_INSTANCES_FOR_SCHEMA = 3   # minimum instances before slot detection runs
MIN_RECURRENCE           = 2   # minimum times a pattern must recur


# ─────────────────────────────────────────────
# PROVISIONAL BUFFER CONSTANTS
# ─────────────────────────────────────────────

PROVISIONAL_TTL = 3600.0   # seconds before unconfirmed provisional dissolves


# ─────────────────────────────────────────────
# SURFACE TEXT SENTINEL
#
# Level 0 nodes store raw text in the `raw` field.
# Level N>0 nodes derive their surface form from components.
# This sentinel marks that derivation is needed.
# ─────────────────────────────────────────────

DERIVED = "__derived__"


# ─────────────────────────────────────────────
# SLOT SENTINEL — used in hyperedge structure
# ─────────────────────────────────────────────

SLOT = "__slot__"


# ─────────────────────────────────────────────
# TIMESTAMP
# ─────────────────────────────────────────────

def now() -> float:
    return time.time()


# ─────────────────────────────────────────────
# PMI — pointwise mutual information
#
# Measures how much two things belong together
# beyond what chance would predict.
#
# PMI(A,B) = log( P(AB) / (P(A) * P(B)) )
#
# High PMI = the combination is non-accidental.
# Low PMI  = the combination is coincidental.
#
# This is the compression signal.
# ─────────────────────────────────────────────

import math

def pmi(count_ab: int, count_a: int, count_b: int, total: int) -> float:
    """
    Pointwise mutual information between A and B.
    Returns value in [-inf, +inf]. Positive = meaningful co-occurrence.
    Normalized to [0, 1] for use as compression signal.
    """
    if count_ab == 0 or count_a == 0 or count_b == 0 or total == 0:
        return 0.0

    p_ab = count_ab / total
    p_a  = count_a  / total
    p_b  = count_b  / total

    raw = math.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)

    # Normalize: PMI ranges from -inf to log(1/p_ab)
    # We clip and map to [0, 1]
    clipped = max(-5.0, min(5.0, raw))
    return (clipped + 5.0) / 10.0


# ─────────────────────────────────────────────
# NOVELTY — how different is this from what we know
#
# Used by ingestion as the boundary test.
# High novelty = crosses the membrane.
# Low novelty  = just reinforcement.
# ─────────────────────────────────────────────

def novelty(best_match_weight: float) -> float:
    """
    Novelty is the complement of the best match strength.
    If the graph has a perfect match (weight=1.0), novelty=0.
    If the graph has no match (weight=0.0), novelty=1.
    """
    return 1.0 - best_match_weight


if __name__ == "__main__":
    print("=== atoms.py ===\n")
    print(f"BASE_DECAY           : {BASE_DECAY}")
    print(f"COMPRESSION_THRESHOLD: {COMPRESSION_THRESHOLD}")
    print(f"BOUNDARY_SENSITIVITY : {BOUNDARY_SENSITIVITY}")
    print()

    print("Decay rates by level:")
    for level in range(6):
        print(f"  level {level}: {decay_rate(level):.6f}")
    print()

    print("PMI examples:")
    # Strong co-occurrence
    print(f"  Strong: {pmi(80, 100, 100, 1000):.4f}")
    # Weak co-occurrence
    print(f"  Weak:   {pmi(3,  100, 100, 1000):.4f}")
    # No co-occurrence
    print(f"  None:   {pmi(0,  100, 100, 1000):.4f}")
    print()

    print("Novelty examples:")
    print(f"  Perfect match : {novelty(1.0):.4f}")
    print(f"  Partial match : {novelty(0.6):.4f}")
    print(f"  No match      : {novelty(0.0):.4f}")
