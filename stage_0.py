"""
stage0.py — Little Deepak: Stage 0 Curriculum
==============================================

Builds the atomic bedrock across three independent streams:
  - Pixel:   synthetic 1D visual patterns (edges, gradients, pulses)
  - Letter:  English bigram/trigram statistics via synthetic streams
  - Phoneme: English phonotactic statistics via synthetic phoneme streams

No cross-stream pairing. No real images or text. No human reward.
Pure sequential statistics, carved into the graph by occurrence and reward.

Reward is fully automated and frequency-proportional within each stream.
The reward multiplier is intentionally high at Stage 0 — we want the
frequent patterns to become dramatically heavier than infrequent ones.

Readiness is measured by stability growth rate. When the top-50 structure
stabilities change less than 1% per 1000 steps across all three streams —
Stage 0 is complete.

Run:
    python stage0.py [--steps 50000] [--log-interval 1000]
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import defaultdict
from typing import Generator

import numpy as np

sys.path.insert(0, ".")
from primeval import Atoms, Config, Primeval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage0")


# ══════════════════════════════════════════════════════════════════════════════
# ENGLISH LETTER BIGRAM TABLE
# Top-100 English bigrams by frequency. Source: standard corpus linguistics.
# ══════════════════════════════════════════════════════════════════════════════

# Bigram → relative frequency (unnormalised)
LETTER_BIGRAM_FREQ: dict[tuple[str, str], float] = {
    ("t","h"): 3.56, ("h","e"): 3.07, ("i","n"): 2.43, ("e","r"): 2.05,
    ("a","n"): 1.99, ("r","e"): 1.85, ("o","n"): 1.76, ("e","n"): 1.75,
    ("a","t"): 1.49, ("e","s"): 1.45, ("e","d"): 1.44, ("t","i"): 1.39,
    ("o","r"): 1.35, ("s","t"): 1.34, ("n","t"): 1.33, ("t","o"): 1.32,
    ("i","t"): 1.23, ("n","d"): 1.22, ("s","e"): 1.22, ("a","l"): 1.20,
    ("o","u"): 1.18, ("h","a"): 1.17, ("n","g"): 1.14, ("a","s"): 1.13,
    ("i","s"): 1.12, ("h","i"): 1.09, ("r","s"): 1.08, ("i","o"): 1.07,
    ("n","e"): 1.06, ("f","o"): 1.04, ("l","e"): 1.03, ("t","e"): 1.02,
    ("o","f"): 1.01, ("r","o"): 1.00, ("l","l"): 0.97, ("a","r"): 0.94,
    ("t","s"): 0.94, ("d","e"): 0.93, ("a","c"): 0.91, ("e","t"): 0.90,
    ("o","m"): 0.89, ("r","i"): 0.87, ("s","s"): 0.86, ("v","e"): 0.85,
    ("a","m"): 0.83, ("c","e"): 0.82, ("l","y"): 0.81, ("c","o"): 0.80,
    ("u","r"): 0.80, ("l","i"): 0.79, ("a","b"): 0.78, ("e","a"): 0.77,
    ("i","c"): 0.76, ("e","l"): 0.75, ("i","l"): 0.75, ("m","e"): 0.74,
    ("o","s"): 0.73, ("s","o"): 0.72, ("t","r"): 0.71, ("p","r"): 0.70,
    ("c","h"): 0.70, ("u","l"): 0.69, ("p","e"): 0.68, ("a","d"): 0.67,
    ("w","h"): 0.66, ("w","i"): 0.65, ("u","t"): 0.64, ("r","n"): 0.63,
    ("e","e"): 0.62, ("f","r"): 0.61, ("u","s"): 0.60, ("o","t"): 0.59,
    ("b","e"): 0.58, ("d","i"): 0.57, ("g","h"): 0.56, ("g","e"): 0.55,
    ("s","h"): 0.54, ("i","e"): 0.53, ("o","l"): 0.52, ("n","o"): 0.51,
    ("a","i"): 0.50, ("a","g"): 0.49, ("i","a"): 0.48, ("n","s"): 0.47,
    ("c","a"): 0.46, ("a","p"): 0.45, ("u","n"): 0.44, ("p","l"): 0.43,
    ("w","a"): 0.42, ("y","o"): 0.41, ("r","a"): 0.40, ("t","t"): 0.39,
    ("e","r"): 0.38, ("o","w"): 0.37, ("u","e"): 0.36, ("g","r"): 0.35,
    ("m","a"): 0.34, ("f","t"): 0.33, ("d","a"): 0.32, ("l","o"): 0.31,
}

# Build transition matrix: for each letter, probability distribution over next
def _build_letter_transitions() -> dict[str, list[tuple[str, float]]]:
    from_dist: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (a, b), freq in LETTER_BIGRAM_FREQ.items():
        from_dist[a].append((b, freq))
    # normalise each row
    result: dict[str, list[tuple[str, float]]] = {}
    for ch, targets in from_dist.items():
        total = sum(f for _, f in targets)
        result[ch] = [(t, f / total) for t, f in targets]
    return result

LETTER_TRANSITIONS = _build_letter_transitions()

# Reward lookup: bigram → normalised reward (0..1)
_max_freq = max(LETTER_BIGRAM_FREQ.values())
LETTER_BIGRAM_REWARD: dict[tuple[str,str], float] = {
    k: v / _max_freq for k, v in LETTER_BIGRAM_FREQ.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# ENGLISH PHONEME BIGRAM TABLE
# Derived from CMU Pronouncing Dictionary phonotactics.
# ══════════════════════════════════════════════════════════════════════════════

PHONEME_BIGRAM_FREQ: dict[tuple[str,str], float] = {
    # Common vowel → consonant transitions
    ("AH","N"): 3.2,  ("AH","T"): 3.0,  ("AH","S"): 2.8,  ("AH","L"): 2.6,
    ("AH","R"): 2.5,  ("AH","D"): 2.4,  ("AH","Z"): 2.2,  ("AH","M"): 2.0,
    ("IH","N"): 3.0,  ("IH","T"): 2.8,  ("IH","NG"): 2.5, ("IH","S"): 2.3,
    ("IH","Z"): 2.1,  ("IH","K"): 1.9,  ("IH","D"): 1.8,  ("IH","L"): 1.7,
    ("EH","N"): 2.5,  ("EH","S"): 2.3,  ("EH","D"): 2.1,  ("EH","R"): 2.0,
    ("AE","N"): 2.4,  ("AE","T"): 2.2,  ("AE","K"): 2.0,  ("AE","S"): 1.8,
    ("IY","N"): 2.3,  ("IY","T"): 2.1,  ("IY","Z"): 1.9,  ("IY","L"): 1.7,
    ("OW","N"): 2.0,  ("OW","T"): 1.8,  ("OW","Z"): 1.6,  ("OW","R"): 1.5,
    ("UW","T"): 1.8,  ("UW","N"): 1.6,  ("UW","Z"): 1.5,  ("UW","L"): 1.4,
    ("ER","N"): 2.2,  ("ER","T"): 2.0,  ("ER","S"): 1.8,  ("ER","D"): 1.6,
    # Common consonant → vowel transitions
    ("T","AH"): 3.1,  ("T","IH"): 2.9,  ("T","EH"): 2.5,  ("T","AE"): 2.2,
    ("N","AH"): 2.8,  ("N","IH"): 2.6,  ("N","EH"): 2.2,  ("N","OW"): 2.0,
    ("S","AH"): 2.7,  ("S","IH"): 2.5,  ("S","EH"): 2.1,  ("S","T"): 2.8,
    ("R","AH"): 2.6,  ("R","IH"): 2.4,  ("R","EH"): 2.0,  ("R","OW"): 1.9,
    ("L","AH"): 2.5,  ("L","IH"): 2.3,  ("L","EH"): 1.9,  ("L","OW"): 1.8,
    ("K","AH"): 2.3,  ("K","AE"): 2.1,  ("K","IH"): 1.9,  ("K","OW"): 1.8,
    ("D","AH"): 2.2,  ("D","IH"): 2.0,  ("D","EH"): 1.8,  ("D","AE"): 1.6,
    ("M","AH"): 2.1,  ("M","IH"): 1.9,  ("M","EH"): 1.7,  ("M","AE"): 1.6,
    ("P","AH"): 1.9,  ("P","IH"): 1.7,  ("P","EH"): 1.5,  ("P","R"): 1.8,
    ("F","AH"): 1.8,  ("F","IH"): 1.6,  ("F","R"): 1.7,   ("F","OW"): 1.5,
    # Common consonant clusters
    ("S","T"): 2.8,   ("S","P"): 2.2,   ("S","K"): 2.0,   ("S","N"): 1.6,
    ("T","R"): 2.4,   ("D","R"): 2.1,   ("P","R"): 2.0,   ("G","R"): 1.8,
    ("N","T"): 2.6,   ("N","D"): 2.3,   ("N","Z"): 2.1,   ("N","K"): 1.8,
    ("NG","Z"): 2.0,  ("NG","D"): 1.7,  ("L","D"): 1.9,   ("L","Z"): 1.8,
}

def _build_phoneme_transitions() -> dict[str, list[tuple[str, float]]]:
    from_dist: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (a, b), freq in PHONEME_BIGRAM_FREQ.items():
        from_dist[a].append((b, freq))
    result: dict[str, list[tuple[str, float]]] = {}
    for ph, targets in from_dist.items():
        total = sum(f for _, f in targets)
        result[ph] = [(t, f / total) for t, f in targets]
    return result

PHONEME_TRANSITIONS = _build_phoneme_transitions()

_max_ph_freq = max(PHONEME_BIGRAM_FREQ.values())
PHONEME_BIGRAM_REWARD: dict[tuple[str,str], float] = {
    k: v / _max_ph_freq for k, v in PHONEME_BIGRAM_FREQ.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# PIXEL PATTERN GENERATORS
# 1D patterns — the visual primitives. 20 pattern types.
# Each pattern is a sequence of grayscale values (0-255).
# ══════════════════════════════════════════════════════════════════════════════

PATTERN_LENGTH = 16   # atoms per pixel sequence
N_PIXEL_PATTERNS = 20

def _make_pixel_patterns() -> list[np.ndarray]:
    """
    Generate 20 canonical pixel patterns:
      0-3:   rising edges (different slopes)
      4-7:   falling edges (different slopes)
      8-11:  pulses (narrow spike, wide spike, double spike, dip)
      12-15: gradients (linear up, linear down, sine, plateau)
      16-19: alternating (fast, slow, sawtooth, square wave)
    """
    L = PATTERN_LENGTH
    x = np.linspace(0, 1, L)
    patterns = []

    # Rising edges
    for steepness in [2, 4, 8, 16]:
        p = 1 / (1 + np.exp(-steepness * (x - 0.5)))
        patterns.append((p * 255).astype(np.uint8))

    # Falling edges
    for steepness in [2, 4, 8, 16]:
        p = 1 / (1 + np.exp(steepness * (x - 0.5)))
        patterns.append((p * 255).astype(np.uint8))

    # Pulses
    pulse_narrow = np.zeros(L, dtype=np.uint8)
    pulse_narrow[L//2] = 255
    patterns.append(pulse_narrow)

    pulse_wide = np.zeros(L, dtype=np.uint8)
    pulse_wide[L//3: 2*L//3] = 200
    patterns.append(pulse_wide)

    double_pulse = np.zeros(L, dtype=np.uint8)
    double_pulse[L//4] = 230
    double_pulse[3*L//4] = 230
    patterns.append(double_pulse)

    dip = np.full(L, 200, dtype=np.uint8)
    dip[L//2] = 30
    patterns.append(dip)

    # Gradients and waves
    patterns.append((x * 255).astype(np.uint8))                          # linear up
    patterns.append(((1 - x) * 255).astype(np.uint8))                    # linear down
    patterns.append(((np.sin(2 * np.pi * x) * 0.5 + 0.5) * 255).astype(np.uint8))  # sine
    plateau = np.zeros(L, dtype=np.uint8)
    plateau[L//4: 3*L//4] = 200
    patterns.append(plateau)                                              # plateau

    # Alternating
    fast_alt = np.array([255 if i % 2 == 0 else 0 for i in range(L)], dtype=np.uint8)
    patterns.append(fast_alt)

    slow_alt = np.array([255 if i % 4 < 2 else 0 for i in range(L)], dtype=np.uint8)
    patterns.append(slow_alt)

    sawtooth = (np.linspace(0, 1, L) % (1/4) * 4 * 255).astype(np.uint8)
    patterns.append(sawtooth)

    square = np.array([200 if np.sin(4 * np.pi * i / L) >= 0 else 50
                       for i in range(L)], dtype=np.uint8)
    patterns.append(square)

    assert len(patterns) == N_PIXEL_PATTERNS, f"Expected 20, got {len(patterns)}"
    return patterns

PIXEL_PATTERNS = _make_pixel_patterns()

# Reward per pattern — uniform at Stage 0, all patterns are equally valid
PIXEL_PATTERN_REWARD = 1.0

def sample_pixel_pattern(noise_std: float = 8.0) -> tuple[list[int], float]:
    """Sample a random pixel pattern with slight noise. Returns (atoms, reward)."""
    idx = random.randint(0, N_PIXEL_PATTERNS - 1)
    pattern = PIXEL_PATTERNS[idx].astype(np.float32)
    noise   = np.random.normal(0, noise_std, len(pattern))
    noisy   = np.clip(pattern + noise, 0, 255).astype(np.uint8)
    atoms   = Atoms.sequence("pixel", noisy.tolist())
    return atoms, PIXEL_PATTERN_REWARD


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def generate_letter_sequence(length: int = 8) -> tuple[list[int], float]:
    """
    Sample a letter sequence by walking the bigram transition matrix.
    Returns (atoms, cumulative_reward).
    Reward is the mean bigram reward across the sequence.
    """
    # Start from a random letter that has outgoing transitions
    starters = list(LETTER_TRANSITIONS.keys())
    ch = random.choice(starters)
    letters = [ch]
    total_reward = 0.0
    n_bigrams = 0

    for _ in range(length - 1):
        transitions = LETTER_TRANSITIONS.get(ch)
        if not transitions:
            # Dead end — pick a new random starter
            ch = random.choice(starters)
            letters.append(ch)
            continue
        targets, probs = zip(*transitions)
        next_ch = random.choices(targets, weights=probs, k=1)[0]
        bigram_reward = LETTER_BIGRAM_REWARD.get((ch, next_ch), 0.0)
        total_reward += bigram_reward
        n_bigrams += 1
        letters.append(next_ch)
        ch = next_ch

    atoms = Atoms.sequence("letter", letters)
    mean_reward = total_reward / n_bigrams if n_bigrams > 0 else 0.0
    return atoms, mean_reward


def generate_phoneme_sequence(length: int = 6) -> tuple[list[int], float]:
    """
    Sample a phoneme sequence by walking the phoneme transition matrix.
    Returns (atoms, cumulative_reward).
    """
    starters = list(PHONEME_TRANSITIONS.keys())
    ph = random.choice(starters)
    phonemes = [ph]
    total_reward = 0.0
    n_bigrams = 0

    for _ in range(length - 1):
        transitions = PHONEME_TRANSITIONS.get(ph)
        if not transitions:
            ph = random.choice(starters)
            phonemes.append(ph)
            continue
        targets, probs = zip(*transitions)
        next_ph = random.choices(targets, weights=probs, k=1)[0]
        bigram_reward = PHONEME_BIGRAM_REWARD.get((ph, next_ph), 0.0)
        total_reward += bigram_reward
        n_bigrams += 1
        phonemes.append(next_ph)
        ph = next_ph

    atoms = Atoms.sequence("phoneme", phonemes)
    mean_reward = total_reward / n_bigrams if n_bigrams > 0 else 0.0
    return atoms, mean_reward


# ══════════════════════════════════════════════════════════════════════════════
# READINESS MONITOR
# Tracks stability growth rate across all three streams.
# Stage 0 is ready when top-50 stabilities change < 1% per 1000 steps.
# ══════════════════════════════════════════════════════════════════════════════

class ReadinessMonitor:
    def __init__(self, window: int = 1000, threshold: float = 0.01):
        self.window    = window
        self.threshold = threshold
        self._prev:    dict[str, float] = {}
        self._history: dict[str, list[float]] = defaultdict(list)

    def snapshot(self, system: Primeval) -> dict[str, float]:
        """
        Compute mean stability of top-50 structures per level.
        Returns rate of change since last snapshot.
        """
        g = system.graph
        rates: dict[str, float] = {}

        for stream in ["letter", "phoneme", "pixel"]:
            if stream == "letter":
                relevant = [
                    nid for nid in g.nodes_at_level(1)
                    if g._nodes.get(nid) and all(
                        Atoms.LETTER_OFF <= c < Atoms.PHONEME_OFF
                        for c in g._nodes[nid].constituents
                    )
                ]
            elif stream == "phoneme":
                relevant = [
                    nid for nid in g.nodes_at_level(1)
                    if g._nodes.get(nid) and all(
                        Atoms.PHONEME_OFF <= c < Atoms.STRUCT_START
                        for c in g._nodes[nid].constituents
                    )
                ]
            else:
                relevant = [
                    nid for nid in g.nodes_at_level(1)
                    if g._nodes.get(nid) and all(
                        c < Atoms.LETTER_OFF
                        for c in g._nodes[nid].constituents
                    )
                ]

            if not relevant:
                rates[stream] = 1.0
                continue

            top50 = sorted(relevant, key=lambda n: -g.stability(n))[:50]
            mean_stab = np.mean([g.stability(n) for n in top50])

            key = stream
            if key in self._prev:
                prev = self._prev[key]
                rate = abs(mean_stab - prev) / (prev + 1e-9)
                rates[stream] = rate
                self._history[stream].append(rate)
            else:
                rates[stream] = 1.0
            self._prev[key] = mean_stab

        return rates

    def is_ready(self, rates: dict[str, float]) -> bool:
        """All streams below threshold rate of change."""
        if len(rates) < 3:
            return False
        return all(r < self.threshold for r in rates.values())


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Stage0Trainer:
    """
    Runs Stage 0 training for Little Deepak.

    Each step ingests one sequence from one stream, applies frequency-
    proportional reward, and logs progress every log_interval steps.

    Streams are interleaved round-robin: pixel → letter → phoneme → repeat.
    This keeps all three streams growing at roughly the same pace.

    The reward_scale parameter controls how aggressively reward is applied
    at Stage 0. Higher = more dramatic gradient between frequent and
    infrequent patterns. Default 5.0 — significantly higher than later stages.
    """

    def __init__(
        self,
        system: Primeval,
        reward_scale:   float = 5.0,
        log_interval:   int   = 1000,
        check_interval: int   = 1000,
    ):
        self.system         = system
        self.reward_scale   = reward_scale
        self.log_interval   = log_interval
        self.check_interval = check_interval
        self.monitor        = ReadinessMonitor(window=1000, threshold=0.01)
        self._stream_cycle  = 0
        self._rates: dict[str, float] = {}

    def step(self) -> str:
        """One training step. Returns which stream was ingested."""
        stream = ["pixel", "letter", "phoneme"][self._stream_cycle % 3]
        self._stream_cycle += 1

        if stream == "pixel":
            atoms, raw_reward = sample_pixel_pattern()
        elif stream == "letter":
            atoms, raw_reward = generate_letter_sequence(length=8)
        else:
            atoms, raw_reward = generate_phoneme_sequence(length=6)

        self.system.ingest_atoms(atoms)
        if raw_reward > 0.0:
            self.system.reward(raw_reward * self.reward_scale)

        return stream

    def run(self, total_steps: int) -> bool:
        """
        Run Stage 0 for up to total_steps steps.
        Returns True if readiness condition was met, False if steps exhausted.
        """
        logger.info("Stage 0 starting. Target steps: %d", total_steps)
        t0 = time.monotonic()

        for step in range(1, total_steps + 1):
            self.step()

            if step % self.check_interval == 0:
                self._rates = self.monitor.snapshot(self.system)
                elapsed = time.monotonic() - t0
                self._log(step, elapsed)

                if self.monitor.is_ready(self._rates):
                    logger.info(
                        "✓ Stage 0 READY at step %d (%.1fs). "
                        "Stability growth rates: %s",
                        step, elapsed,
                        {k: f"{v:.4f}" for k, v in self._rates.items()}
                    )
                    return True

        logger.info(
            "Stage 0 ended at step limit %d. "
            "Final rates: %s (threshold=0.01)",
            total_steps,
            {k: f"{v:.4f}" for k, v in self._rates.items()}
        )
        return False

    def _log(self, step: int, elapsed: float) -> None:
        g    = self.system.graph
        stat = self.system.stats()

        # Top-5 edges per stream
        letter_edges = sorted(
            [(k, v) for k, v in g._weights.items()
             if g._nodes.get(k[0]) and g._nodes.get(k[0]).level == 0
             and Atoms.LETTER_OFF <= k[0] < Atoms.PHONEME_OFF
             and Atoms.LETTER_OFF <= k[1] < Atoms.PHONEME_OFF],
            key=lambda x: -x[1]
        )[:5]

        phoneme_edges = sorted(
            [(k, v) for k, v in g._weights.items()
             if g._nodes.get(k[0]) and g._nodes.get(k[0]).level == 0
             and Atoms.PHONEME_OFF <= k[0] < Atoms.STRUCT_START
             and Atoms.PHONEME_OFF <= k[1] < Atoms.STRUCT_START],
            key=lambda x: -x[1]
        )[:5]

        pixel_edges = sorted(
            [(k, v) for k, v in g._weights.items()
             if g._nodes.get(k[0]) and g._nodes.get(k[0]).level == 0
             and k[0] < Atoms.LETTER_OFF
             and k[1] < Atoms.LETTER_OFF],
            key=lambda x: -x[1]
        )[:3]

        logger.info(
            "Step %6d | %.1fs | nodes=%d edges=%d structs=%d max_lv=%d",
            step, elapsed,
            stat["nodes"], stat["edges"], stat["structures"], stat["max_level"]
        )
        logger.info(
            "  Rates → pixel=%.4f letter=%.4f phoneme=%.4f (ready<0.01)",
            self._rates.get("pixel", 1.0),
            self._rates.get("letter", 1.0),
            self._rates.get("phoneme", 1.0),
        )
        if letter_edges:
            logger.info("  Top letter edges:")
            for (a, b), w in letter_edges:
                la = chr(a - Atoms.LETTER_OFF)
                lb = chr(b - Atoms.LETTER_OFF)
                logger.info("    %r→%r  w=%.1f", la, lb, w)
        if phoneme_edges:
            logger.info("  Top phoneme edges:")
            from primeval import PHONEMES
            for (a, b), w in phoneme_edges:
                pa = PHONEMES[a - Atoms.PHONEME_OFF] if (a - Atoms.PHONEME_OFF) < len(PHONEMES) else "?"
                pb = PHONEMES[b - Atoms.PHONEME_OFF] if (b - Atoms.PHONEME_OFF) < len(PHONEMES) else "?"
                logger.info("    %s→%s  w=%.1f", pa, pb, w)
        if pixel_edges:
            logger.info("  Top pixel edges:")
            for (a, b), w in pixel_edges:
                logger.info("    pixel(%d)→pixel(%d)  w=%.1f", a, b, w)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Little Deepak — Stage 0")
    parser.add_argument("--steps",        type=int,   default=50_000)
    parser.add_argument("--log-interval", type=int,   default=1_000)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints/stage0")
    parser.add_argument("--resume",       action="store_true")
    args = parser.parse_args()

    cfg = Config(
        window_size            = 2,
        occurrence_delta       = 1.0,
        reward_multiplier      = 10.0,   # high at Stage 0
        downward_growth_delta  = 0.1,
        consolidator_interval  = 200,    # run often — bedrock needs bridging
        consolidator_proposal_scale = 0.01,
        consolidator_budget    = 500,    # process top-500 co-activated pairs per pass
        counts_decay_rate      = 0.01,   # stale co-activations fade at 1% per decay pass
        decay_interval         = 5_000,  # slow decay at Stage 0 — let bedrock accumulate
        base_decay_rate        = 1e-4,
        level_decay_factor     = 1.5,
        checkpoint_dir         = args.checkpoint,
        checkpoint_interval    = 10_000,
    )

    system = Primeval(cfg)

    if args.resume:
        import glob, os
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint, "step_*")))
        if checkpoints:
            latest = checkpoints[-1]
            logger.info("Resuming from %s", latest)
            system.load(latest)
        else:
            logger.info("No checkpoint found — starting fresh")

    trainer = Stage0Trainer(
        system        = system,
        reward_scale  = args.reward_scale,
        log_interval  = args.log_interval,
        check_interval= args.log_interval,
    )

    ready = trainer.run(args.steps)

    # Final checkpoint
    system.save(args.checkpoint + "/final")
    logger.info("Saved final checkpoint to %s/final", args.checkpoint)

    # Final summary
    print("\n── Stage 0 Final Summary ────────────────────────────────")
    for k, v in system.stats().items():
        print(f"  {k}: {v}")

    print("\n── Top 10 letter edges (should match English bigrams) ───")
    g = system.graph
    letter_edges = sorted(
        [(k, v) for k, v in g._weights.items()
         if g._nodes.get(k[0]) and g._nodes.get(k[0]).level == 0
         and Atoms.LETTER_OFF <= k[0] < Atoms.PHONEME_OFF
         and Atoms.LETTER_OFF <= k[1] < Atoms.PHONEME_OFF],
        key=lambda x: -x[1]
    )[:10]
    for (a, b), w in letter_edges:
        la, lb = chr(a - Atoms.LETTER_OFF), chr(b - Atoms.LETTER_OFF)
        known  = LETTER_BIGRAM_REWARD.get((la, lb), 0.0)
        print(f"  {la!r}→{lb!r}  w={w:.1f}  known_freq={known:.3f}")

    print("\n── Top 10 phoneme edges ──────────────────────────────────")
    from primeval import PHONEMES
    phoneme_edges = sorted(
        [(k, v) for k, v in g._weights.items()
         if g._nodes.get(k[0]) and g._nodes.get(k[0]).level == 0
         and Atoms.PHONEME_OFF <= k[0] < Atoms.STRUCT_START
         and Atoms.PHONEME_OFF <= k[1] < Atoms.STRUCT_START],
        key=lambda x: -x[1]
    )[:10]
    for (a, b), w in phoneme_edges:
        pa = PHONEMES[a - Atoms.PHONEME_OFF] if (a - Atoms.PHONEME_OFF) < len(PHONEMES) else "?"
        pb = PHONEMES[b - Atoms.PHONEME_OFF] if (b - Atoms.PHONEME_OFF) < len(PHONEMES) else "?"
        known = PHONEME_BIGRAM_REWARD.get((pa, pb), 0.0)
        print(f"  {pa}→{pb}  w={w:.1f}  known_freq={known:.3f}")

    print(f"\n  Stage 0 {'COMPLETE ✓' if ready else 'INCOMPLETE (step limit reached)'}")


if __name__ == "__main__":
    main()