"""
reward.py — Internal Reward System with Hierarchical Propagation.

Fixed in this version:
  - Maturity grows from first cycle: n/(n+50)
    After 1 cycle:  0.02
    After 50 cycles: 0.50
    After 500 cycles: 0.91
  - TOKEN_REWARD_AMOUNT increased to 0.05 (stronger signal)
  - No minimum observation count for token reward ratio
  - compute_reward() is informational only — tester sets fitness
  - propagate_hierarchical() strengthens with 0.05 per correct token
"""

import math
from typing import Dict, Set, List, Optional, TYPE_CHECKING

from hgls.structures import GenerativeStructure

if TYPE_CHECKING:
    from hgls.library import Library

TOKEN_REWARD_AMOUNT  = 0.05   # stronger signal per correct token
TOKEN_PENALTY_AMOUNT = 0.05
CHILD_DECAY          = 0.6    # reward decays going down hierarchy


class InternalRewardSystem:

    def __init__(self):
        self._seen_elements: Set[str]              = set()
        self._level_score_history: Dict[int, list] = {}
        self._level_visit_count:   Dict[int, int]  = {}
        self._reward_log: list = []
        self.total_reward = 0.0
        self.maturity     = 0.0

    def compute_reward(
        self,
        structure: GenerativeStructure,
        outcome:   str,
        score:     float,
    ) -> float:
        """
        Compute intrinsic reward. Informational — does not modify fitness.
        Returns reward value. Maturity grows from first call.
        """
        nov  = self._novelty(structure)
        comp = self._competence(structure.level, score)
        cur  = self._curiosity(structure.level)
        base = 0.4 * nov + 0.4 * comp + 0.2 * cur

        if outcome == 'success':
            total = base * 3.0   # stronger positive signal
        elif outcome == 'failure':
            total = base * 0.1
        else:
            total = base * 0.02

        self.total_reward += total
        self._reward_log.append({'outcome': outcome, 'score': score, 'reward': total})
        self._update_maturity()
        return total

    def propagate_hierarchical(
        self,
        token_mask:    List[int],
        token_structs: List[Set[str]],
        library:       'Library',
    ) -> None:
        """
        Propagate token-level reward/penalty through the hierarchy.
        Stronger amounts (0.05) for clearer signal.
        """
        for mask_val, struct_ids in zip(token_mask, token_structs):
            for sid in struct_ids:
                struct = library.get(sid)
                if struct is None:
                    continue
                if mask_val == 1:
                    struct.fitness = min(1.0, struct.fitness + TOKEN_REWARD_AMOUNT)
                    struct.reward_count += 1
                    self._propagate_to_elements(struct, library, reward=True, depth=0)
                else:
                    struct.fitness = max(0.0, struct.fitness - TOKEN_PENALTY_AMOUNT)
                    struct.penalty_count += 1
                    self._propagate_to_elements(struct, library, reward=False, depth=0)

    def _propagate_to_elements(
        self,
        struct:  GenerativeStructure,
        library: 'Library',
        reward:  bool,
        depth:   int,
    ) -> None:
        if depth >= 4:
            return
        amount = TOKEN_REWARD_AMOUNT * (CHILD_DECAY ** (depth + 1))
        for elem in struct.elements:
            if isinstance(elem, str) and len(elem) > 1:
                child = library.get(elem)
                if child is None:
                    continue
                if reward:
                    child.fitness = min(1.0, child.fitness + amount)
                    child.reward_count += 1
                else:
                    child.fitness = max(0.0, child.fitness - amount)
                    child.penalty_count += 1
                self._propagate_to_elements(child, library, reward, depth + 1)

    def reinforce_correction(
        self,
        structure:   GenerativeStructure,
        topic_words: Optional[List[str]] = None,
    ) -> None:
        """Mark structure as teacher-corrected with topic tags."""
        structure.correction_count += 1
        structure.source = 'correction'
        if topic_words:
            existing = set(structure.topic_tags)
            for w in topic_words:
                if w not in existing:
                    structure.topic_tags.append(w)
        structure.fitness = min(1.0, structure.fitness + 0.15)  # stronger boost

    # ── Intrinsic components ──────────────────────────────────────

    def _novelty(self, structure: GenerativeStructure) -> float:
        key = f"{structure.level}:{structure.elements}"
        if key not in self._seen_elements:
            self._seen_elements.add(key)
            return 1.0
        return 0.05

    def _competence(self, level: int, score: float) -> float:
        hist = self._level_score_history.setdefault(level, [])
        hist.append(score)
        if len(hist) > 30:
            hist.pop(0)
        if len(hist) < 2:
            return score
        recent = sum(hist[-5:]) / len(hist[-5:])
        older  = sum(hist[:-5]) / max(1, len(hist) - 5) if len(hist) > 5 else recent
        return min(1.0, score + max(0.0, recent - older))

    def _curiosity(self, level: int) -> float:
        vc = self._level_visit_count
        vc[level] = vc.get(level, 0) + 1
        frac = vc[level] / max(1, sum(vc.values()))
        return max(0.0, 1.0 - frac * 4)

    def _update_maturity(self) -> None:
        """
        Maturity grows from first cycle.
        n/(n+50): reaches 0.5 at 50 cycles, 0.91 at 500 cycles.
        No artificial delay.
        """
        n = len(self._reward_log)
        self.maturity = n / (n + 50)

    def stats(self) -> dict:
        return {
            'total_reward':   round(self.total_reward, 3),
            'maturity':       round(self.maturity, 4),
            'unique_seen':    len(self._seen_elements),
            'level_visits':   dict(self._level_visit_count),
            'recent_rewards': [round(r['reward'], 3) for r in self._reward_log[-10:]],
        }