"""
reward.py — Internal Reward System.

Supplies intrinsic motivation from three sources:
  novelty     — reward for encountering new structures
  competence  — reward for improvement over recent history
  curiosity   — reward for exploring underrepresented levels

Starts minimal; grows stronger as the system matures.
Maturity is used externally to decay LLM parental dependence.
"""

import math
from typing import Dict, Set
from hgls.structures import GenerativeStructure


class InternalRewardSystem:

    def __init__(self):
        self._seen_elements: Set[str] = set()
        self._level_score_history: Dict[int, list] = {}
        self._level_visit_count:   Dict[int, int]  = {}
        self._reward_log: list = []
        self.total_reward = 0.0
        self.maturity = 0.0   # 0 = early dev, 1 = fully internal

    # ── Public API ────────────────────────────────────────────────

    def compute_reward(
        self,
        structure: GenerativeStructure,
        outcome: str,
        score: float,
    ) -> float:
        """Return total intrinsic reward for this event."""
        nov  = self._novelty(structure)
        comp = self._competence(structure.level, score)
        cur  = self._curiosity(structure.level)

        base = 0.4 * nov + 0.4 * comp + 0.2 * cur

        # Extreme outcomes get amplified reward (spec requirement)
        if outcome == 'success':
            total = base * 2.0
        elif outcome == 'failure':
            total = base * 0.1
        else:                      # mediocre
            total = base * 0.05

        self.total_reward += total
        self._reward_log.append({'outcome': outcome, 'score': score, 'reward': total})
        self._update_maturity()
        return total

    # ── Components ────────────────────────────────────────────────

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
        if len(hist) < 3:
            return score
        recent = sum(hist[-5:])  / len(hist[-5:])
        older  = sum(hist[:-5]) / max(1, len(hist) - 5) if len(hist) > 5 else recent
        return min(1.0, score + max(0.0, recent - older))

    def _curiosity(self, level: int) -> float:
        vc = self._level_visit_count
        vc[level] = vc.get(level, 0) + 1
        frac = vc[level] / max(1, sum(vc.values()))
        return max(0.0, 1.0 - frac * 4)

    def _update_maturity(self):
        n = len(self._reward_log)
        self.maturity = 1.0 / (1.0 + math.exp(-0.01 * (n - 300)))

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'total_reward':    round(self.total_reward, 3),
            'maturity':        round(self.maturity, 4),
            'unique_seen':     len(self._seen_elements),
            'level_visits':    dict(self._level_visit_count),
            'recent_rewards':  [round(r['reward'], 3) for r in self._reward_log[-10:]],
        }