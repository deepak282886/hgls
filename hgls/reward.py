"""
reward.py — Internal Reward System with Hierarchical Propagation.

Three intrinsic signals:
  novelty     — reward for encountering new structures
  competence  — reward for improvement over recent history
  curiosity   — reward for exploring underrepresented levels

New in this version:
  propagate_hierarchical() — given a token mask and the structures that
      contributed each token, propagate reward/penalty UP and DOWN the
      hierarchy. Correct token (1) → reward all contributing structures
      and their element-children. Wrong token (0) → penalise them.

  reinforce_correction() — mark a structure as teacher-corrected, boosting
      its effective fitness for matching topic questions.

Signal maturity grows with experience, eventually making the system
autonomous (no longer needing the LLM parent).
"""

import math
from typing import Dict, Set, List, Optional, TYPE_CHECKING

from hgls.structures import GenerativeStructure

if TYPE_CHECKING:
    from hgls.library import Library

# Reward/penalty amounts per token signal
TOKEN_REWARD_AMOUNT  = 0.02   # per correct token, applied to each contributor
TOKEN_PENALTY_AMOUNT = 0.02   # per wrong token
CHILD_DECAY          = 0.5    # reward decays by this factor per hierarchy level


class InternalRewardSystem:

    def __init__(self):
        self._seen_elements: Set[str]       = set()
        self._level_score_history: Dict[int, list] = {}
        self._level_visit_count: Dict[int, int]    = {}
        self._reward_log: list = []
        self.total_reward = 0.0
        self.maturity     = 0.0

    # ── Intrinsic reward computation ──────────────────────────────

    def compute_reward(
        self,
        structure: GenerativeStructure,
        outcome: str,
        score: float,
    ) -> float:
        """
        Compute intrinsic reward and update structure fitness accordingly.
        Now actually modifies structure.fitness (previously this was ignored).
        """
        nov  = self._novelty(structure)
        comp = self._competence(structure.level, score)
        cur  = self._curiosity(structure.level)

        base = 0.4 * nov + 0.4 * comp + 0.2 * cur

        if outcome == 'success':
            total = base * 2.0
            # Nudge fitness upward — it's already set from test score,
            # but intrinsic reward adds a small additional signal
            structure.fitness = min(1.0, structure.fitness + total * 0.05)
        elif outcome == 'failure':
            total = base * 0.1
            structure.fitness = max(0.0, structure.fitness - total * 0.05)
        else:
            total = base * 0.05
            # Mediocre: no fitness change — mediocre leaves no trace

        self.total_reward += total
        self._reward_log.append({'outcome': outcome, 'score': score, 'reward': total})
        self._update_maturity()
        return total

    # ── Hierarchical token-level propagation ─────────────────────

    def propagate_hierarchical(
        self,
        token_mask: List[int],
        token_structs: List[Set[str]],
        library: 'Library',
    ) -> None:
        """
        Parent-style token-by-token feedback propagated through the hierarchy.

        For each token:
          mask=1 (correct) → reward every contributing structure + their children
          mask=0 (wrong)   → penalise every contributing structure + their children

        The reward/penalty decays as it propagates downward through levels,
        so higher-level structures absorb more signal than primitives.

        This is how a parent teaches a child:
        "Good... good... no that word was wrong... good..."
        and the child knows exactly which connection needs fixing.
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
        struct: GenerativeStructure,
        library: 'Library',
        reward: bool,
        depth: int,
    ) -> None:
        """
        Recursively propagate reward/penalty down to child structures.
        Amount decays by CHILD_DECAY per level so primitives get small signals.
        """
        if depth >= 4:
            return

        amount = TOKEN_REWARD_AMOUNT * (CHILD_DECAY ** (depth + 1))

        for elem in struct.elements:
            # Only follow structure IDs (not primitive single chars)
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

    # ── Teacher correction reinforcement ──────────────────────────

    def reinforce_correction(
        self,
        structure: GenerativeStructure,
        topic_words: Optional[List[str]] = None,
    ) -> None:
        """
        Mark a structure as teacher-corrected.
        Increments correction_count and tags it with topic words.
        This boosts effective_fitness() for matching topic queries.
        """
        structure.correction_count += 1
        structure.source = 'correction'

        if topic_words:
            existing = set(structure.topic_tags)
            for w in topic_words:
                if w not in existing:
                    structure.topic_tags.append(w)

        # Immediate fitness boost for teacher-corrected structures
        structure.fitness = min(1.0, structure.fitness + 0.1)

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

    def _update_maturity(self) -> None:
        n = len(self._reward_log)
        self.maturity = 1.0 / (1.0 + math.exp(-0.01 * (n - 300)))

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'total_reward':   round(self.total_reward, 3),
            'maturity':       round(self.maturity, 4),
            'unique_seen':    len(self._seen_elements),
            'level_visits':   dict(self._level_visit_count),
            'recent_rewards': [round(r['reward'], 3) for r in self._reward_log[-10:]],
        }
