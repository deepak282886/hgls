"""
explorer.py — Internal Exploration Engine

Generates novel targets by combining library structures randomly,
then feeds them to the generative unit's learn() method.

The algorithm decides what survives — not hand-coded rules.
Good combinations score high, get stored, get mutated further.
Bad combinations fail the tester and get stored as failures.
Mediocre combinations are discarded.

This is the system teaching itself through play.
"""

import random
from typing import Optional
from dataclasses import dataclass, field

from hgls.library    import Library
from hgls.tester     import ExtremeTester
from hgls.reward     import InternalRewardSystem
from hgls.curriculum import CurriculumController

CONNECTORS = ['and', 'so', 'then', 'because', 'when', '']


@dataclass
class ExplorationResult:
    combinations_tried: int = 0
    novel_discovered:   int = 0
    examples:           list = field(default_factory=list)


class ExplorationEngine:
    """
    Internal exploration: randomly compose library structures into
    novel targets, let the generative unit learn them.
    The algorithm is the filter — no hand-coded rules.
    """

    def __init__(
        self,
        library:    Library,
        tester:     ExtremeTester,
        reward:     InternalRewardSystem,
        curriculum: CurriculumController,
        llm_parent=None,
    ):
        self.library    = library
        self.curriculum = curriculum
        # Keep references for compatibility — exploration uses learn() directly
        self.tester     = tester
        self.reward     = reward
        self.llm_parent = llm_parent
        self._total_tried = 0
        self._total_novel = 0
        self._units       = None   # set by system after init

    def set_units(self, units: dict) -> None:
        """Inject generative units — called by system after init."""
        self._units = units

    # ── Public ────────────────────────────────────────────────────

    def explore(self, n: int = 60) -> ExplorationResult:
        """
        Run n exploration attempts.
        Each attempt: compose a novel target → let unit learn it.
        Starts at word level minimum — character composition produces no meaning.
        """
        result = ExplorationResult()
        # Compose at word level and above — never at character level
        level  = max(2, min(self.curriculum.get_active_level(), 4))

        for _ in range(n):
            target = self._compose(level)
            if not target:
                continue

            result.combinations_tried += 1
            self._total_tried += 1

            prev_size = len(self.library)

            # Feed to generative unit — the algorithm decides if it survives
            if self._units and level in self._units:
                self._units[level].learn(target)

            if len(self.library) > prev_size:
                result.novel_discovered += 1
                self._total_novel += 1
                if len(result.examples) < 10:
                    result.examples.append(('generated', target))

        return result

    # ── Composition ───────────────────────────────────────────────

    def _compose(self, level: int) -> Optional[str]:
        """
        Pick two random library structures and combine them.
        Simple concatenation with a random connector.
        The tester filters good from bad.
        """
        structs = self.library.get_at_level(level, kind='success')
        if len(structs) < 2:
            # Try lower level
            for lvl in range(level - 1, -1, -1):
                structs = self.library.get_at_level(lvl, kind='success')
                if len(structs) >= 2:
                    break
            if len(structs) < 2:
                return None

        s1, s2 = random.sample(structs[:200], 2)
        g1 = s1.generate(self.library).strip()
        g2 = s2.generate(self.library).strip()

        if not g1 or not g2 or g1 == g2:
            return None
        if len(g1) < 3 or len(g2) < 3:
            return None
        if len(g1) + len(g2) > 60:
            return None

        connector = random.choice(CONNECTORS)
        if connector:
            return f"{g1} {connector} {g2}"
        return f"{g1} {g2}"

    def stats(self) -> dict:
        return {
            'total_tried': self._total_tried,
            'total_novel': self._total_novel,
            'discovery_rate': round(
                self._total_novel / max(1, self._total_tried), 3
            ),
        }