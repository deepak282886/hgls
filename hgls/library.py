"""
library.py — Structured Long-Term Memory.

Stores ONLY extreme outcomes:
  - Successes : high-quality generative structures
  - Failures  : negative examples

New in this version:
  - add_correction()          — stores with correction_count incremented
  - get_by_effective_fitness()— returns structures sorted by contextual fitness
                                (teacher corrections dominate for matching topics)
  - Serialisation uses GenerativeStructure.from_dict() for all new fields
"""

from typing import Dict, List, Optional, Literal
from collections import defaultdict

from hgls.structures import GenerativeStructure

MAX_FAILURES_PER_LEVEL = 150


class Library:
    """
    Compositional long-term memory.
    All structures keyed by UUID.
    Level lookups maintained separately for speed.
    """

    def __init__(self):
        self._store: Dict[str, GenerativeStructure]  = {}
        self._successes_by_level: Dict[int, List[str]] = defaultdict(list)
        self._failures_by_level:  Dict[int, List[str]] = defaultdict(list)
        self._total_successes = 0
        self._total_failures  = 0

    # ── Querying ──────────────────────────────────────────────────

    def has(self, struct_id: str) -> bool:
        return struct_id in self._store

    def get(self, struct_id: str) -> Optional[GenerativeStructure]:
        return self._store.get(struct_id)

    def get_at_level(
        self,
        level: int,
        kind: Literal['success', 'failure', 'all'] = 'success',
    ) -> List[GenerativeStructure]:
        result = []
        if kind in ('success', 'all'):
            for sid in self._successes_by_level.get(level, []):
                s = self._store.get(sid)
                if s:
                    result.append(s)
        if kind in ('failure', 'all'):
            for sid in self._failures_by_level.get(level, []):
                s = self._store.get(sid)
                if s:
                    result.append(s)
        return result

    def get_by_effective_fitness(
        self,
        level: int,
        topic_words: set = None,
        top_k: int = 50,
    ) -> List[GenerativeStructure]:
        """
        Return top-k structures at this level sorted by effective_fitness().
        Teacher corrections dominate for matching topic words.
        """
        structs = self.get_at_level(level, kind='success')
        scored  = sorted(
            structs,
            key=lambda s: s.effective_fitness(topic_words),
            reverse=True,
        )
        return scored[:top_k]

    def success_count_at_level(self, level: int) -> int:
        return len(self._successes_by_level.get(level, []))

    def is_known_failure(self, structure: GenerativeStructure) -> bool:
        for sid in self._failures_by_level.get(structure.level, []):
            s = self._store.get(sid)
            if s and s.elements == structure.elements:
                return True
        return False

    # ── Storage ───────────────────────────────────────────────────

    def add_success(self, structure: GenerativeStructure) -> bool:
        """Store successful structure. Returns True if genuinely new."""
        dup = self._find_duplicate_success(structure)
        if dup:
            dup.fitness    = max(dup.fitness, structure.fitness)
            dup.test_count += 1
            # Merge correction data from new structure into existing
            if structure.correction_count > 0:
                dup.correction_count += structure.correction_count
                for tag in structure.topic_tags:
                    if tag not in dup.topic_tags:
                        dup.topic_tags.append(tag)
            return False

        self._store[structure.id] = structure
        self._successes_by_level[structure.level].append(structure.id)
        self._total_successes += 1
        return True

    def add_failure(self, structure: GenerativeStructure) -> None:
        """Store failure as negative example (capped per level)."""
        level = structure.level
        if len(self._failures_by_level[level]) >= MAX_FAILURES_PER_LEVEL:
            evicted = self._failures_by_level[level].pop(0)
            self._store.pop(evicted, None)

        self._store[structure.id] = structure
        self._failures_by_level[level].append(structure.id)
        self._total_failures += 1

    def add_correction(
        self,
        structure: GenerativeStructure,
        topic_words: List[str] = None,
    ) -> bool:
        """
        Store a teacher-corrected structure.
        Increments correction_count and tags with topic words.
        Corrections dominate over exploration for matching topics.
        """
        structure.correction_count += 1
        structure.source = 'correction'
        if topic_words:
            existing = set(structure.topic_tags)
            for w in topic_words:
                if w not in existing:
                    structure.topic_tags.append(w)
        # Corrections enter with a fitness boost
        structure.fitness = min(1.0, structure.fitness + 0.1)
        return self.add_success(structure)

    # ── Internal helpers ──────────────────────────────────────────

    def _find_duplicate_success(
        self, structure: GenerativeStructure
    ) -> Optional[GenerativeStructure]:
        for sid in self._successes_by_level.get(structure.level, []):
            s = self._store.get(sid)
            if s and s.elements == structure.elements:
                return s
        return None

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        levels = sorted(
            set(list(self._successes_by_level) + list(self._failures_by_level))
        )
        corrections = sum(
            1 for s in self._store.values() if s.correction_count > 0
        )
        return {
            'total_successes': self._total_successes,
            'total_failures':  self._total_failures,
            'total_stored':    len(self._store),
            'corrections':     corrections,
            'by_level': {
                lvl: {
                    'successes': len(self._successes_by_level.get(lvl, [])),
                    'failures':  len(self._failures_by_level.get(lvl, [])),
                }
                for lvl in levels
            },
        }

    def __len__(self):
        return len(self._store)

    # ── Persistence ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'structures': {sid: s.to_dict() for sid, s in self._store.items()},
            'successes_by_level': {
                str(k): v for k, v in self._successes_by_level.items()
            },
            'failures_by_level': {
                str(k): v for k, v in self._failures_by_level.items()
            },
            'total_successes': self._total_successes,
            'total_failures':  self._total_failures,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Library':
        lib = cls()
        for sid, sd in data['structures'].items():
            # Use from_dict to pick up all new fields (correction_count etc.)
            s = GenerativeStructure.from_dict(sd)
            lib._store[sid] = s
        lib._successes_by_level = defaultdict(list, {
            int(k): v for k, v in data['successes_by_level'].items()
        })
        lib._failures_by_level = defaultdict(list, {
            int(k): v for k, v in data['failures_by_level'].items()
        })
        lib._total_successes = data['total_successes']
        lib._total_failures  = data['total_failures']
        return lib
