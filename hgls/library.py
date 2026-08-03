"""
library.py — Structured Long-Term Memory.

Stores ONLY extreme outcomes:
  - Successes : high-quality generative structures → seed future learning
  - Failures  : negative examples   → sharpen the tester

Mediocre outcomes are never stored here.
"""

from typing import Dict, List, Optional, Literal
from collections import defaultdict
from hgls.structures import GenerativeStructure

MAX_FAILURES_PER_LEVEL = 150


class Library:
    """
    Compositional long-term memory.
    All structures are keyed by their UUID.
    Lookups by level are maintained separately for speed.
    """

    def __init__(self):
        self._store: Dict[str, GenerativeStructure] = {}
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
        kind: Literal['success', 'failure', 'all'] = 'success'
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

    def success_count_at_level(self, level: int) -> int:
        return len(self._successes_by_level.get(level, []))

    def is_known_failure(self, structure: GenerativeStructure) -> bool:
        """True if an identical structure already failed at this level."""
        for sid in self._failures_by_level.get(structure.level, []):
            s = self._store.get(sid)
            if s and s.elements == structure.elements:
                return True
        return False

    # ── Storage ───────────────────────────────────────────────────

    def add_success(self, structure: GenerativeStructure) -> bool:
        """
        Store a successful structure.
        Returns True if genuinely new, False if a duplicate was updated instead.
        """
        dup = self._find_duplicate_success(structure)
        if dup:
            dup.fitness    = max(dup.fitness, structure.fitness)
            dup.test_count += 1
            return False

        self._store[structure.id] = structure
        self._successes_by_level[structure.level].append(structure.id)
        self._total_successes += 1
        return True

    def add_failure(self, structure: GenerativeStructure) -> None:
        """Store a failure as a negative example (capped per level)."""
        level = structure.level
        if len(self._failures_by_level[level]) >= MAX_FAILURES_PER_LEVEL:
            evicted = self._failures_by_level[level].pop(0)
            self._store.pop(evicted, None)

        self._store[structure.id] = structure
        self._failures_by_level[level].append(structure.id)
        self._total_failures += 1

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
        return {
            'total_successes': self._total_successes,
            'total_failures':  self._total_failures,
            'total_stored':    len(self._store),
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
        """Serialise the entire library to a JSON-safe dict."""
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
        """Reconstruct a Library from a serialised dict."""
        lib = cls()
        for sid, sd in data['structures'].items():
            s = GenerativeStructure(
                id          = sd['id'],
                level       = sd['level'],
                elements    = sd['elements'],
                source      = sd['source'],
                fitness     = sd['fitness'],
                generation  = sd['generation'],
                description = sd.get('description', ''),
                test_count  = sd.get('test_count', 0),
            )
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