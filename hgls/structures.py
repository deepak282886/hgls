"""
structures.py — the core unit of knowledge in the HGLS.

A GenerativeStructure knows how to reconstruct some input.
Elements are either primitive characters or IDs of lower-level
structures stored in the Library.
"""

import uuid
import random
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.library import Library


@dataclass
class GenerativeStructure:
    """
    A structure that can reconstruct (generate) some input.

    Elements are either:
      - Primitive characters  (single-char str)
      - Structure IDs         (8-char hex str referencing the Library)

    Level indicates hierarchical depth:
      0 = characters
      1 = combinations / syllables
      2 = words
      3 = phrases / sentences
      4 = higher schemas
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: int = 0
    elements: List[Any] = field(default_factory=list)
    source: str = "generated"   # generated | mutated | abstracted | llm | bootstrap
    fitness: float = 0.0
    generation: int = 0
    description: str = ""
    test_count: int = 0

    # ── Core capability ───────────────────────────────────────────

    def generate(self, library: Optional['Library'] = None) -> str:
        """Reconstruct output by recursively expanding all elements."""
        parts = []
        for elem in self.elements:
            if isinstance(elem, str) and len(elem) == 1:
                parts.append(elem)                           # primitive char
            elif library is not None and library.has(elem):
                parts.append(library.get(elem).generate(library))  # recurse
            else:
                parts.append(str(elem))                      # literal fallback
        return ''.join(parts)

    # ── Variation ─────────────────────────────────────────────────

    def mutate(
        self,
        primitives: List[str],
        library: Optional['Library'] = None
    ) -> 'GenerativeStructure':
        """Return a new mutated copy. Leaves this structure untouched."""
        elems = list(self.elements)
        if not elems:
            elems = [random.choice(primitives)]

        op = random.choices(
            ['insert', 'delete', 'replace', 'swap'],
            weights=[3, 2, 3, 2]
        )[0]

        if op == 'insert' and len(elems) < 24:
            idx = random.randint(0, len(elems))
            elems.insert(idx, random.choice(primitives))

        elif op == 'delete' and len(elems) > 1:
            elems.pop(random.randint(0, len(elems) - 1))

        elif op == 'replace' and elems:
            idx = random.randint(0, len(elems) - 1)
            # Occasionally replace with a lower-level library structure
            if library and random.random() < 0.25:
                lower = library.get_at_level(max(0, self.level - 1), kind='success')
                elems[idx] = random.choice(lower).id if lower else random.choice(primitives)
            else:
                elems[idx] = random.choice(primitives)

        elif op == 'swap' and len(elems) > 1:
            i, j = random.sample(range(len(elems)), 2)
            elems[i], elems[j] = elems[j], elems[i]

        return GenerativeStructure(
            level=self.level,
            elements=elems,
            source='mutated',
            generation=self.generation + 1,
        )

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'level': self.level,
            'elements': self.elements,
            'source': self.source,
            'fitness': self.fitness,
            'generation': self.generation,
            'description': self.description,
            'test_count': self.test_count,
        }

    def __repr__(self):
        return (
            f"GenStruct(id={self.id}, lvl={self.level}, "
            f"elems={self.elements[:6]}, fit={self.fitness:.2f}, src={self.source})"
        )