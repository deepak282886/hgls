"""
structures.py — Core unit of knowledge in the HGLS.

A GenerativeStructure knows how to reconstruct some input.
Elements are either primitive characters or IDs of lower-level
structures stored in the Library.

New in this version:
  - generate_with_trace() — returns text + which struct contributed each char
  - correction_count      — how many times a teacher corrected to this structure
  - reward_count          — cumulative token-level reward hits
  - penalty_count         — cumulative token-level penalty hits
  - topic_tags            — topics this structure is relevant to
  - effective_fitness()   — fitness boosted by correction history for matching topics
"""

import uuid
import random
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.library import Library


@dataclass
class GenerativeStructure:
    """
    A structure that can reconstruct (generate) some input.

    Elements are either:
      - Primitive characters  (single-char str)
      - Structure IDs         (8-char hex str referencing the Library)

    Level:
      0 = characters
      1 = combinations / syllables
      2 = words
      3 = phrases / sentences
      4 = schemas
      5 = reasoning patterns
      6 = meta-reasoning
    """
    id: str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: int          = 0
    elements: List[Any] = field(default_factory=list)
    source: str         = "generated"   # generated|mutated|abstracted|llm|bootstrap|correction
    fitness: float      = 0.0
    generation: int     = 0
    description: str    = ""
    test_count: int     = 0

    # Reward tracking
    correction_count: int = 0
    reward_count: int     = 0
    penalty_count: int    = 0
    topic_tags: List[str] = field(default_factory=list)

    # ── Core generation ───────────────────────────────────────────

    def generate(self, library: Optional['Library'] = None) -> str:
        """Reconstruct output by recursively expanding all elements."""
        parts = []
        for elem in self.elements:
            if isinstance(elem, str) and len(elem) == 1:
                parts.append(elem)
            elif library is not None and library.has(elem):
                parts.append(library.get(elem).generate(library))
            else:
                parts.append(str(elem))
        return ''.join(parts)

    def generate_with_trace(
        self,
        library: Optional['Library'] = None,
        _depth: int = 0,
    ) -> Tuple[str, List[Tuple[str, int, int]]]:
        """
        Reconstruct output AND record which structure contributed each character.

        Returns
        -------
        text  : the generated string
        trace : list of (struct_id, char_start, char_end)
                Every character is attributed to its contributing structures.
                A character can appear in multiple trace entries (child + parent).

        This enables hierarchical reward propagation: when a token at
        positions [s, e] is correct, every struct_id in that span is rewarded
        all the way down the hierarchy.
        """
        if _depth > 12:
            return '', []

        parts: List[str] = []
        trace: List[Tuple[str, int, int]] = []
        pos = 0

        for elem in self.elements:
            if isinstance(elem, str) and len(elem) == 1:
                # Primitive — this structure owns this character
                parts.append(elem)
                trace.append((self.id, pos, pos + 1))
                pos += 1

            elif library is not None and library.has(elem):
                # Recurse into child structure
                child = library.get(elem)
                child_text, child_trace = child.generate_with_trace(library, _depth + 1)
                parts.append(child_text)
                for sid, cs, ce in child_trace:
                    trace.append((sid, pos + cs, pos + ce))
                # This structure also owns the whole child span
                if child_text:
                    trace.append((self.id, pos, pos + len(child_text)))
                pos += len(child_text)

            else:
                # Literal multi-char fallback
                s = str(elem)
                parts.append(s)
                trace.append((self.id, pos, pos + len(s)))
                pos += len(s)

        return ''.join(parts), trace

    # ── Contextual fitness ────────────────────────────────────────

    def effective_fitness(self, topic_words: set = None) -> float:
        """
        Fitness adjusted by correction history and token-level reward signal.

        Teacher corrections dominate for matching topics.
        Token reward history shifts fitness up or down based on
        how consistently this structure generated correct tokens.
        """
        base = self.fitness

        # Correction bonus
        if self.correction_count > 0:
            if topic_words and self.topic_tags:
                overlap = len(topic_words & set(self.topic_tags))
                boost = 0.15 * self.correction_count if overlap > 0 else 0.02 * self.correction_count
            else:
                boost = 0.05 * self.correction_count
            base = min(1.0, base + boost)

        # Token reward ratio
        total = self.reward_count + self.penalty_count
        if total >= 5:
            ratio = self.reward_count / total
            base = max(0.0, min(1.0, base + (ratio - 0.5) * 0.4))

        return base

    # ── Variation ─────────────────────────────────────────────────

    def mutate(
        self,
        primitives: List[str],
        library: Optional['Library'] = None,
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
            'id':               self.id,
            'level':            self.level,
            'elements':         self.elements,
            'source':           self.source,
            'fitness':          self.fitness,
            'generation':       self.generation,
            'description':      self.description,
            'test_count':       self.test_count,
            'correction_count': self.correction_count,
            'reward_count':     self.reward_count,
            'penalty_count':    self.penalty_count,
            'topic_tags':       self.topic_tags,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'GenerativeStructure':
        return cls(
            id               = d['id'],
            level            = d['level'],
            elements         = d['elements'],
            source           = d['source'],
            fitness          = d['fitness'],
            generation       = d['generation'],
            description      = d.get('description', ''),
            test_count       = d.get('test_count', 0),
            correction_count = d.get('correction_count', 0),
            reward_count     = d.get('reward_count', 0),
            penalty_count    = d.get('penalty_count', 0),
            topic_tags       = d.get('topic_tags', []),
        )

    def __repr__(self):
        return (
            f"GenStruct(id={self.id}, lvl={self.level}, "
            f"elems={self.elements[:6]}, fit={self.fitness:.2f}, "
            f"src={self.source}, corr={self.correction_count})"
        )
