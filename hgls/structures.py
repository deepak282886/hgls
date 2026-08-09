"""
structures.py — Core unit of knowledge in the HGLS.

Key fix in this version:
  effective_fitness() — corrections now dominate with a strong multiplier.
  A teacher-corrected structure for a matching topic will always
  score higher than any incidentally matching structure.
"""

import uuid
import random
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.library import Library


@dataclass
class GenerativeStructure:
    id: str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: int          = 0
    elements: List[Any] = field(default_factory=list)
    source: str         = "generated"
    fitness: float      = 0.0
    generation: int     = 0
    description: str    = ""
    test_count: int     = 0

    correction_count: int = 0
    reward_count: int     = 0
    penalty_count: int    = 0
    topic_tags: List[str] = field(default_factory=list)

    # ── Generation ────────────────────────────────────────────────

    def generate(self, library: Optional['Library'] = None) -> str:
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
        if _depth > 12:
            return '', []
        parts: List[str] = []
        trace: List[Tuple[str, int, int]] = []
        pos = 0
        for elem in self.elements:
            if isinstance(elem, str) and len(elem) == 1:
                parts.append(elem)
                trace.append((self.id, pos, pos + 1))
                pos += 1
            elif library is not None and library.has(elem):
                child = library.get(elem)
                child_text, child_trace = child.generate_with_trace(library, _depth + 1)
                parts.append(child_text)
                for sid, cs, ce in child_trace:
                    trace.append((sid, pos + cs, pos + ce))
                if child_text:
                    trace.append((self.id, pos, pos + len(child_text)))
                pos += len(child_text)
            else:
                s = str(elem)
                parts.append(s)
                trace.append((self.id, pos, pos + len(s)))
                pos += len(s)
        return ''.join(parts), trace

    # ── Contextual fitness ────────────────────────────────────────

    def effective_fitness(self, topic_words: set = None) -> float:
        """
        Fitness adjusted for corrections and topic relevance.

        Corrections dominate. A structure explicitly taught as a
        correction for a matching topic gets a strong multiplier —
        not a small nudge. This ensures corrections always surface
        above incidentally matching corpus sentences.

        Scoring:
          base fitness from reconstruction quality
          + correction bonus (large, topic-aware)
          + token reward ratio signal
        """
        base = self.fitness

        if self.correction_count > 0:
            if topic_words and self.topic_tags:
                tag_set = set(self.topic_tags)
                overlap = len(topic_words & tag_set)

                if overlap > 0:
                    # Strong bonus for matching topic corrections
                    # Each correction adds 0.5 to effective fitness (capped at 1.0)
                    # This ensures corrections always beat corpus sentences
                    base = min(1.0, base + 0.5 * self.correction_count)
                else:
                    # Small bonus for corrections on other topics
                    base = min(1.0, base + 0.05 * self.correction_count)
            else:
                # No topic context — modest bonus
                base = min(1.0, base + 0.1 * self.correction_count)

        # Token reward history — fine-grained signal after many observations
        total = self.reward_count + self.penalty_count
        if total >= 10:
            ratio  = self.reward_count / total
            base   = max(0.0, min(1.0, base + (ratio - 0.5) * 0.3))

        return base

    # ── Variation ─────────────────────────────────────────────────

    def mutate(
        self,
        primitives: List[str],
        library: Optional['Library'] = None,
    ) -> 'GenerativeStructure':
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
            f"fit={self.fitness:.2f}, corr={self.correction_count}, "
            f"tags={self.topic_tags[:3]})"
        )