"""
explorer.py — Internal Exploration Engine

The hypothesis engine connecting dots between already-established
library structures to generate genuinely novel combinations.

No external input. No curriculum. Runs purely on the library.

Three strategies:
  1. Chain      — take two structures, join them into a novel compound
  2. Substitute — replace a content word in a phrase with another known word
  3. Swap       — exchange consequences between two schemas

Each novel candidate is tested for coherence (word coverage against
the library) before being stored. This ensures only plausible
combinations enter the library — not random noise.

This is the system teaching itself.
"""

import random
from typing import Optional
from dataclasses import dataclass, field

from hgls.structures import GenerativeStructure
from hgls.library    import Library
from hgls.tester     import ExtremeTester
from hgls.reward     import InternalRewardSystem
from hgls.curriculum import CurriculumController
import hgls.persona  as persona

# Words that carry grammar, not meaning — don't substitute these
FUNCTION_WORDS = {
    'i', 'my', 'the', 'a', 'to', 'and', 'when', 'they',
    'she', 'he', 'it', 'we', 'is', 'are', 'in', 'on',
    'up', 'so', 'do', 'be', 'at', 'by', 'of', 'or',
    'then', 'so', 'but', 'not', 'every', 'get',
}

# Minimum fraction of words that must appear in known library outputs
COHERENCE_THRESHOLD = 0.60


@dataclass
class ExplorationResult:
    combinations_tried: int = 0
    novel_discovered:   int = 0
    by_strategy:        dict = field(default_factory=dict)
    examples:           list = field(default_factory=list)  # up to 10 samples


class ExplorationEngine:
    """
    Internal exploration: dots already established in the library
    are connected by the hypothesis engine and tested.
    Valid novel combinations are stored — genuinely new knowledge.
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
        self.tester     = tester
        self.reward     = reward
        self.curriculum = curriculum
        self.llm_parent = llm_parent
        self._total_tried = 0
        self._total_novel = 0

    # ── Public API ────────────────────────────────────────────────

    def explore(self, n: int = 60) -> ExplorationResult:
        """
        Run n internal exploration attempts.
        Returns what was discovered.
        """
        result   = ExplorationResult()
        strategy_map = [
            ('chain',      self._strategy_chain),
            ('substitute', self._strategy_substitute),
            ('swap',       self._strategy_swap),
        ]

        for _ in range(n):
            name, fn = random.choice(strategy_map)
            novel_text = fn()
            result.combinations_tried += 1
            self._total_tried += 1

            if novel_text:
                result.novel_discovered += 1
                result.by_strategy[name] = result.by_strategy.get(name, 0) + 1
                if len(result.examples) < 10:
                    result.examples.append((name, novel_text))
                self._total_novel += 1

        return result

    # ── Strategy 1: Chain ─────────────────────────────────────────

    def _strategy_chain(self) -> Optional[str]:
        """
        Join two phrase-level structures into a novel compound.

        "i brush my teeth" + "and i feel clean"
        → "i brush my teeth and i feel clean"
        """
        level   = min(self.curriculum.get_active_level(), 3)
        structs = self.library.get_at_level(level, kind='success')
        if len(structs) < 2:
            return None

        s1, s2 = random.sample(structs, 2)
        g1 = s1.generate(self.library).strip()
        g2 = s2.generate(self.library).strip()

        if not g1 or not g2 or g1 == g2:
            return None
        if len(g1) < 4 or len(g2) < 4:   # no single chars or tiny fragments
            return None
        if len(g1) + len(g2) > 55:
            return None

        connector = random.choice(['and ', 'so ', 'then '])
        novel = f"{g1} {connector}{g2}"

        return self._store_if_coherent(novel, level)

    # ── Strategy 2: Substitute ────────────────────────────────────

    def _strategy_substitute(self) -> Optional[str]:
        """
        Replace a content word in a known phrase with another known word.

        "i brush my teeth" → "i wash my teeth"  (brush→wash)
        "i love amma"      → "i love bhaiya"     (amma→bhaiya)
        """
        phrases = self.library.get_at_level(3, kind='success')
        words   = self.library.get_at_level(2, kind='success')
        if not phrases or not words:
            return None

        phrase      = random.choice(phrases)
        phrase_text = phrase.generate(self.library)
        parts       = phrase_text.split(' ')

        # Find content-word positions
        substitutable = [
            i for i, w in enumerate(parts)
            if w not in FUNCTION_WORDS and len(w) > 2
        ]
        if not substitutable:
            return None

        idx      = random.choice(substitutable)
        original_word = parts[idx]
        original_cat  = persona.WORD_CATEGORY.get(original_word)

        # Collect candidate words from the word-level library
        word_outputs = [
            w.generate(self.library).strip()
            for w in words
        ]

        # Filter to same semantic category — prevents "i drink bhaiya"
        if original_cat:
            same_cat = [
                w for w in word_outputs
                if persona.WORD_CATEGORY.get(w) == original_cat
                and w != original_word
                and len(w) >= 3
            ]
            candidates = same_cat if same_cat else []
        else:
            candidates = [
                w for w in word_outputs
                if w not in FUNCTION_WORDS and len(w) >= 3 and w != original_word
            ]

        if not candidates:
            return None

        new_word = random.choice(candidates)

        parts[idx] = new_word
        novel = ' '.join(parts)

        if novel == phrase_text:
            return None

        return self._store_if_coherent(novel, 3)

    # ── Strategy 3: Swap ──────────────────────────────────────────

    def _strategy_swap(self) -> Optional[str]:
        """
        Swap the consequence between two schemas to form a novel causal link.

        Schema A: "when i sleep early i wake up happy"
        Schema B: "when i eat well i grow big and strong"
        Novel   : "when i sleep early i grow big and strong"

        New causal connections — things Little Deepak hasn't been
        explicitly taught but can infer from what he knows.
        """
        schemas = self.library.get_at_level(4, kind='success')
        if len(schemas) < 2:
            return None

        s1, s2 = random.sample(schemas, 2)
        t1 = s1.generate(self.library)
        t2 = s2.generate(self.library)

        action1 = self._extract_action(t1)
        cons2   = self._extract_consequence(t2)

        if not action1 or not cons2:
            return None

        novel = f"when {action1} {cons2}"

        # Don't duplicate existing schemas
        existing = {s.generate(self.library) for s in schemas}
        if novel in existing or novel == t1 or novel == t2:
            return None

        return self._store_if_coherent(novel, 4)

    # ── Coherence and storage ─────────────────────────────────────

    def _store_if_coherent(self, text: str, level: int) -> Optional[str]:
        """
        Check coherence then test and store if valid.
        Returns the novel text if it was new and stored, else None.
        """
        if not text or not self._is_coherent(text):
            return None

        hypothesis = GenerativeStructure(
            level=level,
            elements=list(text),
            source='exploration',
        )

        outcome, score = self.tester.test(hypothesis, text, self.library)
        self.reward.compute_reward(hypothesis, outcome, score)

        if outcome == 'success':
            is_new = self.library.add_success(hypothesis)
            return text if is_new else None

        return None

    def _is_coherent(self, text: str) -> bool:
        """
        At least COHERENCE_THRESHOLD fraction of words must already
        exist in known library structure outputs.
        This ensures novel combinations are built from solid foundations
        and aren't random noise.
        """
        if not text:
            return False

        words = text.split()
        if not words:
            return False

        # Collect known vocabulary across all levels
        known = set()
        for lvl in range(5):
            for s in self.library.get_at_level(lvl, kind='success'):
                known.update(s.generate(self.library).split())

        covered = sum(1 for w in words if w in known)
        return (covered / len(words)) >= COHERENCE_THRESHOLD

    # ── Schema parsing ────────────────────────────────────────────

    def _extract_action(self, schema: str) -> Optional[str]:
        """
        "when i sleep early i wake up happy" → "i sleep early"
        """
        if not schema.startswith('when i '):
            return None
        rest = schema[len('when i '):]
        idx  = rest.find(' i ')
        if idx == -1:
            return None
        return 'i ' + rest[:idx]

    def _extract_consequence(self, schema: str) -> Optional[str]:
        """
        "when i sleep early i wake up happy" → "i wake up happy"
        """
        if not schema.startswith('when i '):
            return None
        rest = schema[len('when i '):]
        idx  = rest.find(' i ')
        if idx == -1:
            return None
        return 'i ' + rest[idx + 3:]

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'total_tried':    self._total_tried,
            'total_novel':    self._total_novel,
            'discovery_rate': round(
                self._total_novel / max(1, self._total_tried), 3
            ),
        }