"""
generative_unit.py — Hierarchical Generative Unit.

Executes the core (uniform) learning algorithm at one hierarchical level:

  1. Input stimulates the unit
  2. Generate hypotheses about the input's generative structure
  3. Test each hypothesis via ExtremeTester
  4. Store extreme outcomes (success / failure) in the Library
  5. Generate mutants from successful structures
  6. Test mutants — repeat selection pressure
  7. Attempt abstraction: package recurring sub-sequences as new primitives

One instance exists per curriculum level (0–4); all share the identical algorithm.
"""

import random
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING

from hgls.structures import GenerativeStructure
from hgls.library    import Library
from hgls.tester     import ExtremeTester, Outcome
from hgls.reward     import InternalRewardSystem
from hgls.curriculum import CurriculumController

if TYPE_CHECKING:
    from hgls.llm_parent import LLMParentalInterface

N_HYPOTHESES   = 14    # Initial hypotheses per input
N_MUTANTS      = 5     # Mutants spawned from each success
MAX_MUT_DEPTH  = 3     # Recursive mutation depth


class HierarchicalGenerativeUnit:

    def __init__(
        self,
        level: int,
        library: Library,
        tester: ExtremeTester,
        reward: InternalRewardSystem,
        curriculum: CurriculumController,
        llm_parent: Optional['LLMParentalInterface'] = None,
    ):
        self.level      = level
        self.library    = library
        self.tester     = tester
        self.reward     = reward
        self.curriculum = curriculum
        self.llm_parent = llm_parent
        self._cycles    = 0

    # ── Public ────────────────────────────────────────────────────

    def learn(
        self,
        input_text: str,
        n_hypotheses: int = N_HYPOTHESES,
    ) -> List[Tuple[GenerativeStructure, Outcome, float]]:
        """
        One full learning cycle on input_text.
        Returns list of (structure, outcome, score) for non-mediocre results.
        """
        self._cycles += 1
        primitives = self.curriculum.get_primitives()
        outcomes: List[Tuple[GenerativeStructure, Outcome, float]] = []

        # ── Step 1: generate initial hypotheses ───────────────────
        hypotheses = self._generate_hypotheses(input_text, n_hypotheses, primitives)

        # ── Step 2: LLM parent proposals ─────────────────────────
        if (self.llm_parent
                and self.llm_parent.signal_strength > 0.10
                and self.level >= 1):
            raw_proposals = self.llm_parent.propose_structures(
                input_text, self.level, n=3
            )
            # Normalise: expand any multi-char literal elements into individual
            # characters so spaces are preserved during generate().
            # e.g. ["i brush my", "teeth"] → ['i',' ','b','r','u','s','h',' ','m','y','t','e','e','t','h']
            for struct in raw_proposals:
                normalised = []
                for elem in struct.elements:
                    if isinstance(elem, str) and len(elem) > 1 and not self.library.has(elem):
                        normalised.extend(list(elem))   # expand to chars
                    else:
                        normalised.append(elem)
                struct.elements = normalised
            hypotheses.extend(raw_proposals)

        # ── Step 3: test all hypotheses ───────────────────────────
        successes: List[GenerativeStructure] = []
        for hyp in hypotheses:
            if self.library.is_known_failure(hyp):
                continue

            # LLM evaluation only at phrase/schema level where pure
            # reconstruction is insufficient to judge meaning and values.
            # At char/combination/word levels, reconstruction quality alone is correct.
            use_llm = (self.level >= 3 and self.llm_parent is not None)
            outcome, score = self.tester.test(hyp, input_text, self.library, use_llm)
            self.reward.compute_reward(hyp, outcome, score)

            if outcome == 'success':
                self.library.add_success(hyp)
                successes.append(hyp)
                outcomes.append((hyp, outcome, score))
            elif outcome == 'failure':
                self.library.add_failure(hyp)
                outcomes.append((hyp, outcome, score))
            # mediocre → discarded, no storage

        # ── Step 4: mutation exploration from successes ───────────
        for success in successes:
            mut_outcomes = self._explore_mutations(success, input_text, primitives)
            outcomes.extend(mut_outcomes)

        # ── Step 5: abstraction ────────────────────────────────────
        if len(successes) >= 2:
            self._attempt_abstraction(successes)

        return outcomes

    # ── Hypothesis generation ─────────────────────────────────────

    def _generate_hypotheses(
        self,
        target: str,
        n: int,
        primitives: List[str],
    ) -> List[GenerativeStructure]:
        hyps: List[GenerativeStructure] = []

        # 1. Trivial: target as direct sequence of primitives
        if all(c in primitives for c in target):
            hyps.append(GenerativeStructure(
                level=self.level,
                elements=list(target),
                source='generated',
            ))

        # 2. Random sequences of similar length
        for _ in range(n // 4):
            length = max(1, len(target) + random.randint(-1, 1))
            hyps.append(GenerativeStructure(
                level=self.level,
                elements=[random.choice(primitives) for _ in range(length)],
                source='generated',
            ))

        # 3. Mutations of existing library successes at this level
        lib = self.library.get_at_level(self.level, kind='success')
        if lib:
            sample = random.sample(lib, min(n // 3, len(lib)))
            for s in sample:
                hyps.append(s.mutate(primitives, self.library))

        # 4. Composition: decompose using lower-level library structures
        if self.level > 0:
            decomposed = self._decompose(target)
            if decomposed is not None:
                hyps.append(GenerativeStructure(
                    level=self.level,
                    elements=decomposed,
                    source='generated',
                ))

        # 5. Sub-sequence splits of the target
        if len(target) >= 2:
            for _ in range(n // 4):
                sp = random.randint(1, len(target))
                parts = [target[:sp], target[sp:]]
                parts = [p for p in parts if p]
                if parts:
                    hyps.append(GenerativeStructure(
                        level=self.level,
                        elements=parts,
                        source='generated',
                    ))

        return hyps[:n + 6]

    # ── Decomposition ─────────────────────────────────────────────

    def _decompose(self, target: str) -> Optional[List[str]]:
        """
        Greedy left-to-right decomposition of target
        using validated lower-level library structures.
        """
        lower = self.library.get_at_level(self.level - 1, kind='success')
        if not lower:
            return None

        # Build lookup: generated_text → struct_id (longest first)
        lookup: Dict[str, str] = {}
        for s in sorted(lower, key=lambda x: len(x.elements), reverse=True):
            gen = s.generate(self.library)
            if gen and gen not in lookup:
                lookup[gen] = s.id

        prim_set = set(self.curriculum.get_primitives())

        def _rec(rem: str) -> Optional[List[str]]:
            if not rem:
                return []
            for length in range(min(len(rem), 12), 0, -1):
                prefix = rem[:length]
                if prefix in lookup:
                    rest = _rec(rem[length:])
                    if rest is not None:
                        return [lookup[prefix]] + rest
            if rem[0] in prim_set:
                rest = _rec(rem[1:])
                if rest is not None:
                    return [rem[0]] + rest
            return None

        return _rec(target)

    # ── Mutation exploration ───────────────────────────────────────

    def _explore_mutations(
        self,
        parent: GenerativeStructure,
        target: str,
        primitives: List[str],
        depth: int = 0,
    ) -> List[Tuple[GenerativeStructure, Outcome, float]]:
        if depth >= MAX_MUT_DEPTH:
            return []

        results = []
        for _ in range(N_MUTANTS):
            mutant = parent.mutate(primitives, self.library)
            if self.library.is_known_failure(mutant):
                continue

            outcome, score = self.tester.test(mutant, target, self.library)
            self.reward.compute_reward(mutant, outcome, score)

            if outcome == 'success':
                self.library.add_success(mutant)
                results.append((mutant, outcome, score))
                results.extend(
                    self._explore_mutations(mutant, target, primitives, depth + 1)
                )
            elif outcome == 'failure':
                self.library.add_failure(mutant)
                results.append((mutant, outcome, score))

        return results

    # ── Abstraction ───────────────────────────────────────────────

    def _attempt_abstraction(self, successes: List[GenerativeStructure]) -> None:
        """
        Find common sub-sequences between success pairs and package them
        as new reusable structures (new higher-level primitives).

        Space rules: abstracted fragments must not start or end with a space
        token — that would create glue-less concatenation bugs where two
        fragments joined later produce 'myteeth' instead of 'my teeth'.
        Spaces belong inside a fragment, never at its edges.
        """
        for i in range(len(successes)):
            for j in range(i + 1, min(i + 4, len(successes))):
                common = _lcs(successes[i].elements, successes[j].elements)
                if len(common) < 2:
                    continue
                # Strip leading/trailing space tokens from the fragment
                while common and common[0] == ' ':
                    common = common[1:]
                while common and common[-1] == ' ':
                    common = common[:-1]
                if len(common) < 2:
                    continue
                abstracted = GenerativeStructure(
                    level=self.level,
                    elements=common,
                    source='abstracted',
                    fitness=0.9,
                    description=(
                        f"abstracted from {successes[i].id},{successes[j].id}"
                    ),
                )
                self.library.add_success(abstracted)

    # ── Generation ────────────────────────────────────────────────

    def generate(self, input_text: str, context: list = None) -> str:
        """
        Generate a response using Chain of Thought grounded in the library.

        Step 1: Do I know anything about this?
                Search library for topic overlap.
                If best match is weak → "i am not sure" (honest uncertainty)

        Step 2: What do I know?
                Find the most relevant structures.

        Step 3: What does that tell me / what is my answer?
                Compose from what is known, non-redundantly.

        Uncertainty is a signal to the parent to teach, not a failure.
        """
        topics = self._extract_topics(input_text)

        # Enrich topics from working memory context
        if context:
            for ctx in context[-3:]:
                topics |= self._extract_topics(str(ctx))

        if not topics:
            return 'i am not sure'

        # Build known vocabulary from word-level library
        known_vocab: set = set()
        for struct in self.library.get_at_level(2, kind='success'):
            known_vocab.update(struct.generate(self.library).lower().split())
        known_vocab.update({'a', 'i', 'is', 'at', 'do', 'go', 'my',
                            'me', 'we', 'he', 'be', 'no', 'so', 'to',
                            'up', 'as', 'an', 'or', 'in', 'on', 'if'})

        # Score every library structure by relevant content density
        scored = []
        for level in range(7):
            for struct in self.library.get_at_level(level, kind='success'):
                text = struct.generate(self.library)
                if not text or len(text.split()) < 2:
                    continue
                words = text.lower().split()
                if any(
                    (not w[0].isalpha() if w else True) or
                    (len(w) <= 2 and w not in known_vocab)
                    for w in words
                ):
                    continue
                text_words = set(words)
                overlap    = len(topics & text_words)
                if overlap > 0:
                    info = overlap * len(text_words) * max(struct.fitness, 0.1)
                    scored.append((info, overlap, text))

        # ── Step 1: Do I know anything about this? ────────────────
        if not scored:
            return 'i am not sure about that'

        # Best overlap score the library can produce for this question
        best_overlap = max(s[1] for s in scored)

        # If even the best match shares only 1 word with the question,
        # Deepak genuinely doesn't have relevant knowledge — say so
        if best_overlap <= 1:
            return 'i am not sure. i don\'t know about that yet'

        # ── Step 2-3: Compose from what is known ──────────────────
        scored.sort(reverse=True)

        parts      = []
        seen_words: set = set()
        for _, _, text in scored:
            text_words = set(text.split())
            new_words  = text_words - seen_words
            if len(new_words) >= 4:
                parts.append(text)
                seen_words |= text_words
            if len(parts) >= 2:
                break

        return '. '.join(parts) if parts else 'i am not sure'

    @staticmethod
    def _extract_topics(text: str) -> set:
        """Extract meaningful content words — the signal, not the noise."""
        _STOP = {
            'do', 'you', 'your', 'what', 'how', 'are', 'is', 'can', 'does',
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'and', 'or', 'but',
            'that', 'this', 'it', 'he', 'she', 'they', 'we', 'me', 'my',
            'yes', 'no', 'not', 'so', 'tell', 'about', 'did', 'will',
            'have', 'has', 'was', 'were', 'be', 'been', 'am',
            'little', 'deepak', 'hey', 'hi', 'hello', 'dear',
            'when', 'where', 'who', 'why', 'which', 'today', 'now',
            'just', 'even', 'very', 'much', 'many', 'more', 'some',
        }
        words = set(text.lower().strip().rstrip('?!.').split())
        return {w for w in words if w not in _STOP and len(w) > 2}

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'level':   self.level,
            'cycles':  self._cycles,
            'lib_successes': self.library.success_count_at_level(self.level),
        }


# ── Helpers ───────────────────────────────────────────────────────

def _lcs(a: list, b: list) -> list:
    """Longest common contiguous sub-sequence of two lists."""
    best: list = []
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            if k > len(best):
                best = a[i: i + k]
    return best