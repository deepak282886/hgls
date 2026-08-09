"""
generative_unit.py — Hierarchical Generative Unit.

Key fix in this version:
  generate() scoring formula changed from:
    score = overlap * len(text_words) * eff_fit   ← rewards length
  to:
    score = precision * eff_fit                    ← rewards relevance

  precision = overlap / len(text_words)

  This means a short precise correction that is 100% about
  the topic beats a long corpus sentence that shares 3 words
  incidentally. Corrections always win for their topic.

  Also: generate() now searches ALL levels regardless of
  curriculum stage — corrections at level 4/5 are always
  reachable even when curriculum is at stage CHARACTERS.
"""

import random
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING

from hgls.structures import GenerativeStructure
from hgls.library    import Library
from hgls.tester     import ExtremeTester, Outcome
from hgls.reward     import InternalRewardSystem
from hgls.curriculum import CurriculumController

N_HYPOTHESES  = 14
N_MUTANTS     = 5
MAX_MUT_DEPTH = 3


class HierarchicalGenerativeUnit:

    def __init__(
        self,
        level:      int,
        library:    Library,
        tester:     ExtremeTester,
        reward:     InternalRewardSystem,
        curriculum: CurriculumController,
        llm_parent  = None,
    ):
        self.level      = level
        self.library    = library
        self.tester     = tester
        self.reward     = reward
        self.curriculum = curriculum
        self._cycles    = 0

    def learn(
        self,
        input_text:   str,
        n_hypotheses: int = N_HYPOTHESES,
    ) -> List[Tuple[GenerativeStructure, Outcome, float]]:
        self._cycles += 1
        primitives = self.curriculum.get_primitives()
        outcomes:  List[Tuple[GenerativeStructure, Outcome, float]] = []
        hypotheses = self._generate_hypotheses(input_text, n_hypotheses, primitives)
        successes: List[GenerativeStructure] = []

        for hyp in hypotheses:
            if self.library.is_known_failure(hyp):
                continue
            if self.level >= 3:
                outcome, score, token_mask, token_structs = \
                    self.tester.test_autoregressive(hyp, input_text, self.library)
                self.reward.propagate_hierarchical(token_mask, token_structs, self.library)
            else:
                outcome, score = self.tester.test(hyp, input_text, self.library)

            self.reward.compute_reward(hyp, outcome, score)

            if outcome == 'success':
                self.library.add_success(hyp)
                successes.append(hyp)
                outcomes.append((hyp, outcome, score))
            elif outcome == 'failure':
                self.library.add_failure(hyp)
                outcomes.append((hyp, outcome, score))

        for success in successes:
            outcomes.extend(self._explore_mutations(success, input_text, primitives))

        if len(successes) >= 2:
            self._attempt_abstraction(successes)

        return outcomes

    def learn_correction(
        self,
        correction_text: str,
        topic_words:     List[str] = None,
        n_hypotheses:    int = N_HYPOTHESES,
    ) -> List[Tuple[GenerativeStructure, Outcome, float]]:
        self._cycles += 1
        primitives = self.curriculum.get_primitives()
        outcomes   = []
        hypotheses = self._generate_hypotheses(correction_text, n_hypotheses, primitives)

        for hyp in hypotheses:
            if self.library.is_known_failure(hyp):
                continue
            if self.level >= 3:
                outcome, score, token_mask, token_structs = \
                    self.tester.test_autoregressive(hyp, correction_text, self.library)
                self.reward.propagate_hierarchical(token_mask, token_structs, self.library)
            else:
                outcome, score = self.tester.test(hyp, correction_text, self.library)

            self.reward.compute_reward(hyp, outcome, score)

            if outcome == 'success':
                self.reward.reinforce_correction(hyp, topic_words)
                self.library.add_correction(hyp, topic_words)
                outcomes.append((hyp, outcome, score))
            elif outcome == 'failure':
                self.library.add_failure(hyp)
                outcomes.append((hyp, outcome, score))

        return outcomes

    def generate(self, input_text: str, context: list = None) -> str:
        """
        Generate a response using precision-weighted retrieval.

        Scoring: precision × effective_fitness
          precision = topic_overlap / sentence_length

        A short correction that is entirely about the topic
        scores higher than a long corpus sentence with incidental overlap.
        Corrections with matching topic_tags get effective_fitness boost
        that makes them dominate regardless of sentence length.

        Searches ALL levels — corrections at L4/L5 are always reachable.
        """
        topics = self._extract_topics(input_text)
        if context:
            for ctx in context[-3:]:
                topics |= self._extract_topics(str(ctx))
        if not topics:
            return 'i am not sure'

        # Build known vocabulary for artifact filtering
        known_vocab: set = set()
        for struct in self.library.get_at_level(2, kind='success'):
            known_vocab.update(struct.generate(self.library).lower().split())
        known_vocab.update({
            'a','i','is','at','do','go','my','me','we','he','be',
            'no','so','to','up','as','an','or','in','on','if',
            'the','it','of','and','or','but','not','has','had',
            'was','are','for','with','this','that','they','them',
        })

        scored = []

        # Search ALL levels — do not restrict by curriculum stage
        for level in range(7):
            for struct in self.library.get_at_level(level, kind='success'):
                text = struct.generate(self.library)
                if not text:
                    continue

                words      = text.lower().split()
                word_count = len(words)

                if word_count < 2:
                    continue

                # Artifact filter — skip structures with non-word characters
                # (hypothesis fragments, raw character sequences)
                if any(
                    (not w[0].isalpha() if w else True)
                    for w in words[:3]
                ):
                    continue

                text_words = set(words)
                overlap    = len(topics & text_words)

                if overlap == 0:
                    continue

                # ── Key fix: precision-weighted scoring ──
                # precision: what fraction of the sentence is about the topic
                precision = overlap / max(word_count, 1)

                # effective_fitness: corrections get large boost for matching topics
                eff_fit = struct.effective_fitness(topics)

                # Final score: precision × fitness
                # Short precise corrections beat long incidental matches
                score = precision * eff_fit

                # Corrections with matching tags dominate everything
                if struct.correction_count > 0 and struct.topic_tags:
                    tag_overlap = len(topics & set(struct.topic_tags))
                    if tag_overlap > 0:
                        # Strong multiplier: 3x per correction
                        # ensures corrections always beat corpus sentences
                        score *= (3.0 * struct.correction_count)
                elif struct.correction_count > 0:
                    # Correction but no tag match — modest boost
                    score *= 1.2

                scored.append((score, overlap, text, word_count))

        if not scored:
            return 'i am not sure about that'

        # Need at least 2 topic words matched somewhere

        scored.sort(reverse=True)

        # Compose response from top non-redundant structures
        parts:      List[str] = []
        seen_words: set       = set()

        for score, overlap, text, word_count in scored:
            text_words = set(text.split())
            new_words  = text_words - seen_words

            # Each part must contribute at least 3 new words
            if len(new_words) >= 3:
                parts.append(text)
                seen_words |= text_words

            # Cap at 2 parts — concise and precise
            if len(parts) >= 2:
                break

        return '. '.join(parts) if parts else 'i am not sure'

    # ── Hypothesis generation ──────────────────────────────────────

    def _generate_hypotheses(
        self,
        target:     str,
        n:          int,
        primitives: List[str],
    ) -> List[GenerativeStructure]:
        hyps: List[GenerativeStructure] = []

        if all(c in primitives for c in target):
            hyps.append(GenerativeStructure(
                level=self.level, elements=list(target), source='generated',
            ))

        for _ in range(n // 4):
            length = max(1, len(target) + random.randint(-1, 1))
            hyps.append(GenerativeStructure(
                level=self.level,
                elements=[random.choice(primitives) for _ in range(length)],
                source='generated',
            ))

        lib = self.library.get_at_level(self.level, kind='success')
        if lib:
            weights = [max(0.01, s.effective_fitness()) for s in lib]
            total_w = sum(weights)
            probs   = [w / total_w for w in weights]
            k       = min(n // 3, len(lib))
            try:
                sample = random.choices(lib, weights=probs, k=k)
            except Exception:
                sample = random.sample(lib, k)
            for s in sample:
                hyps.append(s.mutate(primitives, self.library))

        if self.level > 0:
            decomposed = self._decompose(target)
            if decomposed is not None:
                hyps.append(GenerativeStructure(
                    level=self.level, elements=decomposed, source='generated',
                ))

        if len(target) >= 2:
            for _ in range(n // 4):
                sp    = random.randint(1, len(target))
                parts = [p for p in [target[:sp], target[sp:]] if p]
                if parts:
                    hyps.append(GenerativeStructure(
                        level=self.level, elements=parts, source='generated',
                    ))

        return hyps[:n + 6]

    def _decompose(self, target: str) -> Optional[List[str]]:
        lower = self.library.get_at_level(self.level - 1, kind='success')
        if not lower:
            return None

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

    def _explore_mutations(
        self,
        parent:     GenerativeStructure,
        target:     str,
        primitives: List[str],
        depth:      int = 0,
    ) -> List[Tuple[GenerativeStructure, Outcome, float]]:
        if depth >= MAX_MUT_DEPTH:
            return []
        results = []
        for _ in range(N_MUTANTS):
            mutant = parent.mutate(primitives, self.library)
            if self.library.is_known_failure(mutant):
                continue
            if self.level >= 3:
                outcome, score, token_mask, token_structs = \
                    self.tester.test_autoregressive(mutant, target, self.library)
                self.reward.propagate_hierarchical(token_mask, token_structs, self.library)
            else:
                outcome, score = self.tester.test(mutant, target, self.library)
            self.reward.compute_reward(mutant, outcome, score)
            if outcome == 'success':
                self.library.add_success(mutant)
                results.append((mutant, outcome, score))
                results.extend(self._explore_mutations(mutant, target, primitives, depth + 1))
            elif outcome == 'failure':
                self.library.add_failure(mutant)
                results.append((mutant, outcome, score))
        return results

    def _attempt_abstraction(self, successes: List[GenerativeStructure]) -> None:
        for i in range(len(successes)):
            for j in range(i + 1, min(i + 4, len(successes))):
                common = _lcs(successes[i].elements, successes[j].elements)
                if len(common) < 2:
                    continue
                while common and common[0]  == ' ': common = common[1:]
                while common and common[-1] == ' ': common = common[:-1]
                if len(common) < 2:
                    continue
                abstracted = GenerativeStructure(
                    level=self.level, elements=common,
                    source='abstracted', fitness=0.9,
                )
                self.library.add_success(abstracted)

    @staticmethod
    def _extract_topics(text: str) -> set:
        _STOP = {
            'do','you','your','what','how','are','is','can','does',
            'the','a','an','in','on','at','to','and','or','but',
            'that','this','it','he','she','they','we','me','my',
            'yes','no','not','so','tell','about','did','will',
            'have','has','was','were','be','been','am',
            'when','where','who','why','which','just','very',
            'give','name','describe','explain','example',
        }
        words = set(text.lower().strip().rstrip('?!.').split())
        return {w for w in words if w not in _STOP and len(w) > 2}

    def stats(self) -> dict:
        return {
            'level':         self.level,
            'cycles':        self._cycles,
            'lib_successes': self.library.success_count_at_level(self.level),
        }


def _lcs(a: list, b: list) -> list:
    best: list = []
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i+k < len(a) and j+k < len(b) and a[i+k] == b[j+k]:
                k += 1
            if k > len(best):
                best = a[i:i+k]
    return best