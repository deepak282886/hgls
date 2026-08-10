"""
system.py — HGLSystem v0.7

All internal systems now wired into both learning paths:
  - correction path: co_occurrence, reward, emotional_evaluator, navigator
  - respond path: emotional_evaluator, navigator
  - New: reinforce_correct_response() and penalise_wrong_response()
  - Tinkering: no minimum edge count
"""

import os
import json
from typing import List, Optional, Dict, Any

from hgls.structures          import GenerativeStructure
from hgls.library             import Library
from hgls.tester              import ExtremeTester
from hgls.reward              import InternalRewardSystem
from hgls.memory              import WorkingMemory
from hgls.attention           import AttentionMechanism
from hgls.self_model          import SelfModel
from hgls.curriculum          import CurriculumController, Stage
from hgls.sensory_motor       import SensoryMotorInterface
from hgls.generative_unit     import HierarchicalGenerativeUnit
from hgls.explorer            import ExplorationEngine
from hgls.graph               import MemoryGraph
from hgls.co_occurrence       import CoOccurrenceDetector
from hgls.emotional_evaluator import EmotionalEvaluator
from hgls.navigator           import GraphNavigator
from hgls.tinkering           import TinkeringEngine
from hgls.llm_validator       import LLMValidator


class HGLSystem:

    def __init__(self, use_llm: bool = False):
        self.library       = Library()
        self.curriculum    = CurriculumController()
        self.reward        = InternalRewardSystem()
        self.memory        = WorkingMemory()
        self.attention     = AttentionMechanism()
        self.self_model    = SelfModel()
        self.sensory_motor = SensoryMotorInterface()

        self.graph               = MemoryGraph()
        self.emotional_evaluator = EmotionalEvaluator(
            graph=self.graph, library=self.library
        )
        self.navigator    = GraphNavigator(
            graph=self.graph, library=self.library,
            evaluator=self.emotional_evaluator,
        )
        self.tinkering    = TinkeringEngine(
            graph=self.graph, library=self.library,
            evaluator=self.emotional_evaluator,
        )
        self.co_occurrence = CoOccurrenceDetector(
            library=self.library, graph=self.graph,
        )
        self.llm_validator = LLMValidator()
        self.tester        = ExtremeTester()

        self.units: Dict[int, HierarchicalGenerativeUnit] = {
            lvl: HierarchicalGenerativeUnit(
                level=lvl, library=self.library,
                tester=self.tester, reward=self.reward,
                curriculum=self.curriculum,
            )
            for lvl in range(7)
        }

        self.explorer = ExplorationEngine(
            library=self.library, tester=self.tester,
            reward=self.reward, curriculum=self.curriculum,
        )
        self.explorer.set_units(self.units)

        self._cycle_count = 0
        self._bootstrap()

    def _bootstrap(self) -> None:
        primitives = self.curriculum.bootstrap_char_structures()
        for s in primitives:
            self.self_model.mark_external(s)
            self.library.add_success(s)
        print(f"[Bootstrap] Seeded library with {len(primitives)} character primitives.")

    # ── Learning ──────────────────────────────────────────────────

    def run_cycle(self, raw_input: str, target_level: int = None) -> Dict[str, Any]:
        self._cycle_count += 1
        text = self.sensory_motor.receive_input(raw_input)
        if not text:
            return {'cycle': self._cycle_count, 'error': 'empty input'}

        self.memory.push(text)
        level    = target_level if target_level is not None else self.curriculum.get_active_level()
        salience = self.attention.compute_salience(text, level)
        n_hyp    = self.attention.allocate_hypotheses(14, salience)

        # Navigator reads graph topology
        meta = self.navigator.analyse(text, level=level)

        unit          = self.units[level]
        cycle_results = unit.learn(text, n_hypotheses=n_hyp)

        for struct, outcome, score in cycle_results:
            if outcome == 'success':
                if struct.level >= 3:
                    self.co_occurrence.observe_phrase(struct.id)
                self.self_model.mark_self_generated(struct)
                self.reward.compute_reward(struct, outcome, score)

        if cycle_results:
            best_score   = max(sc for _, _, sc in cycle_results)
            top_strategy = max(meta.strategy_weights, key=meta.strategy_weights.get)
            self.navigator.update_from_outcome(top_strategy, best_score)
            best_struct, _, _ = max(cycle_results, key=lambda x: x[2])
            generated = best_struct.generate(self.library)
            self.emotional_evaluator.evaluate(generated, text)

        if target_level is None:
            self.curriculum.tick()
            if self.curriculum.should_advance(self.library):
                new_stage = self.curriculum.advance()
                print(f"\n[Curriculum] Advanced to: {new_stage.name}\n")

        successes = [(s, sc) for s, o, sc in cycle_results if o == 'success']
        failures  = [(s, sc) for s, o, sc in cycle_results if o == 'failure']

        return {
            'cycle':       self._cycle_count,
            'input':       text,
            'level':       level,
            'n_successes': len(successes),
            'n_failures':  len(failures),
            'best_score':  max((sc for _, sc in successes), default=0.0),
            'lib_size':    len(self.library),
            'graph_edges': len(self.graph),
        }

    def ingest_text(
        self,
        text:          str,
        level:         int  = None,
        is_correction: bool = False,
        topic_words:   List[str] = None,
    ) -> Dict[str, Any]:
        text = self.sensory_motor.receive_input(text)
        if not text:
            return {'error': 'empty input'}

        if level is None:
            level = self.curriculum.get_active_level()

        if is_correction:
            if topic_words is None:
                topic_words = list(HierarchicalGenerativeUnit._extract_topics(text))
            results   = self.units[level].learn_correction(text, topic_words=topic_words)
            successes = 0

            for struct, outcome, score in results:
                if outcome == 'success':
                    successes += 1
                    # Wire co-occurrence for phrase-level corrections
                    if struct.level >= 3:
                        self.co_occurrence.observe_phrase(struct.id)
                    # Wire reward system
                    self.reward.compute_reward(struct, outcome, score)
                    self.self_model.mark_self_generated(struct)

            # Navigator learns from correction context
            if results:
                meta = self.navigator.analyse(text, level=level)
                best_score = max((sc for _, o, sc in results if o == 'success'), default=0.0)
                if best_score > 0:
                    top_strategy = max(meta.strategy_weights, key=meta.strategy_weights.get)
                    self.navigator.update_from_outcome(top_strategy, best_score)

            return {
                'text':      text,
                'level':     level,
                'successes': successes,
                'lib_size':  len(self.library),
            }
        else:
            return self.run_cycle(text, target_level=level)

    # ── Generation ────────────────────────────────────────────────

    def respond(self, user_input: str) -> str:
        """
        Generate response. Wires emotional evaluator and navigator
        so they accumulate signal from every query.
        """
        text = self.sensory_motor.receive_input(user_input)
        self.memory.clear()
        self.memory.push(text)
        context = self.memory.get_context()

        # Navigator reads graph topology for this query
        level = self.curriculum.get_active_level()
        meta  = self.navigator.analyse(text, level=level)

        # Generate from all levels
        response = ''
        for lvl in range(6, -1, -1):
            candidate = self.units[lvl].generate(text, context=context)
            if candidate and 'not sure' not in candidate.lower():
                response = candidate
                break
        if not response:
            response = self.units[0].generate(text, context=context)

        # Emotional evaluator scores the response — accumulates signal
        if response:
            self.emotional_evaluator.evaluate(response, text)

        return response

    # ── Reinforcement after evaluation ────────────────────────────

    def reinforce_correct_response(
        self,
        response:    str,
        topic_words: List[str],
    ) -> None:
        """
        System answered correctly. Reward every structure that
        contributed to generating the right response.
        This closes the positive reinforcement loop.
        """
        response_words = set(response.lower().split())
        topic_set      = set(topic_words)

        for level in range(7):
            for struct in self.library.get_at_level(level, 'success'):
                text = struct.generate(self.library)
                if not text:
                    continue
                text_words = set(text.lower().split())
                if len(text_words) < 2:
                    continue

                # Check if this structure's text appears substantially in response
                overlap = len(text_words & response_words)
                if overlap < max(1, len(text_words) // 2):
                    continue

                # Reward this structure
                struct.reward_count += 1
                struct.fitness = min(1.0, struct.fitness + 0.03)

                # Extra boost for correction-tagged structures on matching topic
                if struct.correction_count > 0 and struct.topic_tags:
                    tag_overlap = len(topic_set & set(struct.topic_tags))
                    if tag_overlap > 0:
                        struct.fitness = min(1.0, struct.fitness + 0.05)
                        struct.correction_count += 1  # strengthen further

    def penalise_wrong_response(
        self,
        response:    str,
        topic_words: List[str],
    ) -> None:
        """
        System answered wrongly. Penalise non-correction structures
        that generated the wrong response. Corrections are never penalised.
        """
        if not response or 'not sure' in response.lower():
            return  # honest uncertainty — don't penalise

        response_words = set(response.lower().split())

        for level in range(3, 7):
            for struct in self.library.get_at_level(level, 'success'):
                # Never penalise correction-tagged structures
                if struct.correction_count > 0:
                    continue

                text = struct.generate(self.library)
                if not text:
                    continue
                text_words = set(text.lower().split())
                if len(text_words) < 2:
                    continue

                # Check if this structure appeared in wrong response
                overlap = len(text_words & response_words)
                if overlap >= max(1, len(text_words) // 2):
                    struct.penalty_count += 1
                    struct.fitness = max(0.0, struct.fitness - 0.02)

    # ── Tinkering (no minimum edge count) ─────────────────────────

    def run_tinkering(self) -> dict:
        """Run tinkering engine. No minimum edge requirement."""
        if len(self.graph) < 2:
            return {'proposals': 0, 'accepted': 0}
        proposals = self.tinkering.tinker()
        if not proposals:
            return {'proposals': 0, 'accepted': 0}
        accepted, _ = self.llm_validator.validate_batch(
            proposals, graph=self.graph
        )
        return {'proposals': len(proposals), 'accepted': len(accepted)}

    def scan_cooccurrence(self, sentences_processed: int = None) -> dict:
        return self.co_occurrence.scan_library(
            sentences_processed=sentences_processed, verbose=False
        )

    def explore(self, n: int = 60):
        return self.explorer.explore(n=n)

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str = 'deepak_memory.json') -> None:
        data = {
            'version':          '0.7',
            'curriculum_stage': int(self.curriculum.current_stage),
            'library':          self.library.to_dict(),
        }
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        backup = path + '.bak'
        if os.path.exists(path):
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(path, backup)
        os.rename(tmp, path)

        graph_path = path.replace('.json', '_graph.json')
        self.graph.save(graph_path)
        print(f"[Memory] {len(self.library)} structures + {len(self.graph)} edges → {path}")

    def load(self, path: str = 'deepak_memory.json') -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Memory] Load failed: {e}")
            backup = path + '.bak'
            if os.path.exists(backup):
                try:
                    with open(backup, encoding='utf-8') as f:
                        data = json.load(f)
                    print("[Memory] Loaded from backup.")
                except Exception:
                    return False
            else:
                return False

        self.library = Library.from_dict(data['library'])
        self.curriculum.current_stage = Stage(data.get('curriculum_stage', 0))

        for unit in self.units.values():
            unit.library = self.library
        self.explorer.library            = self.library
        self.co_occurrence.library       = self.library
        self.emotional_evaluator.library = self.library
        self.navigator.library           = self.library
        self.tinkering.library           = self.library

        graph_path = path.replace('.json', '_graph.json')
        self.graph.load(graph_path)

        print(
            f"[Memory] Loaded {len(self.library)} structures + "
            f"{len(self.graph)} edges — stage: {self.curriculum.current_stage.name}"
        )
        return True

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            'cycles':              self._cycle_count,
            'library':             self.library.stats(),
            'graph':               self.graph.stats(),
            'curriculum':          self.curriculum.stats(),
            'reward':              self.reward.stats(),
            'emotional_evaluator': self.emotional_evaluator.stats(),
            'navigator':           self.navigator.stats(),
            'tinkering':           self.tinkering.stats(),
            'tester':              self.tester.stats(),
        }

    def print_stats(self) -> None:
        print('\n' + '=' * 60)
        print('HGLS System Stats v0.7')
        print('=' * 60)
        print(json.dumps(self.stats(), indent=2, default=str))