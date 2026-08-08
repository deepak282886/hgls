"""
system.py — HGLSystem.

Integrates all modules and runs the developmental learning loop.

Changes in this version:
  - _parent_correct() uses learn_correction() so teacher content is tagged
    and carries higher effective fitness for matching topics
  - ingest_text() — pre-training entry point for dataset ingestion
  - Removed all composer references
"""

import os
import json
from typing import List, Optional, Dict, Any

from hgls.structures      import GenerativeStructure
from hgls.library         import Library
from hgls.tester          import ExtremeTester
from hgls.reward          import InternalRewardSystem
from hgls.memory          import WorkingMemory
from hgls.attention       import AttentionMechanism
from hgls.self_model      import SelfModel
from hgls.curriculum      import CurriculumController, Stage
from hgls.sensory_motor   import SensoryMotorInterface
from hgls.generative_unit import HierarchicalGenerativeUnit
from hgls.explorer        import ExplorationEngine


class HGLSystem:
    """
    Hierarchical Generative Learning System

    Module inventory:
      SensoryMotorInterface       — keyboard-level I/O
      HierarchicalGenerativeUnits (×7) — uniform algorithm per level 0-6
      Library                     — long-term memory (extreme outcomes only)
      WorkingMemory               — short-term context buffer
      ExtremeTester               — gradient scoring + autoregressive token test
      InternalRewardSystem        — novelty + competence + curiosity + propagation
      AttentionMechanism          — salience-weighted resource allocation
      SelfModel                   — agency / self vs. external distinction
      CurriculumController        — developmental stage management
      ExplorationEngine           — internal composition between structures
      LLMParentalInterface        — external evaluative signal (optional, fades)
    """

    def __init__(self, use_llm: bool = True):
        self.library       = Library()
        self.curriculum    = CurriculumController()
        self.reward        = InternalRewardSystem()
        self.memory        = WorkingMemory()
        self.attention     = AttentionMechanism()
        self.self_model    = SelfModel()
        self.sensory_motor = SensoryMotorInterface()

        self.llm_parent = None
        if use_llm:
            try:
                from hgls.llm_parent import LLMParentalInterface
                self.llm_parent = LLMParentalInterface()
            except Exception as e:
                print(f"[Warning] LLM parent unavailable: {e}")

        self.tester = ExtremeTester(llm_parent=self.llm_parent)

        self.units: Dict[int, HierarchicalGenerativeUnit] = {
            lvl: HierarchicalGenerativeUnit(
                level=lvl,
                library=self.library,
                tester=self.tester,
                reward=self.reward,
                curriculum=self.curriculum,
                llm_parent=self.llm_parent,
            )
            for lvl in range(7)
        }

        self.explorer = ExplorationEngine(
            library=self.library,
            tester=self.tester,
            reward=self.reward,
            curriculum=self.curriculum,
            llm_parent=self.llm_parent,
        )
        self.explorer.set_units(self.units)

        self._cycle_count = 0
        self._bootstrap()

    # ── Bootstrap ─────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        primitives = self.curriculum.bootstrap_char_structures()
        for s in primitives:
            self.self_model.mark_external(s)
            self.library.add_success(s)
        print(f"[Bootstrap] Seeded library with {len(primitives)} character primitives.")

    # ── Main learning cycle ────────────────────────────────────────

    def run_cycle(self, raw_input: str, target_level: int = None) -> Dict[str, Any]:
        """
        Process one input through the full learning cycle.
        target_level overrides the curriculum stage (used during domain expansion).
        """
        self._cycle_count += 1

        text = self.sensory_motor.receive_input(raw_input)
        if not text:
            return {'cycle': self._cycle_count, 'error': 'empty input'}

        self.memory.push(text)

        level    = target_level if target_level is not None else self.curriculum.get_active_level()
        salience = self.attention.compute_salience(text, level)
        n_hyp    = self.attention.allocate_hypotheses(14, salience)

        unit          = self.units[level]
        cycle_results = unit.learn(text, n_hypotheses=n_hyp)

        for struct, outcome, score in cycle_results:
            if outcome == 'success':
                self.self_model.mark_self_generated(struct)

        advanced = False
        if target_level is None:
            self.curriculum.tick()
            if self.curriculum.should_advance(self.library):
                new_stage = self.curriculum.advance()
                advanced  = True
                print(f"\n[Curriculum] *** Advanced to: {new_stage.name} ***\n")

        if self.llm_parent and self.reward.maturity > 0.5:
            self.llm_parent._decay()

        successes = [(s, sc) for s, o, sc in cycle_results if o == 'success']
        failures  = [(s, sc) for s, o, sc in cycle_results if o == 'failure']

        return {
            'cycle':       self._cycle_count,
            'input':       text,
            'level':       level,
            'n_successes': len(successes),
            'n_failures':  len(failures),
            'best_score':  max((sc for _, sc in successes), default=0.0),
            'advanced':    advanced,
            'lib_size':    len(self.library),
        }

    def run_episode(
        self,
        inputs: List[str],
        verbose: bool = True,
        target_level: int = None,
    ) -> List[Dict]:
        results = []
        for inp in inputs:
            r = self.run_cycle(inp, target_level=target_level)
            results.append(r)
            if verbose:
                self._print_cycle(r)
        return results

    # ── Dataset ingestion (pre-training) ──────────────────────────

    def ingest_text(
        self,
        text: str,
        level: int = None,
        is_correction: bool = False,
        topic_words: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single text chunk for pre-training or fine-tuning.

        Pre-training  (is_correction=False): feeds through run_cycle(),
          normal learning at the specified level.

        Fine-tuning   (is_correction=True): uses learn_correction(),
          tags structures with topic_words so teacher content dominates
          for matching topic queries.

        Call this in a loop from the dataset ingestor.
        """
        text = self.sensory_motor.receive_input(text)
        if not text:
            return {'error': 'empty input'}

        if level is None:
            level = self.curriculum.get_active_level()

        if is_correction:
            if topic_words is None:
                topic_words = list(HierarchicalGenerativeUnit._extract_topics(text))
            results = self.units[level].learn_correction(text, topic_words=topic_words)
            successes = sum(1 for _, o, _ in results if o == 'success')
            return {
                'text':       text,
                'level':      level,
                'mode':       'correction',
                'successes':  successes,
                'lib_size':   len(self.library),
            }
        else:
            return self.run_cycle(text, target_level=level)

    # ── Generation / chat ─────────────────────────────────────────

    def respond(self, user_input: str) -> str:
        """
        Generate a response to user input.
        Tries active level first, falls back through levels.
        LLM parent corrects if wrong (uses learn_correction internally).
        """
        text = self.sensory_motor.receive_input(user_input)
        self.memory.clear()
        self.memory.push(text)
        context = self.memory.get_context()

        level    = self.curriculum.get_active_level()
        response = ''
        for lvl in range(level, -1, -1):
            response = self.units[lvl].generate(text, context=context)
            if response:
                break

        if response and self.llm_parent and self.llm_parent._active():
            response = self._parent_correct(text, response)

        return response

    def _parent_correct(self, user_input: str, response: str) -> str:
        context = (
            f"Someone said to Little Deepak: '{user_input}'. "
            f"Little Deepak replied: '{response}'. "
            f"Is this a natural, correct, age-appropriate reply?"
        )
        judgment, confidence = self.llm_parent.judge(response, context=context)

        if judgment in ('bad',) and confidence > 0.5:
            correction = self._request_correction(user_input, response)
            if correction and correction != response:
                # Mark original as failure
                bad_struct = GenerativeStructure(
                    level=self.curriculum.get_active_level(),
                    elements=list(response),
                    source='generated',
                    fitness=0.0,
                )
                self.library.add_failure(bad_struct)

                # Learn correction with topic tagging
                topic_words = list(
                    HierarchicalGenerativeUnit._extract_topics(user_input)
                )
                level = self.curriculum.get_active_level()
                self.units[level].learn_correction(
                    correction, topic_words=topic_words
                )
                return correction

        return response

    def _request_correction(self, user_input: str, bad_response: str) -> str:
        if not self.llm_parent or not self.llm_parent._active():
            return bad_response
        prompt = (
            f"Someone said to Little Deepak: '{user_input}'.\n"
            f"Little Deepak said: '{bad_response}' — this is wrong or unnatural.\n\n"
            f"Write the correct, natural reply Little Deepak should give. "
            f"Keep it simple, 1-2 sentences, in Little Deepak's voice. "
            f"Start with 'i' or 'yes' or 'namaste'. "
            f"Reply with only the corrected sentence, nothing else."
        )
        try:
            corrected = self.llm_parent._call(prompt, max_tokens=60).strip()
            corrected = corrected.lower().strip().strip('"\'')
            if len(corrected) > 3:
                return corrected
        except Exception:
            pass
        return bad_response

    # ── Exploration ───────────────────────────────────────────────

    def explore(self, n: int = 60):
        return self.explorer.explore(n=n)

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str = 'deepak_memory.json') -> None:
        data = {
            'version':          '0.5',
            'persona':          'Little Deepak',
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
        print(f"[Memory] Saved {len(self.library)} structures → {path}")

    def load(self, path: str = 'deepak_memory.json') -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[Memory] File corrupted ({e}) — checking backup...")
            backup = path + '.bak'
            if os.path.exists(backup):
                try:
                    with open(backup, encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"[Memory] Loaded from backup {backup}")
                except Exception:
                    print("[Memory] Backup also corrupted — starting fresh.")
                    return False
            else:
                print("[Memory] No backup found — starting fresh.")
                return False

        self.library = Library.from_dict(data['library'])
        stage = data.get('curriculum_stage', 0)
        self.curriculum.current_stage = Stage(stage)

        self.tester.llm_parent = self.llm_parent
        self.explorer.library  = self.library
        for unit in self.units.values():
            unit.library = self.library

        print(
            f"[Memory] Loaded {len(self.library)} structures "
            f"from {path} — stage: {self.curriculum.current_stage.name}"
        )
        return True

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> Dict:
        s = {
            'cycles':        self._cycle_count,
            'library':       self.library.stats(),
            'curriculum':    self.curriculum.stats(),
            'reward':        self.reward.stats(),
            'sensory_motor': self.sensory_motor.stats(),
            'self_model':    self.self_model.stats(),
            'tester':        self.tester.stats(),
            'attention':     self.attention.stats(),
            'explorer':      self.explorer.stats(),
            'units':         {lvl: u.stats() for lvl, u in self.units.items()},
        }
        if self.llm_parent:
            s['llm_parent'] = self.llm_parent.stats()
        return s

    def print_stats(self) -> None:
        print('\n' + '=' * 60)
        print('HGLS System Stats')
        print('=' * 60)
        print(json.dumps(self.stats(), indent=2, default=str))

    def _print_cycle(self, r: Dict) -> None:
        if r.get('error'):
            print(f"  Cycle {r['cycle']:4d}: ERROR — {r['error']}")
            return
        adv = ' *** ADVANCED ***' if r.get('advanced') else ''
        print(
            f"  Cycle {r['cycle']:4d} | "
            f"'{r['input'][:18]:<18}' | "
            f"lvl={r['level']} | "
            f"✓={r['n_successes']:2d} ✗={r['n_failures']:2d} | "
            f"best={r['best_score']:.3f} | "
            f"lib={r['lib_size']:4d}"
            + adv
        )
