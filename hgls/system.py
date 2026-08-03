"""
system.py — HGLSystem.

Integrates all modules and runs the developmental learning loop.
Starts from keyboard character primitives and grows upward through
curriculum-controlled stages toward 5-year-old-level text competence.
"""

import os
import json
from difflib import SequenceMatcher
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
from hgls.composer        import ResponseComposer


class HGLSystem:
    """
    Hierarchical Generative Learning System v0.4

    Module inventory:
      SensoryMotorInterface       — keyboard-level I/O
      HierarchicalGenerativeUnits (×5) — uniform learning algorithm per level
      Library                     — long-term memory (extreme outcomes only)
      WorkingMemory               — short-term context buffer
      ExtremeTester               — success / failure / mediocre classification
      InternalRewardSystem        — novelty + competence + curiosity signals
      AttentionMechanism          — salience-weighted resource allocation
      SelfModel                   — agency / self vs. external distinction
      CurriculumController        — developmental stage management
      ExplorationEngine           — internal dot-connecting between structures
      ResponseComposer            — compositional dialogue response generator
      LLMParentalInterface        — external evaluative signal (optional, fades)
    """

    def __init__(self, use_llm: bool = True):
        # Core modules
        self.library       = Library()
        self.curriculum    = CurriculumController()
        self.reward        = InternalRewardSystem()
        self.memory        = WorkingMemory()
        self.attention     = AttentionMechanism()
        self.self_model    = SelfModel()
        self.sensory_motor = SensoryMotorInterface()

        # Optional LLM parent
        self.llm_parent = None
        if use_llm:
            try:
                from hgls.llm_parent import LLMParentalInterface
                self.llm_parent = LLMParentalInterface()
            except Exception as e:
                print(f"[Warning] LLM parent unavailable: {e}")

        self.tester = ExtremeTester(llm_parent=self.llm_parent)

        # One generative unit per level (0–4)
        self.units: Dict[int, HierarchicalGenerativeUnit] = {
            lvl: HierarchicalGenerativeUnit(
                level=lvl,
                library=self.library,
                tester=self.tester,
                reward=self.reward,
                curriculum=self.curriculum,
                llm_parent=self.llm_parent,
            )
            for lvl in range(5)
        }

        # Exploration engine — internal dot-connecting
        self.explorer = ExplorationEngine(
            library=self.library,
            tester=self.tester,
            reward=self.reward,
            curriculum=self.curriculum,
            llm_parent=self.llm_parent,
        )

        # Response composer — compositional dialogue generation
        self.composer = ResponseComposer(
            library=self.library,
            curriculum=self.curriculum,
        )

        self._cycle_count = 0
        self._bootstrap()

    # ── Bootstrap ─────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """Seed the library with atomic character-level primitives."""
        primitives = self.curriculum.bootstrap_char_structures()
        for s in primitives:
            self.self_model.mark_external(s)
            self.library.add_success(s)
        print(
            f"[Bootstrap] Seeded library with {len(primitives)} "
            f"character primitives."
        )

    # ── Main loop ─────────────────────────────────────────────────

    def run_cycle(self, raw_input: str) -> Dict[str, Any]:
        """
        Process one input through the full learning cycle.
        Returns a summary dict.
        """
        self._cycle_count += 1

        # 1. Receive and normalise input
        text = self.sensory_motor.receive_input(raw_input)
        if not text:
            return {'cycle': self._cycle_count, 'error': 'empty input'}

        # 2. Store in working memory
        self.memory.push(text)

        # 3. Compute salience → hypothesis budget
        level    = self.curriculum.get_active_level()
        salience = self.attention.compute_salience(text, level)
        n_hyp    = self.attention.allocate_hypotheses(14, salience)

        # 4. Learn
        unit          = self.units[level]
        cycle_results = unit.learn(text, n_hypotheses=n_hyp)

        # 5. Mark self-generated structures
        for struct, outcome, score in cycle_results:
            if outcome == 'success':
                self.self_model.mark_self_generated(struct)

        # 6. Curriculum tick + advancement check
        self.curriculum.tick()
        advanced = False
        if self.curriculum.should_advance(self.library):
            new_stage = self.curriculum.advance()
            advanced  = True
            print(f"\n[Curriculum] *** Advanced to: {new_stage.name} ***\n")

        # 7. Accelerate LLM decay once the system matures
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
    ) -> List[Dict]:
        """Run a list of inputs as one learning episode."""
        results = []
        for inp in inputs:
            r = self.run_cycle(inp)
            results.append(r)
            if verbose:
                self._print_cycle(r)
        return results

    # ── Generation ────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Find the best completion for a prompt using learned structures.

        Search strategy (in priority order):
          1. A structure whose output starts with the prompt (prefix match)
             — prefer longer completions over shorter ones
          2. A structure whose output equals the prompt exactly
          3. Closest match by string similarity across all levels
        """
        best_struct = None
        best_score  = -1.0
        top_level   = self.curriculum.get_active_level()

        # Search from highest level downward so richer structures win ties
        for level in range(top_level, -1, -1):
            for struct in self.library.get_at_level(level, kind='success'):
                generated = struct.generate(self.library)
                score     = self._generation_score(generated, prompt, struct.fitness)
                if score > best_score:
                    best_score  = score
                    best_struct = struct

            # If we already found a high-confidence prefix match, stop descending
            if best_score >= 2.4:
                break

        return best_struct.generate(self.library) if best_struct else ''

    @staticmethod
    def _generation_score(generated: str, prompt: str, fitness: float = 1.0) -> float:
        """
        Score a candidate structure for the given prompt.

        Priority:
          1. Prefix match weighted by fitness  (score ≥ 2.0)
             — prefers high-fitness structures over longer corrupt ones
          2. Exact match                       (score = 1.5)
          3. Generated is a fragment of prompt (score < 1.0)
          4. General similarity                (score < 0.5)
        """
        if not generated:
            return 0.0
        if generated == prompt:
            return 1.5
        if generated.startswith(prompt):
            # Weight by fitness: a tested, clean structure beats a corrupt longer one
            return 2.0 + fitness * 0.5
        if prompt.startswith(generated):
            return len(generated) / len(prompt) * 0.8
        return SequenceMatcher(None, generated, prompt).ratio() * 0.5

    # ── Dialogue ──────────────────────────────────────────────────

    def respond(self, user_input: str) -> str:
        """
        Generate a response to user input by composing library structures.
        Also learns from the input — every conversation is a learning cycle.
        """
        text     = self.sensory_motor.receive_input(user_input)
        response = self.composer.compose(text)
        # Learn from what was said to us
        self.run_cycle(text)
        return response

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str = 'deepak_memory.json') -> None:
        """Save Little Deepak's entire library to disk."""
        data = {
            'version':          '0.4',
            'persona':          'Little Deepak',
            'curriculum_stage': int(self.curriculum.current_stage),
            'library':          self.library.to_dict(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Memory] Saved {len(self.library)} structures → {path}")

    def load(self, path: str = 'deepak_memory.json') -> bool:
        """
        Load a previously saved library from disk.
        Returns True if loaded, False if file not found.
        """
        if not os.path.exists(path):
            return False
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        # Restore library
        self.library = Library.from_dict(data['library'])

        # Restore curriculum stage
        stage = data.get('curriculum_stage', 0)
        self.curriculum.current_stage = Stage(stage)

        # Rewire all modules that hold a library reference
        self.tester.llm_parent      = self.llm_parent
        self.explorer.library       = self.library
        self.composer.library       = self.library
        for unit in self.units.values():
            unit.library = self.library

        print(
            f"[Memory] Loaded {len(self.library)} structures "
            f"from {path} — stage: {self.curriculum.current_stage.name}"
        )
        return True

    # ── Exploration ───────────────────────────────────────────────

    def explore(self, n: int = 60):
        """
        Run n internal exploration attempts.
        The hypothesis engine connects dots between established
        library structures and tests if novel combinations hold.
        No external input needed.
        """
        return self.explorer.explore(n=n)

    # ── Introspection ─────────────────────────────────────────────

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