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

        # One generative unit per level (0–6)
        # Same algorithm at every level — only the content differs
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

        # Exploration engine — internal dot-connecting
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

    def run_cycle(self, raw_input: str, target_level: int = None) -> Dict[str, Any]:
        """
        Process one input through the full learning cycle.
        target_level overrides the curriculum stage — used during domain expansion
        to learn words at level 2, phrases at level 3, schemas at level 4.
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
        level    = target_level if target_level is not None else self.curriculum.get_active_level()
        salience = self.attention.compute_salience(text, level)
        n_hyp    = self.attention.allocate_hypotheses(14, salience)

        # 4. Learn
        unit          = self.units[level]
        cycle_results = unit.learn(text, n_hypotheses=n_hyp)

        # 5. Mark self-generated structures
        for struct, outcome, score in cycle_results:
            if outcome == 'success':
                self.self_model.mark_self_generated(struct)

        # 6. Curriculum tick + advancement (only when following curriculum naturally)
        advanced = False
        if target_level is None:
            self.curriculum.tick()
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
        target_level: int = None,
    ) -> List[Dict]:
        """Run a list of inputs as one learning episode."""
        results = []
        for inp in inputs:
            r = self.run_cycle(inp, target_level=target_level)
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
        Generate a response using the generative unit directly.
        Same unit that learns also generates — no separate modules.

        Flow:
          1. Generative unit at active level generates from library
          2. Falls back to lower levels if active level has no relevant content
          3. LLM parent corrects if wrong — correction learned immediately
        """
        text = self.sensory_motor.receive_input(user_input)
        self.memory.clear()   # fresh start for each new question
        self.memory.push(text)
        context = self.memory.get_context()

        # Try active level first, fall back through levels
        level    = self.curriculum.get_active_level()
        response = ''
        for lvl in range(level, -1, -1):
            response = self.units[lvl].generate(text, context=context)
            if response:
                break

        # LLM correction
        if response and self.llm_parent and self.llm_parent._active():
            response = self._parent_correct(text, response)

        return response

    def _parent_correct(self, user_input: str, response: str) -> str:
        """
        Ask the LLM parent to evaluate Deepak's response.
        If wrong: mark as failure, propose correction, learn it.
        If right: reinforce the structures that produced it.
        """
        # Ask parent: is this a good response for Little Deepak?
        context = (
            f"Someone said to Little Deepak: '{user_input}'. "
            f"Little Deepak replied: '{response}'. "
            f"Is this a natural, correct, age-appropriate reply?"
        )
        judgment, confidence = self.llm_parent.judge(response, context=context)

        if judgment in ('bad',) and confidence > 0.5:
            # Response is wrong — ask parent for the correct version
            correction = self._request_correction(user_input, response)
            if correction and correction != response:
                # Mark original response as failure
                bad_struct = GenerativeStructure(
                    level=self.curriculum.get_active_level(),
                    elements=list(response),
                    source='generated',
                    fitness=0.0,
                )
                self.library.add_failure(bad_struct)

                # Learn the corrected version
                self.run_cycle(correction)
                return correction

        return response

    def _request_correction(self, user_input: str, bad_response: str) -> str:
        """Ask the LLM parent what Little Deepak should have said."""
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
            # Clean up: lowercase, strip quotes
            corrected = corrected.lower().strip().strip('"\'')
            if len(corrected) > 3:
                return corrected
        except Exception:
            pass
        return bad_response

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str = 'deepak_memory.json') -> None:
        """
        Save Little Deepak's library to disk.
        Writes to a temp file first then renames — prevents corruption
        if the process is interrupted mid-write.
        Also keeps a .bak of the previous good save.
        """
        data = {
            'version':          '0.4',
            'persona':          'Little Deepak',
            'curriculum_stage': int(self.curriculum.current_stage),
            'library':          self.library.to_dict(),
        }
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Rotate: current → .bak, tmp → current
        backup = path + '.bak'
        if os.path.exists(path):
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(path, backup)
        os.rename(tmp, path)
        print(f"[Memory] Saved {len(self.library)} structures → {path}")

    def load(self, path: str = 'deepak_memory.json') -> bool:
        """
        Load a previously saved library from disk.
        Returns True if loaded, False if file not found or corrupted.
        """
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

        # Restore library
        self.library = Library.from_dict(data['library'])

        # Restore curriculum stage
        stage = data.get('curriculum_stage', 0)
        self.curriculum.current_stage = Stage(stage)

        # Rewire all modules that hold a library reference
        self.tester.llm_parent = self.llm_parent
        self.explorer.library  = self.library
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