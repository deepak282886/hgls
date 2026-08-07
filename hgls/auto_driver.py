"""
auto_driver.py — Autonomous Conversation Driver

The LLM plays the role of a curious, patient parent who talks to
Little Deepak continuously. No human needed.

Each turn:
  1. LLM generates a natural input directed at Deepak
  2. Deepak responds using his library
  3. LLM evaluates the response
  4. If wrong: LLM corrects it, bad structure penalised, good one learned
  5. LLM generates the next input, influenced by what just failed

Runs for as long as you want. Saves every SAVE_EVERY turns.
Resumes from saved memory if interrupted.

Usage (via main.py --auto):
    python main.py --auto              # runs until interrupted
    python main.py --auto --turns 500  # runs for 500 turns
"""

import time
import random
from typing import Optional
from openai import OpenAI
import os

import hgls.persona as persona

TOGETHER_BASE = "https://api.together.xyz/v1"
MODEL         = "openai/gpt-oss-20b"

SAVE_EVERY        = 50     # save memory every N turns
SLEEP_BETWEEN     = 2.0    # seconds between turns (rate limit safety)
RETRY_SLEEP       = 60.0   # seconds to wait after rate limit error
MAX_RETRIES       = 5      # retries per turn before giving up
RECENT_FAILS_SIZE = 20     # how many recent failures to track for targeting


class AutoDriver:
    """
    Autonomous conversation driver.
    The LLM talks to Little Deepak continuously, correcting and teaching.
    """

    def __init__(self, system, save_fn, save_every: int = SAVE_EVERY):
        self.system     = system
        self.save_fn    = save_fn        # callable: save_fn() saves memory
        self.save_every = save_every
        self.client     = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            base_url=TOGETHER_BASE,
        )

        self._turn          = 0
        self._corrections   = 0
        self._recent_fails  = []   # recent inputs that produced bad responses
        self._session_log   = []
        self._current_input = None  # held until Deepak answers correctly
        self._attempt_count = 0

    # ── Public ────────────────────────────────────────────────────

    def run(self, max_turns: int = 0, verbose: bool = True) -> dict:
        """
        Run the autonomous conversation loop.
        max_turns = 0 means run until interrupted.
        """
        print('\n' + '=' * 60)
        print('AUTONOMOUS LEARNING SESSION')
        print(f'  Turns    : {"unlimited" if max_turns == 0 else max_turns}')
        print(f'  Save every {self.save_every} turns')
        print('  Press Ctrl+C to stop cleanly')
        print('=' * 60 + '\n')

        try:
            while True:
                if max_turns > 0 and self._turn >= max_turns:
                    break

                self._turn += 1
                self._run_turn(verbose)

                # Periodic save
                if self._turn % self.save_every == 0:
                    self.save_fn()
                    if verbose:
                        print(f'  [Auto] Turn {self._turn} — '
                              f'{len(self.system.library)} structures — '
                              f'{self._corrections} corrections so far\n')

                time.sleep(SLEEP_BETWEEN)

        except KeyboardInterrupt:
            print('\n\n[Auto] Session interrupted by user.')

        finally:
            self.save_fn()
            print(f'[Auto] Session ended. '
                  f'{self._turn} turns, {self._corrections} corrections.')

        return self._stats()

    # ── Turn ─────────────────────────────────────────────────────

    def _run_turn(self, verbose: bool) -> None:
        """
        Run one full conversation turn.
        Keeps repeating the same input until Deepak gets it right,
        just like a parent would with a child.
        """
        for attempt in range(MAX_RETRIES):
            try:
                # 1. Generate input (only on first attempt or after correction)
                if not hasattr(self, '_current_input') or self._current_input is None:
                    self._current_input  = self._generate_input()
                    self._attempt_count  = 0

                user_input = self._current_input
                if not user_input:
                    self._current_input = None
                    return

                # Skip non-English inputs
                if not all(ord(c) < 128 for c in user_input):
                    self._current_input = None
                    return

                # Skip truncated inputs
                words = user_input.strip().split()
                if len(words) < 3:
                    self._current_input = None
                    return
                last = words[-1].rstrip('.,!?')
                if last in ('do', 'is', 'are', 'how', 'what', 'the',
                            'a', 'an', 'and', 'or', 'but', 'your', 'me'):
                    self._current_input = None
                    return

                self._attempt_count += 1

                # 2. Get Deepak's response — generative unit generates directly
                response = self.system.respond(user_input)

                # Filter model reasoning leaks — discard and retry
                if self._is_leaked(response):
                    print(f'\n  [{self._turn:4d}] [leak filtered — retrying]')
                    continue

                # 3. Evaluate
                good, corrected = self._evaluate_and_correct(user_input, response)

                # 4. Display
                final_response = corrected if corrected else response
                self._log(user_input, response, final_response, good)

                status = '✓' if good else '✗'
                attempt_tag = (f' (attempt {self._attempt_count})'
                               if self._attempt_count > 1 else '')
                print(f'\n  [{self._turn:4d}] {status}{attempt_tag}')
                print(f'  Parent : {user_input}')
                print(f'  Deepak : {response}')
                if corrected and corrected != response:
                    print(f'  Corrected → {corrected}')

                # 5. Do NOT learn from the parent's input — that teaches
                # Deepak to echo questions back. Only learn from corrections.
                if good or self._attempt_count >= 10:
                    # Move on — either Deepak nailed it or enough attempts made
                    if not good and self._attempt_count >= 10:
                        print(f'  [Auto] Moving on after {self._attempt_count} attempts.')
                    self._current_input = None
                # else: stay on same input, correction already learned

                # 6. Internal exploration every 10 turns
                if self._turn % 10 == 0:
                    self.system.explorer.explore(n=20)
                    print(f'  [Explore] turn {self._turn} — '
                          f'{len(self.system.library)} structures')

                return

            except Exception as e:
                err = str(e).lower()
                if 'rate' in err or '429' in err:
                    print(f'\n  [Auto] Rate limit — waiting {RETRY_SLEEP}s...')
                    time.sleep(RETRY_SLEEP)
                elif attempt < MAX_RETRIES - 1:
                    print(f'\n  [Auto] Attempt {attempt+1} failed: {e} — retrying...')
                    time.sleep(5)
                else:
                    print(f'\n  [Auto] Turn {self._turn} failed: {e}')

    # ── Input generation ──────────────────────────────────────────

    def _generate_input(self) -> Optional[str]:
        """
        Generate questions that require reasoning, not just recall.
        Biased toward why/how/what happens questions that force
        Deepak to think step by step rather than just retrieve a fact.
        """
        # 70%: target a recent failure to close gaps
        if self._recent_fails and random.random() < 0.7:
            failed = random.choice(self._recent_fails[-10:])
            prompt = (
                f"A 5-year-old Indian child struggled with: \"{failed}\"\n"
                f"Ask a simpler reasoning question about the same topic.\n"
                f"Use why, how, or what happens — not just what.\n"
                f"Write one short question:"
            )
        else:
            # Pick a random library structure as seed
            level   = min(self.system.curriculum.get_active_level(), 4)
            structs = self.system.library.get_at_level(level, kind='success')
            if structs:
                seed      = random.choice(structs[:100])
                seed_text = seed.generate(self.system.library).strip()
                prompt    = (
                    f"Deepak knows: \"{seed_text}\"\n"
                    f"Ask a question that makes him think about why or how "
                    f"or what happens — not just what it is.\n"
                    f"Write one short question for a 5-year-old:"
                )
            else:
                prompt = (
                    f"Write one reasoning question for a 5-year-old Indian child.\n"
                    f"Use why, how, or what happens.\n"
                    f"Write one short question:"
                )

        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=512,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = (resp.choices[0].message.content or '').strip()
            raw = raw.split('\n')[0].strip().strip('"\'')
            raw = self._clean_input(raw)
            if raw and all(ord(c) < 128 for c in raw) and len(raw) > 5:
                return raw
        except Exception as e:
            print(f'  [Auto] Input generation failed: {e}')

        # Fallback
        level   = min(self.system.curriculum.get_active_level(), 3)
        structs = self.system.library.get_at_level(level, kind='success')
        if structs:
            seed = random.choice(structs[:20])
            return f"why do you {seed.generate(self.system.library).strip()}"
        return None

    # ── Evaluation and correction ─────────────────────────────────

    def _evaluate_and_correct(
        self, user_input: str, response: str
    ) -> tuple:
        """
        Evaluate Deepak's response.
        Programmatic checks first (reliable), then LLM for subtler cases.
        """
        # ── Programmatic checks (no LLM needed) ──────────────────

        # Fail: Deepak echoed the parent's question back
        if len(user_input) > 10 and user_input.lower()[:20] in response.lower():
            correction = self._request_correction(user_input, response)
            self._record_fail(user_input, response, correction)
            return False, correction

        # Fail: pure greeting in response to a question
        question_words = {
            'what', 'which', 'how', 'who', 'where', 'when',
            'do', 'did', 'can', 'tell', 'describe', 'have', 'are'
        }
        input_words  = set(user_input.lower().split())
        is_question  = bool(input_words & question_words)
        lazy_replies = {
            'namaste!', 'namaste', 'hi! i am happy to talk to you.',
            'hello! i am little deepak.', 'bye bye!',
            'goodbye! i will study hard.', 'see you! i love learning.',
        }
        if is_question and response.strip().lower() in lazy_replies:
            correction = self._request_correction(user_input, response)
            self._record_fail(user_input, response, correction)
            return False, correction

    def _evaluate_and_correct(
        self, user_input: str, response: str
    ) -> tuple:
        """
        Evaluate and correct using Chain of Thought demonstration.

        Evaluation checks:
          1. Is the response on topic?
          2. Is it age-appropriate?
          3. Does it show any thinking (not just a retrieved fact)?

        When correcting, the parent demonstrates Chain of Thought:
          - what do i know about this
          - what does that tell me
          - so my answer is

        Deepak learns the thinking pattern from the demonstrated correction,
        not just the final answer.
        """
        prompt = (
            f"A 5-year-old Indian child was asked: \"{user_input}\"\n"
            f"The child said: \"{response}\"\n\n"
            f"Evaluate:\n"
            f"1. Is it on topic?\n"
            f"2. Is it age-appropriate?\n"
            f"3. Does it make sense as an answer?\n\n"
            f"If all three yes, write only: YES\n\n"
            f"If any is no, write: NO\n"
            f"Then show how the child should think through it step by step "
            f"using this pattern:\n"
            f"i know [what i know]. [what that means]. so [my answer].\n"
            f"Keep it simple, one sentence per step, in the child's voice."
        )
        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=512,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw   = (resp.choices[0].message.content or '').strip()
            lines = [l.strip() for l in raw.split('\n') if l.strip()]

            if not lines:
                return True, None

            verdict = lines[0].upper()

            if 'YES' in verdict:
                return True, None

            if 'NO' in verdict:
                # Collect the demonstrated chain of thought
                correction = ' '.join(lines[1:]).strip().strip('"\'')
                if self._is_leaked(correction) or not correction or len(correction) < 3:
                    return True, None
                self._record_fail(user_input, response, correction)
                return False, correction

        except Exception:
            pass

        return True, None

    # ── Leak detection ────────────────────────────────────────────

    @staticmethod
    def _is_leaked(text: str) -> bool:
        """
        Detect when the model returns its own reasoning instead of
        Deepak's response. These strings signal a prompt leak.
        """
        if not text:
            return False
        text_lower = text.lower()
        leak_markers = [
            'as a 5-year-old', 'as the child', 'one sentence',
            'first person', 'keep it simple', 'respond as',
            'analysis', 'maybe simple', 'something like',
            'write as a child', 'should be one sentence',
            'i think a common', 'in english', 'indian child maybe',
            'let me think', 'the model', 'llm', 'prompt',
            'step-by-step for the child', 'stepbystep',
            'step by step for', 'child\'s voice', 'childs voice',
            'thinking childs', 'in the child', 'for the child',
        ]
        return any(m in text_lower for m in leak_markers)

    @staticmethod
    def _clean_input(text: str) -> str:
        """Strip markdown formatting from parent questions."""
        import re
        text = re.sub(r'\*+', '', text)   # remove ** bold markers
        text = re.sub(r'_+', '', text)    # remove _ italic markers
        text = re.sub(r'\s+', ' ', text)  # normalise whitespace
        return text.strip().lower()

    def _record_fail(self, user_input: str, response: str,
                     correction: str) -> None:
        """Record failure and apply correction."""
        self._apply_correction(response, correction)
        self._recent_fails.append(user_input)
        if len(self._recent_fails) > RECENT_FAILS_SIZE:
            self._recent_fails.pop(0)
        self._corrections += 1

    def _apply_correction(self, bad: str, good: str) -> None:
        """
        Penalise the bad response and learn the corrected one.
        Filters leaked model reasoning before anything enters the library.
        """
        # Don't store leaked model reasoning in the library
        if self._is_leaked(good) or self._is_leaked(bad):
            return
        from hgls.structures import GenerativeStructure
        level = self.system.curriculum.get_active_level()

        # Mark bad as failure
        bad_struct = GenerativeStructure(
            level=level,
            elements=list(bad),
            source='generated',
            fitness=0.0,
        )
        self.system.library.add_failure(bad_struct)

        # Learn the corrected version — enters library through normal pipeline
        # Next time generate() searches the library, it finds the correction
        self.system.run_cycle(good)

    # ── Logging ───────────────────────────────────────────────────

    def _log(self, user_input, original, final, good):
        self._session_log.append({
            'turn':     self._turn,
            'input':    user_input,
            'original': original,
            'final':    final,
            'good':     good,
        })

    def _stats(self) -> dict:
        return {
            'turns':            self._turn,
            'corrections':      self._corrections,
            'correction_rate':  round(
                self._corrections / max(1, self._turn), 3
            ),
            'library_size':     len(self.system.library),
            'recent_fails':     len(self._recent_fails),
        }