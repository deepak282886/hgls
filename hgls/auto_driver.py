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


# ── Topic seeds ───────────────────────────────────────────────────
# The driver cycles through these to keep conversation broad.
# Weighted toward things Deepak is still weak on.

TOPIC_SEEDS = [
    # identity
    "greet Little Deepak",
    "ask Little Deepak his name",
    "ask how he is feeling today",
    # health
    "ask if he brushes his teeth",
    "ask what he eats for breakfast",
    "ask when he sleeps",
    "ask if he drinks water",
    "ask about his morning routine",
    # family
    "ask about his amma",
    "ask about his appa",
    "ask about his didi",
    "ask about his bhaiya",
    "ask who he loves",
    "ask who he helps at home",
    # respect
    "ask how he greets elders",
    "ask if he says namaste",
    "ask what he does when he makes a mistake",
    # feelings
    "ask if he is happy",
    "ask what makes him happy",
    "ask what makes him sad",
    "ask if he ever feels proud",
    # learning
    "ask about school",
    "ask what he learns",
    "ask about his favourite book",
    "ask if he likes to read",
    "ask about his teacher",
    # values
    "ask if he shares his food",
    "ask if he tells the truth",
    "ask if he helps his friends",
    "ask what he does when a friend is sad",
    # world knowledge
    "ask about colors he can see",
    "ask what he sees outside",
    "ask about animals he knows",
    "ask about the weather",
    "ask about numbers he knows",
    # open
    "ask Little Deepak to tell you something he learned today",
    "ask Little Deepak what a good child does",
    "ask Little Deepak to say something kind",
]


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
        self._topic_idx     = 0
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

                self._attempt_count += 1

                # 2. Get Deepak's response
                response = self.system.composer.compose(
                    self.system.sensory_motor.receive_input(user_input)
                )

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

                # 5. Learn from input
                self.system.run_cycle(
                    self.system.sensory_motor.receive_input(user_input)
                )

                if good:
                    # Deepak nailed it — move to next topic
                    self._current_input = None
                else:
                    # Stay on the same input — try again next turn
                    # The correction has already been learned, so next
                    # attempt should produce a better response
                    pass

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
        Generate the next thing to say to Deepak.
        Biased toward topics where Deepak recently failed.
        """
        # 70% of the time: target a recent failure if we have one
        if self._recent_fails and random.random() < 0.7:
            failed_input = random.choice(self._recent_fails[-10:])
            prompt = (
                f"You are talking to Little Deepak, a good 5-year-old Indian child.\n"
                f"He recently struggled with: '{failed_input}'\n\n"
                f"Ask him about the same topic in a different, simpler way. "
                f"One short friendly sentence. No explanation."
            )
        else:
            # Pick next topic seed
            topic = TOPIC_SEEDS[self._topic_idx % len(TOPIC_SEEDS)]
            self._topic_idx += 1
            prompt = (
                f"You are talking to Little Deepak, a good 5-year-old Indian child.\n"
                f"Your task: {topic}\n\n"
                f"Write one short, friendly sentence to say to him. "
                f"Simple language. No explanation."
            )

        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=80,
                messages=[
                    {'role': 'system', 'content': persona.PARENT_SYSTEM_PROMPT},
                    {'role': 'user',   'content': prompt},
                ],
            )
            raw = resp.choices[0].message.content or ''
            # Clean up — take first sentence only, strip quotes
            raw = raw.strip().strip('"\'').split('\n')[0].strip()
            return raw.lower()
        except Exception as e:
            print(f'  [Auto] Input generation failed: {e}')
            # Fall back to a direct topic seed as plain text
            seed = TOPIC_SEEDS[self._topic_idx % len(TOPIC_SEEDS)]
            self._topic_idx += 1
            return seed

    # ── Evaluation and correction ─────────────────────────────────

    def _evaluate_and_correct(
        self, user_input: str, response: str
    ) -> tuple:
        """
        Ask the LLM parent to evaluate Deepak's response.
        Returns (is_good, corrected_text_or_None).
        """
        prompt = (
            f"Someone said to Little Deepak: '{user_input}'\n"
            f"Little Deepak replied: '{response}'\n\n"
            f"Is this a natural, correct, age-appropriate reply for a "
            f"good 5-year-old Indian child?\n\n"
            f"If YES: reply with just 'good'\n"
            f"If NO: reply with 'bad: ' followed by what Little Deepak "
            f"should have said instead. Keep it simple and in his voice."
        )

        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=80,
                messages=[
                    {'role': 'system', 'content': persona.PARENT_SYSTEM_PROMPT},
                    {'role': 'user',   'content': prompt},
                ],
            )
            raw = (resp.choices[0].message.content or '').strip().lower()

            if raw.startswith('good'):
                return True, None

            elif raw.startswith('bad:'):
                correction = raw[4:].strip().strip('"\'')
                if correction and len(correction) > 3:
                    self._apply_correction(response, correction)
                    self._recent_fails.append(user_input)
                    if len(self._recent_fails) > RECENT_FAILS_SIZE:
                        self._recent_fails.pop(0)
                    self._corrections += 1
                    return False, correction

        except Exception:
            pass

        return True, None   # on failure, assume OK

    def _apply_correction(self, bad: str, good: str) -> None:
        """
        Penalise the bad response and learn the corrected one.
        """
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

        # Learn the corrected version
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