"""
chain_generator.py — Reasoning Chain Generator using GPT-OSS 20B.

Generates the HGLS Universal Reasoning Chain — four tiers:

  Tier 1 — Anchor    : i know [fact grounded in library]
  Tier 2 — Implication: that means [direct logical consequence]
  Tier 3 — Inference  : so [conclusion]
  Tier 4 — Verify     : this is true because [supporting evidence]

Tier 4 is self-verification — GPT-OSS checks its own chain.
Chains that fail self-verification are discarded before teaching.

Four difficulty levels:
  1 — single hop  : one concept, one implication, one conclusion
  2 — two hop     : two concepts combined, chained implication
  3 — cross domain: pattern from one domain applied to another
  4 — meta        : reasoning about reasoning, self-correction

GPT-OSS 20B is called once per pair.
Response is parsed into (question, chain, verified).
Only verified chains enter the teaching loop.
"""

import re
import os
import json
import time
import urllib.request
from typing import Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Together AI config
TOGETHER_API_URL = 'https://api.together.xyz/v1/chat/completions'
MODEL            = 'openai/gpt-oss-20b'

# Retry on failure
MAX_RETRIES = 2
RETRY_DELAY = 1.0

# Generation params
MAX_TOKENS  = 300
TEMPERATURE = 0.7


# ── Prompt templates per difficulty ──────────────────────────────

_PROMPT_D1 = """You are building a reasoning dataset for a learning system.

Generate a question and a four-tier reasoning chain about this concept: {concepts}

Use EXACTLY this format — no deviations:

QUESTION: [a clear why/how/what question about {concepts}]
CHAIN:
i know [one fact about {concepts}].
that means [direct logical consequence].
so [conclusion that follows].
this is true because [one supporting fact or analogy].
VERIFIED: yes

Rules:
- Use simple clear language
- Each line must follow from the previous
- VERIFIED must be yes only if the chain is logically sound
- If the chain is not sound write VERIFIED: no
- Never add extra lines or commentary"""

_PROMPT_D2 = """You are building a reasoning dataset for a learning system.

Generate a question and a two-hop reasoning chain connecting these concepts: {concepts}

Use EXACTLY this format:

QUESTION: [a clear why/how/what question connecting {concepts}]
CHAIN:
i know [fact about {concept_a}].
i know [fact about {concept_b}].
that means [combined implication connecting both].
so [conclusion].
this is true because [supporting evidence].
VERIFIED: yes

Rules:
- Both facts must connect logically to form the implication
- VERIFIED: yes only if the chain is logically sound
- Simple clear language"""

_PROMPT_D3 = """You are building a reasoning dataset for a learning system.

Generate a cross-domain reasoning chain showing how {concept_a} and {concept_b} follow the same pattern.
Also involve: {concept_c}

Use EXACTLY this format:

QUESTION: [a question asking how {concept_a} and {concept_b} are similar]
CHAIN:
i know [pattern in {concept_a}].
i know [same pattern in {concept_b}].
that means [the underlying shared structure].
so [what understanding one tells us about the other].
this is true because [why the analogy holds].
VERIFIED: yes

Rules:
- The analogy must be genuine, not forced
- VERIFIED: yes only if the analogy is real"""

_PROMPT_D4 = """You are building a reasoning dataset for a learning system.

Generate a meta-reasoning chain — reasoning about how to reason about: {concepts}

Use EXACTLY this format:

QUESTION: [a question about how to think about {concepts} correctly]
CHAIN:
i know [common mistake or misconception about {concepts}].
that means [why that mistake leads to wrong conclusions].
so [the correct way to reason about {concepts}].
this is true because [what makes this reasoning reliable].
VERIFIED: yes

Rules:
- The misconception must be real and common
- VERIFIED: yes only if the correction is sound"""

_PROMPTS = {1: _PROMPT_D1, 2: _PROMPT_D2, 3: _PROMPT_D3, 4: _PROMPT_D4}


class ChainGenerator:

    def __init__(self, api_key: str = None):
        self._api_key = api_key or os.environ.get('TOGETHER_API_KEY', '')
        self._total_generated  = 0
        self._total_verified   = 0
        self._total_rejected   = 0
        self._total_errors     = 0
        self._call_times: List[float] = []

    # ── Main: generate one chain ──────────────────────────────────

    def generate(
        self,
        concepts: List[str],
        difficulty: int = 1,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Generate one reasoning chain for given concepts at given difficulty.

        Returns (question, chain, verified).
        Returns (None, None, False) on failure or rejected chain.
        """
        prompt = self._build_prompt(concepts, difficulty)
        if not prompt:
            return None, None, False

        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = self._call_llm(prompt)
                question, chain, verified = self._parse(raw)
                self._total_generated += 1

                if verified and question and chain:
                    self._total_verified += 1
                    return question, chain, True
                else:
                    self._total_rejected += 1
                    return question, chain, False

            except Exception as e:
                self._total_errors += 1
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        return None, None, False

    # ── Prompt building ───────────────────────────────────────────

    def _build_prompt(self, concepts: List[str], difficulty: int) -> Optional[str]:
        template = _PROMPTS.get(difficulty, _PROMPT_D1)
        concepts_str = ' and '.join(concepts)

        if difficulty == 1:
            return template.format(concepts=concepts_str)

        elif difficulty == 2:
            if len(concepts) < 2:
                concepts = concepts + concepts
            return template.format(
                concepts  = concepts_str,
                concept_a = concepts[0],
                concept_b = concepts[1],
            )

        elif difficulty == 3:
            if len(concepts) < 3:
                concepts = (concepts + concepts + concepts)[:3]
            return template.format(
                concepts  = concepts_str,
                concept_a = concepts[0],
                concept_b = concepts[1],
                concept_c = concepts[2],
            )

        elif difficulty == 4:
            return template.format(concepts=concepts_str)

        return None

    # ── LLM call ─────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        t0 = time.time()

        payload = json.dumps({
            'model':       MODEL,
            'messages':    [{'role': 'user', 'content': prompt}],
            'max_tokens':  MAX_TOKENS,
            'temperature': TEMPERATURE,
        }).encode()

        headers = {
            'Content-Type':  'application/json',
            'Authorization': f'Bearer {self._api_key}',
        }

        req  = urllib.request.Request(
            TOGETHER_API_URL,
            data    = payload,
            headers = headers,
            method  = 'POST',
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        output = data['choices'][0]['message']['content'].strip()
        self._call_times.append(time.time() - t0)
        return output

    # ── Response parsing ──────────────────────────────────────────

    def _parse(self, raw: str) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Parse LLM response into (question, chain, verified).

        Expected format:
          QUESTION: ...
          CHAIN:
          i know ...
          that means ...
          so ...
          this is true because ...
          VERIFIED: yes/no
        """
        question = None
        chain_lines = []
        verified = False

        lines = raw.strip().split('\n')

        in_chain = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith('QUESTION:'):
                question = line[9:].strip()
                in_chain = False

            elif line.upper().startswith('CHAIN:'):
                in_chain = True

            elif line.upper().startswith('VERIFIED:'):
                answer = line[9:].strip().lower()
                verified = answer.startswith('yes')
                in_chain = False

            elif in_chain:
                # Valid chain lines start with known prefixes
                low = line.lower()
                if (low.startswith('i know') or
                    low.startswith('that means') or
                    low.startswith('so ') or
                    low.startswith('this is true')):
                    chain_lines.append(line)

        chain = '\n'.join(chain_lines) if chain_lines else None

        # Minimum quality check
        if chain and len(chain_lines) < 3:
            return question, None, False

        return question, chain, verified

    # ── Stats ─────────────────────────────────────────────────────

    def avg_call_time(self) -> float:
        if not self._call_times:
            return 0.0
        return sum(self._call_times[-20:]) / len(self._call_times[-20:])

    def stats(self) -> dict:
        return {
            'total_generated': self._total_generated,
            'total_verified':  self._total_verified,
            'total_rejected':  self._total_rejected,
            'total_errors':    self._total_errors,
            'verify_rate':     round(
                self._total_verified / max(self._total_generated, 1), 3
            ),
            'avg_call_time_s': round(self.avg_call_time(), 2),
        }