"""
llm_validator.py — LLM Validator for Novel Connections.

Only called for novel connections proposed by the tinkering engine
that passed emotional pre-screening.

The LLM's role has shifted from teacher/corrector to validator.
It answers one question per proposal:
  "Is this connection real, meaningful, and generalisable?"

Results are cached — never validate the same pair twice.
Rejected connections are remembered to suppress similar proposals.

LLM calls are rare by design:
  - Emotional evaluator pre-screens most proposals out
  - Cache prevents re-validation
  - System becomes more autonomous as emotional evaluator matures
"""

import json
import os
import time
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.graph import MemoryGraph

# Cache file
VALIDATION_CACHE_PATH = 'deepak_validation_cache.json'

# Together AI model
MODEL = 'mistralai/Mistral-7B-Instruct-v0.1'

# Max proposals per validation batch
BATCH_SIZE = 5

# Validation prompt template
PROMPT_TEMPLATE = """You are helping a learning system decide if a connection between two concepts is real and meaningful.

Concept A: "{source}"
Concept B: "{target}"
Connection type: {strategy}

Is there a genuine, meaningful, generalisable connection between these two concepts?

Answer with exactly:
YES if the connection is real and would help understand both concepts better.
NO if the connection is forced, coincidental, or misleading.

Then in one sentence explain why.

Format:
ANSWER: YES/NO
REASON: <one sentence>"""


class LLMValidator:

    def __init__(self, api_key: str = None):
        self._api_key = api_key
        self._cache: Dict[str, dict] = {}
        self._load_cache()

        self._total_calls    = 0
        self._total_yes      = 0
        self._total_no       = 0
        self._total_cached   = 0
        self._signal_strength = 1.0

    # ── Main: validate proposals ──────────────────────────────────

    def validate_batch(
        self,
        proposals: List[Dict],
        graph: Optional['MemoryGraph'] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate a batch of novel connection proposals.

        Returns (accepted, rejected) lists.
        Accepted proposals are added to the graph as novel edges.
        Rejected proposals are recorded to suppress similar future proposals.
        """
        accepted = []
        rejected = []

        for proposal in proposals[:BATCH_SIZE]:
            src_text = proposal.get('source_text', '')
            tgt_text = proposal.get('target_text', '')
            strategy = proposal.get('strategy', 'novel')

            # Check cache first
            cache_key = self._cache_key(src_text, tgt_text)
            if cache_key in self._cache:
                self._total_cached += 1
                cached = self._cache[cache_key]
                if cached['answer'] == 'YES':
                    accepted.append({**proposal, **cached})
                else:
                    rejected.append({**proposal, **cached})
                continue

            # Call LLM
            answer, reason = self._call_llm(src_text, tgt_text, strategy)
            result = {'answer': answer, 'reason': reason}
            self._cache[cache_key] = result

            if answer == 'YES':
                self._total_yes += 1
                accepted.append({**proposal, **result})
                if graph:
                    src_struct = None
                    tgt_struct = None
                    # Get levels from proposal metadata if available
                    graph.add_novel(
                        source_id    = proposal['source_id'],
                        target_id    = proposal['target_id'],
                        source_level = 0,
                        target_level = 0,
                        metadata     = {'reason': reason, 'strategy': strategy},
                    )
            else:
                self._total_no += 1
                rejected.append({**proposal, **result})
                if graph:
                    graph.add_novel_failure(
                        source_id = proposal['source_id'],
                        target_id = proposal['target_id'],
                    )

        self._save_cache()
        return accepted, rejected

    # ── LLM call ─────────────────────────────────────────────────

    def _call_llm(
        self,
        source_text: str,
        target_text: str,
        strategy: str,
    ) -> Tuple[str, str]:
        """
        Call Together AI to validate a novel connection.
        Returns (answer, reason) where answer is 'YES' or 'NO'.
        """
        self._total_calls += 1

        prompt = PROMPT_TEMPLATE.format(
            source   = source_text[:100],
            target   = target_text[:100],
            strategy = strategy,
        )

        try:
            import urllib.request
            import json as _json

            headers = {
                'Content-Type':  'application/json',
                'Authorization': f'Bearer {self._api_key or os.environ.get("TOGETHER_API_KEY", "")}',
            }

            payload = _json.dumps({
                'model':       MODEL,
                'prompt':      prompt,
                'max_tokens':  60,
                'temperature': 0.1,
                'stop':        ['\n\n'],
            }).encode()

            req = urllib.request.Request(
                'https://api.together.xyz/inference',
                data    = payload,
                headers = headers,
                method  = 'POST',
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data   = _json.loads(resp.read())
                output = data['output']['choices'][0]['text'].strip()

            answer, reason = self._parse_response(output)
            return answer, reason

        except Exception as e:
            # On any error: conservative answer is NO
            return 'NO', f'validation_error: {str(e)[:50]}'

    def _parse_response(self, text: str) -> Tuple[str, str]:
        """Parse LLM response into (answer, reason)."""
        lines  = text.strip().split('\n')
        answer = 'NO'
        reason = ''

        for line in lines:
            line = line.strip()
            if line.startswith('ANSWER:'):
                a = line.replace('ANSWER:', '').strip().upper()
                if 'YES' in a:
                    answer = 'YES'
                elif 'NO' in a:
                    answer = 'NO'
            elif line.startswith('REASON:'):
                reason = line.replace('REASON:', '').strip()

        return answer, reason

    # ── Cache ─────────────────────────────────────────────────────

    def _cache_key(self, src: str, tgt: str) -> str:
        a, b = sorted([src[:50], tgt[:50]])
        return f"{a}|||{b}"

    def _load_cache(self) -> None:
        if os.path.exists(VALIDATION_CACHE_PATH):
            try:
                with open(VALIDATION_CACHE_PATH) as f:
                    self._cache = json.load(f)
                print(f"[LLMValidator] Loaded {len(self._cache)} cached validations.")
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            tmp = VALIDATION_CACHE_PATH + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self._cache, f, ensure_ascii=False)
            os.replace(tmp, VALIDATION_CACHE_PATH)
        except Exception:
            pass

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'total_calls':   self._total_calls,
            'total_yes':     self._total_yes,
            'total_no':      self._total_no,
            'total_cached':  self._total_cached,
            'cache_size':    len(self._cache),
            'yes_rate':      round(self._total_yes / max(self._total_calls, 1), 3),
        }
