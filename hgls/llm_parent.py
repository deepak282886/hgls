"""
llm_parent.py — LLM Parental Interface (Together AI Edition)

Model  : togethercomputer/GPT-NeoXT-Chat-Base-20B
API    : Together AI (OpenAI-compatible endpoint)
Key env: TOGETHER_API_KEY

The parent persona is Little Deepak's loving parent — warm, value-driven,
evaluating all content through the lens of a good Indian childhood.
Signal strength starts at 1.0 and decays with each call, fading as
Little Deepak's internal reward system matures.
"""

import os
import json
from typing import List, Optional, Tuple

from openai import OpenAI

from hgls.structures import GenerativeStructure
import hgls.persona as persona

TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Together AI model string — verify at https://api.together.ai/models
MODEL = "openai/gpt-oss-20b"

DECAY_RATE   = 0.995
MIN_STRENGTH = 0.05


class LLMParentalInterface:

    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            base_url=TOGETHER_BASE_URL,
        )
        self.signal_strength = 1.0
        self._call_count     = 0
        self._enabled        = True

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate_reconstruction(self, generated: str, target: str) -> float:
        """
        Ask the parent: how well does 'generated' match 'target'?
        Returns score in [0, 1].
        """
        if not self._active():
            return 0.5

        prompt = (
            f"Target   : '{target}'\n"
            f"Generated: '{generated}'\n\n"
            f"How well does the generated text match the target? "
            f"Reply with only a single number between 0.0 and 1.0. Nothing else."
        )
        try:
            return max(0.0, min(1.0, float(self._call(prompt, max_tokens=10).strip())))
        except Exception:
            return 0.5

    def propose_structures(
        self, target: str, level: int, n: int = 3
    ) -> List[GenerativeStructure]:
        """
        Ask the parent to propose decompositions of target at this level.
        """
        if not self._active():
            return []

        level_label = {
            0: "individual characters",
            1: "short letter combinations",
            2: "whole words",
            3: "phrase chunks",
        }.get(level, "elements")

        prompt = (
            f"Decompose '{target}' into {level_label}.\n"
            f"Reply ONLY with a JSON array of arrays.\n"
            f"Example for 'cat': [[\"c\",\"a\",\"t\"],[\"ca\",\"t\"]]\n"
            f"Give {n} decompositions. No explanation, no markdown, no extra text."
        )
        try:
            raw = self._call(prompt, max_tokens=200).strip()
            if '```' in raw:
                raw = raw.split('```')[1].lstrip('json').strip()
            parsed = json.loads(raw)
            return [
                GenerativeStructure(
                    level=level,
                    elements=[str(e) for e in decomp],
                    source='llm',
                )
                for decomp in parsed[:n]
                if isinstance(decomp, list)
            ]
        except Exception:
            return []

    def judge(self, text: str, context: str = "") -> Tuple[str, float]:
        """
        Simple parental judgment: correct / good / bad + confidence.
        """
        if not self._active():
            return 'neutral', 0.5

        prompt = (
            f"Context: {context}\n"
            f"Text: '{text}'\n\n"
            f"Is this good content for {persona.NAME}? "
            f"Reply with one word (correct/good/bad) then a confidence 0-1. "
            f"Example: good 0.8"
        )
        try:
            parts      = self._call(prompt, max_tokens=20).strip().split()
            label      = parts[0].lower() if parts else 'neutral'
            confidence = float(parts[1]) if len(parts) > 1 else 0.5
            return label, confidence
        except Exception:
            return 'neutral', 0.5

    # ── Internals ─────────────────────────────────────────────────

    def _active(self) -> bool:
        return self._enabled and self.signal_strength >= MIN_STRENGTH

    def _call(self, prompt: str, max_tokens: int = 100) -> str:
        self._call_count += 1
        resp = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": persona.PARENT_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        self._decay()
        return resp.choices[0].message.content or ""

    def _decay(self) -> None:
        self.signal_strength = max(MIN_STRENGTH, self.signal_strength * DECAY_RATE)

    def disable(self) -> None:
        self._enabled = False

    def stats(self) -> dict:
        return {
            'model':           MODEL,
            'api':             'Together AI',
            'signal_strength': round(self.signal_strength, 4),
            'call_count':      self._call_count,
            'enabled':         self._enabled,
        }