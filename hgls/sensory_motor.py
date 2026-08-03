"""
sensory_motor.py — Sensory-Motor Interface.

All interaction is mediated through keyboard-level primitives.
Normalises raw text into processable form.
Embodiment is explicitly out of scope.
"""

import re
from typing import List

KEYBOARD = set("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-")


class SensoryMotorInterface:

    def __init__(self):
        self._input_history:  List[str] = []
        self._output_history: List[str] = []

    def receive_input(self, raw: str) -> str:
        """Normalise raw text: lowercase, strip unsupported chars, compress whitespace."""
        text = raw.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(c for c in text if c in KEYBOARD)
        self._input_history.append(text)
        return text

    def produce_output(self, text: str) -> str:
        self._output_history.append(text)
        return text

    def segment(self, text: str) -> List[str]:
        """
        Segment input into learnable units.
        Short strings → character list; space-delimited → word list; else whole.
        """
        if len(text) <= 3:
            return list(text)
        if ' ' in text:
            return text.split(' ')
        return [text]

    def get_primitives(self) -> List[str]:
        return sorted(KEYBOARD)

    def stats(self) -> dict:
        return {
            'total_inputs':  len(self._input_history),
            'total_outputs': len(self._output_history),
            'unique_inputs': len(set(self._input_history)),
        }