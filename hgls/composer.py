"""
composer.py — Response Composer

Generates responses to user input by composing multiple
library structures — not retrieval, not next-token prediction.

Process:
  1. Extract key words from user input
  2. Find library structures that overlap with those words
  3. Select a relevant phrase + a supporting schema
  4. Assemble into a natural Little Deepak response

Little Deepak speaks simply, warmly, and from his own experience.
"""

import random
from typing import List, Optional, Tuple
from difflib import SequenceMatcher

from hgls.library    import Library
from hgls.curriculum import CurriculumController
import hgls.persona  as persona

# Question/filler words that don't carry content
STOP_WORDS = {
    'do', 'you', 'your', 'what', 'how', 'are', 'is', 'can', 'does',
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'and', 'or', 'but',
    'that', 'this', 'it', 'he', 'she', 'they', 'we', 'me', 'my',
    'yes', 'no', 'not', 'so', 'tell', 'about', 'know', 'like', 'want',
    'have', 'has', 'did', 'will', 'would', 'could', 'should', 'go',
}

GREETINGS   = {'hi', 'hello', 'hey', 'namaste', 'hiya'}
FAREWELLS   = {'bye', 'goodbye', 'see', 'later', 'goodnight'}
AFFIRMATIVES = ['yes', 'yes!', 'yes,']


class ResponseComposer:
    """
    Composes Little Deepak's responses from his library.
    Each response combines:
      - An affirmation or acknowledgement
      - A relevant phrase from his experience
      - An optional schema showing he understands cause and effect
    """

    def __init__(self, library: Library, curriculum: CurriculumController):
        self.library    = library
        self.curriculum = curriculum
        self._qa_pairs: dict = {}   # question → learned answer from corrections

    def add_learned_answer(self, question: str, answer: str) -> None:
        """Store a correction as a direct question→answer pair."""
        self._qa_pairs[question.lower().strip()] = answer.lower().strip()

    # ── Public ────────────────────────────────────────────────────

    def compose(self, user_input: str) -> str:
        """Compose a response to user_input from library structures."""
        text = user_input.lower().strip().rstrip('?!.')

        # Check learned QA pairs first — corrections take priority
        qa_answer = self._find_qa_answer(text)
        if qa_answer:
            return qa_answer

        # Handle greetings and farewells directly
        words = set(text.split())
        if words & GREETINGS:
            return random.choice([
                'namaste!',
                'hello! i am little deepak.',
                'hi! i am happy to talk to you.',
            ])
        if words & FAREWELLS:
            return random.choice([
                'bye bye!',
                'goodbye! i will study hard.',
                'see you! i love learning.',
            ])

        # Extract content words
        content_words = words - STOP_WORDS

        # Find relevant structures
        phrases = self._find_relevant(content_words, level=3)
        schemas = self._find_relevant(content_words, level=4)

        parts = self._assemble(content_words, phrases, schemas)

        if not parts:
            return self._default_response(content_words)

        return self._format(parts)

    # ── QA lookup ─────────────────────────────────────────────────

    def _find_qa_answer(self, text: str) -> Optional[str]:
        """
        Check if we have a learned answer for this input.
        Uses fuzzy matching so slight rephrasing still finds the answer.
        """
        from difflib import SequenceMatcher
        best_score  = 0.0
        best_answer = None
        for question, answer in self._qa_pairs.items():
            score = SequenceMatcher(None, text, question).ratio()
            if score > best_score:
                best_score  = score
                best_answer = answer
        if best_score >= 0.65:
            return best_answer
        return None

    # ── Assembly ──────────────────────────────────────────────────

    def _assemble(
        self,
        content_words: set,
        phrases: List[Tuple],
        schemas: List[Tuple],
    ) -> List[str]:
        parts = []

        if phrases:
            phrase_text = phrases[0][2].generate(self.library)
            # Affirm if it sounds like a question about what he does
            affirm = random.choice(AFFIRMATIVES) + ' '
            parts.append(affirm + phrase_text)

            # Add a second distinct phrase if available
            if len(phrases) > 1:
                p2 = phrases[1][2].generate(self.library)
                if p2 != phrase_text and len(parts[0]) + len(p2) < 80:
                    parts.append(p2)

        if schemas:
            schema_text = schemas[0][2].generate(self.library)
            # Only add if not redundant with what we already have
            already_said = ' '.join(parts)
            if schema_text not in already_said:
                parts.append(schema_text)

        return parts

    # ── Search ────────────────────────────────────────────────────

    def _find_relevant(
        self,
        content_words: set,
        level: int,
    ) -> List[Tuple]:
        """
        Find library structures at this level that overlap with content_words.
        Returns list of (overlap_count, fitness, structure) sorted descending.
        """
        results = []
        for struct in self.library.get_at_level(level, kind='success'):
            text       = struct.generate(self.library)
            text_words = set(text.split())
            overlap    = len(content_words & text_words)
            if overlap > 0 and len(text) > 3:
                results.append((overlap, struct.fitness, struct))

        results.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return results

    # ── Fallbacks ─────────────────────────────────────────────────

    def _default_response(self, content_words: set) -> str:
        """When nothing relevant is found, respond with a known value."""
        # Pick a random schema Little Deepak knows
        schemas = self.library.get_at_level(4, kind='success')
        if schemas:
            return random.choice(schemas[:10]).generate(self.library)

        phrases = self.library.get_at_level(3, kind='success')
        if phrases:
            return random.choice(phrases[:10]).generate(self.library)

        return 'i am little deepak. i learn every day.'

    @staticmethod
    def _format(parts: List[str]) -> str:
        """Join parts into a natural response."""
        if not parts:
            return ''
        # Capitalise first word, join with '. '
        out = '. '.join(p.strip() for p in parts if p.strip())
        return out