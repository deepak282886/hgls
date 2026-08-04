"""
curriculum_generator.py — LLM-driven Curriculum Generator

The LLM parent generates curriculum content for Little Deepak
on any domain at any level. Results are cached to disk so
nothing is ever regenerated between runs.

This is how a parent teaches a child: talking about the world,
grounded in values, expanding outward from the familiar to the new.
The values stay small. The world is infinite.
"""

import os
import json
import re
from typing import List
from openai import OpenAI

import hgls.persona as persona

CACHE_FILE    = 'curriculum_cache.json'
TOGETHER_BASE = "https://api.together.xyz/v1"
MODEL         = "openai/gpt-oss-20b"

# Items generated per domain per level
COUNTS = {2: 25, 3: 20, 4: 12}

# Developmental domain sequence
# Each domain unlocks after the previous saturates.
# Ordered from closest to Deepak's immediate world outward.
DOMAINS = [
    "body parts",               # head hands legs eyes nose ears mouth
    "colors and shapes",        # red blue green round big small
    "numbers and counting",     # one two three count first second
    "nature and weather",       # sun moon rain tree flower bird sky cloud
    "food and meals",           # rice roti dal milk fruit breakfast lunch
    "home and daily life",      # kitchen bedroom morning bedtime routine
    "feelings and emotions",    # happy sad angry scared excited proud
    "animals",                  # dog cat cow bird elephant fish
    "festivals and celebrations",  # diwali holi birthday new year
    "school and learning",      # pencil book draw write blackboard
    "community and helpers",    # doctor teacher policeman shopkeeper
    "simple world facts",       # sky is blue fire is hot water is wet
    "moral reasoning",          # right wrong fair unfair kind unkind
    "questions and curiosity",  # what where why how wonder explore
    "stories and imagination",  # once upon a time hero adventure dream
]


class CurriculumGenerator:
    """
    Generates curriculum content for Little Deepak on any domain at any level.
    Uses the LLM parent (Together AI) to produce age-appropriate content.
    Caches all results to disk — nothing regenerated unnecessarily.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            base_url=TOGETHER_BASE,
        )
        self._cache      = self._load_cache()
        self._call_count = 0

    # ── Public API ────────────────────────────────────────────────

    def get_content(self, domain: str, level: int) -> List[str]:
        """
        Return content for domain at level.
        Uses cache if available, otherwise generates and caches.
        """
        key = f"{domain}:{level}"
        if key in self._cache:
            return list(self._cache[key])

        print(f'  [Curriculum] Generating "{domain}" level {level}...')
        content = self._generate(domain, level)

        if content:
            self._cache[key] = content
            self._save_cache()
            print(f'  [Curriculum] {len(content)} items. Cached.')
        else:
            print(f'  [Curriculum] No content returned — skipping.')

        return content

    def is_cached(self, domain: str, level: int) -> bool:
        return f"{domain}:{level}" in self._cache

    def clear_cache(self) -> None:
        self._cache = {}
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        print('[Curriculum] Cache cleared.')

    # ── Generation ────────────────────────────────────────────────

    def _generate(self, domain: str, level: int) -> List[str]:
        prompt = self._build_prompt(domain, level)
        self._call_count += 1
        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=600,
                messages=[
                    {'role': 'system', 'content': persona.PARENT_SYSTEM_PROMPT},
                    {'role': 'user',   'content': prompt},
                ],
            )
            raw = resp.choices[0].message.content or ''
            return self._parse(raw, level)
        except Exception as e:
            print(f'  [Curriculum] API error: {e}')
            return []

    def _build_prompt(self, domain: str, level: int) -> str:
        n = COUNTS.get(level, 20)

        if level == 2:   # Words
            return (
                f"Help Little Deepak, a good 5-year-old Indian child, "
                f"learn about '{domain}'.\n\n"
                f"List {n} simple English words about {domain} "
                f"that a 5-year-old would know. "
                f"Positive and age-appropriate only.\n\n"
                f"Reply ONLY with a JSON array. "
                f'Example: ["word1","word2","word3"]\n'
                f"No explanation. No markdown. Just the JSON array."
            )

        if level == 3:   # Phrases
            return (
                f"You are the loving parent of Little Deepak, "
                f"a good 5-year-old Indian child.\n\n"
                f"Write {n} simple sentences about '{domain}' "
                f"that Little Deepak would say. "
                f"Start with 'i'. Positive and age-appropriate.\n"
                f'Example: ["i love the flowers","i see the red bird"]\n\n'
                f"Reply ONLY with a JSON array of sentences. "
                f"No explanation. No markdown. Just the JSON array."
            )

        if level == 4:   # Schemas
            return (
                f"You are the loving parent of Little Deepak, "
                f"a good 5-year-old Indian child.\n\n"
                f"Write {n} cause-and-effect sentences about '{domain}' "
                f"for Little Deepak. "
                f"Pattern: 'when i [action] i [result]'\n"
                f'Example: ["when i count i learn numbers",'
                f'"when i see flowers i feel happy"]\n\n'
                f"Reply ONLY with a JSON array. "
                f"No explanation. No markdown. Just the JSON array."
            )

        return ''

    # ── Parsing ───────────────────────────────────────────────────

    def _parse(self, raw: str, level: int) -> List[str]:
        """Robustly extract a list of strings from the LLM response."""
        raw = raw.strip()
        n   = COUNTS.get(level, 20)

        # 1. Direct JSON parse
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return self._clean(parsed, n)
        except Exception:
            pass

        # 2. Find JSON array anywhere in text
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    return self._clean(parsed, n)
            except Exception:
                pass

        # 3. Strip markdown fences and retry
        for part in raw.split('```'):
            part = part.lstrip('json\n').strip()
            try:
                parsed = json.loads(part)
                if isinstance(parsed, list):
                    return self._clean(parsed, n)
            except Exception:
                continue

        # 4. Line-by-line extraction as last resort
        lines = []
        for line in raw.split('\n'):
            line = re.sub(r'^[\d\.\-\*\s"\']+', '', line).strip().strip('",')
            if 2 <= len(line) <= 80:
                lines.append(line.lower())
        return lines[:n] if lines else []

    @staticmethod
    def _clean(items: list, n: int) -> List[str]:
        return [
            str(i).lower().strip().strip('"\'')
            for i in items
            if i and 2 <= len(str(i).strip()) <= 80
        ][:n]

    # ── Persistence ───────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def stats(self) -> dict:
        return {
            'cached_entries': len(self._cache),
            'api_calls':      self._call_count,
            'cache_file':     CACHE_FILE,
        }