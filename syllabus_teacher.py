"""
syllabus_teacher.py — 6-Stage Curriculum Teaching.

For each topic in the syllabus, teaches in progressive stages
matching how a child naturally learns:

  Stage 1: Topic vocabulary    → syllables (L1) + words (L2)
  Stage 2: Simple facts        → declarative sentences (L3)
  Stage 3: Context sentences   → vocabulary in everyday use (L3)
  Stage 4: Associations        → connected fact pairs (L4)
  Stage 5: Logic               → because/so causal sentences (L4)
  Stage 6: Reasoning chains    → i know → that means → so (L5)

Stages taught progressively per attempt:
  Attempts  1-5:  stages 1-3 only
  Attempts  6-10: stages 1-4
  Attempts 11-15: stages 1-5
  Attempts 16+:   all 6 stages

Evaluation begins after first teaching. No max attempts.
The library gets richer every attempt. The system learns at its own pace.

Tinkering runs every 200 evaluation attempts.
Co-occurrence scan runs every 100 evaluation attempts.
"""

import os
import sys
import re
import json
import time
import urllib.request
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOGETHER_API_URL = 'https://api.together.xyz/v1/chat/completions'
MODEL            = 'openai/gpt-oss-20b'

MEMORY_FILE   = 'deepak_memory.json'
SAVE_EVERY    = 20   # save frequently so corrections persist
TINKER_EVERY  = 200
SCAN_EVERY    = 100
MAX_RETRIES   = 3
RETRY_DELAY   = 2.0

VOWELS = set('aeiou')

_STOP = {
    'the','a','an','is','are','was','were','be','been','have','has',
    'do','does','did','will','to','of','in','on','at','by','for',
    'with','and','or','but','not','so','if','that','this','it',
    'he','she','they','we','i','you','what','how','why','when',
    'who','which','can','just','also','very','your','our',
}


# ── Syllabifier (same as foundation.py) ──────────────────────────

def syllabify(word: str) -> List[str]:
    word = word.lower().strip()
    if not word or not word.isalpha():
        return []
    if len(word) <= 2:
        return [word]

    groups = []
    i = 0
    while i < len(word):
        if word[i] in VOWELS:
            start = i
            while i < len(word) and word[i] in VOWELS:
                i += 1
            groups.append((start, i))
        else:
            i += 1

    if len(groups) <= 1:
        return [word]

    chunks   = []
    prev_end = 0
    for idx in range(len(groups) - 1):
        v1_end   = groups[idx][1]
        v2_start = groups[idx+1][0]
        between  = word[v1_end:v2_start]
        if not between:
            continue
        split = v1_end if len(between) == 1 else v1_end + 1
        chunk = word[prev_end:split]
        if chunk:
            chunks.append(chunk)
        prev_end = split
    last = word[prev_end:]
    if last:
        chunks.append(last)

    result = []
    for chunk in chunks:
        if result and len(chunk) == 1:
            result[-1] += chunk
        else:
            result.append(chunk)
    return result if result else [word]


# ── LLM ──────────────────────────────────────────────────────────

def llm_call(
    prompt:      str,
    api_key:     str,
    max_tokens:  int   = 1024,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call GPT-OSS 20B. Returns content or reasoning text. Never crashes."""
    for attempt in range(MAX_RETRIES):
        try:
            payload = json.dumps({
                'model':       MODEL,
                'messages':    [{'role': 'user', 'content': prompt}],
                'max_tokens':  max_tokens,
                'temperature': temperature,
            }).encode()
            headers = {
                'Content-Type':  'application/json',
                'Authorization': f'Bearer {api_key}',
                'User-Agent':    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }
            req = urllib.request.Request(
                TOGETHER_API_URL,
                data=payload, headers=headers, method='POST'
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            msg = data['choices'][0]['message']
            return (msg.get('content') or msg.get('reasoning') or '').strip()

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
    return None


def test_api_key(api_key: str) -> bool:
    print('[Main] Testing API key...')
    r = llm_call('Say OK', api_key, max_tokens=512)
    if r:
        print('[Main] API key works.')
        return True
    print('[Main] API key FAILED.')
    return False


# ── Parsers ───────────────────────────────────────────────────────

def parse_word_list(raw: str) -> List[str]:
    """Extract single vocabulary words from LLM output."""
    words = []
    seen  = set()
    for line in raw.split('\n'):
        line = re.sub(r'^[\d\.\-\*\•\s]+', '', line).strip().lower()
        line = line.strip('"\'.,;:')
        # Single alpha word, reasonable length
        if line.isalpha() and 2 <= len(line) <= 20 and line not in seen:
            words.append(line)
            seen.add(line)
    return words[:15]


def parse_sentences(raw: str, min_w: int = 3, max_w: int = 20) -> List[str]:
    """Extract simple sentences from LLM output."""
    sents = []
    seen  = set()
    for line in raw.split('\n'):
        line = re.sub(r'^[\d\.\-\*\•\s]+', '', line).strip().lower()
        line = line.strip('"\'')
        line = re.sub(r'[^a-z\s\.\,]', ' ', line)
        line = re.sub(r'\s+', ' ', line).strip().rstrip('.')
        words = line.split()
        if not (min_w <= len(words) <= max_w):
            continue
        alpha = sum(w.isalpha() for w in words) / len(words)
        if alpha < 0.7:
            continue
        if line not in seen:
            sents.append(line)
            seen.add(line)
    return sents[:10]


def parse_chain(raw: str) -> Optional[str]:
    """Extract i know → that means → so reasoning chain."""
    lines = []
    for line in raw.split('\n'):
        line = line.strip()
        low  = line.lower()
        if any(low.startswith(p) for p in ('i know', 'that means', 'so ')):
            # Remove any template placeholder brackets
            line = re.sub(r'\[.*?\]', '', line).strip()
            if len(line.split()) >= 3:
                lines.append(line)
    return '\n'.join(lines) if len(lines) >= 2 else None


def parse_yes_no(raw: str) -> bool:
    """Extract YES/NO from evaluation response."""
    # Look at last 10 words
    words = raw.upper().split()[-10:]
    if 'YES' in words:
        return True
    if 'NO' in words:
        return False
    return 'YES' in raw.upper()


def extract_topics(text: str) -> List[str]:
    words = re.findall(r'[a-z]+', text.lower())
    return [w for w in words if w not in _STOP and len(w) > 2]


# ── Prompts ───────────────────────────────────────────────────────

def prompt_vocab(topic: str, subject: str, grade: int) -> str:
    return f"""List the 10 most important vocabulary words for this topic.
Subject: {subject}, Grade: {grade}, Topic: {topic}

Output ONLY the words, one word per line. No numbers. No explanations.

Example output:
circle
square
triangle
corner
side
round
flat
equal
vertex
angle"""


def prompt_facts(topic: str, subject: str, grade: int) -> str:
    return f"""Write 6 simple facts about this topic for Grade {grade} students.
Subject: {subject}, Topic: {topic}

Rules:
- Each fact is one sentence
- Under 8 words
- Simple vocabulary only
- One fact per line
- No numbers or bullet points

Example output:
a circle is round
a square has four sides
a triangle has three corners
circles have no corners
squares have equal sides
triangles have three angles"""


def prompt_context(topic: str, subject: str, grade: int) -> str:
    return f"""Write 5 sentences showing {topic} in everyday life.
Subject: {subject}, Grade: {grade}

Rules:
- Under 10 words each
- Use real everyday examples
- Simple vocabulary
- One sentence per line
- No numbers

Example output:
the wheel is round like a circle
a window is shaped like a square
a pizza slice looks like a triangle
the sun looks like a circle in the sky
a book cover is a rectangle shape"""


def prompt_associations(topic: str, subject: str, grade: int) -> str:
    return f"""Write 4 pairs of connected facts about {topic} for Grade {grade}.
Subject: {subject}

Rules:
- Join two facts with "and" or "but"
- Under 15 words total per pair
- One pair per line
- No numbers

Example output:
a circle is round and has no corners
a square has four sides and four corners
a triangle has three sides but circles have none
squares have equal sides and right angle corners"""


def prompt_logic(topic: str, subject: str, grade: int) -> str:
    return f"""Write 4 cause-and-effect sentences about {topic} for Grade {grade}.
Subject: {subject}

Rules:
- Use "because" or "so"
- Under 12 words each
- Simple vocabulary
- One sentence per line
- No numbers

Example output:
a circle is round so it can roll
because a square has corners it cannot roll
a triangle has three sides so it has three corners
because a circle has no corners it rolls easily"""


def prompt_reasoning(topic: str, subject: str, grade: int) -> str:
    return f"""Write one reasoning chain about {topic} for Grade {grade}.
Subject: {subject}

Use EXACTLY this format, nothing else:

i know [one fact about {topic}].
that means [what it implies].
so [the conclusion].

Example:
i know a circle is round.
that means it has no flat sides.
so a circle can roll easily."""


def prompt_question(topic: str, subject: str, grade: int, attempt: int) -> str:
    q_starters = [
        "What is",
        "Name one",
        "How many",
        "Which shape",
        "Where do we see",
        "Why do we use",
        "What does",
        "How does",
    ]
    starter = q_starters[attempt % len(q_starters)]
    return f"""Complete this question about {topic} for Grade {grade} {subject}.

Start with: {starter}
Topic: {topic}

Write the complete question in under 10 words.
Output ONLY the question itself. No thinking. No explanation. No preamble.

Example of correct output:
What is a triangle?"""


def prompt_eval(question: str, answer: str, topic: str) -> str:
    return f"""Topic: {topic}
Question: {question}
Student answer: {answer}

Does the answer show correct understanding of the topic?
Write ONLY YES or NO."""


def prompt_correction(question: str, answer: str, chain: str, topic: str) -> str:
    return f"""Topic: {topic}
Correct information: {chain[:200] if chain else topic}
Student gave wrong answer: {answer[:80]}
Question was: {question}

Write ONE corrective sentence starting with exactly "i know".
Under 15 words. Output ONLY that sentence.

Example: i know a circle is round and has no corners."""


# ── Stage teacher ─────────────────────────────────────────────────

class SyllabusTeacher:

    def __init__(self, system, api_key: str):
        from mastery_tracker import MasteryTracker
        self.system   = system
        self.api_key  = api_key
        self.tracker  = MasteryTracker()

        self._total_attempts = 0
        self._total_correct  = 0
        self._session_start  = time.time()

        # Cache: topic → stage content (avoid regenerating every attempt)
        self._stage_cache: dict = {}

    # ── Main loop ──────────────────────────────────────────────────

    def run(self):
        stats = self.tracker.stats()
        print(f'\n{"=" * 60}')
        print('HGLS SYLLABUS TRAINING')
        print(f'  Total topics    : {stats["total_topics"]}')
        print(f'  Already mastered: {stats["mastered"]}')
        print(f'  Current pass    : {stats["pass_number"]}')
        print(f'  Starting at     : {stats["current_topic"]}')
        print(f'{"=" * 60}\n')

        while True:
            rec   = self.tracker.current_topic()
            topic = self.tracker.current_topic_object()
            if rec is None or topic is None:
                print('\n[Teacher] All topics complete for this pass.')
                break
            self._teach_topic(rec, topic)

    # ── Topic teaching ─────────────────────────────────────────────

    def _teach_topic(self, rec, topic):
        subject  = topic.subject
        grade    = topic.grade
        name     = topic.name
        attempts = rec.attempts
        tid      = rec.topic_id

        print(f'\n  → [{subject} G{grade}] {name}  '
              f'(attempt {attempts+1}, rate={rec.correct_rate:.0%})')

        # Determine which stages to teach based on attempt count
        max_stage = (
            3 if attempts < 5  else
            4 if attempts < 10 else
            5 if attempts < 15 else
            6
        )

        # ── Stage 1: Vocabulary ───────────────────────────────────
        # Always teach vocabulary
        self._teach_stage_1(name, subject, grade, tid)

        # ── Stages 2-N: Progressive content ──────────────────────
        chain = None  # keep chain for correction use

        if max_stage >= 2:
            self._teach_stage_2(name, subject, grade, tid)

        if max_stage >= 3:
            self._teach_stage_3(name, subject, grade, tid)

        if max_stage >= 4:
            self._teach_stage_4(name, subject, grade, tid)

        if max_stage >= 5:
            self._teach_stage_5(name, subject, grade, tid)

        if max_stage >= 6:
            chain = self._teach_stage_6(name, subject, grade, tid)

        # Save after teaching so corrections persist
        self._safe_save()

        # ── Evaluation ────────────────────────────────────────────
        q_raw = llm_call(
            prompt_question(name, subject, grade, attempts),
            self.api_key, max_tokens=256, temperature=0.5
        )

        # Extract clean question
        question = self._clean_question(q_raw)
        if not question:
            print(f'    [Warning] Could not generate question. Retrying in 5s...')
            time.sleep(5)
            return

        print(f'    Q: {question}')

        # Get system response
        response = self.system.respond(question)
        print(f'    A: {response[:120]}')

        # Evaluate
        correct = self._evaluate(question, response, chain or name)
        print(f'    {"✓ Correct" if correct else "✗ Wrong"}')

        # Record attempt
        mastered = self.tracker.record_attempt(correct)
        self._total_attempts += 1
        if correct:
            self._total_correct += 1

        # Reinforce or penalise contributing structures
        topic_words = extract_topics(name + ' ' + subject + ' ' + (question or ''))
        if correct:
            self.system.reinforce_correct_response(response, topic_words)
        else:
            self.system.penalise_wrong_response(response, topic_words)

        # Correction if wrong
        if not correct:
            corr_raw = llm_call(
                prompt_correction(question, response, chain or '', name),
                self.api_key, max_tokens=256
            )
            correction = self._clean_correction(corr_raw)
            if correction:
                print(f'    Correction: {correction}')
                topic_words = extract_topics(name + ' ' + question)
                self._ingest(correction, 4, topic_words, name)

        # ── Periodic maintenance ──────────────────────────────────
        if self._total_attempts % SAVE_EVERY == 0:
            self._safe_save()
            self._print_summary()

        if self._total_attempts % SCAN_EVERY == 0 and self._total_attempts > 0:
            self.system.scan_cooccurrence(
                sentences_processed=self._total_attempts * 5
            )

        if self._total_attempts % TINKER_EVERY == 0 and self._total_attempts > 0:
            result = self.system.run_tinkering()
            if result['accepted'] > 0:
                print(f'  [Tinkering] +{result["accepted"]} novel connections')

        # ── Mastery ───────────────────────────────────────────────
        if mastered:
            print(
                f'\n  ★ MASTERED: [{subject} G{grade}] {name} '
                f'({rec.attempts} attempts, {rec.correct_rate:.0%})\n'
            )
            # Clear cache for this topic
            self._stage_cache.pop(tid, None)
            self.tracker.advance()

    # ── Stage implementations ──────────────────────────────────────

    def _teach_stage_1(self, name, subject, grade, tid):
        """Stage 1: Topic vocabulary — syllables at L1, words at L2."""
        cache_key = f'{tid}_vocab'
        if cache_key not in self._stage_cache:
            raw = llm_call(
                prompt_vocab(name, subject, grade),
                self.api_key, max_tokens=512
            )
            words = parse_word_list(raw) if raw else []
            # Add topic name words as well
            for w in name.lower().split():
                if w.isalpha() and len(w) > 2 and w not in words:
                    words.append(w)
            self._stage_cache[cache_key] = words

        words = self._stage_cache[cache_key]
        for word in words:
            # Syllables at level 1
            for syl in syllabify(word):
                if len(syl) >= 2:
                    try:
                        self.system.ingest_text(syl, level=1)
                    except Exception:
                        pass
            # Word at level 2
            try:
                self.system.ingest_text(word, level=2)
            except Exception:
                pass

        if words:
            print(f'    Vocab: {", ".join(words[:6])}...' if len(words) > 6
                  else f'    Vocab: {", ".join(words)}')

    def _teach_stage_2(self, name, subject, grade, tid):
        """Stage 2: Simple facts at L3."""
        cache_key = f'{tid}_facts'
        if cache_key not in self._stage_cache:
            raw   = llm_call(prompt_facts(name, subject, grade), self.api_key, max_tokens=512)
            facts = parse_sentences(raw, min_w=2, max_w=10) if raw else []
            self._stage_cache[cache_key] = facts

        topic_words = extract_topics(name + ' ' + subject)
        for fact in self._stage_cache[cache_key]:
            self._ingest(fact, 3, topic_words, name)

    def _teach_stage_3(self, name, subject, grade, tid):
        """Stage 3: Context sentences at L3."""
        cache_key = f'{tid}_context'
        if cache_key not in self._stage_cache:
            raw   = llm_call(prompt_context(name, subject, grade), self.api_key, max_tokens=512)
            sents = parse_sentences(raw, min_w=3, max_w=12) if raw else []
            self._stage_cache[cache_key] = sents

        topic_words = extract_topics(name + ' ' + subject)
        for sent in self._stage_cache[cache_key]:
            self._ingest(sent, 3, topic_words, name)

    def _teach_stage_4(self, name, subject, grade, tid):
        """Stage 4: Associations at L4."""
        cache_key = f'{tid}_assoc'
        if cache_key not in self._stage_cache:
            raw   = llm_call(prompt_associations(name, subject, grade), self.api_key, max_tokens=512)
            pairs = parse_sentences(raw, min_w=4, max_w=18) if raw else []
            self._stage_cache[cache_key] = pairs

        topic_words = extract_topics(name + ' ' + subject)
        for pair in self._stage_cache[cache_key]:
            self._ingest(pair, 4, topic_words, name)

    def _teach_stage_5(self, name, subject, grade, tid):
        """Stage 5: Logic/causal at L4."""
        cache_key = f'{tid}_logic'
        if cache_key not in self._stage_cache:
            raw   = llm_call(prompt_logic(name, subject, grade), self.api_key, max_tokens=512)
            sents = parse_sentences(raw, min_w=4, max_w=15) if raw else []
            self._stage_cache[cache_key] = sents

        topic_words = extract_topics(name + ' ' + subject)
        for sent in self._stage_cache[cache_key]:
            self._ingest(sent, 4, topic_words, name)

    def _teach_stage_6(self, name, subject, grade, tid) -> Optional[str]:
        """Stage 6: Reasoning chain at L5. Returns chain or None."""
        cache_key = f'{tid}_chain'
        if cache_key not in self._stage_cache:
            raw   = llm_call(prompt_reasoning(name, subject, grade), self.api_key, max_tokens=512)
            chain = parse_chain(raw) if raw else None
            self._stage_cache[cache_key] = chain

        chain = self._stage_cache[cache_key]
        if chain:
            topic_words = extract_topics(name + ' ' + subject)
            for line in chain.split('\n'):
                line = line.strip()
                if len(line.split()) >= 3:
                    self._ingest(line, 5, topic_words, name)
            self._ingest(chain, 5, topic_words, name)

        return chain

    # ── Helpers ────────────────────────────────────────────────────

    def _ingest(self, text: str, level: int, topic_words: List[str], topic: str):
        """Ingest text as correction with topic tags."""
        text = text.strip()
        if len(text.split()) < 2:
            return
        try:
            self.system.ingest_text(
                text, level=level,
                is_correction=True,
                topic_words=topic_words,
            )
        except Exception:
            pass

    def _evaluate(self, question: str, response: str, context: str) -> bool:
        if not response or 'not sure' in response.lower():
            return False
        raw = llm_call(
            prompt_eval(question, response, context),
            self.api_key, max_tokens=256, temperature=0.1
        )
        return parse_yes_no(raw) if raw else False

    def _clean_question(self, raw: str) -> Optional[str]:
        """
        Extract clean question from harmony model output.
        Very aggressive — takes only the shortest clean question found.
        Rejects all reasoning preamble.
        """
        if not raw:
            return None

        # Patterns that indicate reasoning leakage
        bad_starts = [
            'so ', 'but ', 'also ', 'maybe ', 'perhaps ', 'let me',
            'we need', 'the user', 'i think', 'check:', 'count:',
            'output', 'example', 'based on', 'this is', 'note:',
            'alternatively', 'however', 'instead', 'actually',
        ]

        # Find all question-mark sentences
        raw_clean = re.sub(r'\(\d+\)', '', raw)  # remove (1)(2)(3)
        sentences = re.findall(r'[A-Za-z][^.!?]{4,80}\?', raw_clean)

        best = None
        for sent in sentences:
            sent = sent.strip().strip('"''')
            low  = sent.lower()

            # Skip if starts with reasoning phrase
            if any(low.startswith(b) for b in bad_starts):
                continue

            # Skip if contains meta-commentary
            if any(x in low for x in ['word', 'question is', 'ask:', 'must be', 'should be']):
                continue

            words = sent.split()
            if not (3 <= len(words) <= 12):
                continue

            # Prefer shorter questions (less likely to contain reasoning)
            if best is None or len(words) < len(best.split()):
                best = sent

        return best


    def _clean_correction(self, raw: str) -> Optional[str]:
        """Extract correction sentence starting with 'i know'."""
        if not raw:
            return None
        lower = raw.lower()
        idx   = lower.rfind('i know')
        if idx != -1:
            sent = raw[idx:].strip()
            end  = sent.find('.')
            if end != -1:
                sent = sent[:end + 1]
            sent = re.sub(r'\[.*?\]', '', sent).strip()
            if len(sent.split()) >= 3:
                return sent
        return None

    def _safe_save(self):
        try:
            self.system.save(MEMORY_FILE)
        except Exception as e:
            print(f'  [Save] Failed: {e}')

    def _print_summary(self):
        elapsed  = time.time() - self._session_start
        rate     = self._total_attempts / max(elapsed, 1)
        acc      = self._total_correct  / max(self._total_attempts, 1)
        stats    = self.tracker.stats()
        print(
            f'\n  [{time.strftime("%H:%M:%S")}] '
            f'pass={stats["pass_number"]} | '
            f'mastered={stats["mastered"]}/{stats["total_topics"]} | '
            f'attempts={self._total_attempts:,} | '
            f'acc={acc:.0%} | '
            f'lib={len(self.system.library):,} | '
            f'graph={len(self.system.graph):,} | '
            f'rate={rate:.2f}/s\n'
        )


# ── Entry point ───────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='HGLS Syllabus Training')
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--memory',  default=MEMORY_FILE)
    parser.add_argument('--reset',   action='store_true')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('TOGETHER_API_KEY', '')
    if not api_key:
        print('[Error] Set TOGETHER_API_KEY or pass --api-key')
        sys.exit(1)

    if not test_api_key(api_key):
        sys.exit(1)

    from hgls.system import HGLSystem
    system = HGLSystem(use_llm=False)

    if os.path.exists(args.memory) and not args.reset:
        system.load(args.memory)
    elif args.reset:
        for f in ['deepak_progress.json', 'deepak_progress.json.bak']:
            if os.path.exists(f):
                os.remove(f)
        print('[Main] Progress reset. Memory kept (foundation preserved).')

    SyllabusTeacher(system, api_key=api_key).run()

    try:
        system.save(args.memory)
    except Exception as e:
        print(f'[Main] Final save failed: {e}')


if __name__ == '__main__':
    main()