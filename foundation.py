"""
foundation.py — Language Foundation Builder for HGLS.

Builds the word and sentence foundation from corpus data.
No LLM required. Pure data pipeline.

Phase 1 — Words (8000 most common English words, frequency ordered)
  For each word:
    syllabify → teach syllables at level 1
    teach full word at level 2

Phase 2 — Sentences (30,000 Brown corpus sentences, filtered)
  For each sentence:
    filter: 4-15 words, mostly alpha
    teach at level 3

Progress saved in deepak_progress.json under 'foundation' key.
Fully resumable. Run multiple times — picks up where it left off.

Usage:
  python main.py --foundation           # words + sentences
  python main.py --foundation --words   # words only
  python main.py --foundation --sents   # sentences only
"""

import os
import sys
import re
import json
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROGRESS_FILE        = 'deepak_progress.json'
MEMORY_FILE          = 'deepak_memory.json'
N_COMMON_WORDS       = 8000
N_SENTENCES          = 30000
SAVE_EVERY_WORDS     = 500
SAVE_EVERY_SENTENCES = 1000

VOWELS = set('aeiou')


# ── Syllabifier ───────────────────────────────────────────────────

def syllabify(word: str) -> List[str]:
    """
    Split a word into syllable chunks for level 1 learning.

    Algorithm: split at the boundary between vowel groups.
    Between two vowel groups, the consonant cluster is split
    so the first consonant closes the preceding syllable
    and the rest open the next syllable.

    Examples:
      circle      → ['cir', 'cle']
      triangle    → ['trian', 'gle']
      education   → ['edu', 'ca', 'tion']
      photosynthesis → ['pho', 'tos', 'yn', 'the', 'sis']
    """
    word = word.lower().strip()
    if not word or not word.isalpha():
        return []
    if len(word) <= 2:
        return [word]

    # Find vowel group boundaries: (start_idx, end_idx)
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

    # Single vowel group = single syllable
    if len(groups) <= 1:
        return [word]

    chunks   = []
    prev_end = 0

    for idx in range(len(groups) - 1):
        v1_end   = groups[idx][1]
        v2_start = groups[idx + 1][0]
        between  = word[v1_end:v2_start]   # consonants between vowel groups

        if not between:
            continue  # consecutive vowel groups — don't split here

        # Split point in the consonant cluster
        if len(between) == 1:
            split = v1_end          # one consonant → goes with next syllable
        else:
            split = v1_end + 1      # two+ → first closes current, rest open next

        chunk = word[prev_end:split]
        if chunk:
            chunks.append(chunk)
        prev_end = split

    last = word[prev_end:]
    if last:
        chunks.append(last)

    # Merge any single-character fragments
    result = []
    for chunk in chunks:
        if result and len(chunk) == 1:
            result[-1] += chunk
        else:
            result.append(chunk)

    return result if result else [word]


# ── NLTK helpers ──────────────────────────────────────────────────

def ensure_nltk() -> bool:
    """Download required nltk corpora if not already present."""
    try:
        import nltk
    except ImportError:
        print('[Foundation] nltk not installed.')
        print('  Run: pip install nltk')
        return False

    needed = ['brown', 'words']
    for corpus in needed:
        try:
            nltk.data.find(f'corpora/{corpus}')
        except LookupError:
            print(f'  Downloading nltk corpus: {corpus} ...')
            nltk.download(corpus, quiet=True)
    return True


def get_common_words(n: int = N_COMMON_WORDS) -> List[str]:
    """
    Return the N most common English words from the Brown corpus,
    ordered by frequency (most common first).
    Filters: alpha only, length 2-15 characters.
    """
    from nltk.corpus import brown
    from nltk        import FreqDist

    print('[Foundation] Computing word frequencies from Brown corpus...')
    all_words = [w.lower() for w in brown.words() if w.isalpha()]
    freq      = FreqDist(all_words)

    common = [
        word for word, _ in freq.most_common(n * 3)
        if word.isalpha() and 2 <= len(word) <= 15
    ]
    return common[:n]


def get_brown_sentences(n: int = N_SENTENCES) -> List[str]:
    """
    Return N filtered sentences from the Brown corpus.
    Filters: 4-15 words, >75% alpha words, no number-heavy content.
    """
    from nltk.corpus import brown

    print('[Foundation] Loading Brown corpus sentences...')
    sentences = []
    categories = ['news', 'fiction', 'humor', 'romance', 'editorial', 'learned']

    for cat in categories:
        try:
            for sent in brown.sents(categories=[cat]):
                # Join to lowercase string
                text  = ' '.join(sent).lower()
                # Keep only letters and basic punctuation
                text  = re.sub(r'[^a-z\s]', ' ', text)
                text  = re.sub(r'\s+', ' ', text).strip()
                words = text.split()

                if not (4 <= len(words) <= 15):
                    continue
                alpha_ratio = sum(w.isalpha() for w in words) / len(words)
                if alpha_ratio < 0.75:
                    continue

                sentences.append(text)
                if len(sentences) >= n:
                    break
        except Exception:
            continue
        if len(sentences) >= n:
            break

    return sentences[:n]


# ── Foundation Builder ────────────────────────────────────────────

class FoundationBuilder:
    """
    Builds the language foundation bottom-up.
    Characters (already done) → syllables → words → sentences.
    """

    def __init__(self, system, progress_file: str = PROGRESS_FILE):
        self.system         = system
        self.progress_file  = progress_file
        self._words_done    = 0
        self._sents_done    = 0
        self._session_start = time.time()
        self._load_progress()

    # ── Progress ──────────────────────────────────────────────────

    def _load_progress(self):
        if not os.path.exists(self.progress_file):
            return
        try:
            with open(self.progress_file, encoding='utf-8') as f:
                data = json.load(f)
            found = data.get('foundation', {})
            self._words_done = found.get('words_done', 0)
            self._sents_done = found.get('sents_done', 0)
            if self._words_done > 0 or self._sents_done > 0:
                print(
                    f'[Foundation] Resuming — '
                    f'words={self._words_done:,} '
                    f'sentences={self._sents_done:,}'
                )
        except Exception:
            pass

    def _save_progress(self):
        # Load existing data to preserve other keys (mastery records etc.)
        data = {}
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                pass

        data['foundation'] = {
            'words_done': self._words_done,
            'sents_done': self._sents_done,
            'timestamp':  time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        tmp = self.progress_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if os.path.exists(self.progress_file):
            backup = self.progress_file + '.bak'
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(self.progress_file, backup)
        os.rename(tmp, self.progress_file)

    def _safe_save(self):
        try:
            self.system.save(MEMORY_FILE)
            self._save_progress()
        except Exception as e:
            print(f'  [Save] Failed: {e}')

    # ── Teaching primitives ───────────────────────────────────────

    def teach_word(self, word: str) -> bool:
        """
        Teach one word bottom-up:
          syllables at level 1 → full word at level 2.
        Returns True if word was successfully taught.
        """
        word = word.lower().strip()
        if not word or not word.isalpha() or len(word) < 2:
            return False

        # Level 1: syllables
        for syl in syllabify(word):
            if len(syl) >= 2:
                try:
                    self.system.ingest_text(syl, level=1)
                except Exception:
                    pass

        # Level 2: full word
        try:
            self.system.ingest_text(word, level=2)
            return True
        except Exception:
            return False

    def teach_sentence(self, sentence: str) -> bool:
        """Teach one sentence at level 3."""
        try:
            self.system.ingest_text(sentence.strip(), level=3)
            return True
        except Exception:
            return False

    # ── Phase 1: Words ────────────────────────────────────────────

    def build_words(self, word_list: List[str] = None):
        if word_list is None:
            word_list = get_common_words(N_COMMON_WORDS)

        total     = len(word_list)
        remaining = word_list[self._words_done:]

        print(f'\n[Foundation] Teaching {len(remaining):,} words '
              f'(skipping first {self._words_done:,} already done)...\n')

        for word in remaining:
            self.teach_word(word)
            self._words_done += 1

            if self._words_done % 100 == 0:
                elapsed  = time.time() - self._session_start
                rate     = self._words_done / max(elapsed, 1)
                print(
                    f'  words={self._words_done:,}/{total:,} | '
                    f'lib={len(self.system.library):,} | '
                    f'rate={rate:.1f}/s | '
                    f'last="{word}"'
                )

            if self._words_done % SAVE_EVERY_WORDS == 0:
                self._safe_save()

        self._safe_save()
        print(f'\n[Foundation] Words complete. '
              f'Library: {len(self.system.library):,} structures.')

    # ── Phase 2: Sentences ────────────────────────────────────────

    def build_sentences(self, sentence_list: List[str] = None):
        if sentence_list is None:
            sentence_list = get_brown_sentences(N_SENTENCES)

        total     = len(sentence_list)
        remaining = sentence_list[self._sents_done:]

        print(f'\n[Foundation] Teaching {len(remaining):,} sentences '
              f'(skipping first {self._sents_done:,} already done)...\n')

        for sentence in remaining:
            self.teach_sentence(sentence)
            self._sents_done += 1

            if self._sents_done % 500 == 0:
                elapsed = time.time() - self._session_start
                rate    = self._sents_done / max(elapsed, 1)
                print(
                    f'  sentences={self._sents_done:,}/{total:,} | '
                    f'lib={len(self.system.library):,} | '
                    f'graph={len(self.system.graph):,} | '
                    f'rate={rate:.1f}/s'
                )

            if self._sents_done % SAVE_EVERY_SENTENCES == 0:
                self._safe_save()

        self._safe_save()
        print(f'\n[Foundation] Sentences complete. '
              f'Library: {len(self.system.library):,} structures.')

    # ── Run ───────────────────────────────────────────────────────

    def run(self, phase: str = 'all'):
        """
        Run foundation pipeline.
        phase: 'words' | 'sentences' | 'all'
        """
        print(f'\n{"=" * 60}')
        print('HGLS FOUNDATION BUILDER')
        print(f'  Phase           : {phase}')
        print(f'  Target words    : {N_COMMON_WORDS:,}')
        print(f'  Target sentences: {N_SENTENCES:,}')
        print(f'  Library at start: {len(self.system.library):,}')
        print(f'{"=" * 60}')

        if phase in ('words', 'all'):
            self.build_words()

        if phase in ('sentences', 'all'):
            self.build_sentences()

        elapsed = time.time() - self._session_start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        print(f'\n[Foundation] Complete in {h}h {m}m.')
        print(f'  Library: {len(self.system.library):,} structures')
        print(f'  Graph  : {len(self.system.graph):,} edges')
        print(f'\nNext step: python main.py --train')