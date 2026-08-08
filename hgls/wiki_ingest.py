"""
wiki_ingest.py — Wikipedia Ingestion Script for HGLS.

Streams Wikipedia English, processes every sentence bottom-up:
  Level 1 — syllable roots of new words
  Level 2 — full words (only if not already in library)
  Level 3 — the full sentence (phrase/schema)

This order matters. The system cannot build a phrase from words
it hasn't learned, and cannot build words from syllables it hasn't
learned. Every new word is grounded before the sentence that uses it.

Resume: tracks last processed article index in a checkpoint file.
If interrupted, restart picks up from where it left off.

Usage:
  python wiki_ingest.py                        # run until done
  python wiki_ingest.py --save-every 50000     # save every N sentences
  python wiki_ingest.py --max-sentences 2000000 # stop after N sentences
  python wiki_ingest.py --reset-checkpoint     # start from beginning
"""

import re
import os
import sys
import json
import time
import argparse
import signal
from typing import Iterator, List, Set

# ── Syllabifier ────────────────────────────────────────────────────
# Simple rule-based syllable splitter.
# Good enough for seeding level 1 — not meant to be linguistically perfect.

VOWELS = set('aeiou')

def syllabify(word: str) -> List[str]:
    """
    Split a word into approximate syllable chunks for level 1 learning.
    Returns 2-4 character chunks that preserve the sound structure.

    Rules (simple but effective):
      - Split on vowel-consonant-vowel boundaries
      - Minimum chunk size of 2 characters
      - Never split digraphs (th, sh, ch, ph, wh)
    """
    word = word.lower()
    if len(word) <= 3:
        return [word]

    DIGRAPHS = {'th', 'sh', 'ch', 'ph', 'wh', 'ck', 'ng', 'qu'}

    # Find split points
    splits = [0]
    i = 1
    while i < len(word) - 1:
        # Don't split digraphs
        if word[i:i+2] in DIGRAPHS:
            i += 2
            continue
        # Split after vowel followed by consonant followed by vowel
        if (word[i-1] in VOWELS
                and word[i] not in VOWELS
                and word[i+1] in VOWELS
                and i - splits[-1] >= 2):
            splits.append(i)
        i += 1
    splits.append(len(word))

    syllables = []
    for j in range(len(splits) - 1):
        chunk = word[splits[j]:splits[j+1]]
        if chunk:
            syllables.append(chunk)

    # Merge very short chunks (single chars) with neighbours
    merged = []
    for chunk in syllables:
        if merged and len(chunk) == 1:
            merged[-1] += chunk
        else:
            merged.append(chunk)

    return merged if merged else [word]


# ── Sentence extraction ────────────────────────────────────────────

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
CLEAN        = re.compile(r'[^a-z\s]')
MULTI_SPACE  = re.compile(r'\s+')
MIN_WORDS    = 4
MAX_WORDS    = 30


def extract_sentences(text: str) -> List[str]:
    """
    Extract clean, learnable sentences from a Wikipedia article text.

    Filters:
      - Too short (< 4 words) or too long (> 30 words)
      - Mostly numbers or symbols (tables, coordinates, dates)
      - References and citation artifacts
    """
    sentences = []
    for raw in SENTENCE_END.split(text):
        raw = raw.strip()
        if not raw:
            continue

        # Lowercase and strip non-alphabet chars
        clean = CLEAN.sub(' ', raw.lower())
        clean = MULTI_SPACE.sub(' ', clean).strip()

        words = clean.split()
        if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
            continue

        # Skip if more than 30% of original chars were non-alpha
        # (indicates tables, formulas, citation noise)
        alpha_ratio = sum(c.isalpha() for c in raw) / max(len(raw), 1)
        if alpha_ratio < 0.6:
            continue

        sentences.append(clean)

    return sentences


# ── Wikipedia stream ───────────────────────────────────────────────

def stream_wikipedia(checkpoint: int = 0) -> Iterator[tuple]:
    """
    Stream Wikipedia English articles.
    Yields (article_index, title, list_of_sentences).
    Skips articles before checkpoint index for resume.
    """
    from datasets import load_dataset

    print('[Wikipedia] Loading stream (this may take a moment first time)...')
    dataset = load_dataset(
        'wikimedia/wikipedia',
        '20231101.en',
        split='train',
        streaming=True,
        trust_remote_code=True,
    )

    for idx, article in enumerate(dataset):
        if idx < checkpoint:
            if idx % 10000 == 0 and idx > 0:
                print(f'  [Resume] Skipping to checkpoint... {idx}/{checkpoint}')
            continue

        title = article.get('title', '')
        text  = article.get('text', '')
        if not text:
            continue

        sentences = extract_sentences(text)
        if sentences:
            yield idx, title, sentences


# ── Bottom-up ingestion pipeline ──────────────────────────────────

class WikiIngestor:
    """
    Ingests Wikipedia into HGLS bottom-up:
      For each new word seen → learn syllables (L1) → learn word (L2)
      Then learn the sentence (L3).
    """

    def __init__(self, system, save_path: str = 'deepak_memory.json'):
        self.system        = system
        self.save_path     = save_path
        self.checkpoint_path = save_path.replace('.json', '_wiki_checkpoint.json')

        # Track which words are already in the library at level 2
        # Rebuilt from library on resume — cheap dictionary lookup
        self._known_words: Set[str] = set()
        self._rebuild_known_words()

        # Stats
        self._articles_done   = 0
        self._sentences_done  = 0
        self._words_learned   = 0
        self._session_start   = time.time()
        self._interrupted     = False

        # Graceful interrupt
        try:
            signal.signal(signal.SIGINT, self._handle_interrupt)
        except (OSError, ValueError):
            pass  # signal handling not available on all platforms

    def _handle_interrupt(self, sig, frame):
        print('\n\n[WikiIngestor] Interrupt received. Saving and stopping cleanly...')
        self._interrupted = True

    def _rebuild_known_words(self):
        """Scan library level 2 to find already-learned words."""
        for struct in self.system.library.get_at_level(2, kind='success'):
            word = struct.generate(self.system.library).strip()
            if word:
                self._known_words.add(word)
        print(f'[WikiIngestor] Known words from library: {len(self._known_words)}')

    # ── Checkpoint ─────────────────────────────────────────────────

    def load_checkpoint(self) -> int:
        """Return last processed article index, 0 if none."""
        if not os.path.exists(self.checkpoint_path):
            return 0
        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)
            idx = data.get('last_article_index', 0)
            self._articles_done  = data.get('articles_done', 0)
            self._sentences_done = data.get('sentences_done', 0)
            self._words_learned  = data.get('words_learned', 0)
            print(f'[WikiIngestor] Resuming from article {idx} '
                  f'({self._sentences_done} sentences done)')
            return idx
        except Exception:
            return 0

    def save_checkpoint(self, last_article_index: int):
        """Save progress checkpoint."""
        data = {
            'last_article_index': last_article_index,
            'articles_done':      self._articles_done,
            'sentences_done':     self._sentences_done,
            'words_learned':      self._words_learned,
            'timestamp':          time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        tmp = self.checkpoint_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.checkpoint_path)

    # ── Per-word bottom-up learning ───────────────────────────────

    def _learn_word_bottomup(self, word: str):
        """
        For a word not yet in the library:
          1. Learn its syllables at level 1
          2. Learn the full word at level 2
        """
        if word in self._known_words or len(word) < 2:
            return

        # Level 1 — syllable roots
        syllables = syllabify(word)
        for syl in syllables:
            if len(syl) >= 2:
                self.system.ingest_text(syl, level=1, is_correction=False)

        # Level 2 — full word
        self.system.ingest_text(word, level=2, is_correction=False)

        self._known_words.add(word)
        self._words_learned += 1

    # ── Main run loop ─────────────────────────────────────────────

    def run(
        self,
        max_sentences: int  = 0,
        save_every: int     = 50000,
        print_every: int    = 1000,
    ):
        """
        Stream Wikipedia and ingest bottom-up.
        max_sentences=0 means run until all of Wikipedia is processed.
        """
        checkpoint = self.load_checkpoint()

        print(f'\n{"=" * 60}')
        print(f'WIKIPEDIA INGESTION — HGLS v0.5')
        print(f'  Max sentences : {"unlimited" if max_sentences == 0 else max_sentences}')
        print(f'  Save every    : {save_every} sentences')
        print(f'  Memory file   : {self.save_path}')
        print(f'{"=" * 60}\n')

        last_article_idx = checkpoint

        for article_idx, title, sentences in stream_wikipedia(checkpoint):
            if self._interrupted:
                break
            if max_sentences > 0 and self._sentences_done >= max_sentences:
                break

            last_article_idx = article_idx

            for sentence in sentences:
                if self._interrupted:
                    break
                if max_sentences > 0 and self._sentences_done >= max_sentences:
                    break

                # Step 1 — bottom-up: learn any new words in this sentence
                words = sentence.split()
                for word in words:
                    if word not in self._known_words:
                        self._learn_word_bottomup(word)

                # Step 2 — learn the sentence at level 3
                self.system.ingest_text(sentence, level=3, is_correction=False)
                self._sentences_done += 1

            self._articles_done += 1

            # ── Periodic save ─────────────────────────────────────
            if self._sentences_done % save_every < len(sentences):
                self.system.save(self.save_path)
                self.save_checkpoint(last_article_idx)
                self._print_progress(title)

            # ── Periodic print ────────────────────────────────────
            elif self._sentences_done % print_every < len(sentences):
                self._print_progress(title)

        # Final save
        self.system.save(self.save_path)
        self.save_checkpoint(last_article_idx)
        self._print_summary()

    # ── Progress display ──────────────────────────────────────────

    def _print_progress(self, current_title: str = ''):
        elapsed   = time.time() - self._session_start
        rate      = self._sentences_done / max(elapsed, 1)
        lib_size  = len(self.system.library)
        print(
            f'  [{time.strftime("%H:%M:%S")}] '
            f'articles={self._articles_done:,} | '
            f'sentences={self._sentences_done:,} | '
            f'words_learned={self._words_learned:,} | '
            f'lib={lib_size:,} | '
            f'rate={rate:.1f}/s | '
            f'"{current_title[:30]}"'
        )

    def _print_summary(self):
        elapsed  = time.time() - self._session_start
        hours    = int(elapsed // 3600)
        minutes  = int((elapsed % 3600) // 60)
        print(f'\n{"=" * 60}')
        print(f'SESSION COMPLETE')
        print(f'  Articles processed : {self._articles_done:,}')
        print(f'  Sentences learned  : {self._sentences_done:,}')
        print(f'  New words learned  : {self._words_learned:,}')
        print(f'  Library size       : {len(self.system.library):,}')
        print(f'  Time elapsed       : {hours}h {minutes}m')
        print(f'{"=" * 60}\n')


# ── Entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='HGLS Wikipedia Ingestion')
    parser.add_argument('--memory',           default='deepak_memory.json',
                        help='Memory file path (default: deepak_memory.json)')
    parser.add_argument('--save-every',       type=int, default=50000,
                        help='Save every N sentences (default: 50000)')
    parser.add_argument('--print-every',      type=int, default=1000,
                        help='Print progress every N sentences (default: 1000)')
    parser.add_argument('--max-sentences',    type=int, default=0,
                        help='Stop after N sentences, 0=unlimited (default: 0)')
    parser.add_argument('--reset-checkpoint', action='store_true',
                        help='Ignore checkpoint and start from beginning')
    parser.add_argument('--no-llm',          action='store_true',
                        help='Run without LLM parent (recommended for bulk ingestion)')
    args = parser.parse_args()

    # Boot system
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hgls.system import HGLSystem

    use_llm = not args.no_llm
    system  = HGLSystem(use_llm=use_llm)

    # Load existing memory
    if os.path.exists(args.memory):
        system.load(args.memory)
    else:
        print(f'[Main] No existing memory found at {args.memory} — starting fresh.')

    # Reset checkpoint if requested
    ingestor = WikiIngestor(system, save_path=args.memory)
    if args.reset_checkpoint:
        cp = ingestor.checkpoint_path
        if os.path.exists(cp):
            os.remove(cp)
            print('[Main] Checkpoint reset.')

    # Run
    ingestor.run(
        max_sentences=args.max_sentences,
        save_every=args.save_every,
        print_every=args.print_every,
    )


if __name__ == '__main__':
    main()
