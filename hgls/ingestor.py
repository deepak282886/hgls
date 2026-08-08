"""
ingestor.py — Dataset Ingestor for Pre-training and Fine-tuning.

Two modes:

  PRE-TRAINING
    Feed raw text chunks through run_cycle() at the appropriate level.
    Builds the factual knowledge base — the library fills up with structures
    about the world. No teacher correction signal, just volume.

    Compatible datasets:
      Wikipedia dumps, OpenStax textbooks, C4, CommonCrawl, Books3,
      NCERT textbooks, Project Gutenberg, OpenWebText

  FINE-TUNING
    Feed (question, answer) pairs through learn_correction().
    Structures learned here get correction_count > 0 and topic tags,
    so they dominate over pre-trained content for matching topic queries.

    Compatible datasets:
      SlimOrca (CoT), OpenHermes, GSM8K, MetaMath, ARC, TheoremQA,
      CBSE/JEE past papers, NCERT Q&A

Usage:
    from hgls.ingestor import DatasetIngestor
    from hgls.system import HGLSystem

    system = HGLSystem(use_llm=False)
    ingestor = DatasetIngestor(system)

    # Pre-train on raw text
    ingestor.pretrain_texts(["water evaporates when heated", ...], level=3)

    # Fine-tune on Q&A pairs
    ingestor.finetune_qa([
        {"question": "why does water evaporate?",
         "answer": "water evaporates because heat gives molecules enough energy to escape"},
        ...
    ], level=4)
"""

import re
import time
from typing import List, Dict, Optional, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.system import HGLSystem

# How many characters per chunk when splitting raw text
CHUNK_SIZE    = 120
# Overlap between chunks to preserve context at boundaries
CHUNK_OVERLAP = 20
# Pause between batches to avoid overwhelming memory
BATCH_PAUSE   = 0.0


class DatasetIngestor:
    """
    Feeds open source datasets into the HGLS learning system.
    Handles chunking, level assignment, and progress tracking.
    """

    def __init__(self, system: 'HGLSystem'):
        self.system           = system
        self._pretrain_count  = 0
        self._finetune_count  = 0
        self._error_count     = 0

    # ── Pre-training ──────────────────────────────────────────────

    def pretrain_texts(
        self,
        texts: List[str],
        level: int = None,
        batch_size: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """
        Pre-train on a list of raw text strings.
        Each string is chunked and fed through run_cycle().

        level=None → use curriculum's active level (recommended for staged training).
        level=3    → force phrase level (good for factual sentences).
        level=4    → force schema level (good for cause-effect statements).
        """
        total_chunks = 0
        total_success = 0

        if verbose:
            print(f"\n[Ingestor] Pre-training on {len(texts)} texts "
                  f"(level={'auto' if level is None else level})")

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start: batch_start + batch_size]
            batch_chunks = 0
            batch_success = 0

            for text in batch:
                for chunk in self._chunk_text(text):
                    try:
                        result = self.system.ingest_text(chunk, level=level, is_correction=False)
                        batch_chunks  += 1
                        batch_success += result.get('n_successes', 0)
                        self._pretrain_count += 1
                    except Exception as e:
                        self._error_count += 1

            total_chunks  += batch_chunks
            total_success += batch_success

            if verbose:
                print(
                    f"  Batch {batch_start // batch_size + 1} | "
                    f"chunks={batch_chunks} | "
                    f"successes={batch_success} | "
                    f"lib_size={len(self.system.library)}"
                )

            if BATCH_PAUSE > 0:
                time.sleep(BATCH_PAUSE)

        if verbose:
            print(f"[Ingestor] Pre-training complete. "
                  f"Total chunks: {total_chunks}, "
                  f"successes: {total_success}, "
                  f"library: {len(self.system.library)}")

        return {
            'total_chunks':   total_chunks,
            'total_successes': total_success,
            'errors':         self._error_count,
            'lib_size':       len(self.system.library),
        }

    # ── Fine-tuning ───────────────────────────────────────────────

    def finetune_qa(
        self,
        qa_pairs: List[Dict],
        level: int = None,
        batch_size: int = 50,
        verbose: bool = True,
    ) -> Dict:
        """
        Fine-tune on question-answer pairs.
        Each answer is learned as a teacher correction tagged with question topics.
        These structures dominate pre-trained content for matching topic queries.

        qa_pairs: list of dicts with 'question' and 'answer' keys.
                  Optionally 'chain_of_thought' for CoT datasets (SlimOrca, GSM8K).

        Example:
          {"question": "why do we brush teeth?",
           "answer": "brushing removes bacteria and keeps teeth healthy"}

          {"question": "what is 2 + 2?",
           "answer": "i know 2 and 2 are numbers. that means i add them. so 2 + 2 = 4",
           "chain_of_thought": true}
        """
        total_pairs   = 0
        total_success = 0

        if verbose:
            print(f"\n[Ingestor] Fine-tuning on {len(qa_pairs)} Q-A pairs "
                  f"(level={'auto' if level is None else level})")

        for batch_start in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[batch_start: batch_start + batch_size]
            batch_pairs   = 0
            batch_success = 0

            for pair in batch:
                question = pair.get('question', '')
                answer   = pair.get('answer',   '')
                cot      = pair.get('chain_of_thought', None)

                if not question or not answer:
                    continue

                # Extract topic words from the question
                topic_words = _extract_topic_words(question)

                # If CoT provided as a separate field, learn it first as a correction
                if isinstance(cot, str) and cot:
                    try:
                        self.system.ingest_text(
                            cot,
                            level=level,
                            is_correction=True,
                            topic_words=topic_words,
                        )
                    except Exception:
                        pass

                # Learn the answer as a teacher correction
                try:
                    result = self.system.ingest_text(
                        answer,
                        level=level,
                        is_correction=True,
                        topic_words=topic_words,
                    )
                    batch_pairs   += 1
                    batch_success += result.get('successes', 0)
                    self._finetune_count += 1
                except Exception:
                    self._error_count += 1

            total_pairs   += batch_pairs
            total_success += batch_success

            if verbose:
                print(
                    f"  Batch {batch_start // batch_size + 1} | "
                    f"pairs={batch_pairs} | "
                    f"successes={batch_success} | "
                    f"lib_size={len(self.system.library)}"
                )

            if BATCH_PAUSE > 0:
                time.sleep(BATCH_PAUSE)

        if verbose:
            print(f"[Ingestor] Fine-tuning complete. "
                  f"Total pairs: {total_pairs}, "
                  f"successes: {total_success}, "
                  f"library: {len(self.system.library)}")

        return {
            'total_pairs':    total_pairs,
            'total_successes': total_success,
            'errors':         self._error_count,
            'lib_size':       len(self.system.library),
        }

    # ── Streaming ingest (large datasets) ────────────────────────

    def pretrain_stream(
        self,
        text_iterator: Iterator[str],
        level: int = None,
        save_every: int = 10000,
        save_path: str = 'deepak_memory.json',
        verbose: bool = True,
    ) -> Dict:
        """
        Stream-ingest from an iterator — for large datasets like Wikipedia
        or CommonCrawl that don't fit in memory.

        Usage:
            def wiki_lines():
                with open('wiki.txt') as f:
                    for line in f:
                        yield line.strip()

            ingestor.pretrain_stream(wiki_lines(), level=3)
        """
        count   = 0
        success = 0

        for text in text_iterator:
            for chunk in self._chunk_text(text):
                try:
                    result  = self.system.ingest_text(chunk, level=level)
                    success += result.get('n_successes', 0)
                    count   += 1
                except Exception:
                    self._error_count += 1

                if save_every > 0 and count % save_every == 0:
                    self.system.save(save_path)
                    if verbose:
                        print(f"  [Stream] {count} chunks | "
                              f"lib={len(self.system.library)} | saved")

        if verbose:
            print(f"[Ingestor] Stream complete. {count} chunks, "
                  f"lib={len(self.system.library)}")

        return {'total_chunks': count, 'successes': success, 'errors': self._error_count}

    # ── Text chunking ─────────────────────────────────────────────

    def _chunk_text(self, text: str) -> Iterator[str]:
        """
        Split text into learnable chunks.

        Strategy (in priority order):
          1. Split on sentence boundaries (. ! ?)
          2. If a sentence is still too long, split on commas or conjunctions
          3. If still too long, hard-split at CHUNK_SIZE with CHUNK_OVERLAP

        Filters: skip empty chunks, chunks with too many numbers/symbols,
        chunks shorter than 3 words.
        """
        # Normalise whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        if not text:
            return

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Skip chunks that are mostly numbers or symbols (tables, formulas)
            alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
            if alpha_ratio < 0.5:
                continue

            # Skip very short chunks
            words = sent.split()
            if len(words) < 3:
                continue

            # If sentence fits, yield it
            if len(sent) <= CHUNK_SIZE:
                yield sent
            else:
                # Split long sentences at commas or conjunctions
                sub_parts = re.split(r',\s+| and | but | because | so | then ', sent)
                current = ''
                for part in sub_parts:
                    if len(current) + len(part) <= CHUNK_SIZE:
                        current = (current + ' ' + part).strip() if current else part
                    else:
                        if current and len(current.split()) >= 3:
                            yield current
                        current = part
                if current and len(current.split()) >= 3:
                    yield current

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            'pretrain_chunks': self._pretrain_count,
            'finetune_pairs':  self._finetune_count,
            'errors':          self._error_count,
            'lib_size':        len(self.system.library),
        }


# ── Module-level helpers ───────────────────────────────────────────

_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'on',
    'at', 'by', 'for', 'with', 'about', 'from', 'into', 'through',
    'and', 'or', 'but', 'not', 'no', 'so', 'if', 'then', 'that',
    'this', 'it', 'he', 'she', 'they', 'we', 'i', 'you', 'my', 'your',
    'what', 'how', 'why', 'when', 'where', 'who', 'which',
}


def _extract_topic_words(text: str) -> List[str]:
    """Extract meaningful content words as topic tags."""
    words = re.findall(r'[a-z]+', text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]
