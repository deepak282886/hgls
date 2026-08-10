"""
mastery_tracker.py — Topic Mastery Tracker.

90% correct over last 10 attempts = mastered.
Tracks every topic's history. Single JSON file: deepak_progress.json.
Atomic save. Full resume on restart.
"""

import os
import json
import time
from typing import List, Optional
from dataclasses import dataclass, field, asdict

from syllabus import SYLLABUS, Topic


PROGRESS_FILE  = 'deepak_progress.json'
MASTERY_WINDOW = 10      # look at last N attempts
MASTERY_RATE   = 0.90    # 90% correct in window = mastered


@dataclass
class TopicRecord:
    topic_id:    str
    subject:     str
    grade:       int
    name:        str
    status:      str        = 'not_started'   # not_started | in_progress | mastered
    attempts:    int        = 0
    history:     List[bool] = field(default_factory=list)  # True=correct False=wrong
    pass_number: int        = 1
    started_at:  float      = 0.0
    mastered_at: float      = 0.0

    def record_attempt(self, correct: bool) -> bool:
        """Record one attempt. Returns True if topic is now mastered."""
        self.attempts += 1
        self.history.append(correct)
        if self.status == 'not_started':
            self.status    = 'in_progress'
            self.started_at = time.time()

        # Check mastery: last MASTERY_WINDOW attempts
        window = self.history[-MASTERY_WINDOW:]
        if len(window) >= MASTERY_WINDOW:
            rate = sum(window) / len(window)
            if rate >= MASTERY_RATE:
                self.status     = 'mastered'
                self.mastered_at = time.time()
                return True
        return False

    @property
    def correct_rate(self) -> float:
        if not self.history:
            return 0.0
        window = self.history[-MASTERY_WINDOW:]
        return sum(window) / len(window)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['history'] = [int(b) for b in self.history]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TopicRecord':
        d['history'] = [bool(b) for b in d.get('history', [])]
        return cls(**d)


class MasteryTracker:
    """
    Tracks mastery across all syllabus topics.
    Single source of truth: deepak_progress.json
    """

    def __init__(self, progress_file: str = PROGRESS_FILE):
        self.progress_file = progress_file
        self._records: List[TopicRecord] = []
        self._index: dict                = {}  # topic_id → record
        self._current_idx: int           = 0
        self._pass_number: int           = 1

        self._initialise()

    def _initialise(self):
        """Build topic records from syllabus. Load existing progress if any."""
        # Build fresh from syllabus
        for i, topic in enumerate(SYLLABUS):
            tid = f"{topic.subject}_{topic.grade}_{i}"
            rec = TopicRecord(
                topic_id = tid,
                subject  = topic.subject,
                grade    = topic.grade,
                name     = topic.name,
            )
            self._records.append(rec)
            self._index[tid]  = rec

        # Overlay with saved progress
        if os.path.exists(self.progress_file):
            self._load()
        else:
            print(f"[Tracker] Fresh start — {len(self._records)} topics across all subjects/grades.")

    # ── Navigation ────────────────────────────────────────────────

    def current_topic(self) -> Optional[TopicRecord]:
        """Return the current topic being taught."""
        if self._current_idx >= len(self._records):
            return None
        return self._records[self._current_idx]

    def current_topic_object(self) -> Optional[Topic]:
        """Return the syllabus Topic object for the current topic."""
        if self._current_idx >= len(self._records):
            return None
        return SYLLABUS[self._current_idx]

    def advance(self) -> Optional[TopicRecord]:
        """Move to next topic. Returns next record or None if curriculum complete."""
        self._current_idx += 1
        self.save()
        if self._current_idx >= len(self._records):
            print(f"\n[Tracker] *** PASS {self._pass_number} COMPLETE ***")
            self._pass_number += 1
            self._current_idx = 0
            # Reset not_started topics for next pass
            for rec in self._records:
                if rec.status != 'mastered':
                    rec.status    = 'not_started'
                    rec.pass_number = self._pass_number
            self.save()
        return self.current_topic()

    def record_attempt(self, correct: bool) -> bool:
        """
        Record an attempt on the current topic.
        Returns True if topic is now mastered.
        """
        rec = self.current_topic()
        if rec is None:
            return False
        mastered = rec.record_attempt(correct)
        if mastered:
            self.save()
        return mastered

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        total    = len(self._records)
        mastered = sum(1 for r in self._records if r.status == 'mastered')
        in_prog  = sum(1 for r in self._records if r.status == 'in_progress')
        return {
            'total_topics':   total,
            'mastered':       mastered,
            'in_progress':    in_prog,
            'not_started':    total - mastered - in_prog,
            'mastery_rate':   round(mastered / max(total, 1), 3),
            'current_idx':    self._current_idx,
            'pass_number':    self._pass_number,
            'current_topic':  self.current_topic().name if self.current_topic() else 'complete',
        }

    def summary_by_subject(self) -> dict:
        subjects = {}
        for rec in self._records:
            if rec.subject not in subjects:
                subjects[rec.subject] = {'total': 0, 'mastered': 0}
            subjects[rec.subject]['total'] += 1
            if rec.status == 'mastered':
                subjects[rec.subject]['mastered'] += 1
        return subjects

    # ── Persistence ───────────────────────────────────────────────

    def save(self):
        data = {
            'version':     '0.7',
            'current_idx': self._current_idx,
            'pass_number': self._pass_number,
            'records':     [r.to_dict() for r in self._records],
        }
        tmp = self.progress_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if os.path.exists(self.progress_file):
            backup = self.progress_file + '.bak'
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(self.progress_file, backup)
        os.rename(tmp, self.progress_file)

    def _load(self):
        try:
            with open(self.progress_file, encoding='utf-8') as f:
                data = json.load(f)

            self._current_idx = data.get('current_idx', 0)
            self._pass_number = data.get('pass_number', 1)

            saved = {r['topic_id']: r for r in data.get('records', [])}
            for rec in self._records:
                if rec.topic_id in saved:
                    saved_r = TopicRecord.from_dict(saved[rec.topic_id])
                    rec.status      = saved_r.status
                    rec.attempts    = saved_r.attempts
                    rec.history     = saved_r.history
                    rec.pass_number = saved_r.pass_number
                    rec.started_at  = saved_r.started_at
                    rec.mastered_at = saved_r.mastered_at

            mastered = sum(1 for r in self._records if r.status == 'mastered')
            cur = self.current_topic()
            print(
                f"[Tracker] Resuming pass {self._pass_number} — "
                f"topic {self._current_idx}/{len(self._records)} — "
                f"{mastered} mastered — "
                f"current: {cur.name if cur else 'complete'}"
            )
        except Exception as e:
            print(f"[Tracker] Load failed ({e}) — starting fresh.")