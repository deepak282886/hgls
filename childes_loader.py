"""
childes_loader.py

CHILDES data ingestion pipeline.

Reads CHAT format transcripts (.cha files) and streams them
through the living loop as conversational turns.

CHILDES is the ideal training data because:
    - Real parent-child conversation — grounded, corrective
    - Sequential — what follows what is the ground truth
    - Multi-age — from babbling to complex reasoning
    - Annotated — speaker roles preserved (MOT, CHI, FAT etc.)
    - Massive — millions of utterances across hundreds of corpora

Data source: https://childes.talkbank.org
    Download any corpus as a .zip file.
    Point this loader at the zip or the extracted directory.

Usage:
    # From a downloaded zip
    loader = CHILDESLoader(loop)
    loader.load("Brown.zip")

    # From a directory of .cha files
    loader.load("/path/to/Brown/")

    # Stream everything in a directory tree
    loader.load_all("/path/to/Eng-NA/")

    # With progress reporting
    loader.load("Brown.zip", verbose=True)

The loader handles:
    - Speaker normalization (MOT/FAT/SIS → "adult", CHI → "child")
    - Utterance cleaning (remove CHAT annotations, punctuation markers)
    - Context preservation (consecutive turns in same file stay connected)
    - File boundaries (context resets between files)
    - Corrupted files (skipped with warning)
    - Progress tracking and rate reporting
"""

import os
import re
import time
import zipfile
import tempfile
from pathlib     import Path
from dataclasses import dataclass, field
from typing      import Iterator, Optional

from core.loop import LivingLoop


# ─────────────────────────────────────────────
# SPEAKER NORMALIZATION
#
# CHILDES uses many speaker codes.
# We normalize to two roles:
#   "adult" — any adult speaker (MOT, FAT, SIS, GRA, INV etc.)
#   "child" — the target child (CHI)
#
# This preserves the adult→child→adult dynamic
# without overfitting to specific family members.
# ─────────────────────────────────────────────

CHILD_CODES = {"CHI"}

ADULT_CODES = {
    "MOT", "FAT", "SIS", "BRO", "GRA", "GRF", "GRM",
    "UNC", "AUN", "INV", "EXP", "ADU", "MAL", "FEM",
    "MEN", "WOM", "BOY", "GIR", "TEA", "OBS", "CAR",
}

def normalize_speaker(code: str) -> str:
    code = code.upper().strip()
    if code in CHILD_CODES:
        return "child"
    return "adult"


# ─────────────────────────────────────────────
# UTTERANCE CLEANING
#
# CHAT format has many annotation conventions
# we want to strip before ingestion:
#   [: correction]  — corrected form
#   [//]            — retracing
#   &word           — phonological fragment
#   xxx, yyy, www   — unintelligible
#   0word           — omitted word
#   +...            — trailing off
#   <text> [//]     — retraced text
# ─────────────────────────────────────────────

# Patterns to remove entirely
_REMOVE_PATTERNS = [
    r'\[.*?\]',          # anything in square brackets
    r'&\w+',             # phonological fragments
    r'\(\.+\)',          # pauses (.)(...)
    r'<[^>]+>',          # angle bracket groupings
    r'\+\.\.\.',         # trailing off
    r'\+/+',             # interruption markers
    r'0\w+',             # omitted words
    r'\x15\d+_\d+\x15', # time stamps
    r'[^\w\s\'-]',       # non-word chars except apostrophe/hyphen
]

_UNINTELLIGIBLE = {"xxx", "yyy", "www", "zzz"}

def clean_utterance(raw: str) -> str:
    """
    Clean a raw CHAT utterance into plain text.
    Returns empty string if nothing intelligible remains.
    """
    text = raw

    # Remove each pattern
    for pattern in _REMOVE_PATTERNS:
        text = re.sub(pattern, " ", text)

    # Tokenize and filter
    words = []
    for word in text.split():
        word = word.strip(".,!?;:'\"").lower()
        if not word:
            continue
        if word in _UNINTELLIGIBLE:
            continue
        if len(word) < 1:
            continue
        words.append(word)

    return " ".join(words)


# ─────────────────────────────────────────────
# LOAD STATS
# ─────────────────────────────────────────────

@dataclass
class LoadStats:
    files_processed   : int   = 0
    files_skipped     : int   = 0
    utterances_total  : int   = 0
    utterances_ingested: int  = 0
    utterances_skipped: int   = 0   # too short or unintelligible
    tokens_ingested   : int   = 0
    elapsed_sec       : float = 0.0
    errors            : list  = field(default_factory=list)

    def rate(self) -> float:
        """Utterances per second."""
        return self.utterances_ingested / max(0.001, self.elapsed_sec)

    def report(self) -> str:
        lines = [
            f"Files processed  : {self.files_processed}",
            f"Files skipped    : {self.files_skipped}",
            f"Utterances total : {self.utterances_total}",
            f"Utterances used  : {self.utterances_ingested}",
            f"Utterances skipped: {self.utterances_skipped}",
            f"Tokens ingested  : {self.tokens_ingested}",
            f"Elapsed          : {self.elapsed_sec:.1f}s",
            f"Rate             : {self.rate():.1f} utt/s",
        ]
        if self.errors:
            lines.append(f"Errors           : {len(self.errors)}")
            for e in self.errors[:5]:
                lines.append(f"  {e}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# CHILDES LOADER
# ─────────────────────────────────────────────

class CHILDESLoader:
    """
    Streams CHILDES CHAT transcripts through the living loop.

    Usage:
        loader = CHILDESLoader(loop)
        stats  = loader.load("Brown.zip", verbose=True)
        print(stats.report())
    """

    def __init__(
        self,
        loop            : LivingLoop,
        min_utt_words   : int  = 2,     # skip utterances shorter than this
        max_utt_words   : int  = 50,    # skip utterances longer than this
        speaker_filter  : Optional[set] = None,  # None = all speakers
    ):
        self.loop          = loop
        self.min_utt_words = min_utt_words
        self.max_utt_words = max_utt_words
        self.speaker_filter= speaker_filter  # e.g. {"child", "adult"}


    # ─────────────────────────────────────────
    # MAIN LOAD — handles zip, dir, or single file
    # ─────────────────────────────────────────

    def load(
        self,
        path    : str,
        verbose : bool = True,
        limit   : int  = 0,    # max utterances (0 = unlimited)
    ) -> LoadStats:
        """
        Load CHILDES data from a zip, directory, or single .cha file.

        path: path to .zip file, directory of .cha files, or single .cha
        limit: stop after this many utterances (0 = no limit)
        """
        stats = LoadStats()
        t0    = time.time()
        p     = Path(path)

        if not p.exists():
            print(f"Path not found: {path}")
            stats.files_skipped += 1
            return stats

        if str(path).endswith(".zip"):
            self._load_zip(path, stats, verbose, limit)
        elif p.is_dir():
            self._load_directory(path, stats, verbose, limit)
        elif str(path).endswith(".cha"):
            self._load_file(path, stats, verbose)
        else:
            print(f"Unrecognized format: {path}")
            stats.files_skipped += 1

        stats.elapsed_sec = time.time() - t0

        if verbose:
            print(f"\nLoad complete.\n{stats.report()}")
            gs = self.loop.graph.stats()
            print(f"\nGraph: {gs['nodes']} nodes  "
                  f"{gs['edges']} edges  "
                  f"max_level={gs['max_level']}  "
                  f"avg_weight={gs['avg_weight']:.3f}")

        return stats


    def load_all(
        self,
        root_dir : str,
        verbose  : bool = True,
        limit    : int  = 0,
    ) -> LoadStats:
        """
        Recursively load all .cha files and .zip archives
        under root_dir. Use for loading entire CHILDES language
        collections (e.g. all of Eng-NA/).
        """
        stats = LoadStats()
        t0    = time.time()
        root  = Path(root_dir)

        if not root.exists():
            print(f"Directory not found: {root_dir}")
            return stats

        # Find all .zip and .cha files recursively
        zips = sorted(root.rglob("*.zip"))
        chas = sorted(root.rglob("*.cha"))

        if verbose:
            print(f"Found {len(zips)} zip files, {len(chas)} .cha files\n")

        total = 0
        for zp in zips:
            if limit > 0 and total >= limit:
                break
            sub_stats = self.load(str(zp), verbose=False,
                                  limit=limit - total if limit > 0 else 0)
            self._merge_stats(stats, sub_stats)
            total += sub_stats.utterances_ingested

            if verbose:
                print(f"  {zp.name}: {sub_stats.utterances_ingested} utt  "
                      f"total={total}  "
                      f"rate={sub_stats.rate():.0f}/s")

        for cha in chas:
            if limit > 0 and total >= limit:
                break
            sub_stats = self.load(str(cha), verbose=False)
            self._merge_stats(stats, sub_stats)
            total += sub_stats.utterances_ingested

        stats.elapsed_sec = time.time() - t0

        if verbose:
            print(f"\nAll corpora loaded.\n{stats.report()}")

        return stats


    # ─────────────────────────────────────────
    # LOAD ZIP
    # ─────────────────────────────────────────

    def _load_zip(
        self,
        zip_path : str,
        stats    : LoadStats,
        verbose  : bool,
        limit    : int,
    ):
        """Extract zip to temp dir and load all .cha files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir)
            except Exception as e:
                stats.errors.append(f"ZIP error {zip_path}: {e}")
                stats.files_skipped += 1
                return

            self._load_directory(tmpdir, stats, verbose, limit)


    # ─────────────────────────────────────────
    # LOAD DIRECTORY
    # ─────────────────────────────────────────

    def _load_directory(
        self,
        dir_path : str,
        stats    : LoadStats,
        verbose  : bool,
        limit    : int,
    ):
        """Load all .cha files in a directory."""
        cha_files = sorted(Path(dir_path).rglob("*.cha"))

        if not cha_files:
            if verbose:
                print(f"  No .cha files found in {dir_path}")
            return

        if verbose:
            print(f"  Loading {len(cha_files)} files from "
                  f"{Path(dir_path).name}...")

        for cha_file in cha_files:
            if limit > 0 and stats.utterances_ingested >= limit:
                break
            self._load_file(str(cha_file), stats, verbose=False)

            if verbose and stats.files_processed % 10 == 0:
                print(f"    files={stats.files_processed}  "
                      f"utt={stats.utterances_ingested}  "
                      f"nodes={self.loop.graph.stats()['nodes']}")


    # ─────────────────────────────────────────
    # LOAD ONE FILE
    # ─────────────────────────────────────────

    def _load_file(
        self,
        cha_path : str,
        stats    : LoadStats,
        verbose  : bool = False,
    ):
        """
        Load one .cha file. Parse with pylangacq if available,
        fall back to manual parsing if not.
        """
        try:
            utterances = list(self._parse_cha(cha_path))
        except Exception as e:
            stats.errors.append(f"Parse error {cha_path}: {e}")
            stats.files_skipped += 1
            return

        if not utterances:
            stats.files_skipped += 1
            return

        stats.files_processed += 1

        # Reset context at file boundary — each file is a separate session
        self.loop.ingestion.reset_context()

        for speaker, text in utterances:
            stats.utterances_total += 1

            # Filter by speaker if specified
            if self.speaker_filter and speaker not in self.speaker_filter:
                stats.utterances_skipped += 1
                continue

            # Filter by length
            words = text.split()
            if len(words) < self.min_utt_words:
                stats.utterances_skipped += 1
                continue
            if len(words) > self.max_utt_words:
                stats.utterances_skipped += 1
                continue

            # Ingest the turn
            results = self.loop.ingest_turn(speaker, text)
            stats.utterances_ingested += 1
            stats.tokens_ingested     += len(results)

        if verbose:
            print(f"  {Path(cha_path).name}: "
                  f"{stats.utterances_ingested} utterances ingested")


    # ─────────────────────────────────────────
    # PARSE .CHA FILE
    # Try pylangacq first, fall back to manual
    # ─────────────────────────────────────────

    def _parse_cha(self, cha_path: str) -> Iterator[tuple[str, str]]:
        """
        Parse a .cha file into (speaker, text) pairs.
        Uses pylangacq if available, manual parser as fallback.
        """
        try:
            import pylangacq
            reader   = pylangacq.read_chat(cha_path, strict=False)
            utts     = list(reader.utterances())

            for utt in utts:
                # Extract tokens, skip punctuation
                tokens = [
                    t.word.lower()
                    for t in utt.tokens
                    if t.word and t.word not in {
                        ".", "?", "!", ",", ";", ":", "'", '"'
                    }
                ]

                text = " ".join(tokens).strip()
                if not text:
                    continue

                # Clean further
                text = clean_utterance(text)
                if not text:
                    continue

                speaker = normalize_speaker(utt.participant)
                yield speaker, text

        except Exception:
            # Fallback: manual line-by-line parsing
            yield from self._parse_cha_manual(cha_path)


    def _parse_cha_manual(
        self,
        cha_path: str,
    ) -> Iterator[tuple[str, str]]:
        """
        Manual CHAT parser. Handles the most common patterns
        without requiring pylangacq.

        CHAT format:
            *SPK: utterance text .
            %tier: annotation (skip these)
        """
        try:
            with open(cha_path, "r", encoding="utf-8",
                      errors="replace") as f:
                lines = f.readlines()
        except Exception:
            return

        for line in lines:
            line = line.strip()

            # Skip non-utterance lines
            if not line.startswith("*"):
                continue

            # Parse speaker and text
            # Format: *SPK:\ttext .
            match = re.match(r'^\*([A-Z]{2,3}):\s+(.+)$', line)
            if not match:
                continue

            code, raw_text = match.group(1), match.group(2)
            text = clean_utterance(raw_text)

            if not text:
                continue

            speaker = normalize_speaker(code)
            yield speaker, text


    # ─────────────────────────────────────────
    # MERGE STATS
    # ─────────────────────────────────────────

    def _merge_stats(self, target: LoadStats, source: LoadStats):
        target.files_processed    += source.files_processed
        target.files_skipped      += source.files_skipped
        target.utterances_total   += source.utterances_total
        target.utterances_ingested+= source.utterances_ingested
        target.utterances_skipped += source.utterances_skipped
        target.tokens_ingested    += source.tokens_ingested
        target.errors.extend(source.errors)


    # ─────────────────────────────────────────
    # STREAM MODE
    # For very large corpora — yields progress periodically
    # ─────────────────────────────────────────

    def stream(
        self,
        path      : str,
        report_every: int = 1000,   # utterances between progress reports
    ) -> Iterator[LoadStats]:
        """
        Stream-load with periodic progress yields.
        Use for large corpora where you want live feedback.

        for stats in loader.stream("Eng-NA.zip", report_every=500):
            print(f"  {stats.utterances_ingested} utterances  "
                  f"{stats.rate():.0f}/s")
        """
        stats = LoadStats()
        t0    = time.time()
        p     = Path(path)

        if str(path).endswith(".zip"):
            cha_files = self._extract_and_list(path)
        elif p.is_dir():
            cha_files = sorted(p.rglob("*.cha"))
        elif str(path).endswith(".cha"):
            cha_files = [p]
        else:
            return

        for cha_file in cha_files:
            self._load_file(str(cha_file), stats, verbose=False)

            if stats.utterances_ingested % report_every < 10:
                stats.elapsed_sec = time.time() - t0
                yield stats

        stats.elapsed_sec = time.time() - t0
        yield stats


    def _extract_and_list(self, zip_path: str) -> list[Path]:
        """Extract zip and return list of .cha paths. Caller manages temp."""
        # Note: caller must handle temp dir lifecycle for streaming
        # For simplicity, extract to a fixed location
        extract_dir = Path(zip_path).parent / (Path(zip_path).stem + "_extracted")
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        return sorted(extract_dir.rglob("*.cha"))


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────

def load_childes(
    path    : str,
    db_path : str  = "graph.db",
    verbose : bool = True,
    limit   : int  = 0,
) -> tuple[LivingLoop, LoadStats]:
    """
    Convenience function. Load CHILDES data into a fresh or
    existing graph and return the running loop.

    Returns (loop, stats) — loop is already started.

    Usage:
        loop, stats = load_childes("Brown.zip")
        interface   = Interface(loop)
        interface.conversation()
    """
    from core.graph      import Graph
    from core.loop       import LivingLoop, LoopConfig
    from core.interface  import Interface

    graph  = Graph(db_path)
    config = LoopConfig(
        compress_every   = 50,
        project_every    = 3,
        decay_every_sec  = 30.0,
        verbose          = verbose,
    )
    loop   = LivingLoop(graph, config)
    loader = CHILDESLoader(loop)

    loop.start()

    if verbose:
        print(f"Loading CHILDES data from: {path}")
        print(f"Graph DB: {db_path}\n")

    stats = loader.load(path, verbose=verbose, limit=limit)

    return loop, stats


if __name__ == "__main__":
    import tempfile, os
    from core.graph      import Graph
    from core.loop       import LivingLoop, LoopConfig
    from core.interface  import Interface

    print("=== childes_loader.py smoke test ===\n")

    # Test with synthetic .cha file — same format as real CHILDES
    cha_content = """@UTF8
@Begin
@Languages: eng
@Participants: CHI Adam Target_Child, MOT Mother
@ID: eng|Brown|CHI|2;0.|male|||Target_Child|||
@ID: eng|Brown|MOT|||||Mother|||
*MOT: what is that ?
*CHI: kitty !
*MOT: yes that is a kitty .
*CHI: big kitty .
*MOT: yes it is a big kitty .
*CHI: more kitty ?
*MOT: do you want to see more kitties ?
*CHI: yes more .
*MOT: okay let us look for more kitties .
*CHI: what that ?
*MOT: that is a doggy .
*CHI: doggy run .
*MOT: yes the doggy is running .
*CHI: fast doggy .
*MOT: yes the doggy runs very fast .
*CHI: kitty run ?
*MOT: can the kitty run ?
*CHI: yes kitty run fast .
*MOT: yes the kitty can also run fast .
*CHI: doggy bigger ?
*MOT: is the doggy bigger than the kitty ?
*CHI: yes doggy bigger .
*MOT: you are right the doggy is bigger than the kitty .
*CHI: kitty smaller .
*MOT: yes the kitty is smaller than the doggy .
@End
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test .cha file
        cha_path = os.path.join(tmpdir, "test.cha")
        with open(cha_path, "w", encoding="utf-8") as f:
            f.write(cha_content)

        # Set up system
        db_path = os.path.join(tmpdir, "graph.db")
        graph   = Graph(db_path)
        config  = LoopConfig(
            compress_every   = 20,
            project_every    = 2,
            decay_every_sec  = 60.0,
            verbose          = True,
        )
        loop    = LivingLoop(graph, config)
        loader  = CHILDESLoader(loop, min_utt_words=1)

        print("Loading test .cha file...\n")
        stats = loader.load(cha_path, verbose=True)
        print()

        # Run some ticks
        for _ in range(5):
            loop.tick()

        # Test interface
        print("\nTesting interface after CHILDES load...\n")
        interface = Interface(loop)

        queries = [
            "what is the kitty",
            "is the doggy bigger",
            "can the kitty run",
        ]

        for q in queries:
            response = interface.respond(q)
            print(f"Q: {q}")
            print(f"A: {response.text}  "
                  f"[conf={response.confidence:.3f}]")
            print()

        graph.close()

    print("\nTo load real CHILDES data:")
    print("  1. Download a corpus from https://childes.talkbank.org")
    print("  2. Save the .zip file locally")
    print("  3. Run:")
    print("       from childes_loader import load_childes")
    print("       loop, stats = load_childes('Brown.zip')")
