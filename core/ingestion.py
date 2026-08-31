"""
ingestion.py

Ingestion pipeline — builds the graph from text.

Four ingestion modes corresponding to the four hierarchy levels:

    1. ingest_complete_characters()
       Ingest every character in CHAR_VOCAB — finite and exhaustive.
       Run once at bootstrap. Creates ~97 character nodes.

    2. ingest_complete_words()
       Ingest every word in a dictionary file — finite and exhaustive.
       Run once at bootstrap. Creates ~200k+ word nodes.

    3. ingest() / ingest_text()
       Standard ingestion from text datasets (JSONL, TXT).
       Creates SENTENCE-level nodes and connects them.
       Also creates WORD and CHARACTER links if max_level is set.

    4. ingest_axioms()
       Ingest structured axiom corpus from frontier model extraction.
       Creates AXIOM-level nodes with dependency edges.
       Dependency ordering ensures foundational axioms are
       ingested before derived ones.

Document hierarchy (new — no PARAGRAPH or DOCUMENT nodes):
    Text → sentences → words → characters

    AXIOM nodes are first-class — they do not decompose
    from documents. They come directly from the frontier model
    and are ingested via ingest_axioms().

Deduplication: the graph never stores the same text twice.
Identical sentences seen a thousand times strengthen existing
edges rather than creating duplicate nodes.
"""

import os
import re
import json
import time
from pathlib     import Path
from typing      import Iterator, Optional
from dataclasses import dataclass, field

from core.atoms      import (
    Level, CHAR_VOCAB,
    text_to_sentences, text_to_words, text_to_characters, text_to_axioms,
    decompose,
)
from core.graph      import Graph
from core.edge       import EdgeType
from core.normalizer import Normalizer
from core.bellman    import BellmanManager
from core.reward     import node_reward as compute_node_reward


# ─────────────────────────────────────────────
# INGESTION CONFIG
# ─────────────────────────────────────────────

@dataclass
class IngestionConfig:
    """
    Configuration for the ingestion pipeline.

    max_level         : deepest level to create nodes at during
                        text ingestion.
                        SENTENCE  = sentence nodes only (fastest)
                        WORD      = sentence + word nodes
                        CHARACTER = all levels (slowest, most complete)
    batch_size        : documents per Bellman sweep cycle
    max_nodes         : stop after this many total nodes (0 = unlimited)
    min_sentence_words: skip sentences shorter than this
    verbose           : print progress
    bellman_every     : run delta propagation every N edges
    """
    max_level            : Level = Level.SENTENCE
    batch_size           : int   = 100
    max_nodes            : int   = 0
    min_sentence_words   : int   = 3
    verbose              : bool  = True
    bellman_every        : int   = 10


# ─────────────────────────────────────────────
# INGESTION STATS
# ─────────────────────────────────────────────

@dataclass
class IngestionStats:
    documents_read  : int   = 0
    nodes_created   : int   = 0
    nodes_skipped   : int   = 0   # duplicates
    edges_created   : int   = 0
    elapsed_sec     : float = 0.0
    errors          : int   = 0

    level_nodes     : dict  = field(default_factory=lambda: {
        lv.name: 0 for lv in Level
    })

    def report(self) -> str:
        rate  = self.nodes_created / (self.elapsed_sec + 1e-9)
        lines = [
            f"Documents read   : {self.documents_read}",
            f"Nodes created    : {self.nodes_created}  ({rate:.1f}/s)",
            f"Nodes skipped    : {self.nodes_skipped}  (duplicates)",
            f"Edges created    : {self.edges_created}",
            f"Errors           : {self.errors}",
            f"Elapsed          : {self.elapsed_sec:.1f}s",
            f"Level breakdown  :",
        ]
        for lv, count in self.level_nodes.items():
            if count > 0:
                lines.append(f"    {lv:12s}: {count}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# DATASET READERS
# ─────────────────────────────────────────────

def read_txt(path: str) -> Iterator[str]:
    """Read a plain text file. Double-newline blocks = documents."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    docs = [d.strip() for d in content.split("\n\n") if d.strip()]
    if not docs:
        if content.strip():
            yield content.strip()
        return

    if len(docs) == 1:
        yield docs[0]
        return

    current = []
    current_words = 0
    for chunk in docs:
        words = len(chunk.split())
        if current_words + words > 500 and current:
            yield "\n\n".join(current)
            current = []
            current_words = 0
        current.append(chunk)
        current_words += words
    if current:
        yield "\n\n".join(current)


def read_jsonl(path: str) -> Iterator[str]:
    """
    Read a JSONL file. Handles common field names:
    text, content, problem+solution, question+answer.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" in obj and obj["text"]:
                    yield str(obj["text"])
                elif "content" in obj and obj["content"]:
                    yield str(obj["content"])
                elif "problem" in obj and "solution" in obj:
                    yield f"{obj['problem']}\n\n{obj['solution']}"
                elif "question" in obj and "answer" in obj:
                    yield f"{obj['question']}\n\n{obj['answer']}"
                elif isinstance(obj, str):
                    yield obj
            except json.JSONDecodeError:
                continue


def read_dataset(path: str) -> Iterator[str]:
    """Read any supported dataset format. Recursively reads directories."""
    p = Path(path)

    if p.is_dir():
        for child in sorted(p.rglob("*")):
            if child.is_file() and child.suffix in (".txt", ".jsonl", ".json"):
                yield from read_dataset(str(child))
        return

    if not p.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    if p.suffix == ".txt":
        yield from read_txt(path)
    elif p.suffix in (".jsonl", ".json"):
        yield from read_jsonl(path)
    else:
        yield from read_txt(path)


# ─────────────────────────────────────────────
# NODE CREATION WITH DEDUPLICATION
# ─────────────────────────────────────────────

def get_or_create_node(
    text       : str,
    level      : Level,
    graph      : Graph,
    normalizer : Normalizer,
    seen_words : set,
    context    : list[str],
    stats      : IngestionStats,
) -> tuple[Optional[int], bool]:
    """
    Get an existing node or create a new one.
    Returns (node_id, is_new). is_new=False means duplicate found.
    """
    if not text or not text.strip():
        return None, False

    existing_id = graph.text_exists(text.strip(), level)
    if existing_id is not None:
        stats.nodes_skipped += 1
        return existing_id, False

    reward = compute_node_reward(
        text.strip(), level,
        seen_words = seen_words,
        context    = context,
    )

    node = graph.add_node(text.strip(), level, reward=reward)
    normalizer.on_node_added(node, graph)

    stats.nodes_created          += 1
    stats.level_nodes[level.name] += 1

    return node.node_id, True


def create_edge_if_needed(
    source_id     : int,
    target_id     : int,
    source_text   : str,
    target_text   : str,
    level         : Level,
    graph         : Graph,
    normalizer    : Normalizer,
    bm            : BellmanManager,
    stats         : IngestionStats,
    edge_count_ref: list,
    bellman_every : int,
    edge_type     : EdgeType = EdgeType.SEQUENTIAL,
):
    """Create an edge if it does not exist. Triggers Bellman delta."""
    if source_id == target_id:
        return
    if graph.edge_exists(source_id, target_id):
        return

    graph.add_edge(source_id, target_id,
                   source_text, target_text, level,
                   edge_type=edge_type)
    normalizer.on_edge_added(source_id, graph)
    bm.on_edge_added(source_id, target_id)
    stats.edges_created += 1
    edge_count_ref[0]   += 1


# ─────────────────────────────────────────────
# BOOTSTRAP: COMPLETE CHARACTER INGESTION
# ─────────────────────────────────────────────

def ingest_complete_characters(
    graph      : Graph,
    normalizer : Normalizer,
    bm         : BellmanManager,
    verbose    : bool = True,
) -> IngestionStats:
    """
    Ingest every character in CHAR_VOCAB — finite and exhaustive.

    This is the true foundation of the graph. Every word is built
    from these characters. Every sentence is built from these words.
    By ingesting all characters first, the graph has complete
    coverage at Level 0 before anything else is built.

    Creates ~97 character nodes with sequential edges between
    characters that commonly appear together (all pairs from
    the vocabulary). Run once at bootstrap.

    Characters are NOT deduplication-dependent — there are
    exactly VOCAB_SIZE characters and each is unique.
    """
    stats        = IngestionStats()
    t0           = time.time()
    edge_counter = [0]

    if verbose:
        print(f"Ingesting complete character vocabulary ({len(CHAR_VOCAB)} characters)...")

    prev_char_id = None

    for char in CHAR_VOCAB:
        existing_id = graph.text_exists(char, Level.CHARACTER)
        if existing_id is not None:
            char_id = existing_id
        else:
            # Whitespace chars (' ', '\n', '\t') have empty .strip()
            # so we bypass get_or_create_node and insert directly
            reward = compute_node_reward(
                char, Level.CHARACTER,
                seen_words=set(), context=[],
            ) if char.strip() else 0.5

            node = graph.add_node(char, Level.CHARACTER, reward=reward)
            normalizer.on_node_added(node, graph)
            char_id = node.node_id
            stats.nodes_created += 1
            stats.level_nodes[Level.CHARACTER.name] += 1

        if char_id is None:
            continue

        # Sequential edge from previous character in vocabulary
        # This creates a baseline connectivity across the character space
        if prev_char_id is not None:
            create_edge_if_needed(
                prev_char_id, char_id,
                CHAR_VOCAB[CHAR_VOCAB.index(char) - 1], char,
                Level.CHARACTER, graph, normalizer, bm, stats,
                edge_counter, 10,
            )

        prev_char_id = char_id

    bm.sweep(verbose=False)
    stats.elapsed_sec   = time.time() - t0
    stats.documents_read = 1

    if verbose:
        print(f"  Created {stats.nodes_created} character nodes, "
              f"{stats.edges_created} edges in {stats.elapsed_sec:.2f}s")

    return stats


# ─────────────────────────────────────────────
# BOOTSTRAP: COMPLETE WORD INGESTION
# ─────────────────────────────────────────────

def ingest_complete_words(
    word_list_path : str,
    graph          : Graph,
    normalizer     : Normalizer,
    bm             : BellmanManager,
    config         : Optional[IngestionConfig] = None,
    verbose        : bool = True,
) -> IngestionStats:
    """
    Ingest every word from a dictionary/word list — finite and exhaustive.

    The word list file should have one word per line.
    Standard Unix word lists work:
        /usr/share/dict/words  (Linux)
        Or download: https://github.com/dwyl/english-words

    For each word:
        1. Create WORD node
        2. Create CHARACTER nodes for each letter (if not exists)
        3. Create hierarchical edges: char → word

    This completes the lexical layer. Every word that can appear
    in any sentence already exists in the graph before sentences
    are ingested. Co-occurrence edges form during sentence ingestion.

    Words without corresponding character nodes are still created —
    characters were already ingested by ingest_complete_characters().
    """
    if config is None:
        config = IngestionConfig(
            max_level  = Level.WORD,
            verbose    = verbose,
            batch_size = 1000,
        )

    stats        = IngestionStats()
    t0           = time.time()
    batch_count  = 0
    char_word_edges  : list = []
    word_batch       : list = []   # (text, level, reward) for batch insert
    BATCH_SIZE = 2000

    path = Path(word_list_path)
    if not path.exists():
        raise FileNotFoundError(f"Word list not found: {word_list_path}")

    if verbose:
        print(f"Ingesting complete word list from: {word_list_path}")

    def flush_word_batch():
        """Insert a batch of word nodes and their char→word edges."""
        if not word_batch:
            return

        # Batch insert all word nodes at once
        word_ids = graph.add_nodes_batch(word_batch)

        new_count = sum(
            1 for i, (_, _, _) in enumerate(word_batch)
            if word_ids[i] is not None
        )
        stats.nodes_created += len([w for w in word_ids if w is not None])
        stats.level_nodes[Level.WORD.name] += new_count

        # Collect char→word edges
        for i, (word, level, reward) in enumerate(word_batch):
            word_id = word_ids[i]
            if word_id is None:
                continue
            chars = text_to_characters(word)
            for char in chars:
                char_id = graph.text_exists(char, Level.CHARACTER)
                if char_id is not None:
                    char_word_edges.append((
                        char_id, word_id,
                        char, word,
                        Level.CHARACTER, EdgeType.HIERARCHICAL,
                    ))

        word_batch.clear()

        # Flush edges in one batch
        if char_word_edges:
            n_inserted = graph.add_edges_batch(char_word_edges)
            stats.edges_created += n_inserted
            char_word_edges.clear()

        # Notify normalizer for new nodes
        # (skip per-node notification for speed — normalizer
        #  recalibrates during Bellman sweep at the end)

        if verbose:
            elapsed = time.time() - t0
            rate    = stats.nodes_created / (elapsed + 1e-9)
            print(f"  words={stats.nodes_created:7d}  "
                  f"edges={stats.edges_created:7d}  "
                  f"rate={rate:.0f}/s")

    with open(word_list_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            word = line.strip()
            if not word or len(word) < 1:
                continue

            # Skip non-ASCII for now
            try:
                word.encode('ascii')
            except UnicodeEncodeError:
                continue

            reward = compute_node_reward(
                word, Level.WORD,
                seen_words = set(),
                context    = [],
            )
            word_batch.append((word, Level.WORD, reward))
            batch_count += 1

            if batch_count >= BATCH_SIZE:
                flush_word_batch()
                batch_count = 0

    # Flush remaining
    flush_word_batch()

    bm.sweep(verbose=False)
    stats.elapsed_sec    = time.time() - t0
    stats.documents_read = 1

    if verbose:
        print(f"\nWord ingestion complete.")
        print(f"  Words created : {stats.nodes_created}")
        print(f"  Edges created : {stats.edges_created}")
        print(f"  Elapsed       : {stats.elapsed_sec:.1f}s")

    return stats


# ─────────────────────────────────────────────
# AXIOM INGESTION
# ─────────────────────────────────────────────

def ingest_axioms(
    axiom_path : str,
    graph      : Graph,
    normalizer : Normalizer,
    bm         : BellmanManager,
    verbose    : bool = True,
) -> IngestionStats:
    """
    Ingest a structured axiom corpus into the graph.

    Axiom JSONL format (one per line):
        {
            "axiom": "For any function x^n, the derivative is n*x^(n-1).",
            "domain": "calculus",
            "type": "rule",
            "depends_on": ["power_rule_def", "derivative_def"],
            "id": "power_rule"
        }

    Each axiom becomes a Level.AXIOM node.
    Dependency edges are created from each axiom to its dependencies
    — foundational axioms point toward derived axioms.

    Ingestion order matters: axioms with no dependencies are
    ingested first so Bellman propagates correctly. The function
    handles dependency ordering automatically via topological sort.

    The axiom node's reward is computed by the reward function
    but also boosted by the number of axioms that depend on it —
    more generative axioms get higher reward.
    """
    stats        = IngestionStats()
    t0           = time.time()
    edge_counter = [0]

    path = Path(axiom_path)
    if not path.exists():
        raise FileNotFoundError(f"Axiom file not found: {axiom_path}")

    if verbose:
        print(f"Ingesting axioms from: {axiom_path}")

    # Load all axioms first for dependency resolution
    axioms      = []
    axiom_by_id = {}

    with open(axiom_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Support multiple formats
                if "axiom" in obj:
                    text = obj["axiom"]
                elif "text" in obj:
                    text = obj["text"]
                elif "statement" in obj:
                    text = obj["statement"]
                else:
                    continue

                axiom_record = {
                    "text"       : str(text).strip(),
                    "domain"     : obj.get("domain", "general"),
                    "type"       : obj.get("type", "rule"),
                    "depends_on" : obj.get("depends_on", []),
                    "id"         : obj.get("id", ""),
                }
                if axiom_record["text"] and len(axiom_record["text"]) > 5:
                    axioms.append(axiom_record)
                    if axiom_record["id"]:
                        axiom_by_id[axiom_record["id"]] = axiom_record
            except json.JSONDecodeError:
                continue

    if not axioms:
        if verbose:
            print("  No axioms found in file.")
        return stats

    # Topological sort — axioms with no dependencies first
    def get_depth(axiom, visited=None):
        if visited is None:
            visited = set()
        aid = axiom.get("id", "")
        if aid in visited:
            return 0
        visited.add(aid)
        deps = axiom.get("depends_on", [])
        if not deps:
            return 0
        dep_depths = []
        for dep_id in deps:
            dep = axiom_by_id.get(dep_id)
            if dep:
                dep_depths.append(get_depth(dep, visited))
        return max(dep_depths, default=0) + 1

    axioms_sorted = sorted(axioms, key=lambda a: get_depth(a))

    if verbose:
        print(f"  Found {len(axioms)} axioms. Ingesting in dependency order...")

    # Ingest axioms in order
    axiom_node_ids = {}   # axiom_id → node_id in graph

    for axiom in axioms_sorted:
        text   = axiom["text"]
        domain = axiom["domain"]

        # Reward boosted by how many other axioms depend on this one
        dependents = sum(
            1 for a in axioms
            if axiom.get("id", "") in a.get("depends_on", [])
        )

        seen_words : set = set()
        context          = [f"domain:{domain}", f"type:{axiom['type']}"]

        axiom_id, is_new = get_or_create_node(
            text, Level.AXIOM,
            graph, normalizer,
            seen_words, context, stats,
        )
        if axiom_id is None:
            continue

        # Boost reward for generative axioms (many dependents)
        if dependents > 0 and is_new:
            node = graph.get_node(axiom_id)
            if node:
                boosted_reward = node.reward + (dependents * 0.5)
                graph.update_node_V(axiom_id, boosted_reward)

        if axiom.get("id"):
            axiom_node_ids[axiom["id"]] = axiom_id

        # Also ingest component sentences as SENTENCE nodes
        sentences = text_to_sentences(text)
        prev_sent_id = None
        for sent_text in sentences:
            if len(sent_text.split()) < 3:
                continue
            sent_id, _ = get_or_create_node(
                sent_text, Level.SENTENCE,
                graph, normalizer,
                seen_words, context, stats,
            )
            if sent_id is None:
                continue

            # Hierarchical edge: sentence → axiom
            create_edge_if_needed(
                sent_id, axiom_id,
                sent_text, text,
                Level.SENTENCE, graph, normalizer, bm, stats,
                edge_counter, 10,
            )

            if prev_sent_id is not None:
                prev_node = graph.get_node(prev_sent_id)
                if prev_node:
                    create_edge_if_needed(
                        prev_sent_id, sent_id,
                        prev_node.text, sent_text,
                        Level.SENTENCE, graph, normalizer, bm, stats,
                        edge_counter, 10,
                    )
            prev_sent_id = sent_id
            seen_words |= set(re.findall(r'\b\w+\b', sent_text.lower()))

        # Dependency edges: this axiom → axioms it depends on
        for dep_id in axiom.get("depends_on", []):
            dep_node_id = axiom_node_ids.get(dep_id)
            if dep_node_id is not None:
                create_edge_if_needed(
                    axiom_id, dep_node_id,
                    text, graph.get_node(dep_node_id).text if graph.get_node(dep_node_id) else "",
                    Level.AXIOM, graph, normalizer, bm, stats,
                    edge_counter, 10,
                )

    bm.sweep(verbose=False)
    stats.elapsed_sec    = time.time() - t0
    stats.documents_read = len(axioms)

    if verbose:
        print(f"\nAxiom ingestion complete.")
        print(f"  Axioms ingested : {stats.nodes_created}")
        print(f"  Edges created   : {stats.edges_created}")
        print(f"  Elapsed         : {stats.elapsed_sec:.1f}s")

    return stats


# ─────────────────────────────────────────────
# INGEST ONE DOCUMENT
# Hierarchy: text → sentences → words → characters
# No PARAGRAPH or DOCUMENT nodes — only the four
# meaningful levels.
# ─────────────────────────────────────────────

def ingest_document(
    text        : str,
    graph       : Graph,
    normalizer  : Normalizer,
    bm          : BellmanManager,
    config      : IngestionConfig,
    stats       : IngestionStats,
    seen_words  : set,
    edge_counter: list,
) -> dict:
    """
    Ingest a single document into the graph.

    Flow:
        text → sentences → (words → characters)

    No paragraph or document nodes. The document is a container
    for sentences — it doesn't persist as a node.
    """
    doc_stats = {"nodes": 0, "edges": 0, "sentences": 0}

    if not text or not text.strip():
        return doc_stats

    sentences    = text_to_sentences(text)
    prev_sent_id = None
    sent_context : list[str] = []

    for sent_text in sentences:
        words = sent_text.split()
        if len(words) < config.min_sentence_words:
            continue

        sent_id, sent_new = get_or_create_node(
            sent_text, Level.SENTENCE,
            graph, normalizer,
            seen_words, sent_context, stats,
        )
        if sent_id is None:
            continue

        doc_stats["sentences"] += 1
        seen_words |= set(re.findall(r'\b\w+\b', sent_text.lower()))

        # Sequential edge: previous sentence → this sentence
        if prev_sent_id is not None:
            prev_node = graph.get_node(prev_sent_id)
            if prev_node:
                create_edge_if_needed(
                    prev_sent_id, sent_id,
                    prev_node.text, sent_text,
                    Level.SENTENCE, graph, normalizer, bm, stats,
                    edge_counter, config.bellman_every,
                )

        prev_sent_id = sent_id
        sent_context = sent_context[-3:] + [sent_text]

        if config.max_level == Level.SENTENCE:
            continue

        # ── Word level ───────────────────────────────────
        words_list   = text_to_words(sent_text)
        prev_word_id = None
        word_context : list[str] = []

        for word in words_list:
            word = word.strip()
            if not word:
                continue

            word_id, word_new = get_or_create_node(
                word, Level.WORD,
                graph, normalizer,
                seen_words, word_context, stats,
            )
            if word_id is None:
                continue

            # Hierarchical edge: word → sentence
            create_edge_if_needed(
                word_id, sent_id,
                word, sent_text,
                Level.WORD, graph, normalizer, bm, stats,
                edge_counter, config.bellman_every,
            )

            # Sequential edge: previous word → this word
            if prev_word_id is not None:
                prev_node = graph.get_node(prev_word_id)
                if prev_node:
                    create_edge_if_needed(
                        prev_word_id, word_id,
                        prev_node.text, word,
                        Level.WORD, graph, normalizer, bm, stats,
                        edge_counter, config.bellman_every,
                    )

            prev_word_id = word_id
            word_context = word_context[-3:] + [word]

            if config.max_level == Level.WORD:
                continue

            # ── Character level ──────────────────────────
            chars        = text_to_characters(word)
            prev_char_id = None

            for char in chars:
                char_id, _ = get_or_create_node(
                    char, Level.CHARACTER,
                    graph, normalizer,
                    set(), [], stats,
                )
                if char_id is None:
                    continue

                # Hierarchical edge: char → word
                create_edge_if_needed(
                    char_id, word_id,
                    char, word,
                    Level.CHARACTER, graph, normalizer, bm, stats,
                    edge_counter, config.bellman_every,
                )

                # Sequential edge: previous char → this char
                if prev_char_id is not None:
                    prev_node = graph.get_node(prev_char_id)
                    if prev_node:
                        create_edge_if_needed(
                            prev_char_id, char_id,
                            prev_node.text, char,
                            Level.CHARACTER, graph, normalizer, bm, stats,
                            edge_counter, config.bellman_every,
                        )

                prev_char_id = char_id

    return doc_stats


# ─────────────────────────────────────────────
# MAIN INGESTION PIPELINE
# ─────────────────────────────────────────────

def ingest(
    dataset_path : str,
    graph        : Graph,
    normalizer   : Normalizer,
    bm           : BellmanManager,
    config       : Optional[IngestionConfig] = None,
) -> IngestionStats:
    """
    Main ingestion entry point for text datasets.
    Reads from dataset_path and ingests all sentences.
    """
    if config is None:
        config = IngestionConfig()

    stats        = IngestionStats()
    t0           = time.time()
    seen_words   : set  = set()
    edge_counter : list = [0]
    batch_count  = 0

    if config.verbose:
        print(f"Starting ingestion from: {dataset_path}")
        print(f"Max level: {config.max_level.name}")
        print(f"Batch size: {config.batch_size} documents\n")

    try:
        for doc_text in read_dataset(dataset_path):
            if config.max_nodes > 0 and stats.nodes_created >= config.max_nodes:
                if config.verbose:
                    print(f"\nNode limit reached: {config.max_nodes}")
                break

            try:
                ingest_document(
                    doc_text, graph, normalizer, bm, config,
                    stats, seen_words, edge_counter,
                )
                stats.documents_read += 1
            except Exception as e:
                stats.errors += 1
                if config.verbose:
                    print(f"  Error on document {stats.documents_read}: {e}")
                continue

            if config.verbose and stats.documents_read % 10 == 0:
                elapsed = time.time() - t0
                rate    = stats.nodes_created / (elapsed + 1e-9)
                print(
                    f"  docs={stats.documents_read:5d}  "
                    f"nodes={stats.nodes_created:6d}  "
                    f"edges={stats.edges_created:6d}  "
                    f"rate={rate:.0f}n/s  "
                    f"elapsed={elapsed:.1f}s"
                )

            batch_count += 1
            if batch_count >= config.batch_size:
                if config.verbose:
                    print(f"\n  Running Bellman sweep...")
                sweep_stats = bm.sweep(verbose=False)
                if config.verbose:
                    print(f"  Sweep: {sweep_stats}\n")
                batch_count = 0

    except KeyboardInterrupt:
        if config.verbose:
            print("\nIngestion interrupted.")

    if config.verbose:
        print("\nFinal Bellman sweep...")
    bm.sweep(verbose=config.verbose)

    stats.elapsed_sec = time.time() - t0

    if config.verbose:
        print(f"\nIngestion complete.\n")
        print(stats.report())
        gs = graph.stats()
        print(f"\nGraph: {gs['node_count']} nodes, {gs['edge_count']} edges")

    return stats


def ingest_text(
    text       : str,
    graph      : Graph,
    normalizer : Normalizer,
    bm         : BellmanManager,
    config     : Optional[IngestionConfig] = None,
) -> IngestionStats:
    """Ingest a raw text string directly. Convenience wrapper."""
    if config is None:
        config = IngestionConfig(verbose=False)

    stats        = IngestionStats()
    t0           = time.time()
    seen_words   : set  = set()
    edge_counter : list = [0]

    ingest_document(text, graph, normalizer, bm, config,
                    stats, seen_words, edge_counter)
    stats.documents_read = 1
    bm.sweep(verbose=False)
    stats.elapsed_sec = time.time() - t0
    return stats


# ─────────────────────────────────────────────
# CONTRASTIVE INGESTION
# ─────────────────────────────────────────────

def ingest_contrastive(
    text       : str,
    graph      : Graph,
    normalizer : Normalizer,
    bm         : BellmanManager,
    config     : Optional[IngestionConfig] = None,
) -> IngestionStats:
    """
    Ingest text as a contrastive (negative) example.

    Nodes enter the graph normally — they may be valid elsewhere.
    Sequential edges between them are CONTRASTIVE — negative reward.
    MCTS learns to avoid this transition pattern.
    """
    if config is None:
        config = IngestionConfig(max_level=Level.SENTENCE, verbose=False)

    stats        = IngestionStats()
    t0           = time.time()
    seen_words   : set  = set()
    edge_counter : list = [0]

    sentences    = text_to_sentences(text)
    prev_sent_id = None
    sent_context : list[str] = []

    for sent_text in sentences:
        if len(sent_text.split()) < config.min_sentence_words:
            continue

        sent_id, _ = get_or_create_node(
            sent_text, Level.SENTENCE,
            graph, normalizer,
            seen_words, sent_context, stats,
        )
        if sent_id is None:
            continue

        if prev_sent_id is not None and prev_sent_id != sent_id:
            if not graph.edge_exists(prev_sent_id, sent_id):
                prev_node = graph.get_node(prev_sent_id)
                if prev_node:
                    create_edge_if_needed(
                        prev_sent_id, sent_id,
                        prev_node.text, sent_text,
                        Level.SENTENCE, graph, normalizer, bm, stats,
                        edge_counter, 10,
                        edge_type=EdgeType.CONTRASTIVE,
                    )

        prev_sent_id = sent_id
        sent_context = sent_context[-2:] + [sent_text]
        seen_words  |= set(re.findall(r'\b\w+\b', sent_text.lower()))

    bm.sweep(verbose=False)
    stats.elapsed_sec    = time.time() - t0
    stats.documents_read = 1
    return stats


def ingest_distillation_batch(
    positive_texts : list[str],
    negative_texts : list[str],
    graph          : Graph,
    normalizer     : Normalizer,
    bm             : BellmanManager,
    config         : Optional[IngestionConfig] = None,
    verbose        : bool = False,
) -> tuple[IngestionStats, IngestionStats]:
    """
    Ingest a batch of agent outputs.
    positive_texts → normal ingestion
    negative_texts → contrastive ingestion
    Returns (positive_stats, negative_stats).
    """
    if config is None:
        config = IngestionConfig(max_level=Level.SENTENCE, verbose=False)

    pos_stats = IngestionStats()
    neg_stats = IngestionStats()
    t0        = time.time()
    seen_words   : set  = set()
    edge_counter : list = [0]

    for text in positive_texts:
        try:
            ingest_document(text, graph, normalizer, bm, config,
                            pos_stats, seen_words, edge_counter)
            pos_stats.documents_read += 1
        except Exception as e:
            pos_stats.errors += 1
            if verbose:
                print(f"  Positive ingestion error: {e}")

    for text in negative_texts:
        try:
            s = ingest_contrastive(text, graph, normalizer, bm, config)
            neg_stats.documents_read += 1
            neg_stats.nodes_created  += s.nodes_created
            neg_stats.edges_created  += s.edges_created
        except Exception as e:
            neg_stats.errors += 1
            if verbose:
                print(f"  Contrastive ingestion error: {e}")

    bm.sweep(verbose=verbose)

    pos_stats.elapsed_sec = time.time() - t0
    neg_stats.elapsed_sec = pos_stats.elapsed_sec

    if verbose:
        print(f"  Positive: {pos_stats.nodes_created} nodes, "
              f"{pos_stats.edges_created} edges")
        print(f"  Contrastive: {neg_stats.nodes_created} nodes, "
              f"{neg_stats.edges_created} contrastive edges")

    return pos_stats, neg_stats


if __name__ == "__main__":
    import tempfile

    print("=== ingestion.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        from core.normalizer import Normalizer
        from core.bellman    import BellmanManager
        import os

        db_path = os.path.join(tmpdir, "graph.db")
        graph   = Graph(db_path)
        norm    = Normalizer()
        bm      = BellmanManager(graph, norm)

        # ── Test 1: character bootstrap ──
        print("── Test 1: complete character ingestion ──\n")
        stats = ingest_complete_characters(graph, norm, bm, verbose=True)

        # ── Test 2: sentence ingestion ──
        print("\n── Test 2: sentence ingestion ──\n")
        sample = (
            "The derivative of x squared is 2x by the power rule. "
            "We apply the rule by bringing down the exponent. "
            "The exponent decreases by one after differentiation."
        )
        config = IngestionConfig(max_level=Level.SENTENCE, verbose=False)
        stats2 = ingest_text(sample, graph, norm, bm, config)
        print(f"Sentences created: {stats2.nodes_created}")

        # ── Test 3: axiom ingestion ──
        print("\n── Test 3: axiom ingestion ──\n")
        axiom_file = os.path.join(tmpdir, "axioms.jsonl")
        with open(axiom_file, "w") as f:
            axioms = [
                {"id": "deriv_def", "axiom": "The derivative measures instantaneous rate of change.", "domain": "calculus", "type": "definition", "depends_on": []},
                {"id": "power_rule", "axiom": "For any function x^n, the derivative is n times x^(n-1).", "domain": "calculus", "type": "rule", "depends_on": ["deriv_def"]},
                {"id": "sum_rule", "axiom": "The derivative of a sum equals the sum of the derivatives.", "domain": "calculus", "type": "rule", "depends_on": ["deriv_def"]},
            ]
            for a in axioms:
                f.write(json.dumps(a) + "\n")

        stats3 = ingest_axioms(axiom_file, graph, norm, bm, verbose=True)

        gs = graph.stats()
        print(f"\nFinal graph: {gs['node_count']} nodes, {gs['edge_count']} edges")
        graph.close()