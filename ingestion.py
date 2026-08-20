"""
ingestion.py

Ingestion pipeline — reads raw text datasets and builds the graph.

Flow for each document:

    Document text
        → paragraphs  (Level 3 nodes)
            → sentences  (Level 2 nodes)
                → words      (Level 1 nodes)
                    → characters (Level 0 nodes)

At each level:
    1. Check if node already exists (deduplication)
    2. Create node if new
    3. Create sequential edge to previous node at same level
    4. Create hierarchical edge to parent node one level up
    5. Run normalizer on new node and new edges
    6. Trigger Bellman delta propagation

Deduplication is critical — the graph never stores the same
text unit twice. Identical sentences seen a thousand times
strengthen edges rather than creating a thousand duplicate nodes.

Supported dataset formats:
    - Plain text (.txt) — one document, newlines separate paragraphs
    - JSONL (.jsonl)    — one JSON object per line, reads "text" field
    - Directory         — recursively reads all .txt and .jsonl files

Configurable ingestion depth:
    max_level = SENTENCE  → sentence + paragraph + document nodes only
    max_level = WORD      → also create word nodes (larger graph)
    max_level = CHARACTER → all levels (very large, use carefully)

Default: max_level = SENTENCE (best balance for text reasoning)
"""

import os
import json
import time
from pathlib    import Path
from typing     import Iterator, Optional
from dataclasses import dataclass, field

from core.atoms      import Level, decompose, text_to_paragraphs, text_to_sentences
from core.graph      import Graph
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

    max_level         : deepest level to create nodes at
                        SENTENCE = sentence+paragraph+document
                        WORD     = also words
                        CHARACTER= all levels
    batch_size        : documents per batch before Bellman sweep
    max_nodes         : stop after this many total nodes (0 = unlimited)
    min_sentence_words: skip sentences shorter than this
    min_paragraph_sents: skip paragraphs with fewer sentences
    verbose           : print progress
    bellman_every     : run delta propagation every N edges
                        (full sweep handled by BellmanManager)
    """
    max_level            : Level = Level.SENTENCE
    batch_size           : int   = 100
    max_nodes            : int   = 0
    min_sentence_words   : int   = 3
    min_paragraph_sents  : int   = 1
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

    # Per-level counts
    level_nodes     : dict  = field(default_factory=lambda: {
        lv.name: 0 for lv in Level
    })

    def report(self) -> str:
        rate = self.nodes_created / (self.elapsed_sec + 1e-9)
        lines = [
            f"Documents read   : {self.documents_read}",
            f"Nodes created    : {self.nodes_created}  "
            f"({rate:.1f}/s)",
            f"Nodes skipped    : {self.nodes_skipped}  "
            f"(duplicates)",
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
    """
    Read a plain text file.
    Each double-newline-separated block is one document.
    Single documents with no double-newlines are yielded whole.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Split on double newlines — each chunk is a document
    docs = [d.strip() for d in content.split("\n\n") if d.strip()]
    if not docs:
        if content.strip():
            yield content.strip()
        return

    # If only one chunk and it's short, yield as one document
    if len(docs) == 1:
        yield docs[0]
        return

    # Group into ~500 word documents
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
    Read a JSONL file.
    Reads the "text" field from each JSON line.
    Also handles "content" and "problem"+"solution" fields
    (for math datasets like AMPS/OpenWebMath).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f):
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
                    # Math dataset format
                    yield f"{obj['problem']}\n\n{obj['solution']}"
                elif "question" in obj and "answer" in obj:
                    yield f"{obj['question']}\n\n{obj['answer']}"
                elif isinstance(obj, str):
                    yield obj
            except json.JSONDecodeError:
                continue


def read_dataset(path: str) -> Iterator[str]:
    """
    Read any supported dataset format.
    Dispatches based on file extension.
    Recursively reads directories.
    """
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
        # Try as plain text
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

    Returns (node_id, is_new).
    is_new = False means a duplicate was found.
    """
    if not text or not text.strip():
        return None, False

    # Deduplication check
    existing_id = graph.text_exists(text.strip(), level)
    if existing_id is not None:
        stats.nodes_skipped += 1
        return existing_id, False

    # Compute reward from raw signal
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
    source_id  : int,
    target_id  : int,
    source_text: str,
    target_text: str,
    level      : Level,
    graph      : Graph,
    normalizer : Normalizer,
    bm         : BellmanManager,
    stats      : IngestionStats,
    edge_count_ref: list,   # mutable counter [n_edges_since_bellman]
    bellman_every : int,
):
    """
    Create an edge between two nodes if it does not exist.
    Triggers Bellman delta propagation every bellman_every edges.
    """
    if source_id == target_id:
        return
    if graph.edge_exists(source_id, target_id):
        return

    graph.add_edge(source_id, target_id,
                   source_text, target_text, level)
    normalizer.on_edge_added(source_id, graph)
    bm.on_edge_added(source_id, target_id)
    stats.edges_created += 1
    edge_count_ref[0]   += 1


# ─────────────────────────────────────────────
# INGEST ONE DOCUMENT
# ─────────────────────────────────────────────

def ingest_document(
    text       : str,
    graph      : Graph,
    normalizer : Normalizer,
    bm         : BellmanManager,
    config     : IngestionConfig,
    stats      : IngestionStats,
    seen_words : set,
    edge_counter: list,
) -> dict:
    """
    Ingest a single document into the graph.

    Decomposes text into the configured hierarchy levels,
    creates nodes and edges at each level.

    Returns dict with counts for this document.
    """
    doc_stats = {"nodes": 0, "edges": 0, "sentences": 0}

    if not text or not text.strip():
        return doc_stats

    # ── Document level ───────────────────────────────────────
    doc_id, _ = get_or_create_node(
        text, Level.DOCUMENT, graph, normalizer,
        seen_words, [], stats
    )

    # ── Paragraph level ──────────────────────────────────────
    paragraphs   = text_to_paragraphs(text)
    prev_para_id = None
    para_context = []

    for para_text in paragraphs:
        if not para_text.strip():
            continue

        para_id, para_new = get_or_create_node(
            para_text, Level.PARAGRAPH, graph, normalizer,
            seen_words, para_context, stats
        )
        if para_id is None:
            continue

        # Hierarchical edge: paragraph → document
        if doc_id is not None:
            create_edge_if_needed(
                para_id, doc_id,
                para_text, text,
                Level.PARAGRAPH, graph, normalizer, bm, stats,
                edge_counter, config.bellman_every,
            )

        # Sequential edge: previous paragraph → this one
        if prev_para_id is not None:
            prev_node = graph.get_node(prev_para_id)
            if prev_node:
                create_edge_if_needed(
                    prev_para_id, para_id,
                    prev_node.text, para_text,
                    Level.PARAGRAPH, graph, normalizer, bm, stats,
                    edge_counter, config.bellman_every,
                )

        prev_para_id = para_id
        para_context = para_context[-2:] + [para_text]

        if config.max_level == Level.PARAGRAPH:
            continue

        # ── Sentence level ───────────────────────────────────
        sentences    = text_to_sentences(para_text)
        prev_sent_id = None
        sent_context = []

        for sent_text in sentences:
            words = sent_text.split()
            if len(words) < config.min_sentence_words:
                continue

            sent_id, sent_new = get_or_create_node(
                sent_text, Level.SENTENCE, graph, normalizer,
                seen_words, sent_context, stats
            )
            if sent_id is None:
                continue

            doc_stats["sentences"] += 1

            # Update seen words from this sentence
            import re
            seen_words |= set(re.findall(r'\b\w+\b', sent_text.lower()))

            # Hierarchical edge: sentence → paragraph
            create_edge_if_needed(
                sent_id, para_id,
                sent_text, para_text,
                Level.SENTENCE, graph, normalizer, bm, stats,
                edge_counter, config.bellman_every,
            )

            # Sequential edge: previous sentence → this one
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
            words_list   = sent_text.split()
            prev_word_id = None
            word_context = []

            for word in words_list:
                word = word.strip()
                if not word:
                    continue

                word_id, word_new = get_or_create_node(
                    word, Level.WORD, graph, normalizer,
                    seen_words, word_context, stats
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

                # Sequential edge: previous word → this one
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
                chars        = list(word)
                prev_char_id = None

                for char in chars:
                    if not char.strip():
                        continue

                    char_id, _ = get_or_create_node(
                        char, Level.CHARACTER, graph, normalizer,
                        set(), [], stats
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

                    # Sequential edge: previous char → this one
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
    Main ingestion entry point.

    Reads a dataset from dataset_path and ingests all text
    into the graph according to config.

    Runs Bellman sweeps periodically via BellmanManager.

    Returns IngestionStats with full run diagnostics.
    """
    if config is None:
        config = IngestionConfig()

    stats        = IngestionStats()
    t0           = time.time()
    seen_words   = set()
    edge_counter = [0]   # mutable list so nested functions can modify
    batch_count  = 0

    if config.verbose:
        print(f"Starting ingestion from: {dataset_path}")
        print(f"Max level: {config.max_level.name}")
        print(f"Batch size: {config.batch_size} documents")
        print()

    try:
        for doc_text in read_dataset(dataset_path):
            # Check node limit
            if config.max_nodes > 0 and stats.nodes_created >= config.max_nodes:
                if config.verbose:
                    print(f"\nNode limit reached: {config.max_nodes}")
                break

            try:
                doc_stats = ingest_document(
                    doc_text, graph, normalizer, bm, config,
                    stats, seen_words, edge_counter
                )
                stats.documents_read += 1

            except Exception as e:
                stats.errors += 1
                if config.verbose:
                    print(f"  Error on document {stats.documents_read}: {e}")
                continue

            # Progress reporting
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

            # Batch Bellman sweep
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
            print("\nIngestion interrupted by user.")

    # Final Bellman sweep
    if config.verbose:
        print("\nFinal Bellman sweep...")
    bm.sweep(verbose=config.verbose)

    stats.elapsed_sec = time.time() - t0

    if config.verbose:
        print(f"\nIngestion complete.\n")
        print(stats.report())
        gs = graph.stats()
        print(f"\nGraph state:")
        print(f"  Total nodes: {gs['node_count']}")
        print(f"  Total edges: {gs['edge_count']}")
        print(f"  Avg V:       {gs['avg_V']:.3f}")
        print(f"  Max V:       {gs['max_V']:.3f}")

    return stats


# ─────────────────────────────────────────────
# INGEST FROM STRING (for testing / small inputs)
# ─────────────────────────────────────────────

def ingest_text(
    text       : str,
    graph      : Graph,
    normalizer : Normalizer,
    bm         : BellmanManager,
    config     : Optional[IngestionConfig] = None,
) -> IngestionStats:
    """
    Ingest a raw text string directly.
    Convenience wrapper for single documents or test inputs.
    """
    if config is None:
        config = IngestionConfig(verbose=False)

    stats        = IngestionStats()
    t0           = time.time()
    seen_words   = set()
    edge_counter = [0]

    ingest_document(
        text, graph, normalizer, bm, config,
        stats, seen_words, edge_counter
    )
    stats.documents_read = 1
    bm.sweep(verbose=False)
    stats.elapsed_sec = time.time() - t0
    return stats


if __name__ == "__main__":
    import tempfile, os

    print("=== ingestion.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        from core.normalizer import Normalizer
        from core.bellman    import BellmanManager
        from core.traversal  import traverse_from_node

        db_path = os.path.join(tmpdir, "graph.db")
        graph   = Graph(db_path)
        norm    = Normalizer()
        bm      = BellmanManager(graph, norm)

        # ── Test 1: ingest raw text ──────────────────────
        print("── Test 1: ingest raw text (sentence level) ──\n")

        sample_text = """
Detective Maria arrived at the abandoned warehouse at midnight.
The building smelled of rust and old machinery.
She found a trail of footprints leading to the back room.
The footprints were fresh, made within the last hour.

A broken window let in cold night air from the alley.
In the corner she discovered a locked metal box.
The box had three combination locks each with four digits.

Her flashlight flickered and died.
She replaced the batteries and continued her search.
Scrawled on the wall were the numbers 1987, 2034, and 0451.
She entered the codes and the box clicked open.

Inside were photographs and a folded letter.
The letter named someone she recognized from the case files.
Maria called her partner with the new evidence.
The mystery was finally close to being solved.
""".strip()

        config = IngestionConfig(
            max_level  = Level.SENTENCE,
            verbose    = True,
            batch_size = 10,
        )

        stats = ingest_text(sample_text, graph, norm, bm, config)

        print(f"\nIngestion stats:")
        print(stats.report())

        gs = graph.stats()
        print(f"\nGraph state:")
        for k, v in gs.items():
            if k != "db_path":
                print(f"  {k}: {v}")

        # ── Test 2: traverse after ingestion ─────────────
        print("\n── Test 2: traverse from first node ──\n")
        top = graph.top_nodes(n=1, level=Level.SENTENCE)
        if top:
            start = top[0]
            print(f"Starting from: {start.text}")
            from core.traversal import traverse_from_node
            result = traverse_from_node(
                start.node_id, graph, norm,
                mode="greedy", max_depth=8,
            )
            print(f"\nOptimal path ({result.summary()}):")
            for i, text in enumerate(result.texts):
                print(f"  [{i}] {text}")

        # ── Test 3: JSONL ingestion ───────────────────────
        print("\n── Test 3: JSONL dataset ingestion ──\n")
        jsonl_path = os.path.join(tmpdir, "test.jsonl")
        with open(jsonl_path, "w") as f:
            examples = [
                {"text": "The area of a circle is pi times the radius squared."},
                {"text": "To find the derivative, apply the power rule."},
                {"text": "The sum of angles in a triangle is 180 degrees."},
                {"problem": "What is 2 + 2?", "solution": "The answer is 4. Addition combines two quantities."},
            ]
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        graph2  = Graph(os.path.join(tmpdir, "graph2.db"))
        norm2   = Normalizer()
        bm2     = BellmanManager(graph2, norm2)

        config2 = IngestionConfig(
            max_level = Level.SENTENCE,
            verbose   = True,
            batch_size= 10,
        )

        stats2 = ingest(jsonl_path, graph2, norm2, bm2, config2)

        graph.close()
        graph2.close()