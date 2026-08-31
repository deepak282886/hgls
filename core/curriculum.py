"""
curriculum.py

Frontier model conversation loop for sentence generation.

The curriculum agent has one job: generate high-quality sentences
across every domain in the axiom registry. Not answers — reasoning
chains. Not summaries — derivations. The agent thinks out loud and
every thought becomes graph material.

Two modes:

    1. GUIDED — the agent works through a structured curriculum,
       covering each domain systematically. Ensures complete coverage.
       Uses the axiom domain registry as the curriculum source.

    2. GAP-FILLING — the agent queries the graph to find sparse
       regions (low-V nodes, weak connectivity) and generates
       sentences specifically targeting those gaps. Used after
       the graph has enough structure to identify its own weaknesses.

The agent uses Ollama (local) for sentence generation.
For axiom extraction use axiom.py with Together AI instead.
Sentence generation is high-volume and benefits from being local.

Generation target: 50-200 reasoning sentences per domain topic.
Quality over quantity — the reward function filters low-quality output.

Usage:
    python curriculum.py --mode guided --output sentences.jsonl
    python curriculum.py --mode gap --db graph.db --output gaps.jsonl
    python curriculum.py --ingest sentences.jsonl --db graph.db
"""

import os
import re
import json
import time
import random
import argparse
import requests
from pathlib     import Path
from dataclasses import dataclass, field
from typing      import Iterator, Optional


# ─────────────────────────────────────────────
# CURRICULUM CONFIG
# ─────────────────────────────────────────────

OLLAMA_URL     = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 120


@dataclass
class CurriculumConfig:
    """Configuration for the curriculum agent."""
    model           : str   = "deepseek-r1:1.5b"
    mode            : str   = "guided"      # "guided" or "gap"
    sentences_per_topic : int = 20          # target sentences per topic
    temperature     : float = 0.8
    max_tokens      : int   = 512
    verbose         : bool  = True
    reward_threshold: float = 0.0


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

CURRICULUM_SYSTEM_PROMPT = """You are a mathematical reasoning agent building a knowledge graph.

Your task: reason through mathematical and computational topics step by step.
Each sentence you produce should be a clear, standalone statement that
logically follows from the previous one.

Rules:
- Think out loud — show every reasoning step
- Use precise language — define terms before using them
- State WHY each step follows
- Produce complete sentences, not fragments
- Build from simple to complex within each response

Your output becomes training material. Every sentence matters."""


# ─────────────────────────────────────────────
# TOPIC REGISTRY
# Structured topics for guided curriculum.
# More granular than axiom domains — each topic
# produces one focused conversation.
# ─────────────────────────────────────────────

CURRICULUM_TOPICS = [

    # ── Propositional Logic ──
    {"domain": "logic",       "topic": "What are the basic logical connectives and how do truth tables work?"},
    {"domain": "logic",       "topic": "Explain modus ponens and modus tollens with examples."},
    {"domain": "logic",       "topic": "What is a tautology? Give three examples and explain why they are always true."},
    {"domain": "logic",       "topic": "Explain logical equivalence and De Morgan's laws step by step."},
    {"domain": "logic",       "topic": "What is the difference between necessary and sufficient conditions?"},

    # ── Proof Techniques ──
    {"domain": "proof",       "topic": "Walk through a proof by contradiction that the square root of 2 is irrational."},
    {"domain": "proof",       "topic": "Explain mathematical induction and prove the sum of first n integers is n(n+1)/2."},
    {"domain": "proof",       "topic": "What is a direct proof? Prove that the sum of two even numbers is even."},
    {"domain": "proof",       "topic": "Explain proof by contrapositive with a worked example."},
    {"domain": "proof",       "topic": "Prove there are infinitely many prime numbers step by step."},

    # ── Set Theory ──
    {"domain": "sets",        "topic": "Define a set and explain the axiom of extensionality."},
    {"domain": "sets",        "topic": "What are union, intersection, and complement? Explain with Venn diagrams."},
    {"domain": "sets",        "topic": "What is a function? Define domain, codomain, and range precisely."},
    {"domain": "sets",        "topic": "Explain injective, surjective, and bijective functions with examples."},
    {"domain": "sets",        "topic": "What is cardinality? Explain why the rationals are countable."},

    # ── Arithmetic ──
    {"domain": "arithmetic",  "topic": "State and prove the fundamental theorem of arithmetic."},
    {"domain": "arithmetic",  "topic": "Explain the Euclidean algorithm for finding GCD step by step."},
    {"domain": "arithmetic",  "topic": "What is modular arithmetic? Explain with clock arithmetic examples."},
    {"domain": "arithmetic",  "topic": "Prove Fermat's little theorem with an example."},
    {"domain": "arithmetic",  "topic": "What are prime numbers? Prove that 2 is the only even prime."},

    # ── Algebra ──
    {"domain": "algebra",     "topic": "Derive the quadratic formula by completing the square step by step."},
    {"domain": "algebra",     "topic": "Explain what a group is with the four group axioms and an example."},
    {"domain": "algebra",     "topic": "What are the field axioms? Show how real numbers satisfy them."},
    {"domain": "algebra",     "topic": "Factor x squared minus 5x plus 6 completely, explaining each step."},
    {"domain": "algebra",     "topic": "Solve a system of two linear equations by substitution and elimination."},
    {"domain": "algebra",     "topic": "Explain logarithm properties: log(ab), log(a/b), log(a^n) with proofs."},

    # ── Calculus — Limits ──
    {"domain": "calculus",    "topic": "Define the limit of a function using epsilon-delta notation."},
    {"domain": "calculus",    "topic": "Compute the limit of sin(x)/x as x approaches 0, step by step."},
    {"domain": "calculus",    "topic": "What does continuity mean? Give examples of continuous and discontinuous functions."},
    {"domain": "calculus",    "topic": "Explain L'Hopital's rule and when to apply it with worked examples."},

    # ── Calculus — Derivatives ──
    {"domain": "calculus",    "topic": "Define the derivative from first principles using the limit definition."},
    {"domain": "calculus",    "topic": "Prove the power rule for derivatives step by step."},
    {"domain": "calculus",    "topic": "Explain and derive the product rule for differentiation."},
    {"domain": "calculus",    "topic": "State and explain the chain rule with three worked examples."},
    {"domain": "calculus",    "topic": "Find the derivative of x squared plus 3x minus 5, explaining each step."},
    {"domain": "calculus",    "topic": "What are critical points? Find and classify the critical points of x cubed minus 3x."},
    {"domain": "calculus",    "topic": "Explain the derivative of exponential and logarithmic functions."},
    {"domain": "calculus",    "topic": "Derive the derivatives of sine and cosine from first principles."},

    # ── Calculus — Integrals ──
    {"domain": "calculus",    "topic": "State the fundamental theorem of calculus and explain what it means."},
    {"domain": "calculus",    "topic": "Compute the integral of x squared from 0 to 3, step by step."},
    {"domain": "calculus",    "topic": "Explain integration by substitution with two worked examples."},
    {"domain": "calculus",    "topic": "What is integration by parts? Derive the formula and apply it."},

    # ── Linear Algebra ──
    {"domain": "linear",      "topic": "What is a vector space? State the eight vector space axioms with examples."},
    {"domain": "linear",      "topic": "Define linear independence and basis. Find a basis for R^2."},
    {"domain": "linear",      "topic": "Explain matrix multiplication step by step with a 2x2 example."},
    {"domain": "linear",      "topic": "What is a determinant? Compute the determinant of a 2x2 matrix."},
    {"domain": "linear",      "topic": "What are eigenvalues and eigenvectors? Find them for a simple 2x2 matrix."},

    # ── Geometry ──
    {"domain": "geometry",    "topic": "State Euclid's five postulates and explain their significance."},
    {"domain": "geometry",    "topic": "Prove the Pythagorean theorem using area decomposition."},
    {"domain": "geometry",    "topic": "Why does the sum of angles in a triangle equal 180 degrees?"},
    {"domain": "geometry",    "topic": "Derive the formula for the area of a circle."},
    {"domain": "geometry",    "topic": "Explain the distance formula and its derivation from the Pythagorean theorem."},

    # ── Probability ──
    {"domain": "probability", "topic": "State Kolmogorov's axioms of probability and explain each one."},
    {"domain": "probability", "topic": "Explain conditional probability and Bayes theorem with a medical test example."},
    {"domain": "probability", "topic": "Compute the expected value of rolling a fair six-sided die."},
    {"domain": "probability", "topic": "What is the central limit theorem? Explain its significance intuitively."},

    # ── Computation ──
    {"domain": "computation", "topic": "What is an algorithm? Explain correctness and termination."},
    {"domain": "computation", "topic": "Explain Big-O notation with examples of O(1), O(n), O(n^2), O(log n)."},
    {"domain": "computation", "topic": "What is recursion? Write a recursive definition of factorial."},
    {"domain": "computation", "topic": "Explain what a loop invariant is and how to use one to prove correctness."},

    # ── File System ──
    {"domain": "filesystem",  "topic": "What is a file? Explain paths, directories, and the file hierarchy."},
    {"domain": "filesystem",  "topic": "Explain file permissions: read, write, execute for user, group, others."},
    {"domain": "filesystem",  "topic": "What happens when you create, read, write, and delete a file?"},

    # ── Shell ──
    {"domain": "shell",       "topic": "Explain how shell pipes work and give three examples."},
    {"domain": "shell",       "topic": "What is stdin, stdout, and stderr? Explain redirection operators."},
    {"domain": "shell",       "topic": "Explain environment variables and how they affect processes."},
    {"domain": "shell",       "topic": "Walk through what happens when you run a bash script step by step."},
    {"domain": "shell",       "topic": "Explain how to use grep, awk, and sed for text processing."},

    # ── Error Reading ──
    {"domain": "debugging",   "topic": "How do you read a Python traceback and find the root cause?"},
    {"domain": "debugging",   "topic": "Explain the difference between syntax errors, runtime errors, and logic errors."},
    {"domain": "debugging",   "topic": "What does 'permission denied' mean and how do you fix it?"},
    {"domain": "debugging",   "topic": "How do you interpret an exit code? What does exit code 1 vs 127 mean?"},
]


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────

def generate_reasoning(
    topic   : str,
    config  : CurriculumConfig,
) -> str:
    """Generate a reasoning chain for a topic via Ollama."""

    payload = {
        "model"   : config.model,
        "messages": [
            {"role": "system", "content": CURRICULUM_SYSTEM_PROMPT},
            {"role": "user",   "content": topic},
        ],
        "stream"  : False,
        "options" : {
            "temperature"    : config.temperature,
            "num_predict"    : config.max_tokens,
            "top_p"          : 0.9,
            "repeat_penalty" : 1.1,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json    = payload,
            timeout = OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data     = response.json()
        message  = data.get("message", {})

        # DeepSeek-R1 separates thinking from content
        thinking = message.get("thinking", "").strip()
        content  = message.get("content",  "").strip()

        # Thinking is the primary reasoning chain
        if thinking and content:
            return thinking + "\n\n" + content
        return thinking or content

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Run: ollama serve"
        )
    except Exception:
        return ""


def extract_sentences(text: str) -> list[str]:
    """
    Extract clean sentences from generated text.
    Filters out very short fragments, meta-commentary,
    and non-informative lines.
    """
    from core.atoms import text_to_sentences

    # Clean up
    text = re.sub(r'<think>|</think>', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # remove bold
    text = re.sub(r'\*(.+?)\*',     r'\1', text)   # remove italic
    text = re.sub(r'#+\s*',         '',    text)   # remove headers
    text = re.sub(r'\n{3,}',        '\n\n', text)

    sentences = text_to_sentences(text)

    # Filter
    clean = []
    skip_patterns = [
        r'^okay',
        r'^alright',
        r'^so,?\s+i',
        r'^let me',
        r'^i need to',
        r'^i\'m going',
        r'^i think',
        r'^hmm',
        r'^well,',
        r'^\d+\.',               # numbered list markers
        r'^[-•]\s',              # bullet points
    ]
    skip_re = re.compile('|'.join(skip_patterns), re.IGNORECASE)

    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 5:
            continue
        if skip_re.match(sent):
            continue
        clean.append(sent)

    return clean


def score_sentence(text: str, seen_words: set) -> float:
    """Score a single sentence using the reward function."""
    from core.atoms  import Level
    from core.reward import node_reward
    return node_reward(text, Level.SENTENCE,
                       seen_words=seen_words, context=[])


# ─────────────────────────────────────────────
# GUIDED MODE
# Work through the curriculum topic by topic
# ─────────────────────────────────────────────

def run_guided(
    output_path : str,
    config      : CurriculumConfig,
) -> dict:
    """
    Run guided curriculum — generate reasoning chains
    for every topic in CURRICULUM_TOPICS.

    Saves high-reward sentences as positive examples
    and low-reward sentences as negative (for contrastive ingestion).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "topics_done"   : 0,
        "sentences_pos" : 0,
        "sentences_neg" : 0,
        "total_tokens"  : 0,
    }

    if config.verbose:
        print(f"Guided curriculum: {len(CURRICULUM_TOPICS)} topics")
        print(f"Model: {config.model}")
        print(f"Output: {output_path}\n")

    # Shuffle for variety
    topics = CURRICULUM_TOPICS.copy()
    random.shuffle(topics)

    with open(output_path, "w", encoding="utf-8") as out:

        for topic_rec in topics:
            topic  = topic_rec["topic"]
            domain = topic_rec["domain"]

            if config.verbose:
                print(f"[{domain}] {topic[:70]}...")

            try:
                raw = generate_reasoning(topic, config)
            except ConnectionError as e:
                print(f"  Error: {e}")
                break
            except Exception as e:
                print(f"  Generation error: {e}")
                continue

            if not raw or len(raw.strip()) < 50:
                if config.verbose:
                    print(f"  Empty output — skipping")
                continue

            sentences  = extract_sentences(raw)
            seen_words : set = set()
            pos_count  = 0
            neg_count  = 0

            for sent in sentences:
                score = score_sentence(sent, seen_words)
                seen_words |= set(re.findall(r'\b\w+\b', sent.lower()))

                record = {
                    "text"  : sent,
                    "domain": domain,
                    "topic" : topic,
                    "score" : score,
                    "label" : "positive" if score > config.reward_threshold else "negative",
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if score > config.reward_threshold:
                    pos_count += 1
                else:
                    neg_count += 1

            stats["topics_done"]   += 1
            stats["sentences_pos"] += pos_count
            stats["sentences_neg"] += neg_count

            if config.verbose:
                total = pos_count + neg_count
                print(f"  → {total} sentences  "
                      f"pos={pos_count}  neg={neg_count}")

    if config.verbose:
        print(f"\nCurriculum complete.")
        print(f"  Topics done    : {stats['topics_done']}")
        print(f"  Positive sents : {stats['sentences_pos']}")
        print(f"  Negative sents : {stats['sentences_neg']}")
        print(f"  Saved to       : {output_path}")

    return stats


# ─────────────────────────────────────────────
# GAP-FILLING MODE
# Find sparse regions in graph and target them
# ─────────────────────────────────────────────

def find_graph_gaps(
    graph,
    n_samples : int = 20,
) -> list[str]:
    """
    Find sparse regions in the graph.

    Strategy:
        1. Sample low-V sentence nodes (these are weakly connected)
        2. Extract their keywords
        3. Generate topics targeting those keywords

    Returns list of gap-filling topic prompts.
    """
    from core.atoms import Level

    # Get low-V sentence nodes — these are the graph's weak spots
    all_sentence_nodes = graph.nodes_by_level(Level.SENTENCE, limit=5000)

    # Sort by V ascending — lowest V first
    weak_nodes = sorted(all_sentence_nodes, key=lambda n: n.V)[:n_samples]

    prompts = []
    for node in weak_nodes:
        # Extract key concepts from the weak node's text
        words = re.findall(r'\b[a-zA-Z]{4,}\b', node.text)
        stop  = {'that', 'this', 'with', 'from', 'have', 'been', 'they',
                 'will', 'when', 'what', 'which', 'some', 'more', 'also',
                 'into', 'than', 'then', 'over', 'after', 'before'}
        keywords = [w for w in words if w.lower() not in stop][:3]

        if keywords:
            prompt = (
                f"Explain step by step how the following concepts connect "
                f"and what you can derive from them: {', '.join(keywords)}."
            )
            prompts.append(prompt)

    return prompts


def run_gap_filling(
    db_path     : str,
    output_path : str,
    config      : CurriculumConfig,
    n_gaps      : int = 50,
) -> dict:
    """
    Run gap-filling curriculum — find sparse graph regions
    and generate sentences targeting those gaps.
    """
    from core.graph import Graph

    graph = Graph(db_path)

    if config.verbose:
        gs = graph.stats()
        print(f"Gap-filling curriculum")
        print(f"  Graph: {gs['node_count']} nodes, {gs['edge_count']} edges")
        print(f"  Finding {n_gaps} gap topics...")

    gap_prompts = find_graph_gaps(graph, n_samples=n_gaps)
    graph.close()

    if config.verbose:
        print(f"  Found {len(gap_prompts)} gap topics\n")

    # Use guided mode with gap prompts
    # Temporarily replace curriculum topics
    original_topics = CURRICULUM_TOPICS.copy()
    gap_topic_records = [
        {"domain": "gap", "topic": p}
        for p in gap_prompts
    ]

    # Patch and run
    import core.curriculum as self_mod
    self_mod.CURRICULUM_TOPICS = gap_topic_records

    stats = run_guided(output_path, config)

    # Restore
    self_mod.CURRICULUM_TOPICS = original_topics

    return stats


# ─────────────────────────────────────────────
# INGEST GENERATED SENTENCES
# ─────────────────────────────────────────────

def ingest_curriculum_output(
    sentences_path : str,
    db_path        : str  = "graph.db",
    norm_path      : str  = "normalizer.json",
    verbose        : bool = True,
) -> dict:
    """
    Ingest a curriculum output JSONL into the graph.

    Records with label="positive" → normal ingestion
    Records with label="negative" → contrastive ingestion
    """
    import json as json_mod
    from core.graph      import Graph
    from core.normalizer import Normalizer
    from core.bellman    import BellmanManager
    from core.ingestion  import ingest_distillation_batch, IngestionConfig
    from core.atoms      import Level

    graph = Graph(db_path)
    norm  = Normalizer()

    if Path(norm_path).exists():
        try:
            with open(norm_path) as f:
                norm.load(json_mod.load(f))
        except Exception:
            pass

    bm = BellmanManager(graph, norm)

    config = IngestionConfig(
        max_level  = Level.SENTENCE,
        verbose    = False,
        batch_size = 200,
    )

    positive : list[str] = []
    negative : list[str] = []
    batch_size = 200
    total_pos  = 0
    total_neg  = 0

    with open(sentences_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json_mod.loads(line)
                text  = rec.get("text", "")
                label = rec.get("label", "positive")
                if not text:
                    continue
                if label == "positive":
                    positive.append(text)
                else:
                    negative.append(text)

                if len(positive) + len(negative) >= batch_size:
                    pos_s, neg_s = ingest_distillation_batch(
                        positive, negative, graph, norm, bm,
                        config, verbose=False,
                    )
                    total_pos += pos_s.nodes_created
                    total_neg += neg_s.nodes_created
                    positive   = []
                    negative   = []

                    if verbose:
                        gs = graph.stats()
                        print(f"  graph={gs['node_count']} nodes  "
                              f"pos_nodes={total_pos}  neg_nodes={total_neg}")

            except json.JSONDecodeError:
                continue

    # Flush remaining
    if positive or negative:
        ingest_distillation_batch(
            positive, negative, graph, norm, bm, config, verbose=False,
        )

    bm.sweep(verbose=verbose)

    try:
        with open(norm_path, "w") as f:
            json_mod.dump(norm.save(), f)
    except Exception:
        pass

    gs = graph.stats()
    graph.close()

    if verbose:
        print(f"\nIngestion complete.")
        print(f"  Graph: {gs['node_count']} nodes, {gs['edge_count']} edges")

    return {"total_pos": total_pos, "total_neg": total_neg}


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum agent for sentence generation"
    )
    subparsers = parser.add_subparsers(dest="command")

    # run command
    p_run = subparsers.add_parser("run", help="Run curriculum generation")
    p_run.add_argument("--mode",    default="guided",
                       choices=["guided", "gap"],
                       help="guided=structured curriculum, gap=fill graph gaps")
    p_run.add_argument("--output",  default="sentences.jsonl",
                       help="Output JSONL file")
    p_run.add_argument("--model",   default="deepseek-r1:1.5b")
    p_run.add_argument("--db",      default="graph.db",
                       help="Graph DB (required for gap mode)")
    p_run.add_argument("--n-gaps",  type=int, default=50,
                       help="Number of gap topics to generate (gap mode)")
    p_run.add_argument("--temperature", type=float, default=0.8)
    p_run.add_argument("--max-tokens",  type=int,   default=512)
    p_run.add_argument("--threshold",   type=float, default=0.0,
                       help="Reward threshold for positive/negative split")

    # ingest command
    p_ingest = subparsers.add_parser("ingest",
                                     help="Ingest generated sentences into graph")
    p_ingest.add_argument("sentences", help="Path to sentences JSONL file")
    p_ingest.add_argument("--db",   default="graph.db")
    p_ingest.add_argument("--norm", default="normalizer.json")

    # list command
    p_list = subparsers.add_parser("list",
                                   help="List curriculum topics")
    p_list.add_argument("--domain", default=None,
                        help="Filter by domain")

    args = parser.parse_args()

    if args.command == "run":
        config = CurriculumConfig(
            model            = args.model,
            mode             = args.mode,
            temperature      = args.temperature,
            max_tokens       = args.max_tokens,
            reward_threshold = args.threshold,
            verbose          = True,
        )
        if args.mode == "guided":
            run_guided(args.output, config)
        else:
            run_gap_filling(args.db, args.output, config, args.n_gaps)

    elif args.command == "ingest":
        ingest_curriculum_output(
            args.sentences,
            db_path   = args.db,
            norm_path = args.norm,
            verbose   = True,
        )

    elif args.command == "list":
        domain_filter = args.domain
        topics = CURRICULUM_TOPICS
        if domain_filter:
            topics = [t for t in topics if t["domain"] == domain_filter]

        domains = {}
        for t in topics:
            domains.setdefault(t["domain"], []).append(t["topic"])

        for domain, topic_list in sorted(domains.items()):
            print(f"\n[{domain}] ({len(topic_list)} topics)")
            for topic in topic_list:
                print(f"  • {topic}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()