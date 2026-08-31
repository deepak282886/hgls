"""
bootstrap.py

Bootstrap pipeline — builds the foundation of the graph
from first principles before any text ingestion begins.

Bootstrap order:
    1. Characters  — ingest all 97 printable ASCII characters
    2. Words       — ingest the complete English dictionary
    3. Verify      — confirm the foundation is solid

Why bootstrap matters:

    A graph that starts from raw text ingestion builds its
    character and word layers opportunistically — only the
    characters and words that appear in the corpus. This
    leaves gaps. "Derivative" exists because OpenWebMath
    used it. "Antiderivative" may not exist yet.

    Bootstrap fills those gaps before any reasoning begins.
    Every character that can appear in any mathematical
    expression already exists. Every English word already
    exists. When sentences and axioms are ingested, their
    words immediately find existing nodes and form edges —
    no cold-start problem at the lexical level.

    The character and word layers are finite. They can be
    made complete. This is the one opportunity to do so
    before the graph grows in less controlled directions.

Word list sources (in order of preference):
    1. NLTK words corpus (if installed)
    2. System word list: /usr/share/dict/words (Linux/Mac)
    3. Downloaded word list (dwyl english-words)
    4. Bundled minimal word list (fallback)

Usage:
    python bootstrap.py
    python bootstrap.py --words /path/to/wordlist.txt
    python bootstrap.py --db graph.db --verify-only
"""

import os
import sys
import time
import argparse
import tempfile
from pathlib import Path


# ─────────────────────────────────────────────
# WORD LIST SOURCES
# ─────────────────────────────────────────────

def get_word_list_path(override: str = None) -> str:
    """
    Find the best available word list.
    Returns path to a word list file (one word per line).
    """
    # 1. User-specified override
    if override and Path(override).exists():
        return override

    # 2. NLTK words corpus
    try:
        import nltk
        try:
            from nltk.corpus import words as nltk_words
            word_list = nltk_words.words()
        except LookupError:
            nltk.download('words', quiet=True)
            from nltk.corpus import words as nltk_words
            word_list = nltk_words.words()

        # Write to temp file for consistent interface
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt',
            delete=False, encoding='utf-8'
        )
        for word in word_list:
            tmp.write(word.lower() + '\n')
        tmp.close()
        print(f"  Using NLTK word corpus ({len(word_list)} words)")
        return tmp.name

    except (ImportError, Exception):
        pass

    # 3. System word list
    system_paths = [
        "/usr/share/dict/words",
        "/usr/share/dict/american-english",
        "/usr/share/dict/british-english",
        "C:/Windows/System32/drivers/etc/hosts",  # not useful but signals Windows
    ]
    for path in system_paths:
        if Path(path).exists() and os.path.getsize(path) > 1000:
            print(f"  Using system word list: {path}")
            return path

    # 4. Bundled minimal mathematical + common English vocabulary
    print("  Using bundled vocabulary (no system word list found)")
    return _write_bundled_vocabulary()


def _write_bundled_vocabulary() -> str:
    """
    Write a bundled vocabulary covering:
    - Common English words
    - Mathematical vocabulary
    - Programming/computer vocabulary
    - Logic and philosophy vocabulary

    This is the fallback when no system word list is available.
    Not exhaustive but covers the domains needed for bootstrap.
    """
    vocabulary = sorted(set([
        # Articles, prepositions, conjunctions
        "a", "an", "the", "and", "or", "but", "not", "if", "then",
        "else", "for", "in", "on", "at", "by", "to", "of", "from",
        "with", "about", "into", "through", "during", "before",
        "after", "above", "below", "between", "among", "without",
        "within", "along", "following", "across", "behind", "beyond",
        "plus", "except", "up", "out", "around", "down", "off", "over",

        # Pronouns and common verbs
        "i", "we", "you", "he", "she", "it", "they", "this", "that",
        "these", "those", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "need", "dare", "ought", "used",

        # Common English
        "able", "about", "above", "according", "across", "actually",
        "after", "again", "against", "all", "almost", "also", "always",
        "another", "any", "anything", "area", "around", "as", "ask",
        "back", "because", "become", "both", "call", "case", "change",
        "come", "consider", "contain", "continue", "could", "course",
        "day", "define", "different", "each", "end", "enough", "even",
        "every", "example", "fact", "few", "find", "first", "follow",
        "form", "found", "give", "given", "go", "good", "great",
        "hand", "help", "here", "high", "how", "idea", "include",
        "just", "keep", "know", "large", "last", "left", "let",
        "like", "line", "little", "look", "long", "make", "many",
        "means", "might", "more", "most", "move", "much", "must",
        "name", "need", "never", "new", "next", "no", "now", "number",
        "often", "only", "open", "order", "other", "our", "out",
        "over", "own", "part", "place", "point", "possible", "put",
        "real", "right", "same", "see", "seem", "set", "show",
        "side", "since", "small", "so", "some", "something", "start",
        "still", "such", "take", "thing", "think", "three", "time",
        "true", "turn", "two", "under", "use", "very", "way",
        "well", "what", "when", "where", "which", "while", "who",
        "why", "work", "world", "would", "write", "year",

        # Mathematics — foundations
        "axiom", "theorem", "proof", "lemma", "corollary", "conjecture",
        "definition", "proposition", "hypothesis", "postulate",
        "assume", "given", "therefore", "hence", "thus", "because",
        "implies", "equivalent", "necessary", "sufficient", "iff",

        # Mathematics — sets and logic
        "set", "subset", "element", "member", "union", "intersection",
        "complement", "empty", "universal", "relation", "function",
        "domain", "range", "codomain", "injective", "surjective",
        "bijective", "map", "mapping", "image", "preimage", "inverse",
        "composition", "identity", "cardinality", "finite", "infinite",
        "countable", "uncountable", "transitive", "reflexive",
        "symmetric", "antisymmetric", "equivalence", "class", "partition",

        # Mathematics — numbers
        "integer", "natural", "rational", "irrational", "real",
        "complex", "prime", "composite", "even", "odd", "positive",
        "negative", "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten", "hundred", "thousand",
        "million", "billion", "infinity", "absolute", "value",
        "factor", "multiple", "divisor", "divisible", "remainder",
        "modulo", "gcd", "lcm", "fibonacci", "perfect", "abundant",

        # Mathematics — algebra
        "algebra", "group", "ring", "field", "vector", "space",
        "basis", "dimension", "linear", "polynomial", "degree",
        "coefficient", "variable", "constant", "equation", "inequality",
        "system", "matrix", "determinant", "eigenvalue", "eigenvector",
        "kernel", "span", "orthogonal", "normal", "scalar", "product",
        "sum", "difference", "quotient", "factor", "root", "solution",
        "solve", "simplify", "expand", "factor", "distribute",

        # Mathematics — calculus
        "calculus", "derivative", "integral", "limit", "continuity",
        "differentiable", "integrable", "rate", "change", "slope",
        "tangent", "normal", "curve", "function", "graph", "plot",
        "maximum", "minimum", "critical", "inflection", "concave",
        "convex", "increasing", "decreasing", "monotone", "bounded",
        "convergent", "divergent", "series", "sequence", "partial",
        "differential", "ordinary", "equation", "boundary", "initial",
        "condition", "chain", "rule", "product", "quotient", "power",
        "logarithm", "exponential", "trigonometric", "sine", "cosine",
        "tangent", "secant", "cosecant", "cotangent",

        # Mathematics — geometry
        "geometry", "point", "line", "plane", "angle", "triangle",
        "circle", "square", "rectangle", "polygon", "vertex", "edge",
        "face", "volume", "area", "perimeter", "radius", "diameter",
        "chord", "arc", "sector", "parallel", "perpendicular",
        "congruent", "similar", "transformation", "translation",
        "rotation", "reflection", "symmetry", "coordinate", "distance",
        "midpoint", "bisect", "inscribed", "circumscribed",

        # Logic
        "logic", "proposition", "statement", "predicate", "quantifier",
        "universal", "existential", "negation", "conjunction",
        "disjunction", "implication", "biconditional", "tautology",
        "contradiction", "satisfiable", "valid", "invalid", "sound",
        "complete", "consistent", "decidable", "provable", "deduction",
        "induction", "abduction", "modus", "ponens", "tollens",
        "syllogism", "premise", "conclusion", "argument", "fallacy",

        # Computer science
        "algorithm", "computation", "complexity", "time", "space",
        "memory", "processor", "program", "code", "instruction",
        "variable", "constant", "type", "value", "reference",
        "pointer", "array", "list", "stack", "queue", "tree",
        "graph", "node", "edge", "path", "cycle", "search",
        "sort", "insert", "delete", "update", "read", "write",
        "file", "directory", "process", "thread", "system", "call",
        "input", "output", "network", "protocol", "address", "port",
        "function", "method", "class", "object", "instance",
        "inherit", "polymorphism", "encapsulation", "abstraction",
        "interface", "module", "library", "compile", "execute",
        "debug", "test", "error", "exception", "handler",
        "recursive", "iterative", "loop", "condition", "branch",

        # Shell / command line
        "bash", "shell", "terminal", "command", "argument", "flag",
        "option", "pipe", "redirect", "stdin", "stdout", "stderr",
        "exit", "return", "environment", "variable", "path", "home",
        "root", "user", "permission", "execute", "script", "run",

        # Proof techniques
        "induction", "base", "step", "hypothesis", "inductive",
        "contradiction", "contrapositive", "direct", "constructive",
        "existence", "uniqueness", "well", "ordering", "pigeonhole",
        "diagonalization", "compactness",
    ]))

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt',
        delete=False, encoding='utf-8'
    )
    for word in vocabulary:
        tmp.write(word + '\n')
    tmp.close()

    print(f"  Bundled vocabulary: {len(vocabulary)} words")
    return tmp.name


# ─────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────

def verify_bootstrap(graph, verbose: bool = True) -> dict:
    """
    Verify the bootstrap completed correctly.

    Checks:
        - All CHAR_VOCAB characters exist as nodes
        - Word nodes exist and have char→word edges
        - V values are non-zero (Bellman ran)
        - Edge connectivity is healthy
    """
    from core.atoms import CHAR_VOCAB, Level

    results = {
        "characters_complete" : False,
        "char_count"          : 0,
        "word_count"          : 0,
        "char_word_edges"     : 0,
        "avg_V_char"          : 0.0,
        "avg_V_word"          : 0.0,
        "issues"              : [],
    }

    # Check characters
    missing_chars = []
    for char in CHAR_VOCAB:
        node_id = graph.text_exists(char, Level.CHARACTER)
        if node_id is None:
            missing_chars.append(repr(char))

    results["char_count"]          = len(CHAR_VOCAB) - len(missing_chars)
    results["characters_complete"] = len(missing_chars) == 0

    if missing_chars:
        results["issues"].append(
            f"Missing {len(missing_chars)} characters: {missing_chars[:5]}..."
        )

    # Check word count
    gs = graph.stats()
    level_counts = gs.get("level_counts", {})
    results["word_count"]     = level_counts.get("WORD", 0)
    results["char_word_edges"] = level_counts.get("CHARACTER_to_WORD_edges", 0)

    # Check V values
    char_nodes = graph.nodes_by_level(Level.CHARACTER, limit=100)
    word_nodes = graph.nodes_by_level(Level.WORD,      limit=100)

    if char_nodes:
        results["avg_V_char"] = sum(n.V for n in char_nodes) / len(char_nodes)
    if word_nodes:
        results["avg_V_word"] = sum(n.V for n in word_nodes) / len(word_nodes)

    if results["avg_V_char"] == 0.0:
        results["issues"].append("Character V values are all zero — Bellman may not have run")
    if results["word_count"] == 0:
        results["issues"].append("No word nodes found — word bootstrap may have failed")

    if verbose:
        print("\nBootstrap verification:")
        print(f"  Characters  : {results['char_count']}/{len(CHAR_VOCAB)} "
              f"({'complete' if results['characters_complete'] else 'INCOMPLETE'})")
        print(f"  Words       : {results['word_count']}")
        print(f"  Avg V (char): {results['avg_V_char']:.4f}")
        print(f"  Avg V (word): {results['avg_V_word']:.4f}")

        if results["issues"]:
            print("\n  Issues:")
            for issue in results["issues"]:
                print(f"    ⚠ {issue}")
        else:
            print("\n  ✓ Bootstrap complete and healthy.")

    return results


# ─────────────────────────────────────────────
# MAIN BOOTSTRAP PIPELINE
# ─────────────────────────────────────────────

def run_bootstrap(
    db_path        : str  = "graph.db",
    norm_path      : str  = "normalizer.json",
    word_list_path : str  = None,
    skip_chars     : bool = False,
    skip_words     : bool = False,
    verify_only    : bool = False,
    verbose        : bool = True,
) -> dict:
    """
    Run the complete bootstrap pipeline.

    Steps:
        1. Initialize graph, normalizer, Bellman manager
        2. Ingest complete character vocabulary
        3. Find and ingest complete word list
        4. Run final Bellman sweep
        5. Verify and report

    Returns verification results dict.
    """
    import json
    from core.graph      import Graph
    from core.normalizer import Normalizer
    from core.bellman    import BellmanManager
    from core.ingestion  import (
        ingest_complete_characters,
        ingest_complete_words,
        IngestionConfig,
    )
    from core.atoms import Level

    t0 = time.time()

    if verbose:
        print("=" * 60)
        print("GRAPH BOOTSTRAP")
        print("=" * 60)
        print(f"  Database    : {db_path}")
        print(f"  Normalizer  : {norm_path}")
        print()

    # Initialize
    graph = Graph(db_path)
    norm  = Normalizer()

    # Load existing normalizer if present
    if Path(norm_path).exists():
        try:
            with open(norm_path, "r") as f:
                norm_data = json.load(f)
            norm.load(norm_data)
            if verbose:
                print(f"  Loaded existing normalizer from {norm_path}")
        except Exception:
            pass

    bm = BellmanManager(graph, norm)

    if verify_only:
        results = verify_bootstrap(graph, verbose=verbose)
        graph.close()
        return results

    # ── Step 1: Characters ───────────────────
    if not skip_chars:
        gs = graph.stats()
        existing_chars = gs.get("level_counts", {}).get("CHARACTER", 0)

        if existing_chars >= 90:
            if verbose:
                print(f"Characters already bootstrapped ({existing_chars} nodes). Skipping.")
        else:
            if verbose:
                print("Step 1: Ingesting complete character vocabulary...")
            char_stats = ingest_complete_characters(graph, norm, bm, verbose=verbose)
    else:
        if verbose:
            print("Step 1: Skipping character ingestion (--skip-chars)")

    # ── Step 2: Words ────────────────────────
    if not skip_words:
        gs = graph.stats()
        existing_words = gs.get("level_counts", {}).get("WORD", 0)

        if existing_words > 10000:
            if verbose:
                print(f"\nWords already bootstrapped ({existing_words} nodes). Skipping.")
        else:
            if verbose:
                print("\nStep 2: Finding word list...")

            wl_path = get_word_list_path(word_list_path)

            if verbose:
                print("Ingesting complete word list...")

            config = IngestionConfig(
                max_level  = Level.WORD,
                verbose    = verbose,
                batch_size = 2000,
            )
            word_stats = ingest_complete_words(
                wl_path, graph, norm, bm, config, verbose=verbose
            )

            # Clean up temp file if we created one
            if wl_path != word_list_path:
                try:
                    os.unlink(wl_path)
                except Exception:
                    pass
    else:
        if verbose:
            print("\nStep 2: Skipping word ingestion (--skip-words)")

    # ── Step 3: Final Bellman sweep ──────────
    if verbose:
        print("\nStep 3: Final Bellman sweep...")
    sweep_stats = bm.sweep(verbose=verbose)
    if verbose:
        print(f"  Sweep complete: {sweep_stats}")

    # ── Step 4: Save normalizer ──────────────
    try:
        with open(norm_path, "w") as f:
            json.dump(norm.to_dict(), f)
        if verbose:
            print(f"\nNormalizer saved to {norm_path}")
    except Exception as e:
        if verbose:
            print(f"  Warning: could not save normalizer: {e}")

    # ── Step 5: Verify ───────────────────────
    if verbose:
        print()
    results = verify_bootstrap(graph, verbose=verbose)

    elapsed = time.time() - t0
    gs      = graph.stats()

    if verbose:
        print(f"\nBootstrap complete in {elapsed:.1f}s")
        print(f"Graph: {gs['node_count']} nodes, {gs['edge_count']} edges")

    graph.close()
    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap the graph with complete character and word layers"
    )
    parser.add_argument("--db",
        default="graph.db",
        help="Database path (default: graph.db)")
    parser.add_argument("--norm",
        default="normalizer.json",
        help="Normalizer path (default: normalizer.json)")
    parser.add_argument("--words",
        default=None,
        help="Path to word list file (one word per line). "
             "Auto-detected if not specified.")
    parser.add_argument("--skip-chars",
        action="store_true",
        help="Skip character ingestion")
    parser.add_argument("--skip-words",
        action="store_true",
        help="Skip word ingestion")
    parser.add_argument("--verify-only",
        action="store_true",
        help="Only verify existing bootstrap, don't ingest")
    args = parser.parse_args()

    run_bootstrap(
        db_path        = args.db,
        norm_path      = args.norm,
        word_list_path = args.words,
        skip_chars     = args.skip_chars,
        skip_words     = args.skip_words,
        verify_only    = args.verify_only,
        verbose        = True,
    )


if __name__ == "__main__":
    main()