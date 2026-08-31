"""
axiom.py

Axiom extraction and management.

Extracts structured axiom sets from frontier models via Together AI
and manages the axiom corpus for graph ingestion.

The frontier model's job is not to answer questions.
Its job is to produce the foundational frameworks from which
all answers within a domain can be derived by sequencing.

Axiom extraction prompt strategy:
    "Give me the complete minimal set of axioms for [domain].
     Each axiom must be:
     - A single atomic truth that cannot be derived from the others
     - Stated precisely enough to be applied mechanically
     - Accompanied by its logical dependencies on other axioms
     - Accompanied by its axiom type (definition/rule/constraint/relation/procedure)"

This produces axiom nodes with explicit dependency structure —
exactly what the graph needs for Bellman to propagate correctly
and MCTS to sequence efficiently.

Domains covered:
    Mathematics  : logic, sets, arithmetic, algebra, calculus, geometry
    Computation  : programs, memory, processes, files, shell, networks
    Language     : grammar, meaning, inference, argument structure
    Reasoning    : proof techniques, fallacies, heuristics

Usage:
    # Extract axioms using Together AI
    python axiom.py --domain calculus --output axioms/calculus.jsonl

    # Extract all domains
    python axiom.py --all --output-dir axioms/

    # Ingest existing axiom corpus
    python axiom.py --ingest axioms/ --db graph.db
"""

import os
import re
import json
import time
import argparse
from pathlib     import Path
from dataclasses import dataclass, field
from typing      import Optional


# ─────────────────────────────────────────────
# DOMAIN REGISTRY
# Every domain the agent will extract axioms for.
# Order matters — foundational domains first.
# ─────────────────────────────────────────────

DOMAINS = {

    # ── Logic (most foundational — everything else depends on this) ──
    "propositional_logic": {
        "name"       : "Propositional Logic",
        "description": "The rules governing truth values and logical connectives",
        "foundational": True,
        "depends_on" : [],
    },
    "predicate_logic": {
        "name"       : "Predicate Logic (First-Order Logic)",
        "description": "Quantifiers, predicates, and reasoning about objects",
        "foundational": True,
        "depends_on" : ["propositional_logic"],
    },
    "proof_techniques": {
        "name"       : "Proof Techniques",
        "description": "Direct proof, contradiction, induction, contrapositive, construction",
        "foundational": True,
        "depends_on" : ["propositional_logic", "predicate_logic"],
    },

    # ── Set Theory ──
    "set_theory": {
        "name"       : "Set Theory (ZFC)",
        "description": "The axioms of Zermelo-Fraenkel set theory with choice",
        "foundational": True,
        "depends_on" : ["predicate_logic"],
    },
    "relations_functions": {
        "name"       : "Relations and Functions",
        "description": "Relations, functions, injections, surjections, bijections, composition",
        "foundational": False,
        "depends_on" : ["set_theory"],
    },

    # ── Arithmetic and Number Theory ──
    "peano_arithmetic": {
        "name"       : "Peano Arithmetic",
        "description": "The axioms defining the natural numbers and arithmetic",
        "foundational": True,
        "depends_on" : ["set_theory"],
    },
    "integer_arithmetic": {
        "name"       : "Integer Arithmetic",
        "description": "Properties of integers: divisibility, primes, modular arithmetic",
        "foundational": False,
        "depends_on" : ["peano_arithmetic"],
    },
    "real_numbers": {
        "name"       : "Real Number System",
        "description": "Field axioms, order axioms, completeness axiom",
        "foundational": True,
        "depends_on" : ["peano_arithmetic"],
    },

    # ── Algebra ──
    "group_theory": {
        "name"       : "Group Theory",
        "description": "Axioms of groups, subgroups, homomorphisms, cosets",
        "foundational": False,
        "depends_on" : ["set_theory", "relations_functions"],
    },
    "linear_algebra": {
        "name"       : "Linear Algebra",
        "description": "Vector spaces, linear maps, matrices, eigenvalues",
        "foundational": False,
        "depends_on" : ["real_numbers", "group_theory"],
    },
    "polynomial_algebra": {
        "name"       : "Polynomial Algebra",
        "description": "Polynomials, roots, factoring, the fundamental theorem of algebra",
        "foundational": False,
        "depends_on" : ["real_numbers"],
    },

    # ── Calculus and Analysis ──
    "limits_continuity": {
        "name"       : "Limits and Continuity",
        "description": "Epsilon-delta definitions, limit laws, continuity",
        "foundational": True,
        "depends_on" : ["real_numbers"],
    },
    "differential_calculus": {
        "name"       : "Differential Calculus",
        "description": "Derivatives, differentiation rules, applications",
        "foundational": False,
        "depends_on" : ["limits_continuity"],
    },
    "integral_calculus": {
        "name"       : "Integral Calculus",
        "description": "Integrals, integration techniques, fundamental theorem",
        "foundational": False,
        "depends_on" : ["differential_calculus"],
    },

    # ── Geometry ──
    "euclidean_geometry": {
        "name"       : "Euclidean Geometry",
        "description": "Euclid's five postulates and their consequences",
        "foundational": True,
        "depends_on" : ["real_numbers"],
    },
    "analytic_geometry": {
        "name"       : "Analytic Geometry",
        "description": "Coordinate systems, distance, lines, curves",
        "foundational": False,
        "depends_on" : ["euclidean_geometry", "linear_algebra"],
    },

    # ── Probability and Statistics ──
    "probability_axioms": {
        "name"       : "Probability Theory (Kolmogorov Axioms)",
        "description": "Sample spaces, events, probability measure, conditional probability",
        "foundational": True,
        "depends_on" : ["set_theory", "real_numbers"],
    },

    # ── Computation ──
    "computation_model": {
        "name"       : "Model of Computation",
        "description": "What a program is, what execution means, state and transition",
        "foundational": True,
        "depends_on" : ["predicate_logic"],
    },
    "data_structures": {
        "name"       : "Fundamental Data Structures",
        "description": "Arrays, lists, trees, graphs — definitions and operations",
        "foundational": False,
        "depends_on" : ["computation_model", "set_theory"],
    },
    "algorithms": {
        "name"       : "Algorithm Design Axioms",
        "description": "Correctness, termination, time complexity, space complexity",
        "foundational": False,
        "depends_on" : ["computation_model"],
    },
    "filesystem": {
        "name"       : "File System Axioms",
        "description": "Files, directories, paths, permissions, read/write/create/delete",
        "foundational": True,
        "depends_on" : ["computation_model"],
    },
    "processes": {
        "name"       : "Process and Execution Axioms",
        "description": "Processes, stdin/stdout/stderr, exit codes, signals, environment",
        "foundational": True,
        "depends_on" : ["computation_model", "filesystem"],
    },
    "shell_commands": {
        "name"       : "Shell Command Axioms",
        "description": "Bash syntax, pipes, redirection, variables, scripts, common commands",
        "foundational": False,
        "depends_on" : ["processes", "filesystem"],
    },
    "networking": {
        "name"       : "Network Axioms",
        "description": "IP addresses, TCP/UDP, HTTP, sockets, clients and servers",
        "foundational": False,
        "depends_on" : ["processes"],
    },

    # ── Language and Reasoning ──
    "argument_structure": {
        "name"       : "Argument Structure",
        "description": "Premises, conclusions, valid and invalid argument forms",
        "foundational": True,
        "depends_on" : ["propositional_logic"],
    },
    "reading_inference": {
        "name"       : "Reading and Inference",
        "description": "How to extract claims, identify assumptions, evaluate evidence",
        "foundational": False,
        "depends_on" : ["argument_structure"],
    },
}


# ─────────────────────────────────────────────
# EXTRACTION PROMPT
# ─────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are an axiom extraction specialist. Your job is to identify and state the minimal foundational axioms of a given domain.

Rules for good axioms:
1. Each axiom must be ATOMIC — it cannot be derived from the other axioms in your list
2. Each axiom must be PRECISE — stated formally enough to be applied mechanically
3. Each axiom must be NECESSARY — the domain cannot function without it
4. Each axiom must be GENERATIVE — it should enable deriving many things

Output format — respond ONLY with a JSON array, no other text:
[
  {
    "id": "unique_snake_case_id",
    "axiom": "The precise statement of the axiom.",
    "type": "definition|rule|constraint|relation|procedure",
    "depends_on": ["id_of_axiom_this_depends_on"],
    "generates": "One sentence explaining what can be derived from this axiom.",
    "example": "One concrete example of this axiom in use."
  }
]

Be comprehensive but minimal. Include every axiom that is truly foundational.
Do not include theorems that can be derived from the axioms.
Do not add any text before or after the JSON array."""


def build_extraction_prompt(domain_key: str) -> str:
    """Build the user prompt for extracting axioms of a given domain."""
    domain = DOMAINS[domain_key]
    deps   = domain.get("depends_on", [])

    dep_names = [DOMAINS[d]["name"] for d in deps if d in DOMAINS]
    dep_str   = (
        f"\n\nThis domain builds on: {', '.join(dep_names)}. "
        f"Do not re-state axioms from those domains — only state what is new here."
        if dep_names else ""
    )

    return (
        f"Extract the complete minimal set of foundational axioms for: "
        f"{domain['name']}\n\n"
        f"Domain description: {domain['description']}"
        f"{dep_str}\n\n"
        f"Return ONLY the JSON array of axioms."
    )


# ─────────────────────────────────────────────
# TOGETHER AI CLIENT
# ─────────────────────────────────────────────

@dataclass
class TogetherConfig:
    """Configuration for Together AI extraction."""
    api_key    : str   = ""
    model      : str   = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens : int   = 4096
    temperature: float = 0.2    # Low temperature for precise axiom extraction
    verbose    : bool  = True


def extract_axioms_together(
    domain_key : str,
    config     : TogetherConfig,
) -> list[dict]:
    """
    Extract axioms for a domain using Together AI.

    Uses a frontier model (Llama 3.3 70B by default) for high-quality
    axiom extraction. Low temperature ensures precise, consistent output.

    Returns list of axiom dicts ready for ingestion.
    """
    try:
        from together import Together
    except ImportError:
        raise ImportError(
            "together package required. Install with: pip install together"
        )

    client = Together(api_key=config.api_key)
    domain = DOMAINS[domain_key]

    if config.verbose:
        print(f"  Extracting: {domain['name']}...")

    prompt = build_extraction_prompt(domain_key)

    try:
        response = client.chat.completions.create(
            model    = config.model,
            messages = [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = config.max_tokens,
            temperature = config.temperature,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Strip markdown code blocks if present
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*',     '', content, flags=re.MULTILINE)
        content = content.strip()

        axioms = json.loads(content)

        if not isinstance(axioms, list):
            raise ValueError(f"Expected list, got {type(axioms)}")

        # Add domain metadata to each axiom
        for axiom in axioms:
            axiom["domain"]      = domain_key
            axiom["domain_name"] = domain["name"]
            axiom["foundational"]= domain.get("foundational", False)

        if config.verbose:
            print(f"    → {len(axioms)} axioms extracted")

        return axioms

    except json.JSONDecodeError as e:
        if config.verbose:
            print(f"    ✗ JSON parse error: {e}")
            print(f"    Raw response: {content[:200]}...")
        return []

    except Exception as e:
        if config.verbose:
            print(f"    ✗ Error: {e}")
        return []


# ─────────────────────────────────────────────
# OLLAMA FALLBACK
# If Together AI not available, use local Ollama
# ─────────────────────────────────────────────

def extract_axioms_ollama(
    domain_key : str,
    model      : str  = "deepseek-r1:1.5b",
    verbose    : bool = True,
) -> list[dict]:
    """
    Extract axioms using local Ollama model.
    Lower quality than frontier model but works offline.
    """
    import requests

    domain = DOMAINS[domain_key]
    prompt = build_extraction_prompt(domain_key)

    if verbose:
        print(f"  Extracting via Ollama: {domain['name']}...")

    payload = {
        "model"   : model,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream"  : False,
        "options" : {
            "temperature": 0.2,
            "num_predict": 4096,
        },
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json    = payload,
            timeout = 180,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "").strip()

        # Also check thinking field for DeepSeek-R1
        thinking = response.json().get("message", {}).get("thinking", "")

        # Try to find JSON in the response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if not json_match and thinking:
            json_match = re.search(r'\[.*\]', thinking, re.DOTALL)

        if not json_match:
            if verbose:
                print(f"    ✗ No JSON array found in response")
            return []

        axioms = json.loads(json_match.group())

        if not isinstance(axioms, list):
            return []

        for axiom in axioms:
            axiom["domain"]      = domain_key
            axiom["domain_name"] = domain["name"]
            axiom["foundational"]= domain.get("foundational", False)

        if verbose:
            print(f"    → {len(axioms)} axioms extracted")

        return axioms

    except Exception as e:
        if verbose:
            print(f"    ✗ Error: {e}")
        return []


# ─────────────────────────────────────────────
# CORPUS MANAGEMENT
# ─────────────────────────────────────────────

def save_axioms(axioms: list[dict], output_path: str):
    """Save axioms to a JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for axiom in axioms:
            f.write(json.dumps(axiom, ensure_ascii=False) + "\n")


def load_axioms(path: str) -> list[dict]:
    """Load axioms from a JSONL file or directory of JSONL files."""
    axioms = []
    p = Path(path)

    if p.is_dir():
        for child in sorted(p.glob("*.jsonl")):
            axioms.extend(load_axioms(str(child)))
        return axioms

    if not p.exists():
        return []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    axioms.append(obj)
            except json.JSONDecodeError:
                continue

    return axioms


def validate_axioms(axioms: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Validate axiom list. Returns (valid_axioms, error_messages).
    Checks for required fields and basic structural validity.
    """
    valid  = []
    errors = []

    seen_ids = set()

    for i, axiom in enumerate(axioms):
        # Required fields
        if "axiom" not in axiom and "text" not in axiom and "statement" not in axiom:
            errors.append(f"Axiom {i}: missing 'axiom'/'text'/'statement' field")
            continue

        text = axiom.get("axiom") or axiom.get("text") or axiom.get("statement")
        if not text or len(str(text).strip()) < 10:
            errors.append(f"Axiom {i}: text too short")
            continue

        # Normalize to 'axiom' field
        axiom["axiom"] = str(text).strip()

        # Ensure ID
        if not axiom.get("id"):
            axiom["id"] = f"axiom_{i:04d}"

        # Deduplicate IDs
        if axiom["id"] in seen_ids:
            axiom["id"] = f"{axiom['id']}_{i}"
        seen_ids.add(axiom["id"])

        # Ensure depends_on is a list
        if not isinstance(axiom.get("depends_on"), list):
            axiom["depends_on"] = []

        # Ensure domain
        if not axiom.get("domain"):
            axiom["domain"] = "general"

        # Ensure type
        if axiom.get("type") not in ("definition", "rule", "constraint", "relation", "procedure"):
            axiom["type"] = "rule"

        valid.append(axiom)

    return valid, errors


# ─────────────────────────────────────────────
# EXTRACTION PIPELINE
# ─────────────────────────────────────────────

def extract_all_domains(
    output_dir   : str,
    together_key : str  = None,
    together_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ollama_model : str  = "deepseek-r1:1.5b",
    domains      : list = None,
    delay_sec    : float = 1.0,
    verbose      : bool = True,
) -> dict:
    """
    Extract axioms for all domains and save to output_dir.

    Uses Together AI if api_key provided, else Ollama.
    Extracts in dependency order — foundational domains first.

    Returns dict mapping domain_key → number of axioms extracted.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sort domains by dependency depth
    def dep_depth(key, visited=None):
        if visited is None:
            visited = set()
        if key in visited:
            return 0
        visited.add(key)
        deps = DOMAINS[key].get("depends_on", [])
        if not deps:
            return 0
        return max(dep_depth(d, visited) for d in deps if d in DOMAINS) + 1

    domain_keys = domains or list(DOMAINS.keys())
    domain_keys = sorted(domain_keys, key=lambda k: dep_depth(k))

    use_together = bool(together_key)
    results      = {}

    if verbose:
        print(f"\nExtracting axioms for {len(domain_keys)} domains")
        print(f"Method: {'Together AI (' + together_model + ')' if use_together else 'Ollama (' + ollama_model + ')'}")
        print(f"Output: {output_dir}\n")

    for domain_key in domain_keys:
        if domain_key not in DOMAINS:
            if verbose:
                print(f"  Unknown domain: {domain_key}")
            continue

        output_path = os.path.join(output_dir, f"{domain_key}.jsonl")

        # Skip if already extracted
        if Path(output_path).exists():
            existing = load_axioms(output_path)
            if existing:
                if verbose:
                    print(f"  {domain_key}: already extracted ({len(existing)} axioms)")
                results[domain_key] = len(existing)
                continue

        # Extract
        if use_together:
            config = TogetherConfig(
                api_key     = together_key,
                model       = together_model,
                verbose     = verbose,
            )
            axioms = extract_axioms_together(domain_key, config)
        else:
            axioms = extract_axioms_ollama(domain_key, ollama_model, verbose)

        # Validate
        valid, errors = validate_axioms(axioms)
        if errors and verbose:
            for err in errors[:3]:
                print(f"    Warning: {err}")

        # Save
        if valid:
            save_axioms(valid, output_path)
            results[domain_key] = len(valid)
        else:
            results[domain_key] = 0
            if verbose:
                print(f"    ✗ No valid axioms extracted for {domain_key}")

        # Rate limiting
        if delay_sec > 0:
            time.sleep(delay_sec)

    if verbose:
        total = sum(results.values())
        print(f"\nExtraction complete.")
        print(f"  Domains: {len(results)}")
        print(f"  Total axioms: {total}")
        for domain, count in results.items():
            status = "✓" if count > 0 else "✗"
            print(f"  {status} {domain}: {count} axioms")

    return results


# ─────────────────────────────────────────────
# INGEST AXIOM CORPUS INTO GRAPH
# ─────────────────────────────────────────────

def ingest_axiom_corpus(
    axiom_dir  : str,
    db_path    : str  = "graph.db",
    norm_path  : str  = "normalizer.json",
    verbose    : bool = True,
) -> dict:
    """
    Ingest all axioms from axiom_dir into the graph.

    Ingests in dependency order:
        Foundational domains first (logic, sets, arithmetic)
        Derived domains after (calculus, computation, shell)

    This ensures Bellman propagates correctly — foundational
    axiom nodes get high V before derived axioms reference them.
    """
    import json
    from core.graph      import Graph
    from core.normalizer import Normalizer
    from core.bellman    import BellmanManager
    from core.ingestion  import ingest_axioms

    graph = Graph(db_path)
    norm  = Normalizer()

    if Path(norm_path).exists():
        try:
            with open(norm_path) as f:
                norm.load(json.load(f))
        except Exception:
            pass

    bm = BellmanManager(graph, norm)

    # Sort by dependency depth
    def dep_depth(key, visited=None):
        if visited is None:
            visited = set()
        if key in visited:
            return 0
        visited.add(key)
        deps = DOMAINS.get(key, {}).get("depends_on", [])
        if not deps:
            return 0
        return max(dep_depth(d, visited) for d in deps if d in DOMAINS) + 1

    axiom_files = sorted(
        Path(axiom_dir).glob("*.jsonl"),
        key=lambda p: dep_depth(p.stem),
    )

    results = {"domains": 0, "axioms": 0, "errors": 0}

    for axiom_file in axiom_files:
        if verbose:
            print(f"  Ingesting: {axiom_file.name}")
        try:
            stats = ingest_axioms(str(axiom_file), graph, norm, bm, verbose=False)
            results["domains"] += 1
            results["axioms"]  += stats.nodes_created
        except Exception as e:
            results["errors"] += 1
            if verbose:
                print(f"    Error: {e}")

    # Final sweep
    if verbose:
        print("\nFinal Bellman sweep...")
    bm.sweep(verbose=verbose)

    # Save normalizer
    try:
        with open(norm_path, "w") as f:
            json.dump(norm.save(), f)
    except Exception:
        pass

    gs = graph.stats()
    if verbose:
        print(f"\nAxiom ingestion complete.")
        print(f"  Domains ingested : {results['domains']}")
        print(f"  Axiom nodes      : {results['axioms']}")
        print(f"  Graph total      : {gs['node_count']} nodes, {gs['edge_count']} edges")

    graph.close()
    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract and manage the axiom corpus"
    )
    subparsers = parser.add_subparsers(dest="command")

    # extract command
    p_extract = subparsers.add_parser("extract", help="Extract axioms from frontier model")
    p_extract.add_argument("--domain",  default=None, help="Single domain to extract (default: all)")
    p_extract.add_argument("--output-dir", default="axioms", help="Output directory for axiom JSONL files")
    p_extract.add_argument("--together-key", default=os.environ.get("TOGETHER_API_KEY", ""), help="Together AI API key")
    p_extract.add_argument("--together-model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    p_extract.add_argument("--ollama-model", default="deepseek-r1:1.5b")
    p_extract.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")

    # ingest command
    p_ingest = subparsers.add_parser("ingest", help="Ingest axiom corpus into graph")
    p_ingest.add_argument("--axiom-dir", default="axioms", help="Directory of axiom JSONL files")
    p_ingest.add_argument("--db",   default="graph.db")
    p_ingest.add_argument("--norm", default="normalizer.json")

    # list command
    p_list = subparsers.add_parser("list", help="List available domains")

    # validate command
    p_val = subparsers.add_parser("validate", help="Validate axiom files")
    p_val.add_argument("path", help="JSONL file or directory to validate")

    args = parser.parse_args()

    if args.command == "extract":
        domains = [args.domain] if args.domain else None
        extract_all_domains(
            output_dir    = args.output_dir,
            together_key  = args.together_key,
            together_model= args.together_model,
            ollama_model  = args.ollama_model,
            domains       = domains,
            delay_sec     = args.delay,
            verbose       = True,
        )

    elif args.command == "ingest":
        ingest_axiom_corpus(
            axiom_dir = args.axiom_dir,
            db_path   = args.db,
            norm_path = args.norm,
            verbose   = True,
        )

    elif args.command == "list":
        print(f"\nAvailable domains ({len(DOMAINS)}):\n")
        for key, domain in DOMAINS.items():
            deps = domain.get("depends_on", [])
            dep_str = f"  ← {', '.join(deps)}" if deps else "  (foundational)"
            print(f"  {key:30s} {domain['name']}{dep_str}")

    elif args.command == "validate":
        axioms = load_axioms(args.path)
        valid, errors = validate_axioms(axioms)
        print(f"Loaded    : {len(axioms)} axioms")
        print(f"Valid     : {len(valid)}")
        print(f"Errors    : {len(errors)}")
        for err in errors:
            print(f"  ✗ {err}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()