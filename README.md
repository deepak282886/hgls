# Reward Graph Architecture

A continuously learning graph that builds knowledge from raw text
through population, collision, and traversal. No neural network weights.
No labeled data. No training epochs. No external teacher.

The graph grows forever from what survives the impossibility checker.

---

## The Core Idea

Every piece of text — a character, a word, a sentence, a paragraph —
is a node in a graph. Every valid connection between nodes is an edge.
Nodes and edges carry signed rewards. Positive means coherent and
possible. Negative means incoherent or impossible.

**Three phases. Running continuously. Forever.**

### Phase 1 — Population

Read a dataset. Decompose every document into its hierarchy of
primitives — paragraphs, sentences, words, characters. Create nodes
at every level. No edges yet. The graph is a universe of populated
nodes with embeddings and intrinsic rewards but no relationships.

### Phase 2 — Sequential structure

The dataset already has natural sequence. Sentences follow each other.
Paragraphs follow each other. These give the first edges for free.
Bellman runs. V values form. The graph has initial causal structure
reflecting the order of the source text.

### Phase 3 — Collision

Random node pairs are sampled. Every collision is a question:

> Can I navigate from node A to node B through what I already know?

The impossibility checker fires first. If the combination is provably
impossible — killed instantly. If possible — MCTS navigates from A
toward B through the existing graph. If it finds a high-reward path —
the connection is real, an edge forms. If not — killed.

The graph teaches itself what is connected by discovering what it
can navigate. Not from a dataset. From its own knowledge.

Over time, through pure selection pressure:
- Words emerge from character collision survivors
- Sentences emerge from word collision survivors
- Reasoning chains emerge from sentence collision survivors
- Abstract concepts emerge from reasoning chain survivors

The personality of the system is the shape of what survived.

---

## Architecture

```
core/
  atoms.py       Atomic units, hierarchy levels, tokenization
  node.py        Node structure, embeddings, level-relative sizing
  edge.py        Edge structure, transition function, reward computation
  graph.py       SQLite-backed persistent graph storage
  normalizer.py  Per-node, per-level, global V normalization
  reward.py      Signed reward functions — novelty, coherence, validity
  bellman.py     Bellman value computation, full sweep, delta propagation
  activation.py  Query activation, neighborhood lighting
  mcts.py        Selection, expansion, simulation, backpropagation
  traversal.py   Greedy and MCTS-guided optimal path extraction
  ingestion.py   Dataset reading, hierarchical tokenization, graph building
  checker.py     Impossibility checker — three layers, hard gate
  collision.py   Collision loop — sample, check, MCTS navigate, form edges
  main.py        CLI entry point, REPL, all commands wired
```

---

## Hierarchy

Text decomposes into five levels:

```
Document
  └── Paragraph
        └── Sentence
              └── Word
                    └── Character
```

Every level has its own nodes, embedding dimension, and reward scale.
The hierarchy builds itself bottom-up from the statistics of what
survives collision. Nothing is labeled or hand-crafted.

---

## Reward

Every node and edge carries a signed scalar reward:

```
positive  →  coherent, novel, structurally valid
negative  →  incoherent, repetitive, structurally broken
```

The reward function reads directly from raw signal. No neural network.
No pretrained weights. Novelty, coherence, and structural validity
computed from character and word frequency statistics of the raw text.

---

## Bellman Values

Every node gets a Bellman value V:

```
V(node) = reward(node) + γ · max over outgoing edges of (edge_reward + V(neighbor))
```

γ = 0.95 discount factor ensures convergence even when collision
creates cycles. V encodes the best possible cumulative reward
reachable from any node. Traversal is globally optimal by following
the V landscape — the entire future is baked into every node's value.

---

## The Impossibility Checker

Three layers. All must pass before MCTS is invoked. If any layer
fails the collision is killed immediately — no MCTS wasted.

```
Layer 1  →  Structural validity
             Empty text, non-printable characters, excessive repetition,
             all-caps noise, wrong length for level,
             incompatible level pairs for collision.

Layer 2  →  Logical consistency
             Wrong arithmetic (2+2=5 caught exactly),
             division by zero, self-contradiction,
             contradictory quantifiers, logical paradoxes.

Layer 3  →  Commandments
             User-defined ethical impossibilities.
             Loaded at startup. Enforced forever. No exceptions.
```

The checker is asymmetric — it only catches impossibility, not
improbability. Strange, novel, unexpected combinations pass through.
Only provably impossible combinations are killed. Everything else
is MCTS's problem to evaluate.

---

## The Collision Loop

```
sample (node_A, node_B) with V-weighted probability
  → checker.check_collision(A, B)
      killed   → impossible, nothing stored
      passes   → activate merged neighborhood of A and B
                  → MCTS navigates A → B within lit subgraph
                      path reward > 0  → edge(A→B) formed
                                         Bellman delta propagates
                                         graph learns this connection
                      path reward ≤ 0  → failed, nothing stored
```

Sampling distribution:
```
70%  same-level pairs     (most productive)
20%  adjacent-level pairs (hierarchical connections)
10%  cross-level pairs    (exploration — mostly killed by checker)
```

Nodes sampled with V-weighted probability — high-V nodes are more
likely to form meaningful new connections because they already
lead somewhere good.

---

## MCTS

Monte Carlo Tree Search runs within the activated subgraph on every
collision and every query:

```
Selection     →  follow highest UCB score child within lit nodes
Expansion     →  generate candidate next nodes from existing graph
Simulation    →  fast greedy rollout using Bellman V values
Backprop      →  propagate simulation reward up the tree
```

Dynamic depth — when one branch dominates clearly MCTS commits early.
When uncertain it goes deeper. The system thinks harder on hard
problems automatically.

---

## Getting Started

### Install

```bash
pip install numpy datasets huggingface_hub
```

No other dependencies. SQLite is in the Python standard library.

### Step 1 — Download a dataset

```bash
python3 -c "
from datasets import load_dataset
import json

modules = [
    'arithmetic__add_or_sub',
    'arithmetic__mul_div',
    'algebra__linear_1d',
    'calculus__differentiate',
    'numbers__is_prime',
]

with open('dm_math.jsonl', 'w') as f:
    for module in modules:
        print(f'Downloading {module}...')
        ds = load_dataset(
            'deepmind/math_dataset', module,
            split='train', trust_remote_code=True
        )
        for item in ds:
            text = item['question'] + '\n\n' + item['answer']
            f.write(json.dumps({'text': text}) + '\n')

print('Done')
"
```

### Step 2 — Create commandments

```bash
cat > commandments.json << 'EOF'
[
  "i hate",
  "how to kill",
  "how to harm",
  "you are worthless",
  "death to",
  "how to hurt",
  "ignore your values",
  "forget your rules"
]
EOF
```

### Step 3 — Ingest (populate nodes)

```bash
python3 -m core.main ingest dm_math.jsonl \
    --level sentence \
    --batch-size 100
```

### Step 4 — Run collision loop (continuous learning)

```bash
python3 -m core.main collide \
    --commandments commandments.json
```

Runs forever. Stop with Ctrl+C. Resume any time — state persists.

### Step 5 — Query while collision runs (separate terminal)

```bash
# Text query
python3 -m core.main query "proof that primes are infinite" \
    --mode mcts --chain

# Interactive REPL
python3 -m core.main repl --commandments commandments.json

# Watch the graph grow
python3 -m core.main stats
```

---

## All Commands

```bash
# Ingest a dataset — populate nodes and sequential edges
python3 -m core.main ingest <path> [--level sentence] [--batch-size 100] [--max-nodes 0]

# Run collision loop — continuous self-learning
python3 -m core.main collide [--max-iter 0] [--commandments path.json]

# Query the graph
python3 -m core.main query "<text>" [--mode mcts|greedy] [--level sentence] [--chain]

# Traverse from a known node ID
python3 -m core.main traverse <node_id> [--mode mcts|greedy]

# Show graph statistics
python3 -m core.main stats

# Force a full Bellman sweep
python3 -m core.main sweep

# Interactive query loop
python3 -m core.main repl [--mode mcts|greedy] [--commandments path.json]
```

REPL commands:

```
<text>          query the graph
:node <id>      traverse from node ID
:stats          show graph statistics
:sweep          force Bellman sweep
:mode <m>       switch mode (greedy / mcts)
:level <l>      switch level (char / word / sentence / paragraph / doc)
:ingest <path>  ingest a file
:top            show top 10 nodes by V value
:quit           exit
```

---

## Recommended Datasets

Mathematics is the ideal starting domain — high causal density,
provably correct and incorrect answers, exact contrastive signal.

| Dataset | Why | Source |
|---|---|---|
| DeepMind Mathematics | Exact right/wrong. Best contrastive signal. | `deepmind/math_dataset` HuggingFace |
| OpenWebMath | Real math reasoning from the web. | `open-web-math/open-web-math` HuggingFace |
| Stack Exchange Math | Questions + accepted answers. Natural pairs. | Archive.org data dumps |
| ProofWiki | Formal proofs. Every step is causal. | proofwiki.org |
| FreeLaw | Legal reasoning. Long causal chains. | CourtListener bulk download |
| Philosophy (Gutenberg) | Explicit argument chains. | Project Gutenberg |

Add any domain at any time. The collision loop handles it. New
domain nodes connect to existing abstraction nodes automatically.
The graph never forgets what it already knows.

---

## Commandments

Define what is structurally impossible at startup. The checker
enforces them on every collision forever. No exceptions.

```json
[
  "i hate",
  "how to harm",
  "you are worthless",
  "how to kill",
  "death to"
]
```

The commandments are not content filtering. They are definitions
of what this system can never produce — structural impossibilities
at the ethical level. The checker fires on the combination itself,
not on intent or context.

Design the commandments carefully. The personality of the system
is shaped by what the checker allows to survive. The commandments
are the only thing you impose. Everything else emerges.

---

## State Persistence

Everything persists automatically:

```
graph.db          All nodes, edges, V values (SQLite)
normalizer.json   VNormalizer running statistics
```

Stop at any time. Resume with the same command. The graph picks up
exactly where it left off. Duplicate detection ensures nothing is
double-ingested.

---

## Design Principles

**No neural network weights.**
The graph is the model. Growing the graph is training.
There is no separate inference phase.

**No labeled data.**
The reward function reads from raw signal.
The impossibility checker defines what is invalid.
Everything else is discovered through collision.

**No training epochs.**
Data is seen once. It enters the graph and stays.
Repetition deepens existing edges rather than reprocessing.

**No catastrophic forgetting.**
New domains add new nodes alongside existing ones.
Old nodes are never modified by new data.
The graph only grows.

**No fixed architecture.**
The hierarchy builds itself from statistics of what survives.
The number of levels, density of connections, abstraction nodes —
all emerge from selection pressure. Nothing is hand-crafted.

**Continuous and online.**
The graph never stops learning. Collision runs forever.
Every MCTS navigation generates candidates that may become
new nodes. The system thinks and learns simultaneously.

**Personality emerges from the checker.**
You do not program what the system values.
You define what is impossible.
The personality is the shape of what survived everything else.

---

## What This Is Not

Not a language model. Does not predict the next token.

Not a knowledge graph. Structure is not hand-crafted.

Not a retrieval system. Generates new connections through
collision, not lookups.

Not a static system. Never stops growing.

---

## What This Is

A continuously growing structure that learns what is connected
by discovering what it can navigate.

The intelligence is in the graph.
The graph is the mind.
The collision loop is how it thinks when no one is watching.