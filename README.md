# hgls

A living graph that learns from conversation.

Not a neural network. No weights. No training loop in the traditional sense.
A continuously running process that builds structure from sequential observation —
the same way a child learns language by absorbing what follows what.

---

## The Idea

Truth is what consistently follows what, across time and levels of abstraction.
Letters follow letters — words emerge. Words follow words — sentences emerge.
Sentences follow sentences — reasoning patterns emerge. The system discovers
these levels itself. Nothing is pre-decided.

The graph is alive in a specific sense:

- **Ingestion** — the membrane. reads input, routes familiar vs novel.
- **Compression** — the metabolism. promotes patterns into stable concepts.
- **Projection** — anticipation. generates hypotheses from known schemas.
- **Decay** — homeostasis. unused structure fades. the graph stays current.

These four processes run continuously. The graph grows upward from observation.
When you stop feeding it data it turns inward — testing its own hypotheses,
exploring uncertain regions, maintaining itself. It never just sits there.

---

## Architecture

```
core/
    atoms.py        — three parameters, all primitives derived from them
    graph.py        — weighted directed hypergraph, SQLite backed
    ingestion.py    — process 1: the membrane
    compression.py  — process 2: the metabolism
    projection.py   — process 3: anticipation
    decay.py        — process 4: homeostasis
    loop.py         — the organism: coordinates all four processes
    memory.py       — working memory for active thought episodes
    thinking.py     — parallel rollouts with schema guidance
    interface.py    — conversation interface

childes_loader.py   — CHILDES CHAT format data loader
run.py              — command line entry point
```

**Three parameters control everything:**

```python
BASE_DECAY            = 0.001   # how fast unused structure fades
COMPRESSION_THRESHOLD = 0.7     # how much recurrence before a pattern becomes a node
BOUNDARY_SENSITIVITY  = 0.3     # how novel something must be to cross the membrane
```

Everything else — levels, schemas, decay rates, learning rate — emerges from
these three and from the data.

---

## Setup

```bash
pip install pylangacq
```

No other dependencies beyond the Python standard library and SQLite.

---

## Training Data

The system learns best from real conversation. The ideal starting point is
the CHILDES database — decades of recorded parent-child conversation transcripts.

1. Register (free) at **https://childes.talkbank.org**
2. Download any corpus as a `.zip` file
3. The Brown corpus is the classic starting point — Adam, Eve, and Sarah
   recorded from age 1.5 to 5

---

## Usage

```bash
# Load a corpus and start chatting
python run.py train Brown.zip
python run.py chat

# Check what the graph has learned
python run.py status

# Load more data on top of existing graph
python run.py train Providence.zip

# Load an entire collection
python run.py train /path/to/Eng-NA/ --recursive

# Limit utterances for quick testing
python run.py train Brown.zip --limit 1000
```

During `chat`, special commands:
```
status   — show graph health
context  — show current working memory
quit     — exit
```

---

## How It Works

**Ingestion** reads a token stream. Each token is tested against the graph.
Familiar tokens reinforce existing nodes and edges. Novel tokens enter a
provisional state and are watched for recurrence. Context is maintained
across turns — what followed what is the primary signal.

**Compression** runs periodically. Provisional nodes that have recurred enough
are promoted to stable. Stable nodes that reliably co-occur get compressed into
level-1 nodes. When multiple level-1 nodes share structure but vary in some
positions — those varying positions become slots. The shared structure becomes
a schema (hyperedge). This is how the system forms reusable patterns.

**Projection** fires whenever a new schema stabilizes. It generates hypothetical
instances by combining known slot values in new ways — predicting what should
exist that hasn't been seen yet. Hypotheticals that get confirmed by observation
strengthen their parent schema. Hypotheticals that get contradicted refine it.

**Decay** runs continuously. Every node and edge loses weight at a rate derived
from its level — higher level means slower decay, because relearning a schema
costs more than relearning a word. Nodes that are activated reset their decay.
Nodes that aren't weaken and eventually dissolve. The graph reflects what is
currently useful, not just what was ever observed.

**Thinking** at query time uses working memory — a temporary scratch pad holding
the current context, active schemas, and attention map. Four rollout strategies
run in parallel: exploitation (follow high-weight paths), exploration (follow
uncertain paths), schema-guided (apply a firing schema's structure), and
analogical (find structurally similar situations). The best rollout becomes
the response. All rollouts update graph weights — every thought is a mini
training episode.

---

## What The Graph Stores

```
Node:       a pattern at some abstraction level
            level 0 = raw token
            level N = pattern of level N-1 nodes
            weight  = observational support [0,1]
            state   = provisional | stable | hypothetical | decaying

Edge:       a transition that has been observed
            weight  = how reliably target follows source

Hyperedge:  a schema — pattern with slots
            structure  = ordered list of (node_id | SLOT)
            instances  = known instances of this schema
            constraints= relationships between slots
```

The graph is append-and-decay — it never hard-deletes, only dissolves through
weight reaching zero. The same pattern can reform if it recurs.

---

## Levels

Levels are not pre-decided. They emerge from compression.

- **Level 0**: raw input tokens — whatever the stream provides
- **Level 1**: patterns of level-0 tokens that reliably co-occur
- **Level 2**: patterns of level-1 patterns
- **Level N**: emerges when level N-1 has enough stable structure to compress from

With conversational data the system typically develops 2-3 levels naturally.
With more structured data (mathematical text, formal reasoning) it goes deeper.

---

## Graph Persistence

The graph lives in `graph.db` — a SQLite file in the working directory.
Every session builds on the last. Stop and restart anytime.

```bash
# Train across multiple sessions
python run.py train Brown.zip        # session 1
python run.py train Providence.zip   # session 2 — adds to same graph
python run.py train Sachs.zip        # session 3 — keeps growing
python run.py chat                   # conversation with everything learned
```

---

## Known Issues

- **Thread safety on Windows**: the `train-chat` command runs training and
  the background loop thread concurrently, which causes SQLite write conflicts
  on Windows. Use `train` then `chat` as separate commands instead.
  Fix: adding a write lock to `graph.py` resolves this.

---

## What This Is Not

- Not a language model. No transformer, no attention mechanism, no embeddings
  from pretraining.
- Not a knowledge graph. Nodes don't have types or ontological roles — only
  structural position and observational weight.
- Not a retrieval system. There is no index and no similarity search — only
  graph traversal guided by edge weights and Bellman values.

It is a compression engine that happens to produce language-like output because
language is the data it compresses. Feed it mathematical reasoning and it
compresses that. Feed it code and it compresses that. The structure that emerges
reflects the structure of whatever it observes.

---

## Roadmap

- Thread-safe SQLite writes (write lock in graph.py)
- Schema slot constraint detection (arithmetic, logical relationships)
- Multi-level compression (level-2 and above nodes)
- Data loaders for StackExchange, GSM8K, Socratic dialogues
- Evaluation metrics beyond graph health (prediction accuracy on held-out turns)
- Visualization of the graph structure as it grows