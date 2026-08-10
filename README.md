# HGLS v0.6 — Hierarchical Generative Learning System

## What This Is

A learning system that builds knowledge from scratch using a single uniform
algorithm at every level. No neural networks. No transformers. No pre-training
on someone else's weights.

Starts from 43 character primitives and builds upward:

```
characters → syllables → words → phrases → schemas → reasoning → meta-reasoning
```

Every structure at every level is connected all the way down to individual
characters and digits. "evaporation" traces to 'e','v','a','p','o','r','a','t','i','o','n'.
"1969" traces to '1','9','6','9'. Nothing floats without a foundation.

---

## What's New in v0.6 — The Memory Graph

The library is no longer a flat store. It is now a living graph.

### Memory Graph
Every structure in the library is a node. Edges connect them in three ways:

- **Compositional** — parent structure contains child. Already implicit in the
  hierarchy, now explicit and queryable. "evaporation" → its syllables → its characters.

- **Co-occurrence** — two structures appear together frequently in learned inputs.
  After 50 co-occurrences the edge becomes permanent. Recurrence IS validation.
  "water" and "evaporates" appear together thousands of times — the connection is real.

- **Novel** — proposed by the tinkering engine, confirmed by LLM. Links distant
  regions of the graph. Genuine discoveries. Rare and high quality.

### Emotional Evaluator
Replaces raw SequenceMatcher as the primary signal. Starts from two primitive
states — positive and negative — and grows finer gradient through experience.

States that emerge over time:
- `familiar_confirming` — dense region, expected connection
- `novel_exciting` — sparse region, structurally coherent
- `almost_right` — high surface similarity, one element wrong
- `deeply_surprising` — links two disconnected dense regions (discovery)
- `curious` — moderate novelty, worth exploring
- `contradicting` — conflicts with an existing strong edge
- `novel_incoherent` — sparse region, no structural support

None of these are programmed. They emerge from the gradient growing
through millions of evaluation cycles.

### Graph Navigator
Reads graph topology before every hypothesis generation cycle.
Selects traversal strategy based on what it finds:

- Dense known region → exploit existing edges, mutate nearby structures
- Sparse frontier → decompose input from lower-level validated pieces
- Bridge between disconnected regions → cross-region connection strategy
- Isolated → rebuild bottom-up from characters

Learns which strategies work for which topology over time.

### Tinkering Engine
Runs every 500 cycles during idle time. Three strategies:

- **Extension** — follow edge chains outward. A→B→C proposes A→C directly.
- **Analogy** — find subgraphs with similar topology in different domains.
  "water cycle" and "economic cycle" have identical input→process→output structure.
- **Compression** — dense clusters that appear together become new primitives
  at the next level up.

Proposals are pre-screened by the emotional evaluator before reaching the LLM.
Only genuinely exciting connections get validated.

### LLM Validator
LLM role shifts from teacher/corrector to validator.
One yes/no per novel connection proposal.
Results cached permanently — never validate the same pair twice.
Rejected connections suppressed to avoid re-proposing.
LLM calls are rare by design — emotional evaluator filters most proposals out.

### The Flywheel
```
Recurrence → base graph edges
Base graph → tinkering proposes novel connections
LLM validates → richer graph
Richer graph → finer emotional gradient
Finer gradient → better tinkering → better novel connections
Better connections → even richer graph
```
It compounds. The system gets smarter at getting smarter.

---

## Architecture

```
hgls/
  persona.py              Identity, values, all curriculum seeds
  curriculum.py           7 stages + 43 primitives (a-z, 0-9, punctuation)
  curriculum_generator.py 55 LLM-generated domains, cached
  structures.py           GenerativeStructure — core unit of knowledge
  library.py              Long-term memory — extreme outcomes only
  tester.py               Gradient scoring + autoregressive token test
  reward.py               Novelty + competence + curiosity + propagation
  generative_unit.py      The algorithm — learn() and generate()
  explorer.py             Internal exploration — random composition
  memory.py               Working memory — cleared per question
  attention.py            Salience-weighted resource allocation
  self_model.py           Agency tracking
  sensory_motor.py        Keyboard-level I/O normalisation
  llm_parent.py           Together AI — teacher and corrector
  auto_driver.py          Autonomous conversation driver
  graph.py                Edge store — the memory graph
  co_occurrence.py        Passive + batch co-occurrence edge builder
  emotional_evaluator.py  Graph-aware gradient signal
  navigator.py            Topology reading + traversal strategy selection
  tinkering.py            Novel connection proposer
  llm_validator.py        Confirms novel connections
  ingestor.py             Dataset ingestion — pre-training and fine-tuning
  wiki_ingest.py          Wikipedia streaming ingestion
  system.py               Main orchestrator — all modules wired
main.py                   Entry point
```

---

## Core Algorithm (Identical at Every Level)

```
1. Input arrives
2. Navigator reads graph topology — dense or sparse?
3. Traversal strategy selected
4. Hypotheses generated using strategy weights
5. Emotional evaluator pre-screens hypotheses
6. Autoregressive token-level test (phrase level and above)
7. Hierarchical reward propagates through all contributing structures
8. Extreme success (≥ 0.99) → stored in library
9. Extreme failure (≤ 0.30) → stored as negative example
10. Mediocre → discarded, leaves no trace
11. Co-occurrence edges observed for learned phrase structures
12. Every 500 cycles: tinkering engine proposes novel connections
13. Emotional pre-screening filters proposals
14. LLM validates remaining proposals
15. Novel edges enter graph
```

---

## File Structure

```
C:\Users\deepa\hgls\
  main.py          ← entry point, lives here
  hgls\            ← the package
    system.py
    wiki_ingest.py
    graph.py
    ... (all other modules)
```

---

## Commands

### Wikipedia Ingestion — run this first
```bash
python main.py --wiki --no-llm
```
Streams all of Wikipedia English bottom-up:
- New words → syllables learned at level 1, word at level 2
- Sentence → learned at level 3
- Co-occurrence edges form passively as phrases are learned
- Resumes automatically if interrupted — just run the same command again

```bash
# Start from scratch
python main.py --wiki --no-llm --reset

# Stop after N sentences
python main.py --wiki --no-llm --max 2000000
```

### Co-occurrence Scan — run once after Wikipedia
```bash
python main.py --scan
```
Scans the full library and builds graph edges from everything already learned.
Run this once when Wikipedia ingestion is complete to bootstrap the graph.

### Autonomous Learning
```bash
python main.py --auto
python main.py --auto --turns 500
```
LLM parent drives continuous learning. Tinkering engine runs every 500 cycles.
Novel connections proposed, validated, and added to the graph.

### Chat with Deepak
```bash
python main.py --chat
```

### Stats
```bash
python main.py --stats
```
Prints full system state — library size, graph edges, emotional evaluator
maturity, tinkering pass rate, LLM validator yes/no ratio.

---

## Dataset Order

### Layer 1 — Language Foundation (levels 2-3)
Wikipedia English — 20M sentences, cleanest factual text. Run first.
C4 / OpenWebText — broader vocabulary, natural prose. Run after Wikipedia.

### Layer 2 — Factual Depth (level 4)
OpenStax textbooks — physics, chemistry, biology, history, economics.
NCERT textbooks — Indian curriculum base for JEE/NEET reasoning.

### Layer 3 — Reasoning Fine-tuning (switch to finetune_qa)
SlimOrca — 500k Chain of Thought examples. The i know → that means → so
pattern is baked in. Massively reinforces what emerged from corrections.
MetaMath + GSM8K — mathematical reasoning, step-by-step solutions.
ARC + TheoremQA — science reasoning.

### Layer 4 — Domain Mastery
JEE / CBSE past papers — hardest domain-specific Q&A.
The Stack — code across all languages.

---

## Ingest Your Own Datasets

```python
from hgls.system import HGLSystem
from hgls.ingestor import DatasetIngestor

system = HGLSystem(use_llm=False)
system.load('deepak_memory.json')
ingestor = DatasetIngestor(system)

# Raw text — pre-training
ingestor.pretrain_texts(your_sentences, level=3)

# Q&A pairs — fine-tuning (teacher corrections, topic-tagged)
ingestor.finetune_qa([
    {'question': 'why does water evaporate?',
     'answer': 'water evaporates because heat gives molecules energy to escape'},
])

# Large datasets — streaming
def my_lines():
    with open('large_dataset.txt') as f:
        for line in f:
            yield line.strip()

ingestor.pretrain_stream(my_lines(), save_every=50000)

system.save('deepak_memory.json')
```

---

## Key Design Principles

1. One algorithm only — reconstruction quality drives everything
2. Only extreme outcomes stored — mediocre leaves no trace
3. Higher levels build from validated lower levels only
4. Successful sub-sequences become new primitives (abstraction)
5. Teacher corrections dominate for their topic (contextual fitness)
6. Co-occurrence edges form from recurrence — recurrence IS validation
7. Novel connections proposed by tinkering, confirmed by LLM
8. Emotional gradient grows from graph richness — not programmed
9. Navigator learns which traversal strategies work where
10. LLM parent and validator both fade as internal signal matures
11. Numbers and letters are equal primitives — both connected bottom-up
12. The same unit that learns also generates — no separate modules

---

## Saves

```
deepak_memory.json        full library (structures + fitness + corrections)
deepak_memory.json.bak    previous good save
deepak_graph.json         graph edges (co-occurrence + compositional + novel)
deepak_validation_cache.json  LLM validation results (never re-validates)
deepak_memory_wiki_checkpoint.json  Wikipedia resume position
```

---

## Current State

- 43 character primitives (a-z, 0-9, punctuation, space)
- 7 curriculum stages
- Autoregressive token-level reward with hierarchical propagation
- Contextual fitness — corrections dominate for their topic
- Memory graph with compositional, co-occurrence, and novel edges
- Emotional evaluator growing from positive/negative primitives
- Graph navigator with learned traversal strategies
- Tinkering engine building novel connections
- LLM validator confirming discoveries
- Wikipedia ingestion running

---

## What's Next

- Multiprocessing with level sharding
- OpenStax + NCERT ingestion after Wikipedia
- SlimOrca fine-tuning for reasoning reinforcement
- Mathematics datasets
- Semantic hypothesis generator using graph regions