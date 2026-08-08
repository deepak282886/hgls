# HGLS v0.5 — Hierarchical Generative Learning System

## What This Is

A learning system that builds knowledge from scratch using a single uniform
algorithm at every level. No neural networks. No transformers. No pre-training
on someone else's weights.

It starts from 43 character primitives and builds upward:

```
characters → syllables → words → phrases → schemas → reasoning → meta-reasoning
```

Every structure at every level is connected all the way down to individual
characters. "evaporation" traces to 'e','v','a','p','o','r','a','t','i','o','n'.
"42" traces to '4','2'. Nothing floats without a foundation.

---

## Architecture

```
hgls/
  persona.py              Identity, values, all curriculum seeds
  curriculum.py           7 developmental stages + 43 character primitives
  curriculum_generator.py 55 LLM-generated domains, cached
  structures.py           GenerativeStructure — core unit of knowledge
  library.py              Long-term memory — extreme outcomes only
  tester.py               ExtremeTester — gradient scoring + autoregressive test
  reward.py               Novelty + competence + curiosity + hierarchical propagation
  generative_unit.py      The algorithm — learn() and generate()
  explorer.py             Internal exploration — random composition
  memory.py               Working memory — cleared per question
  attention.py            Salience-weighted resource allocation
  self_model.py           Tracks self-generated vs external structures
  sensory_motor.py        Keyboard-level I/O normalisation
  llm_parent.py           Together AI — teacher and corrector
  auto_driver.py          Autonomous conversation driver
  system.py               Main orchestrator
  ingestor.py             Dataset ingestion — pre-training and fine-tuning
  wiki_ingest.py          Wikipedia streaming ingestion
```

---

## Core Algorithm (Identical at Every Level)

```
1. Input stimulates the unit
2. Generate hypotheses about the generative structure of the input
3. Test each hypothesis — autoregressive token-level scoring at phrase level+
4. Propagate reward hierarchically through every contributing structure
5. Extreme success (≥ 0.99) → stored in library
6. Extreme failure (≤ 0.30) → stored as negative example
7. Mediocre → discarded, leaves no trace
8. Mutate successful structures and test again
9. Package recurring sub-sequences as new primitives (abstraction)
```

This is the only learning mechanism. Everything else grows from it.

---

## What Changed in v0.5

### Autoregressive Token-Level Reward
Every generated token gets a 1 (correct) or 0 (wrong) compared to the target.
Not the whole response — each individual token.

```
generated : i brush my teeht every day
target    : i brush my teeth every day
mask      : 1  1     1  0     1     1
```

### Hierarchical Propagation
When a token is correct, the reward flows down through every structure that
contributed to it — phrase → word → syllable → character. When wrong, the
penalty flows the same way. The system knows exactly which connection at which
level caused the error.

This is not backpropagation. It is how a parent teaches a child —
token by token, in real time, through the hierarchy.

### Contextual Fitness
Teacher corrections carry higher weight than exploration-generated structures.
A correction tagged with topic words dominates for matching topic queries.
Exploration noise doesn't drown out what the teacher taught.

### Numbers
Digits 0-9 are now seeded as level 0 primitives alongside letters.
"1969" traces all the way to individual digit characters.

### Dataset Ingestion Pipeline
`ingestor.py` — pre-training and fine-tuning from open source datasets.
`wiki_ingest.py` — Wikipedia streaming with bottom-up ingestion.

---

## Running

### Install
```bash
pip install datasets
```

### Wikipedia Ingestion (start here)
```bash
# Run until done — resumes automatically if interrupted
python hgls/wiki_ingest.py --no-llm

# Or limit sentences for a weekend run
python hgls/wiki_ingest.py --no-llm --max-sentences 2000000

# Start fresh
python hgls/wiki_ingest.py --no-llm --reset-checkpoint
```

Progress looks like:
```
[08:03:19] articles=847 | sentences=12,450 | words_learned=8,231 | lib=94,302 | rate=89.3/s
```

Rate starts slow (new vocabulary). Climbs as vocabulary saturates.

### Chat with Deepak
```bash
python main.py --chat
```

### Autonomous Learning
```bash
python main.py --auto              # runs until Ctrl+C
python main.py --auto --turns 500  # exactly 500 turns
```

### Pre-train on Your Own Dataset
```python
from hgls.system import HGLSystem
from hgls.ingestor import DatasetIngestor

system = HGLSystem(use_llm=False)
system.load('deepak_memory.json')
ingestor = DatasetIngestor(system)

# Raw text — Wikipedia, OpenStax, NCERT, C4
ingestor.pretrain_texts(your_list_of_sentences, level=3)

# Q&A pairs — SlimOrca, GSM8K, MetaMath, JEE papers
ingestor.finetune_qa([
    {'question': 'why does water evaporate?',
     'answer': 'water evaporates because heat gives molecules enough energy to escape'},
])

system.save('deepak_memory.json')
```

### Stream Large Datasets
```python
def my_lines():
    with open('large_dataset.txt') as f:
        for line in f:
            yield line.strip()

ingestor.pretrain_stream(my_lines(), level=3, save_every=50000)
```

---

## Dataset Order

Layer 1 — Language Foundation (levels 2-3)
  Wikipedia English     — 20M sentences, cleanest factual text
  C4 / OpenWebText      — broader vocabulary, natural prose

Layer 2 — Factual Depth (level 4)
  OpenStax textbooks    — physics, chemistry, biology, history, economics
  NCERT textbooks       — Indian curriculum, JEE/NEET base

Layer 3 — Reasoning Fine-tuning (switch to finetune_qa)
  SlimOrca              — 500k Chain of Thought examples
  MetaMath + GSM8K      — mathematical reasoning, step-by-step
  ARC + TheoremQA       — science reasoning

Layer 4 — Domain Mastery
  JEE / CBSE papers     — hardest domain-specific Q&A
  The Stack             — code, all languages

---

## Key Design Principles

1. One algorithm only — reconstruction quality drives everything
2. Only extreme outcomes stored — mediocre leaves no trace
3. Higher levels build from validated lower levels only
4. Successful sub-sequences become new primitives (abstraction)
5. Teacher corrections dominate for their topic (contextual fitness)
6. LLM parent fades as internal reward matures
7. The same unit that learns also generates — no separate modules
8. Numbers and letters are equal primitives — both connected bottom-up

---

## Current State

- 43 character primitives (a-z, 0-9, punctuation, space)
- 7 curriculum stages
- Autoregressive token-level reward with hierarchical propagation
- Chain of Thought emerging from correction loop
- Wikipedia ingestion running

---

## What's Next

- Multiprocessing with level sharding
- OpenStax + NCERT ingestion after Wikipedia
- SlimOrca fine-tuning for reasoning
- Mathematics datasets