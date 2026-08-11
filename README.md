# HGLS — Hierarchical Generative Learning System

A learning system that builds knowledge from scratch.
No neural networks. No pre-training. No hardcoded rules.

The graph is the knowledge. Everything else serves the graph.

---

## Core Idea

Input activates atoms. Atoms are connected through a graph.
The system follows the strongest path forward from those atoms.
You signal reward or no reward. The graph adjusts.
Over time the graph becomes the knowledge.

Same process at every level. Same process for every modality.

---

## Three Modalities, One Graph

```
Text   — 41 atoms  (a-z, 0-9, punctuation)
Voice  — 41 atoms  (English phonemes)
Vision — 36 atoms  (pixel patches: brightness × color × edge)
```

All 118 primitives live at level 0 in the same graph.
They compose upward through the same mechanism:

```
Level 0 — atoms        (characters, phonemes, pixel patches)
Level 1 — clusters     (syllables, phoneme groups, visual edges)
Level 2 — units        (words, spoken words, visual regions)
Level 3 — sequences    (sentences, utterances, scenes)
Level 4 — associations (cross-modal links form here)
Level 5 — schemas      (concepts grounded in all three modalities)
Level 6 — reasoning
```

A concept like "apple" eventually becomes a node connected to
its spelling, its sound, and its visual appearance simultaneously.
That convergence is not programmed — it emerges from cross-modal
co-activation being rewarded repeatedly.

---

## Six Files

```
atoms.py     — 118 primitives and input encoders
memory.py    — Graph, Node, Edge — the knowledge structure
eval.py      — coherence scoring, strength adjustment
tinkerer.py  — novel connection engine
algo.py      — activate, propagate, reward, abstract
system.py    — single entry point for all interaction
```

---

## How It Works

### Learning

```
input → activate atoms → propagate through graph → eval scores path
→ you signal reward or no reward
→ eval adjusts every edge on the path
→ strong edges grow stronger, weak edges grow weaker
→ nothing is deleted, nothing decays
```

### Tinkerer

Engaged when propagation finds weak or missing paths.
Looks at co-activated nodes that are not yet connected.
Proposes connections using three strategies:

- **Extension** — A→B and B→C exist, propose A→C
- **Analogy** — A and X have similar neighbourhood structure, B and Y are connected, propose A→B
- **Compression** — dense cluster of mutually connected nodes → flag for abstraction to next level

Tinkerer learns which strategy works in which situation
because strategy choices and their outcomes are also edges in the graph.

### Abstraction

When a cluster of nodes is densely connected and co-activates
repeatedly on rewarded interactions, algo abstracts it into
a single new node at the next level up.
That node starts with the average strength of its members
and connects back to all of them.

This is how characters become syllables, syllables become words,
words become phrases, and so on — without any rule saying when to abstract.
The graph structure decides.

### Emotional Intensity

Eval does not apply fixed amounts. Intensity emerges from the situation:

```
high coherence + reward     → strong confirmation   → large reinforcement
low  coherence + reward     → discovery             → moderate reinforcement
high coherence + no reward  → surprise              → moderate weakening
low  coherence + no reward  → clear error           → large weakening
```

Strong paths resist weakening. Weak paths are still finding their place.
Adjustment is always inversely proportional to current strength.

---

## Your Training Modules

You hand-design what gets taught and when.
The system exposes three methods:

```python
from system import System

s = System('knowledge.json')

# single modality
path, score, intensity = s.learn(s.text('hello'), 'text', reward=True)

# all three modalities at once
paths, cross, avg = s.learn_multi(
    text_atoms   = s.text('apple'),
    voice_atoms  = s.phonemes(['æ', 'p', 'ə', 'l']),
    vision_atoms = s.patches([...]),
    reward       = True,
)

# query without reward
path, score = s.query(s.text('hello'), 'text')
```

Score tells you how well the system knows the input.
High score — strong familiar path.
Low score  — novel or not yet learned.

Your external module decides what deserves reward.
That decision shapes everything the system becomes.

---

## Commands

```bash
# Query the system interactively
python main.py --chat

# See current state (nodes and edges by level)
python main.py --state

# Teach one input with reward
python main.py --teach "hello" --reward

# Teach one input with no reward
python main.py --teach "hello" --no-reward

# Use a different knowledge file
python main.py --chat --memory my_knowledge.json
```

---

## Files Written to Disk

```
knowledge.json      — the full graph (nodes + edges)
knowledge.json.bak  — previous save
```

That is all.

---

## Design Principles

1. The graph is the knowledge — not a representation of it
2. One algorithm at every level and every modality
3. No hardcoded thresholds anywhere
4. No hardcoded reward amounts — intensity emerges from the situation
5. Nothing is deleted, nothing decays — edges only move up or down
6. Strong edges resist change — stable knowledge is hard to shake
7. Tinkerer grows the graph, eval shapes the strengths
8. Abstraction is not scheduled — it emerges from density
9. Cross-modal concepts emerge from convergent co-activation
10. Your external module is the teacher — the system learns from your signal