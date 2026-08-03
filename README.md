# Hierarchical Generative Learning System (HGLS)
### Version 0.4 — Little Deepak Edition

A system that grows hierarchical generative knowledge using a single uniform learning algorithm, starting from keyboard characters and building upward through curriculum-controlled developmental stages.

The persona being built is **Little Deepak** — a general good 5-year-old Indian child whose learning is grounded in health habits, family values, honesty, and curiosity.

---

## How It Works

The core idea is simple: every unit at every level of the hierarchy runs the **identical algorithm**:

1. Input stimulates the unit
2. Generate hypotheses about the generative structure of the input
3. Test each hypothesis (reconstruction quality)
4. **Extreme successes** → stored in the library
5. **Extreme failures** → stored as negative examples
6. **Mediocre outcomes** → discarded entirely
7. Mutate successful structures and repeat
8. Package recurring sub-sequences as new higher-level primitives (abstraction)

This is the only learning mechanism in the system.

---

## Little Deepak's Curriculum

Learning is structured around Little Deepak's world, in five stages:

| Stage | Level | Content |
|-------|-------|---------|
| Characters | 0 | a–z, the atomic primitives |
| Combinations | 1 | Syllable roots from Deepak's key vocabulary |
| Words | 2 | His world: health, family, values, learning |
| Phrases | 3 | Good habits stated simply: *i brush my teeth* |
| Schemas | 4 | Cause and effect: *when i sleep early i wake up happy* |

### Little Deepak's Values (priority order)

1. **Health habits** — brush, wash, drink water, eat well, sleep early
2. **Family and respect** — love for amma and appa, namaste, helping at home, caring for didi and bhaiya
3. **Honesty and kindness** — tell the truth, say sorry, share, be gentle
4. **Curiosity and learning** — school, reading, asking questions, writing neatly

All curriculum content lives in `hgls/persona.py`. To change what Little Deepak learns, edit that file.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Together AI API key

```bash
export TOGETHER_API_KEY=your_key_here
```

Get a key at [api.together.ai](https://api.together.ai).

### 3. Run

```bash
# With LLM parent (Together AI GPT-NeoXT-20B)
python main.py

# Offline — pure reconstruction quality, no API calls
python main.py --no-llm
```

---

## Module Map

```
hgls/
  persona.py          — Little Deepak: identity, vocabulary, values, parent prompt
  curriculum.py       — Developmental stage controller; sources content from persona.py
  structures.py       — GenerativeStructure: the core unit of knowledge
  library.py          — Long-term memory: stores only extreme outcomes
  tester.py           — ExtremeTester: success / failure / mediocre classification
  generative_unit.py  — The uniform learning algorithm (one instance per level)
  reward.py           — Internal reward: novelty + competence + curiosity
  llm_parent.py       — Together AI parental interface (signal fades over time)
  memory.py           — Working memory: limited-capacity context buffer
  attention.py        — Salience-weighted resource allocation
  self_model.py       — Tracks self-generated vs external structures (agency)
  sensory_motor.py    — Keyboard-level I/O normalisation
  system.py           — Main orchestrator: integrates all modules
main.py               — Entry point and demo
```

---

## LLM Parent

The LLM (Together AI — `togethercomputer/GPT-NeoXT-Chat-Base-20B`) acts as Little Deepak's parent:

- Evaluates reconstruction quality at higher levels
- Proposes candidate decompositions
- Judges content against Little Deepak's values

**The signal fades over time.** It starts at strength 1.0 and decays with each call. As Little Deepak's internal reward system matures, he depends less on external evaluation.

To change the model, edit the `MODEL` constant in `hgls/llm_parent.py`.

---

## Key Design Rules (from spec v0.4)

- **One learning algorithm only**, identical at every level
- **Reconstruction is the fundamental test**
- **Only extreme outcomes are stored** — mediocre is discarded and leaves no trace
- **Higher levels build only from validated lower-level structures**
- **Successful sub-sequences become new primitives** (abstraction)
- **Scope: keyboard-mediated text only** — embodiment is out of scope
- **LLM parent fades** as internal criteria strengthen

---

## Extending Little Deepak

To add new vocabulary or habits, edit `hgls/persona.py`:

- Add words to `LEVEL_3_WORDS`
- Add phrases to `LEVEL_4_PHRASES`
- Add causal schemas to `LEVEL_5_SCHEMAS`
- Add positive markers to `POSITIVE_MARKERS` so the parent recognises them

No other file needs to change for curriculum updates.