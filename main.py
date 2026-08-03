"""
main.py — Little Deepak

Hierarchical Generative Learning System v0.4
Persona: Little Deepak — a good 5-year-old Indian child

Usage:
    python main.py              # learn + explore, with LLM parent
    python main.py --no-llm     # learn + explore, offline
    python main.py --chat       # talk to Little Deepak after learning
    python main.py --reset      # forget saved memory and start fresh
"""

import sys
import time

from hgls import HGLSystem
from hgls.curriculum import Stage
import hgls.persona as persona

MEMORY_FILE            = 'deepak_memory.json'
MAX_PASSES_PER_STAGE   = 10
EXPLORATION_ROUNDS     = 80


# ── Curriculum stage runner ───────────────────────────────────────

def run_until_advanced(system, inputs, label, verbose=True):
    bar = '─' * max(1, 55 - len(label))
    print(f'── {label} {bar}')
    passes   = 0
    prev     = system.curriculum.current_stage
    is_final = prev >= Stage.SCHEMAS

    while passes < MAX_PASSES_PER_STAGE:
        passes += 1
        system.run_episode(inputs, verbose=verbose)
        if not is_final and system.curriculum.current_stage != prev:
            break

    if passes == MAX_PASSES_PER_STAGE and not is_final:
        print('  [Note] Max passes reached — moving on.')
    print()


# ── Exploration session ───────────────────────────────────────────

def run_exploration(system, label=''):
    tag = f' ({label})' if label else ''
    print(f'── Internal Exploration{tag} {"─" * max(1, 38 - len(label))}')
    result = system.explore(n=EXPLORATION_ROUNDS)
    print(f'  Tried: {result.combinations_tried}  '
          f'Novel: {result.novel_discovered}  '
          f'({result.novel_discovered/max(1,result.combinations_tried):.0%})')
    if result.examples:
        print('  Sample discoveries:')
        for strategy, text in result.examples[:5]:
            print(f'    [{strategy:10}] "{text}"')
    print()


# ── Generation probe ──────────────────────────────────────────────

def generation_probe(system):
    print('── Generation Probe ─────────────────────────────────────')
    probes = [
        "i brush", "when i sleep", "i love amma",
        "when i tell the truth", "i share", "when i help",
        "i go to school",
    ]
    for p in probes:
        print(f"  '{p}'  →  '{system.generate(p)}'")
    print()


# ── Dialogue loop ─────────────────────────────────────────────────

def dialogue_loop(system):
    print('── Talk to Little Deepak ────────────────────────────────')
    print('  Little Deepak is ready. Type "bye" to exit.\n')
    while True:
        try:
            user = input('  You    : ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() in ('bye', 'quit', 'exit'):
            print('  Deepak : bye bye!')
            break
        response = system.respond(user)
        print(f'  Deepak : {response}')
        print()


# ── Main ──────────────────────────────────────────────────────────

def main():
    use_llm = '--no-llm'  not in sys.argv
    reset   = '--reset'   in sys.argv
    chat    = '--chat'    in sys.argv

    print('=' * 60)
    print('Hierarchical Generative Learning System v0.4')
    print(f'Persona : {persona.NAME}')
    print(f'LLM     : {"Together AI — GPT-NeoXT-20B" if use_llm else "disabled"}')
    print('=' * 60)
    print()

    system = HGLSystem(use_llm=use_llm)

    # ── Persistence: try to load saved memory ─────────────────────
    if not reset and system.load(MEMORY_FILE):
        print('  Little Deepak remembers his previous learning.\n')
        already_learned = True
    else:
        already_learned = False
        print()

    # ── Curriculum (skipped if memory loaded) ────────────────────
    if not already_learned:
        stages = [
            ("Characters",              persona.LEVEL_1_CHARS),
            ("Combinations",            persona.LEVEL_2_COMBINATIONS),
            ("Words — Deepak's World",  persona.LEVEL_3_WORDS),
            ("Phrases — Good Habits",   persona.LEVEL_4_PHRASES),
            ("Schemas — Cause-Effect",  persona.LEVEL_5_SCHEMAS),
        ]
        for label, inputs in stages:
            run_until_advanced(system, inputs, label, verbose=True)
            run_exploration(system, label=label)

        # Save after full learning
        system.save(MEMORY_FILE)
        print()

    # ── Always run a short exploration on startup ─────────────────
    print('── Exploration on startup ───────────────────────────────')
    run_exploration(system, label='startup')

    # ── Generation probe ──────────────────────────────────────────
    generation_probe(system)

    # ── Dialogue loop ─────────────────────────────────────────────
    if chat:
        dialogue_loop(system)
        # Save any new learning from the conversation
        system.save(MEMORY_FILE)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'Total time: {time.time() - t0:.1f}s')