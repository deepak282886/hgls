"""
main.py — Little Deepak

Hierarchical Generative Learning System v0.4
Persona: Little Deepak — a good 5-year-old Indian child

Usage:
    python main.py               # learn from scratch + domain expansion
    python main.py --no-llm      # persona seed only
    python main.py --chat        # talk to Little Deepak
    python main.py --auto        # autonomous LLM-driven learning session
    python main.py --auto --turns 200   # run exactly 200 turns
    python main.py --reset       # forget library, start fresh
    python main.py --clear-cache # regenerate curriculum content
"""

import sys
import time

from hgls import HGLSystem
from hgls.curriculum import Stage
from hgls.curriculum_generator import CurriculumGenerator, DOMAINS
from hgls.auto_driver import AutoDriver
from hgls.llm_parent import MODEL as LLM_MODEL
import hgls.persona as persona

MEMORY_FILE          = 'deepak_memory.json'
MAX_PASSES_PER_STAGE = 10
MAX_DOMAIN_PASSES    = 5
EXPLORATION_ROUNDS   = 80


# ── Curriculum helpers ────────────────────────────────────────────

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
        print('  [Note] Max passes reached.')
    print()


def run_until_saturated(system, content, level, max_passes=MAX_DOMAIN_PASSES):
    """Run content at a specific level until library stops growing."""
    prev_size = len(system.library)
    for i in range(1, max_passes + 1):
        results  = system.run_episode(content, verbose=False, target_level=level)
        new_size = len(system.library)
        hits     = sum(r.get('n_successes', 0) for r in results)
        print(f'    Pass {i}: lib {prev_size} → {new_size} '
              f'(+{new_size - prev_size}) ✓={hits}')
        if new_size == prev_size:
            print(f'    ✓ Saturated.')
            break
        prev_size = new_size


def run_exploration(system, label=''):
    tag = f' ({label})' if label else ''
    print(f'── Exploration{tag} {"─"*max(1, 45-len(label))}')
    result = system.explore(n=EXPLORATION_ROUNDS)
    print(f'  Tried: {result.combinations_tried}  '
          f'Novel: {result.novel_discovered}  '
          f'({result.novel_discovered/max(1,result.combinations_tried):.0%})')
    if result.examples:
        for strategy, text in result.examples[:3]:
            print(f'    [{strategy:10}] "{text}"')
    print()


# ── Domain expansion ──────────────────────────────────────────────

def run_domain_expansion(system, generator):
    """
    Expand Little Deepak's knowledge across all domains.
    The LLM parent generates content; the system learns it.
    Each domain runs words → phrases → schemas, exploring between each.
    """
    print('\n' + '=' * 60)
    print('DOMAIN EXPANSION — Little Deepak grows his world')
    print('=' * 60)

    level_map = {2: 'Words', 3: 'Phrases', 4: 'Schemas'}

    for i, domain in enumerate(DOMAINS):
        print(f'\n── Domain {i+1}/{len(DOMAINS)}: {domain.upper()} '
              f'{"─" * max(1, 38 - len(domain))}')

        had_content = False

        for level in [2, 3, 4]:
            content = generator.get_content(domain, level)
            if not content:
                print(f'  [{level_map[level]}] No content — skipping')
                continue

            had_content = True
            cached_tag  = '[cached]' if generator.is_cached(domain, level) else '[new]'
            print(f'\n  {level_map[level]} {cached_tag} — {len(content)} items')
            print(f'  Sample: {content[:4]}')

            run_until_saturated(system, content, level)
            run_exploration(system, label=domain)

        if had_content:
            system.save(MEMORY_FILE)
            print(f'  [Memory] {len(system.library)} structures saved.')


# ── Generation probe ──────────────────────────────────────────────

def generation_probe(system):
    print('── Generation Probe ─────────────────────────────────────')
    probes = [
        'i brush', 'when i sleep', 'i love amma',
        'when i tell the truth', 'i share', 'when i help',
        'i see the', 'i feel', 'when i count',
    ]
    for p in probes:
        print(f"  '{p}'  →  '{system.generate(p)}'")
    print()


# ── Dialogue ──────────────────────────────────────────────────────

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
    use_llm   = '--no-llm'      not in sys.argv
    reset     = '--reset'       in sys.argv
    chat      = '--chat'        in sys.argv
    auto      = '--auto'        in sys.argv
    clr_cache = '--clear-cache' in sys.argv

    # Parse --turns N
    max_turns = 0
    if '--turns' in sys.argv:
        idx = sys.argv.index('--turns')
        if idx + 1 < len(sys.argv):
            try:
                max_turns = int(sys.argv[idx + 1])
            except ValueError:
                pass

    print('=' * 60)
    print('Hierarchical Generative Learning System v0.4')
    print(f'Persona : {persona.NAME}')
    print(f'LLM     : {"Together AI — " + LLM_MODEL if use_llm else "disabled"}')
    print('=' * 60)
    print()

    system    = HGLSystem(use_llm=use_llm)
    generator = CurriculumGenerator() if use_llm else None

    if clr_cache and generator:
        generator.clear_cache()

    # ── Load or seed from scratch ─────────────────────────────────
    already_learned = not reset and system.load(MEMORY_FILE)

    if already_learned:
        print('  Little Deepak remembers his previous learning.\n')
    else:
        print()
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

        system.save(MEMORY_FILE)
        print()

    # ── Domain expansion (only when learning from scratch) ────────
    if not already_learned:
        if use_llm and generator:
            run_domain_expansion(system, generator)
        else:
            print('[Note] Domain expansion needs LLM. '
                  'Remove --no-llm to expand.\n')

    # ── Startup exploration (brief, always) ───────────────────────
    if not chat:
        run_exploration(system, label='startup')
        generation_probe(system)

    # ── Dialogue ──────────────────────────────────────────────────
    if chat:
        generation_probe(system)
        dialogue_loop(system)
        system.save(MEMORY_FILE)

    # ── Autonomous learning session ────────────────────────────────
    elif auto:
        if not use_llm:
            print('[Auto] Autonomous mode requires LLM. Remove --no-llm.')
            return
        driver = AutoDriver(
            system=system,
            save_fn=lambda: system.save(MEMORY_FILE),
        )
        stats = driver.run(max_turns=max_turns)
        print('\n[Auto] Final stats:', stats)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'Total time: {time.time() - t0:.1f}s')