"""
diagnose3.py — Check emotional evaluator, reward system, and navigator state.
python diagnose3.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hgls.system import HGLSystem

system = HGLSystem(use_llm=False)
system.load('deepak_memory.json')

print('=' * 60)
print('INTERNAL EVALUATOR STATE')
print('=' * 60)

ev = system.emotional_evaluator
print(f'\nMaturity        : {ev.maturity:.6f}')
print(f'Eval count      : {ev._eval_count}')
print(f'Positive count  : {ev._positive_count}')
print(f'Negative count  : {ev._negative_count}')
print(f'Surprise count  : {ev._surprise_count}')
print(f'Contradiction   : {ev._contradiction_count}')
print(f'\nCurrent weights:')
print(f'  surface   : {ev._w_surface:.3f}')
print(f'  coherence : {ev._w_coherence:.3f}')
print(f'  novelty   : {ev._w_novelty:.3f}')
print(f'  surprise  : {ev._w_surprise:.3f}')

# Test it on some inputs
print('\nTest evaluations:')
tests = [
    ('i know a triangle has three sides', 'i know a triangle has three sides'),
    ('i know a triangle has three sides', 'i am not sure'),
    ('shapes fit together', 'a circle is round'),
    ('numbers help us count', 'there are nine numbers from one to nine'),
]
for gen, tgt in tests:
    result = ev.evaluate(gen, tgt)
    print(f'  score={result["score"]:.3f} state={result["state"]:<25} '
          f'surface={result["surface"]:.3f}')
    print(f'    gen: "{gen[:40]}"')
    print(f'    tgt: "{tgt[:40]}"')

print()
print('=' * 60)
print('REWARD SYSTEM STATE')
print('=' * 60)
r = system.reward
print(f'\nMaturity      : {r.maturity:.4f}')
print(f'Total reward  : {r.total_reward:.2f}')
print(f'Unique seen   : {len(r._seen_elements)}')
print(f'Recent rewards: {[round(x["reward"],3) for x in r._reward_log[-10:]]}')

print()
print('=' * 60)
print('NAVIGATOR STATE')
print('=' * 60)
nav = system.navigator
print(f'\nNav count     : {nav._nav_count}')
print(f'Avg outcomes  : {nav.stats()["avg_outcomes"]}')

print()
print('=' * 60)
print('GRAPH STATE')
print('=' * 60)
g = system.graph
print(f'\nTotal edges   : {len(g)}')
print(f'Pending co-occ: {len(g._pending_co_occ)}')
print(f'By type       : {g.stats()["by_type"]}')

# Show top pending co-occurrences
if g._pending_co_occ:
    top = sorted(g._pending_co_occ.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f'\nTop pending co-occurrences (need threshold to become edges):')
    for (a, b), count in top:
        sa = system.library.get(a)
        sb = system.library.get(b)
        ta = sa.generate(system.library)[:20] if sa else a
        tb = sb.generate(system.library)[:20] if sb else b
        print(f'  count={count:3d}  "{ta}" ↔ "{tb}"')

print()
print('=' * 60)
print('TINKERING STATE')
print('=' * 60)
print(f'\n{system.tinkering.stats()}')

print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'\nEmotional evaluator maturity: {ev.maturity:.6f}')
print('  → At 0.000xxx it is essentially primitive (just pos/neg)')
print('  → Needs ~50,000 evaluations to reach meaningful differentiation')
print(f'\nNavigator visits: {nav._nav_count}')
print('  → Traversal strategy learning requires many cycles')
print(f'\nGraph edges: {len(g)} (need 50+ for tinkering to activate)')
print(f'Pending co-occ pairs: {len(g._pending_co_occ)}')
threshold = max(3, len(system.library) // 10000)
print(f'Current co-occ threshold: ~{threshold} (lib_size/10000)')