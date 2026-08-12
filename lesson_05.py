"""
Lesson 05 — Cup (core concept).

Second object. Little Deepak sees a cup and hears its name.

Text  : c, u, p
Voice : /k/ /ʌ/ /p/ + papa's voice
Vision: mid brightness, warm (ceramic), horizontal edge (rim of cup)

Structure mirrors lesson 03:
  Phase 1 — word alone (text + voice)
  Phase 2 — word + sight (all three)
  Phase 3 — consolidation with variation (warm cup / cool cup)

Run:
  python lesson_05.py

Requires knowledge.json from lesson_04.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import System
from atoms  import encode_text, encode_voice, encode_voice_signature, encode_patches


PAPA_VOICE = dict(pitch='mid', tone='warm', tempo='mid')

CUP_TEXT  = encode_text('cup')
CUP_VOICE = encode_voice(['k', 'ʌ', 'p'], **PAPA_VOICE)

# Cup visual: mid brightness, warm ceramic, horizontal rim edge
CUP_PATCHES_WARM = encode_patches([
    {'brightness': 0.5, 'r': 0.75, 'g': 0.5, 'b': 0.3, 'edge_angle': 10.0},
])
# Cold drink cup — cooler color
CUP_PATCHES_COOL = encode_patches([
    {'brightness': 0.5, 'r': 0.4,  'g': 0.5, 'b': 0.75, 'edge_angle': 10.0},
])


def teach(s: System) -> None:

    print('\n[Phase 1] Word alone — text + voice, 10 reps')
    for rep in range(10):
        paths, cross, avg = s.learn_multi(
            text_atoms  = CUP_TEXT,
            voice_atoms = CUP_VOICE,
            reward      = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    s.save()

    print('\n[Phase 2] Word + sight — all three, 15 reps')
    for rep in range(15):
        paths, cross, avg = s.learn_multi(
            text_atoms   = CUP_TEXT,
            voice_atoms  = CUP_VOICE,
            vision_atoms = CUP_PATCHES_WARM,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    s.save()

    print('\n[Phase 3] Consolidation — warm + cool cup, 15 reps')
    for rep in range(15):
        patches = CUP_PATCHES_WARM if rep % 2 == 0 else CUP_PATCHES_COOL
        paths, cross, avg = s.learn_multi(
            text_atoms   = CUP_TEXT,
            voice_atoms  = CUP_VOICE,
            vision_atoms = patches,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    s.save()


def test(s: System) -> None:

    print('\n[Results]')

    _, cup_text  = s.query(CUP_TEXT,          'text')
    _, cup_voice = s.query(CUP_VOICE,         'voice')
    _, cup_warm  = s.query(CUP_PATCHES_WARM,  'vision')
    _, cup_cool  = s.query(CUP_PATCHES_COOL,  'vision')
    _, ball_text = s.query(encode_text('ball'), 'text')

    papa_sig  = encode_voice_signature(**PAPA_VOICE)
    cup_phone = CUP_VOICE[0]
    sig_edge  = s.graph.get_edge(papa_sig[0], cup_phone) if papa_sig else None
    sig_str   = sig_edge.strength if sig_edge else 0.0

    st = s.state()

    print(f"  Cup text  score    : {cup_text:.0f}")
    print(f"  Cup voice score    : {cup_voice:.0f}")
    print(f"  Cup warm  vision   : {cup_warm:.0f}")
    print(f"  Cup cool  vision   : {cup_cool:.0f}")
    print(f"  Ball text score    : {ball_text:.0f}")
    print(f"  Papa sig → cup /k/ : {sig_str:.0f}")
    print(f"  Total nodes        : {st['total_nodes']}")
    print(f"  Total edges        : {st['total_edges']}")

    print('\n[Checks]')
    checks = [
        ('Cup text path is strong',       cup_text  > 100),
        ('Cup voice path is strong',      cup_voice > 100),
        ('Warm cup recognised',           cup_warm  > 100),
        ('Cool cup recognised',           cup_cool  > 100),
        ('Cup distinct from ball',        abs(cup_text - ball_text) > 10),
        ('Papa sig connected to cup',     sig_str   > 0),
    ]

    passed = 0
    for label, condition in checks:
        status = 'PASS' if condition else 'FAIL'
        print(f"  {status}  {label}")
        if condition:
            passed += 1

    print(f"\n  {passed}/{len(checks)} passed")
    if passed == len(checks):
        print('  Little Deepak knows cup.')


if __name__ == '__main__':
    if not os.path.exists('knowledge.json'):
        print('[Lesson 05] Run lessons 01-04 first.')
        sys.exit(1)
    s = System('knowledge.json')
    teach(s)
    test(s)