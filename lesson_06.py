"""
Lesson 06 — Cup reinforcement + Spoon (core concept).

Two things happen this lesson:

  Part A — Cup reinforcement (mama's voice, action context)
    Same as lesson 04 did for ball.
    "drink from the cup", "mama's cup", "your cup"

  Part B — Spoon (core concept, papa's voice)
    Text  : s, p, o, o, n
    Voice : /s/ /p/ /uː/ /n/ + papa's voice
    Vision: bright, neutral, diagonal edge (curved shiny handle)

Run:
  python lesson_06.py

Requires knowledge.json from lesson_05.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import System
from atoms  import encode_text, encode_voice, encode_voice_signature, encode_patches


MAMA_VOICE = dict(pitch='high', tone='warm', tempo='slow')
PAPA_VOICE = dict(pitch='mid',  tone='warm', tempo='mid')

# ── Cup reinforcement ─────────────────────────────────────────────

CUP_TEXT       = encode_text('cup')
CUP_VOICE_MAMA = encode_voice(['k', 'ʌ', 'p'], **MAMA_VOICE)
CUP_VOICE_PAPA = encode_voice(['k', 'ʌ', 'p'], **PAPA_VOICE)
CUP_PATCHES    = encode_patches([
    {'brightness': 0.5, 'r': 0.75, 'g': 0.5, 'b': 0.3, 'edge_angle': 10.0},
])

DRINK_CUP = encode_text('drink from the cup')
YOUR_CUP  = encode_text('your cup')
MAMAS_CUP = encode_text('mamas cup')

# ── Spoon concept ─────────────────────────────────────────────────

SPOON_TEXT    = encode_text('spoon')
SPOON_VOICE   = encode_voice(['s', 'p', 'uː', 'n'], **PAPA_VOICE)

# Spoon: bright, neutral (silver/metal), diagonal edge (curved handle)
SPOON_PATCHES = encode_patches([
    {'brightness': 0.85, 'r': 0.75, 'g': 0.75, 'b': 0.75, 'edge_angle': 50.0},
])
# Spoon from side — more horizontal
SPOON_PATCHES_SIDE = encode_patches([
    {'brightness': 0.8, 'r': 0.72, 'g': 0.72, 'b': 0.72, 'edge_angle': 10.0},
])


def teach(s: System) -> None:

    # ── Part A: Cup reinforcement ─────────────────────────────────
    print('\n[Part A] Cup — mama\'s voice, 10 reps')
    for rep in range(10):
        voice   = CUP_VOICE_MAMA if rep % 2 == 0 else CUP_VOICE_PAPA
        paths, cross, avg = s.learn_multi(
            text_atoms   = CUP_TEXT,
            voice_atoms  = voice,
            vision_atoms = CUP_PATCHES,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    print('\n[Part A] Cup in phrase context, 5 reps each')
    for phrase in [DRINK_CUP, YOUR_CUP, MAMAS_CUP]:
        for rep in range(5):
            voice = CUP_VOICE_MAMA if rep % 2 == 0 else CUP_VOICE_PAPA
            s.learn_multi(
                text_atoms   = phrase,
                voice_atoms  = voice,
                vision_atoms = CUP_PATCHES,
                reward       = True,
            )
        print(f"  phrase done  cross={cross:.4f}")

    s.save()

    # ── Part B: Spoon core ────────────────────────────────────────
    print('\n[Part B] Spoon — word alone, 10 reps')
    for rep in range(10):
        paths, cross, avg = s.learn_multi(
            text_atoms  = SPOON_TEXT,
            voice_atoms = SPOON_VOICE,
            reward      = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    print('\n[Part B] Spoon — word + sight, 15 reps')
    for rep in range(15):
        patches = SPOON_PATCHES if rep % 2 == 0 else SPOON_PATCHES_SIDE
        paths, cross, avg = s.learn_multi(
            text_atoms   = SPOON_TEXT,
            voice_atoms  = SPOON_VOICE,
            vision_atoms = patches,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    s.save()


def test(s: System) -> None:

    print('\n[Results]')

    _, cup_text   = s.query(CUP_TEXT,           'text')
    _, cup_mama   = s.query(CUP_VOICE_MAMA,     'voice')
    _, cup_papa   = s.query(CUP_VOICE_PAPA,     'voice')
    _, spoon_text = s.query(SPOON_TEXT,         'text')
    _, spoon_v    = s.query(SPOON_VOICE,        'voice')
    _, spoon_vis  = s.query(SPOON_PATCHES,      'vision')
    _, spoon_side = s.query(SPOON_PATCHES_SIDE, 'vision')
    _, ball_text  = s.query(encode_text('ball'),'text')

    papa_sig   = encode_voice_signature(**PAPA_VOICE)
    mama_sig   = encode_voice_signature(**MAMA_VOICE)
    cup_phone  = CUP_VOICE_PAPA[0]
    spoon_phone = SPOON_VOICE[0]

    cup_papa_sig = s.graph.get_edge(papa_sig[0], cup_phone)  if papa_sig else None
    cup_mama_sig = s.graph.get_edge(mama_sig[0], cup_phone)  if mama_sig else None
    spoon_sig    = s.graph.get_edge(papa_sig[0], spoon_phone) if papa_sig else None

    st = s.state()

    print(f"  Cup text         : {cup_text:.0f}")
    print(f"  Cup mama voice   : {cup_mama:.0f}")
    print(f"  Cup papa voice   : {cup_papa:.0f}")
    print(f"  Spoon text       : {spoon_text:.0f}")
    print(f"  Spoon voice      : {spoon_v:.0f}")
    print(f"  Spoon diagonal   : {spoon_vis:.0f}")
    print(f"  Spoon side       : {spoon_side:.0f}")
    print(f"  Ball text        : {ball_text:.0f}")
    print(f"  Total nodes      : {st['total_nodes']}")
    print(f"  Total edges      : {st['total_edges']}")

    print('\n[Checks]')
    checks = [
        ('Cup known by mama voice',       cup_mama  > 100),
        ('Cup known by papa voice',       cup_papa  > 100),
        ('Spoon text path strong',        spoon_text > 100),
        ('Spoon voice path strong',       spoon_v   > 100),
        ('Spoon diagonal vision strong',  spoon_vis > 100),
        ('Spoon side vision recognised',  spoon_side > 100),
        ('Spoon distinct from ball',      abs(spoon_text - ball_text) > 10),
        ('Papa sig connected to spoon',   spoon_sig is not None and spoon_sig.strength > 0),
    ]

    passed = 0
    for label, condition in checks:
        status = 'PASS' if condition else 'FAIL'
        print(f"  {status}  {label}")
        if condition:
            passed += 1

    print(f"\n  {passed}/{len(checks)} passed")
    if passed >= 7:
        print('  Little Deepak knows cup (both voices) and spoon.')


if __name__ == '__main__':
    if not os.path.exists('knowledge.json'):
        print('[Lesson 06] Run lessons 01-05 first.')
        sys.exit(1)
    s = System('knowledge.json')
    teach(s)
    test(s)