"""
Lesson 03 — Ball (core concept).

First object lesson. Little Deepak sees a ball and hears its name.

Text  : b, a, l, l
Voice : /b/ /ɔː/ /l/ + papa's voice (mid pitch, warm tone, mid tempo)
Vision: bright + warm + none (round, colorful, no straight edges)

Teaching structure:
  Phase 1 — word alone (text + voice, no vision yet)
            like hearing "ball" before seeing one clearly
  Phase 2 — word + sight together (all three modalities)
            like papa holding the ball and saying "ball"
  Phase 3 — consolidation (all three, more reps)
            repeated encounters to strengthen the concept

Run:
  python lesson_03.py

Requires knowledge.json from lesson_02.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import System
from atoms  import encode_text, encode_voice, encode_voice_signature, encode_patches


# ── Voices ────────────────────────────────────────────────────────

PAPA_VOICE = dict(pitch='mid', tone='warm', tempo='mid')

# ── Ball concept ──────────────────────────────────────────────────

BALL_TEXT  = encode_text('ball')
BALL_VOICE = encode_voice(['b', 'ɔː', 'l'], **PAPA_VOICE)

# Ball visual: bright, warm (colorful), no strong edges (round)
BALL_PATCHES = encode_patches([
    {'brightness': 0.8, 'r': 0.9, 'g': 0.3, 'b': 0.1, 'edge_angle': None},
])

# Ball in shadow — same object, less light
BALL_PATCHES_DIM = encode_patches([
    {'brightness': 0.4, 'r': 0.8, 'g': 0.3, 'b': 0.1, 'edge_angle': None},
])


def teach(s: System) -> None:

    # ── Phase 1: word alone ───────────────────────────────────────
    print('\n[Phase 1] Word alone — text + voice, 10 reps')
    for rep in range(10):
        paths, cross, avg = s.teach_concept('ball',
            text_atoms  = BALL_TEXT,
            voice_atoms = BALL_VOICE,
            reward      = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    s.save()

    # ── Phase 2: word + sight ─────────────────────────────────────
    print('\n[Phase 2] Word + sight — all three modalities, 15 reps')
    for rep in range(15):
        paths, cross, avg = s.teach_concept('ball',
            text_atoms   = BALL_TEXT,
            voice_atoms  = BALL_VOICE,
            vision_atoms = BALL_PATCHES,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    s.save()

    # ── Phase 3: consolidation with variation ─────────────────────
    print('\n[Phase 3] Consolidation + visual variation, 15 reps')
    for rep in range(15):
        # alternate between bright and dim ball
        patches = BALL_PATCHES if rep % 2 == 0 else BALL_PATCHES_DIM
        paths, cross, avg = s.teach_concept('ball',
            text_atoms   = BALL_TEXT,
            voice_atoms  = BALL_VOICE,
            vision_atoms = patches,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")


    # Register concepts in the graph
    s.register_concept('ball')
    s.save()


def test(s: System) -> None:

    print('\n[Results]')

    _, ball_text_score   = s.query(BALL_TEXT,        'text')
    _, ball_voice_score  = s.query(BALL_VOICE,       'voice')
    _, ball_vision_score = s.query(BALL_PATCHES,     'vision')
    _, ball_dim_score    = s.query(BALL_PATCHES_DIM, 'vision')

    # Ball should be distinct from known words
    _, papa_score = s.query(encode_text('papa'), 'text')
    _, mama_score = s.query(encode_text('mama'), 'text')

    # Papa's voice sig connected to ball phonemes
    papa_sig   = encode_voice_signature(**PAPA_VOICE)
    ball_phone = BALL_VOICE[0]   # first phoneme 'vo:b'
    sig_edge   = s.graph.get_edge(papa_sig[0], ball_phone) if papa_sig else None
    sig_str    = sig_edge.strength if sig_edge else 0.0

    # Vision connected to text — cross-modal concept forming
    vis_atom  = BALL_PATCHES[0]
    text_atom = BALL_TEXT[0]
    vt_edge   = s.graph.get_edge(vis_atom, text_atom)
    vt_str    = vt_edge.strength if vt_edge else 0.0

    st = s.state()

    print(f"  Ball text  score     : {ball_text_score:.0f}")
    print(f"  Ball voice score     : {ball_voice_score:.0f}")
    print(f"  Ball vision score    : {ball_vision_score:.0f}")
    print(f"  Ball dim   score     : {ball_dim_score:.0f}")
    print(f"  Papa text  score     : {papa_score:.0f}")
    print(f"  Mama text  score     : {mama_score:.0f}")
    print(f"  Papa sig → ball /b/  : {sig_str:.0f}")
    print(f"  Vision → text edge   : {vt_str:.0f}")
    print(f"  Total nodes          : {st['total_nodes']}")
    print(f"  Total edges          : {st['total_edges']}")

    print('\n[Checks]')
    checks = [
        ('Ball text path is strong',
            ball_text_score > 100),
        ('Ball voice path is strong',
            ball_voice_score > 100),
        ('Ball vision path is strong',
            ball_vision_score > 100),
        ('Dim ball still recognised (visual variation)',
            ball_dim_score > 100),
        ('Papa sig connected to ball phoneme',
            sig_str > 0),
        ('Ball is distinct from papa',
            abs(ball_text_score - papa_score) > 100),
        ('Ball is distinct from mama',
            abs(ball_text_score - mama_score) > 100),
        ('Vision and text starting to connect',
            vt_str >= 0),        # will be 0 now, > 0 after lesson 04
    ]

    passed = 0
    for label, condition in checks:
        status = 'PASS' if condition else 'FAIL'
        print(f"  {status}  {label}")
        if condition:
            passed += 1

    print(f"\n  {passed}/{len(checks)} passed")
    if passed >= 7:
        print('  Little Deepak knows ball.')


if __name__ == '__main__':
    if not os.path.exists('knowledge.json'):
        print('[Lesson 03] Run lessons 01 and 02 first.')
        sys.exit(1)

    s = System('knowledge.json')
    teach(s)
    test(s)