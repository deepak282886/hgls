"""
Lesson 04 — Ball reinforcement (mama's voice + action context).

Same concept as lesson 03, now from mama's perspective.
Little Deepak hears mama say "ball" — same word, different voice.
The concept deepens and stabilises across both voices.

New this lesson:
  - Mama's voice saying "ball"
  - Action context: "roll the ball", "big ball"
    (introduces ball in short phrase context)
  - Different visual angles: ball close up vs far away

After this lesson the ball concept should be:
  - Grounded in both papa's and mama's voice
  - Connected across text + voice + vision solidly
  - Appearing in simple phrase context

Run:
  python lesson_04.py

Requires knowledge.json from lesson_03.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import System
from atoms  import encode_text, encode_voice, encode_voice_signature, encode_patches


# ── Voices ────────────────────────────────────────────────────────

MAMA_VOICE = dict(pitch='high', tone='warm', tempo='slow')
PAPA_VOICE = dict(pitch='mid',  tone='warm', tempo='mid')

# ── Ball concept ──────────────────────────────────────────────────

BALL_TEXT  = encode_text('ball')
BALL_VOICE_MAMA = encode_voice(['b', 'ɔː', 'l'], **MAMA_VOICE)
BALL_VOICE_PAPA = encode_voice(['b', 'ɔː', 'l'], **PAPA_VOICE)

# Visual variations
BALL_BRIGHT  = encode_patches([   # close up, bright
    {'brightness': 0.8, 'r': 0.9, 'g': 0.3, 'b': 0.1, 'edge_angle': None},
])
BALL_MID = encode_patches([       # mid distance
    {'brightness': 0.55, 'r': 0.85, 'g': 0.3, 'b': 0.1, 'edge_angle': None},
])
BALL_FAR = encode_patches([       # far away, dimmer
    {'brightness': 0.3, 'r': 0.8, 'g': 0.25, 'b': 0.1, 'edge_angle': None},
])

# Simple phrase contexts — ball appears with known words
ROLL_BALL  = encode_text('roll the ball')
BIG_BALL   = encode_text('big ball')
YOUR_BALL  = encode_text('your ball')


def teach(s: System) -> None:

    # ── Phase 1: mama says ball ───────────────────────────────────
    print('\n[Phase 1] Mama says ball — text + mama voice, 15 reps')
    for rep in range(15):
        patches = [BALL_BRIGHT, BALL_MID, BALL_FAR][rep % 3]
        paths, cross, avg = s.teach_concept('ball',
            text_atoms   = BALL_TEXT,
            voice_atoms  = BALL_VOICE_MAMA,
            vision_atoms = patches,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    s.save()

    # ── Phase 2: both voices, alternating ────────────────────────
    print('\n[Phase 2] Both voices alternating, 10 reps each')
    for rep in range(10):
        voice = BALL_VOICE_PAPA if rep % 2 == 0 else BALL_VOICE_MAMA
        patches = [BALL_BRIGHT, BALL_MID][rep % 2]
        paths, cross, avg = s.teach_concept('ball',
            text_atoms   = BALL_TEXT,
            voice_atoms  = voice,
            vision_atoms = patches,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    s.save()

    # ── Phase 3: phrase context ───────────────────────────────────
    print('\n[Phase 3] Ball in phrase context, 5 reps each phrase')
    for phrase_text in [ROLL_BALL, BIG_BALL, YOUR_BALL]:
        for rep in range(5):
            voice = BALL_VOICE_MAMA if rep % 2 == 0 else BALL_VOICE_PAPA
            paths, cross, avg = s.teach_concept(s.text_from_atoms(phrase_text),
                text_atoms   = phrase_text,
                voice_atoms  = voice,
                vision_atoms = BALL_BRIGHT,
                reward       = True,
            )
        phrase_str = ''.join(
            c for c in str(phrase_text)[:20]
        )
        print(f"  phrase taught  cross={cross:.4f}  avg={avg:.0f}")


    # Register concepts in the graph
    s.register_concept('ball')
    s.register_concept('roll the ball')
    s.register_concept('big ball')
    s.register_concept('your ball')
    s.save()


def test(s: System) -> None:

    print('\n[Results]')

    _, ball_text       = s.query(BALL_TEXT,          'text')
    _, ball_papa       = s.query(BALL_VOICE_PAPA,    'voice')
    _, ball_mama       = s.query(BALL_VOICE_MAMA,    'voice')
    _, ball_bright     = s.query(BALL_BRIGHT,        'vision')
    _, ball_mid        = s.query(BALL_MID,           'vision')
    _, ball_far        = s.query(BALL_FAR,           'vision')
    _, roll_ball       = s.query(ROLL_BALL,          'text')

    mama_sig  = encode_voice_signature(**MAMA_VOICE)
    ball_phone = BALL_VOICE_MAMA[0]   # 'vo:b'
    mama_sig_edge = s.graph.get_edge(mama_sig[0], ball_phone) if mama_sig else None
    mama_sig_str  = mama_sig_edge.strength if mama_sig_edge else 0.0

    papa_sig  = encode_voice_signature(**PAPA_VOICE)
    papa_sig_edge = s.graph.get_edge(papa_sig[0], ball_phone) if papa_sig else None
    papa_sig_str  = papa_sig_edge.strength if papa_sig_edge else 0.0

    st = s.state()

    print(f"  Ball text score       : {ball_text:.0f}")
    print(f"  Ball papa voice       : {ball_papa:.0f}")
    print(f"  Ball mama voice       : {ball_mama:.0f}")
    print(f"  Ball bright vision    : {ball_bright:.0f}")
    print(f"  Ball mid   vision     : {ball_mid:.0f}")
    print(f"  Ball far   vision     : {ball_far:.0f}")
    print(f"  Roll ball text        : {roll_ball:.0f}")
    print(f"  Papa sig → /b/        : {papa_sig_str:.0f}")
    print(f"  Mama sig → /b/        : {mama_sig_str:.0f}")
    print(f"  Total nodes           : {st['total_nodes']}")
    print(f"  Total edges           : {st['total_edges']}")
    print(f"  Level-1 nodes         : {st['by_level'].get(1,{}).get('nodes',0)}")

    print('\n[Checks]')
    checks = [
        ('Ball text path is strong',
            ball_text > 100),
        ('Papa voice knows ball',
            ball_papa > 100),
        ('Mama voice knows ball',
            ball_mama > 100),
        ('Bright ball recognised',
            ball_bright > 100),
        ('Mid distance ball recognised',
            ball_mid > 100),
        ('Far ball still recognised',
            ball_far > 100),
        ('Ball in phrase context recognised',
            roll_ball > 100),
        ('Papa sig connected to ball',
            papa_sig_str > 0),
        ('Mama sig connected to ball',
            mama_sig_str > 0),
        ('Both voices know ball',
            ball_papa > 100 and ball_mama > 100),
    ]

    passed = 0
    for label, condition in checks:
        status = 'PASS' if condition else 'FAIL'
        print(f"  {status}  {label}")
        if condition:
            passed += 1

    print(f"\n  {passed}/{len(checks)} passed")
    if passed >= 9:
        print('  Ball concept is solid across both voices and all visual distances.')


if __name__ == '__main__':
    if not os.path.exists('knowledge.json'):
        print('[Lesson 04] Run lessons 01-03 first.')
        sys.exit(1)

    s = System('knowledge.json')
    teach(s)
    test(s)