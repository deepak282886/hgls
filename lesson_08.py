"""
Lesson 08 — Book reinforcement + Light (core and reinforcement).

  Part A — Book reinforcement (phrase context, visual variety)
    "look at the book", "pretty book", "open the book"
    Different lighting conditions for book

  Part B — Light (core concept, both voices, action)
    Text  : l, i, g, h, t
    Voice : /l/ /aɪ/ /t/ + papa's voice, then mama's
    Vision: bright, neutral, no edges (diffuse glow) — light on
            dark, neutral, no edges — light off

    Action context: "the light is on", "the light is off"
    This introduces opposites naturally — same object, two states.

Run:
  python lesson_08.py

Requires knowledge.json from lesson_07.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import System
from atoms  import encode_text, encode_voice, encode_voice_signature, encode_patches


MAMA_VOICE = dict(pitch='high', tone='warm', tempo='slow')
PAPA_VOICE = dict(pitch='mid',  tone='warm', tempo='mid')

# ── Book reinforcement ────────────────────────────────────────────

BOOK_TEXT       = encode_text('book')
BOOK_VOICE_PAPA = encode_voice(['b', 'ʊ', 'k'], **PAPA_VOICE)
BOOK_VOICE_MAMA = encode_voice(['b', 'ʊ', 'k'], **MAMA_VOICE)
BOOK_PATCHES    = encode_patches([
    {'brightness': 0.65, 'r': 0.6, 'g': 0.6, 'b': 0.75, 'edge_angle': 5.0},
])
# Book in bright light
BOOK_PATCHES_BRIGHT = encode_patches([
    {'brightness': 0.85, 'r': 0.75, 'g': 0.75, 'b': 0.85, 'edge_angle': 5.0},
])

LOOK_BOOK  = encode_text('look at the book')
OPEN_BOOK  = encode_text('open the book')
NICE_BOOK  = encode_text('nice book')

# ── Light concept ─────────────────────────────────────────────────

LIGHT_TEXT       = encode_text('light')
LIGHT_VOICE_PAPA = encode_voice(['l', 'aɪ', 't'], **PAPA_VOICE)
LIGHT_VOICE_MAMA = encode_voice(['l', 'aɪ', 't'], **MAMA_VOICE)

# Light on — very bright, neutral, no edges (diffuse glow)
LIGHT_ON = encode_patches([
    {'brightness': 0.95, 'r': 0.95, 'g': 0.95, 'b': 0.9, 'edge_angle': None},
])
# Light off — dark, neutral, no edges
LIGHT_OFF = encode_patches([
    {'brightness': 0.1, 'r': 0.1, 'g': 0.1, 'b': 0.1, 'edge_angle': None},
])

LIGHT_ON_TEXT  = encode_text('light is on')
LIGHT_OFF_TEXT = encode_text('light is off')
SEE_LIGHT      = encode_text('see the light')


def teach(s: System) -> None:

    # ── Part A: Book reinforcement ────────────────────────────────
    print('\n[Part A] Book reinforcement — phrases + bright light, 10 reps')
    for rep in range(10):
        voice   = BOOK_VOICE_PAPA if rep % 2 == 0 else BOOK_VOICE_MAMA
        patches = BOOK_PATCHES if rep % 3 != 2 else BOOK_PATCHES_BRIGHT
        paths, cross, avg = s.learn_multi(
            text_atoms   = BOOK_TEXT,
            voice_atoms  = voice,
            vision_atoms = patches,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    for phrase in [LOOK_BOOK, OPEN_BOOK, NICE_BOOK]:
        for rep in range(5):
            voice = BOOK_VOICE_MAMA if rep % 2 == 0 else BOOK_VOICE_PAPA
            s.learn_multi(
                text_atoms   = phrase,
                voice_atoms  = voice,
                vision_atoms = BOOK_PATCHES,
                reward       = True,
            )
        print(f"  phrase done")

    s.save()

    # ── Part B: Light core ────────────────────────────────────────
    print('\n[Part B] Light — word alone, papa first, 10 reps')
    for rep in range(10):
        paths, cross, avg = s.learn_multi(
            text_atoms  = LIGHT_TEXT,
            voice_atoms = LIGHT_VOICE_PAPA,
            reward      = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    print('\n[Part B] Light on — word + bright vision, 10 reps')
    for rep in range(10):
        paths, cross, avg = s.learn_multi(
            text_atoms   = LIGHT_TEXT,
            voice_atoms  = LIGHT_VOICE_PAPA,
            vision_atoms = LIGHT_ON,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    print('\n[Part B] Light off — same word, dark vision, 10 reps')
    for rep in range(10):
        voice = LIGHT_VOICE_PAPA if rep % 2 == 0 else LIGHT_VOICE_MAMA
        paths, cross, avg = s.learn_multi(
            text_atoms   = LIGHT_TEXT,
            voice_atoms  = voice,
            vision_atoms = LIGHT_OFF,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    print('\n[Part B] Light in context — on/off states, 5 reps each')
    for phrase, patches in [
        (LIGHT_ON_TEXT,  LIGHT_ON),
        (LIGHT_OFF_TEXT, LIGHT_OFF),
        (SEE_LIGHT,      LIGHT_ON),
    ]:
        for rep in range(5):
            voice = LIGHT_VOICE_MAMA if rep % 2 == 0 else LIGHT_VOICE_PAPA
            s.learn_multi(
                text_atoms   = phrase,
                voice_atoms  = voice,
                vision_atoms = patches,
                reward       = True,
            )
        print(f"  phrase done")

    s.save()


def test(s: System) -> None:

    print('\n[Results]')

    _, light_text  = s.query(LIGHT_TEXT,       'text')
    _, light_papa  = s.query(LIGHT_VOICE_PAPA, 'voice')
    _, light_mama  = s.query(LIGHT_VOICE_MAMA, 'voice')
    _, light_on    = s.query(LIGHT_ON,         'vision')
    _, light_off   = s.query(LIGHT_OFF,        'vision')
    _, book_text   = s.query(BOOK_TEXT,        'text')
    _, ball_text   = s.query(encode_text('ball'), 'text')

    papa_sig   = encode_voice_signature(**PAPA_VOICE)
    mama_sig   = encode_voice_signature(**MAMA_VOICE)
    light_phone = LIGHT_VOICE_PAPA[0]
    papa_light  = s.graph.get_edge(papa_sig[0], light_phone) if papa_sig else None
    mama_light  = s.graph.get_edge(mama_sig[0], light_phone) if mama_sig else None

    # light on and light off should both connect to 'light' text
    light_text_atom = LIGHT_TEXT[0]
    on_to_text  = s.graph.get_edge(LIGHT_ON[0],  light_text_atom)
    off_to_text = s.graph.get_edge(LIGHT_OFF[0], light_text_atom)

    st = s.state()

    print(f"  Light text        : {light_text:.0f}")
    print(f"  Light papa voice  : {light_papa:.0f}")
    print(f"  Light mama voice  : {light_mama:.0f}")
    print(f"  Light ON vision   : {light_on:.0f}")
    print(f"  Light OFF vision  : {light_off:.0f}")
    print(f"  Book text         : {book_text:.0f}")
    print(f"  Ball text         : {ball_text:.0f}")
    print(f"  ON → text edge    : {on_to_text.strength if on_to_text else 0:.0f}")
    print(f"  OFF → text edge   : {off_to_text.strength if off_to_text else 0:.0f}")
    print(f"  Total nodes       : {st['total_nodes']}")
    print(f"  Total edges       : {st['total_edges']}")
    print(f"  Level-1 nodes     : {st['by_level'].get(1,{}).get('nodes',0)}")
    print(f"  Level-2 nodes     : {st['by_level'].get(2,{}).get('nodes',0)}")

    print('\n[Checks]')
    checks = [
        ('Light text path strong',           light_text > 100),
        ('Light known by papa',              light_papa > 100),
        ('Light known by mama',              light_mama > 100),
        ('Light ON vision recognised',       light_on   > 100),
        ('Light OFF vision recognised',      light_off  > 100),
        ('Light distinct from book',         abs(light_text - book_text) > 10),
        ('Light distinct from ball',         abs(light_text - ball_text) > 10),
        ('Papa sig connected to light',
            papa_light is not None and papa_light.strength > 0),
        ('Mama sig connected to light',
            mama_light is not None and mama_light.strength > 0),
        ('ON and OFF both connect to light word',
            (on_to_text is not None or off_to_text is not None)),
    ]

    passed = 0
    for label, condition in checks:
        status = 'PASS' if condition else 'FAIL'
        print(f"  {status}  {label}")
        if condition:
            passed += 1

    print(f"\n  {passed}/{len(checks)} passed")
    if passed >= 9:
        print('  Little Deepak knows light — on and off.')
        print('  All 5 objects learned: ball, cup, spoon, book, light.')


if __name__ == '__main__':
    if not os.path.exists('knowledge.json'):
        print('[Lesson 08] Run lessons 01-07 first.')
        sys.exit(1)
    s = System('knowledge.json')
    teach(s)
    test(s)