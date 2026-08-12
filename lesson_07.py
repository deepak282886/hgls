"""
Lesson 07 — Spoon reinforcement + Book (core concept).

  Part A — Spoon reinforcement (mama's voice, action context)
    "eat with the spoon", "your spoon", "little spoon"

  Part B — Book (core concept, papa's voice then mama's)
    Text  : b, o, o, k
    Voice : /b/ /ʊ/ /k/ + papa's voice, then mama's
    Vision: mid brightness, cool (white pages), horizontal edges (flat pages)

Run:
  python lesson_07.py

Requires knowledge.json from lesson_06.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import System
from atoms  import encode_text, encode_voice, encode_voice_signature, encode_patches


MAMA_VOICE = dict(pitch='high', tone='warm', tempo='slow')
PAPA_VOICE = dict(pitch='mid',  tone='warm', tempo='mid')

# ── Spoon reinforcement ───────────────────────────────────────────

SPOON_TEXT       = encode_text('spoon')
SPOON_VOICE_MAMA = encode_voice(['s', 'p', 'uː', 'n'], **MAMA_VOICE)
SPOON_VOICE_PAPA = encode_voice(['s', 'p', 'uː', 'n'], **PAPA_VOICE)
SPOON_PATCHES    = encode_patches([
    {'brightness': 0.85, 'r': 0.75, 'g': 0.75, 'b': 0.75, 'edge_angle': 50.0},
])

EAT_SPOON   = encode_text('eat with the spoon')
YOUR_SPOON  = encode_text('your spoon')
LITTLE_SPOON = encode_text('little spoon')

# ── Book concept ──────────────────────────────────────────────────

BOOK_TEXT       = encode_text('book')
BOOK_VOICE_PAPA = encode_voice(['b', 'ʊ', 'k'], **PAPA_VOICE)
BOOK_VOICE_MAMA = encode_voice(['b', 'ʊ', 'k'], **MAMA_VOICE)

# Book: mid brightness, cool (white pages), horizontal (flat edge)
BOOK_PATCHES_OPEN = encode_patches([
    {'brightness': 0.65, 'r': 0.6, 'g': 0.6, 'b': 0.75, 'edge_angle': 5.0},
])
# Book closed — darker spine, vertical edge
BOOK_PATCHES_CLOSED = encode_patches([
    {'brightness': 0.4, 'r': 0.4, 'g': 0.4, 'b': 0.55, 'edge_angle': 80.0},
])

READ_BOOK  = encode_text('read the book')
YOUR_BOOK  = encode_text('your book')
BIG_BOOK   = encode_text('big book')


def teach(s: System) -> None:

    # ── Part A: Spoon reinforcement ───────────────────────────────
    print('\n[Part A] Spoon — mama\'s voice, 10 reps')
    for rep in range(10):
        voice = SPOON_VOICE_MAMA if rep % 2 == 0 else SPOON_VOICE_PAPA
        paths, cross, avg = s.learn_multi(
            text_atoms   = SPOON_TEXT,
            voice_atoms  = voice,
            vision_atoms = SPOON_PATCHES,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    print('\n[Part A] Spoon in phrase context, 5 reps each')
    for phrase in [EAT_SPOON, YOUR_SPOON, LITTLE_SPOON]:
        for rep in range(5):
            voice = SPOON_VOICE_MAMA if rep % 2 == 0 else SPOON_VOICE_PAPA
            s.learn_multi(
                text_atoms   = phrase,
                voice_atoms  = voice,
                vision_atoms = SPOON_PATCHES,
                reward       = True,
            )
        print(f"  phrase done")

    s.save()

    # ── Part B: Book core ─────────────────────────────────────────
    print('\n[Part B] Book — word alone, papa first, 10 reps')
    for rep in range(10):
        paths, cross, avg = s.learn_multi(
            text_atoms  = BOOK_TEXT,
            voice_atoms = BOOK_VOICE_PAPA,
            reward      = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    print('\n[Part B] Book — word + sight, open + closed, 15 reps')
    for rep in range(15):
        voice   = BOOK_VOICE_PAPA if rep % 2 == 0 else BOOK_VOICE_MAMA
        patches = BOOK_PATCHES_OPEN if rep % 3 != 2 else BOOK_PATCHES_CLOSED
        paths, cross, avg = s.learn_multi(
            text_atoms   = BOOK_TEXT,
            voice_atoms  = voice,
            vision_atoms = patches,
            reward       = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg={avg:.0f}")

    print('\n[Part B] Book in phrase context, 5 reps each')
    for phrase in [READ_BOOK, YOUR_BOOK, BIG_BOOK]:
        for rep in range(5):
            voice = BOOK_VOICE_MAMA if rep % 2 == 0 else BOOK_VOICE_PAPA
            s.learn_multi(
                text_atoms   = phrase,
                voice_atoms  = voice,
                vision_atoms = BOOK_PATCHES_OPEN,
                reward       = True,
            )
        print(f"  phrase done")

    s.save()


def test(s: System) -> None:

    print('\n[Results]')

    _, spoon_text  = s.query(SPOON_TEXT,           'text')
    _, spoon_mama  = s.query(SPOON_VOICE_MAMA,     'voice')
    _, book_text   = s.query(BOOK_TEXT,            'text')
    _, book_papa   = s.query(BOOK_VOICE_PAPA,      'voice')
    _, book_mama   = s.query(BOOK_VOICE_MAMA,      'voice')
    _, book_open   = s.query(BOOK_PATCHES_OPEN,    'vision')
    _, book_closed = s.query(BOOK_PATCHES_CLOSED,  'vision')
    _, ball_text   = s.query(encode_text('ball'),  'text')
    _, cup_text    = s.query(encode_text('cup'),   'text')

    mama_sig  = encode_voice_signature(**MAMA_VOICE)
    papa_sig  = encode_voice_signature(**PAPA_VOICE)
    book_phone = BOOK_VOICE_PAPA[0]
    book_papa_sig = s.graph.get_edge(papa_sig[0], book_phone) if papa_sig else None
    book_mama_sig = s.graph.get_edge(mama_sig[0], book_phone) if mama_sig else None

    st = s.state()

    print(f"  Spoon text        : {spoon_text:.0f}")
    print(f"  Spoon mama voice  : {spoon_mama:.0f}")
    print(f"  Book text         : {book_text:.0f}")
    print(f"  Book papa voice   : {book_papa:.0f}")
    print(f"  Book mama voice   : {book_mama:.0f}")
    print(f"  Book open vision  : {book_open:.0f}")
    print(f"  Book closed vision: {book_closed:.0f}")
    print(f"  Ball text         : {ball_text:.0f}")
    print(f"  Cup  text         : {cup_text:.0f}")
    print(f"  Total nodes       : {st['total_nodes']}")
    print(f"  Total edges       : {st['total_edges']}")
    print(f"  Level-1 nodes     : {st['by_level'].get(1,{}).get('nodes',0)}")

    print('\n[Checks]')
    checks = [
        ('Spoon known by mama voice',         spoon_mama  > 100),
        ('Book text path strong',             book_text   > 100),
        ('Book known by papa voice',          book_papa   > 100),
        ('Book known by mama voice',          book_mama   > 100),
        ('Open book recognised',              book_open   > 100),
        ('Closed book recognised',            book_closed > 100),
        ('Book distinct from ball',           abs(book_text - ball_text) > 10),
        ('Book distinct from cup',            abs(book_text - cup_text)  > 10),
        ('Papa sig connected to book',
            book_papa_sig is not None and book_papa_sig.strength > 0),
        ('Mama sig connected to book',
            book_mama_sig is not None and book_mama_sig.strength > 0),
    ]

    passed = 0
    for label, condition in checks:
        status = 'PASS' if condition else 'FAIL'
        print(f"  {status}  {label}")
        if condition:
            passed += 1

    print(f"\n  {passed}/{len(checks)} passed")
    if passed >= 9:
        print('  Little Deepak knows spoon (both voices) and book (both voices).')


if __name__ == '__main__':
    if not os.path.exists('knowledge.json'):
        print('[Lesson 07] Run lessons 01-06 first.')
        sys.exit(1)
    s = System('knowledge.json')
    teach(s)
    test(s)