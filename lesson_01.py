"""
Lesson 01 — Little Deepak learns its name and papa.

Two concepts taught from scratch:

  Concept 1: "little deepak" — its own name
    Text  : l, i, t, t, l, e, d, e, e, p, a, k
    Voice : phonemes + papa's voice signature

  Concept 2: "papa"
    Text  : p, a, p, a
    Voice : phonemes + papa's voice signature

Run:
  python lesson_01.py

This will create knowledge.json in the same folder.
Run again — it loads and continues from where it left off.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import System
from atoms  import encode_text, encode_voice, encode_voice_signature


# ── Voices ────────────────────────────────────────────────────────

PAPA_VOICE     = dict(pitch='mid',  tone='warm',  tempo='mid')
STRANGER_VOICE = dict(pitch='high', tone='sharp', tempo='fast')

# ── Concepts ──────────────────────────────────────────────────────

NAME_TEXT  = encode_text('little deepak')
NAME_VOICE = encode_voice(
    ['l', 'ɪ', 't', 'ə', 'l', 'd', 'iː', 'p', 'æ', 'k'],
    **PAPA_VOICE
)

PAPA_TEXT  = encode_text('papa')
PAPA_VOICE_ATOMS    = encode_voice(['p', 'æ', 'p', 'ə'], **PAPA_VOICE)
PAPA_VOICE_STRANGER = encode_voice(['p', 'æ', 'p', 'ə'], **STRANGER_VOICE)


def teach(s: System, reps: int = 20) -> None:

    print('\n[Teaching: little deepak]')
    for rep in range(reps):
        paths, cross, avg = s.teach_concept('little deepak',
            text_atoms  = NAME_TEXT,
            voice_atoms = NAME_VOICE,
            reward      = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg_intensity={avg:.2f}")

    s.save()

    print('\n[Teaching: papa]')
    for rep in range(reps):
        paths, cross, avg = s.teach_concept('papa',
            text_atoms  = PAPA_TEXT,
            voice_atoms = PAPA_VOICE_ATOMS,
            reward      = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg_intensity={avg:.2f}")


    # Register concepts in the graph
    s.register_concept('little deepak')
    s.register_concept('papa')
    s.save()


def test(s: System) -> None:

    print('\n[Results]')

    _, name_score  = s.query(NAME_TEXT,         'text')
    _, papa_score  = s.query(PAPA_TEXT,         'text')
    _, name_vscore = s.query(NAME_VOICE,        'voice')
    _, papa_vscore = s.query(PAPA_VOICE_ATOMS,  'voice')
    _, papa_stranger_score = s.query(PAPA_VOICE_STRANGER, 'voice')

    papa_sig      = encode_voice_signature(**PAPA_VOICE)
    stranger_sig  = encode_voice_signature(**STRANGER_VOICE)
    papa_phoneme  = PAPA_VOICE_ATOMS[0]

    papa_sig_edge     = s.graph.get_edge(papa_sig[0],     papa_phoneme) if papa_sig     else None
    stranger_sig_edge = s.graph.get_edge(stranger_sig[0], papa_phoneme) if stranger_sig else None

    papa_sig_strength     = papa_sig_edge.strength     if papa_sig_edge     else 0.0
    stranger_sig_strength = stranger_sig_edge.strength if stranger_sig_edge else 0.0

    st = s.state()

    print(f"  Name  text  score : {name_score:.2f}")
    print(f"  Papa  text  score : {papa_score:.2f}")
    print(f"  Name  voice score : {name_vscore:.2f}")
    print(f"  Papa  voice score : {papa_vscore:.2f}")
    print(f"  Papa by stranger  : {papa_stranger_score:.2f}")
    print(f"  Papa sig strength : {papa_sig_strength:.2f}")
    print(f"  Stranger sig str  : {stranger_sig_strength:.2f}")
    print(f"  Total nodes       : {st['total_nodes']}")
    print(f"  Total edges       : {st['total_edges']}")
    print(f"  Levels formed     : {sorted(st['by_level'].keys())}")

    print('\n[Checks]')
    checks = [
        ('Name path is strong',                   name_score  > 100),
        ('Papa path is strong',                   papa_score  > 100),
        ('Name and papa are distinct',            abs(name_score - papa_score) > 10),
        ('Name voice path is strong',             name_vscore > 100),
        ('Papa voice path is strong',             papa_vscore > 100),
        ("Papa's voice knows papa, stranger's doesn't",
                                                  papa_sig_strength > stranger_sig_strength),
        ("Papa's voice scores higher than stranger's",
                                                  papa_vscore > papa_stranger_score),
        ('Abstractions have formed',              max(st['by_level'].keys()) >= 1),
    ]

    passed = 0
    for label, condition in checks:
        status = 'PASS' if condition else 'FAIL'
        print(f"  {status}  {label}")
        if condition:
            passed += 1

    print(f"\n  {passed}/{len(checks)} passed")
    if passed == len(checks):
        print('  Little Deepak knows its name and papa.')


if __name__ == '__main__':
    s = System('knowledge.json')
    teach(s, reps=20)
    test(s)