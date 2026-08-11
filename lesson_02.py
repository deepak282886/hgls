"""
Lesson 02 — Little Deepak learns mama.

Builds on Lesson 01. Teaches one new concept:

  Concept: "mama"
    Text  : m, a, m, a
    Voice : phonemes + mama's voice signature
            (high pitch, warm tone, slow tempo — distinct from papa)

Key distinction from papa:
  /m/ is unique to mama in what the system has learned so far.
  /p/ is unique to papa.
  Both share /æ/ and /ə/ — same vowel sounds.
  The voice signatures are different.
  Together these make mama a distinct concept.

Run:
  python lesson_01.py   (first time only)
  python lesson_02.py

Continues from existing knowledge.json.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import System
from atoms  import encode_text, encode_voice, encode_voice_signature


# ── Voices ────────────────────────────────────────────────────────

MAMA_VOICE = dict(pitch='high', tone='warm', tempo='slow')
PAPA_VOICE = dict(pitch='mid',  tone='warm', tempo='mid')

# ── Concepts ──────────────────────────────────────────────────────

MAMA_TEXT        = encode_text('mama')
MAMA_VOICE_ATOMS = encode_voice(['m', 'æ', 'm', 'ə'], **MAMA_VOICE)
MAMA_BY_PAPA     = encode_voice(['m', 'æ', 'm', 'ə'], **PAPA_VOICE)

PAPA_TEXT        = encode_text('papa')
PAPA_VOICE_ATOMS = encode_voice(['p', 'æ', 'p', 'ə'], **PAPA_VOICE)
NAME_TEXT        = encode_text('little deepak')

MAMA_SIG = encode_voice_signature(**MAMA_VOICE)
PAPA_SIG = encode_voice_signature(**PAPA_VOICE)

# /m/ is unique to mama in what has been taught — use it as the
# specific phoneme to test voice signature attachment
MAMA_UNIQUE_PHONEME = 'vo:m'   # not present in papa or name
PAPA_UNIQUE_PHONEME = 'vo:p'   # not present in mama


def teach(s: System, reps: int = 25) -> None:

    print('\n[Teaching: mama]')
    for rep in range(reps):
        paths, cross, avg = s.learn_multi(
            text_atoms  = MAMA_TEXT,
            voice_atoms = MAMA_VOICE_ATOMS,
            reward      = True,
        )
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1:2d}  cross={cross:.4f}  avg_intensity={avg:.2f}")

    s.save()


def test(s: System) -> None:

    print('\n[Results]')

    _, mama_score    = s.query(MAMA_TEXT,        'text')
    _, papa_score    = s.query(PAPA_TEXT,        'text')
    _, name_score    = s.query(NAME_TEXT,        'text')
    _, mama_vscore   = s.query(MAMA_VOICE_ATOMS, 'voice')
    _, papa_vscore   = s.query(PAPA_VOICE_ATOMS, 'voice')

    # Sig → unique phoneme edges — the meaningful distinction
    mama_sig_to_mama_m = s.graph.get_edge(MAMA_SIG[0], MAMA_UNIQUE_PHONEME)
    papa_sig_to_mama_m = s.graph.get_edge(PAPA_SIG[0], MAMA_UNIQUE_PHONEME)
    mama_sig_to_papa_p = s.graph.get_edge(MAMA_SIG[0], PAPA_UNIQUE_PHONEME)
    papa_sig_to_papa_p = s.graph.get_edge(PAPA_SIG[0], PAPA_UNIQUE_PHONEME)

    mama_sig_mama_str = mama_sig_to_mama_m.strength if mama_sig_to_mama_m else 0.0
    papa_sig_mama_str = papa_sig_to_mama_m.strength if papa_sig_to_mama_m else 0.0
    mama_sig_papa_str = mama_sig_to_papa_p.strength if mama_sig_to_papa_p else 0.0
    papa_sig_papa_str = papa_sig_to_papa_p.strength if papa_sig_to_papa_p else 0.0

    st = s.state()
    nodes_at_l1 = st['by_level'].get(1, {}).get('nodes', 0)

    print(f"  Mama  text  score         : {mama_score:.2f}")
    print(f"  Papa  text  score         : {papa_score:.2f}")
    print(f"  Name  text  score         : {name_score:.2f}")
    print(f"  Mama  voice score         : {mama_vscore:.2f}")
    print(f"  Papa  voice score         : {papa_vscore:.2f}")
    print(f"  Mama sig → /m/ (mama)     : {mama_sig_mama_str:.2f}")
    print(f"  Papa sig → /m/ (mama)     : {papa_sig_mama_str:.2f}")
    print(f"  Mama sig → /p/ (papa)     : {mama_sig_papa_str:.2f}")
    print(f"  Papa sig → /p/ (papa)     : {papa_sig_papa_str:.2f}")
    print(f"  Total nodes               : {st['total_nodes']}")
    print(f"  Total edges               : {st['total_edges']}")
    print(f"  Level-1 abstract nodes    : {nodes_at_l1}")

    print('\n[Checks]')
    checks = [
        ('Mama text path is strong',
            mama_score > 100),

        # mama and papa share tx:a which dominates text score.
        # Real distinction is in their unique phonemes: /m/ vs /p/.
        # Check voice distinctness instead.
        ('Mama voice is distinct from papa voice',
            abs(mama_vscore - papa_vscore) > 10),

        ('Mama is distinct from name (text)',
            abs(mama_score - name_score) > 10),

        ('Mama voice path is strong',
            mama_vscore > 100),

        ("Mama's voice sig connected to mama's unique phoneme /m/",
            mama_sig_mama_str > 0),

        ("Mama's sig knows /m/ better than papa's sig does",
            mama_sig_mama_str > papa_sig_mama_str),

        ("Papa's sig knows /p/ better than mama's sig does",
            papa_sig_papa_str > mama_sig_papa_str),

        ('Level-1 abstractions exist',
            nodes_at_l1 > 0),
    ]

    passed = 0
    for label, condition in checks:
        status = 'PASS' if condition else 'FAIL'
        print(f"  {status}  {label}")
        if condition:
            passed += 1

    print(f"\n  {passed}/{len(checks)} passed")
    if passed == len(checks):
        print('  Little Deepak knows mama, papa, and its own name.')


if __name__ == '__main__':
    if not os.path.exists('knowledge.json'):
        print('[Lesson 02] Run lesson_01.py first to build knowledge.json')
        sys.exit(1)

    s = System('knowledge.json')
    teach(s, reps=25)
    test(s)