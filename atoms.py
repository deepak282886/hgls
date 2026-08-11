"""
atoms.py — Primitives for all three modalities plus voice signature.

Text     : 41 characters (a-z, 0-9, space, basic punctuation)
Voice    : 41 English phonemes (IPA) + 27 voice signature atoms
Vision   : 36 pixel patch atoms (brightness × color × edge)

Voice has two layers:
  phonemes        — WHAT is said  (content)
  voice signature — WHO is saying it (acoustic character)
    pitch × tone × tempo = 3 × 3 × 3 = 27 atoms

All atoms are level 0 nodes in the graph.
Bootstrap seeds the graph with all atoms at startup.
"""

from typing import List, Optional
from memory import Graph, Node


# ── Text atoms ────────────────────────────────────────────────────

TEXT_ATOMS = list(
    'abcdefghijklmnopqrstuvwxyz'
    '0123456789'
    ' .,!?'
)

# ── Voice atoms — phonemes (IPA English) ─────────────────────────

CONSONANTS = [
    'p', 'b', 't', 'd', 'k', 'g',
    'f', 'v', 'θ', 'ð', 's', 'z',
    'ʃ', 'ʒ', 'h',
    'tʃ', 'dʒ',
    'm', 'n', 'ŋ',
    'l', 'r', 'w', 'j',
]

VOWELS = [
    'iː', 'ɪ', 'e', 'æ',
    'ɑː', 'ɒ', 'ɔː', 'ʊ', 'uː',
    'ʌ', 'ɜː', 'ə',
    'eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'əʊ',
]

VOICE_ATOMS = CONSONANTS + VOWELS

# ── Voice signature atoms — WHO is speaking ───────────────────────
# Three features, each with three discrete values.
# 3 × 3 × 3 = 27 atoms.
#
# pitch : low  | mid  | high
# tone  : warm | sharp | breathy
# tempo : slow | mid  | fast
#
# Each combination is a unique atom describing
# the acoustic character of one voice.
# A child learns to distinguish mama's voice from papa's voice
# because their signature atoms are different even when
# the phonemes are the same.

PITCH_VALUES = ['low',  'mid',  'high']
TONE_VALUES  = ['warm', 'sharp', 'breathy']
TEMPO_VALUES = ['slow', 'mid',  'fast']

def _sig_atom_id(pitch: str, tone: str, tempo: str) -> str:
    return f"vs:{pitch}:{tone}:{tempo}"

VOICE_SIG_ATOMS = [
    _sig_atom_id(pitch, tone, tempo)
    for pitch in PITCH_VALUES
    for tone  in TONE_VALUES
    for tempo in TEMPO_VALUES
]

# ── Vision atoms ──────────────────────────────────────────────────
# brightness × color × edge = 3 × 3 × 4 = 36 atoms

BRIGHTNESS_VALUES = ['dark',  'mid',   'bright']
COLOR_VALUES      = ['warm',  'cool',  'neutral']
EDGE_VALUES       = ['none',  'horizontal', 'vertical', 'diagonal']

def _vision_atom_id(brightness: str, color: str, edge: str) -> str:
    return f"px:{brightness}:{color}:{edge}"

VISION_ATOMS = [
    _vision_atom_id(b, c, e)
    for b in BRIGHTNESS_VALUES
    for c in COLOR_VALUES
    for e in EDGE_VALUES
]


# ── Bootstrap ─────────────────────────────────────────────────────

def bootstrap(graph: Graph) -> None:
    """
    Seed the graph with all atoms across all modalities.
    Called once at startup if graph is empty.
    All atoms start at strength 0 — nothing pre-weighted.
    """
    count = 0

    for char in TEXT_ATOMS:
        node = Node(
            id       = f"tx:{char}",
            level    = 0,
            modality = 'text',
            elements = [char],
            strength = 0.0,
        )
        if not graph.has_node(node.id):
            graph.add_node(node)
            count += 1

    for phoneme in VOICE_ATOMS:
        node = Node(
            id       = f"vo:{phoneme}",
            level    = 0,
            modality = 'voice',
            elements = [phoneme],
            strength = 0.0,
        )
        if not graph.has_node(node.id):
            graph.add_node(node)
            count += 1

    for sig in VOICE_SIG_ATOMS:
        node = Node(
            id       = sig,
            level    = 0,
            modality = 'voice_sig',
            elements = [sig],
            strength = 0.0,
        )
        if not graph.has_node(node.id):
            graph.add_node(node)
            count += 1

    for patch_id in VISION_ATOMS:
        node = Node(
            id       = patch_id,
            level    = 0,
            modality = 'vision',
            elements = [patch_id],
            strength = 0.0,
        )
        if not graph.has_node(node.id):
            graph.add_node(node)
            count += 1

    print(f"[Atoms] Bootstrapped {count} primitives — "
          f"text={len(TEXT_ATOMS)} "
          f"phonemes={len(VOICE_ATOMS)} "
          f"voice_sig={len(VOICE_SIG_ATOMS)} "
          f"vision={len(VISION_ATOMS)}")


# ── Encoders ──────────────────────────────────────────────────────

def encode_text(text: str) -> List[str]:
    """Map a string to text atom node ids. Unknown chars skipped."""
    known = {f"tx:{c}" for c in TEXT_ATOMS}
    return [f"tx:{c}" for c in text.lower() if f"tx:{c}" in known]


def encode_phonemes(phoneme_sequence: List[str]) -> List[str]:
    """Map a phoneme sequence to phoneme atom node ids."""
    known = {f"vo:{p}" for p in VOICE_ATOMS}
    return [f"vo:{p}" for p in phoneme_sequence if f"vo:{p}" in known]


def encode_voice_signature(
    pitch: str,   # 'low' | 'mid' | 'high'
    tone:  str,   # 'warm' | 'sharp' | 'breathy'
    tempo: str,   # 'slow' | 'mid' | 'fast'
) -> List[str]:
    """
    Encode a voice signature as a single atom node id.
    Returns a list for consistency with other encoders.
    """
    sid = _sig_atom_id(pitch, tone, tempo)
    if sid in {s for s in VOICE_SIG_ATOMS}:
        return [sid]
    return []


def encode_voice(
    phoneme_sequence: List[str],
    pitch: str = 'mid',
    tone:  str = 'warm',
    tempo: str = 'mid',
) -> List[str]:
    """
    Encode a full voice input — phonemes + signature together.
    This is the main encoder for voice input.

    phoneme_sequence : what is being said
    pitch/tone/tempo : who is saying it

    Both layers activate together so the system learns
    to associate content with speaker.
    """
    return encode_phonemes(phoneme_sequence) + \
           encode_voice_signature(pitch, tone, tempo)


def encode_patches(patch_list: List[dict]) -> List[str]:
    """
    Map a list of patch feature dicts to vision atom node ids.
    Each dict: {brightness, r, g, b, edge_angle}
    """
    return [_encode_patch(p) for p in patch_list]


def _encode_patch(p: dict) -> str:
    brightness = p['brightness']
    r, g, b    = p['r'], p['g'], p['b']
    edge_angle = p.get('edge_angle')

    if brightness < 0.35:
        b_label = 'dark'
    elif brightness < 0.7:
        b_label = 'mid'
    else:
        b_label = 'bright'

    warmth = r - b
    if warmth > 0.1:
        c_label = 'warm'
    elif warmth < -0.1:
        c_label = 'cool'
    else:
        c_label = 'neutral'

    if edge_angle is None:
        e_label = 'none'
    elif edge_angle < 22.5 or edge_angle >= 157.5:
        e_label = 'horizontal'
    elif 67.5 <= edge_angle < 112.5:
        e_label = 'vertical'
    else:
        e_label = 'diagonal'

    return _vision_atom_id(b_label, c_label, e_label)