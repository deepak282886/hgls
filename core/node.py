"""
node.py

Defines the Node — the fundamental unit of the graph.

Every node represents an event at a specific level of the hierarchy.
A character, a word, a sentence, a paragraph, a document —
all are nodes. Same structure. Different level. Different embedding size.

What a node holds:
    - Stable integer ID — never changes even as meaning evolves
    - Raw text — the signal this node represents
    - Level — where in the hierarchy this node lives
    - Embedding — fixed-size float vector encoding the node's meaning
    - Reward — signed scalar, how valuable this node is
    - V — Bellman value, best cumulative reward reachable from here
    - Children — IDs of Level N-1 nodes this node is composed of
    - Parents — IDs of Level N+1 nodes this node contributes to
    - Visit count — how many times MCTS has visited this node
    - Created at — ingestion timestamp for delta propagation ordering
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from core.atoms import Level, EMBEDDING_DIM, REWARD_SCALE, text_to_characters, char_to_id


# ─────────────────────────────────────────────
# NODE DATACLASS
# ─────────────────────────────────────────────

@dataclass
class Node:
    # Identity
    node_id    : int
    text       : str
    level      : Level

    # Numerical representation — what MCTS sees
    embedding  : np.ndarray        # shape: (EMBEDDING_DIM[level],)
    reward     : float = 0.0       # signed, raw (before normalization)
    norm_reward: float = 0.0       # normalized reward (set by normalizer)
    V          : float = 0.0       # Bellman value (recomputed by bellman.py)
    norm_V     : float = 0.0       # normalized V (set by normalizer)

    # Graph connectivity — stored as integer IDs
    # Actual edge objects live in edge.py / graph.py
    children   : list[int] = field(default_factory=list)  # level N-1 nodes
    parents    : list[int] = field(default_factory=list)   # level N+1 nodes

    # MCTS statistics
    visit_count : int   = 0
    total_reward: float = 0.0   # accumulated reward across MCTS visits

    # Metadata
    created_at  : float = field(default_factory=time.time)
    is_terminal : bool  = False   # no outgoing edges — natural endpoint


    def mean_reward(self) -> float:
        """Average reward seen across MCTS visits. Used in UCB."""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count


    def dim(self) -> int:
        """Embedding dimension for this node's level."""
        return EMBEDDING_DIM[self.level]


    def reward_scale(self) -> float:
        """Reward normalization anchor for this node's level."""
        return REWARD_SCALE[self.level]


    def to_dict(self) -> dict:
        """
        Serialize to dict for SQLite storage.
        Embedding stored as raw bytes.
        """
        return {
            "node_id"    : self.node_id,
            "text"       : self.text,
            "level"      : int(self.level),
            "embedding"  : self.embedding.tobytes(),
            "reward"     : self.reward,
            "norm_reward": self.norm_reward,
            "V"          : self.V,
            "norm_V"     : self.norm_V,
            "visit_count": self.visit_count,
            "total_reward": self.total_reward,
            "created_at" : self.created_at,
            "is_terminal": int(self.is_terminal),
            "children"   : ",".join(str(c) for c in self.children),
            "parents"    : ",".join(str(p) for p in self.parents),
        }


    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        """Deserialize from SQLite row dict."""
        level = Level(d["level"])
        dim   = EMBEDDING_DIM[level]
        emb   = np.frombuffer(d["embedding"], dtype=np.float32).copy()

        # Pad or trim if dim mismatch (schema evolution safety)
        if len(emb) != dim:
            tmp = np.zeros(dim, dtype=np.float32)
            tmp[:min(len(emb), dim)] = emb[:min(len(emb), dim)]
            emb = tmp

        children_raw = d.get("children", "")
        parents_raw  = d.get("parents",  "")

        return cls(
            node_id     = d["node_id"],
            text        = d["text"],
            level       = level,
            embedding   = emb,
            reward      = d["reward"],
            norm_reward = d["norm_reward"],
            V           = d["V"],
            norm_V      = d["norm_V"],
            visit_count = d["visit_count"],
            total_reward= d["total_reward"],
            created_at  = d["created_at"],
            is_terminal = bool(d["is_terminal"]),
            children    = [int(x) for x in children_raw.split(",") if x],
            parents     = [int(x) for x in parents_raw.split(",") if x],
        )


    def __repr__(self) -> str:
        preview = self.text[:40].replace('\n', ' ')
        return (
            f"Node(id={self.node_id}, level={self.level.name}, "
            f"r={self.reward:+.3f}, V={self.V:+.3f}, "
            f"visits={self.visit_count}, text={repr(preview)})"
        )


# ─────────────────────────────────────────────
# EMBEDDING COMPUTATION
#
# No neural network. The embedding is computed
# directly from the raw signal using structural
# properties of the text at each level.
#
# This is the raw signal representation —
# the graph and reward function learn what matters.
# We are not preprocessing meaning into the embedding.
# We are encoding structure.
# ─────────────────────────────────────────────

def compute_embedding(text: str, level: Level) -> np.ndarray:
    """
    Compute a structural embedding for a text unit at a given level.
    Returns a float32 numpy array of shape (EMBEDDING_DIM[level],).

    The embedding encodes:
        - Character frequency distribution (what characters appear)
        - Positional structure (where things appear)
        - Length and density signals
        - Level-specific structural features

    No pretrained weights. No neural network.
    Pure signal structure — the graph learns the rest.
    """
    dim = EMBEDDING_DIM[level]
    emb = np.zeros(dim, dtype=np.float32)

    if not text:
        return emb

    chars  = list(text)
    n      = len(chars)
    ids    = [char_to_id(c) for c in chars]

    if level == Level.CHARACTER:
        # Single character — one-hot style over reduced vocab
        # dim=16: encode character class and position in vocab
        cid = char_to_id(text[0]) if text else 0
        emb[0] = cid / 97.0                          # normalized vocab position
        emb[1] = float(text.isalpha())               # is letter
        emb[2] = float(text.isdigit())               # is digit
        emb[3] = float(text.isspace())               # is whitespace
        emb[4] = float(text in '.,!?;:')             # is punctuation
        emb[5] = float(text.isupper())               # is uppercase
        emb[6] = float(text.islower())               # is lowercase
        # remaining dims: sinusoidal position encoding of vocab id
        for k in range(7, dim):
            emb[k] = np.sin(cid / (10000 ** ((k-7) / max(dim-7, 1))))

    elif level == Level.WORD:
        # dim=64
        # Encode character frequency, length, structural features
        freq = np.zeros(97, dtype=np.float32)
        for cid in ids:
            if cid < 97:
                freq[cid] += 1
        freq /= (n + 1e-9)

        # First 32 dims: character frequency (top 32 most informative slots)
        emb[:32] = freq[:32]

        # Next dims: structural word features
        emb[32] = min(n, 30) / 30.0                 # normalized length
        emb[33] = float(text[0].isupper())          # starts with capital
        emb[34] = float(text.isupper())             # all caps
        emb[35] = float(text.isalpha())             # pure alphabetic
        emb[36] = float(any(c.isdigit() for c in text))  # contains digit
        emb[37] = float(text[-1] in '.!?')          # ends sentence
        emb[38] = float(text[-1] in ',;:')          # ends clause
        emb[39] = float('-' in text)                # hyphenated
        emb[40] = float("'" in text)                # contraction or possessive

        # Remaining: sinusoidal encoding of character id sequence (first 23 chars)
        for k, cid in enumerate(ids[:23]):
            slot = 41 + k
            if slot < dim:
                emb[slot] = np.sin(cid / 97.0 * np.pi)

    elif level == Level.SENTENCE:
        # dim=128
        words = text.split()
        nw    = len(words)

        # Character frequency over full sentence (first 48 dims)
        freq = np.zeros(97, dtype=np.float32)
        for cid in ids:
            if cid < 97:
                freq[cid] += 1
        freq /= (n + 1e-9)
        emb[:48] = freq[:48]

        # Word count features
        emb[48] = min(nw, 60) / 60.0               # normalized word count
        emb[49] = min(n,  300) / 300.0             # normalized char count
        emb[50] = float(text.strip()[-1] == '.')   # declarative
        emb[51] = float(text.strip()[-1] == '?')   # interrogative
        emb[52] = float(text.strip()[-1] == '!')   # exclamatory
        emb[53] = float(text[0].isupper())         # proper start

        # Lexical diversity — unique words / total words
        unique_words = len(set(w.lower() for w in words))
        emb[54] = unique_words / (nw + 1e-9)

        # Average word length
        avg_wlen = np.mean([len(w) for w in words]) if words else 0
        emb[55] = min(avg_wlen, 15) / 15.0

        # Digit presence ratio
        digit_chars = sum(c.isdigit() for c in text)
        emb[56] = digit_chars / (n + 1e-9)

        # Positional character encoding (dims 57-127)
        step = max(1, n // (dim - 57))
        for k, pos in enumerate(range(0, n, step)):
            slot = 57 + k
            if slot >= dim:
                break
            cid = ids[pos] if pos < len(ids) else 0
            emb[slot] = np.sin(cid / 97.0 * np.pi) * np.cos(pos / (n + 1e-9) * np.pi)

    elif level == Level.PARAGRAPH:
        # dim=256
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        ns = len(sentences)
        words  = text.split()
        nw     = len(words)

        # Character frequency (first 64 dims)
        freq = np.zeros(97, dtype=np.float32)
        for cid in ids:
            if cid < 97:
                freq[cid] += 1
        freq /= (n + 1e-9)
        emb[:64] = freq[:64]

        # Structural features
        emb[64] = min(ns, 20) / 20.0              # sentence count
        emb[65] = min(nw, 200) / 200.0            # word count
        emb[66] = min(n, 1500) / 1500.0           # char count
        emb[67] = len(set(w.lower() for w in words)) / (nw + 1e-9)  # lexical diversity
        emb[68] = text.count('?') / (ns + 1e-9)  # question density
        emb[69] = text.count(',') / (nw + 1e-9)  # comma density (clause complexity)
        emb[70] = text.count('"') / (n + 1e-9)   # quote presence
        emb[71] = float(any(c.isdigit() for c in text))  # contains numbers

        # Sentence length variance — measures structural regularity
        if sentences:
            slens = [len(s.split()) for s in sentences]
            emb[72] = np.std(slens) / (np.mean(slens) + 1e-9)

        # Positional encoding over characters (dims 73-255)
        step = max(1, n // (dim - 73))
        for k, pos in enumerate(range(0, n, step)):
            slot = 73 + k
            if slot >= dim:
                break
            cid = ids[pos] if pos < len(ids) else 0
            emb[slot] = np.sin(cid / 97.0 * np.pi) * np.cos(pos / (n + 1e-9) * np.pi)

    elif level == Level.DOCUMENT:
        # dim=512
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        np_ = len(paragraphs)
        words = text.split()
        nw    = len(words)

        # Character frequency (first 97 dims — full vocab)
        freq = np.zeros(97, dtype=np.float32)
        for cid in ids:
            if cid < 97:
                freq[cid] += 1
        freq /= (n + 1e-9)
        emb[:97] = freq

        # High-level structural features
        emb[97]  = min(np_, 50)  / 50.0
        emb[98]  = min(nw,  5000) / 5000.0
        emb[99]  = min(n,   30000) / 30000.0
        emb[100] = len(set(w.lower() for w in words)) / (nw + 1e-9)
        emb[101] = text.count('\n') / (n + 1e-9)
        emb[102] = text.count('?') / (nw + 1e-9)
        emb[103] = text.count('!') / (nw + 1e-9)
        emb[104] = text.count('"') / (nw + 1e-9)

        # Positional encoding over sampled characters (dims 105-511)
        step = max(1, n // (dim - 105))
        for k, pos in enumerate(range(0, n, step)):
            slot = 105 + k
            if slot >= dim:
                break
            cid = ids[pos] if pos < len(ids) else 0
            emb[slot] = np.sin(cid / 97.0 * np.pi) * np.cos(pos / (n + 1e-9) * np.pi)

    # L2 normalize so all embeddings live on the unit hypersphere
    # This makes cosine similarity = dot product everywhere
    norm = np.linalg.norm(emb)
    if norm > 1e-9:
        emb /= norm

    return emb


# ─────────────────────────────────────────────
# NODE FACTORY
# ─────────────────────────────────────────────

def make_node(node_id: int, text: str, level: Level,
              reward: float = 0.0,
              children: Optional[list[int]] = None,
              parents:  Optional[list[int]] = None) -> Node:
    """
    Create a new node with computed embedding.
    This is the single entry point for node creation —
    all other modules call this rather than constructing
    Node directly, ensuring embedding is always computed.
    """
    emb = compute_embedding(text, level)
    return Node(
        node_id  = node_id,
        text     = text,
        level    = level,
        embedding= emb,
        reward   = reward,
        V        = reward,           # V initialized to own reward
        children = children or [],
        parents  = parents  or [],
    )


# ─────────────────────────────────────────────
# SIMILARITY
#
# Cosine similarity between two node embeddings.
# Used by activation.py to spread query signal.
# Embeddings are L2-normalized so this is just dot product.
# ─────────────────────────────────────────────

def similarity(a: Node, b: Node) -> float:
    """
    Cosine similarity between two nodes.
    Returns value in [-1, 1].
    Nodes at different levels have different embedding dims —
    in that case project smaller into larger via zero-padding.
    """
    ea, eb = a.embedding, b.embedding
    if ea.shape != eb.shape:
        max_dim = max(len(ea), len(eb))
        tmp_a = np.zeros(max_dim, dtype=np.float32)
        tmp_b = np.zeros(max_dim, dtype=np.float32)
        tmp_a[:len(ea)] = ea
        tmp_b[:len(eb)] = eb
        ea, eb = tmp_a, tmp_b
    dot = float(np.dot(ea, eb))
    return max(-1.0, min(1.0, dot))


if __name__ == "__main__":
    import time
    print("=== node.py smoke test ===\n")

    samples = [
        ("a",                                      Level.CHARACTER),
        ("Detective",                              Level.WORD),
        ("She found footprints leading to the back room.", Level.SENTENCE),
        ("The building smelled of rust and old machinery.\nShe found footprints.", Level.PARAGRAPH),
    ]

    nodes = []
    for i, (text, level) in enumerate(samples):
        t0 = time.time()
        n = make_node(i, text, level, reward=float(i) * 0.5)
        elapsed = (time.time() - t0) * 1000
        nodes.append(n)
        print(f"{n}")
        print(f"  embedding dim : {n.dim()}")
        print(f"  embedding norm: {np.linalg.norm(n.embedding):.4f}  (should be ~1.0)")
        print(f"  compute time  : {elapsed:.2f}ms")
        print()

    print("Similarity matrix:")
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            s = similarity(a, b)
            print(f"  sim({i},{j}) = {s:+.4f}", end="")
        print()

    print("\nSerialization round-trip:")
    n = nodes[2]
    d = n.to_dict()
    n2 = Node.from_dict(d)
    print(f"  Original : {n}")
    print(f"  Recovered: {n2}")
    print(f"  Embedding match: {np.allclose(n.embedding, n2.embedding)}")