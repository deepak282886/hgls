"""
server.py — Little Deepak's brain server.

Run: python server.py
Then open ui.html in your browser.

No hardcoded word lists. The graph IS the knowledge.
Known concepts are registered as level-2 concept nodes in the graph
when each lesson teaches them. The server reads those nodes at runtime.

Endpoints:
  POST /speak   { text }           → { response, score, state }
  GET  /state                      → { nodes, edges, known_words }
  POST /teach   { text, reward }   → { intensity }
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask      import Flask, request, jsonify
from flask_cors import CORS

from system import System
from atoms  import encode_text

app    = Flask(__name__)
CORS(app)

MEMORY = 'knowledge.json'
system = System(MEMORY)


# ── Response — purely from the graph ──────────────────────────────

def respond_from_graph(input_text: str) -> dict:
    """
    Recognition purely through the graph — bottom up.

    Input atoms activate at level 0.
    Activation propagates UP through edges to concept nodes at level 2.
    The concept node with the highest incoming activation from the
    input atoms IS the recognition result.

    No string matching. No hardcoded lists.
    The graph structure determines the answer.

    Confidence = top concept activation / total activation across all concepts.
    High = the system clearly knows this. Low = weak or no match.
    """
    text       = input_text.strip().lower()
    text_atoms = encode_text(text)

    if not text_atoms:
        return {'word': None, 'score': 0.0, 'confidence': 0.0}

    # Propagate up through the graph
    results = system.al.propagate_up(text_atoms)

    if not results:
        return {'word': None, 'score': 0.0, 'confidence': 0.0}

    best_node, best_score = results[0]
    best_word = best_node.elements[0] if best_node and best_node.elements else None

    # Confidence: top score as fraction of total activation
    total = sum(score for _, score in results)
    confidence = best_score / max(total, 1.0)

    return {
        'word':       best_word,
        'score':      best_score,
        'confidence': confidence,
    }


# ── Endpoints ──────────────────────────────────────────────────────

@app.route('/speak', methods=['POST'])
def speak():
    data = request.json or {}
    text = data.get('text', '').strip().lower()

    if not text:
        return jsonify({'error': 'no input'}), 400

    text_atoms      = encode_text(text)
    path, graph_score = system.query(text_atoms, 'text')
    response        = respond_from_graph(text)
    tinkerer_active = graph_score < 50.0

    st = system.state()
    return jsonify({
        'input':           text,
        'score':           graph_score,
        'response_word':   response['word'],
        'response_score':  response['score'],
        'confidence':      response['confidence'],
        'tinkerer_active': tinkerer_active,
        'nodes':           st['total_nodes'],
        'edges':           st['total_edges'],
        'level1':          st['by_level'].get(1, {}).get('nodes', 0),
    })


@app.route('/teach', methods=['POST'])
def teach():
    data   = request.json or {}
    text   = data.get('text', '').strip().lower()
    reward = data.get('reward', True)

    if not text:
        return jsonify({'error': 'no input'}), 400

    text_atoms = encode_text(text)
    paths, cross, avg = system.learn_multi(
        text_atoms = text_atoms,
        reward     = reward,
    )

    # Register as a concept in the graph
    system.register_concept(text)
    system.save()

    st = system.state()
    return jsonify({
        'taught':    text,
        'intensity': avg,
        'nodes':     st['total_nodes'],
        'edges':     st['total_edges'],
    })


@app.route('/state', methods=['GET'])
def state():
    st       = system.state()
    concepts = system.known_concepts()
    return jsonify({
        'nodes':       st['total_nodes'],
        'edges':       st['total_edges'],
        'level1':      st['by_level'].get(1, {}).get('nodes', 0),
        'level2':      st['by_level'].get(2, {}).get('nodes', 0),
        'known_words': concepts,
    })


if __name__ == '__main__':
    print('\n' + '='*50)
    print('Little Deepak — Brain Server')
    print('='*50)
    st       = system.state()
    concepts = system.known_concepts()
    print(f'  Nodes    : {st["total_nodes"]}')
    print(f'  Edges    : {st["total_edges"]}')
    print(f'  Concepts : {len(concepts)}')
    if concepts:
        print(f'  Knows    : {", ".join(concepts[:8])}{"…" if len(concepts)>8 else ""}')
    print('\nOpen ui.html in your browser.')
    print('='*50 + '\n')
    app.run(port=5000, debug=False)