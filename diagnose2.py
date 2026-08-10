"""
diagnose2.py — Check if corrections are being stored and retrieved.
python diagnose2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hgls.system import HGLSystem

system = HGLSystem(use_llm=False)
system.load('deepak_memory.json')

print(f'Library size: {len(system.library)}')
print()

# Check correction-tagged structures
corrections = []
for level in range(7):
    for s in system.library.get_at_level(level, 'success'):
        if s.correction_count > 0:
            text = s.generate(system.library)
            corrections.append((level, s.correction_count, s.topic_tags[:5], text[:80]))

print(f'Total correction-tagged structures: {len(corrections)}')
print()
print('Sample corrections:')
for level, count, tags, text in corrections[:20]:
    print(f'  L{level} corr={count} tags={tags}')
    print(f'    "{text}"')
    print()

# Simulate a query
print('=' * 50)
print('Simulating query: "What shape has three sides?"')
print()

topics = {'shape', 'three', 'sides', 'triangle'}
print(f'Topic words: {topics}')
print()

# Score all structures
scored = []
for level in range(7):
    for s in system.library.get_at_level(level, 'success'):
        text = s.generate(system.library)
        if not text:
            continue
        words = text.lower().split()
        if len(words) < 2:
            continue
        text_words = set(words)
        overlap = len(topics & text_words)
        if overlap == 0:
            continue
        precision = overlap / len(words)
        eff_fit   = s.effective_fitness(topics)
        score     = precision * eff_fit
        if s.correction_count > 0:
            tag_overlap = len(topics & set(s.topic_tags))
            if tag_overlap > 0:
                score *= (1.0 + 0.5 * s.correction_count)
        scored.append((score, overlap, level, s.correction_count, text[:80]))

scored.sort(reverse=True)
print(f'Top 10 matches:')
for score, overlap, level, corr, text in scored[:10]:
    print(f'  score={score:.3f} overlap={overlap} L{level} corr={corr}')
    print(f'    "{text}"')
    print()

if not scored:
    print('  NO MATCHES FOUND')
    print('  The corrections are not being stored or retrieved.')