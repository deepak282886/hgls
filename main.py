"""
main.py — Entry point.

Commands:

  python main.py --chat
      Query the system interactively with text.
      Shows path length and coherence score for each input.
      No reward given — pure query mode.

  python main.py --state
      Print current system state.
      Nodes and edges by level.

  python main.py --teach "hello" --reward
      Teach one text input with reward.

  python main.py --teach "hello" --no-reward
      Teach one text input with no reward.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KNOWLEDGE_FILE = 'knowledge.json'


def cmd_chat(args):
    from system import System
    s = System(args.memory)

    print('\n' + '=' * 50)
    print('Query mode  (type "quit" to exit)')
    print('Score shows how well the system knows this input.')
    print('=' * 50 + '\n')

    while True:
        try:
            text = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye.')
            break
        if not text:
            continue
        if text.lower() in ('quit', 'exit', 'bye'):
            break

        atoms        = s.text(text)
        path, score  = s.query(atoms, 'text')
        print(f'  score={score:.4f}  path_length={len(path)}\n')


def cmd_state(args):
    import json
    from system import System
    s    = System(args.memory)
    st   = s.state()
    print('\n' + '=' * 50)
    print('System State')
    print('=' * 50)
    print(f'  Total nodes : {st["total_nodes"]}')
    print(f'  Total edges : {st["total_edges"]}')
    print('\n  By level:')
    for lvl, d in st['by_level'].items():
        print(f'    Level {lvl}: {d["nodes"]:>6} nodes  '
              f'modalities={d["modalities"]}')


def cmd_teach(args):
    from system import System
    s     = System(args.memory)
    atoms = s.text(args.teach)
    reward = not args.no_reward

    path, score, intensity = s.learn(atoms, 'text', reward=reward)
    print(f'  reward={reward}  score={score:.4f}  '
          f'intensity={intensity:.4f}  path_length={len(path)}')
    s.save()


def main():
    parser = argparse.ArgumentParser(
        description='HGLS — Hierarchical Generative Learning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--memory', default=KNOWLEDGE_FILE,
        help=f'Knowledge file (default: {KNOWLEDGE_FILE})'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--chat',  action='store_true', help='Query mode')
    group.add_argument('--state', action='store_true', help='Print state')
    group.add_argument('--teach', metavar='TEXT',      help='Teach one input')

    parser.add_argument('--no-reward', action='store_true',
                        help='Teach with no reward (default: reward=True)')

    args = parser.parse_args()

    if args.chat:
        cmd_chat(args)
    elif args.state:
        cmd_state(args)
    elif args.teach:
        cmd_teach(args)


if __name__ == '__main__':
    main()