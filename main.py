"""
main.py — HGLS v0.7 Entry Point.

Commands:

  python main.py --foundation
      Phase 1: Teach 8000 common English words (syllables → words)
      Phase 2: Teach 30000 everyday sentences from Brown corpus
      Run this FIRST before --train.
      Fully resumable — run again after interruption.

  python main.py --foundation --words
      Words phase only.

  python main.py --foundation --sents
      Sentences phase only.

  python main.py --train
      Run syllabus training (Class 1 through 12, all subjects).
      Requires --foundation to have run first.
      Fully resumable — run again after interruption.

  python main.py --train --reset
      Reset syllabus progress (keeps foundation/memory).
      Starts curriculum from Class 1 again.

  python main.py --chat
      Talk to the system interactively.

  python main.py --stats
      Print full system stats and mastery progress.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEMORY_FILE = 'deepak_memory.json'


# ── Commands ──────────────────────────────────────────────────────

def cmd_foundation(args):
    from hgls.system   import HGLSystem
    from foundation    import FoundationBuilder, ensure_nltk

    if not ensure_nltk():
        print('[Main] Please install nltk: pip install nltk')
        sys.exit(1)

    system = HGLSystem(use_llm=False)
    if os.path.exists(args.memory):
        system.load(args.memory)

    builder = FoundationBuilder(system, progress_file='deepak_progress.json')

    phase = 'all'
    if args.words and not args.sents:
        phase = 'words'
    elif args.sents and not args.words:
        phase = 'sentences'

    builder.run(phase=phase)

    try:
        system.save(args.memory)
    except Exception as e:
        print(f'[Main] Final save failed: {e}')


def cmd_train(args):
    api_key = args.api_key or os.environ.get('TOGETHER_API_KEY', '')
    if not api_key:
        print('[Error] Training requires Together AI API key.')
        print('  Set environment variable: TOGETHER_API_KEY=your_key')
        print('  Or pass: --api-key your_key')
        sys.exit(1)

    from hgls.system      import HGLSystem
    from syllabus_teacher import SyllabusTeacher, test_api_key

    if not test_api_key(api_key):
        sys.exit(1)

    system = HGLSystem(use_llm=False)

    if args.reset:
        # Reset only progress, keep memory/foundation
        for f in ['deepak_progress.json', 'deepak_progress.json.bak']:
            if os.path.exists(f):
                os.remove(f)
        print('[Main] Syllabus progress reset. Foundation preserved.')

    if os.path.exists(args.memory):
        system.load(args.memory)
    else:
        print(f'[Main] No memory found at {args.memory}.')
        print('[Main] Consider running --foundation first to build vocabulary.')

    teacher = SyllabusTeacher(system, api_key=api_key)
    teacher.run()

    try:
        system.save(args.memory)
    except Exception as e:
        print(f'[Main] Final save failed: {e}')


def cmd_chat(args):
    from hgls.system import HGLSystem

    system = HGLSystem(use_llm=False)
    if os.path.exists(args.memory):
        system.load(args.memory)
    else:
        print('[Main] No memory found — starting fresh.')

    print('\n' + '=' * 50)
    print('Chat  (type "quit" to exit)')
    print('=' * 50 + '\n')

    while True:
        try:
            user_input = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye.')
            break
        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'bye'):
            print('Bye.')
            break
        response = system.respond(user_input)
        print(f'System: {response}\n')


def cmd_stats(args):
    import json
    from hgls.system import HGLSystem

    system = HGLSystem(use_llm=False)
    if not system.load(args.memory):
        print('[Main] No memory found.')
        return

    system.print_stats()

    # Foundation progress
    if os.path.exists('deepak_progress.json'):
        try:
            with open('deepak_progress.json') as f:
                data = json.load(f)
            found = data.get('foundation', {})
            if found:
                print('\n' + '=' * 60)
                print('FOUNDATION PROGRESS')
                print('=' * 60)
                print(f'  Words taught    : {found.get("words_done", 0):,} / 8,000')
                print(f'  Sentences taught: {found.get("sents_done", 0):,} / 30,000')
        except Exception:
            pass

    # Mastery progress
    try:
        from mastery_tracker import MasteryTracker
        tracker = MasteryTracker()
        stats   = tracker.stats()

        print('\n' + '=' * 60)
        print('MASTERY PROGRESS')
        print('=' * 60)
        print(f'  Pass            : {stats["pass_number"]}')
        print(f'  Total topics    : {stats["total_topics"]}')
        print(f'  Mastered        : {stats["mastered"]} ({stats["mastery_rate"]:.0%})')
        print(f'  In progress     : {stats["in_progress"]}')
        print(f'  Not started     : {stats["not_started"]}')
        print(f'  Current topic   : {stats["current_topic"]}')

        print('\n  By Subject:')
        for subject, data in sorted(tracker.summary_by_subject().items()):
            pct = data['mastered'] / max(data['total'], 1)
            bar = '█' * int(pct * 20)
            print(f'    {subject:<20} {data["mastered"]:3}/{data["total"]:3}  {bar}')
    except Exception as e:
        print(f'  [Stats] Mastery tracker error: {e}')


# ── Entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HGLS v0.7 — Hierarchical Generative Learning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--memory', default=MEMORY_FILE,
        help=f'Memory file path (default: {MEMORY_FILE})'
    )
    parser.add_argument(
        '--api-key', default=None,
        help='Together AI API key (or set TOGETHER_API_KEY env var)'
    )

    # Primary commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--foundation', action='store_true',
                       help='Build language foundation (words + sentences)')
    group.add_argument('--train',      action='store_true',
                       help='Run syllabus training')
    group.add_argument('--chat',       action='store_true',
                       help='Chat with the system')
    group.add_argument('--stats',      action='store_true',
                       help='Print system and mastery stats')

    # Foundation options
    parser.add_argument('--words', action='store_true',
                        help='Foundation: words phase only')
    parser.add_argument('--sents', action='store_true',
                        help='Foundation: sentences phase only')

    # Training options
    parser.add_argument('--reset', action='store_true',
                        help='Reset syllabus progress (keeps memory/foundation)')

    args = parser.parse_args()

    if args.foundation:
        cmd_foundation(args)
    elif args.train:
        cmd_train(args)
    elif args.chat:
        cmd_chat(args)
    elif args.stats:
        cmd_stats(args)


if __name__ == '__main__':
    main()