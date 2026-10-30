#!/usr/bin/env python3
"""Reject tracked private/local runtime files; never inspect their contents."""
import argparse
from pathlib import PurePosixPath
import subprocess
import sys


def forbidden(path):
    value=PurePosixPath(path);name=value.name.casefold()
    if any(part in {'.codex','.agents','.claude','.aws','.ssh','.vscode','.idea','.papermerge'}
           or part.startswith('.venv') for part in value.parts):return True
    if path.startswith('environment/') and len(value.parts)>2:return True
    if path=='gui/queue_state.json' or path.startswith('gui/worker_state/'):return True
    if path.startswith(('newResults/','newPlots/','experiments/cache/')):return True
    if path.startswith('docs/thesis-figures/') and (value.suffix.lower()=='.pdf'
            or name=='installation.json'):return True
    if (path.startswith('docs/results/') and value.suffix.lower()=='.pdf'
            and 'source_snapshots' not in value.parts
            and any(part in ('figures','thesis_figures','publication_curves') for part in value.parts)):return True
    if name in {'.env','browser_share.json','connection.json','supervisor.pid'}:return True
    document=value.suffix.lower() in {'.md','.txt','.rst','.pdf','.docx','.html'}
    context=('handoff' in name or
        any(agent in name for agent in ('chatgpt','claude')) and
        any(kind in name for kind in ('prompt','catch_up','catchup','context')))
    return document and context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', help='Check a Git revision instead of the staged index.')
    args = parser.parse_args(argv)
    command = ['git','ls-tree','-r','--name-only','-z',args.revision] if args.revision else ['git','ls-files','-z']
    names = subprocess.check_output(command).decode().split('\0')
    blocked = [name for name in names if name and forbidden(name)]
    if blocked:
        print('Publication blocked: local runtime/private files must not be tracked:\n'+'\n'.join(blocked), file=sys.stderr)
        return 1
    print('Publication policy passed: no tracked local/private runtime files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
