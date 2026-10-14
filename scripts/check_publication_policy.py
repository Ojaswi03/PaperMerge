#!/usr/bin/env python3
"""Reject tracked handoff documents; never inspect document contents."""
import argparse
from pathlib import PurePosixPath
import subprocess
import sys


def forbidden(path):
    name=PurePosixPath(path).name.casefold()
    document=PurePosixPath(path).suffix.lower() in {'.md','.txt','.rst','.pdf','.docx','.html'}
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
        print('Publication blocked: handoff documents must not be tracked:\n'+'\n'.join(blocked), file=sys.stderr)
        return 1
    print('Publication policy passed: no tracked handoff documents.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
