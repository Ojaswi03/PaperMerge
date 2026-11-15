#!/usr/bin/env python3
"""Choose the server-specific noise-only launcher using the actual hostname."""
import argparse
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {
    'hal9k': 'run_fedsharp_noise_only.py',
    'themachine': 'run_fedsharp_noise_only_themachine.py',
}


def command_for_host(hostname, argv):
    host = hostname.split('.')[0]
    if host not in LAUNCHERS:
        raise ValueError('Noise-only execution supports hal9k and themachine; detected ' + hostname)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--role', default='driver')
    control, _ = pre.parse_known_args(argv)
    if host == 'hal9k' and control.role == 'export':
        parser = argparse.ArgumentParser(description='Verify and export the hal9k noise-only campaign.')
        parser.add_argument('--role', choices=('export',), required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--destination', type=Path, required=True)
        args = parser.parse_args(argv)
        return [sys.executable, str(ROOT / 'scripts/export_fedsharp_noise_only.py'),
                '--output', str(args.output), '--destination', str(args.destination)]
    return [sys.executable, str(ROOT / 'scripts' / LAUNCHERS[host]), *argv]


def main(argv=None):
    command = command_for_host(socket.gethostname(), list(sys.argv[1:] if argv is None else argv))
    # Replace this process. The existing launcher keeps its actual argv, host,
    # supervisor ownership checks, background behavior and frozen producer pins.
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
