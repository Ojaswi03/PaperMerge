#!/usr/bin/env python3
"""Rebuild CIFAR-10 fitting images and require the saved plan's exact checksum."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basil_core.cache_recovery import recover


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Install only after exact identity checks and backup.')
    args = parser.parse_args()
    print(json.dumps(recover(args.output, apply=args.apply), indent=2), flush=True)
