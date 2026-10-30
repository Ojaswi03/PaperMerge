#!/usr/bin/env python3
"""Reuse a written CIFAR rebuild against a verified archive; require a new plan."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basil_core.cache_recovery import install_rebuild


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Failed experiment folder; preserved unchanged.')
    parser.add_argument('--rebuild', type=Path, required=True, help='Existing cache-recovery folder.')
    parser.add_argument('--reference-plan', type=Path, required=True, help='Archived plan beside its SHA256.json.')
    args = parser.parse_args()
    result = install_rebuild(args.output, args.rebuild, args.reference_plan,
                             progress=lambda message: print(message, flush=True))
    print(json.dumps(result, indent=2), flush=True)
