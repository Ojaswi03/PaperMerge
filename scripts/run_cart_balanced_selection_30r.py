#!/usr/bin/env python3
"""Explicitly authorized clean 30-round pair. Default is a no-training plan.

The horizon is fixed, never an accuracy-triggered extension. W&B and the
production queue are disabled. Neither attacks, noise nor EBM are executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ROUNDS = 30
STREAM_ACTIVATIONS = True
CONTROLS = ('receiver_local_ss', 'balanced_owner_ss')
RESULT = ROOT/'newResults/nonIID/cart_owner_gradients/clean_balanced_selection_fixed_anchor_30r_v1'
PLOTS = ROOT/'newPlots/nonIID/cart_owner_gradients/clean_balanced_selection_fixed_anchor_30r_v1'
BASELINE = ROOT/'gui/worker_state/cart_balanced_selection_30r_before.json'


def fingerprints():
    from scripts.run_cart_balanced_selection_diagnostic import fingerprints as previous
    return dict(previous(), **{p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'scripts/run_cart_balanced_selection_30r.py',
        'scripts/verify_saved_evidence.py',)})


def config(device, control):
    from scripts.run_cart_balanced_selection_diagnostic import config as previous
    cfg = previous(device, control)
    cfg.update(nRounds=ROUNDS, purpose='bounded_clean_30_round_fixed_anchor_selection_comparison',
               automaticExtension=False, stopOnAccuracy=False, activationStorage='individual_atomic_json')
    return cfg


def verify_configs(device):
    a, b = (config(device, c) for c in CONTROLS)
    differences = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    if differences != ['condition', 'snapshotSelectionRule', 'startingModelRule']:
        raise ValueError('Unpaired control configurations.')
    return differences


def create_worker(*args, **kwargs):
    from scripts.run_cart_balanced_selection_diagnostic import create_worker as factory
    return factory(*args, **kwargs)


def choose_start(*args, **kwargs):
    from scripts.run_cart_balanced_selection_diagnostic import choose_start as selector
    return selector(*args, **kwargs)


def proof_path(device):
    return ROOT/f'gui/worker_state/CART_BALANCED_SELECTION_FIXED_ANCHOR_30R_{device}_VERIFICATION.json'


def verify_evidence():
    """Check all pre-existing research evidence; allow only this run's additions."""
    from scripts.verify_saved_evidence import inventory
    from scripts.run_cart_feature_diagnostic import write_json
    import scripts.verify_saved_evidence as evidence
    original = json.loads(BASELINE.read_text())
    evidence.TREES = tuple(original['trees'])
    current = inventory()
    old = original['files']
    changed = [p for p in old if current.get(p) != old[p]]
    allowed = tuple(str(p.relative_to(ROOT))+'/' for p in (RESULT, PLOTS))
    unexpected = [p for p in current.keys()-old.keys() if not p.startswith(allowed)]
    result = dict(before=len(old), after=len(current), matched=len(old)-len(changed),
                  changed=changed, unexpectedAdditions=unexpected,
                  status='unchanged' if not changed and not unexpected else 'failed')
    write_json(ROOT/'gui/worker_state/cart_balanced_selection_30r_integrity.json', result)
    if changed or unexpected:
        raise ValueError('Retained evidence integrity differs; inspect integrity report.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify-device', action='store_true')
    action.add_argument('--execute', action='store_true')
    action.add_argument('--report', action='store_true')
    args = parser.parse_args()
    study = sys.modules[__name__]
    if args.report:
        from scripts.report_cart_balanced_selection_diagnostic import report
        report(study=study)
        print(json.dumps(verify_evidence()))
        return
    if not args.execute and not args.verify_device:
        print(json.dumps(dict(configs=[config(args.device, c) for c in CONTROLS],
            differences=verify_configs(args.device), trainingRunsStarted=0,
            result=str(RESULT), plots=str(PLOTS)), indent=2))
        return
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled')
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    from scripts.run_cart_owner_selection_diagnostic import verify, execute
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf, args.device)
    if args.verify_device:
        verify(tf, args.device, study=study)
        return
    print(json.dumps(verify_evidence()), flush=True)
    try:
        execute(tf, args.device, study=study)
        from scripts.report_cart_balanced_selection_diagnostic import report
        report(study=study)
    finally:
        # Also check evidence after a genuine numerical/execution failure.
        print(json.dumps(verify_evidence()), flush=True)


if __name__ == '__main__':
    main()
