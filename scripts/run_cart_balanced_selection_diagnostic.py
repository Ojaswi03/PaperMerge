#!/usr/bin/env python3
"""Clean three-round selection comparison, fixed mu=.01; default never trains.

The owner-balanced selector is a CART adaptation, NOT paper BASIL selection.
No threat, channel, EBM, W&B, test-feedback or production queue is enabled.
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
RESULT = ROOT/'newResults/nonIID/cart_owner_gradients/clean_balanced_selection_fixed_anchor_3r_v1'
PLOTS = ROOT/'newPlots/nonIID/cart_owner_gradients/clean_balanced_selection_fixed_anchor_3r_v1'
CONTROLS = ('receiver_local_ss', 'balanced_owner_ss')


def fingerprints():
    from scripts.run_cart_owner_selection_diagnostic import fingerprints as previous
    paths = ('scripts/run_cart_balanced_selection_diagnostic.py',
             'scripts/report_cart_balanced_selection_diagnostic.py',
             'basil_core/cart_owner_protocol.py')
    return dict(previous(), **{p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})


def config(device, control):
    from scripts.run_cart_owner_selection_diagnostic import config as previous
    if control not in CONTROLS:
        raise ValueError('Unknown bounded selection condition.')
    cfg = previous(device, 'basil_ss')
    cfg.update(condition=control, purpose='bounded_clean_fixed_anchor_selection_comparison',
        startingModelRule='lowest_receiver_local_loss' if control == CONTROLS[0] else 'balanced_owner_training_ce',
        snapshotSelectionRule='lowest_receiver_local_loss' if control == CONTROLS[0] else 'balanced_owner_training_ce',
        snapshotAnchorRule='fixed_frozen_selected_snapshot', snapshotAnchorMu=.01,
        cartRevision='owner_gradients_fixed_anchor_balanced_selection_v1',
        selectionProbeUsage='512 unaugmented TRAIN examples per available owner; identical batches for all candidates',
        helperThreatScope='honest_auxiliary_snapshot_only',
        balancedSelectionIsPaperBasil=False, helperChannelNoise=False,
        futureOwnersAvailable=False)
    return cfg


def verify_configs(device):
    a, b = (config(device, c) for c in CONTROLS)
    differences = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    if differences != ['condition', 'snapshotSelectionRule', 'startingModelRule']:
        raise ValueError('Unpaired clean selection configuration.')
    return differences


def create_worker(model, helper_model, *, device='CPU', diagnostic_rounds=3):
    from basil_core.cart_owner_gradient_worker import OwnerGradientCartWorker

    class FixedAnchorOwnerWorker(OwnerGradientCartWorker):
        def _anchor_control(self, selected_loss, target, current, support):
            # Prespecified mu, not selected loss/global accuracy, controls the
            # anchor. Registry verification/transport remains unchanged.
            return dict(mu=.01, rule='fixed_frozen_selected_snapshot',
                        supportedClasses=[i for i, count in enumerate(support) if count > 0])

        def train(self, *args, **kwargs):
            params, record = super().train(*args, **kwargs)
            record.update(cartRevision='owner_gradients_fixed_anchor_balanced_selection_v1',
                          snapshotAnchorRule='fixed_frozen_selected_snapshot', snapshotAnchorMu=.01)
            return params, record

    return FixedAnchorOwnerWorker(model, helper_model, device=device, diagnostic_rounds=diagnostic_rounds)


def choose_start(receiver, candidates, *, worker, images, labels, selections,
                 available_owners, control):
    from basil_core.cart_owner_protocol import training_probes, select_balanced_snapshot, TRUSTED_AUXILIARY
    from basil_core.research_protocol import select_snapshot
    if control not in CONTROLS:
        raise ValueError('Unknown selection condition.')
    # Both conditions evaluate identical candidate/probe pairs for diagnostic
    # evidence. Only the configured TRAIN-side score determines selection.
    probes = training_probes(labels, {i: selections[i] for i in available_owners})
    balanced, records, _ = select_balanced_snapshot(receiver, candidates, probes,
        lambda params, owner, ids: worker.probe(params, images[list(ids)], labels[list(ids)])[0],
        helper_scope=TRUSTED_AUXILIARY, available_owners=available_owners)
    lookup = {id(s.params): r for s, r in zip(candidates, records)}
    local, _ = select_snapshot(receiver, candidates,
        lambda params: (lookup[id(params)]['receiverLocalLoss'], 0.))
    # Accuracy is not a selection criterion. The zero above is an unused API
    # placeholder, not persisted as an observed accuracy.
    selected = local if control == CONTROLS[0] else balanced
    loss = lookup[id(selected.params)]['receiverLocalLoss']
    decision = dict(selectionApplied=True, availableOwners=list(available_owners),
        receiverLocalMinimumSender=local.sender_id, balancedMinimumSender=balanced.sender_id,
        selectedBy='receiverLocalLoss' if control == CONTROLS[0] else 'balancedTrainingLoss',
        selectionProbeIndices={str(p.owner): list(p.indices) for p in probes}, sourceSplit='train')
    return selected, records, loss, decision


def proof_path(device):
    return ROOT/f'gui/worker_state/CART_BALANCED_SELECTION_FIXED_ANCHOR_3R_{device}_VERIFICATION.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify-device', action='store_true')
    action.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.verify_device and not args.execute:
        print(json.dumps(dict(configs=[config(args.device, c) for c in CONTROLS],
                              differences=verify_configs(args.device), trainingRunsStarted=0), indent=2))
        return
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled')
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    from scripts.run_cart_owner_selection_diagnostic import verify, execute
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf, args.device)
    study = sys.modules[__name__]
    if args.verify_device:
        verify(tf, args.device, study=study)
    else:
        execute(tf, args.device, study=study)


if __name__ == '__main__':
    main()
