#!/usr/bin/env python3
"""Read-only failure diagnosis and zero-update owner-gradient proof.

Default is plan only. No CIFAR optimization, experiment queue or W&B actions.
New output paths refuse overwrite; completed research evidence is read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT/'newResults/nonIID/cart_representation/clean_30_rounds_fixed_gamma_v1'
RESULT = ROOT/'newResults/nonIID/cart_retention_failure/saved_30r_v1'
PROOF = ROOT/'gui/worker_state/CART_OWNER_GRADIENT_CPU_VERIFICATION.json'


def audit():
    import numpy as np
    from basil_core.cart_feature_memory import FeaturePacket
    from basil_core.cart_owner_gradients import feature_supervision_prior
    from basil_core.forgetting_audit import classification_metrics
    from basil_core.research_protocol import params_hash
    from scripts import report_cart_representation_diagnostic as review
    from scripts.run_cart_feature_diagnostic import write_json
    if RESULT.exists(): raise ValueError('Refusing to overwrite retention-audit evidence.')
    review.RESULT = SOURCE
    manifest, _, _, records, summary, checks = review.review_learning()
    findings = {}; hashes = {}
    for name in ('run.json', 'summary.json', 'config.json', 'partition_audit.json'):
        hashes[name] = hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
    for control in manifest['config']['controls']:
        last = records[control][-1]
        assert (last['round'], last['node']) == (29, 9)
        for name in ('activations.json', 'final_nodes.npz', 'round_metrics.csv'):
            hashes[f'{control}/{name}'] = hashlib.sha256((SOURCE/control/name).read_bytes()).hexdigest()
        packets = [FeaturePacket.from_dict(p) for p in last['featureMemory']['packets']]
        if any(p.receiver != 9 or p.round_id != 29 or p.model_hash != last['inputHash'] for p in packets):
            raise ValueError('Frozen replay packet provenance mismatch.')
        with np.load(SOURCE/control/'final_nodes.npz', allow_pickle=False) as saved:
            params = [saved[f'node9_w{j}'].copy() for j in range(10)]
            sender = last['selectedSender']
            if last['selectedSnapshotRound'] != 29 or not 4 <= sender <= 8:
                raise ValueError('Final selected snapshot cannot be reconstructed from current-round predecessors.')
            selected = [saved[f'node{sender}_w{j}'].copy() for j in range(10)]
        if params_hash(params) != last['outputHash'] or params_hash(selected) != last['inputHash']:
            raise ValueError('Saved trained/selected model checksum mismatch.')
        features = np.concatenate([np.asarray(p.features) for p in packets])
        labels = np.concatenate([np.full(len(p.features), p.class_id, np.int32) for p in packets])
        frozen = classification_metrics(features@params[-2]+params[-1], labels)
        stages = last['epochs'][-1]['sameImageAudit']['interventions']
        logits_current = stages['current_encoder_current_head']
        steps = last['optimizerTelemetry']
        prior = feature_supervision_prior(9, [p.class_id for p in packets], beta=1.)
        findings[control] = dict(**summary[control],
            finalNode=9, selectedSender=sender, selectedSnapshotRound=29,
            frozenReplayAccuracy=frozen['accuracy'], frozenReplayCE=frozen['crossEntropy'],
            frozenReplayOwnProbability=frozen['meanProbabilities'][9],
            frozenReplayPredictedCounts=frozen['predictedCounts'],
            trainingProbeOwnProbability=logits_current['meanProbabilities'][9],
            selectedEncoderCurrentHeadAccuracy=stages['selected_encoder_current_head']['accuracy'],
            selectedEncoderSelectedHeadAccuracy=stages['selected_encoder_selected_head']['accuracy'],
            encoderRepresentation=last['epochs'][-1]['sameImageAudit']['representation'],
            meanFinalActivationComponents={k: float(np.mean([s[k] for s in steps])) for k in (
                'baseCrossEntropy', 'featureReplayLoss', 'featureReplayCorrectionRatio', 'snapshotAnchorPenalty')},
            constantLogitStationaryPrior=prior)
    out = dict(status='completed', sourceRun=str(SOURCE.relative_to(ROOT)), sourceHashes=hashes,
        savedChecksPassed=checks, optimizerSteps=0, testSetLoaded=False,
        trainingProbeOnlyForNewAnalysis=True, existingTestMetricsReadOnly=True,
        findings=findings, candidateEnabled=False, newResearchRunsStarted=0)
    RESULT.mkdir(parents=True)
    write_json(RESULT/'audit.json', out)
    lines = ['# Completed CART retention failure: diagnosis and next step', '',
        '**Both controls completed 30/30 rounds. This audit performs zero optimizer updates.**', '',
        '| Metric | Feature replay | + Encoder preservation |', '|---|---:|---:|']
    for label, key, scale in (('Final full-test accuracy', 'finalFullTestAccuracy', 100),
        ('Final mean-node full-test accuracy', 'finalMeanNodeTestAccuracy', 100),
        ('Own-class TRAINING-probe accuracy', 'meanOwnClassAccuracy', 100),
        ('Other-class TRAINING-probe accuracy', 'meanOtherClassAccuracy', 100),
        ('Mean old-class squared feature drift', 'meanOldSquaredFeatureDrift', 1),
        ('Final Node-9 frozen old-feature accuracy', 'frozenReplayAccuracy', 100),
        ('Final Node-9 frozen old-feature CE', 'frozenReplayCE', 1)):
        a, b = (findings[c][key]*scale for c in manifest['config']['controls'])
        lines.append(f'| {label} | {a:.6f} | {b:.6f} |')
    lines += ['', 'Accuracy rows are percentages; drift/CE rows are not. Frozen features are owner TRAINING '
        'examples, not the CIFAR test set or identical images to the analysis probe. '
        'The earlier same-image intervention remains separately labelled.', '',
        '## Confirmed mechanism limitations, not a hidden numerical fix', '',
        '1. **Relative class supervision is imbalanced.** Current CE has coefficient 1; all nine '
        'older classes combined have coefficient beta=1, so each old class has coefficient 1/9. '
        'The existing implementation correctly follows the approved group-weighted objective; '
        'it is not equal supervision across all ten classes.', '',
        '2. **Detached feature replay has no encoder gradient.** Selected-encoder preservation '
        'uses current-class images only. It reduces drift without teaching the encoder '
        'to distinguish absent classes. A registry is metadata, not missing-class supervision.', '',
        '3. **The incoming reference is itself weak.** Final Node-9 selected-encoder/selected-head '
        'and selected-encoder/current-head probe accuracies are 10% in both controls. '
        'Preserving such a reference is not cumulative class learning.', '',
        'For a constant-logit classifier, `CE_current + mean(CE_old)` has the stationary prior '
        '`p_current=1/2`, `p_each_old=1/18`. Uniform balanced supervision instead has `p_c=1/10`. '
        '**This analytic example is not proof that weighting alone causes the CNN failure.** '
        'Feature coordinates differ between current images and detached replay; their representation '
        'quality and head changes also matter.', '',
        '## Candidate requiring explicit communication approval', '',
        'Use a separate, clean **class-balanced owner-gradient CART** diagnostic: older owners '
        'compute CE gradients on their own training images at the active trainer\'s exact current '
        'weights and return gradients, not images. The active trainer alone applies SGD.', '',
        '$$F_{bal}(\\theta)=\\frac{1}{|C|}\\sum_{c\\in C}CE_c(\\theta),\\qquad '
        '\\nabla F_{bal}=\\frac{1}{|C|}\\sum_{c\\in C}\\nabla CE_c.$$', '',
        'Here `C` contains the current class and genuinely available earlier owners, never invented '
        'future classes. Gradients reach both encoder and classifier. This is **gradient aggregation**, '
        'not model averaging: strict selected-weight handoff can remain unchanged. It nonetheless '
        '**changes the information/communication protocol** and is not plain one-class BASIL.', '',
        'The candidate mathematical interface and synthetic tests are implemented separately in '
        '`basil_core/cart_owner_gradients.py`. No engine/queue wiring, approved owner minibatch size, '
        'new CIFAR learning run, or successful learning claim follows from those tests.', '',
        'A full 117,706-parameter float32 gradient is 470,824 coordinate bytes. Nine helpers '
        'return 4,237,416 bytes per step, about 202.06 MiB per 50-step activation, before '
        'outbound parameter synchronization and serialization. This is substantially more '
        'communication than 16 detached features/class. Provenance checks do not prevent '
        'malicious gradients or provide privacy. Start clean, not with attacks/noise.', '',
        'Do not retry the failed loss with larger coefficients or launch another 30/500-round '
        'campaign merely to seek a better test score. A bounded approved mechanism check is '
        'needed before research execution.', '',
        f'Saved-result contracts verified: {checks}. New optimizer updates: 0. Existing result hashes '
        'are retained in `audit.json` for reproducible read-only attribution.', '']
    (RESULT/'CART_RETENTION_FAILURE_REVIEW.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='audited', checksPassed=checks, optimizerSteps=0,
        findings=findings, report=str(RESULT/'CART_RETENTION_FAILURE_REVIEW.md'))))


def verify_owner_objective():
    """Actual approved CNN, training data only, zero SGD updates; CPU scope."""
    import numpy as np
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    prepare_environment('CPU')
    import tensorflow as tf
    configure_tensorflow(tf, 'CPU')
    from basil_core.cart_owner_gradients import compute_owner_response, aggregate_balanced_owner_gradients
    from basil_core.research_protocol import build_model, keyed_rng, params_hash
    from scripts.run_cart_feature_diagnostic import write_json
    if PROOF.exists(): raise ValueError('Refusing to overwrite candidate objective proof.')
    source = json.loads((SOURCE/'run.json').read_text())
    if source['status'] != 'completed': raise ValueError('Completed source comparison required.')
    cache = ROOT/'experiments/cache/cifar10'
    x = np.load(cache/'x_train_float32.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(cache/'y_train_int32.npy', mmap_mode='r', allow_pickle=False)
    if x.shape != (50000, 32, 32, 3) or not np.array_equal(np.bincount(y, minlength=10), [5000]*10):
        raise ValueError('Full one-class TRAINING cache required.')
    excluded = set(json.loads((SOURCE/'partition_audit.json').read_text())['analysisProbe'])
    batches = []
    for owner in range(10):
        assigned = np.flatnonzero(y == owner)
        excluded.update(map(int, keyed_rng(2025, 'selection_batch', owner).permutation(assigned)[:512]))
        pool = np.asarray([i for i in assigned if int(i) not in excluded])
        ids = keyed_rng(2025, 'cart_owner_gradient_zero_update', owner).permutation(pool)[:16]
        batches.append((np.asarray(x[ids]), np.asarray(y[ids]), ids))
    model = build_model('basil_paper_cnn', 2025)
    canonical = [w.numpy().copy() for w in model.trainable_variables]
    if params_hash(canonical) != source['initialModelHash']: raise ValueError('Canonical model identity mismatch.')
    with np.load(SOURCE/'feature_replay_preserve/final_nodes.npz', allow_pickle=False) as saved:
        collapsed = [saved[f'node9_w{j}'].copy() for j in range(10)]
    outcomes = {}
    for state, params in (('canonical', canonical), ('completed_preservation_node9', collapsed)):
        for w, p in zip(model.trainable_variables, params): w.assign(p)
        responses = [compute_owner_response(model, params, bx, by, ids, owner=c,
            receiver=9, round_id=29, optimizer_step=0) for c, (bx, by, ids) in enumerate(batches)]
        measured, metadata = aggregate_balanced_owner_gradients(params, responses,
            receiver=9, round_id=29, optimizer_step=0, available_classes=tuple(range(10)))
        with tf.GradientTape() as tape:
            logits = model(tf.convert_to_tensor(np.concatenate([b[0] for b in batches])))
            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.convert_to_tensor(np.concatenate([b[1] for b in batches])), logits=logits))
        expected = tape.gradient(ce, model.trainable_variables)
        if any(g is None for g in expected): raise ValueError('Disconnected direct balanced CE gradient.')
        delta = np.concatenate([(a-g.numpy()).ravel() for a, g in zip(measured, expected)])
        norm = np.linalg.norm(np.concatenate([g.numpy().ravel() for g in expected]))
        relative = float(np.linalg.norm(delta)/max(norm, 1e-12))
        maximum = float(np.max(np.abs(delta)))
        if relative > 1e-5 or maximum > 1e-6: raise ValueError('Owner aggregate differs from direct balanced objective.')
        unchanged = params_hash([w.numpy() for w in model.trainable_variables]) == params_hash(params)
        if not unchanged: raise ValueError('Zero-update derivative proof changed model state.')
        outcomes[state] = dict(gradientRelativeDifference=relative, maxAbsoluteGradientDifference=maximum,
            modelUnchanged=True, directCE=float(ce), ownerMeanCE=metadata['crossEntropy'],
            olderClassEncoderGradientNorms={str(r.owner): float(np.sqrt(sum(np.sum(g.astype(np.float64)**2)
                for g in r.gradients[:-2]))) for r in responses if r.owner != 9},
            metadata=metadata)
    result = dict(status='passed', device='CPU', tensorflowVersion=tf.__version__,
        sourceManifestHash=hashlib.sha256((SOURCE/'run.json').read_bytes()).hexdigest(),
        sourceModuleHash=hashlib.sha256((ROOT/'basil_core/cart_owner_gradients.py').read_bytes()).hexdigest(),
        actualModelParameterCount=sum(p.size for p in canonical), optimizerSteps=0,
        newResearchRunsStarted=0, testSetLoaded=False, trainingOnly=True,
        ownerBatchSizeForDerivativeProof=16, researchOwnerBatchSizeApproved=False,
        GPUVerified=False, outcomes=outcomes)
    write_json(PROOF, result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--audit', action='store_true')
    action.add_argument('--verify-owner-objective', action='store_true')
    args = parser.parse_args()
    if args.audit: audit()
    elif args.verify_owner_objective: verify_owner_objective()
    else: print(json.dumps(dict(mode='plan_only', optimizerSteps=0,
        source=str(SOURCE), candidateEnabled=False, researchTrainingImplemented=False)))


if __name__ == '__main__': main()
