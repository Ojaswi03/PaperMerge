"""Opt-in CART integration contracts; no queue, training or cloud side effects.

Balanced owner evaluation is our adaptation, NOT BASIL Eq. (3). Hash-bound
responses are not Byzantine certificates. Full-node adversaries fail closed
until a helper defense is specified; a trusted auxiliary boundary is narrower.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

REVISION = 'cart_owner_balanced_selection_local_ebm_v1'
FULL_NODE = 'full_node_threats'
TRUSTED_AUXILIARY = 'honest_auxiliary_snapshot_only'


def source_hashes():
    from scripts.run_cart_owner_selection_diagnostic import fingerprints
    root = Path(__file__).resolve().parents[1]
    return dict(fingerprints(), **{name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('basil_core/cart_owner_protocol.py', 'basil_core/cart_owner_ebm.py',
                     'scripts/check_cart_owner_integration.py')})


def device_proof_current():
    path = Path(__file__).resolve().parents[1]/'gui/worker_state/CART_OWNER_INTEGRATION_GPU_VERIFICATION.json'
    try:
        proof = json.loads(path.read_text())
        return proof.get('status') == 'passed' and proof.get('sourceHashes') == source_hashes()
    except (OSError, ValueError):
        return False


def require_helper_boundary(scope, attacker_ids):
    attackers = tuple(attacker_ids)
    if len(set(attackers)) != len(attackers) or any(type(i) is not int or not 0 <= i < 10 for i in attackers):
        raise ValueError('Distinct attacker IDs 0..9 required.')
    if scope not in (FULL_NODE, TRUSTED_AUXILIARY):
        raise ValueError('Explicit helper threat scope required.')
    if scope == FULL_NODE and attackers:
        raise ValueError('Full-node Byzantine helper defense is unresolved; research execution blocked.')
    return dict(helperThreatScope=scope, helperByzantineDefenseVerified=False,
                snapshotAttackers=list(attackers), helperChannel='clean simulated auxiliary channel',
                privacyGuaranteed=False, physicalRPC=False)


@dataclass(frozen=True)
class TrainingProbe:
    owner: int
    indices: tuple[int, ...]

    def __post_init__(self):
        if (type(self.owner) is not int or not 0 <= self.owner < 10 or
                type(self.indices) is not tuple or not 1 <= len(self.indices) <= 512 or
                len(set(self.indices)) != len(self.indices) or
                any(type(i) is not int or not 0 <= i < 50000 for i in self.indices)):
            raise ValueError('Immutable bounded owner TRAIN indices required.')


def training_probes(train_labels, indices_by_owner, *, source_split='train'):
    """Freeze explicitly supplied TRAIN batches, identical across candidates."""
    import numpy as np
    if source_split != 'train':
        raise ValueError('Only training-side candidate evidence is permitted.')
    labels = np.asarray(train_labels)
    if labels.shape != (50000,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Complete CIFAR training labels required, not test/validation labels.')
    if not np.array_equal(np.bincount(labels, minlength=10), [5000]*10):
        raise ValueError('Full ten-class training provenance required.')
    probes = []
    for owner, indices in sorted(indices_by_owner.items()):
        ids = np.asarray(indices)
        if (type(owner) is not int or not 0 <= owner < 10 or ids.ndim != 1 or
                not np.issubdtype(ids.dtype, np.integer) or not 1 <= len(ids) <= 512 or
                len(set(ids.tolist())) != len(ids) or np.any((ids < 0) | (ids >= 50000)) or
                np.any(labels[ids] != owner)):
            raise ValueError('Distinct bounded owner-local TRAIN indices required.')
        probes.append(TrainingProbe(owner, tuple(int(i) for i in ids)))
    if not probes:
        raise ValueError('Available training owners required; never invent absent classes.')
    return tuple(probes)


def select_balanced_snapshot(receiver, candidates, probes, evaluator, *,
                             helper_scope, attacker_ids=(), available_owners):
    """Mean owner-local CE; evaluate all candidates on the SAME per-owner batches.

    evaluator(params, owner, training_indices) returns locally computed CE.
    Sender telemetry and global evaluation are not arguments or criteria.
    This does not authenticate malicious owner claims. The caller must execute
    evaluation at the declared trusted logical boundary, not accept reported CE.
    """
    from .research_protocol import params_hash
    require_helper_boundary(helper_scope, attacker_ids)
    if type(receiver) is not int or not 0 <= receiver < 10:
        raise ValueError('Receiver 0..9 required.')
    owners = tuple(available_owners)
    if (not owners or receiver not in owners or len(set(owners)) != len(owners) or
            any(type(c) is not int or not 0 <= c < 10 for c in owners)):
        raise ValueError('Distinct available owners including receiver required.')
    probes = tuple(probes)
    if (any(not isinstance(p, TrainingProbe) for p in probes) or
            [p.owner for p in probes] != sorted(owners)):
        raise ValueError('Exact available-owner probe coverage required.')
    candidates = tuple(candidates)
    allowed = {(receiver-distance) % 10 for distance in range(1, 6)}
    if (len(candidates) != 5 or {s.sender_id for s in candidates} != allowed or
            any(type(s.round_id) is not int or s.round_id < -1 for s in candidates)):
        raise ValueError('Exactly five distinct predecessor snapshots required.')
    records = []
    for snapshot in candidates:
        digest = params_hash(snapshot.params)
        losses = {}
        for probe in probes:
            value = float(evaluator(snapshot.params, probe.owner, probe.indices))
            if not math.isfinite(value) or value < 0:
                raise FloatingPointError('Non-finite/negative owner selection CE.')
            losses[probe.owner] = value
            if params_hash(snapshot.params) != digest:
                raise ValueError('Candidate evaluation changed snapshot parameters.')
        records.append(dict(senderId=snapshot.sender_id, snapshotRound=snapshot.round_id,
                            parameterHash=digest, ownerLosses=losses,
                            balancedTrainingLoss=math.fsum(losses.values())/len(losses),
                            receiverLocalLoss=losses[receiver]))
    minimum = min(r['balancedTrainingLoss'] for r in records)
    indices = [i for i, r in enumerate(records) if r['balancedTrainingLoss'] <= minimum+1e-8]
    selected = min(indices, key=lambda i: (receiver-candidates[i].sender_id) % 10)
    return candidates[selected], records, records[selected]['receiverLocalLoss']


def snapshot_links(trained, *, seed, round_id, sender, attacker_ids,
                   sigma_e, attack_start=20):
    """Existing Hidden model attack, THEN independent absolute noise per link.

    This protects no helper channel and claims no exact attack-paper reproduction.
    Clean trained parameters are never modified. No Gaussian attack is used.
    """
    from .attacks import hiddenAttack
    from .research_protocol import copy_params, params_hash, keyed_rng, apply_channel_noise
    import numpy as np
    if (type(round_id) is not int or round_id < 0 or type(sender) is not int or
            not 0 <= sender < 10 or attack_start != 20 or
            not math.isfinite(sigma_e) or sigma_e < 0):
        raise ValueError('Fixed R20 attack, valid node/round and absolute sigma required.')
    attacker_ids = tuple(attacker_ids)
    require_helper_boundary(TRUSTED_AUXILIARY, attacker_ids)
    params = copy_params(trained)
    if not params or any(not np.isfinite(p).all() for p in params):
        raise FloatingPointError('Non-finite honest outbound model.')
    before = params_hash(params)
    active = sender in attacker_ids and round_id >= attack_start
    outbound = hiddenAttack(params, rng=keyed_rng(seed, 'hidden_attack', round_id, sender)) if active else params
    if any(not np.isfinite(p).all() for p in outbound):
        raise FloatingPointError('Non-finite attacked outbound model.')
    outbound_hash = params_hash(outbound)
    links = []
    for distance in range(1, 6):
        receiver = (sender+distance) % 10
        received, statistics = apply_channel_noise(outbound, semantics='paper_absolute_gaussian',
            sigma=sigma_e, rng=keyed_rng(seed, 'channel_noise', round_id, sender, receiver))
        if any(not np.isfinite(p).all() for p in received):
            raise FloatingPointError('Non-finite received snapshot.')
        links.append(dict(receiver=receiver, params=received, noise=statistics,
                          attackActive=active, trainedHash=before, attackedHash=outbound_hash,
                          receivedHash=params_hash(received)))
    return links


def integration_plan(helper_scope=FULL_NODE):
    """Zero-training design/configuration checks; no research queue or horizon chosen."""
    conditions = []
    device_verified = device_proof_current()
    for letter in 'ABCD':
        attackers = [] if letter == 'A' else [0, 1, 5, 7]
        try:
            boundary = require_helper_boundary(helper_scope, attackers)
            blockers = ['balanced selection learning validation pending', 'research horizon not authorized']
        except ValueError as error:
            boundary = dict(helperThreatScope=helper_scope)
            blockers = [str(error), 'balanced selection learning validation pending', 'research horizon not authorized']
        if not device_verified:
            blockers.append('new GPU revision unverified')
        conditions.append(dict(condition=letter, methodRevision=REVISION,
            experimentProtocol='sequential_basil_one_class_v1', partitionStrategy='one_class_per_node',
            nNodes=10, samplesPerNode=5000, localEpochs=5, batchSize=512, nRounds=None,
            optimizer='sgd', momentum=0., optimizerStateMode='reset_each_activation',
            learningRate=.05, learningRateSchedule='basil_round_decay',
            gradientClipNorm=None, weightDecayCoefficient=0., modelArchitecture='basil_paper_cnn',
            aggregationMode='strict_sequential_handoff', basilMemorySize=5,
            snapshotSelection=True, snapshotSelectionRule='balanced_owner_training_ce',
            cartAnchorRule='existing_receiver_supported_confidence_gap', cartMuMax=.01,
            helperSamples=16, resolvedAttackerIds=attackers, attackStartRound=20,
            attackType='delayed_hidden_parameter_attack_v1',
            channelNoiseSemantics='paper_absolute_gaussian', channelNoiseStartRound=0,
            channelNoiseSigmaAbsolute=.01 if letter in 'CD' else 0.,
            ebmMode='gradient_norm_objective' if letter == 'D' else 'none',
            ebmInnerObjective='receiver_local_ce', ebmCoefficientSemantics='channel_coordinate_variance',
            helperChannelNoise=False, **boundary, researchValid=False, blockers=blockers,
            liveTrackingEnabled=False, queueStarted=False))
    return dict(trainingRunsStarted=0, researchReady=False, representativeGpuProofCurrent=device_verified,
                conditions=conditions)
