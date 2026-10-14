"""Fixed, paired 30-round CART study using certified TRAIN computations.

Configuration and recovery inspection are display independent and never train.
TRAIN replication and a trusted coordinator are explicit protocol assumptions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'cart_verified_one_class_v1'
STUDY = 'cart_verified_abcd_30r_v1'
CONFIG_ROOT = ROOT/'gui/configs/nonIID/cart_verified_30r'
QUEUE_PRESET = ROOT/'gui/queues/noniid_abcd_30r_verified_gpu.json'
FAULTS = {'0': 'poison', '1': 'forged', '5': 'withhold', '7': 'exit'}
LABELS = {'A': 'Clean / no attack / no noise / CART + anchor',
          'B': 'Hidden R20 / clean channel / CART + anchor',
          'C': 'Hidden R20 / absolute noise 0.010 / CART + anchor',
          'D': 'Hidden R20 / absolute noise 0.010 / CART + EBM + anchor'}


def configurations(partition_hash, initial_hash):
    configs = []
    for letter in 'ABCD':
        attack, noise, ebm = letter != 'A', letter in 'CD', letter == 'D'
        name = f'{letter}_noniid_verified_cart_30r_gpu'
        configs.append(dict(schemaVersion=5, experimentProtocol=PROTOCOL,
            noniidStudyId=STUDY, conditionId=letter, runId=name, experimentName=name,
            displayName=f'{letter} — Non-IID — {LABELS[letter]} — 30 rounds GPU',
            dataset='cifar10', nonIID=True, partitionStrategy='one_class_per_node',
            classAssignment='node_id_equals_class_id', nNodes=10, samplesPerNode=5000,
            trainingSamples=50000, evaluationSamples=10000, nRounds=30,
            modelArchitecture='basil_paper_cnn', parameterCount=117706,
            aggregationMode='strict_sequential_handoff', ringOrder='fixed_0_to_9',
            approach='cart', cartAlgorithm='verified_owner_gradients',
            snapshotSelection=True, useBasil=True, basilMemorySize=5,
            assumedByzantineCount=4, snapshotSelectionRule='certified_balanced_available_owner_train_ce',
            snapshotPlausibilityGuard='none', helperSamplesPerOwner=16, selectionSamplesPerOwner=512,
            localEpochs=5, epochSemantics='full_local_dataset', batchSize=512,
            optimizer='sgd', momentum=0., optimizerStateMode='reset_each_activation',
            learningRate=.05, learningRateSchedule='basil_round_decay',
            gradientClipNorm=None, weightDecayCoefficient=0., adaptiveWeightDecayMode='off',
            snapshotAnchorMu=.01, snapshotAnchorMuMax=0., monteCarloNoiseSamples=0,
            localObjective='verified_cart_source_ebm_snapshot_anchor' if ebm else 'verified_cart_snapshot_anchor',
            ebmMode='gradient_norm_objective' if ebm else 'none',
            ebmInnerObjective='receiver_local_ce', ebmCoefficientSemantics='channel_coordinate_variance',
            legacyLambdaIgnored=True, useChannelNoise=noise,
            channelNoiseSemantics='paper_absolute_gaussian', channelNoiseSigmaAbsolute=.01 if noise else 0.,
            channelNoiseStartRound=0, attackerCount=4 if attack else 0,
            resolvedAttackerIds=[0,1,5,7] if attack else [], attackerIds='0,1,5,7' if attack else '',
            attackHidden=attack, attackHiddenStart=20, attackStartRound=20,
            attackType='delayed_hidden_parameter_attack_v1',
            attackSourceStatus='project_model_space_transformation_not_exact_paper_reproduction',
            attackGaussian=False, attackSignFlip=False, attackModelPoison=False,
            attackScaling=False, attackAlie=False, attackIpm=False, attackNoiseAmp=False,
            helperFaults=FAULTS if attack else {}, helperFaultStartRound=20,
            trainerFailurePolicy='verified_coordinator_recompute',
            informationModel='replicated_train_trusted_coordinator',
            helperChannelSemantics='authenticated_exact_recomputation_no_channel_noise',
            helperExecution='concurrent_independent_cpu_endpoints_canonical_vote_order',
            seed=2025, partitionSeed=2025, initializationSeed=2025,
            shuffleSeed=2025, augmentationSeed=2025, attackerSelectionSeed=2025, channelSeed=2025,
            augmentation='seeded_flip_pad_crop', partitionHash=partition_hash, initialModelHash=initial_hash,
            executionDevice='GPU', executionMode='serial_gpu_coordinator_cpu_verifiers',
            gpuMemoryLimitMb=6144, replicaMemoryLimitMb=2048, liveTrackingEnabled=False,
            researchValid=False, purpose='fixed_30_round_noniid_feasibility_study', fullResearchReady=False,
            resultDirectory=f'newResults/nonIID/{STUDY}/{name}',
            plotDirectory=f'newPlots/nonIID/{STUDY}/{name}',
            comparisonResultsDirectory=f'newResults/nonIID/{STUDY}/comparison',
            comparisonPlotsDirectory=f'newPlots/nonIID/{STUDY}/comparison'))
    verify_pairing(configs)
    return configs


def validate_configuration(config):
    letter = config.get('conditionId')
    if letter not in LABELS:
        raise ValueError('Verified non-IID study requires condition A, B, C or D.')
    hashes = [config.get(k, '') for k in ('partitionHash', 'initialModelHash')]
    if any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
        raise ValueError('Actual partition and canonical initialization SHA-256 hashes required.')
    expected = configurations_unchecked(*hashes)[ord(letter)-ord('A')]
    mismatches = [k for k, value in expected.items() if config.get(k) != value]
    if mismatches:
        raise ValueError('Fixed verified non-IID configuration differs: ' + ', '.join(mismatches))
    for key, tree in (('resultDirectory','newResults/nonIID'), ('plotDirectory','newPlots/nonIID')):
        if not (ROOT/config[key]).resolve().is_relative_to((ROOT/tree).resolve()):
            raise ValueError('Non-IID outputs must stay in nonIID.')
    return config


def configurations_unchecked(partition_hash, initial_hash):
    # Config construction verifies differences only; it never calls this validator.
    return configurations(partition_hash, initial_hash)


def verify_pairing(configs):
    if len(configs) != 4 or [c['conditionId'] for c in configs] != list('ABCD'):
        raise ValueError('Exactly four conditions in A/B/C/D order required.')
    identity = {'conditionId','runId','experimentName','displayName','resultDirectory','plotDirectory'}
    permitted = {'A/B': {'attackerCount','resolvedAttackerIds','attackerIds','attackHidden','helperFaults'},
                 'B/C': {'useChannelNoise','channelNoiseSigmaAbsolute'},
                 'C/D': {'localObjective','ebmMode'}}
    diffs = {}
    for label, a, b in (('A/B',0,1),('B/C',1,2),('C/D',2,3)):
        diff = {k for k in configs[a].keys() | configs[b].keys() if configs[a].get(k) != configs[b].get(k)} - identity
        if diff != permitted[label]:
            raise ValueError(f'Unexpected pairing difference {label}: {sorted(diff)}')
        diffs[label] = sorted(diff)
    return diffs


def gui_approved(config):
    if config.get('experimentProtocol') != PROTOCOL:
        return False
    try:
        validate_configuration(config)
    except (ValueError, TypeError, KeyError):
        return False
    return True


def source_hashes():
    from basil_core.verified_cart_runner import fingerprints
    hashes = fingerprints()
    for name in ('basil_core/noniid_verified_study.py', 'basil_core/parallel_verifier_service.py',
                 'scripts/run_noniid_verified_condition.py', 'scripts/run_verified_cart_learning.py',
                 'gui/services/noniid_telemetry.py'):
        hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return hashes


def require_transport_proof():
    from basil_core.verified_cart_runner import fingerprints
    path=ROOT/'gui/worker_state/CART_PARALLEL_VERIFIER_CHECK.json'
    proof=json.loads(path.read_text())
    expected=dict(fingerprints(),**{'basil_core/parallel_verifier_service.py':
        hashlib.sha256((ROOT/'basil_core/parallel_verifier_service.py').read_bytes()).hexdigest()})
    checker=ROOT/'scripts/check_parallel_verifiers.py'
    if (proof.get('status')!='passed' or proof.get('trainingUpdates')!=0 or
            proof.get('sourceHashes')!=expected or
            proof.get('checkerSha256')!=hashlib.sha256(checker.read_bytes()).hexdigest()):
        raise ValueError('Current concurrent verifier equivalence proof required.')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recovery_plan(directory, config):
    """Inspect the committed checkpoint pointer; never reconstruct by training."""
    validate_configuration(config)
    directory = Path(directory)
    pointer = directory/'progress.json'
    if not pointer.exists():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Existing non-IID evidence has no committed checkpoint; inspect it before retrying.')
        return dict(mode='fresh', completedRounds=0, startActivation=0, step=0)
    state = json.loads(pointer.read_text())
    if state.get('config') != config or state.get('sourceHashes') != source_hashes():
        raise ValueError('Checkpoint configuration/source mismatch; exact resume blocked.')
    cursor = state['cursor']
    if (len(cursor) != 3 or any(type(v) is not int for v in cursor) or
            not 0 <= cursor[0] <= 30 or not 0 <= cursor[1] < 10 or not 0 <= cursor[2] <= 50 or
            (cursor[0] == 30 and cursor[1:] != [0,0])):
        raise ValueError('Invalid saved round/node/step cursor.')
    checkpoint = state.get('checkpoint')
    if checkpoint not in {'checkpoint_0.json', 'checkpoint_1.json'}:
        raise ValueError('Invalid checkpoint slot.')
    path = directory/checkpoint
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != state.get('checkpointSha256'):
        raise ValueError('Committed checkpoint is missing or corrupt; no silent rollback.')
    if len(state.get('roundMetrics', [])) != cursor[0]:
        raise ValueError('Completed-round metrics do not match checkpoint cursor.')
    if state.get('status') == 'completed' and cursor != [30,0,0]:
        raise ValueError('False completion; thirty full rounds required.')
    return dict(mode='completed' if state.get('status') == 'completed' else 'checkpoint',
                completedRounds=cursor[0], startActivation=cursor[0]*10+cursor[1], step=cursor[2])
