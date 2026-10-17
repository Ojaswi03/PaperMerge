"""Fixed 100/500-round local-prior studies; preparation never starts training.

The loss worker, partition, threats and RNG streams are unchanged. A completed
100-round protocol/numerical audit (not an accuracy target) unlocks 500 rounds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'dirichlet_prior_fixed_horizons_v1'
BASE_STUDY = 'dirichlet_prior_anchor_confirmation_30r_v1'
EVIDENCE = ROOT/'gui/worker_state/dirichlet_horizon_readiness.json'
RECEIPT = ROOT/'gui/worker_state/dirichlet_prior_100r_validation.json'


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def study_name(rounds):
    if type(rounds) is not int or rounds not in (100, 500):
        raise ValueError('Only the explicitly requested 100/500-round horizons are allowed.')
    return f'dirichlet_prior_anchor_{rounds}r_v1'


def configurations(partition_hash, initial_hash, evidence_sha256, rounds):
    from basil_core.dirichlet_calibration import readiness_configs
    study = study_name(rounds)
    configs = readiness_configs(partition_hash, initial_hash, 'threat', _validate=False)
    for cfg in configs:
        letter = cfg['conditionId']
        cfg.update(horizonStudyRevision=REVISION, nRounds=rounds,
            noniidStudyId=study, researchValid=True, fullResearchReady=True,
            purpose='fixed_horizon_dirichlet_local_prior_research',
            displayName=f'{letter} — Dirichlet α=0.2 — fresh {rounds} rounds GPU',
            resultDirectory=f'newResults/nonIID/{study}/{letter}',
            plotDirectory=f'newPlots/nonIID/{study}/{letter}',
            prerequisiteStudy=BASE_STUDY if rounds == 100 else study_name(100),
            readinessEvidenceSha256=evidence_sha256,
            liveTrackingEnabled=rounds == 500,
            trackingNamespace='NONIID RUN', trackingGroup=study,
            wandbEntity='ojaswisinha15-ohio-university', wandbProject='Research Paper')
        # Derived from the pinned full-data partition, not adjustable batching.
        cfg.update(optimizerStepsPerNode=[30,45,50,10,95,70,30,55,45,80],
            optimizerStepsPerRound=510,totalOptimizerSteps=510*rounds)
    return configs


def validate_configuration(config):
    from basil_core.dirichlet_calibration import validate_diagnostic_config
    if config.get('horizonStudyRevision') != REVISION:
        raise ValueError('Explicit fixed-horizon study revision required.')
    digest = config.get('readinessEvidenceSha256', '')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Pinned completed-confirmation evidence required.')
    for field in ('partitionHash','initialModelHash'):
        value=config.get(field,'')
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Canonical full-data partition/initialization hashes required.')
    expected = configurations(config['partitionHash'], config['initialModelHash'], digest, config['nRounds'])
    match = next((c for c in expected if c['conditionId'] == config.get('conditionId')), None)
    if match is None or any(config.get(k) != v for k, v in match.items()):
        raise ValueError('Fixed-horizon configuration changed outside the approved horizon/tracking fields.')
    if any(k.startswith('diagnostic') for k in config):
        raise ValueError('Reduced data are not allowed in the fixed-horizon study.')
    # Validate the exact canonical 30-round scientific template as well. Do not
    # pass the extended config recursively through the calibration router.
    from basil_core.dirichlet_calibration import readiness_configs
    template = readiness_configs(config['partitionHash'], config['initialModelHash'], 'threat', _validate=False)
    validate_diagnostic_config(next(c for c in template if c['conditionId'] == config['conditionId']))
    return config


def execution_fingerprints(config):
    from basil_core.protocol_checkpoint import checkpoint_implementation
    return checkpoint_implementation(config)


def validate_evidence(config):
    validate_configuration(config)
    if checksum(EVIDENCE) != config['readinessEvidenceSha256']:
        raise ValueError('Completed 30-round approval is missing or changed.')
    evidence = json.loads(EVIDENCE.read_text())
    if evidence['status'] != 'protocol_and_numerical_checks_passed':
        raise ValueError('Completed confirmation has not passed the protocol audit.')
    if (evidence['partitionHash'], evidence['initialModelHash']) != (config['partitionHash'], config['initialModelHash']):
        raise ValueError('Partition/initialization pairing differs from the completed confirmation.')
    if evidence['implementationSha256'] != execution_fingerprints(config):
        raise ValueError('Implementation changed after fixed-horizon preparation; revalidate before execution.')
    for name, digest in evidence['manifests'].items():
        if checksum(ROOT/f'newResults/nonIID/{BASE_STUDY}/{name}/run.json') != digest:
            raise ValueError('Completed 30-round evidence changed.')
    if config['nRounds'] == 500:
        if not RECEIPT.is_file():
            raise ValueError('500 rounds blocked: complete and validate ALL A/B/C/D 100-round runs first.')
        receipt = json.loads(RECEIPT.read_text())
        if (receipt.get('status') != 'protocol_and_numerical_checks_passed'
                or receipt.get('baseEvidenceSha256') != checksum(EVIDENCE)
                or receipt.get('implementationSha256') != execution_fingerprints(config)):
            raise ValueError('The 100-round validation receipt is stale or invalid.')
        for name, digest in receipt['manifests'].items():
            if checksum(ROOT/f'newResults/nonIID/{study_name(100)}/{name}/run.json') != digest:
                raise ValueError('Completed 100-round evidence changed after validation.')
        if set(receipt['manifests']) != set('ABCD'):
            raise ValueError('All four completed 100-round conditions are required.')
    return evidence


def gui_approved(config):
    try:
        validate_evidence(config)
    except (ValueError, KeyError, TypeError, OSError):
        return False
    return True


def display_label(config):
    validate_configuration(config)
    attack = 'Hidden R20' if config['attackHidden'] else 'no attack'
    noise = 'absolute σₑ=0.010' if config['useChannelNoise'] else 'no noise'
    objective = 'prior CE + EBM + anchor' if config['ebmMode'] == 'gradient_norm_objective' else 'prior CE + anchor'
    tracking = 'W&B' if config['liveTrackingEnabled'] else 'local only'
    return f"{config['conditionId']} — Dirichlet α=0.2 — {attack} / {noise} / {objective} — {config['nRounds']} rounds GPU / {tracking}"


def recovery_plan(directory, config):
    # Reuse the established checkpoint reader and failed-run guards, with this
    # family's approval callback; never reinterpret a 30/100-round checkpoint.
    from basil_core.dirichlet_prior_study import _recovery_plan
    validate_evidence(config)
    manifest=Path(directory)/'run.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if previous.get('status')=='tracking_unavailable':
            if (previous['config']!=config or previous['completedActivations']!=0
                    or previous['completedRounds']!=0 or (Path(directory)/'checkpoint.npz').exists()
                    or previous['implementationSha256']!=execution_fingerprints(config)):
                raise ValueError('A startup tracking failure has inconsistent scientific state.')
            return dict(mode='startup_retry',savedActivations=0,startActivation=0,completedRounds=0)
    return _recovery_plan(directory, config)


def tracking_settings(config):
    """Do not reuse the old IID prepared cloud IDs or mutate global settings."""
    if not config.get('liveTrackingEnabled'):
        return {'enabled': False}
    return dict(enabled=True, mode='online', entity=config['wandbEntity'],
        project=config['wandbProject'], group=config['trackingGroup'], preparedRuns={})


def inspect_completed(source, *, expected_rounds):
    """Read existing arrays, telemetry and real checkpoint parameters; zero SGD."""
    import gzip
    from reporting.dirichlet_calibration import review, verify_final_transmissions
    source = Path(source)
    result, _ = review(source, names=tuple('ABCD'))
    transmissions = verify_final_transmissions(source, tuple('ABCD'))
    histories = {}
    manifests = {}
    runtimes = {}
    for letter, summary in result['summaries'].items():
        if summary['rounds'] != expected_rounds:
            raise ValueError('Prerequisite horizon is incomplete.')
        manifests[letter] = checksum(source/letter/'run.json')
        runtimes[letter] = summary['runtimeSeconds']
        histories[letter] = json.loads(gzip.decompress((source/letter/'activation_telemetry.json.gz').read_bytes()))
        cfg = json.loads((source/letter/'run.json').read_text())['config']
        if expected_rounds==100:validate_configuration(cfg)
        partition_audit=json.loads((source/letter/'partition_audit.json').read_text())
        expected_steps=[node['optimizerStepsPerActivation'] for node in partition_audit['nodes']]
        if expected_steps!=[30,45,50,10,95,70,30,55,45,80]:
            raise ValueError('Actual per-node optimizer work differs from the pinned partition.')
        for record in histories[letter]:
            expected_attack = cfg['attackHidden'] and record['nodeId'] in cfg['resolvedAttackerIds'] and record['round'] >= 20
            if record['attackActive'] != expected_attack:
                raise ValueError('Attack activation does not match the fixed schedule.')
            if (record['outboundAttackHash'] != record['trainedOutputHash']) != expected_attack:
                raise ValueError('Attack application/hash disagrees with activation.')
            for link in record['outgoingLinks']:
                if bool(link['noiseL2Norm']) != cfg['useChannelNoise']:
                    raise ValueError('Actual noise does not match the configured channel.')
                if link['coordinateSigma'] != (.01 if cfg['useChannelNoise'] else 0.):
                    raise ValueError('Absolute coordinate sigma differs.')
    for a, b in zip(histories['A'][:200], histories['B'][:200]):
        if a['trainedOutputHash'] != b['trainedOutputHash']:
            raise ValueError('Paired A/B pre-attack initialization/training diverged.')
    paired = 0
    for c, d in zip(histories['C'], histories['D']):
        for left, right in zip(c['outgoingLinks'], d['outgoingLinks']):
            if left['noiseHash'] != right['noiseHash']:
                raise ValueError('C/D underlying channel draws are not paired.')
            paired += 1
    identity = {(s['partitionHash'], s['initialModelHash']) for s in result['summaries'].values()}
    if len(identity) != 1:
        raise ValueError('Prerequisite partition/initialization mismatch.')
    partition, initial = identity.pop()
    return dict(status='protocol_and_numerical_checks_passed', rounds=expected_rounds,
        partitionHash=partition, initialModelHash=initial, manifests=manifests,
        pairedNoisyLinks=paired, transmissionAudit=transmissions,
        runtimeSecondsByCondition=runtimes,
        accuracyThresholdUsed=False, corruptedSelectionThresholdUsed=False,
        scientificInterpretation='Correct execution is not proof of beneficial EBM or universal robustness.')
