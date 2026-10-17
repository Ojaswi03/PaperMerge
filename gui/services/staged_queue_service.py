"""Dependency handling for the combined non-IID queue; never starts training."""
from __future__ import annotations

import json
from basil_core.execution_profile import profile_path


def is_later_stage(config):
    return (config.get('horizonStudyRevision')=='dirichlet_prior_fixed_horizons_v1'
        and config.get('nRounds')==500)


def dependency_state(config):
    """A waiting entry may be displayed, but is not permission to execute it."""
    from basil_core.dirichlet_horizon_study import (ROOT,RECEIPT,configurations,
        validate_configuration,validate_evidence,study_name)
    if not is_later_stage(config):return dict(state='ready')
    validate_configuration(config)
    if RECEIPT.exists():
        validate_evidence(config)
        return dict(state='ready')
    parents=configurations(config['partitionHash'],config['initialModelHash'],
        config['readinessEvidenceSha256'],100)
    validate_evidence(parents[0])
    if profile_path(ROOT/config['resultDirectory']).exists():
        raise ValueError('500-round output exists without a valid prerequisite receipt; investigate it.')
    missing=[]
    for parent in parents:
        letter=parent['conditionId'];path=profile_path(ROOT/parent['resultDirectory'])/'run.json'
        if not path.is_file():missing.append(letter);continue
        manifest=json.loads(path.read_text())
        if manifest.get('config')!=parent:
            raise ValueError(f'{letter} 100-round prerequisite configuration differs.')
        if manifest.get('status')=='failed':
            raise ValueError(f'{letter} 100-round prerequisite failed; 500-round stage is blocked.')
        complete=(manifest.get('status')=='completed' and manifest.get('completedRounds')==100
            and manifest.get('completedActivations')==1000)
        if not complete:
            if manifest.get('status')=='completed':
                raise ValueError(f'{letter} claims completion without all 100 rounds / 1,000 activations.')
            missing.append(letter)
    return dict(state='waiting' if missing else 'needs_validation',missing=missing,
        prerequisite=study_name(100))


def validate_queue_for_loading(entries):
    """Allow deferred entries to be queued; never authorize their runner."""
    from gui.state.experiment_state import to_persisted,validate_experiment
    seen=set()
    for entry in entries:
        config=to_persisted(entry.config)
        if config.get('horizonStudyRevision'):
            key=(config['nRounds'],config['conditionId'])
            if key in seen:raise ValueError('Duplicate condition/horizon in the staged queue.')
            seen.add(key)
        if entry.config.form_errors:raise ValueError('Queue entry has invalid form fields.')
        issues=validate_experiment(entry.config)
        if issues:
            if is_later_stage(config):dependency_state(config)
            else:raise ValueError('Queue contains an invalid configuration: '+issues[0].message)


def validate_completed_100():
    """Read-only scientific audit plus atomic receipt; no SGD or W&B calls."""
    from basil_core.dirichlet_horizon_study import (ROOT,EVIDENCE,RECEIPT,checksum,
        configurations,inspect_completed,execution_fingerprints,study_name,validate_evidence)
    from scripts.prepare_dirichlet_horizon_queues import save_new
    base=json.loads(EVIDENCE.read_text())
    config=configurations(base['partitionHash'],base['initialModelHash'],checksum(EVIDENCE),100)[0]
    validate_evidence(config)
    receipt=inspect_completed(ROOT/'newResults/nonIID'/study_name(100),expected_rounds=100)
    receipt.update(baseEvidenceSha256=checksum(EVIDENCE),implementationSha256=execution_fingerprints(config))
    save_new(RECEIPT,receipt)
    for cfg in configurations(base['partitionHash'],base['initialModelHash'],checksum(EVIDENCE),500):
        validate_evidence(cfg)
    return receipt
