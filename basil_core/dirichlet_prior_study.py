"""Evidence-gated 30-round GUI confirmation of local-prior BASIL, not CART.

Only configuration/JSON operations on import. The original raw-CE Dirichlet
family remains separate and unchanged. This does not authorize 500 rounds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from basil_core.execution_profile import profile_path

ROOT=Path(__file__).resolve().parents[1]
CONFIG_ROOT=ROOT/'gui/configs/nonIID/dirichlet_prior_confirmation_30r'
QUEUE_PRESET=ROOT/'gui/queues/noniid_dirichlet_prior_abcd_30r_gpu.json'
EVIDENCE=profile_path(ROOT/'gui/worker_state/dirichlet_prior_readiness.json')
STUDY='dirichlet_prior_anchor_confirmation_30r_v1'


def display_label(config):
    """Human-readable title derived from actual validated flags, not a nickname."""
    validate_configuration(config)
    attack='Hidden R20' if config['attackHidden'] else 'no attack'
    noise='absolute σₑ=0.010' if config['useChannelNoise'] else 'no noise'
    objective='prior CE + EBM + anchor' if config['ebmMode']=='gradient_norm_objective' else 'prior CE + anchor'
    return f"{config['conditionId']} — Dirichlet α=0.2 — {attack} / {noise} / {objective} — 30 rounds GPU"


def configurations(partition_hash,initial_hash,evidence_sha256):
    from basil_core.dirichlet_calibration import readiness_configs
    cfgs=readiness_configs(partition_hash,initial_hash,'threat')
    for cfg in cfgs:
        letter=cfg['conditionId']
        cfg.update(resultDirectory=f'newResults/nonIID/{STUDY}/{letter}',
            plotDirectory=f'newPlots/nonIID/{STUDY}/{letter}',
            readinessEvidenceSha256=evidence_sha256)
    return cfgs


def validate_configuration(config):
    from basil_core.dirichlet_calibration import validate_diagnostic_config,GATE_REVISION
    if config.get('lossStudyRevision')!=GATE_REVISION or config.get('readinessStage')!='threat':
        raise ValueError('Only the separate 30-round local-prior confirmation family is permitted.')
    validate_diagnostic_config(config)
    letter=config.get('conditionId')
    if letter not in tuple('ABCD') or config['runId']!=letter:
        raise ValueError('Explicit A/B/C/D condition required.')
    for field,prefix in [('resultDirectory','newResults'),('plotDirectory','newPlots')]:
        if config[field]!=f'{prefix}/nonIID/{STUDY}/{letter}':
            raise ValueError('Confirmation paths must be fresh, separate non-IID outputs.')
    digest=config.get('readinessEvidenceSha256','')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Verified readiness evidence checksum required.')
    return config


def validate_evidence(config):
    validate_configuration(config)
    raw=EVIDENCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=config['readinessEvidenceSha256']:
        raise ValueError('Readiness evidence changed or does not match configuration.')
    evidence=json.loads(raw)
    if evidence.get('status')!='ready_for_30_round_confirmation':
        raise ValueError('Learning/threat gates have not approved a 30-round confirmation.')
    for key in ('partitionHash','initialModelHash'):
        if evidence[key]!=config[key]:raise ValueError('Readiness evidence pairing mismatch.')
    stages={'clean':(10,('baseline','prior_adjusted')),
            'noise':(3,('noise_ce','noise_ebm')),'attack_gate':(23,tuple('BCD'))}
    if set(evidence['gates'])!=set(stages):raise ValueError('All three completed readiness gates required.')
    for stage,(rounds,names) in stages.items():
        proofs=evidence['gates'][stage]
        if set(proofs)!=set(names):raise ValueError('Readiness condition missing.')
        for name in names:
            directory=profile_path(ROOT/f'newResults/nonIID/dirichlet_prior_readiness_{stage}_{rounds}r_v1'/name)
            manifest_raw=(directory/'run.json').read_bytes()
            if hashlib.sha256(manifest_raw).hexdigest()!=proofs[name]['manifestSha256']:
                raise ValueError('Completed diagnostic manifest changed after approval.')
            manifest=json.loads(manifest_raw)
            if (manifest['status']!='completed' or manifest['completedRounds']!=rounds or
                    manifest['completedActivations']!=10*rounds or
                    any(manifest[k]!=config[k] for k in ('partitionHash','initialModelHash'))):
                raise ValueError('Incomplete or unpaired gate evidence.')
            for filename in ('metrics.npz','round_metrics.csv','checkpoint.npz','activation_telemetry.json.gz'):
                if not (directory/filename).is_file():raise ValueError('Required gate artifact missing.')
    required_sources={'basil_core/models.py','basil_core/attacks.py','basil_core/trainer.py',
        'basil_core/data/cifar.py','basil_core/research_protocol.py','basil_core/iid_study.py',
        'basil_core/noniid_study.py','basil_core/dirichlet_calibration.py'}
    if set(evidence['scientificSourceSha256'])!=required_sources:
        raise ValueError('All scientific source fingerprints required.')
    for name,digest in evidence['scientificSourceSha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Scientific source changed after readiness checks: {name}')
    return evidence


def gui_approved(config):
    try:validate_evidence(config)
    except (ValueError,KeyError,TypeError,OSError):return False
    return True


def recovery_plan(directory,config):
    validate_evidence(config)
    return _recovery_plan(directory,config)


def _recovery_plan(directory,config):
    """Shared exact checkpoint mechanics; caller supplies family approval."""
    from zipfile import BadZipFile
    from basil_core.protocol_checkpoint import load_checkpoint
    directory=profile_path(directory)
    manifest_path=directory/'run.json'
    if not manifest_path.exists():
        if directory.exists():raise ValueError('Existing output without a manifest requires investigation.')
        return dict(mode='fresh',savedActivations=0,startActivation=0,completedRounds=0)
    manifest=json.loads(manifest_path.read_text())
    if manifest['config']!=config:raise ValueError('Saved configuration differs from the confirmation.')
    if manifest['status']=='failed':raise ValueError('A failed run must be investigated before recovery.')
    errors=[]
    for path in (directory/'checkpoint.npz',directory/'checkpoint.previous.npz'):
        if not path.exists():continue
        try:
            state=load_checkpoint(path,config,partition_hash=config['partitionHash'],initial_hash=config['initialModelHash'])
        except (ValueError,KeyError,OSError,EOFError,BadZipFile) as error:
            errors.append(str(error));continue
        # Immutable telemetry referenced by the atomic checkpoint is the
        # authoritative committed history. No guessed weight-only replay and
        # no requirement for a second, separately appended JSONL history.
        completed=state['completed_rounds']==config['nRounds']
        if manifest['status']=='completed' and not completed:
            raise ValueError('Completed manifest disagrees with the verified final checkpoint; investigate before proceeding.')
        return dict(mode='completed' if completed and manifest['status']=='completed' else 'checkpoint',
            checkpoint=str(path),savedActivations=state['next_activation'],
            startActivation=state['next_activation'],completedRounds=state['completed_rounds'])
    raise ValueError('No verified activation-boundary checkpoint: '+'; '.join(errors))
