#!/usr/bin/env python3
"""Inspect completed gates; --prepare writes a fresh idle preset, never trains."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def checksum(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def inspect_gates():
    """No threshold tuning: inspect finite contracts, collapse and admission.

    Zero corrupted admission is a conservative confirmation-screen condition,
    not a theorem about arbitrary Byzantine nodes or an algorithm-correctness
    assertion. A faithful minimum-loss rule can fail this scientific screen.
    """
    import numpy as np
    from reporting.dirichlet_calibration import review,verify_final_transmissions
    from basil_core.dirichlet_diagnosis import confusion_summary
    stages={'clean':(10,('baseline','prior_adjusted')),
        'noise':(3,('noise_ce','noise_ebm')),'attack_gate':(23,tuple('BCD'))}
    reports={};proofs={};identities=set();constant_nodes={}
    for stage,(rounds,names) in stages.items():
        root=ROOT/f'newResults/nonIID/dirichlet_prior_readiness_{stage}_{rounds}r_v1'
        result,_=review(root,names=names)
        result['transmissionAudit']=verify_final_transmissions(root,names)
        reports[stage]=result;proofs[stage]={}
        for name in names:
            directory=root/name
            manifest=json.loads((directory/'run.json').read_text())
            identities.add((manifest['partitionHash'],manifest['initialModelHash']))
            proofs[stage][name]={'manifestSha256':checksum(directory/'run.json')}
            if stage=='attack_gate':
                confusions=[]
                for path in directory.glob('audit*/confusions.json'):
                    confusions.extend(json.loads(path.read_text()))
                finals={r['node']:r['confusion'] for r in confusions if r['round']==22 and r['stage']==5}
                if set(finals)!=set(range(10)):raise ValueError('Missing final ten-node confusion matrices.')
                constant_nodes[name]=[node for node,m in finals.items() if confusion_summary(np.asarray(m))['constantPredictionClass'] is not None]
    if len(identities)!=1:raise ValueError('Diagnostic pairing identity mismatch.')
    partition,initial=next(iter(identities))
    # Audit-only global candidate evaluation must not change training/selection.
    records=lambda path:json.loads(gzip.decompress(path.read_bytes()))
    a=records(ROOT/'newResults/nonIID/dirichlet_prior_readiness_clean_10r_v1/prior_adjusted/activation_telemetry.json.gz')
    b=records(ROOT/'newResults/nonIID/dirichlet_prior_readiness_attack_gate_23r_v1/B/activation_telemetry.json.gz')
    keys=('inputHash','trainedOutputHash','outboundAttackHash','inputSourceNode','inputSnapshotRound','globalTestAccuracy')
    if len(a)!=100 or any(a[i][key]!=b[i][key] for i in range(100) for key in keys):
        raise ValueError('Clean/pre-attack warm-up changed across audit wiring.')
    noise_pairing={}
    for stage,rounds,left,right in [('noise',3,'noise_ce','noise_ebm'),('attack_gate',23,'C','D')]:
        root=ROOT/f'newResults/nonIID/dirichlet_prior_readiness_{stage}_{rounds}r_v1'
        lhs=records(root/left/'activation_telemetry.json.gz')
        rhs=records(root/right/'activation_telemetry.json.gz')
        if len(lhs)!=len(rhs) or any(
                [link['noiseHash'] for link in l['outgoingLinks']]!=
                [link['noiseHash'] for link in r['outgoingLinks']] for l,r in zip(lhs,rhs)):
            raise ValueError('Matched absolute-channel Gaussian draws changed between paired losses.')
        noise_pairing[stage]=sum(len(r['outgoingLinks']) for r in lhs)
    failures=[]
    for name in 'BCD':
        summary=reports['attack_gate']['summaries'][name]
        if constant_nodes[name]:failures.append(f'{name}: constant predictors at nodes {constant_nodes[name]}')
        if summary['contracts']['actuallyCorruptedSelections']:
            failures.append(f"{name}: {summary['contracts']['actuallyCorruptedSelections']} corrupted snapshots admitted")
    # Integrity is checked once in preparation, never as a slow GUI poll.
    before=json.loads((ROOT/'gui/worker_state/dirichlet_readiness_before.json').read_text())['files']
    changed=[name for name,entry in before.items() if not (ROOT/name).is_file() or checksum(ROOT/name)!=entry['sha256']]
    if changed:raise ValueError(f'Preserved research artifacts changed: {changed[:10]}')
    scientific=('basil_core/models.py','basil_core/attacks.py','basil_core/trainer.py',
        'basil_core/data/cifar.py','basil_core/research_protocol.py','basil_core/iid_study.py',
        'basil_core/noniid_study.py','basil_core/dirichlet_calibration.py')
    gpu=json.loads((ROOT/'gui/worker_state/DIRICHLET_READINESS_GPU_VERIFICATION.json').read_text())
    if gpu.get('status')!='passed' or not gpu.get('ebmVerified') or gpu['objectiveSha256']!=checksum(ROOT/'basil_core/dirichlet_calibration.py'):
        raise ValueError('Current prior CE / EBM GPU objective not verified.')
    evidence=dict(status='not_ready' if failures else 'ready_for_30_round_confirmation',
        trainingRunsStarted=0,production500RoundsApproved=False,partitionHash=partition,initialModelHash=initial,
        gates=proofs,scientificSourceSha256={name:checksum(ROOT/name) for name in scientific},
        predictorCollapse=constant_nodes,scientificScreenFailures=failures,
        preAttackMatchedActivations=100,pairedChannelTransmissions=noise_pairing,
        protectedFiles=len(before),protectedFilesChanged=0,
        limitations=['Single seed','Three post-onset rounds only','Not universal Byzantine protection',
            'No test-based coefficient tuning; no claim of >50% accuracy','500-round campaign not approved'])
    return evidence,reports


def prepare(evidence):
    from basil_core.dirichlet_prior_study import EVIDENCE,CONFIG_ROOT,QUEUE_PRESET,configurations,validate_evidence
    from scripts.run_noniid_basil_study import save_json
    from basil_core.dirichlet_study import verify_pairing
    from gui.state.experiment_state import from_persisted,to_persisted,validate_experiment
    if (ROOT/'gui/.queue_active.json').exists():raise ValueError('An active GUI queue must not be changed.')
    if evidence['status']!='ready_for_30_round_confirmation':raise ValueError('Scientific confirmation screen failed; no queue prepared.')
    # This is a local approval record, not a public handoff document.
    if EVIDENCE.exists() and json.loads(EVIDENCE.read_text())!=evidence:
        raise ValueError('Existing approval differs; do not silently replace pinned evidence.')
    if not EVIDENCE.exists():save_json(EVIDENCE,evidence)
    cfgs=configurations(evidence['partitionHash'],evidence['initialModelHash'],checksum(EVIDENCE))
    for cfg in cfgs:
        validate_evidence(cfg)
        state=from_persisted(cfg)
        if validate_experiment(state) or to_persisted(state)!=cfg:raise ValueError('GUI configuration round-trip invalid.')
        output=ROOT/cfg['resultDirectory']
        if output.exists():raise ValueError('Fresh confirmation must not overwrite evidence.')
        target=CONFIG_ROOT/f"{cfg['conditionId']}_dirichlet_prior_30r.json"
        if target.exists() and json.loads(target.read_text())!=cfg:raise ValueError('Existing configuration differs.')
        if not target.exists():save_json(target,cfg)
    if QUEUE_PRESET.exists() and json.loads(QUEUE_PRESET.read_text())!=cfgs:raise ValueError('Existing preset differs.')
    if not QUEUE_PRESET.exists():save_json(QUEUE_PRESET,cfgs)
    # Do not replace the user's queue_state.json or start any worker.
    return dict(status='prepared_idle',runs=4,order=list('ABCD'),rounds=30,trainingRunsStarted=0,
        preset=str(QUEUE_PRESET.relative_to(ROOT)),differences=verify_pairing(cfgs),queueAutoStarted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    try:evidence,reports=inspect_gates()
    except (ValueError,OSError,KeyError) as error:
        if args.prepare:raise
        print(json.dumps(dict(status='blocked',trainingRunsStarted=0,reason=str(error))));return
    result=prepare(evidence) if args.prepare else evidence
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
