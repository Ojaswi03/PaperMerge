#!/usr/bin/env python3
"""Prepare local-only Dirichlet α=.2 A/B/C/D and an idle queue; no training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def prepare():
    if (ROOT/'gui/.queue_active.json').exists():
        raise ValueError('Stop the active queue before replacing its configuration.')
    from basil_core.iid_runtime import prepare_environment
    prepare_environment('CPU')
    import numpy as np
    from basil_core.dirichlet_study import CONFIG_ROOT,QUEUE_PRESET,configurations,partition_audit,verify_pairing,validate_configuration
    from basil_core.research_protocol import build_model,params_hash
    from gui.state.experiment_state import from_persisted,to_persisted,validate_experiment
    from scripts.run_noniid_basil_study import save_json
    cache=ROOT/'experiments/cache/cifar10'
    audit,_=partition_audit(np.load(cache/'y_train_int32.npy',mmap_mode='r',allow_pickle=False),
        np.load(cache/'y_test_int32.npy',mmap_mode='r',allow_pickle=False))
    model=build_model('basil_paper_cnn',2025)
    params=[v.numpy() for v in model.trainable_variables]
    if sum(v.size for v in params)!=117706:raise ValueError('Approved CNN architecture changed.')
    configs=configurations(audit['partitionHash'],params_hash(params))
    for cfg in configs:
        validate_configuration(cfg)
        state=from_persisted(cfg)
        if validate_experiment(state) or to_persisted(state)!=cfg:
            raise ValueError('GUI configuration validation/round trip failed.')
        save_json(CONFIG_ROOT/f"{cfg['runId']}.json",cfg)
    old=ROOT/'gui/queue_state.json'
    backup=ROOT/'gui/queues/queue_before_dirichlet_alpha02.json'
    if old.exists() and not backup.exists():save_json(backup,json.loads(old.read_text()))
    save_json(QUEUE_PRESET,configs);save_json(old,configs)
    from gui.services.queue_service import QueueService
    state=QueueService(old).load();QueueService.describe_recovery(state,ROOT)
    evidence=dict(status='prepared_idle',trainingRunsStarted=0,order=list('ABCD'),rounds=30,
        queue=str(QUEUE_PRESET.relative_to(ROOT)),partitionAudit=audit,initialModelHash=params_hash(params),
        differences=verify_pairing(configs),informationModel='receiver_local_train_only',
        wandbEnabled=False,entries=[dict(condition=e.config.extra['conditionId'],status=e.status.value,
            hint=e.execution_hint) for e in state.entries])
    if any(e.status.value!='pending' or not e.execution_hint.startswith('Fresh') for e in state.entries):
        raise ValueError('New queue must be fresh and pending; no guessed restart.')
    save_json(ROOT/'gui/worker_state/dirichlet_alpha02_preparation.json',evidence)
    print(json.dumps(evidence,indent=2));return evidence


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
