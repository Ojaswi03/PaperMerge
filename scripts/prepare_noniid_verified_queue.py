#!/usr/bin/env python3
"""Prepare four verified-CART configurations and an idle GUI queue; zero training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def prepare():
    if (ROOT/'gui/.queue_active.json').exists():
        raise ValueError('A GUI queue is active; stop it before replacing queued configurations.')
    from basil_core.iid_runtime import prepare_environment
    prepare_environment('CPU')
    import numpy as np
    from basil_core.noniid_study import partition_audit
    from basil_core.research_protocol import build_model, params_hash
    from basil_core.noniid_verified_study import CONFIG_ROOT,QUEUE_PRESET,configurations,verify_pairing,validate_configuration,require_transport_proof
    from gui.state.experiment_state import from_persisted,to_persisted,validate_experiment
    from scripts.run_noniid_verified_condition import atomic_json
    from scripts.run_verified_cart_learning import require_proof
    require_proof()
    require_transport_proof()
    cache=ROOT/'experiments/cache/cifar10'
    train=np.load(cache/'y_train_int32.npy',mmap_mode='r',allow_pickle=False)
    test=np.load(cache/'y_test_int32.npy',mmap_mode='r',allow_pickle=False)
    audit,_=partition_audit(train,test)
    model=build_model('basil_paper_cnn',2025)
    params=[w.numpy() for w in model.trainable_variables]
    if sum(w.size for w in params)!=117706:raise ValueError('Approved architecture count changed.')
    configs=configurations(audit['partitionHash'],params_hash(params))
    for config in configs:
        validate_configuration(config)
        state=from_persisted(config)
        if validate_experiment(state) or to_persisted(state)!=config:
            raise ValueError('Configuration failed GUI validation or round trip.')
        atomic_json(CONFIG_ROOT/f"{config['runId']}.json",config)
    # Preserve the previous queue as configuration evidence, not research output.
    old=ROOT/'gui/queue_state.json'
    backup=ROOT/'gui/queues/queue_before_noniid_verified_30r.json'
    if old.exists() and not backup.exists():atomic_json(backup,json.loads(old.read_text()))
    atomic_json(QUEUE_PRESET,configs)
    atomic_json(old,configs)
    from gui.services.queue_service import QueueService
    queue=QueueService(old).load();QueueService.describe_recovery(queue,ROOT)
    evidence=dict(status='prepared_idle',trainingRunsStarted=0,queue=str(QUEUE_PRESET.relative_to(ROOT)),
        order=list('ABCD'),rounds=30,partitionAudit=audit,initialModelHash=params_hash(params),
        parameterCount=117706,differences=verify_pairing(configs),
        entries=[dict(condition=e.config.extra['conditionId'],status=e.status.value,
            completedRounds=e.completed_rounds,hint=e.execution_hint) for e in queue.entries])
    atomic_json(ROOT/'gui/worker_state/noniid_verified_30r_preparation.json',evidence)
    print(json.dumps(evidence,indent=2))
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()


if __name__=='__main__':main()
