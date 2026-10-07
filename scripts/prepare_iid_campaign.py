#!/usr/bin/env python3
"""Prepare four idle GUI configurations, without training or starting workers."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_DETERMINISTIC_OPS']='1'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    import numpy as np
    from basil_core.iid_campaign import campaign_configs,verify_campaign,CONFIG_ROOT,QUEUE_PATH,COMPARISON
    from basil_core.iid_study import TEST_NAMES,partition_audit
    from basil_core.research_protocol import build_model,params_hash
    from gui.state.experiment_state import from_persisted,to_persisted,validate_experiment
    from gui.services.queue_service import QueueService
    results=ROOT/'newResults/IID'
    old=[json.loads((results/name/'run.json').read_text()) for name in TEST_NAMES]
    if any(m['status']!='completed' or m['completedRounds']!=50 for m in old):raise ValueError('Expected both completed 50-round references.')
    for field in ('iidPartitionHash','initialModelHash'):
        if old[0][field]!=old[1][field]:raise ValueError(f'Existing references disagree: {field}')
    cache=ROOT/'experiments/cache/cifar10'
    train_labels=np.load(cache/'y_train_int32.npy',mmap_mode='r').reshape(-1)
    test_labels=np.load(cache/'y_test_int32.npy',mmap_mode='r').reshape(-1)
    audit,indices=partition_audit(train_labels,test_labels,seed=2025)
    if audit['partitionHash']!=old[0]['iidPartitionHash']:raise ValueError('IID partition changed.')
    with np.load(results/TEST_NAMES[0]/'partition_indices.npz',allow_pickle=False) as saved:
        if any(not np.array_equal(chunk,saved[f'node_{i}']) for i,chunk in enumerate(indices)):raise ValueError('Actual saved indices differ.')
    model=build_model('basil_paper_cnn',2025)
    params=[v.numpy() for v in model.trainable_variables]
    initial_hash=params_hash(params)
    if sum(p.size for p in params)!=117706 or initial_hash!=old[0]['initialModelHash']:raise ValueError('Canonical initialization changed.')
    configs=campaign_configs(audit['partitionHash'],initial_hash)
    clean_seconds=old[0]['runtimeSeconds']*2
    ebm_seconds=old[1]['runtimeSeconds']*2
    for i,cfg in enumerate(configs):
        # Estimates are metadata only and are kept common to avoid contrast differences.
        cfg['estimatedCampaignRuntimeSeconds']=3*clean_seconds+ebm_seconds
        cfg['runtimeEstimateModel']=dict(ceSeconds=clean_seconds,sourceEbmSeconds=ebm_seconds)
        if validate_experiment(from_persisted(cfg)):raise ValueError('GUI validation failed.')
        if to_persisted(from_persisted(cfg))!=cfg:raise ValueError('GUI configuration round-trip changed fields.')
        write_json(CONFIG_ROOT/f"{cfg['experimentName']}.json",cfg)
        for field in ('resultDirectory','plotDirectory'):(ROOT/cfg[field]).mkdir(parents=True,exist_ok=True)
    if (ROOT/'gui/.queue_active.json').exists():raise ValueError('An active queue must not be replaced.')
    write_json(QUEUE_PATH,configs)
    current=ROOT/'gui/queue_state.json'
    if current.exists():
        previous=json.loads(current.read_text())
        if previous and previous!=configs:write_json(ROOT/'gui/queues/previous_queue_before_iid_abcd.json',previous)
    write_json(current,configs)
    queue=QueueService(QUEUE_PATH).load()
    if len(queue.entries)!=4 or any(e.status.value!='pending' for e in queue.entries):raise ValueError('Queue must be idle with four pending entries.')
    comparison=results/COMPARISON;comparison.mkdir(parents=True,exist_ok=True)
    (ROOT/'newPlots/IID'/COMPARISON).mkdir(parents=True,exist_ok=True)
    evidence=dict(status='prepared_idle',researchRunsStarted=0,conditions=list('ABCD'),rounds=100,
        partitionHash=audit['partitionHash'],initialModelHash=initial_hash,parameterCount=117706,
        scientificDifferences=verify_campaign(configs),resumeSupported=False,startModes={c:'fresh R0→R99' for c in 'ABCD'},
        queuePreset=str(QUEUE_PATH.relative_to(ROOT)),actualAttackers={'A':[],'B':[0,1,5,7],'C':[0,1,5,7],'D':[0,1,5,7]},
        sigmaStandardDeviation=.010,sigmaVariance=.0001,
        runtimeEstimates=dict(freshCampaignHours=(3*clean_seconds+ebm_seconds)/3600,
            hypotheticalVerifiedResumeHours=(2.5*clean_seconds+.5*ebm_seconds)/3600,
            resumeEstimateIsNotAnAvailableExecutionMode=True),
        noiseStream='keyed_rng(seed=2025, channel_noise, round, sender, receiver); same underlying draws in C/D')
    write_json(comparison/'preparation.json',evidence)
    write_json(comparison/'iid_partition_audit.json',audit)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':prepare()
