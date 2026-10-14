"""Paired local-only BASIL study; reuse the historical Dirichlet partitioner.

Configuration inspection imports no numerical/GPU libraries. No helper data,
owner gradients, prototype replay or coordinator TRAIN replication is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from basil_core.protocol_compatibility import DIRICHLET_PROTOCOL as PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'dirichlet_alpha02_local_anchor_30r_v1'
CONFIG_ROOT = ROOT/'gui/configs/nonIID/dirichlet_alpha02_30r'
QUEUE_PRESET = ROOT/'gui/queues/noniid_dirichlet_alpha02_abcd_30r_gpu.json'
LABELS = {'A': 'No attack / no noise / CE + anchor',
          'B': 'Hidden R20 / clean channel / CE + anchor',
          'C': 'Hidden R20 / absolute noise 0.010 / CE + anchor',
          'D': 'Hidden R20 / absolute noise 0.010 / CE + EBM + anchor'}


def configurations(partition_hash, initial_hash):
    from basil_core.noniid_study import study_configs
    configs = study_configs(.01, rounds=50)
    for cfg in configs:
        letter = cfg['conditionId']
        name = f'{letter}_dirichlet_alpha02_local_anchor_30r_gpu'
        cfg.update(experimentProtocol=PROTOCOL, protocolRevision=STUDY,
            lossStudyRevision='dirichlet_local_snapshot_anchor_v1', noniidStudyId=STUDY,
            partitionStrategy='dirichlet', dirichletAlpha=.2, nonIID=True,
            classAssignment='seeded_dirichlet_label_allocation',
            partitionHash=partition_hash, initialModelHash=initial_hash,
            trainingSamples=50000, evaluationSamples=10000, parameterCount=117706,
            conditionId=letter, runId=name, experimentName=name,
            displayName=f'{letter} — Dirichlet α=0.2 — {LABELS[letter]} — 30 rounds GPU',
            nRounds=30, researchValid=False, purpose='fixed_30_round_dirichlet_feasibility_study',
            fullResearchReady=False, informationModel='receiver_local_train_only',
            helperExecution='disabled', cartAlgorithm='disabled', useBasil=True, approach='basil',
            executionMode='deterministic_single_process',
            resultDirectory=f'newResults/nonIID/{STUDY}/{name}',
            plotDirectory=f'newPlots/nonIID/{STUDY}/{name}')
    verify_pairing(configs)
    return configs


def verify_pairing(configs):
    if len(configs)!=4 or [c['conditionId'] for c in configs]!=list('ABCD'):
        raise ValueError('Exactly A/B/C/D in order required.')
    identity={'conditionId','runId','experimentName','displayName','resultDirectory','plotDirectory'}
    allowed={'A/B':{'attackerCount','resolvedAttackerIds','attackHidden'},
             'B/C':{'useChannelNoise','channelNoiseSigmaAbsolute'},
             'C/D':{'localObjective','ebmMode'}}
    diffs={}
    for contrast,a,b in [('A/B',0,1),('B/C',1,2),('C/D',2,3)]:
        diff={k for k in configs[a].keys()|configs[b].keys()
              if configs[a].get(k)!=configs[b].get(k)}-identity
        if diff!=allowed[contrast]:raise ValueError(f'Unpaired {contrast}: {sorted(diff)}')
        diffs[contrast]=sorted(diff)
    return diffs


def validate_configuration(config):
    from basil_core.research_protocol import validate_config
    validate_config(config)
    if config.get('conditionId') not in LABELS:raise ValueError('A/B/C/D required.')
    hashes=[config.get(key,'') for key in ('partitionHash','initialModelHash')]
    if any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
        raise ValueError('Actual canonical partition and initialization hashes required.')
    expected=configurations(*hashes)[ord(config['conditionId'])-ord('A')]
    diff=[k for k,v in expected.items() if config.get(k)!=v]
    if diff:raise ValueError('Fixed Dirichlet configuration differs: '+', '.join(diff))
    forbidden=('diagnosticSamplesPerClass','diagnosticTestSamplesPerClass','helperFaults',
               'recoverySourceDirectory','classImportanceEstimator')
    if any(k in config for k in forbidden):raise ValueError('No subset, helper, or other-study resume permitted.')
    return config


def gui_approved(config):
    if config.get('experimentProtocol')!=PROTOCOL:return False
    try:validate_configuration(config)
    except (ValueError,KeyError,TypeError):return False
    return True


def partition_audit(train_labels, test_labels, seed=2025):
    import numpy as np
    from basil_core.data.cifar import _dirichletPartition
    from basil_core.research_protocol import keyed_rng
    train=np.asarray(train_labels,dtype=np.int32).reshape(-1)
    test=np.asarray(test_labels,dtype=np.int32).reshape(-1)
    if len(train)!=50000 or not np.array_equal(np.bincount(train,minlength=10),[5000]*10):
        raise ValueError('Full CIFAR-10 TRAIN labels required.')
    if len(test)!=10000 or not np.array_equal(np.bincount(test,minlength=10),[1000]*10):
        raise ValueError('Full evaluation-only TEST labels required.')
    indices=_dirichletPartition(train,10,.2,keyed_rng(seed,'dirichlet_partition'))
    joined=np.concatenate(indices)
    if not np.array_equal(np.sort(joined),np.arange(50000)):
        raise ValueError('Partition is not disjoint complete TRAIN coverage.')
    digest=hashlib.sha256(); nodes=[]
    for i,chunk in enumerate(indices):
        if not len(chunk):raise ValueError('Empty node partition; refuse silent resampling.')
        digest.update(np.asarray(chunk,dtype='<i8').tobytes())
        batches=(len(chunk)+511)//512
        nodes.append(dict(nodeId=i,samples=len(chunk),
            classCounts=np.bincount(train[chunk],minlength=10).tolist(),
            batchesPerEpoch=batches,optimizerStepsPerActivation=5*batches,
            finalBatchSize=(len(chunk)-1)%512+1))
    return dict(partitionHash=digest.hexdigest(),partitionStrategy='dirichlet',dirichletAlpha=.2,
        partitionSeed=seed,rngStream='dirichlet_partition',assignedSamples=len(joined),
        uniqueSamples=len(np.unique(joined)),nodes=nodes,testClassCounts=[1000]*10,
        testUsage='evaluation_only',unequalNodeSizes=True),indices


def recovery_plan(directory, config):
    from basil_core.protocol_checkpoint import recovery_plan as inspect_checkpoint
    validate_configuration(config)
    plan=inspect_checkpoint(directory,config)
    if plan['mode']=='verified_replay':
        raise ValueError('Only full stateful checkpoints can resume this new study.')
    return plan
