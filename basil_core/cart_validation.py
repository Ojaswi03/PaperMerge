"""Fixed TRAIN-only comparison of unchanged prior, KL CART and loss-barrier CART."""
from __future__ import annotations

import hashlib

REVISION='cart_train_validation_v1'
ROUNDS=10
REPLICATION_STAGE='replication_30r'
REPLICATION_SEEDS=(2025,2026,2027)
REPLICATION_ROUNDS=30
HOLDOUT_SEED=2025
STAGES=('clean','budgeted_clean','noisy_attack',REPLICATION_STAGE)


def validate_training_arrays(images,labels):
    """Check the committed loader's TRAIN-cache format without TEST I/O."""
    import numpy as np
    if images.shape!=(50000,32,32,3) or images.dtype!=np.dtype('float32'):
        raise ValueError('TRAIN images must be float32 with shape (50000, 32, 32, 3).')
    if labels.shape!=(50000,) or labels.dtype!=np.dtype('int32'):
        raise ValueError('TRAIN labels must be int32 with shape (50000,).')
    split_training(labels)  # Validates all ten classes before partitioning.
    for start in range(0,len(images),256):
        block=images[start:start+256]
        if not np.isfinite(block).all() or np.any(block<0) or np.any(block>1):
            raise ValueError('TRAIN images must contain finite normalized values in [0, 1].')


def split_training(labels):
    import numpy as np
    from basil_core.research_protocol import keyed_rng
    labels=np.asarray(labels)
    if (labels.shape!=(50000,) or not np.issubdtype(labels.dtype,np.integer) or
            np.any(labels<0) or np.any(labels>=10) or
            not np.array_equal(np.bincount(labels,minlength=10),[5000]*10)):
        raise ValueError('The complete original TRAIN label vector is required.')
    validation=np.concatenate([keyed_rng(HOLDOUT_SEED,'cart_heldout_train',c).choice(
        np.flatnonzero(labels==c),100,replace=False) for c in range(10)])
    fit=np.flatnonzero(~np.isin(np.arange(50000),validation))
    assert len(fit)==49000 and not np.intersect1d(fit,validation).size
    return fit,validation


def partition(labels,seed=2025):
    import numpy as np
    from basil_core.data.cifar import _dirichletPartition
    from basil_core.research_protocol import keyed_rng
    if type(seed) is not int or seed not in REPLICATION_SEEDS:
        raise ValueError('Only the fixed validation seeds 2025, 2026 and 2027 are allowed.')
    chunks=_dirichletPartition(labels,10,.2,keyed_rng(seed,'dirichlet_partition'))
    digest=hashlib.sha256()
    for chunk in chunks:digest.update(np.asarray(chunk,dtype='<i8').tobytes())
    return digest.hexdigest(),chunks


def configurations(partition_hash,initial_hash,device='GPU',stage='clean',*,seed=2025):
    from basil_core.dirichlet_study import configurations as previous
    from basil_core.research_protocol import resolve_attackers
    if stage not in STAGES:raise ValueError('Unknown fixed validation stage.')
    if type(seed) is not int or seed not in (REPLICATION_SEEDS if stage==REPLICATION_STAGE else (2025,)):
        raise ValueError('Seed differs from the predeclared validation profile.')
    methods=([('prior',0.,'barrier','A'),('kl',1.,'kl','A'),('barrier',1.,'barrier','A')] if stage=='clean' else
        [('prior',0.,'barrier','A'),('budgeted',1.,'budgeted','A')] if stage=='budgeted_clean' else
        [(f'{method}_{letter}',beta,mode,letter) for letter in 'ABCD'
            for method,beta,mode in [('prior',0.,'barrier'),('budgeted',1.,'budgeted')]] if stage==REPLICATION_STAGE else
        [('prior_B',0.,'barrier','B'),('budgeted_B',1.,'budgeted','B'),
         ('prior_C',0.,'barrier','C'),('budgeted_C',1.,'budgeted','C'),
         ('prior_D',0.,'barrier','D'),('budgeted_D',1.,'budgeted','D')])
    output=[]
    for name,beta,mode,letter in methods:
        cfg=dict(previous(partition_hash,initial_hash)[ord(letter)-ord('A')])
        cfg.update(lossStudyRevision=REVISION,protocolRevision=REVISION,cartRevision=REVISION,
            noniidStudyId=REVISION,cartValidationStage=stage,cartBeta=beta,cartMemoryMode=mode,cartProbePerClass=16,
            cartCorrectionRatioCap=.1 if mode=='budgeted' else None,
            cartBarrierSlack=.001,trainingSamples=49000,evaluationSamples=1000,
            evaluationDataOrigin='heldout_TRAIN',officialTestUsed=False,
            nRounds=5 if stage=='noisy_attack' else ROUNDS,
            purpose='implementation_smoke_test' if stage=='noisy_attack' else 'bounded_cart_train_validation',
            attackStartRound=1 if stage=='noisy_attack' else 20,researchValid=False,
            fullResearchReady=False,diagnosticTestSamplesPerClass=100,
            logitCalibrationEnabled=True,logitCalibrationMethod='prior_adjustment',
            logitCalibrationTau=1.,logitCalibrationCountFloor=1.,cartAlgorithm=REVISION,
            localObjective='cart_train_validation_prior_ebm_anchor' if letter=='D' else 'cart_train_validation_prior_anchor',
            executionDevice=device,runId=name,experimentName=name,
            displayName=f'{name}: fixed TRAIN-only CART validation',
            resultDirectory=f'newResults/nonIID/{REVISION}/{name}',
            plotDirectory=f'newPlots/nonIID/{REVISION}/{name}')
        if stage==REPLICATION_STAGE:
            cfg.update(seed=seed,attackerSelectionSeed=seed,
                resolvedAttackerIds=list(resolve_attackers(seed,cfg['attackerCount'])),
                nRounds=REPLICATION_ROUNDS,attackStartRound=20,
                purpose='fixed_cart_three_seed_train_validation',
                noniidStudyId=f'{REVISION}_{REPLICATION_STAGE}_seed_{seed}',
                cartReplicationSeeds=list(REPLICATION_SEEDS),cartHoldoutSeed=HOLDOUT_SEED,
                cartPrimaryEndpoint='mean_node_validation_accuracy_round_indices_20_to_29',
                resultDirectory=f'newResults/nonIID/{REVISION}/{REPLICATION_STAGE}/seed-{seed}/{name}',
                plotDirectory=f'newPlots/nonIID/{REVISION}/{REPLICATION_STAGE}/seed-{seed}/{name}')
        output.append(cfg)
    return output


def validate_config(config):
    from basil_core.research_protocol import validate_config as base
    from basil_core.cart_local_memory import _sha
    if not all(_sha(config.get(k)) for k in ('partitionHash','initialModelHash')):
        raise ValueError('Actual partition and initialization identities required.')
    base(dict(config,lossStudyRevision='dirichlet_local_snapshot_anchor_v1',
        localObjective='source_ebm_snapshot_anchor' if config['ebmMode']=='gradient_norm_objective' else 'snapshot_anchor_only'))
    expected=next((c for c in configurations(config['partitionHash'],config['initialModelHash'],
        config['executionDevice'],config.get('cartValidationStage','clean'),seed=config.get('seed'))
        if c['runId']==config.get('runId')),None)
    if config!=expected:raise ValueError('Only a predeclared fixed TRAIN-only comparison is allowed.')


def selected_configurations(partition_hash,initial_hash,device='GPU',stage='clean',*,seed=2025,pair=None):
    configs=configurations(partition_hash,initial_hash,device,stage,seed=seed)
    if pair is not None:
        if stage!=REPLICATION_STAGE or pair not in ('A','B','C','D'):
            raise ValueError('Pair selection is restricted to a complete A/B/C/D replication pair.')
        configs=[c for c in configs if c['conditionId']==pair]
    return configs


def make_worker(model,config):
    from basil_core.cart_barrier import BarrierCartWorker
    from basil_core.cart_local_memory import LocalMemoryCartWorker
    kind=LocalMemoryCartWorker if config['cartMemoryMode']=='kl' else BarrierCartWorker
    extra=dict(correction_ratio_cap=config['cartCorrectionRatioCap']) if kind is BarrierCartWorker else {}
    worker=kind(model,beta=config['cartBeta'],seed=config['seed'],device=config['executionDevice'],**extra)
    worker.diagnostic_revision=REVISION
    worker.validation_mode=config['cartMemoryMode']
    return worker
