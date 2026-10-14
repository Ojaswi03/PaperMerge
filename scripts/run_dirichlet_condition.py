#!/usr/bin/env python3
"""Inspect a local Dirichlet condition; --execute explicitly authorizes training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_runtime(config):
    """Full cached partition + real GPU forward/gradient checks, zero SGD."""
    import os
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled',WANDB_MODE='disabled')
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow,runtime_metadata
    prepare_environment('GPU')
    import numpy as np
    import tensorflow as tf
    from basil_core.dirichlet_study import partition_audit
    from basil_core.noniid_study import SnapshotAnchorWorker
    from basil_core.research_protocol import build_model,params_hash,keyed_rng
    from basil_core.protocol_checkpoint import checkpoint_implementation
    from scripts.run_noniid_basil_study import save_json
    configure_tensorflow(tf,'GPU');runtime=runtime_metadata(tf,config)
    cache=ROOT/'experiments/cache/cifar10'
    y=np.load(cache/'y_train_int32.npy',mmap_mode='r',allow_pickle=False)
    ty=np.load(cache/'y_test_int32.npy',mmap_mode='r',allow_pickle=False)
    x=np.load(cache/'x_train_float32.npy',mmap_mode='r',allow_pickle=False)
    tx=np.load(cache/'x_test_float32.npy',mmap_mode='r',allow_pickle=False)
    audit,indices=partition_audit(y,ty,config['seed'])
    assert x.shape==(50000,32,32,3) and tx.shape==(10000,32,32,3)
    assert audit['partitionHash']==config['partitionHash']
    with tf.device('/CPU:0'):
        model=build_model(config['modelArchitecture'],config['seed'])
        initial=[v.numpy() for v in model.trainable_variables]
    assert params_hash(initial)==config['initialModelHash']
    with tf.device('/GPU:0'):
        worker=SnapshotAnchorWorker(build_model(config['modelArchitecture'],config['seed']),mu=.01,device='GPU')
        worker.load(initial)
        for reference,param in zip(worker.reference,initial):reference.assign(param)
        assert all('GPU:' in v.handle.device for v in worker.model.trainable_variables)
        rows=[]
        for node,chunk in enumerate(indices):
            batch=keyed_rng(config['seed'],'snapshot_selection_batch',node).choice(chunk,512,replace=False)
            loss,acc=worker.probe(initial,x[batch],y[batch])
            rows.append(dict(node=node,selectionLoss=loss,selectionAccuracy=acc,samples=len(chunk)))
        batch=keyed_rng(config['seed'],'snapshot_selection_batch',0).choice(indices[0],512,replace=False)
        repeats=[]
        for legacy in (1.,999999.):
            grads,values=worker._gradients(tf.constant(x[batch]),tf.constant(y[batch]),tf.constant([7,8]),
                config['ebmMode'],'paper_absolute_gaussian',tf.constant(config['channelNoiseSigmaAbsolute']),tf.constant(legacy))
            repeats.append(([g.numpy() for g in grads],[float(v) for v in values]))
        for a,b in zip(repeats[0][0],repeats[1][0]):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(repeats[0][1],repeats[1][1])
        assert params_hash(worker.export())==config['initialModelHash']
        assert params_hash([r.numpy() for r in worker.reference])==config['initialModelHash']
        expected=config['channelNoiseSigmaAbsolute']**2 if config['ebmMode']=='gradient_norm_objective' else 0.
        np.testing.assert_allclose(repeats[0][1][3],expected,rtol=1e-6,atol=1e-10)
    report=dict(status='passed',trainingUpdates=0,researchRunsStarted=0,condition=config['conditionId'],
        runtime=runtime,sourceHashes=checkpoint_implementation(config),partitionAudit=audit,
        initialModelHash=params_hash(initial),parameterCount=sum(v.size for v in initial),
        selectionChecks=rows,legacyLambdaIgnored=True,referenceUnchanged=True,
        appliedCoefficient=repeats[0][1][3],baseCE=repeats[0][1][0],
        objective=repeats[0][1][2],gradientNorm=repeats[0][1][5],
        gpuMemory=tf.config.experimental.get_memory_info('GPU:0'),wandbEnabled=False)
    save_json(ROOT/f"gui/worker_state/DIRICHLET_{config['conditionId']}_RUNTIME_VERIFICATION.json",report)
    print(json.dumps(report,allow_nan=False));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--execute',action='store_true')
    action.add_argument('--validate-runtime',action='store_true',help='Full cached data and GPU gradients, zero optimizer updates.')
    parser.add_argument('--recover',action='store_true')
    args=parser.parse_args()
    from basil_core.dirichlet_study import validate_configuration,recovery_plan
    config=json.loads(args.config.read_text())
    validate_configuration(config)
    plan=recovery_plan(ROOT/config['resultDirectory'],config)
    print(json.dumps(dict(status='configuration_valid',trainingStarted=False,
        condition=config['conditionId'],rounds=config['nRounds'],recovery=plan)),flush=True)
    if args.execute:
        from scripts.run_noniid_basil_study import execute
        try:execute(config,recover=args.recover)
        except InterruptedError:
            raise SystemExit(130)
    elif args.validate_runtime:validate_runtime(config)


if __name__=='__main__':main()
