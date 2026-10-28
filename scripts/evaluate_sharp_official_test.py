#!/usr/bin/env python3
"""Evaluate frozen completed SHARP checkpoints on official TEST, without training.

This is a separate explicit action after development and protocol selection.
It does not change TRAIN-holdout records or feed TEST into CART decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load_official_arrays(dataset):
    import numpy as np
    cache=ROOT/'experiments/cache'/dataset
    if dataset=='mnist':
        archive=Path.home()/'.keras/datasets/mnist.npz'
        if not archive.exists():archive=cache/'mnist.npz'
        with np.load(archive,allow_pickle=False) as raw:
            x=raw['x_test'].astype(np.float32)[...,None]/255.;y=raw['y_test'].astype(np.int32)
        origin=str(archive)
    elif dataset=='cifar10':
        if all((cache/name).exists() for name in ('x_test_float32.npy','y_test_int32.npy')):
            x=np.load(cache/'x_test_float32.npy',mmap_mode='r',allow_pickle=False)
            y=np.load(cache/'y_test_int32.npy',mmap_mode='r',allow_pickle=False)
            origin=str(cache/'x_test_float32.npy')
        else:
            source=Path.home()/'.keras/datasets/cifar-10-batches-py-target/cifar-10-batches-py/test_batch'
            with source.open('rb') as handle:raw=pickle.load(handle,encoding='bytes')
            x=raw[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1).astype(np.float32)/255.
            y=np.asarray(raw[b'labels'],np.int32);origin=str(source)
    elif dataset=='cifar100':
        archive=cache/'cifar-100-python.tar.gz'
        if hashlib.md5(archive.read_bytes()).hexdigest()!='eb9058c3a382ffc7106e4002c42a8d85':
            raise ValueError('Official CIFAR-100 archive checksum mismatch.')
        with tarfile.open(archive,'r:gz') as handle:
            raw=pickle.load(handle.extractfile('cifar-100-python/test'),encoding='bytes')
        x=raw[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1).astype(np.float32)/255.
        y=np.asarray(raw[b'fine_labels'],np.int32);origin=str(archive)+'::test'
    else:raise ValueError('Unknown dataset.')
    classes=100 if dataset=='cifar100' else 10
    shape=(28,28,1) if dataset=='mnist' else (32,32,3)
    if (x.shape!=(10000,*shape) or y.shape!=(10000,) or x.dtype!=np.float32 or y.dtype!=np.int32
            or not np.array_equal(np.unique(y),np.arange(classes))
            or not np.isfinite(x).all() or x.min()<0 or x.max()>1):
        raise ValueError('Expected the full official TEST split; no subset is allowed.')
    return x,y,origin


def aggregate_final_results(results):
    import numpy as np
    groups={}
    for row in results:
        cfg=row['config'];basis={k:v for k,v in cfg.items() if k not in ('seed','runId','baseRunId')}
        key=json.dumps(basis,sort_keys=True)
        group=groups.setdefault(key,dict(config=basis,seeds=[],seedMeanAccuracy=[]))
        if cfg['seed'] in group['seeds']:raise ValueError('Duplicate final TEST seed result.')
        group['seeds'].append(cfg['seed']);group['seedMeanAccuracy'].append(row['meanAccuracy'])
    for group in groups.values():
        order=np.argsort(group['seeds'])
        group['seeds']=[group['seeds'][i] for i in order]
        group['seedMeanAccuracy']=[group['seedMeanAccuracy'][i] for i in order]
        values=group['seedMeanAccuracy'];group['meanAccuracy']=float(np.mean(values))
        group['sampleStd']=float(np.std(values,ddof=1)) if len(values)>1 else None
    return list(groups.values())


def evaluate(args):
    import numpy as np
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import load_state,evaluate as infer
    from basil_core.sharp_data import digest
    from scripts.run_sharp_study import check_source,write_json
    plan=json.loads((args.output/'plan.json').read_text())
    guarded={c['method']=='sharp_guarded' for c in plan['configs']}
    if not guarded or len(guarded)>1:raise ValueError('Use one nonempty, frozen method family per final-evaluation plan.')
    if guarded=={True}:
        from basil_core.sharp_guarded_protocol import check_source as check_guarded,validate_config
        source=dict(plan['source']);launcher='scripts/run_guarded_sharp_study.py'
        extra=source.pop(launcher,None);check_guarded(source)
        if extra is not None and hashlib.sha256((ROOT/launcher).read_bytes()).hexdigest()!=extra:
            raise ValueError('Guarded launcher changed from the frozen final-evaluation plan.')
    else:
        from basil_core.sharp_study import validate_config
        check_source(plan['source'])
    for cfg in plan['configs']:validate_config(cfg)
    for name,digest_value in plan.get('executionProgram',{}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest_value:
            raise ValueError('A frozen training/continuation executor changed.')
    # Check completion before loading any official TEST example.
    manifests=[]
    for cfg in plan['configs']:
        folder=args.output/cfg['runId'];metadata=json.loads((folder/'run.json').read_text())
        if metadata['status']!='COMPLETED' or metadata['config']!=cfg or metadata['source']!=plan['source']:
            raise ValueError('All selected training runs must complete with frozen identities before final testing.')
        manifests.append((cfg,folder,metadata))
    if args.device=='GPU':
        from scripts.run_sharp_study import setup_gpu
        tf=setup_gpu()
    else:
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
    cache={};results=[]
    for cfg,folder,metadata in manifests:
        if cfg['dataset'] not in cache:
            x,y,origin=load_official_arrays(cfg['dataset'])
            identity=dict(origin=origin,imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y))
            cache[cfg['dataset']]=(x,y,identity)
        x,y,identity=cache[cfg['dataset']]
        destination=folder/'official_test_evaluation.json'
        if destination.exists():
            existing=json.loads(destination.read_text())
            if existing['config']!=cfg or existing['testIdentity']!=identity:
                raise ValueError('Existing final TEST receipt has a different identity; never overwrite it.')
            results.append(existing);continue
        worker=SharpWorker(cfg,args.device)
        contract={key:metadata[key] for key in ('config','source','data','partition','runtime','initialHash')}
        state=load_state(folder/'checkpoint.npz',contract,worker.dimension)
        if state['cursor']!=cfg['nodes']*cfg['rounds']:raise ValueError('Final checkpoint is incomplete.')
        accuracies=[];classes=[]
        for params in state['nodeParams']:
            accuracy,per_class=infer(worker,params,x,y,cfg['classes'])
            accuracies.append(accuracy);classes.append(per_class.tolist())
        result=dict(config=cfg,source=plan['source'],testIdentity=identity,officialTestUsed=True,
            purpose='final_frozen_checkpoint_evaluation_only',trainingPerformed=False,
            cartDecisionsUsedTest=False,nodeAccuracy=accuracies,perClass=classes,
            meanAccuracy=float(np.mean(accuracies)),worstNodeAccuracy=float(np.min(accuracies)),
            worstClassMean=float(np.asarray(classes).mean(axis=0).min()),
            evaluationDevice=args.device,evaluationProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_json(destination,result);results.append(result)
        print(f'{cfg["runId"]}: official TEST mean {result["meanAccuracy"]:.2%}',flush=True)
    write_json(args.output/'official_test_summary.json',dict(status='COMPLETED_FINAL_TEST_EVALUATION',
        runs=results,groups=aggregate_final_results(results),officialTestUsed=True,trainingPerformed=False,
        aggregation='Final node means within seed, then matching complete configuration and sigma across seeds.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('CPU','GPU'),default='GPU')
    parser.add_argument('--profile',choices=('local','hpc'),default='local')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.device=='GPU' and not args.worker:
        from basil_core.execution_profile import ExecutionProfile,gpu_inventory
        from scripts.run_sharp_study import worker_environment
        profile=ExecutionProfile(name=args.profile,gpu=args.gpu,
            output_root=str(args.output) if args.profile=='hpc' else '',
            memory_mib=8192 if args.profile=='hpc' else 6144,headroom_mib=2048 if args.profile=='hpc' else 1024).bind(gpu_inventory())
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker'],cwd=ROOT,
            env=worker_environment(profile),check=True)
    else:
        if args.device=='CPU':os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',MPLCONFIGDIR='/tmp/papermerge-mpl')
        evaluate(args)


if __name__=='__main__':main()
