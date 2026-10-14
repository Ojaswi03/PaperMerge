#!/usr/bin/env python3
"""No-training independent CNN recomputation/fault-injection diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def check(device):
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow, runtime_metadata
    prepare_environment(device)
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,device)
    from basil_core.research_protocol import build_model,params_hash
    from basil_core.replicated_training_service import TrainingReference,TrainingReplica,ReplicatedTrainingService
    from basil_core.replicated_verification import ComputeRequest,ComputeResult,sign_result
    tf.config.experimental.enable_op_determinism()
    # Synthetic TRAIN-labelled evidence: no CIFAR test loading, SGD or accuracy claim.
    x=np.random.default_rng(2025).uniform(size=(160,32,32,3)).astype(np.float32)
    y=np.repeat(np.arange(10,dtype=np.int32),16)
    ref=TrainingReference(x,y,source_split='train',synthetic=True)
    with tf.device('/GPU:0' if device=='GPU' else '/CPU:0'):
        replicas={i:TrainingReplica(i,build_model('basil_paper_cnn',2025),ref) for i in range(10)}
        service=ReplicatedTrainingService(replicas)
        params=[w.numpy() for w in replicas[0].model.trainable_variables]
        count=sum(p.size for p in params)
        assert count==117706
        original=params_hash(params)
        cases=[]
        for kind in ('gradient','selection_ce'):
            req=ComputeRequest('no-training-certificate-check',0,2,1,0,kind,original,ref.root,
                tuple(range(32,48)),(7,8) if kind=='gradient' else ())
            expected=replicas[3].compute(req,params)
            for fault in ('none','poison','withhold'):
                for r in replicas.values():r.reply=TrainingReplica.reply.__get__(r)
                for i in (() if fault=='none' else (0,1,5,7)):
                    r=replicas[i]
                    if fault=='withhold':
                        def unavailable(*_):raise TimeoutError('Injected missing response')
                        r.reply=unavailable
                    else:
                        def poison(request,p,r=r):
                            good=r.compute(request,p)
                            bad=ComputeResult.from_values(999.,[np.full_like(a,-100.) for a in good.arrays()])
                            return sign_result(r._private,r.identity,request,bad)
                        r.reply=poison
                start=time.monotonic()
                result,proof=service.evaluate(req,params)
                elapsed=time.monotonic()-start
                assert result.digest==expected.digest
                assert original==params_hash(params)
                cases.append(dict(kind=kind,fault=fault,seconds=elapsed,
                    exactHonestResultMatch=True,crossEntropy=result.cross_entropy,
                    gradientNorm=float(np.sqrt(sum(np.sum(a.astype(np.float64)**2) for a in result.arrays()))),
                    certificate=proof))
    paths=('basil_core/replicated_verification.py','basil_core/replicated_training_service.py',
           'scripts/check_replicated_training.py','tests/test_replicated_verification.py')
    return dict(status='helper_certificates_verified',device=device,parameterCount=count,
        runtime=runtime_metadata(tf,{'executionDevice':device}),initialModelHash=original,syntheticDataRoot=ref.root,
        cases=cases,optimizerUpdates=0,researchExperimentsStarted=0,wandbEnabled=False,
        fullResearchReady=False,physicalRPC=False,
        sourceHashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--device',choices=('CPU','GPU'),default='CPU')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=check(args.device) if args.check else dict(fullResearchReady=False,
        researchExperimentsStarted=0,message='Use --check for synthetic CNN evaluations; no SGD.')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
