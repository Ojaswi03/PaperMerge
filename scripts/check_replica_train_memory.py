#!/usr/bin/env python3
"""Full TRAIN-corpus replica RAM check; one forward evaluation, zero SGD."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def check():
    import fcntl
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    prepare_environment('CPU')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,'CPU')
    from basil_core.isolated_replicas import SandboxEndpoint
    from basil_core.replica_resources import require_host_headroom
    from basil_core.replicated_verification import ComputeRequest,CertificateCollector,digest_json
    from basil_core.research_protocol import build_model,params_hash
    budget=require_host_headroom()
    lock=ROOT/'gui/worker_state/cart_isolated_pool.lock'
    with lock.open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cache=ROOT/'experiments/cache/cifar10'
        x=np.load(cache/'x_train_float32.npy',mmap_mode='r',allow_pickle=False)
        y=np.load(cache/'y_train_int32.npy',mmap_mode='r',allow_pickle=False)
        if x.shape!=(50000,32,32,3) or y.shape!=(50000,) or not np.array_equal(np.bincount(y,minlength=10),[5000]*10):
            raise ValueError('Complete TRAIN-only cache required.')
        h=hashlib.sha256(digest_json(dict(split='train',synthetic=False,shape=x.shape)).encode())
        h.update(x.data);h.update(y.data);root=h.hexdigest()
        model=build_model('basil_paper_cnn',2025);params=[w.numpy() for w in model.trainable_variables]
        with tempfile.TemporaryDirectory(prefix='cart-full-train-memory-') as temp:
            fixture=Path(temp)/'train.npz';np.savez(fixture,images=x,labels=y,synthetic=np.asarray(False))
            endpoint=SandboxEndpoint(fixture,1,timeout=15.)
            try:
                assert endpoint.bootstrap['dataRoot']==root
                req=ComputeRequest('full-train-memory-only',0,0,0,0,'selection_ce',params_hash(params),root,
                    tuple(map(int,np.flatnonzero(y==0)[:16])))
                vote=endpoint.reply(req,params)
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                from basil_core.replicated_verification import vote_message
                Ed25519PublicKey.from_public_bytes(endpoint.public_key).verify(vote.signature,
                    vote_message(vote.signer,vote.request_hash,vote.result.digest))
                cgroup=Path('/sys/fs/cgroup')/endpoint.resources['cgroup'].lstrip('/')
                peak=int((cgroup/'memory.peak').read_text())
                events={k:int(v) for k,v in (line.split() for line in (cgroup/'memory.events').read_text().splitlines())}
                if events.get('oom',0) or events.get('oom_kill',0):raise AssertionError('Full TRAIN replica encountered OOM.')
                return dict(status='passed',samples=50000,classCounts=[5000]*10,testExamplesLoaded=0,
                    dataRoot=root,memoryPeakBytes=peak,memoryEvents=events,limits=endpoint.resources,
                    hostBudget=budget,selectionCrossEntropy=vote.result.cross_entropy,
                    replicasTested=1,quorumClaimed=False,optimizerUpdates=0,researchExperimentsStarted=0)
            finally:endpoint.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();result=check() if a.check else dict(researchExperimentsStarted=0)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
