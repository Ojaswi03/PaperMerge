#!/usr/bin/env python3
"""Bounded namespace/RPC audit with synthetic updates, not research training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def check(device='CPU'):
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    prepare_environment(device)
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,device)
    from basil_core.research_protocol import build_model,params_hash
    from basil_core.replicated_training_service import TrainingReference,TrainingReplica
    from basil_core.replicated_verification import ComputeRequest
    from basil_core.isolated_replicas import IsolatedReplicaPool
    from basil_core.verified_cart_runner import VerifiedCartRunner,fingerprints
    x=np.random.default_rng(2025).uniform(size=(160,32,32,3)).astype(np.float32)
    y=np.repeat(np.arange(10,dtype=np.int32),16)
    ref=TrainingReference(x,y,source_split='train',synthetic=True)
    with tf.device('/CPU:0'):
        model=build_model('basil_paper_cnn',2025)
        params=[w.numpy() for w in model.trainable_variables]
    def active_model():
        with tf.device('/GPU:0' if device=='GPU' else '/CPU:0'):
            active=build_model('basil_paper_cnn',2025)
            for w,p in zip(active.trainable_variables,params):w.assign(p)
            return active
    reference=TrainingReplica(3,model,ref)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='cart-isolated-') as temporary:
        fixture=Path(temporary)/'train.npz'
        np.savez(fixture,images=x,labels=y,synthetic=np.asarray(True))
        # Mix four distinct faults in real isolated processes, not patched objects.
        faults={0:'poison',1:'forged',5:'withhold',7:'exit'}
        start=time.monotonic()
        with IsolatedReplicaPool(fixture,faults=faults,timeout=2.) as pool:
            startup=time.monotonic()-start
            for kind in ('gradient','selection_ce'):
                req=ComputeRequest('isolated-no-training-check',0,2,20,0,kind,
                    params_hash(params),ref.root,tuple(range(32,48)),(7,8) if kind=='gradient' else ())
                # CPU variables alone do not pin forward/backward operations.
                # The oracle must use the same CPU backend as the sandboxes,
                # even when the active training coordinator runs on GPU.
                with tf.device('/CPU:0'):
                    expected=reference.compute(req,params)
                start=time.monotonic()
                result,proof=pool.service.evaluate(req,params)
                assert result.digest==expected.digest, (
                    f'{kind}: CPU oracle {expected.digest} != certified {result.digest}; '
                    f'CE oracle={expected.cross_entropy}, certified={result.cross_entropy}')
                assert len(proof['matchingSigners'])==5
                cases.append(dict(kind=kind,seconds=time.monotonic()-start,
                    exactHonestResultMatch=True,matchingSigners=proof['matchingSigners'],
                    rejected=proof['rejected'],unavailable=proof['unavailableReplicas']))
            bootstraps={str(i):e.bootstrap for i,e in pool.endpoints.items()}
            resources={str(i):e.resources for i,e in pool.endpoints.items()}
            memory_budget=pool.memoryBudget
            runner=VerifiedCartRunner(active_model(),ref,pool.service,
                run_id='isolated-synthetic-step',ebm=True,sigma_e=.01,device=device)
            runner.round=1
            for memory in runner.memories:
                for s in memory.candidates():s.round_id=0 if s.sender_id>=memory.receiver_id else 1
            selected=runner.select({c:tuple(range(16*c,16*c+16)) for c in range(10)})
            assert selected['sender']==9  # Identical candidates: nearest predecessor.
            batch=(tuple(range(16)),(7,8),{c:(tuple(range(16*c,16*c+16)),(7,8)) for c in range(1,10)})
            first=runner.advance(unit_batch=batch)
            checkpoint=Path(temporary)/'checkpoint.json';runner.save(checkpoint)
            uninterrupted=runner.advance(unit_batch=batch)
            cursor=runner.step
            try:
                runner.advance(unit_batch=batch,proposed_params=[np.zeros_like(p) for p in runner.params])
                raise AssertionError('Malicious active update was accepted')
            except ValueError as error:
                assert 'Active-node' in str(error) and runner.step==cursor
        # All endpoint keys/processes are gone. Restore data/state with explicitly
        # recorded fresh trusted membership; compare computation, not signatures.
        with IsolatedReplicaPool(fixture,timeout=2.) as restarted:
            resumed=VerifiedCartRunner(active_model(),ref,restarted.service,
                run_id='isolated-synthetic-step',ebm=True,sigma_e=.01,device=device)
            resumed.restore(checkpoint)
            replay=resumed.advance(unit_batch=batch)
            assert replay['outputHash']==uninterrupted['outputHash']
            assert replay['objective']==uninterrupted['objective']
            assert len(resumed.sessions)==2
            assert params_hash(resumed.anchor)==selected['hash']
            recovered=VerifiedCartRunner(active_model(),ref,restarted.service,
                run_id='explicit-recovery-step',ebm=True,sigma_e=.01,device=device,
                trainer_failure_policy='verified_coordinator_recompute')
            recovered.round=1
            for memory in recovered.memories:
                for s in memory.candidates():s.round_id=0 if s.sender_id>=memory.receiver_id else 1
            recovered.select({c:tuple(range(16*c,16*c+16)) for c in range(10)})
            recovery=recovered.advance(unit_batch=batch,proposal_failure='timeout')
            assert recovery['outputHash']==first['outputHash'] and recovery['recoveredByCoordinator']
    return dict(status='passed',cases=cases,startupSeconds=startup,bootstraps=bootstraps,
        syntheticCommittedSteps=4,maliciousProposalRejected=True,
        explicitTrainerRecoveryMatchesHonest=True,resourceLimits=resources,memoryBudget=memory_budget,
        resumeOutputBitIdentical=True,membershipSessions=2,
        firstStep=first['outputHash'],resumedStep=replay['outputHash'],
        sourceHashes=fingerprints(),researchExperimentsStarted=0,fullResearchReady=False,
        physicalRPC=True,transport='local pipes / bounded JSON',device=device,helperDevice='CPU',
        oracleDevice='CPU',checkerSha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
        privateKeysExported=False,faults=faults)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--coordinator-device',choices=('CPU','GPU'),default='CPU')
    a=p.parse_args()
    result=check(a.coordinator_device) if a.check else dict(researchExperimentsStarted=0,fullResearchReady=False)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
