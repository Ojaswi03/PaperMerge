#!/usr/bin/env python3
"""Bounded real-TRAIN, GPU EBM/Hidden/channel/helper-fault correctness replay.

Default does nothing. Two matched five-epoch activations at an explicitly
constructed R20 boundary; not twenty executed rounds or an A/B/C/D campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUTPUT=ROOT/'gui/worker_state/cart_full_batch_threat_v1'
FAULTS={0:'poison',1:'forged',5:'withhold',7:'exit'}
ATTACKERS=(0,1,5,7)
RUN_ID='cart_full_batch_boundary_replay_v1'


def scope():
    return dict(purpose='bounded_full_batch_boundary_replay',researchValid=False,
        logicalRound=20,roundsActuallyExecuted=0,matchedActivations=2,stepsPerActivation=50,
        localEpochs=5,batchSize=512,lastBatchSize=392,samplesPerLocalClass=5000,
        sigma_e=.01,variance=.0001,mu=.01,ebmInnerObjective='receiver_local_ce',
        attackerIds=list(ATTACKERS),helperFaults=FAULTS,wandb=False,testExamplesLoaded=0,
        informationModel='replicated_train_trusted_coordinator',device='GPU',helperDevice='CPU',
        fullResearchReady=False)


def health(pool):
    """Do not invent zero OOM counters for deliberately terminated peers."""
    values={}
    for identity,endpoint in pool.endpoints.items():
        if endpoint.process is None:
            values[str(identity)]=dict(status='closed_by_transport',memoryCountersAvailable=False)
            continue
        root=Path('/sys/fs/cgroup')/endpoint.resources['cgroup'].lstrip('/')
        events={k:int(v) for k,v in (s.split() for s in (root/'memory.events').read_text().splitlines())}
        if events.get('oom',0) or events.get('oom_kill',0):raise RuntimeError(f'Replica {identity} OOM')
        values[str(identity)]=dict(status='running',memoryCountersAvailable=True,
            peakBytes=int((root/'memory.peak').read_text()),events=events)
    return values


def channel_contract(params,round_id,*,links=None):
    """Independent algebra/RNG reconstruction; no gradient or SGD here."""
    import numpy as np
    from basil_core.research_protocol import params_hash,keyed_rng
    from basil_core.cart_owner_protocol import snapshot_links
    clean_hash=params_hash(params)
    active=round_id>=20
    rng=keyed_rng(2025,'hidden_attack',round_id,0)
    attacked=[((1-.55)*p+.55*(-1.4*p+rng.normal(0,.05*max(np.std(p),.01),p.shape))).astype(np.float32)
              for p in params] if active else params
    actual=links if links is not None else snapshot_links(params,seed=2025,round_id=round_id,sender=0,
        attacker_ids=ATTACKERS,sigma_e=.01)
    noises=[];summaries=[]
    for link in actual:
        receiver=link['receiver'];rng=keyed_rng(2025,'channel_noise',round_id,0,receiver)
        noise=[rng.normal(0.,.01,p.shape).astype(np.float32) for p in attacked]
        expected=[(p+n).astype(np.float32) for p,n in zip(attacked,noise)]
        assert link['attackActive']==active and link['attackedHash']==params_hash(attacked)
        assert link['trainedHash']==clean_hash and link['receivedHash']==params_hash(expected)
        for a,b in zip(link['params'],expected):np.testing.assert_array_equal(a,b)
        assert link['noise']['noiseHash']==params_hash(noise)
        assert link['noise']['effectiveCoordinateSigma']==.01
        std=float(np.std(np.concatenate([n.ravel() for n in noise]).astype(np.float64)))
        assert abs(std-.01)<.0001
        noises.append(link['noise']['noiseHash'])
        summaries.append(dict(receiver=receiver,attackActive=active,empiricalCoordinateStd=std,
            trainedHash=clean_hash,attackedHash=link['attackedHash'],receivedHash=link['receivedHash'],**link['noise']))
    assert len(set(noises))==5 and params_hash(params)==clean_hash
    return summaries


def mathematical_contract(runner):
    """Two lambda values, CE control and independent nested derivative at one state."""
    import numpy as np
    import tensorflow as tf
    from basil_core.replicated_verification import ComputeRequest
    from basil_core.research_protocol import params_hash
    from basil_core.research_audit import augment_batch
    from basil_core.cart_owner_ebm import owner_objective_gradients
    from basil_core.trainer import lossFn
    ids,key,helpers=runner.next_batch();responses=[]
    for owner,(indices,aug) in helpers.items():
        req=ComputeRequest(RUN_ID,0,owner,20,runner.step,'gradient',params_hash(runner.params),
            runner.reference.root,indices,aug)
        response,_=runner.service.gradient_response(req,runner.params);responses.append(response)
    for w,p in zip(runner.model.trainable_variables,runner.params):w.assign(p)
    with tf.device('/CPU:0'):images=augment_batch(runner.reference.images[list(ids)],np.asarray(key,np.int32))
    labels=runner.reference.labels[list(ids)];weights=runner.model.trainable_variables
    kwargs=dict(receiver=0,round_id=20,optimizer_step=runner.step,mu=.01,sigma_e=.01)
    full,values,meta=owner_objective_gradients(runner.model,images,labels,runner.anchor,responses,
        ebm=True,legacy_lambda=1.,**kwargs)
    repeat,repeated,_=owner_objective_gradients(runner.model,images,labels,runner.anchor,responses,
        ebm=True,legacy_lambda=999999.,**kwargs)
    clean,clean_values,_=owner_objective_gradients(runner.model,images,labels,runner.anchor,responses,
        ebm=False,legacy_lambda=999999.,**kwargs)
    for a,b in zip(full,repeat):np.testing.assert_array_equal(a.numpy(),b.numpy())
    np.testing.assert_array_equal([float(v) for v in values],[float(v) for v in repeated])
    for p,a,b in zip(runner.params,full,repeat):
        np.testing.assert_array_equal(p-.025*a.numpy(),p-.025*b.numpy())
    # External helper gradients are already certified CPU derivatives. Use a
    # linear surrogate with those derivatives at this state, not GPU recomputation
    # of CPU helpers or a claim of CPU/GPU bit equality.
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce=lossFn(labels,runner.model(images,training=True))
        base=inner.gradient(ce,weights)
        coefficient=tf.constant(.01,tf.float32)**2
        gradient_sq=tf.add_n([tf.reduce_sum(g*g) for g in base])
        anchor=.01/2*tf.add_n([tf.reduce_sum((w-tf.constant(a))**2) for w,a in zip(weights,runner.anchor)])
        helper_linear=tf.add_n([tf.reduce_sum(w*tf.constant(r.gradients[j]))
            for r in responses for j,w in enumerate(weights)])
        objective_for_derivative=(ce+helper_linear)/10+coefficient*gradient_sq+anchor
    expected=outer.gradient(objective_for_derivative,weights)
    max_error=0.
    for a,b in zip(full,expected):
        np.testing.assert_allclose(a.numpy(),b.numpy(),atol=2e-6,rtol=1e-4)
        max_error=max(max_error,float(np.max(np.abs(a.numpy()-b.numpy()))))
    expected_value=(float(ce)+sum(r.cross_entropy for r in responses))/10+float(coefficient*gradient_sq+anchor)
    np.testing.assert_allclose(float(values[2]),expected_value,atol=2e-6,rtol=1e-6)
    assert clean_values[3]==0 and clean_values[6]==0
    delta=float(np.sqrt(sum(np.sum((a.numpy().astype(np.float64)-b.numpy())**2) for a,b in zip(full,clean))))
    np.testing.assert_allclose(delta,meta['ebmCorrectionNorm'],atol=2e-6,rtol=1e-3)
    assert delta>0 and params_hash([w.numpy() for w in weights])==params_hash(runner.params)
    return dict(batchSize=len(ids),step=runner.step,baseCE=float(ce),gradientNormSquared=float(gradient_sq),
        objective=float(values[2]),ebmCoefficient=float(coefficient),ebmPenalty=meta['ebmPenalty'],
        ebmCorrectionNorm=meta['ebmCorrectionNorm'],ebmCorrectionRatio=meta['ebmCorrectionRatio'],
        ceVsEbmGradientDifference=delta,maxIndependentGradientError=max_error,
        lambdaValues=[1,999999],lambdaLossIdentical=True,lambdaGradientIdentical=True,
        lambdaProspectiveUpdateIdentical=True,modelUnchanged=True)


def check():
    from scripts.run_verified_cart_learning import require_proof,OUTPUT as SOURCE,atomic_json
    proof_hash=require_proof()
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled',WANDB_MODE='disabled')
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow,runtime_metadata
    prepare_environment('GPU')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,'GPU')
    from basil_core.replicated_training_service import TrainingReference
    from basil_core.replica_wire import unpack_arrays
    from basil_core.replicated_verification import digest_json
    from basil_core.research_protocol import build_model,params_hash,keyed_rng,Snapshot
    from basil_core.cart_owner_protocol import snapshot_links
    from basil_core.verified_cart_runner import VerifiedCartRunner,fingerprints
    from basil_core.isolated_replicas import IsolatedReplicaPool
    source_progress=json.loads((SOURCE/'progress.json').read_text())
    if source_progress['status']!='completed':raise ValueError('Completed clean source fixture required.')
    source_path=SOURCE/source_progress['checkpoint'];envelope=json.loads(source_path.read_text());source=envelope['state']
    if digest_json(source)!=envelope['digest'] or source['config']['sourceHashes']!=fingerprints():
        raise ValueError('Source fixture integrity/source mismatch.')
    node_params=[[p.copy() for p in unpack_arrays(source['weights'][digest])] for digest in source['nodeParams']]
    cache=ROOT/'experiments/cache/cifar10'
    images=np.load(cache/'x_train_float32.npy',mmap_mode='r',allow_pickle=False)
    labels=np.load(cache/'y_train_int32.npy',mmap_mode='r',allow_pickle=False)
    reference=TrainingReference(images,labels,source_split='train')
    assert reference.root==source['config']['dataRoot']
    probes={i:tuple(map(int,keyed_rng(2025,'selection_batch',i).permutation(np.flatnonzero(labels==i))[:512])) for i in range(10)}
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/'manifest.json',dict(**scope(),sourceCheckpoint=str(source_path.relative_to(ROOT)),
        sourceCheckpointSha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),dataRoot=reference.root,
        gpuProofSha256=proof_hash,sourceHashes=fingerprints(),checkerSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=runtime_metadata(tf,{'executionDevice':'GPU'})))
    cases={};contracts=[];started=time.monotonic();runner=None;current_case='setup';last_saved=None
    try:
        with tempfile.TemporaryDirectory(prefix='cart-full-batch-') as temporary:
            fixture=Path(temporary)/'train.npz'
            np.savez(fixture,images=images,labels=labels,synthetic=np.asarray(False))
            for current_case,faults in (('honest',{}),('four_faults',FAULTS)):
                with IsolatedReplicaPool(fixture,faults=faults,timeout=2. if faults else 15.) as pool:
                    with tf.device('/GPU:0'):
                        model=build_model('basil_paper_cnn',2025)
                        for w,p in zip(model.trainable_variables,node_params[9]):w.assign(p)
                    runner=VerifiedCartRunner(model,reference,pool.service,run_id=RUN_ID,device='GPU',
                        ebm=True,sigma_e=.01,attacker_ids=ATTACKERS,trainer_failure_policy='verified_coordinator_recompute')
                    # Explicit test fixture, NOT resume or fabricated R1..R19 metrics.
                    runner.round=20
                    for sender in range(10):
                        for link in snapshot_links(node_params[sender],seed=2025,round_id=19,
                                sender=sender,attacker_ids=ATTACKERS,sigma_e=.01):
                            runner.memories[link['receiver']].receive(Snapshot(sender,19,link['params'],channel_metadata=link['noise']))
                    initial_health=health(pool)
                    selection=runner.select(probes)
                    if current_case=='four_faults':
                        assert selection['hash']==cases['honest']['selection']['hash']
                        assert selection['sender']==cases['honest']['selection']['sender']
                    runner.save(OUTPUT/f'{current_case}_selected.json');last_saved=f'{current_case}_selected.json'
                    for step in range(50):
                        if current_case=='honest' and step in (0,9):contracts.append(mathematical_contract(runner))
                        kwargs={}
                        if current_case=='four_faults':
                            if step==0:kwargs['proposed_params']=[np.zeros_like(p) for p in runner.params]
                            if step==1:kwargs['proposal_failure']='timeout'
                        record=runner.advance(**kwargs)
                        if current_case=='four_faults':
                            baseline=cases['honest']['records'][step]
                            assert record['outputHash']==baseline['outputHash'] and record['objective']==baseline['objective']
                        if runner.step%10==0:
                            runner.save(OUTPUT/f'{current_case}_epoch_{runner.step//10}.json')
                            last_saved=f'{current_case}_epoch_{runner.step//10}.json'
                            observation=dict(case=current_case,epoch=runner.step//10,steps=runner.step,
                                objective=record['objective'],ebmPenalty=record['ebmPenalty'],
                                ebmCorrectionRatio=record['ebmCorrectionRatio'],resources=health(pool))
                            atomic_json(OUTPUT/f'{current_case}_epoch_{runner.step//10}_health.json',observation)
                            print(json.dumps({k:v for k,v in observation.items() if k!='resources'}),flush=True)
                    honest_output=[p.copy() for p in runner.params]
                    links=runner.finish_activation()
                    channel=channel_contract(honest_output,20,links=links)
                    if current_case=='honest':
                        warmup=dict(round0=channel_contract(honest_output,0),round19=channel_contract(honest_output,19))
                    else:
                        assert [x['receivedHash'] for x in channel]==[x['receivedHash'] for x in cases['honest']['channel']]
                        assert runner.records[0]['trainerFailure']=='incorrect_update' and runner.records[0]['recoveredByCoordinator']
                        assert runner.records[1]['trainerFailure']=='timeout' and runner.records[1]['recoveredByCoordinator']
                    # Node 1 selects from the actual freshly delivered Hidden+noisy
                    # snapshot and four R19 fixtures; no extra optimizer activation.
                    receiver_selection=runner.select(probes)
                    runner.save(OUTPUT/f'{current_case}_delivered_selected.json');last_saved=f'{current_case}_delivered_selected.json'
                    cases[current_case]=dict(selection=selection,records=runner.records,
                        outputHash=params_hash(honest_output),channel=channel,nextReceiverSelection=receiver_selection,
                        resourceLimits={str(i):e.resources for i,e in pool.endpoints.items()},
                        resourceStart=initial_health,resourceEnd=health(pool),helperCalls=pool.service.calls)
                    atomic_json(OUTPUT/f'{current_case}_evidence.json',cases[current_case])
                del runner,model;runner=None
            assert cases['honest']['nextReceiverSelection']['hash']==cases['four_faults']['nextReceiverSelection']['hash']
            summary=dict(status='passed',**scope(),actualOptimizerUpdates=100,fullActivationsExecuted=2,
                fullLocalEpochsExecuted=10,batchSizes=[512,392],mathematicalContracts=contracts,
                allFiftyUpdatesExactlyMatch=True,selectionExactlyMatches=True,
                outgoingAttackedNoisyLinksExactlyMatch=True,incorrectProposalRecovered=True,
                explicitTrainerTimeoutRecovered=True,warmupChecks=warmup,
                nextReceiverSelectedSender=cases['honest']['nextReceiverSelection']['sender'],
                nextReceiverSelectedCorruptedSnapshot=cases['honest']['nextReceiverSelection']['sender']==0,
                seconds=time.monotonic()-started,gpuMemory=tf.config.experimental.get_memory_info('GPU:0'),
                sourceHashes=fingerprints(),testAccuracyMeasured=False)
            atomic_json(OUTPUT/'summary.json',summary)
            return summary
    except BaseException as error:
        failure=dict(status='failed',case=current_case,exceptionType=type(error).__name__,message=str(error),
            cursor=[runner.round,runner.node,runner.step] if runner else None,lastSavedCheckpoint=last_saved,
            seconds=time.monotonic()-started)
        atomic_json(OUTPUT/'failure.json',failure);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    print(json.dumps(check() if args.check else dict(config=scope(),trainingRunsStarted=0),indent=2))


if __name__=='__main__':main()
