#!/usr/bin/env python3
"""Paired three-round owner-gradient CART diagnostic; default never trains."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RESULT = ROOT/'newResults/nonIID/cart_owner_gradients/clean_three_round_selection_pair_v1'
PLOTS = ROOT/'newPlots/nonIID/cart_owner_gradients/clean_three_round_selection_pair_v1'
CONTROLS = ('basil_ss','immediate_predecessor')


def fingerprints():
    from scripts.run_cart_owner_gradient_diagnostic import fingerprints as prior
    return dict(prior(), **{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ('scripts/run_cart_owner_selection_diagnostic.py','scripts/report_cart_owner_selection_diagnostic.py')})


def config(device, control):
    from scripts.run_cart_owner_gradient_diagnostic import config as prior
    if control not in CONTROLS: raise ValueError('Unknown selection control.')
    cfg=prior(device)
    cfg.update(condition=control,nRounds=3,purpose='bounded_clean_snapshot_selection_ablation',
        snapshotSelection=control=='basil_ss',
        startingModelRule='lowest_receiver_local_loss' if control=='basil_ss' else 'immediate_counterclockwise_predecessor',
        ownerAvailability='already visited owners; all other nine owners after the first traversal',
        selectionProbeUsage='all candidates evaluated on identical own TRAIN batch; controls differ only in chosen source',
        snapshotAnchorRule='unchanged CART receiver-supported confidence/gap; selected actual input CE used in both',
        resumePolicy='fresh canonical starts; evidence checkpoints only, no automatic resume')
    return cfg


def verify_configs(device):
    a,b=(config(device,c) for c in CONTROLS)
    changed={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    if changed!={'condition','snapshotSelection','startingModelRule'}:
        raise ValueError('Unintended control configuration difference.')
    return sorted(changed)


def choose_start(receiver, candidates, evaluator, *, snapshot_selection):
    from basil_core.research_protocol import select_snapshot
    minimum,diagnostics=select_snapshot(receiver,candidates,evaluator)
    if snapshot_selection:
        selected=minimum
    else:
        options=[s for s in candidates if s.sender_id==(receiver-1)%10]
        if len(options)!=1: raise ValueError('Exactly one immediate predecessor required.')
        selected=options[0]
    loss=next(d['receiverLocalLoss'] for d in diagnostics
        if (d['senderId'],d['snapshotRound'])==(selected.sender_id,selected.round_id))
    return selected,diagnostics,loss,dict(minimumLossSender=minimum.sender_id,
        minimumLossSnapshotRound=minimum.round_id,selectionApplied=snapshot_selection)


def proof_path(device):
    return ROOT/f'gui/worker_state/CART_OWNER_SELECTION_3R_{device}_VERIFICATION.json'


def verify(tf,device, *, study=None):
    """Independent all-ten-class gradient/SGD check, synthetic inputs only."""
    import numpy as np
    from basil_core.cart_owner_gradient_worker import OwnerGradientCartWorker
    from basil_core.research_protocol import build_model,params_hash
    from basil_core.research_audit import augment_batch
    from scripts.run_cart_feature_diagnostic import write_json
    # Explicit opt-in study hooks avoid runtime monkey-patching and preserve the
    # original diagnostic defaults. Each revision has its own source proof.
    proof_target = study.proof_path if study else proof_path
    hashes = study.fingerprints if study else fingerprints
    validate = study.verify_configs if study else verify_configs
    factory = study.create_worker if study else OwnerGradientCartWorker
    if proof_target(device).exists(): raise ValueError('Refusing proof overwrite.')
    with tf.device('/CPU:0'):
        initial=[w.numpy() for w in build_model('basil_paper_cnn',2025).trainable_variables]
    source=np.random.default_rng(8).uniform(0,1,(160,32,32,3)).astype(np.float32)
    labels=np.repeat(np.arange(10,dtype=np.int32),16)
    pools=[np.flatnonzero(labels==c) for c in range(10)]
    x=np.random.default_rng(7).uniform(0,1,(512,32,32,3)).astype(np.float32)
    outcomes={}
    for target in dict.fromkeys(('CPU',device)):
        with tf.device(f'/{target}:0'):
            worker=factory(build_model('basil_paper_cnn',2025),
                build_model('basil_paper_cnn',2025),device=target,diagnostic_rounds=getattr(study, 'ROUNDS', 3))
            worker.source_x,worker.source_y,worker.owner_pools=source,labels,pools
            worker.older_classes=tuple(range(1,10)); worker.receiver=0
            worker.seed,worker.round_id=2025,1; worker.anchor_mu.assign(.007)
            for r,p in zip(worker.reference,initial): r.assign(p+.001)
            results={}
            for size in (512,392):
                worker.load(initial); worker.owner_records=[]
                actual,values=worker._gradients(tf.constant(x[:size]),tf.zeros(size,tf.int32),tf.constant([7,8]),
                    'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(1.))
                trace=worker.owner_records[0]
                with tf.device('/CPU:0'):
                    batches=[(augment_batch(x[:size],[7,8]),tf.zeros(size,tf.int32))]
                    for owner in trace['owners']:
                        ids=owner['trainingIndices']
                        batches.append((augment_batch(source[ids],owner['augmentationKey']),tf.constant(labels[ids])))
                with tf.GradientTape() as tape:
                    terms=[tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                        logits=worker.model(images,training=True))) for images,y in batches]
                    loss=tf.add_n(terms)/10 + worker.anchor_mu/2*tf.add_n([
                        tf.reduce_sum((w-r)**2) for w,r in zip(worker.model.trainable_variables,worker.reference)])
                expected=tape.gradient(loss,worker.model.trainable_variables)
                for a,b in zip(actual,expected): np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=1e-3,atol=2e-6)
                np.testing.assert_allclose(float(values[2]),float(loss),rtol=1e-6)
                assert trace['classWeights']=={str(c):.1 for c in range(10)}
                assert trace['ownerOptimizerSteps']==0 and trace['extraOwnerResponses']==9
                helper_before=params_hash([w.numpy() for w in worker.helper_model.trainable_variables])
                worker.owner_records=[]
                repeated,other=worker._gradients(tf.constant(x[:size]),tf.zeros(size,tf.int32),tf.constant([7,8]),
                    'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(999999.))
                for a,b in zip(actual,repeated): np.testing.assert_array_equal(a.numpy(),b.numpy())
                np.testing.assert_array_equal([float(v) for v in values],[float(v) for v in other])
                tf.keras.optimizers.SGD(.05).apply_gradients(zip(actual,worker.model.trainable_variables))
                assert helper_before==params_hash([w.numpy() for w in worker.helper_model.trainable_variables])
                results[size]=([g.numpy() for g in actual],worker.export())
            outcomes[target]=results
    comparisons={}
    for size in (512,392):
        ga,wa=outcomes['CPU'][size]; gb,wb=outcomes[device][size]
        relative=float(np.linalg.norm(np.concatenate([(a-b).ravel() for a,b in zip(ga,gb)]))/max(
            np.linalg.norm(np.concatenate([g.ravel() for g in ga])),1e-12))
        maximum=max(float(np.max(np.abs(a-b))) for a,b in zip(wa,wb))
        if relative>.01 or maximum>1e-4: raise ValueError('CPU/GPU tolerance failed.')
        comparisons[str(size)]=dict(relativeGradientDifference=relative,maximumUpdateDifference=maximum)
    proof=dict(status='passed',device=device,sourceHashes=hashes(),comparisons=comparisons,
        batchSizes=[512,392],ownerSamples=16,availableClasses=list(range(10)),
        independentObjective='mean of ten class CEs plus frozen CART anchor',
        legacyLambdaIgnored=True,helperOptimizerSteps=0,parameterCount=sum(p.size for p in initial),
        tolerance=dict(relativeGradient=.01,maximumUpdate=1e-4),configurationDifferences=validate(device))
    write_json(proof_target(device),proof); print(json.dumps(proof),flush=True)


def execute(tf,device, *, study=None):
    import numpy as np
    from basil_core.cart_owner_gradient_worker import OwnerGradientCartWorker
    from basil_core.class_registry import ClassRegistry
    from basil_core.research_protocol import build_model,initialize_memories,keyed_rng,params_hash,Snapshot,broadcast,basil_round_learning_rate
    from basil_core.research_audit import ProtocolAudit,tensor_stats
    from basil_core.forgetting_audit import classification_metrics
    from basil_core.memory_safety import RuntimeMemoryGuard,check_host_memory
    from basil_core.iid_runtime import runtime_metadata
    from scripts.run_cart_representation_diagnostic import load_training
    from scripts.run_cart_feature_diagnostic import write_json
    from scripts.run_strict_cart_diagnostic import expected_candidates
    # All hooks are explicit. Saved source fingerprints cover the runner and
    # study implementation; no scientific defaults are silently replaced.
    RESULT = study.RESULT if study else globals()['RESULT']
    PLOTS = study.PLOTS if study else globals()['PLOTS']
    CONTROLS = study.CONTROLS if study else globals()['CONTROLS']
    config = study.config if study else globals()['config']
    fingerprints = study.fingerprints if study else globals()['fingerprints']
    proof_path = study.proof_path if study else globals()['proof_path']
    verify_configs = study.verify_configs if study else globals()['verify_configs']
    factory = study.create_worker if study else OwnerGradientCartWorker
    rounds = getattr(study, 'ROUNDS', 3)
    stream = getattr(study, 'STREAM_ACTIVATIONS', False)
    if type(rounds) is not int or rounds not in (3, 30):
        raise ValueError('Only explicitly approved three- or thirty-round comparisons are allowed.')
    if any(config(device, c)['nRounds'] != rounds for c in CONTROLS):
        raise ValueError('Study horizon and executed configuration differ.')
    proof=json.loads(proof_path(device).read_text())
    if proof['status']!='passed' or proof['sourceHashes']!=fingerprints(): raise ValueError('Current-source device proof required.')
    differences=verify_configs(device)
    if RESULT.exists() or PLOTS.exists(): raise ValueError('Refusing result overwrite/retry/resume.')
    check_host_memory(); images,labels=load_training()
    cache=ROOT/'experiments/cache/cifar10'
    test_x=np.load(cache/'x_test_float32.npy',mmap_mode='r',allow_pickle=False)
    test_y=np.load(cache/'y_test_int32.npy',mmap_mode='r',allow_pickle=False)
    if test_x.shape!=(10000,32,32,3) or not np.array_equal(np.bincount(test_y,minlength=10),[1000]*10):
        raise ValueError('Complete balanced evaluation-only test cache required.')
    assignments=[np.flatnonzero(labels==i) for i in range(10)]
    selections=[keyed_rng(2025,'selection_batch',i).permutation(c)[:512] for i,c in enumerate(assignments)]
    probes=[]; pools=[]
    for i,c in enumerate(assignments):
        pool=np.setdiff1d(c,selections[i]); probe=keyed_rng(2025,'cart_feature_analysis_probe',i).permutation(pool)[:64]
        probes.extend(map(int,probe)); pools.append(np.setdiff1d(pool,probe))
    probe_ids=np.asarray(probes,np.int64)
    with tf.device('/CPU:0'):
        initial=[w.numpy() for w in build_model('basil_paper_cnn',2025).trainable_variables]
    digest=hashlib.sha256()
    for c in assignments: digest.update(np.asarray(c,dtype='<i8').tobytes())
    runtime=runtime_metadata(tf,{'executionDevice':device})
    manifest=dict(status='running',**runtime,sourceHashes=fingerprints(),initialModelHash=params_hash(initial),
        partitionHash=digest.hexdigest(),parameterCount=sum(p.size for p in initial),
        conditions=list(CONTROLS),configurationDifferences=differences,activationsCompleted=0,
        optimizerSteps=0,helperOptimizerSteps=0,controlsCompleted=[],
        roundsCompletedByControl={c:0 for c in CONTROLS},wandbEnabled=False,
        configs={c:config(device,c) for c in CONTROLS})
    RESULT.mkdir(parents=True)
    write_json(RESULT/'run.json',manifest)
    write_json(RESULT/'partition_audit.json',dict(samplesPerNode=[len(c) for c in assignments],
        nodeClasses=list(range(10)),coverage=50000,testSamples=10000,testClassCounts=[1000]*10,
        analysisProbe=probes,ownerResponsesExcludeSSAndAnalysisProbe=True,
        analysisProbeNotHeldOutFromOrdinarySGD=True,partitionHash=digest.hexdigest()))
    stop=False
    def request_stop(_signal,_frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    class OwnerAudit(ProtocolAudit):
        def failure(self,worker,error):
            if self.pending is None:return
            self.write('failure.json',dict(self.current,exception=str(error),replayPerformed=False,
                parameterStatsAtFailure=[tensor_stats(p) for p in worker.export()]))
            np.savez_compressed(self.directory/'failure_batch.npz',x=self.pending['x'],y=self.pending['y'],
                key=self.pending['key'],**{f'weight_{i}':p for i,p in enumerate(self.pending['params'])})
    guard=RuntimeMemoryGuard(tf,device); started=time.perf_counter(); context={}
    try:
        for control in CONTROLS:
            directory=RESULT/control; directory.mkdir(); cfg=config(device,control)
            write_json(directory/'config.json',cfg)
            state=dict(status='running',config=cfg,**runtime,sourceHashes=fingerprints(),
                partitionHash=manifest['partitionHash'],initialModelHash=params_hash(initial),
                activationsCompleted=0,optimizerSteps=0,roundsCompleted=0)
            write_json(directory/'run.json',state); control_started=time.perf_counter()
            audit=OwnerAudit(directory/'audit')
            try:
                with tf.device(f'/{device}:0'):
                    worker=factory(build_model('basil_paper_cnn',2025),
                        build_model('basil_paper_cnn',2025),device=device,diagnostic_rounds=rounds)
                    worker.audit=audit; memories=initialize_memories(initial)
                    registries=[ClassRegistry() for _ in range(10)]; visited=set(); records=[]; final_weights={}
                    def metrics(x,y):
                        logits=[]
                        for start in range(0,len(y),512):
                            try:
                                z=worker.predict(tf.convert_to_tensor(x[start:start+512]))
                                tf.debugging.assert_all_finite(z,'Non-finite evaluation-only logits')
                            except Exception as error:
                                audit.evaluation_failure(worker,x[start:start+512],y[start:start+512],start,error);raise
                            logits.append(z.numpy())
                        return classification_metrics(np.concatenate(logits),np.asarray(y))
                    def evaluate():
                        values=metrics(images[probe_ids],labels[probe_ids])
                        audit.last_confusion=np.asarray(values['confusion']);audit.evaluation()
                        return dict(trainingAnalysis=values)
                    for activation in range(rounds * 10):
                        round_id,node=divmod(activation,10)
                        context=dict(condition=control,round=round_id,node=node);audit.activation(round_id,node)
                        candidates=memories[node].candidates()
                        if [(s.sender_id,s.round_id) for s in candidates]!=expected_candidates(round_id,node):
                            raise ValueError('Rolling predecessor contract differs.')
                        ss=selections[node]
                        if study:
                            selected,diagnostics,loss,decision=study.choose_start(node,candidates,
                                worker=worker,images=images,labels=labels,selections=selections,
                                available_owners=sorted(visited|{node}),control=control)
                        else:
                            selected,diagnostics,loss,decision=choose_start(node,candidates,
                                lambda p:worker.probe(p,images[ss],labels[ss]),snapshot_selection=cfg['snapshotSelection'])
                        worker.load(selected.params); input_hash=params_hash(worker.export())
                        if input_hash!=params_hash(selected.params): raise ValueError('Strict handoff differs.')
                        previous_hash=records[-1]['outputHash'] if records else params_hash(initial)
                        if not cfg['snapshotSelection'] and input_hash!=previous_hash:
                            raise ValueError('Immediate-predecessor exact handoff differs.')
                        owners=sorted(visited-{node})
                        params,record=worker.train(selected.params,images[assignments[node]],labels[assignments[node]],
                            source_x=images,source_y=labels,owner_pools=pools,older_classes=owners,
                            seed=2025,round_id=round_id,node_id=node,epochs=5,batch_size=512,
                            learning_rate=basil_round_learning_rate(.05,round_id),ebm_mode='none',
                            semantics='paper_absolute_gaussian',sigma=0.,local_registry=registries[node],
                            received_registry=ClassRegistry.fromDict(selected.retention_state) if selected.retention_state else ClassRegistry(),
                            selection_x=images[ss],selection_y=labels[ss],selected_loss=loss,
                            evaluation=evaluate,should_stop=lambda:stop)
                        if record['steps']!=50 or any(e['samplesSeen']!=5000 or e['batchSizes']!=[512]*9+[392] for e in record['epochs']):
                            raise ValueError('Five-full-epoch contract failed.')
                        record.update(round=round_id,node=node,condition=control,inputHash=input_hash,
                            outputHash=params_hash(params),selectedSender=selected.sender_id,
                            selectedSnapshotRound=selected.round_id,selection=diagnostics,selectionDecision=decision,
                            learningRate=basil_round_learning_rate(.05,round_id),
                            globalTestEvaluationOnly=metrics(test_x,test_y))
                        if params_hash(worker.export())!=params_hash(params):raise ValueError('Evaluation altered weights.')
                        records.append(record);visited.add(node)
                        broadcast(memories,Snapshot(node,round_id,params,retention_state=record['outputRegistry']))
                        for j,p in enumerate(params):final_weights[f'node{node}_w{j}']=p
                        if stream:
                            # One immutable activation per file: bounded live
                            # history and linear I/O, without changing updates.
                            target = directory/'activations'/f'{activation:04d}.json'
                            if target.exists():
                                raise ValueError('Refusing activation evidence overwrite.')
                            write_json(target, record)
                            records[:] = records[-1:]
                        else:
                            write_json(directory/'activations.json',records)
                        # Atomic evidence checkpoint includes rolling cache, not
                        # just final weights. No unverified resume is offered.
                        payload=dict(final_weights); metadata=[]
                        for receiver,memory in enumerate(memories):
                            for snapshot in memory.candidates():
                                key=f'r{receiver}s{snapshot.sender_id}'
                                payload.update({f'{key}w{j}':p for j,p in enumerate(snapshot.params)})
                                metadata.append(dict(receiver=receiver,sender=snapshot.sender_id,
                                    round=snapshot.round_id,hash=params_hash(snapshot.params),registry=snapshot.retention_state))
                        payload['metadata']=json.dumps(dict(nextActivation=activation+1,snapshots=metadata,
                            registries=[r.toDict() for r in registries],visitedClasses=sorted(visited),
                            config=cfg,sourceHashes=fingerprints(),initialModelHash=params_hash(initial),runtime=runtime))
                        temporary=directory/'checkpoint.tmp.npz';np.savez_compressed(temporary,**payload)
                        temporary.replace(directory/'checkpoint.npz');del payload,metadata
                        state.update(activationsCompleted=activation+1,optimizerSteps=(activation+1)*50,
                            roundsCompleted=(activation+1)//10,memorySafety=guard.checkpoint())
                        write_json(directory/'run.json',state)
                        manifest['activationsCompleted']+=1;manifest['optimizerSteps']+=50
                        manifest['roundsCompletedByControl'][control]=(activation+1)//10
                        manifest['context']=context;write_json(RESULT/'run.json',manifest)
                        print(json.dumps(dict(condition=control,round=round_id,nodeCompleted=node,
                            globalTestAccuracy=record['globalTestEvaluationOnly']['accuracy'],
                            elapsedSeconds=time.perf_counter()-started)),flush=True)
                    np.savez_compressed(directory/'final_nodes.npz',**final_weights)
                    state.update(status='completed',runtimeSeconds=time.perf_counter()-control_started,
                        numericalFailures=0,oomEvents=0,memorySafety=guard.checkpoint())
                    write_json(directory/'run.json',state);manifest['controlsCompleted'].append(control)
                    del worker,memories,registries,records,final_weights
            except Exception as error:
                state.update(status='interrupted' if isinstance(error,InterruptedError) else 'failed',
                    error=str(error),context=context,runtimeSeconds=time.perf_counter()-control_started)
                write_json(directory/'run.json',state);raise
            finally:audit.close()
        manifest.update(status='completed',runtimeSeconds=time.perf_counter()-started,
            numericalFailures=0,oomEvents=0,memorySafety=guard.checkpoint())
    except Exception as error:
        manifest.update(status='interrupted' if isinstance(error,InterruptedError) else 'failed',
            error=str(error),context=context,runtimeSeconds=time.perf_counter()-started)
        write_json(RESULT/'run.json',manifest);raise
    write_json(RESULT/'run.json',manifest)
    print(json.dumps(dict(status='completed',result=str(RESULT),runtimeSeconds=manifest['runtimeSeconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['CPU','GPU'],default='GPU')
    action=parser.add_mutually_exclusive_group();action.add_argument('--verify-device',action='store_true');action.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute and not args.verify_device:
        print(json.dumps(dict(configs=[config(args.device,c) for c in CONTROLS],
            differences=verify_configs(args.device),trainingRunsStarted=0),indent=2));return
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled',WANDB_MODE='disabled')
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf,args.device)
    if args.verify_device:verify(tf,args.device)
    else:execute(tf,args.device)


if __name__=='__main__':main()
