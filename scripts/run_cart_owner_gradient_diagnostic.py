#!/usr/bin/env python3
"""One explicitly authorized clean traversal; default is a zero-training plan."""
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
sys.path.insert(0, str(ROOT))
RESULT = ROOT/'newResults/nonIID/cart_owner_gradients/clean_one_traversal_16_v1'
PLOTS = ROOT/'newPlots/nonIID/cart_owner_gradients/clean_one_traversal_16_v1'


def fingerprints():
    from scripts.run_cart_feature_diagnostic import SOURCES
    paths = (*SOURCES, 'basil_core/cart_owner_gradients.py',
        'basil_core/cart_owner_gradient_worker.py', 'scripts/run_cart_owner_gradient_diagnostic.py',
        'scripts/report_cart_owner_gradient_diagnostic.py')
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def config(device):
    return dict(methodRevision='cart_balanced_owner_gradient_candidate_v1',
        experimentProtocol='sequential_basil_one_class_v1', researchValid=False,
        purpose='bounded_clean_owner_gradient_diagnostic', seed=2025, device=device,
        nNodes=10, nRounds=1, localEpochs=5, samplesPerNode=5000, batchSize=512,
        optimizer='sgd', momentum=0., optimizerStateMode='reset_each_activation',
        learningRate=.05, learningRateSchedule='basil_round_decay', gradientClipNorm=None,
        weightDecayCoefficient=0., modelArchitecture='basil_paper_cnn',
        aggregationMode='strict_sequential_handoff', basilMemorySize=5,
        snapshotSelectionRule='lowest_receiver_local_loss', snapshotPlausibilityGuard='none',
        actualAttackers=[], channelNoise=False, ebm=False, cartMuMax=.01,
        helperSamples=16, helperSamplesApproval='explicit operator approval before run',
        ownerResponseRefresh='every current SGD step; no stale-gradient reuse',
        classWeighting='equal mean of available owner class objectives',
        featureReplay=False, representationPreservation=False, liveTrackingEnabled=False,
        testUsage='full 10000 test images evaluation only after each activation',
        analysisUsage='same 640 TRAIN probe; not held out; no selection/control feedback',
        auxiliaryCommunication='additional clean simulated owner-gradient channel; not plain BASIL',
        rawImagesTransferred=False, physicalRPC=False, helperOptimizerSteps=0,
        queueStarted=False, automaticPromotion=False, checkpointScope='evidence; no automatic resume')


def proof_path(device):
    return ROOT/f'gui/worker_state/CART_OWNER_WORKER_{device}_VERIFICATION.json'


def verify(tf, device):
    """Synthetic full-CNN update validation; no CIFAR learning trajectory."""
    import numpy as np
    from basil_core.cart_owner_gradient_worker import OwnerGradientCartWorker
    from basil_core.research_protocol import build_model, params_hash
    from basil_core.research_audit import augment_batch
    from scripts.run_cart_feature_diagnostic import write_json
    if proof_path(device).exists(): raise ValueError('Refusing to overwrite device evidence.')
    with tf.device('/CPU:0'):
        initial = [w.numpy() for w in build_model('basil_paper_cnn', 2025).trainable_variables]
    source = np.random.default_rng(8).uniform(0, 1, (16,32,32,3)).astype(np.float32)
    source_y = np.zeros(16, np.int32)
    x = np.random.default_rng(7).uniform(0, 1, (512,32,32,3)).astype(np.float32)
    outcomes = {}
    for target in dict.fromkeys(('CPU',device)):
        with tf.device(f'/{target}:0'):
            worker = OwnerGradientCartWorker(build_model('basil_paper_cnn',2025),
                build_model('basil_paper_cnn',2025), device=target)
            worker.load(initial)
            for a,p in zip(worker.reference,initial): a.assign(p+.001)
            worker.anchor_mu.assign(.007)
            worker.source_x, worker.source_y = source, source_y
            worker.owner_pools = [np.arange(16)]*10
            worker.older_classes, worker.receiver = (0,), 1
            worker.seed, worker.round_id = 2025, 0
            results = {}
            for size in (512,392):
                worker.owner_records = []
                a,v = worker._gradients(tf.constant(x[:size]),tf.ones(size,tf.int32),tf.constant([7,8]),
                    'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(1.))
                record = worker.owner_records[0]
                owner = record['owners'][0]
                with tf.device('/CPU:0'):
                    local_x = augment_batch(x[:size],tf.constant([7,8]))
                    old_x = augment_batch(source[owner['trainingIndices']],owner['augmentationKey'])
                # Independent objective on the ACTIVE model, not the helper API.
                with tf.GradientTape() as tape:
                    ce_current = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.ones(size,tf.int32),logits=worker.model(local_x)))
                    ce_old = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.zeros(16,tf.int32),logits=worker.model(old_x)))
                    penalty = worker.anchor_mu/2*tf.add_n([tf.reduce_sum((w-r)**2)
                        for w,r in zip(worker.model.trainable_variables,worker.reference)])
                    objective = (ce_current+ce_old)/2+penalty
                reference = tape.gradient(objective,worker.model.trainable_variables)
                for g,r in zip(a,reference): np.testing.assert_allclose(g.numpy(),r.numpy(),rtol=1e-3,atol=2e-6)
                np.testing.assert_allclose(float(v[2]),float(objective),rtol=1e-6)
                worker.owner_records = []
                b,u = worker._gradients(tf.constant(x[:size]),tf.ones(size,tf.int32),tf.constant([7,8]),
                    'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(999999.))
                for g,h in zip(a,b): np.testing.assert_array_equal(g.numpy(),h.numpy())
                np.testing.assert_array_equal([float(z) for z in v],[float(z) for z in u])
                before_helper = params_hash([w.numpy() for w in worker.helper_model.trainable_variables])
                assert before_helper == params_hash(initial)
                tf.keras.optimizers.SGD(.05).apply_gradients(zip(a,worker.model.trainable_variables))
                results[size] = ([g.numpy() for g in a],worker.export())
                assert params_hash([w.numpy() for w in worker.helper_model.trainable_variables])==before_helper
                worker.load(initial)
            outcomes[target] = results
    comparisons = {}
    for size in (512,392):
        ga,wa = outcomes['CPU'][size]; gb,wb = outcomes[device][size]
        relative = np.linalg.norm(np.concatenate([(a-b).ravel() for a,b in zip(ga,gb)]))/max(
            np.linalg.norm(np.concatenate([g.ravel() for g in ga])),1e-12)
        maximum = max(float(np.max(np.abs(a-b))) for a,b in zip(wa,wb))
        if relative>.01 or maximum>1e-4: raise ValueError('CPU/GPU numerical tolerance exceeded.')
        comparisons[str(size)] = dict(relativeGradientDifference=float(relative),maximumUpdateDifference=maximum)
    proof = dict(status='passed',device=device,sourceHashes=fingerprints(),parameterCount=sum(p.size for p in initial),
        comparisons=comparisons,lambdaIgnored=True,helperUpdates=0,
        independentObjective='(CE_current + CE_old)/2 + mu/2 * snapshot_drift_squared',
        tolerance=dict(relativeGradient=.01,maximumUpdate=1e-4),researchRunsStarted=0)
    write_json(proof_path(device),proof)
    print(json.dumps(proof),flush=True)


def execute(tf, device):
    import numpy as np
    from basil_core.cart_owner_gradient_worker import OwnerGradientCartWorker
    from basil_core.class_registry import ClassRegistry
    from basil_core.research_protocol import build_model, initialize_memories, keyed_rng, select_snapshot, params_hash, Snapshot, broadcast
    from basil_core.research_audit import ProtocolAudit
    from basil_core.forgetting_audit import classification_metrics
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    from basil_core.iid_runtime import runtime_metadata
    from scripts.run_cart_representation_diagnostic import load_training
    from scripts.run_cart_feature_diagnostic import write_json
    from scripts.run_strict_cart_diagnostic import expected_candidates
    proof = json.loads(proof_path(device).read_text())
    if proof['status']!='passed' or proof['sourceHashes']!=fingerprints():
        raise ValueError('Matching current-source device proof required.')
    if RESULT.exists() or PLOTS.exists(): raise ValueError('Refusing evidence overwrite/retry/resume.')
    check_host_memory()
    images,labels = load_training()
    cache = ROOT/'experiments/cache/cifar10'
    test_x = np.load(cache/'x_test_float32.npy',mmap_mode='r',allow_pickle=False)
    test_y = np.load(cache/'y_test_int32.npy',mmap_mode='r',allow_pickle=False)
    if test_x.shape!=(10000,32,32,3) or not np.array_equal(np.bincount(test_y,minlength=10),[1000]*10):
        raise ValueError('Complete balanced evaluation-only test cache required.')
    assignments = [np.flatnonzero(labels==i) for i in range(10)]
    selections = [keyed_rng(2025,'selection_batch',i).permutation(c)[:512] for i,c in enumerate(assignments)]
    probes, pools = [],[]
    for i,c in enumerate(assignments):
        pool = np.setdiff1d(c,selections[i])
        probe = keyed_rng(2025,'cart_feature_analysis_probe',i).permutation(pool)[:64]
        probes.extend(map(int,probe)); pools.append(np.setdiff1d(pool,probe))
    probe_ids = np.asarray(probes,np.int64)
    with tf.device('/CPU:0'):
        initial = [w.numpy() for w in build_model('basil_paper_cnn',2025).trainable_variables]
    digest = hashlib.sha256()
    for c in assignments: digest.update(np.asarray(c,dtype='<i8').tobytes())
    manifest = dict(status='running',config=config(device),sourceHashes=fingerprints(),
        **runtime_metadata(tf,{'executionDevice':device}),initialModelHash=params_hash(initial),
        partitionHash=digest.hexdigest(),parameterCount=sum(p.size for p in initial),
        activationsCompleted=0,optimizerSteps=0,helperOptimizerSteps=0,wandbEnabled=False)
    RESULT.mkdir(parents=True)
    write_json(RESULT/'config.json',config(device)); write_json(RESULT/'run.json',manifest)
    write_json(RESULT/'partition_audit.json',dict(samplesPerNode=[len(c) for c in assignments],
        classes=list(range(10)),coverage=50000,testSamples=10000,testClassCounts=[1000]*10,
        analysisProbe=probes,ownerResponsesExcludeSSAndAnalysisProbe=True,
        analysisProbeNotHeldOutFromOrdinarySGD=True,partitionHash=digest.hexdigest()))
    stop = False
    def request_stop(_signal,_frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    class OwnerAudit(ProtocolAudit):
        def failure(self, worker, error):
            # The inherited graph replay represents ordinary anchored CE, not
            # this Python owner service. Never replay the wrong objective.
            if self.pending is None: return
            from basil_core.research_audit import tensor_stats
            self.write('failure.json', dict(self.current, exception=str(error),
                replayPerformed=False, reason='owner service requires exact per-owner step context',
                parameterStatsAtFailure=[tensor_stats(p) for p in worker.export()]))
            np.savez_compressed(self.directory/'failure_batch.npz',x=self.pending['x'],
                y=self.pending['y'],key=self.pending['key'],
                **{f'weight_{i}':p for i,p in enumerate(self.pending['params'])})
    audit = OwnerAudit(RESULT/'audit'); guard = RuntimeMemoryGuard(tf,device)
    started = time.perf_counter(); context = {}; records = []; final_weights = {}
    try:
        with tf.device(f'/{device}:0'):
            worker = OwnerGradientCartWorker(build_model('basil_paper_cnn',2025),
                build_model('basil_paper_cnn',2025),device=device)
            worker.audit = audit
            memories = initialize_memories(initial); registries = [ClassRegistry() for _ in range(10)]
            def metrics(x,y):
                logits = []
                for start in range(0,len(y),512):
                    try:
                        z = worker.predict(tf.convert_to_tensor(x[start:start+512]))
                        tf.debugging.assert_all_finite(z,'Non-finite evaluation-only logits')
                    except Exception as error:
                        audit.evaluation_failure(worker,x[start:start+512],y[start:start+512],start,error)
                        raise
                    logits.append(z.numpy())
                return classification_metrics(np.concatenate(logits),np.asarray(y))
            def evaluate():
                values = metrics(images[probe_ids],labels[probe_ids])
                audit.last_confusion = np.asarray(values['confusion']); audit.evaluation()
                return dict(trainingAnalysis=values)
            for node in range(10):
                context = dict(round=0,node=node); audit.activation(0,node)
                candidates = memories[node].candidates()
                if [(s.sender_id,s.round_id) for s in candidates]!=expected_candidates(0,node):
                    raise ValueError('Rolling predecessor contract failed.')
                ss = selections[node]
                selected,diagnostics = select_snapshot(node,candidates,
                    lambda p: worker.probe(p,images[ss],labels[ss]))
                worker.load(selected.params)
                if params_hash(worker.export())!=params_hash(selected.params): raise ValueError('Strict handoff failed.')
                loss = next(d['receiverLocalLoss'] for d in diagnostics
                    if (d['senderId'],d['snapshotRound'])==(selected.sender_id,selected.round_id))
                params,record = worker.train(selected.params,images[assignments[node]],labels[assignments[node]],
                    source_x=images,source_y=labels,owner_pools=pools,older_classes=range(node),
                    seed=2025,round_id=0,node_id=node,epochs=5,batch_size=512,learning_rate=.05,
                    ebm_mode='none',semantics='paper_absolute_gaussian',sigma=0.,
                    local_registry=registries[node],received_registry=ClassRegistry.fromDict(selected.retention_state)
                    if selected.retention_state else ClassRegistry(),selection_x=images[ss],selection_y=labels[ss],
                    selected_loss=loss,evaluation=evaluate,should_stop=lambda:stop)
                if record['steps']!=50 or any(e['samplesSeen']!=5000 or e['batchSizes']!=[512]*9+[392] for e in record['epochs']):
                    raise ValueError('Five-full-epoch contract failed.')
                record.update(round=0,node=node,inputHash=params_hash(selected.params),outputHash=params_hash(params),
                    selectedSender=selected.sender_id,selectedSnapshotRound=selected.round_id,selection=diagnostics,
                    globalTestEvaluationOnly=metrics(test_x,test_y))
                if params_hash(worker.export())!=params_hash(params): raise ValueError('Evaluation altered parameters.')
                records.append(record)
                broadcast(memories,Snapshot(node,0,params,retention_state=record['outputRegistry']))
                for j,p in enumerate(params): final_weights[f'node{node}_w{j}']=p
                write_json(RESULT/'activations.json',records)
                np.savez_compressed(RESULT/'latest_nodes.npz',**final_weights)
                manifest.update(activationsCompleted=node+1,optimizerSteps=(node+1)*50,
                    context=context,memorySafety=guard.checkpoint())
                write_json(RESULT/'run.json',manifest)
                print(json.dumps(dict(nodeCompleted=node,trainingProbeAccuracy=record['epochs'][-1]['trainingAnalysis']['accuracy'],
                    globalTestAccuracy=record['globalTestEvaluationOnly']['accuracy'],
                    elapsedSeconds=time.perf_counter()-started)),flush=True)
        manifest.update(status='completed',runtimeSeconds=time.perf_counter()-started,
            numericalFailures=0,oomEvents=0,memorySafety=guard.checkpoint())
    except Exception as error:
        manifest.update(status='interrupted' if isinstance(error,InterruptedError) else 'failed',
            error=str(error),context=context,runtimeSeconds=time.perf_counter()-started)
        write_json(RESULT/'run.json',manifest)
        raise
    finally: audit.close()
    write_json(RESULT/'run.json',manifest)
    print(json.dumps(dict(status='completed',result=str(RESULT),runtimeSeconds=manifest['runtimeSeconds'])),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['CPU','GPU'],default='GPU')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify-device',action='store_true')
    action.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if not args.execute and not args.verify_device:
        print(json.dumps(dict(config=config(args.device),trainingRunsStarted=0),indent=2)); return
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled',WANDB_MODE='disabled')
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf,args.device)
    if args.verify_device: verify(tf,args.device)
    else: execute(tf,args.device)


if __name__=='__main__': main()
