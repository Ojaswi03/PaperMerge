#!/usr/bin/env python3
"""Serial, resumable GUI worker for the fixed verified-CART non-IID study.

Without --execute this validates configuration and recovery and performs zero SGD.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.noniid-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, separators=(',', ':'), allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(name): os.unlink(name)


def commit(runner, state, directory):
    """Two atomic slots: progress points only to a completely written checkpoint."""
    generation = state.get('generation', -1)+1
    path = directory/f'checkpoint_{generation % 2}.json'
    runner.save(path)
    payload=dict(state,generation=generation, checkpoint=path.name,
        checkpointSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cursor=[runner.round, runner.node, runner.step])
    atomic_json(directory/'progress.json', payload)
    state.update(payload)
    print(json.dumps(dict(event='checkpoint_saved',activation=runner.round*10+runner.node,
        completedRounds=runner.round,checkpoint=str(path))),flush=True)


def helper_faults(config, round_id):
    return {int(k): v for k, v in config['helperFaults'].items()} if round_id >= 20 else {}


def selected_telemetry(config, receiver, round_id, selection):
    attackers = config['resolvedAttackerIds']
    return dict(receiver=receiver, round=round_id, selectedSender=selection['sender'],
        selectedSnapshotRound=selection['round'], selectedHash=selection['hash'],
        selectedAttackerDesignated=selection['sender'] in attackers,
        selectedActuallyCorrupted=selection['sender'] in attackers and selection['round'] >= 20,
        snapshotAge=round_id-selection['round'], candidates=selection['records'])


def publish(directory, state, runtime):
    rounds = state['roundMetrics']
    last = rounds[-1] if rounds else {}
    atomic_json(directory/'run.json', dict(config=state['config'], status=state['status'],
        researchValid=False, completedRounds=len(rounds), completedActivations=state['cursor'][0]*10+state['cursor'][1],
        currentCursor=state['cursor'], partitionHash=state['config']['partitionHash'],
        initialModelHash=state['config']['initialModelHash'], sourceHashes=state['sourceHashes'],
        device=runtime['device'], runtime=runtime, elapsedSeconds=state['elapsedSeconds'],
        finalAverageAccuracy=last.get('mean_node_accuracy'), finalWorstAccuracy=last.get('worst_node_accuracy'),
        finalFullTestAccuracy=last.get('full_test_accuracy'), roundMetrics=rounds,
        metricDefinitions=dict(full_test_accuracy='Node 9 honest trained model before outbound corruption; full TEST',
            mean_node_accuracy='Mean full-TEST accuracy of ten stored honest trained node states',
            worst_node_accuracy='Minimum full-TEST accuracy of ten stored honest trained node states',
            full_test_loss='Node 9 full-TEST cross-entropy'),
        threatQualification='Trusted coordinator and replicated TRAIN; certified helper computation; snapshots may still be attack-corrupted',
        exactStateTransfer=True, bitIdenticalCpuContinuation=False,
        artifactPaths=dict(metrics=str(directory/'metrics.npz'))))
    import csv
    with (directory/'round_metrics.csv').open('w', newline='') as stream:
        names = ['round','full_test_accuracy','mean_node_accuracy','worst_node_accuracy','full_test_loss',
                 *[f'class_{i}_accuracy' for i in range(10)], 'corrupted_selection_rate',
                 'attacker_designated_selection_rate','selected_snapshot_age','balanced_training_ce',
                 'ebm_penalty','ebm_correction_norm','ebm_correction_ratio','channel_noise_norm','noise_to_model_norm_ratio']
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rounds)
    import numpy as np
    np.savez_compressed(directory/'metrics.npz', fullTestAccuracy=np.asarray([r['full_test_accuracy'] for r in rounds]),
        averageAccuracy=np.asarray([r['mean_node_accuracy'] for r in rounds]),
        worstNodeAccuracy=np.asarray([r['worst_node_accuracy'] for r in rounds]),
        fullTestLoss=np.asarray([r['full_test_loss'] for r in rounds]),
        fullTestPerClassAccuracy=np.asarray([[r[f'class_{i}_accuracy'] for i in range(10)] for r in rounds]),
        perNodeAccuracy=np.asarray([r['per_node_accuracy'] for r in rounds]))


def execute(config, *, validate_only=False):
    from basil_core.noniid_verified_study import recovery_plan, source_hashes, validate_configuration, require_transport_proof
    validate_configuration(config)
    from basil_core.execution_profile import profile_path
    directory = profile_path(ROOT/config['resultDirectory'])
    plan = recovery_plan(directory, config)
    if plan['mode'] == 'completed':
        raise ValueError('This condition already completed; no extra training authorized.')
    from scripts.run_verified_cart_learning import require_proof, evaluate
    proof_hash = require_proof()
    transport_proof_hash=require_transport_proof()
    gate_path = profile_path(ROOT/'gui/worker_state/cart_full_batch_threat_v1/summary.json')
    gate = json.loads(gate_path.read_text())
    from basil_core.verified_cart_runner import fingerprints
    if (gate.get('status') != 'passed' or gate.get('sourceHashes') != fingerprints() or
            not gate.get('allFiftyUpdatesExactlyMatch') or not gate.get('incorrectProposalRecovered')):
        raise ValueError('Current full-batch threat/EBM gate required.')
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled')
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow, runtime_metadata, check_recovery_device
    prepare_environment('GPU')
    import tensorflow as tf
    import numpy as np
    configure_tensorflow(tf, 'GPU')
    runtime = runtime_metadata(tf, {'executionDevice':'GPU'})
    print(json.dumps(dict(event='device_ready', **runtime)), flush=True)
    from basil_core.research_protocol import build_model, params_hash, keyed_rng
    from basil_core.replicated_training_service import TrainingReference
    from basil_core.isolated_replicas import IsolatedReplicaPool
    from basil_core.parallel_verifier_service import ParallelVerifierService
    from basil_core.verified_cart_runner import VerifiedCartRunner
    from basil_core.noniid_study import partition_audit
    cache = ROOT/'experiments/cache/cifar10'
    images = np.load(cache/'x_train_float32.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(cache/'y_train_int32.npy', mmap_mode='r', allow_pickle=False)
    test_x = np.load(cache/'x_test_float32.npy', mmap_mode='r', allow_pickle=False)
    test_y = np.load(cache/'y_test_int32.npy', mmap_mode='r', allow_pickle=False)
    audit, _ = partition_audit(labels, test_y)
    if test_x.shape != (10000,32,32,3) or audit['partitionHash'] != config['partitionHash']:
        raise ValueError('Full data/partition identity differs from prepared configuration.')
    reference = TrainingReference(images, labels, source_split='train')
    probes = {i:tuple(map(int,keyed_rng(2025,'selection_batch',i).permutation(np.flatnonzero(labels == i))[:512])) for i in range(10)}
    with tf.device('/CPU:0'):
        canonical = build_model('basil_paper_cnn', 2025)
        initial = [w.numpy() for w in canonical.trainable_variables]
    if params_hash(initial) != config['initialModelHash'] or sum(p.size for p in initial) != 117706:
        raise ValueError('Canonical model initialization changed.')
    with tf.device('/GPU:0'):
        model = build_model('basil_paper_cnn', 2025)
        for w,p in zip(model.trainable_variables, initial): w.assign(p)
    if validate_only:
        # Real device/model/data/IPC/checkpoint startup, but no optimizer call.
        with tempfile.TemporaryDirectory(prefix='cart-gui-runtime-check-') as temp:
            fixture=Path(temp)/'train.npz'
            np.savez(fixture,images=images,labels=labels,synthetic=np.asarray(False))
            with IsolatedReplicaPool(fixture,timeout=15.) as pool, ParallelVerifierService(pool.endpoints) as service:
                runner=VerifiedCartRunner(model,reference,service,run_id=config['runId'],device='GPU',
                    ebm=config['conditionId']=='D',sigma_e=config['channelNoiseSigmaAbsolute'],
                    attacker_ids=tuple(config['resolvedAttackerIds']),trainer_failure_policy='verified_coordinator_recompute')
                sizes=[]
                for step in range(50):
                    runner.step=step;sizes.append(len(runner.next_batch()[0]))
                runner.step=0
                if sizes!=([512]*9+[392])*5:raise ValueError('Full-epoch schedule mismatch.')
                selection=runner.select({0:probes[0]})
                if selection['hash']!=config['initialModelHash'] or selection['sender']!=9:
                    raise ValueError('Initial strict handoff/tie-breaking mismatch.')
                path=Path(temp)/'initial_checkpoint.json';runner.save(path)
                restored=VerifiedCartRunner(model,reference,service,run_id=config['runId'],device='GPU',
                    ebm=config['conditionId']=='D',sigma_e=config['channelNoiseSigmaAbsolute'],
                    attacker_ids=tuple(config['resolvedAttackerIds']),trainer_failure_policy='verified_coordinator_recompute')
                restored.restore(path)
                normalized_selection=json.loads(json.dumps(runner.selection,allow_nan=False))
                if params_hash(restored.params)!=params_hash(runner.params) or restored.selection!=normalized_selection:
                    raise ValueError('Initial checkpoint did not restore exactly.')
                from scripts.check_verified_cart_full_batch import health
                resources=health(pool)
                result=dict(status='passed',trainingUpdates=0,condition=config['conditionId'],runtime=runtime,
                    partitionHash=audit['partitionHash'],initialModelHash=params_hash(initial),
                    selectedSender=selection['sender'],checkpointRestoredExactly=True,
                    fullActivationBatchSizes=sizes,replicaResources=resources,sourceHashes=source_hashes(),
                    integratedProofSha256=proof_hash,concurrentVerifierProofSha256=transport_proof_hash)
                atomic_json(ROOT/'gui/worker_state/NONIID_VERIFIED_GUI_RUNTIME_CHECK.json',result)
                print(json.dumps({k:v for k,v in result.items() if k!='replicaResources'},indent=2),flush=True)
                return 0
    if plan['mode'] == 'fresh':
        directory.mkdir(parents=True, exist_ok=True)
        state = dict(config=config, sourceHashes=source_hashes(), status='running', generation=-1,
            roundMetrics=[], roundNodes=[], roundSelections=[], roundLinks=[], epochEvaluations=[],
            archiveHashes={}, elapsedSeconds=0., helperPhase=False, resourceHistory=[])
        atomic_json(directory/'partition_audit.json', audit)
        atomic_json(directory/'manifest.json', dict(config=config, sourceHashes=source_hashes(),
            runtime=runtime, dataRoot=reference.root, integratedProofSha256=proof_hash,
            concurrentVerifierProofSha256=transport_proof_hash,
            fullBatchProofSha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            testUsage='evaluation_only', trainClassCounts=[5000]*10, testClassCounts=[1000]*10))
    else:
        state = json.loads((directory/'progress.json').read_text())
        manifest = json.loads((directory/'manifest.json').read_text())
        check_recovery_device(manifest['runtime'], runtime)
        if manifest['dataRoot'] != reference.root:
            raise ValueError('Restored TRAIN content hash differs.')
        for name,digest in state['archiveHashes'].items():
            path = directory/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Committed activation evidence corrupt: {name}')
        state['status'] = 'running'
    stop = [False]
    previous_handlers = {s:signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for s in previous_handlers: signal.signal(s, lambda *_:stop.__setitem__(0,True))
    started, prior_elapsed = time.monotonic(), state['elapsedSeconds']
    runner = None
    try:
        with tempfile.TemporaryDirectory(prefix='cart-30r-train-') as temp, ExitStack() as stack:
            fixture = Path(temp)/'train.npz'
            np.savez(fixture, images=images, labels=labels, synthetic=np.asarray(False))
            start_round = plan['completedRounds']
            faults = helper_faults(config, start_round)
            pool = stack.enter_context(IsolatedReplicaPool(fixture, faults=faults, timeout=15.))
            service=stack.enter_context(ParallelVerifierService(pool.endpoints))
            runner = VerifiedCartRunner(model, reference, service, run_id=config['runId'], device='GPU',
                ebm=config['conditionId']=='D', sigma_e=config['channelNoiseSigmaAbsolute'],
                attacker_ids=tuple(config['resolvedAttackerIds']), trainer_failure_policy='verified_coordinator_recompute')
            if plan['mode'] == 'checkpoint': runner.restore(directory/state['checkpoint'])
            else: commit(runner, state, directory)
            state['helperPhase'] = bool(faults)
            print(json.dumps(dict(restored=dict(progress=(runner.round*10+runner.node+runner.step/50)/300,
                round=runner.round, priorRuntimeSeconds=prior_elapsed,
                startProgress=(runner.round*10+runner.node+runner.step/50)/300,
                accuracyHistory=[[r['round']+1,r['mean_node_accuracy'],r['worst_node_accuracy']] for r in state['roundMetrics']],
                activationHistory=[], recoveryTotal=0,recoveryCompleted=0))), flush=True)
            while runner.round < 30:
                if stop[0]:
                    state.update(status='stopped', elapsedSeconds=prior_elapsed+time.monotonic()-started)
                    commit(runner,state,directory); publish(directory,state,runtime)
                    print(json.dumps(dict(event='stopped',cursor=state['cursor'])),flush=True)
                    return 130
                if helper_faults(config,runner.round) and not state['helperPhase']:
                    service.close()
                    pool.close()
                    pool = stack.enter_context(IsolatedReplicaPool(fixture,faults=helper_faults(config,runner.round),timeout=15.))
                    service=stack.enter_context(ParallelVerifierService(pool.endpoints))
                    runner.service=service; runner._record_membership(); state['helperPhase']=True
                    commit(runner,state,directory)
                round_id,node = runner.round,runner.node
                if runner.anchor is None:
                    selection = runner.select({i:probes[i] for i in (range(node+1) if round_id == 0 else range(10))})
                    state['activeSelection'] = selected_telemetry(config,node,round_id,selection)
                    state['epochEvaluations'] = [evaluate(model,runner.params,test_x,test_y,node=node,epoch=0)]
                    commit(runner,state,directory)
                while runner.step < 50 and not stop[0]:
                    receipt=runner.advance()
                    if runner.step % 10 == 0:
                        epoch=runner.step//10
                        observation=evaluate(model,runner.params,test_x,test_y,node=node,epoch=epoch)
                        from scripts.check_verified_cart_full_batch import health
                        observation['replicaResources']=health(pool)
                        before=state['epochEvaluations'][0]
                        observation['forgettingPerClass']=(np.asarray(before['perClass'])-observation['perClass']).tolist()
                        observation['snapshotDistanceSquared']=sum(float(np.sum((p.astype(np.float64)-a)**2)) for p,a in zip(runner.params,runner.anchor))
                        state['epochEvaluations'].append(observation)
                        state['elapsedSeconds']=prior_elapsed+time.monotonic()-started
                        commit(runner,state,directory)
                        print(json.dumps(dict(event='epoch_complete',round=round_id,node=node,epoch=epoch,
                            step=runner.step, accuracy=observation['accuracy'], objective=receipt['objective'],
                            seconds=state['elapsedSeconds'])),flush=True)
                if stop[0]: continue
                final=state['epochEvaluations'][-1]
                selection=state['activeSelection']
                records=list(runner.records)
                links=runner.finish_activation()
                link_stats=[{k:v for k,v in link.items() if k!='params'} for link in links]
                name=f'activation_{round_id*10+node:04d}.json'
                artifact=dict(round=round_id,node=node,selection=selection,epochs=state['epochEvaluations'],
                    records=records,outgoing=link_stats,trainedHash=final['parameterHash'])
                atomic_json(directory/name,artifact)
                state['archiveHashes'][name]=hashlib.sha256((directory/name).read_bytes()).hexdigest()
                state['roundNodes'].append(final);state['roundSelections'].append(selection)
                state['roundLinks'].extend(link_stats)
                # Completed proofs are archived. Keeping all 15,000 proofs in a
                # checkpoint would eventually exceed the bounded storage budget.
                runner.records=[]
                state['epochEvaluations']=[]; state.pop('activeSelection',None)
                if node==9:
                    measured=round_metrics(round_id,state,directory)
                    state['roundMetrics'].append(measured)
                    state['roundNodes']=[];state['roundSelections']=[];state['roundLinks']=[]
                state['elapsedSeconds']=prior_elapsed+time.monotonic()-started
                commit(runner,state,directory); publish(directory,state,runtime)
                from gui.services.noniid_telemetry import node_event
                network=node_event(config,artifact)
                print(json.dumps(dict(round=round_id,node=node,accuracy=final['accuracy'],networkUpdate=network)),flush=True)
                if node==9:
                    print(json.dumps(dict(roundComplete=dict(round=round_id,averageAccuracy=measured['mean_node_accuracy'],
                        worstNodeAccuracy=measured['worst_node_accuracy'],perNodeAccuracy=measured['per_node_accuracy']))),flush=True)
            state.update(status='completed',elapsedSeconds=prior_elapsed+time.monotonic()-started)
            commit(runner,state,directory);publish(directory,state,runtime)
        if os.environ.get('PAPERMERGE_SKIP_PLOTS')!='1':
            from reporting.noniid_verified_plots import report_condition, report_comparison
            report_condition(directory,profile_path(ROOT/config['plotDirectory']))
            report_comparison(ROOT,config)
        print(json.dumps(dict(event='completed',rounds=30)),flush=True)
        return 0
    except BaseException as error:
        failure=dict(event='failed',exceptionType=type(error).__name__,error=str(error),
            cursor=[runner.round,runner.node,runner.step] if runner else None,
            lastCommittedCheckpoint=state.get('checkpoint'),config=config)
        if runner is not None:
            try: runner.save(directory/'failure_checkpoint.json')
            except Exception as save_error: failure['failureCheckpointError']=str(save_error)
        atomic_json(directory/'failure.json',failure)
        print(json.dumps(failure),flush=True)
        raise
    finally:
        for s,handler in previous_handlers.items():signal.signal(s,handler)


def round_metrics(round_id, state, directory):
    import numpy as np
    nodes=state['roundNodes']
    if len(nodes)!=10 or [n['node'] for n in nodes] != list(range(10)):
        raise ValueError('Exactly ten measured node states required per round.')
    records=[r for node in range(10) for r in json.loads((directory/f'activation_{round_id*10+node:04d}.json').read_text())['records']]
    if len(records)!=500:raise ValueError('Exactly 500 committed SGD steps per round required.')
    selections=state['roundSelections']; links=state['roundLinks']
    result=dict(round=round_id,full_test_accuracy=nodes[9]['accuracy'],
        mean_node_accuracy=float(np.mean([n['accuracy'] for n in nodes])),
        worst_node_accuracy=min(n['accuracy'] for n in nodes),full_test_loss=nodes[9]['loss'],
        per_node_accuracy=[n['accuracy'] for n in nodes],per_node_per_class_accuracy=[n['perClass'] for n in nodes],
        corrupted_selection_rate=float(np.mean([s['selectedActuallyCorrupted'] for s in selections])),
        attacker_designated_selection_rate=float(np.mean([s['selectedAttackerDesignated'] for s in selections])),
        selected_snapshot_age=float(np.mean([s['snapshotAge'] for s in selections])),
        balanced_training_ce=float(np.mean([r['crossEntropy'] for r in records])),
        ebm_penalty=float(np.mean([r['ebmPenalty'] for r in records])),
        ebm_correction_norm=float(np.mean([r['ebmCorrectionNorm'] for r in records])),
        ebm_correction_ratio=float(np.mean([r['ebmCorrectionRatio'] for r in records])),
        channel_noise_norm=float(np.mean([l['noise']['noiseL2Norm'] for l in links])),
        noise_to_model_norm_ratio=float(np.mean([l['noise']['relativeNoiseL2'] for l in links])))
    result.update({f'class_{i}_accuracy':v for i,v in enumerate(nodes[9]['perClass'])})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--execute',action='store_true')
    action.add_argument('--validate-runtime',action='store_true',help='Real GPU/IPC/checkpoint startup with zero SGD updates.')
    parser.add_argument('--recover',action='store_true',help='Compatibility flag; committed state is always inspected and restored.')
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text())
    from basil_core.noniid_verified_study import validate_configuration,recovery_plan
    validate_configuration(config)
    plan=recovery_plan(ROOT/config['resultDirectory'],config)
    if not args.execute and not args.validate_runtime:
        print(json.dumps(dict(status='valid',config=config,recovery=plan,trainingRunsStarted=0),indent=2))
        return 0
    return execute(config,validate_only=args.validate_runtime)


if __name__=='__main__': raise SystemExit(main())
