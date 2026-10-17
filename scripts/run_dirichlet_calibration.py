#!/usr/bin/env python3
"""Run explicit bounded Dirichlet diagnostics, never auto-start a GUI queue.

Full TRAIN partitions, five epochs and evaluation-only TEST; the original
three-round pair and later readiness stages remain separate. No W&B or tuning.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULT = ROOT / 'newResults/nonIID/dirichlet_logit_calibration_3r_v1'


def save_json(path, payload):
    from scripts.run_noniid_basil_study import save_json as atomic_json
    atomic_json(path, payload)


def verify_runtime(*,method='fedlc',readiness=False):
    """Synthetic CPU/GPU gradients and one SGD update; no CIFAR run."""
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow,GPU_MEMORY_LIMIT_MB
    prepare_environment('GPU')
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import build_model,params_hash
    from basil_core.dirichlet_calibration import DirichletCalibrationWorker,calibration_offsets,REVISION,GATE_REVISION
    configure_tensorflow(tf,'GPU')
    x=np.random.default_rng(17).uniform(0,1,(512,32,32,3)).astype(np.float32)
    y=np.arange(512,dtype=np.int32)%10
    rows={}
    outcomes={}
    for device in ('CPU','GPU'):
        with tf.device(f'/{device}:0'):
            worker=DirichletCalibrationWorker(build_model('basil_paper_cnn',2025),calibration=True,device=device,method=method,
                revision=GATE_REVISION if readiness else REVISION)
            start=worker.export()
            for anchor,value in zip(worker.reference,start):anchor.assign(value+.001)
            frozen=params_hash([v.numpy() for v in worker.reference])
            worker.offsets.assign(calibration_offsets(np.array([0,1,3,16,64,128,512,1000,2000,5000]),method=method))
            repeated=[]
            for legacy in (1.,999999.):
                gradients,values=worker._gradients(tf.constant(x),tf.constant(y),tf.constant([7,8]),
                    'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(legacy))
                repeated.append(([g.numpy() for g in gradients],np.array([float(v) for v in values])))
            for first,second in zip(repeated[0][0],repeated[1][0]):np.testing.assert_array_equal(first,second)
            np.testing.assert_array_equal(repeated[0][1],repeated[1][1])
            for size in (19,37,160,183,311,378,385,392,444,477,502):
                partial,_=worker._gradients(tf.constant(x[:size]),tf.constant(y[:size]),tf.constant([7,8]),
                    'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(999999.))
                assert all(np.isfinite(g.numpy()).all() for g in partial)
            tf.keras.optimizers.SGD(.05,momentum=0.).apply_gradients(zip(gradients,worker.model.trainable_variables))
            assert params_hash([v.numpy() for v in worker.reference])==frozen
            assert all(device in g.device for g in gradients)
            outcomes[device]=repeated[0]
            rows[device]=dict(parameterCount=sum(v.size for v in start),legacyLambdaIgnored=True,referenceUnchanged=True)
            if readiness:
                worker.load(start)
                corrected=[]
                for legacy in (1.,999999.):
                    gradients,values=worker._gradients(tf.constant(x),tf.constant(y),tf.constant([7,8]),
                        'gradient_norm_objective','paper_absolute_gaussian',tf.constant(.01),tf.constant(legacy))
                    corrected.append(([g.numpy() for g in gradients],np.array([float(v) for v in values])))
                for a,b in zip(corrected[0][0],corrected[1][0]):np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(corrected[0][1],corrected[1][1])
                assert abs(float(values[3])-.0001)<1e-10 and float(values[6])>0
                for size in (19,37,160,183,311,378,385,392,444,477,502):
                    partial,_=worker._gradients(tf.constant(x[:size]),tf.constant(y[:size]),tf.constant([7,8]),
                        'gradient_norm_objective','paper_absolute_gaussian',tf.constant(.01),tf.constant(999999.))
                    assert all(np.isfinite(g.numpy()).all() for g in partial)
                rows[device]['ebmCoefficient']=float(values[3])
                rows[device]['ebmCorrectionNorm']=float(values[6])
                outcomes[device+'EBM']=corrected[0]
    diff=np.concatenate([(a-b).ravel() for a,b in zip(outcomes['CPU'][0],outcomes['GPU'][0])])
    base=np.concatenate([a.ravel() for a in outcomes['CPU'][0]])
    relative=float(np.linalg.norm(diff)/np.linalg.norm(base))
    assert relative<.01,relative
    if readiness:
        difference=np.concatenate([(a-b).ravel() for a,b in zip(outcomes['CPUEBM'][0],outcomes['GPUEBM'][0])])
        reference=np.concatenate([a.ravel() for a in outcomes['CPUEBM'][0]])
        ebm_relative=float(np.linalg.norm(difference)/np.linalg.norm(reference))
        assert ebm_relative<.01,ebm_relative
    np.testing.assert_allclose(outcomes['CPU'][1][[0,2,13]],outcomes['GPU'][1][[0,2,13]],atol=5e-7,rtol=1e-4)
    memory=tf.config.experimental.get_memory_info('GPU:0')
    assert memory['peak']<.75*GPU_MEMORY_LIMIT_MB*1024**2
    report=dict(status='passed',researchRunsStarted=0,method=method,checks=rows,gradientRelativeL2Difference=relative,
        gpuMemory=memory,tau=1.,mu=.01,coefficient=0.,batchSizes=[512,19,37,160,183,311,378,385,392,444,477,502],
        objectiveSha256=hashlib.sha256((ROOT/'basil_core/dirichlet_calibration.py').read_bytes()).hexdigest())
    filename='DIRICHLET_PRIOR_GPU_VERIFICATION.json' if method=='prior_adjustment' else 'DIRICHLET_CALIBRATION_GPU_VERIFICATION.json'
    if readiness:
        filename='DIRICHLET_READINESS_GPU_VERIFICATION.json'
        report.update(ebmVerified=True,ebmGradientRelativeL2Difference=ebm_relative,revision=GATE_REVISION)
    save_json(ROOT/'gui/worker_state'/filename,report)
    print(json.dumps(report),flush=True)
    return report


def write_round_metrics(output, rows):
    """Atomically publish measured rows; an empty prefix has a header, not data."""
    fields=['round','full_test_accuracy','mean_node_accuracy','worst_node_accuracy','full_test_loss']
    fields += [f'class_{c}_accuracy' for c in range(10)]
    target=Path(output)/'round_metrics.csv'
    temporary=target.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
        handle.flush();os.fsync(handle.fileno())
    temporary.replace(target)


def saved_round_rows(state):
    """Reconstruct only fully evaluated, checkpoint-committed rounds on resume."""
    if state is None:return []
    rows=[]
    for r in range(state['completed_rounds']):
        accuracy=state['round_node_accuracy'][r]
        row=dict(round=r,full_test_accuracy=float(accuracy[9]),
            mean_node_accuracy=float(accuracy.mean()),worst_node_accuracy=float(accuracy.min()),
            full_test_loss=float(state['round_node_loss'][r,9]))
        row.update({f'class_{c}_accuracy':float(state['round_node_per_class'][r,9,c]) for c in range(10)})
        rows.append(row)
    return rows


def emit_restored(state, prior_time, config):
    rows=saved_round_rows(state)
    progress=state['next_activation']/(config['nRounds']*config['nNodes'])
    print(json.dumps({'restored':dict(round=state['completed_rounds'],progress=progress,
        startProgress=progress,priorRuntimeSeconds=prior_time,recoveryCompleted=0,recoveryTotal=0,
        accuracyHistory=[(r['round']+1,r['mean_node_accuracy'],r['worst_node_accuracy']) for r in rows],
        activationHistory=[(r['round']+(r['nodeId']+1)/config['nNodes'],r['globalTestAccuracy'])
            for r in state['telemetry']])},allow_nan=False),flush=True)


def run_pair(device, *, recover=False, prior_only=False, output_root=None, stage='initial',
             configs_override=None, gui_events=False):
    if stage=='threat' and configs_override is None:
        raise ValueError('Confirmation runs require one evidence-approved config; use run_dirichlet_prior_condition.py.')
    result_root=Path(output_root or RESULT).resolve()
    if not result_root.is_relative_to((ROOT/'newResults/nonIID').resolve()):
        raise ValueError('Small non-IID diagnostics must stay in newResults/nonIID.')
    horizon = bool(configs_override and configs_override[0].get('horizonStudyRevision'))
    if configs_override is not None:
        if horizon:
            from basil_core.dirichlet_horizon_study import validate_evidence
        else:
            from basil_core.dirichlet_prior_study import validate_evidence
        if stage!='threat' or prior_only or len(configs_override)!=1:
            raise ValueError('A GUI execution must select exactly one approved confirmation condition.')
        validate_evidence(configs_override[0])
        if configs_override[0]['executionDevice']!=device:
            raise ValueError('Requested device differs from the approved configuration.')
    tracking_enabled=bool(horizon and configs_override[0].get('liveTrackingEnabled'))
    if not tracking_enabled:
        os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled')
    else:
        os.environ.pop('PAPERMERGE_LIVE_TRACKING',None)
        os.environ['WANDB_MODE']='online'
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow, runtime_metadata
    prepare_environment(device)
    import numpy as np
    import tensorflow as tf
    from basil_core.dirichlet_calibration import diagnostic_configs, DirichletCalibrationWorker,prior_diagnostic_config,readiness_configs
    from basil_core.dirichlet_study import partition_audit
    from basil_core.research_protocol import build_model, params_hash, run_research_protocol
    from basil_core.protocol_checkpoint import checkpoint_implementation, save_checkpoint, load_checkpoint, TelemetryStore
    from basil_core.memory_safety import check_host_memory, RuntimeMemoryGuard
    from basil_core.dirichlet_diagnosis import array_identity_hash
    from basil_core.research_audit import ProtocolAudit
    class CalibrationAudit(ProtocolAudit):
        def begin_batch(self,*args,**kwargs):
            with tf.device('/CPU:0'):
                super().begin_batch(*args,**kwargs)
        def gradients(self,gradients,values):
            super().gradients(gradients,values)
            self.current.update(zip(DirichletCalibrationWorker.extra_metric_names,map(float,values[6:])))
        def failure(self,worker,error):
            self.write('failure.json',dict(**getattr(self,'current',{}),exception=str(error),
                graphReplaySkipped=True, localObjective=cfg['localObjective'],
                referenceHash=params_hash([v.numpy() for v in worker.reference])))
            if self.pending:
                np.savez_compressed(self.directory/'failure_batch.npz',x=self.pending['x'],y=self.pending['y'],
                    key=self.pending['key'],**{f'weight_{i}':p for i,p in enumerate(self.pending['params'])},
                    **{f'anchor_{i}':a.numpy() for i,a in enumerate(worker.reference)})
    configure_tensorflow(tf, device)
    runtime = runtime_metadata(tf, dict(executionDevice=device))
    if gui_events:
        print(json.dumps(dict(event='device_ready',device=runtime['device'],
            deviceName=runtime.get('deviceName',''))),flush=True)
    if device=='GPU':
        filename='DIRICHLET_PRIOR_GPU_VERIFICATION.json' if prior_only else 'DIRICHLET_CALIBRATION_GPU_VERIFICATION.json'
        if stage!='initial':filename='DIRICHLET_READINESS_GPU_VERIFICATION.json'
        verified=json.loads((ROOT/'gui/worker_state'/filename).read_text())
        if verified['status']!='passed' or verified['objectiveSha256']!=hashlib.sha256((ROOT/'basil_core/dirichlet_calibration.py').read_bytes()).hexdigest():
            raise ValueError('Current calibrated objective requires synthetic GPU validation before execution.')
        if stage!='initial' and (verified.get('revision')!='dirichlet_prior_readiness_diagnostic_v1' or verified.get('ebmVerified') is not True):
            raise ValueError('Readiness runs require verified prior CE and full second-order EBM, not a CE-only GPU report.')
    check_host_memory()
    cache = ROOT/'experiments/cache/cifar10'
    x, y, tx, ty = [np.load(cache/f, mmap_mode='r', allow_pickle=False) for f in (
        'x_train_float32.npy','y_train_int32.npy','x_test_float32.npy','y_test_int32.npy')]
    if x.shape!=(50000,32,32,3) or tx.shape!=(10000,32,32,3):
        raise ValueError('Full cached CIFAR image sets required.')
    partition, _ = partition_audit(y, ty)
    with tf.device('/CPU:0'):
        canonical = build_model('basil_paper_cnn', 2025)
        initial_hash = params_hash(canonical.get_weights())
    configs = [prior_diagnostic_config(partition['partitionHash'],initial_hash)] if prior_only else diagnostic_configs(partition['partitionHash'], initial_hash)
    names=('prior_adjusted',) if prior_only else ('baseline','calibrated')
    if stage!='initial':
        configs=readiness_configs(partition['partitionHash'],initial_hash,stage)
        names=tuple(c['runId'] for c in configs)
    if configs_override is not None:
        configs=[dict(c) for c in configs_override];names=tuple(c['runId'] for c in configs)
        for name,cfg in zip(names,configs):
            if cfg['partitionHash']!=partition['partitionHash'] or cfg['initialModelHash']!=initial_hash:
                raise ValueError('Actual data partition / initialization differs from approved pairing.')
            if (ROOT/cfg['resultDirectory']).resolve()!=result_root/name:
                raise ValueError('Approved single-condition output path differs from execution root.')
    else:
        for name,cfg in zip(names,configs):
            cfg['executionDevice'] = device
            cfg['resultDirectory']=str((result_root/name).relative_to(ROOT))
            cfg['plotDirectory']=str(Path('newPlots/nonIID')/result_root.relative_to(ROOT/'newResults/nonIID')/name)
    source_files = checkpoint_implementation(configs[0])
    data_hashes = {k:array_identity_hash(v) for k,v in zip(
        ('trainImages','trainLabels','testImages','testLabels'),(x,y,tx,ty))}
    stopped = False
    def stop_requested(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    outcomes = {}
    for name, cfg in zip(names,configs):
        if stopped: raise InterruptedError('Stopped before next condition.')
        output = result_root/name
        previous = None
        state = None
        records = []
        if output.exists():
            if not recover: raise ValueError('Evidence exists: explicit --recover required; no overwrite.')
            previous = json.loads((output/'run.json').read_text())
            if (previous['config']!=cfg or previous['implementationSha256']!=source_files or
                    previous['dataSha256']!=data_hashes or previous['numericalExecution']!=runtime['numericalExecution']):
                raise ValueError('Resume identity / source / numerical policy mismatch.')
            if previous['status']=='completed':
                outcomes[name]=previous; continue
            startup_retry=horizon and previous.get('status')=='tracking_unavailable'
            if startup_retry:
                from basil_core.dirichlet_horizon_study import recovery_plan
                if recovery_plan(output,cfg)['mode']!='startup_retry':raise ValueError('Unverified startup retry.')
            else:
                checkpoint_path=output/'checkpoint.npz'
                if configs_override is not None:
                    if horizon:
                        from basil_core.dirichlet_horizon_study import recovery_plan
                    else:
                        from basil_core.dirichlet_prior_study import recovery_plan
                    checkpoint_path=recovery_plan(output,cfg)['checkpoint']
                state = load_checkpoint(checkpoint_path,cfg,
                    partition_hash=partition['partitionHash'], initial_hash=initial_hash)
                records = list(state['telemetry'])
        output.mkdir(parents=True, exist_ok=True)
        # Future diagnostic repetitions retain the exact executed source bytes,
        # not just hashes of an evolving, potentially uncommitted worktree.
        snapshots=output/'source_snapshots';snapshots.mkdir(exist_ok=True)
        for filename,digest in source_files.items():
            raw=(ROOT/filename).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Source changed before execution.')
            target=snapshots/digest
            if target.exists() and target.read_bytes()!=raw:raise ValueError('Archived source hash collision / corruption.')
            if not target.exists():target.write_bytes(raw)
        audit_path=output/('audit' if state is None else f"audit_resume_{state['next_activation']}")
        attempt=0
        while audit_path.exists():
            attempt+=1
            audit_path=output/f"audit_resume_{state['next_activation'] if state else 0}_{attempt}"
        audit=CalibrationAudit(audit_path,candidate_evaluation=stage!='initial')
        save_json(output/'config.json',cfg); save_json(output/'partition_audit.json',partition)
        started = time.perf_counter()
        prior_time = previous.get('runtimeSeconds',0.) if previous else 0.
        guard = RuntimeMemoryGuard(tf,device)
        store = TelemetryStore(output/'checkpoint.npz')
        metadata = dict(status='running', config=cfg, **runtime, purpose=cfg['purpose'], researchValid=cfg['researchValid'],
            runId=cfg['runId'],experimentProtocol=cfg['experimentProtocol'],
            pythonVersion=platform.python_version(),numpyVersion=np.__version__,tensorflowVersion=tf.__version__,
            gitCommit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            gitWorktreeDirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
            executionMode='deterministic_single_process',multiprocessingUsed=False,
            resolvedAttackerIds=cfg['resolvedAttackerIds'],attackType=cfg['attackType'],
            attackStartRound=cfg['attackStartRound'],attackSourceStatus=cfg['attackSourceStatus'],
            sigma_e=cfg['channelNoiseSigmaAbsolute'],sigma_e_squared=cfg['channelNoiseSigmaAbsolute']**2,
            legacyLambdaIgnored=True,rngDerivation='SHA256(seed|stream|round|node|epoch|batch or sender|receiver)',
            localObjective=cfg['localObjective'], objectiveScope='local_calibration_extension_not_old_baseline',
            snapshotSelectionLoss='ordinary_receiver_local_ce', epochSemantics='full_local_dataset',
            candidateGlobalEvaluationOnly=stage!='initial',
            partitionHash=partition['partitionHash'], initialModelHash=initial_hash, dataSha256=data_hashes,
            implementationSha256=source_files, timestamp=datetime.now(timezone.utc).isoformat(),
            sourceSnapshotDirectory='source_snapshots',
            parameterCount=117706, wandbEnabled=tracking_enabled, helpersEnabled=False, trainSharing=False,
            attackEnabled=cfg['attackHidden'], channelNoiseEnabled=cfg['useChannelNoise'],
            ebmEnabled=cfg['ebmMode']=='gradient_norm_objective',
            ebmBaseObjective='adjusted_cross_entropy' if cfg['logitCalibrationEnabled'] else 'ordinary_cross_entropy',
            testUsage='evaluation_only; never loss/selection/parameter choice/stopping',
            calibrationTau=1., countFloor=1, snapshotAnchorMu=.01,
            completedRounds=state['completed_rounds'] if state else 0,
            completedActivations=state['next_activation'] if state else 0,
            runtimeSeconds=prior_time)
        # Observation only, with a separate non-IID group and no old IID IDs.
        tracking=None
        if tracking_enabled:
            from basil_core.live_tracking import LiveTracker
            from basil_core.dirichlet_horizon_study import tracking_settings
            tracking=LiveTracker(cfg,{**metadata,'liveTracking':(previous or {}).get('liveTracking',{})},
                settings=tracking_settings(cfg),notify=lambda payload:print(json.dumps(payload),flush=True))
            metadata['liveTracking']=tracking.metadata
            if tracking.metadata['status']!='active':
                metadata.update(status='tracking_unavailable',startupFailure=True,
                    error='W&B initialization failed before any scientific update')
                save_json(output/'run.json',metadata)
                audit.close()
                raise ValueError('W&B could not initialize: no scientific updates started. Check local login/connectivity.')
        save_json(output/'run.json',metadata)
        rows=saved_round_rows(state)
        if tracking is not None:
            for row in rows:tracking.round(row)
        write_round_metrics(output,rows)
        if gui_events and state is not None:emit_restored(state,prior_time,cfg)
        def checkpoint(current):
            save_checkpoint(output/'checkpoint.npz',cfg,current,
                partition_hash=partition['partitionHash'], initial_hash=initial_hash, telemetry_store=store)
            metadata.update(completedRounds=current['completed_rounds'], completedActivations=current['next_activation'],
                runtimeSeconds=prior_time+time.perf_counter()-started, memorySafety=guard.checkpoint())
            if tracking is not None:metadata['liveTracking']=dict(tracking.metadata)
            save_json(output/'run.json',metadata)
        def activation(record):
            if record['inputHash']!=record['snapshotAnchorReferenceHash'] or record['optimizerSteps']!=partition['nodes'][record['nodeId']]['optimizerStepsPerActivation']:
                raise ValueError('Handoff / epoch contract failed.')
            records.append(record)
            if tracking is not None:tracking.activation(record)
            if gui_events:
                from gui.services.research_telemetry import activation_payload
                payload=activation_payload(record,cfg['experimentName'])
            else:payload=dict(condition=name,activation=len(records),round=record['round'],node=record['nodeId'])
            print(json.dumps(payload,allow_nan=False),flush=True)
        def round_done(row):
            r=row['round']
            if r!=len(rows):raise ValueError('Measured round rows must be contiguous, never duplicated.')
            item=dict(round=r,full_test_accuracy=row['fullTestAccuracy'],
                mean_node_accuracy=row['averageAccuracy'],worst_node_accuracy=row['worstNodeAccuracy'],
                full_test_loss=row['fullTestLoss'])
            item.update({f'class_{c}_accuracy':float(row['perClassAccuracy'][c]) for c in range(10)})
            rows.append(item);write_round_metrics(output,rows)
            if tracking is not None:
                measured=[record for record in records if record['round']==r]
                steps=[s for record in measured for s in record['optimizerTelemetry']]
                links=[link for record in measured for link in record['outgoingLinks']]
                tracked=dict(item)
                for key,field in [('baseCrossEntropy','local_prior_ce'),('ebmPenalty','ebm_penalty'),
                        ('baseGradientNorm','ordinary_gradient_norm'),('ebmCorrectionNorm','ebm_correction_norm'),
                        ('ebmCorrectionRatio','ebm_correction_ratio'),('snapshotAnchorPenalty','snapshot_anchor_penalty')]:
                    tracked[field]=float(np.mean([s[key] for s in steps]))
                tracked.update(channel_noise_norm=float(np.mean([l['noiseL2Norm'] for l in links])),
                    noise_to_model_norm_ratio=float(np.mean([l['relativeNoiseL2'] for l in links])))
                tracking.round(tracked)
            print(json.dumps({'roundComplete':row} if gui_events else
                dict(condition=name,event='round_completed',**row),allow_nan=False),flush=True)
        try:
            with tf.device(f'/{device}:0'):
                result = run_research_protocol(cfg,x,y,tx,ty,
                    worker_factory=lambda model:DirichletCalibrationWorker(model,
                        calibration=cfg['logitCalibrationEnabled'],device=device,method=cfg.get('logitCalibrationMethod','fedlc'),
                        revision=cfg['lossStudyRevision']),
                    activation_callback=activation,round_callback=round_done,
                    resume_state=state,checkpoint_callback=checkpoint,should_stop=lambda:stopped,audit=audit)
            for node, params in enumerate(result['finalNodeParams']):
                np.savez_compressed(output/f'node_{node}_weights.npz',**{f'weight_{i}':p for i,p in enumerate(params)})
            np.savez_compressed(output/'metrics.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
            (output/'activation_telemetry.json.gz').write_bytes(gzip.compress(
                json.dumps(result['telemetry'],allow_nan=False).encode(),mtime=0))
            final_rows=[]
            for round_id in range(cfg['nRounds']):
                row=dict(round=round_id, full_test_accuracy=float(result['fullTestAccuracy'][round_id]),
                    mean_node_accuracy=float(result['averageAccuracy'][round_id]),
                    worst_node_accuracy=float(result['worstNodeAccuracy'][round_id]),
                    full_test_loss=float(result['fullTestLoss'][round_id]))
                row.update({f'class_{c}_accuracy':float(result['fullTestPerClassAccuracy'][round_id,c]) for c in range(10)})
                final_rows.append(row)
            if rows!=final_rows:raise ValueError('Live round rows disagree with final measured arrays.')
            write_round_metrics(output,final_rows)
            metadata.update(status='completed',completedRounds=cfg['nRounds'],completedActivations=10*cfg['nRounds'],
                runtimeSeconds=prior_time+time.perf_counter()-started, finalMetrics=rows[-1])
            save_json(output/'run.json',metadata); outcomes[name]=metadata
        except BaseException as error:
            metadata.update(status='interrupted' if isinstance(error,InterruptedError) else 'failed',
                error=str(error),runtimeSeconds=prior_time+time.perf_counter()-started)
            # Save the exact current failing state without replacing the last
            # committed activation checkpoint or allocating another GPU model.
            last=records[-1] if records else None
            save_json(output/'failure.json',dict(exception=str(error),lastCompletedActivation=last,
                completedRounds=metadata['completedRounds'],completedActivations=metadata['completedActivations']))
            save_json(output/'run.json',metadata)
            raise
        finally:
            if tracking is not None:
                tracking.finish(metadata['status'])
                metadata['liveTracking']=dict(tracking.metadata)
                save_json(output/'run.json',metadata)
            audit.close()
    if configs_override is None:
        save_json(result_root/('prior_stage.json' if prior_only else 'comparison.json'),dict(status='completed', conditions=outcomes,
            contrasts='Prospective bounded diagnostics; not production evidence', stage=stage,researchValid=False))
    return outcomes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--execute',action='store_true')
    action.add_argument('--verify-runtime',action='store_true')
    parser.add_argument('--device',choices=('CPU','GPU'),default='GPU')
    parser.add_argument('--recover',action='store_true')
    parser.add_argument('--prior-only',action='store_true',help='Only the second, TRAIN-prior adjustment diagnostic; preserve the initial pair.')
    parser.add_argument('--stage',choices=('initial','clean','noise','attack_gate','threat'),default='initial',
        help='Prospective diagnostics: clean=10, noise=3, attack_gate=23 B/C/D; threat=30 A/B/C/D. Default is dry-run.')
    parser.add_argument('--output-root',type=Path,help='Unused responsibility-based namespace inside newResults/nonIID; never overwrite evidence.')
    args=parser.parse_args()
    if args.prior_only and args.stage!='initial':parser.error('--prior-only belongs to the original three-round diagnostic.')
    if args.output_root is None:
        rounds={'clean':10,'noise':3,'attack_gate':23,'threat':30}.get(args.stage)
        args.output_root=RESULT if rounds is None else ROOT/f'newResults/nonIID/dirichlet_prior_readiness_{args.stage}_{rounds}r_v1'
    if not args.output_root.resolve().is_relative_to((ROOT/'newResults/nonIID').resolve()):
        parser.error('--output-root must be inside newResults/nonIID')
    if args.verify_runtime:
        verify_runtime(method='prior_adjustment' if args.prior_only or args.stage!='initial' else 'fedlc',readiness=args.stage!='initial');return
    if not args.execute:
        variants={'clean':['baseline','prior_adjusted'],'noise':['noise_ce','noise_ebm'],'attack_gate':list('BCD'),'threat':list('ABCD')}
        print(json.dumps(dict(trainingStarted=False, rounds={'clean':10,'noise':3,'attack_gate':23,'threat':30}.get(args.stage,3),localEpochs=5,
            fullData=True, variants=variants.get(args.stage,['prior_adjusted'] if args.prior_only else ['baseline','calibrated']), device=args.device,
            tau=1.,mu=.01,noise=args.stage in {'noise','attack_gate','threat'},attack=args.stage in {'attack_gate','threat'},wandb=False)))
        return
    run_pair(args.device,recover=args.recover,prior_only=args.prior_only,output_root=args.output_root,stage=args.stage)


if __name__=='__main__':main()
