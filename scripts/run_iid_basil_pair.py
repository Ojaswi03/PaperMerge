#!/usr/bin/env python3
"""Prepare or execute only the two paired IID BASIL tests, with preflight gates."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basil_core.iid_runtime import prepare_environment, runtime_metadata, check_recovery_device

# Legacy CLI invocations remain CPU-only; the GUI child selects GPU explicitly.
prepare_environment(os.environ.get('PAPERMERGE_IID_DEVICE','CPU'))

import numpy as np
import tensorflow as tf
from basil_core.data.cifar import _datasetArrays, loadCifar10
from basil_core.iid_study import (
    RESULT_ROOT, PLOT_ROOT, TEST_NAMES, CALIBRATED_SIGMAS, IidWorker,
    paired_configs, partition_audit, preflight_config, verify_pair, iid_output,
)
from basil_core.research_protocol import (
    build_model, params_hash, evaluate_params, run_research_protocol, validate_config, keyed_rng,
)
from basil_core.research_audit import ProtocolAudit, array_hash
from basil_core.protocol_checkpoint import (
    recovery_plan, read_activation_records, load_checkpoint, save_checkpoint,
    verify_replayed_activation,
)


def save_json(path, data):
    path = iid_output(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temporary.replace(path)


def prepare(configs):
    for root in (ROOT/"newResults",ROOT/"newPlots"):
        for family in ("IID","nonIID"):
            (root/family).mkdir(parents=True,exist_ok=True)
    for config in configs:
        existing=RESULT_ROOT/config["experimentName"]/"run.json"
        if existing.exists() and json.loads(existing.read_text())["config"]!=config:
            raise ValueError(f"Refusing to alter the configuration of an existing research run: {existing}")
        for root in (RESULT_ROOT,PLOT_ROOT):
            (root/config["experimentName"]).mkdir(parents=True,exist_ok=True)
        save_json(RESULT_ROOT/config["experimentName"]/"config.json",config)
    for root in (RESULT_ROOT,PLOT_ROOT):
        (root/"comparison").mkdir(exist_ok=True)
    save_json(RESULT_ROOT/"comparison"/"pairing.json",{
        "intentionalDifferences":verify_pair(*configs),"configs":configs,
        "sigmaStatus":configs[1]["sigmaApproval"],"calibratedCandidates":CALIBRATED_SIGMAS})
    from reporting.iid_study_plots import write_study_summary
    write_study_summary()


def execute(config, data, audit, *, family, recover=False):
    runtime=runtime_metadata(tf,config)
    if config.get('recoverySourceDirectory'):
        if not recover:raise ValueError('An explicit --recover is required for the CPU-to-GPU state transition.')
        from basil_core.iid_device_transition import transfer_cpu_state
        transfer_cpu_state(ROOT,config,data,audit,runtime)
    with tf.device(runtime['tensorflowDevice']):
        return _execute(config,data,audit,family=family,recover=recover,runtime=runtime)


def _execute(config, data, audit, *, family, recover, runtime):
    validate_config(config)
    name = config["experimentName"]
    campaign=bool(config.get('iidCampaignId'))
    output = iid_output(ROOT/config['resultDirectory']) if campaign else iid_output(RESULT_ROOT/name/(family if family != "research" else ""))
    output.mkdir(parents=True,exist_ok=True)
    previous=None; plan=None; resume_state=None; saved_records=[]
    if (output/"run.json").exists():
        previous=json.loads((output/"run.json").read_text())
        check_recovery_device(previous,runtime)
        if family=="preflight" and previous.get("preflightPassed") and previous["config"]==config and previous["iidPartitionHash"]==audit["partitionHash"]:
            print("Reusing matching completed full-data preflight:",output,flush=True)
            return previous
        if not recover or not campaign:
            raise ValueError(f"Refusing to overwrite an existing run: {output}")
        plan=recovery_plan(output,config)
        if plan['mode']=='completed':raise ValueError('This experiment already completed.')
        saved_records=read_activation_records(output/'partial_activation_telemetry.jsonl')
        if plan['mode']=='checkpoint':
            resume_state=load_checkpoint(plan['checkpoint'],config,
                partition_hash=audit['partitionHash'],initial_hash=previous['initialModelHash'])
    save_json(output/"config.json",config)
    train_x,train_y,test_x,test_y = data
    if config.get("diagnosticSamplesPerClass"):
        # Training subsampling is performed by the shared engine; only its
        # explicitly labelled smoke evaluation subset is formed here.
        chosen = np.concatenate([np.flatnonzero(test_y==c)[:config["diagnosticTestSamplesPerClass"]] for c in range(10)])
        test_x,test_y = test_x[chosen],test_y[chosen]
    # All devices start from the approved CPU-generated initialization bytes.
    with tf.device('/CPU:0'):
        cpu_initial_worker=IidWorker(build_model(config["modelArchitecture"],config["seed"]))
        canonical_params=cpu_initial_worker.export()
    def make_worker(model):
        if runtime['device']=='GPU':
            from basil_core.iid_gpu_worker import GpuIidWorker
            model=build_model(config['modelArchitecture'],config['seed'])
            worker=GpuIidWorker(model)
            worker.load(canonical_params)
            if not all('GPU:' in v.handle.device for v in model.trainable_variables):
                raise RuntimeError('GPU model variables were not placed on CUDA. Refusing CPU fallback.')
            return worker
        return IidWorker(model)
    initial_worker = make_worker(None) if runtime['device']=='GPU' else cpu_initial_worker
    initial_hash = params_hash(initial_worker.export())
    if campaign and (initial_hash!=config['expectedInitialModelHash'] or audit['partitionHash']!=config['expectedPartitionHash']):
        raise ValueError('Partition/initialization differs from the prepared paired campaign.')
    initial_accuracy,_ = evaluate_params(initial_worker,initial_worker.export(),test_x,test_y)
    data_hashes={key:array_hash(value) for key,value in zip(
        ('trainImagesHash','trainLabelsHash','testImagesHash','testLabelsHash'),
        (train_x,train_y,test_x,test_y))}
    if previous:
        expected_data=previous.get('dataSha256')
        if expected_data is None:
            expected_data=json.loads((output/'audit/initial.json').read_text())
        if any(expected_data.get(key)!=value for key,value in data_hashes.items()):
            raise ValueError('Recovery dataset bytes differ from the original run.')
        if previous['tensorflowVersion']!=tf.__version__:
            raise ValueError('Recovery requires the original TensorFlow version.')
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ("basil_core/iid_study.py","basil_core/research_protocol.py","basil_core/research_audit.py",
         "basil_core/data/cifar.py","basil_core/models.py","basil_core/attacks.py","basil_core/trainer.py",
         "scripts/run_iid_basil_pair.py")}
    metadata = {"experimentName":name,"experimentProtocol":config["experimentProtocol"],
        "gitCommit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "gitWorktreeDirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True)),
        "implementationSha256":source_hashes,
        "timestamp":datetime.now(timezone.utc).isoformat(),"tensorflowVersion":tf.__version__,
        "pythonVersion":platform.python_version(),**runtime,"executionMode":"deterministic_single_process",
        "multiprocessingUsed":False,
        "seed":config["seed"],"seedStreams":{s:config["seed"] for s in
            ("partition","initialization","shuffle","augmentation","attacker")},
        "partitionSeed":config["seed"],"shuffleSeed":config["seed"],
        "augmentationSeed":config["seed"],"attackerSeed":config["seed"],
        "initializationSeed":int(keyed_rng(config["seed"],"model_initialization").integers(0,2**30)),
        "rngDerivation":"SHA256(seed|stream|round|node|epoch|batch or sender|receiver), separate streams",
        "initialModelHash":initial_hash,"initialGlobalAccuracy":initial_accuracy,
        "dataSha256":data_hashes,
        "iidPartitionHash":audit["partitionHash"],"resolvedAttackerIds":config["resolvedAttackerIds"],
        "attackType":config["attackType"],"attackStartRound":config["attackStartRound"],
        "hiddenAttackParameters":{"blendRatio":.55,"attackStrength":1.4},
        "nNodes":10,"b":4,"S":5,"nRounds":config["nRounds"],"localEpochs":5,"batchSize":512,
        "learningRate":.05,"learningRateSchedule":config["learningRateSchedule"],"optimizer":"sgd",
        "channelNoiseSemantics":config["channelNoiseSemantics"],
        "sigma_e":config["channelNoiseSigmaAbsolute"],"sigma_e_squared":config["channelNoiseSigmaAbsolute"]**2,
        "localObjective":config["localObjective"],"ebmMode":config["ebmMode"],"legacyLambdaIgnored":True,
        "researchValid":config["researchValid"],"config":config,"status":"running",
        "completedRounds":0,"channelNoiseSeed":config["seed"],
        "sigmaChoiceReason":"A-priori moderate middle calibration value; explicitly authorized before Test 2 outcomes",
        "metricDefinitions":{"full_test_accuracy":"End-of-ring Node 9 honest trained model on all 10000 test images, before outbound noise",
            "mean_node_accuracy":"Arithmetic mean of full-test accuracy of ten stored node trained states; no model averaging",
            "worst_node_accuracy":"Minimum full-test accuracy of ten stored node trained states",
            "full_test_loss":"Mean full-test CE of end-of-ring Node 9 model",
            "per_class_accuracy":"Node 9 class-conditional full-test accuracy"},
        "prePostChannelAccuracy":"not_measured; channel norms and hashes are recorded per link"}
    prior_runtime=float(previous.get('runtimeSeconds',0)) if previous else 0.
    print(json.dumps(dict(event='device_ready',**runtime)),flush=True)
    if campaign:metadata.update(iidCampaignId=config['iidCampaignId'],conditionId=config['conditionId'],
        actualAttackerCount=config['attackerCount'],executionStartMode=plan['mode'] if plan else 'fresh',resumeSupported=True)
    if previous:
        metadata.update(timestamp=previous['timestamp'],completedRounds=previous.get('completedRounds',0),
            recoveryProvenance=dict(originalImplementationSha256=previous['implementationSha256'],
                recoveryImplementationSha256=source_hashes,mode=plan['mode'],
                savedActivations=len(saved_records),startedAt=datetime.now(timezone.utc).isoformat()))
        if 'latestRound' in previous:metadata['latestRound']=previous['latestRound']
        for key in ('deviceTransition','executionDeviceSegments'):
            if key in previous:metadata[key]=previous[key]
        attempt=output/'recovery_attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        attempt.mkdir(parents=True)
        shutil.copy2(output/'run.json',attempt/'run_before_recovery.json')
        audit_path=attempt/'audit'
    else:audit_path=output/'audit'
    save_json(output/"run.json",metadata)
    save_json(output/"iid_partition_audit.json",audit)
    logger = logging.getLogger(name)
    handler = logging.FileHandler(output/"execution.log")
    logger.addHandler(handler); logger.setLevel(logging.INFO)
    started = time.perf_counter()
    if runtime['device']=='GPU':
        from basil_core.iid_gpu_worker import GpuProtocolAudit
        observer=GpuProtocolAudit(audit_path)
    else:observer = ProtocolAudit(audit_path)
    round_records=list(resume_state['telemetry'][resume_state['completed_rounds']*10:]) if resume_state else []
    round_rows=[]
    csv_path=output/"round_metrics.csv"
    has_csv=csv_path.exists() and csv_path.stat().st_size>0
    if previous:
        if has_csv:
            with csv_path.open(newline='') as source:
                round_rows=[{k:float(v) if v else None for k,v in r.items()} for r in csv.DictReader(source)]
        if [int(r['round']) for r in round_rows]!=list(range(len(round_rows))):
            raise ValueError('Saved round metrics are not contiguous.')
        log_path=output/'partial_activation_telemetry.jsonl'
        valid_text=''.join(json.dumps(r,allow_nan=False)+'\n' for r in saved_records)
        raw_text=log_path.read_text() if log_path.exists() else ''
        if raw_text and (len(raw_text.splitlines())!=len(saved_records) or not raw_text.endswith('\n')):
            shutil.copy2(log_path,attempt/'activation_log_before_recovery.jsonl')
            log_path.write_text(valid_text)
    csv_handle=csv_path.open("a" if previous else "w",newline="")
    fields=["round","full_test_accuracy","mean_node_accuracy","worst_node_accuracy","full_test_loss",
        *[f"class_{c}_accuracy" for c in range(10)],"mean_local_ce","honest_selection_rate",
        "byzantine_selection_rate","attacked_source_selection_rate","mean_snapshot_age",
        "ebm_penalty","ordinary_gradient_norm","ebm_correction_norm","ebm_correction_ratio",
        "channel_noise_norm","noise_to_model_norm_ratio"]
    csv_writer=csv.DictWriter(csv_handle,fieldnames=fields)
    if not previous or not has_csv:csv_writer.writeheader();csv_handle.flush()
    stopped = False
    def stop(_signum,_frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    cursor=resume_state['next_activation'] if resume_state else 0
    print(json.dumps({'restored':dict(progress=cursor/(config['nRounds']*10),
        round=resume_state['completed_rounds'] if resume_state else 0,
        priorRuntimeSeconds=prior_runtime,startProgress=cursor/(config['nRounds']*10),
        accuracyHistory=[(int(r['round'])+1,r['mean_node_accuracy'],r['worst_node_accuracy'])
            for r in round_rows[:resume_state['completed_rounds']]] if resume_state else [],
        activationHistory=[(r['round']+(r['nodeId']+1)/10,r['globalTestAccuracy'])
            for r in saved_records[:cursor]],
        recoveryTotal=len(saved_records),recoveryCompleted=cursor)}),flush=True)
    def checkpoint(state):
        save_checkpoint(output/'checkpoint.npz',config,state,
            partition_hash=audit['partitionHash'],initial_hash=initial_hash)
    def activation(record):
        round_records.append(record)
        index=record['round']*10+record['nodeId']
        if index<len(saved_records):
            verify_replayed_activation(record,saved_records[index])
            print(json.dumps({'recovery':dict(completed=index+1,total=len(saved_records))}),flush=True)
        else:
            with (output/"partial_activation_telemetry.jsonl").open("a") as handle:
                handle.write(json.dumps(record,allow_nan=False)+"\n")
        from gui.services.research_telemetry import activation_payload
        print(json.dumps(activation_payload(record,name)),flush=True)
    def round_complete(record):
        if "fullTestAccuracy" in record:
            steps=[s for a in round_records for s in a["optimizerTelemetry"]]
            links=[l for a in round_records for l in a["outgoingLinks"]]
            ages=[a["selectedSnapshotAgeRounds"] for a in round_records if a["selectedSnapshotAgeRounds"] is not None]
            mean=lambda values:float(np.mean(values)) if values else None
            row=dict(round=record["round"],full_test_accuracy=record["fullTestAccuracy"],
                mean_node_accuracy=record["averageAccuracy"],worst_node_accuracy=record["worstNodeAccuracy"],
                full_test_loss=record["fullTestLoss"],mean_local_ce=mean([a["localTrainingLoss"] for a in round_records]),
                honest_selection_rate=mean([not a["selectedSenderByzantine"] for a in round_records]),
                byzantine_selection_rate=mean([a["selectedSenderByzantine"] for a in round_records]),
                attacked_source_selection_rate=mean([a["selectedSenderByzantine"] and a["inputSnapshotRound"]>=config["attackStartRound"] for a in round_records]),
                mean_snapshot_age=mean(ages),ebm_penalty=mean([s["ebmPenalty"] for s in steps]),
                ordinary_gradient_norm=mean([s["baseGradientNorm"] for s in steps]),
                ebm_correction_norm=mean([s["ebmCorrectionNorm"] for s in steps]),
                ebm_correction_ratio=mean([s["ebmCorrectionRatio"] for s in steps]),
                channel_noise_norm=mean([l["noiseL2Norm"] for l in links]),
                noise_to_model_norm_ratio=mean([l["relativeNoiseL2"] for l in links]))
            row.update({f"class_{c}_accuracy":v for c,v in enumerate(record["perClassAccuracy"])})
            if record['round']<len(round_rows):
                if row!=round_rows[record['round']]:raise ValueError(f'Recovery round metrics diverged at {record["round"]}.')
            else:
                csv_writer.writerow(row);csv_handle.flush();round_rows.append(row)
            if record['round']+1>=metadata['completedRounds']:metadata['latestRound']=row
            metadata.update(completedRounds=max(metadata['completedRounds'],record["round"]+1),runtimeSeconds=prior_runtime+time.perf_counter()-started)
            save_json(output/"run.json",metadata)
            if not campaign:
                from reporting.iid_study_plots import write_study_summary
                write_study_summary()
        network_record={**record,'perNodeAccuracy':[r['globalTestAccuracy'] for r in round_records]}
        round_records.clear()
        logger.info("Round completed: %s",record)
        print(json.dumps({"test":name,"roundComplete":network_record}),flush=True)
    try:
        result = run_research_protocol(config,train_x,train_y,test_x,test_y,
            activation_callback=activation,round_callback=round_complete,audit=observer,
            worker_factory=make_worker,should_stop=lambda:stopped,resume_state=resume_state,
            checkpoint_callback=checkpoint if campaign else None)
        actual_partition = hashlib.sha256()
        for chunk in result["partitionIndices"]:
            actual_partition.update(np.asarray(chunk,dtype="<i8").tobytes())
        metadata.update(fullDatasetPartitionHash=audit["partitionHash"],
            iidPartitionHash=actual_partition.hexdigest(),
            trainingSamplesUsed=sum(len(c) for c in result["partitionIndices"]),
            localSampleCounts=[len(c) for c in result["partitionIndices"]],
            evaluationSamples=len(test_y),evaluationClassCounts=np.bincount(test_y,minlength=10).tolist())
        np.savez_compressed(output/"partition_indices.npz",**{f"node_{i}":c for i,c in enumerate(result["partitionIndices"])})
        metrics = {k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(output/"metrics.npz",**metrics)
        for node,params in enumerate(result["finalNodeParams"]):
            np.savez_compressed(output/f"node_{node}_weights.npz",**{f"weight_{i}":p for i,p in enumerate(params)})
        with gzip.open(output/"activation_telemetry.json.gz","wt") as handle:
            json.dump(result["telemetry"],handle,allow_nan=False)
        selections = result["telemetry"]
        steps = [s for r in selections for s in r["optimizerTelemetry"]]
        smoke = family == "smoke"
        expected_steps = 5 if smoke else 50
        checks = {"finiteTraining":all(np.isfinite(s["ebmObjective"]) for s in steps),
            "fiveFullEpochs":all(r["optimizerSteps"]==expected_steps and len(r["epochs"])==5 and
                all(e["samplesSeen"]==(512 if smoke else 5000) and e["batchSizes"][-1]==(512 if smoke else 392)
                    for e in r["epochs"]) for r in selections),
            "basilSelectionMinimum":all(abs(r["selectedCandidateLoss"]-min(c["receiverLocalLoss"] for c in r["candidates"]))<=1e-8 for r in selections),
            "rollingDistinctMemory":all(len(set(r["memorySenderIds"]))==5 for r in selections),
            "attackSchedule":all(r["attackActive"]==(r["configuredByzantine"] and r["round"]>=config["attackStartRound"]) for r in selections),
            "channelSchedule":all(r["channelNoiseActive"]==config["useChannelNoise"] for r in selections),
            "coefficient":all(np.isclose(s["ebmCoefficient"],config["channelNoiseSigmaAbsolute"]**2 if config['ebmMode']=='gradient_norm_objective' else 0.,rtol=1e-6) for s in steps),
            "accuracyAboveInitialization":float(result["averageAccuracy"][-1])>initial_accuracy}
        if campaign:
            failed=[key for key,value in checks.items() if key!='accuracyAboveInitialization' and not value]
            if failed:raise ValueError(f'Completed IID protocol checks failed: {failed}')
        metadata.update(status="completed",runtimeSeconds=prior_runtime+time.perf_counter()-started,
            completedRounds=config["nRounds"],
            finalAverageAccuracy=float(result["averageAccuracy"][-1]),finalWorstAccuracy=float(result["worstNodeAccuracy"][-1]),
            finalPerNodeAccuracy=result["roundNodeAccuracy"][-1].tolist(),
            finalPerClassAccuracy=result["roundNodePerClassAccuracy"][-1].mean(axis=0).tolist(),
            bestAverageAccuracy=float(result["averageAccuracy"].max()),bestRound=int(result["averageAccuracy"].argmax()),
            selectedByzantine=sum(r["selectedSenderByzantine"] for r in selections),
            selectedHonest=sum(not r["selectedSenderByzantine"] for r in selections),
            attackActiveActivations=sum(r["attackActive"] for r in selections),preflightChecks=checks,
            preflightPassed=all(checks.values()))
        if round_rows:
            metadata["roundAggregateStatistics"]={k:float(np.mean([r[k] for r in round_rows])) for k in fields
                if k not in {"round","mean_snapshot_age"} and not k.startswith("class_")}
            metadata["finalFullTestAccuracy"]=float(result["fullTestAccuracy"][-1])
            metadata["bestFullTestAccuracy"]=float(result["fullTestAccuracy"].max())
            metadata["bestFullTestRound"]=int(result["fullTestAccuracy"].argmax())
        save_json(output/"run.json",metadata)
        from reporting.iid_study_plots import generate_run_plots
        plot_output = iid_output(ROOT/config['plotDirectory'],plots=True) if campaign else PLOT_ROOT/name/(family if family != "research" else "")
        generate_run_plots(output,plot_output)
        if campaign:
            from reporting.iid_campaign_plots import report_campaign
            report_campaign(device=runtime['device'])
        return metadata
    except Exception as error:
        logger.exception("IID execution failed")
        save_json(output/"run.json",{**metadata,"status":"stopped" if stopped else "failed",
            "error":str(error),"runtimeSeconds":prior_runtime+time.perf_counter()-started})
        raise
    finally:
        observer.close();csv_handle.close(); handler.close(); logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-absolute",type=float,help="Explicit operator-approved sigma_e; no default")
    parser.add_argument("--preflight",action="store_true",help="Three full-data rounds; attack warm-up remains 20")
    parser.add_argument("--smoke",action="store_true",help="Three reduced-data rounds; attack starts at 1, researchValid=false")
    parser.add_argument("--run",action="store_true",help="Run both 100-round tests only after both preflights pass")
    parser.add_argument("--rounds",type=int,choices=(50,100),default=100)
    parser.add_argument("--replace-iid-outputs",action="store_true",help="Archive only the six current IID result/plot directories before replacing them")
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    configs = paired_configs(args.sigma_absolute,rounds=args.rounds)
    for config in configs: validate_config(config)
    if args.replace_iid_outputs:
        archive=Path(tempfile.mkdtemp(prefix="papermerge_iid_preliminary_"))
        for root in (RESULT_ROOT,PLOT_ROOT):
            for name in (*TEST_NAMES,"comparison"):
                target=root/name
                if target.is_symlink():raise ValueError(f"Refusing to archive a symlink: {target}")
                if target.exists():
                    destination=archive/root.parent.name/name;destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.move(str(target),str(destination))
        print("Previous IID directories archived recoverably at",archive,flush=True)
    prepare(configs)
    if not (args.preflight or args.run or args.smoke):
        print("Prepared exactly two configs. Sigma status:",configs[1]["sigmaApproval"]); return
    train,test = loadCifar10(cacheDir=ROOT/"experiments"/"cache"/"cifar10")
    data = (*_datasetArrays(train),*_datasetArrays(test))
    audit, indices = partition_audit(data[1],data[3])
    save_json(RESULT_ROOT/"comparison"/"iid_partition_audit.json",audit)
    np.savez_compressed(RESULT_ROOT/"comparison"/"iid_partition_indices.npz",**{f"node_{i}":v for i,v in enumerate(indices)})
    lines = ["node,total,"+",".join(f"class_{i}" for i in range(10))]
    lines += [",".join(map(str,[r["nodeId"],r["samples"],*r["classCounts"]])) for r in audit["nodes"]]
    (RESULT_ROOT/"comparison"/"iid_partition_audit.csv").write_text("\n".join(lines)+"\n")
    if args.run:
        initial_hashes=[];checks=[]
        for config in configs:
            worker=IidWorker(build_model(config["modelArchitecture"],config["seed"]))
            initial_hashes.append(params_hash(worker.export()))
            order=keyed_rng(config["seed"],"data_order",0,0,0).permutation(5000)[:512]
            chosen=indices[0][order]
            key=keyed_rng(config["seed"],"augmentation",0,0,0,0).integers(0,2**30,size=2,dtype=np.int32)
            grads,values=worker._gradients(tf.convert_to_tensor(data[0][chosen]),tf.convert_to_tensor(data[1][chosen],tf.int32),tf.convert_to_tensor(key),
                config["ebmMode"],"paper_absolute_gaussian",tf.constant(config["channelNoiseSigmaAbsolute"],tf.float32),tf.constant(0.,tf.float32))
            optimizer=tf.keras.optimizers.SGD(.05,momentum=0.)
            optimizer.apply_gradients(zip(grads,worker.model.trainable_variables))
            if any(not np.isfinite(p).all() for p in worker.export()):raise FloatingPointError("Non-finite minimal sanity update")
            checks.append(dict(test=config["experimentName"],initialHash=initial_hashes[-1],finiteFirstUpdate=True,
                ce=float(values[0]),coefficient=float(values[3]),correctionNorm=float(values[6]),variance=config["channelNoiseSigmaAbsolute"]**2))
        if initial_hashes[0]!=initial_hashes[1]:raise ValueError("Initial model hashes are not paired.")
        save_json(RESULT_ROOT/"comparison"/"minimal_sanity_checks.json",{"checks":checks,"partitionHash":audit["partitionHash"],"intentionalDifferences":verify_pair(*configs)})
        failures=[]
        for config in configs:
            try:execute(config,data,audit,family="research")
            except Exception as error:
                failures.append({"test":config["experimentName"],"error":str(error)})
                print("Affected run failed:",failures[-1],flush=True)
            from reporting.iid_study_plots import write_study_summary
            write_study_summary()
        if not failures:
            from reporting.iid_study_plots import generate_comparison
            generate_comparison([RESULT_ROOT/n for n in TEST_NAMES],PLOT_ROOT/"comparison")
        if failures:raise SystemExit(1)
        return
    family = "smoke" if args.smoke else "preflight"
    outcomes = []
    for config in configs:
        if config["channelNoiseSigmaAbsolute"] is None:
            print("Test 2 blocked: select an explicitly approved sigma_e from the calibrated candidates",CALIBRATED_SIGMAS)
            continue
        outcomes.append(execute(preflight_config(config,smoke=args.smoke),data,audit,family=family))
    save_json(RESULT_ROOT/"comparison"/f"{family}_outcomes.json",outcomes)
    if len(outcomes)==2:
        from reporting.iid_study_plots import generate_comparison
        generate_comparison([RESULT_ROOT/n/family for n in TEST_NAMES],PLOT_ROOT/"comparison"/family)
    from reporting.iid_study_plots import write_study_summary
    write_study_summary()


if __name__ == "__main__":
    main()
