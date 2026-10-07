#!/usr/bin/env python3
"""Run one research-protocol config and save outside protected artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys
import time
import signal
import logging
import hashlib
import os

# Configure the research worker before TensorFlow is imported. Historical
# runners retain their own runtime policy; explicit environment overrides win.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))

from basil_core.data.cifar import _datasetArrays, loadCifar10
from basil_core.research_protocol import run_research_protocol, validate_config, params_hash, build_model
from basil_core.artifact_paths import writable_output
from gui.research_protocol import RESULT_ROOT
from basil_core.protocol_compatibility import IID_PROTOCOL, normalize_config


def atomic_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--result-root",default=str(RESULT_ROOT)); parser.add_argument("--gpu-memory-mb",type=int,default=0); parser.add_argument("--allow-production",action="store_true",help="Explicit approval after reviewing the baseline gate")
    parser.add_argument("--audit-directory")
    parser.add_argument("--audit-candidates",action="store_true")
    parser.add_argument("--deterministic",action="store_true")
    args=parser.parse_args()
    config=normalize_config(json.loads(Path(args.config).read_text(encoding="utf-8"))); validate_config(config)
    if config["experimentProtocol"] == IID_PROTOCOL:
        raise SystemExit("Use scripts/run_iid_basil_pair.py for paired IID output routing and preflight gates.")
    if config.get("researchValid",True) and not args.allow_production:
        raise SystemExit("Production is gated. Review the clean baseline first; explicit --allow-production is required.")
    import tensorflow as tf
    # CPU parallel gradient reductions still differed by a few ULPs with
    # op determinism enabled. Use one ordered CPU lane for paired research.
    cpu_threads = 4 if tf.config.list_physical_devices("GPU") else 1
    tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    for device in tf.config.list_physical_devices("GPU"):
        if args.gpu_memory_mb:
            tf.config.set_logical_device_configuration(device,[tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_memory_mb)])
        else: tf.config.experimental.set_memory_growth(device,True)
    family="production" if config.get("researchValid",True) else "preflight" if config.get("purpose")=="full_data_preflight" else "smoke"
    output=writable_output(Path(args.result_root)/family/config["runId"])
    if output.exists(): raise SystemExit(f"Result already exists: {output}")
    output.mkdir(parents=True)
    logging.basicConfig(filename=output/"execution.log",level=logging.INFO)
    started=time.perf_counter(); started_at=datetime.now().astimezone().isoformat()
    stopped=False
    def request_stop(_signal,_frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    code_digest=hashlib.sha256()
    for source in ("basil_core/research_protocol.py","basil_core/models.py","basil_core/attacks.py","basil_core/data/cifar.py","basil_core/trainer.py","scripts/run_research_protocol.py"):
        code_digest.update(source.encode());code_digest.update((PROJECT_ROOT/source).read_bytes())
    initial_metadata={"schemaVersion":5,"runId":config["runId"],"experimentProtocol":config["experimentProtocol"],"config":config,"researchValid":config["researchValid"],"status":"preparing","startedAt":started_at,"implementationSha256":code_digest.hexdigest(),"runtimeEnvironment":{"tensorflow":tf.__version__,"numpy":np.__version__,"intraOpThreads":4,"interOpThreads":2,"devices":[d.device_type for d in tf.config.list_physical_devices()]}}
    initial_metadata["runtimeEnvironment"].update(intraOpThreads=cpu_threads, interOpThreads=1,
        oneDnn=os.environ["TF_ENABLE_ONEDNN_OPTS"],
        opDeterminism=True, normAccumulation="float64", tensorDtype="float32")
    atomic_json(output/"run.json",initial_metadata)
    try:
        train,test=loadCifar10(cacheDir=PROJECT_ROOT/"experiments"/"cache"/"cifar10"); train_x,train_y=_datasetArrays(train); test_x,test_y=_datasetArrays(test)
    except Exception as error:
        logging.exception("CIFAR data preparation failed")
        atomic_json(output/"run.json",{**initial_metadata,"status":"failed","error":str(error)})
        raise SystemExit("CIFAR data preparation failed. See the execution log.")
    if config.get("diagnosticTestSamplesPerClass"):
        amount=int(config["diagnosticTestSamplesPerClass"]); chosen=np.concatenate([np.flatnonzero(test_y==class_id)[:amount] for class_id in range(10)]); test_x,test_y=test_x[chosen],test_y[chosen]
    prefix="@@CAMPAIGN_EVENT@@"
    print(prefix+json.dumps({"event":"started","runId":config["runId"]}),flush=True)
    def activation_event(record):
        from gui.services.research_telemetry import node_event
        with (output/"partial_activation_telemetry.jsonl").open("a",encoding="utf-8") as trace:
            trace.write(json.dumps(record,separators=(",",":"))+"\n")
        print(prefix+json.dumps({"event":"activation_complete","round":record["round"],"nodeId":record["nodeId"],"globalTestAccuracy":record["globalTestAccuracy"]}),flush=True)
        print(prefix+json.dumps({"event":"node_update",**node_event(record)}),flush=True)
    audit = None
    if args.audit_directory:
        if config.get("researchValid",True):
            raise SystemExit("Detailed audits are restricted to non-research diagnostics.")
        from basil_core.research_audit import ProtocolAudit
        audit = ProtocolAudit(args.audit_directory, candidate_evaluation=args.audit_candidates)
    try:
        result=run_research_protocol(config,train_x,train_y,test_x,test_y,
            activation_callback=activation_event,
            round_callback=lambda event:print(prefix+json.dumps({"event":"round_complete",**event}),flush=True),should_stop=lambda:stopped,audit=audit)
    except Exception as error:
        status="stopped" if isinstance(error,InterruptedError) else "failed"
        logging.exception("Run %s %s",config["runId"],status)
        atomic_json(output/"run.json",{**initial_metadata,"status":status,"error":str(error),"runtimeSeconds":time.perf_counter()-started})
        message="Non-finite training value; see execution.log." if "Non-finite" in str(error) else str(error).splitlines()[0]
        print(prefix+json.dumps({"event":status,"runId":config["runId"],"message":message}),flush=True)
        raise SystemExit(0 if status=="stopped" else 1)
    finally:
        if audit:
            audit.close()
    try:
        metrics={key:value for key,value in result.items() if isinstance(value,np.ndarray)}; np.savez_compressed(output/"metrics.npz",**metrics)
        for node_id,params in enumerate(result["finalNodeParams"]): np.savez_compressed(output/f"node_{node_id}_weights.npz",**{f"weight_{index}":value for index,value in enumerate(params)})
        with gzip.open(output/"activation_telemetry.json.gz","wt",encoding="utf-8") as handle: json.dump(result["telemetry"],handle,separators=(",",":"))
    except Exception as error:
        logging.exception("Result persistence failed for %s",config["runId"])
        atomic_json(output/"run.json",{**initial_metadata,"status":"failed","errorCategory":"persistence","error":str(error)})
        print(prefix+json.dumps({"event":"failed","message":"Result persistence failed; see execution.log."}),flush=True)
        raise SystemExit(1)
    metadata={**initial_metadata,"status":"completed","resolvedAttackerIds":result["resolvedAttackerIds"].tolist(),"runtimeSeconds":time.perf_counter()-started,"finalAverageAccuracy":float(result["averageAccuracy"][-1]),"finalWorstAccuracy":float(result["worstNodeAccuracy"][-1]),"finalBestAccuracy":float(result["bestNodeAccuracy"][-1]),"initializationHash":params_hash([v.numpy() for v in build_model(config["modelArchitecture"],config["seed"]).trainable_variables]),"rngDerivation":"SHA256(seed|stream|round|sender|receiver); independent keyed streams","trainClassCounts":np.bincount(train_y,minlength=10).tolist(),"evaluationClassCounts":np.bincount(test_y,minlength=10).tolist(),"artifactPaths":{"metrics":str(output/"metrics.npz"),"telemetry":str(output/"activation_telemetry.json.gz"),"nodeWeights":str(output/"node_*_weights.npz")}}
    atomic_json(output/"run.json",metadata); print(prefix+json.dumps({"event":"completed","runId":config["runId"],"finalAverageAccuracy":metadata["finalAverageAccuracy"],"finalWorstAccuracy":metadata["finalWorstAccuracy"]}),flush=True); print(json.dumps(metadata,indent=2))


if __name__=="__main__":main()
