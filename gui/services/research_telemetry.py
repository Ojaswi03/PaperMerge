"""Read-only bridge from activation records to the existing ring renderer."""
import gzip
import json
from pathlib import Path
import numpy as np
from basil_core.protocol_compatibility import normalize_config


def node_event(record):
    node=record["nodeId"];source=record["inputSourceNode"]
    return {"round":record["round"]+1,"nodeId":node,
        "attack":{"configured":record["configuredByzantine"],"active":record["attackActive"],"relativeNorm":record["attackRelativeParameterChange"]},
        "incoming":{"senderId":source,"sourceRound":record["inputSnapshotRound"]},
        "selection":{"candidates":[{"senderId":c["senderId"],"distance":(node-c["senderId"])%10,"loss":c["receiverLocalLoss"],"plausible":True,"selected":c["senderId"]==source} for c in record["candidates"]]},
        "training":{"stateSignature":record["trainedOutputHash"],"momentumMode":"reset_each_activation","momentumNorm":0.,"baseGradientNorm":record["baseGradientNorm"],"preclipGradientNorm":record["robustObjectiveGradientNorm"],"clipFraction":0.,"optimizerSteps":record["optimizerSteps"]},
        "ebm":{"mode":record["ebmMode"],"coefficient":record["ebmCoefficient"],"effectiveCoordinateSigma":record["effectiveCoordinateSigma"]},
        "outgoing":[{"receiverId":link["receiverId"],"noiseNorm":link["noiseL2Norm"],"relativeNoise":link["relativeNoiseL2"],"coordinateSigma":link["coordinateSigma"]} for link in record["outgoingLinks"]],
        "globalTestAccuracy":record["globalTestAccuracy"],"perClassGlobalTestAccuracy":record["perClassGlobalTestAccuracy"]}


def activation_payload(record,experiment_name):
    return {'test':experiment_name,'round':record['round'],'node':record['nodeId'],
            'accuracy':record['globalTestAccuracy'],'networkUpdate':node_event(record)}


def load_replay(run_path):
    root=Path(run_path).parent
    metadata=json.loads(Path(run_path).read_text())
    with gzip.open(root/"activation_telemetry.json.gz","rt") as source:records=json.load(source)
    with np.load(root/"metrics.npz",allow_pickle=False) as saved:
        metrics={key:saved[key].copy() for key in saved.files}
    metrics.update(avg_history=metrics["averageAccuracy"],worst_history=metrics["worstNodeAccuracy"],per_node_history=metrics["roundNodeAccuracy"])
    return {"config":normalize_config(metadata["config"]),"metrics":metrics,"telemetry":{},"researchRecords":{(r["round"],r["nodeId"]):node_event(r) for r in records}}
