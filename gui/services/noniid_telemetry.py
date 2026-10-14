"""Read-only replay of measured verified CART activation archives."""
import json
from pathlib import Path


def node_event(config, activation):
    selection=activation['selection'];final=activation['epochs'][-1]
    node=activation['node'];round_id=activation['round']
    return dict(round=round_id+1,nodeId=node,globalTestAccuracy=final['accuracy'],
        perClassGlobalTestAccuracy=final['perClass'],
        incoming=dict(senderId=selection['selectedSender'],sourceRound=selection['selectedSnapshotRound'],
            attacked=selection['selectedActuallyCorrupted'],stateSignature=selection['selectedHash']),
        selection=dict(candidates=[dict(senderId=c['senderId'],loss=c['balancedTrainingLoss'],
            distance=(node-c['senderId'])%10,selected=c['senderId']==selection['selectedSender'],
            plausible=True,snapshotRound=c['snapshotRound']) for c in selection['candidates']]),
        attack=dict(configured=node in config['resolvedAttackerIds'],active=node in config['resolvedAttackerIds'] and round_id>=20),
        outgoing=[dict(receiverId=l['receiver'],relativeNoise=l['noise']['relativeNoiseL2'],
            noiseNorm=l['noise']['noiseL2Norm'],coordinateSigma=l['noise'].get('coordinateSigma',config['channelNoiseSigmaAbsolute']),
            attackActive=l['attackActive']) for l in activation['outgoing']],
        training=dict(localEpochs=5,optimizerSteps=50),
        ebm=dict(mode=config['ebmMode'],coefficient=config['channelNoiseSigmaAbsolute']**2 if config['conditionId']=='D' else 0.),
        cart=dict(mu=.01),telemetryComplete=True)


def load_replay(run_path):
    import numpy as np
    run_path=Path(run_path);metadata=json.loads(run_path.read_text());config=metadata['config']
    rows=metadata['roundMetrics'];records={}
    if not rows:raise ValueError('No fully measured round is available for replay.')
    for r in rows:
        for node in range(10):
            path=run_path.parent/f"activation_{r['round']*10+node:04d}.json"
            records[(r['round'],node)]=node_event(config,json.loads(path.read_text()))
    return dict(config=config,telemetry={},researchRecords=records,
        metrics=dict(avg_history=np.asarray([r['mean_node_accuracy'] for r in rows]),
            worst_history=np.asarray([r['worst_node_accuracy'] for r in rows]),
            per_node_history=np.asarray([r['per_node_accuracy'] for r in rows])))
