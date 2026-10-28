#!/usr/bin/env python3
"""Recover a complete diagnostic's summary without changing/retraining models.

Handles the early fresh-hop diagnostic's all-own route: mean selected peer hop
is undefined, so it is recorded as null. Original source identities are kept.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def recover(root):
    import numpy as np
    from scripts.probe_cart_fresh_continuity import intervention_hashes
    from basil_core.sharp_cart_v2_protocol import check_source
    from scripts.run_cart_v2_study import write_json
    root=Path(root);plan=json.loads((root/'plan.json').read_text());check_source(plan['source'])
    if plan['interventionProgramSha256']!=intervention_hashes():raise ValueError('Diagnostic code identity changed.')
    recovered=[];cases=[]
    for case in plan['cases']:
        folder=root/case['name'];target=folder/'result.json'
        if not target.exists():
            with np.load(folder/'checkpoint.npz',allow_pickle=False) as arrays:state=json.loads(arrays['metadata'].tobytes())
            contract=state['contract'];cfg=case['config']
            if (contract['diagnostic']!=case or contract['source']!=plan['source'] or
                    contract['interventionProgramSha256']!=plan['interventionProgramSha256'] or
                    state['cursor']!=cfg['nodes']*cfg['rounds'] or len(state['roundMetrics'])!=cfg['rounds']):
                raise ValueError('Only complete, identically frozen diagnostic state can be summarized.')
            records=state['telemetry'][cfg['nodes']:];routing=[r['selection'].get('continuity',{}) for r in records]
            hops=[(r['node']-s)%cfg['nodes'] for r in records for s in r['selection']['selectedSenders']]
            last=state['roundMetrics'][-1]
            row=dict(case=case,status='COMPLETED_DIAGNOSTIC',contract=contract,
                meanAccuracy=last['meanAccuracy'],worstNodeAccuracy=last['worstNodeAccuracy'],roundMetrics=state['roundMetrics'],
                meanSelectedCount=float(np.mean([r['selection']['selectedCount'] for r in records])),
                meanSelectedHop=float(np.mean(hops)) if hops else None,
                meanLastRoundTrainCE=float(np.mean([r['training']['localTrainingLoss'] for r in state['telemetry'][-cfg['nodes']:]])),
                elapsedSeconds=state['elapsedSeconds'],officialTestUsed=False,
                ownRoutes=sum(r.get('route')=='own' for r in routing),peerRoutes=sum(r.get('route')=='peer' for r in routing),
                classBoundChecks=sum(r.get('empiricalInitializationBound',False) for r in routing),
                calibrationDraws=sum(r.get('calibrationDraws',0) for r in routing),
                actualRouteCounts={name:sum(r.get('actualRoute')==name for r in routing) for name in ('fresh','pool','own')},
                summaryRecovery=dict(reason='Post-training JSON export could not encode undefined mean peer hop for all-own routes.',
                    programSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),trainingModified=False),
                interpretation='Single-seed diagnostic; no successful robustness or novelty claim.')
            write_json(target,row);recovered.append(case['name'])
        cases.append(json.loads(target.read_text()))
    result=dict(status='COMPLETE_TRAINING_WITH_RECOVERED_SUMMARY',recovered=recovered,cases=cases)
    target=root/'recovered_summary.json'
    if target.exists():raise FileExistsError('Recovery already recorded; preserve the existing receipt.')
    write_json(target,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(dict(recovered=recover(a.output)['recovered'],trainingModified=False)))
