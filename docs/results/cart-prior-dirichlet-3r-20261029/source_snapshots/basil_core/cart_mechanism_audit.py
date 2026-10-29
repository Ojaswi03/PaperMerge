"""Read-only metric, partition and activation diagnostics with explicit limits."""
from collections import Counter
import json
from pathlib import Path


def metrics_summary(accuracy,per_class=None):
    import numpy as np
    a=np.asarray(accuracy,dtype=float)
    if a.ndim!=2 or not a.size or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):
        raise ValueError('Expected finite round-by-client accuracy in [0,1].')
    curve=a.mean(axis=1);start=min(29,len(curve)-1)
    result=dict(final=float(curve[-1]),primary=float(a[-min(10,len(a)):].mean()),
        round20=float(curve[19]) if len(curve)>=20 else None,
        gainAfterRound30=float(curve[-1]-curve[start]) if len(curve)>30 else None,
        worstClient=float(a[-1].min()),clientP10=float(np.quantile(a[-1],.1)),
        medianClient=float(np.median(a[-1])),clientsAtOrBelow10_5Percent=int((a[-1]<=.105).sum()))
    if per_class is not None:
        pc=np.asarray(per_class,dtype=float)
        if pc.shape[:2]!=a.shape or pc.ndim!=3 or not np.isfinite(pc).all() or np.any(pc<0) or np.any(pc>1):
            raise ValueError('Invalid round/client/class recall array.')
        result['finalMeanClassRecall']=pc[-1].mean(axis=0).tolist()
    return result


def partition_summary(audit):
    import numpy as np
    h=np.asarray(audit['classCounts'],dtype=float)
    if h.ndim!=2 or np.any(h<0) or np.any(h.sum(axis=1)<=0):raise ValueError('Invalid owner class histogram.')
    p=h/h.sum(axis=1,keepdims=True)
    entropy=-(p*np.log(np.maximum(p,1e-300))).sum(axis=1)/np.log(h.shape[1])
    return dict(partitionHash=audit['partitionHash'],holdoutHash=audit['holdoutHash'],
        meanNormalizedClassEntropy=float(entropy.mean()),meanDominantClassFraction=float(p.max(axis=1).mean()),
        meanClassesPresent=float((h>0).sum(axis=1).mean()),minimumClassesPresent=int((h>0).sum(axis=1).min()),
        minimumLocalSamples=int(h.sum(axis=1).min()),maximumLocalSamples=int(h.sum(axis=1).max()))


def telemetry_summary(records,nodes):
    """Aggregate routes and committed-step scales by round, never infer ancestry."""
    buckets={}
    for i,r in enumerate(records):
        if (r['round'],r['node'],r['activation'])!=(i//nodes,i%nodes,i+1):
            raise ValueError('Incomplete or unordered activation telemetry.')
        row=buckets.setdefault(r['round'],dict(displayRound=r['round']+1,activations=0,routes=Counter(),
            commitScales=Counter(),commitChecks=0,qualifiedClassCounts=[],selectedCounts=[],
            rlaCoefficients=[],localTrainCE=[],calibrationDraws=0))
        row['activations']+=1;s=r['selection'];c=s.get('continuity',{});t=r.get('training',{})
        row['routes'][c.get('actualRoute','pool')]+=1
        row['selectedCounts'].append(s['selectedCount']);row['qualifiedClassCounts'].append(len(c.get('qualifiedClasses',[])))
        row['calibrationDraws']+=c.get('calibrationDraws',0)
        qc=(r.get('cart') or {}).get('qualityCommit',{})
        if qc.get('applied'):
            row['commitChecks']+=1;row['commitScales'][str(qc['scale'])]+=1
        for name,target in [('rlaCoefficient','rlaCoefficients'),('localTrainingLoss','localTrainCE')]:
            if t.get(name) is not None:row[target].append(float(t[name]))
    for row in buckets.values():
        for name in ('qualifiedClassCounts','selectedCounts','rlaCoefficients','localTrainCE'):
            values=row.pop(name);row['mean'+name[0].upper()+name[1:]]=sum(values)/len(values) if values else None
        row['routes']=dict(row['routes']);row['commitScales']=dict(row['commitScales'])
        row['ownRouteFraction']=row['routes'].get('own',0)/row['activations']
        row['zeroCommitFractionAmongChecks']=row['commitScales'].get('0.0',0)/row['commitChecks'] if row['commitChecks'] else None
    return list(buckets.values())


def inspect(root):
    import numpy as np
    root=Path(root);plan=json.loads((root/'plan.json').read_text());results=[]
    for view in plan['configs']:
        folder=root/view['runId'];run_path=folder/'run.json'
        if not run_path.exists():continue
        run=json.loads(run_path.read_text());cfg=view.get('comparisonCase',{}).get('protocolConfig',view)
        row=dict(runId=view['runId'],dataset=cfg['dataset'],partition=cfg['partition'],seed=cfg['seed'],
            condition=cfg['condition'],arm=view.get('comparisonCase',{}).get('arm'),status=run['status'])
        if (folder/'metrics.npz').exists():
            with np.load(folder/'metrics.npz',allow_pickle=False) as a:
                row['metrics']=metrics_summary(a['roundNodeAccuracy'],a['roundNodePerClassAccuracy'] if 'roundNodePerClassAccuracy' in a else None)
        if run.get('partition'):row['partitionAudit']=partition_summary(run['partition'])
        if (folder/'mechanisms.json').exists():
            saved=json.loads((folder/'mechanisms.json').read_text())
            if saved.get('source')!=run['source'] or saved.get('runId')!=view['runId']:raise ValueError('Mechanism receipt identity mismatch.')
            row['mechanisms']=saved['rounds']
        elif (folder/'checkpoint.npz').exists():
            with np.load(folder/'checkpoint.npz',allow_pickle=False) as a:meta=json.loads(a['metadata'].tobytes())
            if meta['contract']!=run['contract']:raise ValueError('Checkpoint and run contract differ.')
            row['mechanisms']=telemetry_summary(meta['telemetry'],cfg['nodes'])
        else:row['mechanisms']=None
        results.append(row)
    return dict(revision=plan.get('revision'),runs=results,
        limitation='Metrics are development-holdout observations. Missing telemetry cannot establish routing/commit causes; no independent inference is performed.')
