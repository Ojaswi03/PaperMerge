"""Read-only, source-linked metrics for mixed-attack continuation host fragments."""
import csv
import json
from pathlib import Path

import numpy as np

from basil_core import round_continuation as continuation
from basil_core import basil_sharp_thesis as study


def collect(root):
    root=Path(root).resolve();plan=continuation.read(root/'plan.json')
    if not plan.get('continuations'):raise ValueError('Expected an explicit continuation plan.')
    hashes={'plan.json':continuation.sha(root/'plan.json')};rows=[];seen=set()
    for name,expected in plan['source'].items():
        path=root/'source_snapshots'/name
        if path.is_symlink() or not path.resolve().is_relative_to(root/'source_snapshots') or continuation.sha(path)!=expected:
            raise ValueError('Continuation source snapshot changed: '+name)
        hashes['source_snapshots/'+name]=expected
    for view in plan['configs']:
        run=view['runId'];cfg=study.effective_config(view);item=plan['continuations'][run]
        if run in seen or Path(run).name!=run:raise ValueError('Duplicate/unsafe continuation run.')
        seen.add(run);folder=root/run;record=continuation.read(folder/'run.json')
        parent=continuation.source_record(item)
        if continuation.extended_view(parent['config'],cfg['rounds'])!=view:
            raise ValueError('Continuation numerical configuration changed.')
        if (record['status']!='COMPLETED' or record['config']!=view or record['source']!=plan['source']
                or record['completedActivations']!=cfg['nodes']*cfg['rounds']
                or record['optimizerSteps']!=cfg['nodes']*cfg['rounds']*cfg['localSteps']
                or record['data']!=parent['data'] or record['partition']!=parent['partition']
                or record['initialHash']!=parent['initialHash']):raise ValueError('Incomplete/mismatched continuation.')
        receipt=continuation.read(folder/'continuation.json')
        if (receipt['sourceCheckpointSha256']!=item['seedSha256'] or receipt['targetRunId']!=run
                or not receipt['prefixMetricsExactlyPreserved'] or receipt['targetRuntime']!=record['runtime']):
            raise ValueError('Continuation import provenance mismatch.')
        with np.load(folder/'metrics.npz',allow_pickle=False) as archive:
            a=archive['roundNodeAccuracy'];pc=archive['roundNodePerClassAccuracy'];honest=archive['roundHonestMeanAccuracy']
            if a.shape!=(cfg['rounds'],cfg['nodes']) or pc.shape!=(*a.shape,cfg['classes']) or honest.shape!=(cfg['rounds'],):
                raise ValueError('Continuation metric shape differs.')
            for values in (a,pc,honest):
                if not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid continuation accuracy.')
            np.testing.assert_allclose(a,pc.mean(axis=2),rtol=0,atol=1e-12)
            original=study.ROOT/item['sourceArchive']/item['sourceRunId']/'metrics.npz'
            if continuation.sha(original)!=item['sourceMetricsSha256']:raise ValueError('Parent metrics changed.')
            with np.load(original,allow_pickle=False) as before:
                for name in ('roundNodeAccuracy','roundNodePerClassAccuracy','roundHonestMeanAccuracy'):
                    np.testing.assert_array_equal(archive[name][:item['sourceRounds']],before[name])
            row=dict(dataset=cfg['dataset'],partition=cfg['partition'],seed=cfg['seed'],
                variant=view['comparisonCase']['arm'],attackType=cfg['attackType'],
                environment=view['comparisonCase']['environment'],runId=run,curve=a.mean(axis=1),
                primaryAccuracyPercent=float(a[-10:].mean()*100),finalAccuracyPercent=float(a[-1].mean()*100),
                sourcePrimaryAccuracyPercent=float(a[item['sourceRounds']-10:item['sourceRounds']].mean()*100))
        for name in ('run.json','metrics.npz','mechanisms.json','continuation.json'):
            hashes[run+'/'+name]=continuation.sha(folder/name)
        rows.append(row)
    if set(plan['continuations'])!=seen:raise ValueError('Continuation source mapping differs from target coverage.')
    if (root/'SHA256.json').exists():
        saved=continuation.read(root/'SHA256.json')
        if any(saved.get(name)!=value for name,value in hashes.items()):raise ValueError('Export checksums differ.')
    return plan,rows,hashes


def generate(root,destination):
    plan,rows,inputs=collect(root);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    fields=['dataset','partition','seed','variant','attackType','environment','runId',
        'primaryAccuracyPercent','finalAccuracyPercent','sourcePrimaryAccuracyPercent']
    with (destination/'seed_endpoints.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:row[k] for k in fields})
    summary=dict(configurations=len(rows),host=plan['host'],continuationQueueId=plan['continuationQueueId'],
        finalRounds=sorted({len(r['curve']) for r in rows}),prefixMetricsExactlyPreserved=True,
        interpretation='Server fragment, not an independent full-matrix study. Combine disjoint host exports before three-seed comparisons.',
        inputSha256=inputs,svgFiles=[])
    (destination/'continuation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
