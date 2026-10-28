#!/usr/bin/env python3
"""Read-only full queue, manager health and equal-round CART/control comparisons.

Reads JSON telemetry only; never loads TensorFlow or model checkpoints.
Live comparisons are exploratory, not final research endpoints.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core.cart_campaign_io import read_json,retry_io


def optional(path,default=None):
    try:return read_json(path)
    except FileNotFoundError:return default


def health(path,payload,host):
    if not path.exists():return dict(status='MISSING')
    pid=payload.get('ownerPid');alive=None
    if host==socket.gethostname() and type(pid) is int:
        try:os.kill(pid,0);alive=True
        except ProcessLookupError:alive=False
        except PermissionError:alive=True
    return dict(status=payload.get('status'),pid=pid,pidExistsOnThisHost=alive,
        modifiedUtc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
        ageSeconds=max(0.,time.time()-path.stat().st_mtime),error=payload.get('error'))


def history(root,cfg,manifest):
    rows=optional(root/cfg['runId']/'round_metrics.json',[])
    limit=min(len(rows),manifest.get('completedActivations',0)//cfg['nodes'])
    rows=rows[:limit]
    if len(rows)>cfg['rounds']:raise ValueError('Round history exceeds the frozen horizon.')
    for i,row in enumerate(rows):
        values=row['nodeAccuracy']
        if (row['round']!=i or len(values)!=cfg['nodes'] or
            any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in values)
            or abs(statistics.mean(values)-row['meanAccuracy'])>1e-12):
            raise ValueError('Invalid saved round metrics: '+cfg['runId'])
    return rows


def identity(cfg):return cfg['dataset'],cfg['partition'],cfg['seed'],cfg['sigma'],cfg['condition']


def paired(plan,manifests,curves):
    configs=plan['configs'];lookup={identity(c):c for c in configs};pairs=[];waiting=[]
    if len(lookup)!=len(configs):raise ValueError('Ambiguous repeated configuration identities.')
    strip=lambda c:{k:v for k,v in c.items() if k not in
        ('condition','cart','switches','runId','baseRunId','guardedRunId')}
    flags=lambda c:{k:v for k,v in c['switches'].items() if k not in ('name','cart')}
    for cart in configs:
        if not cart['cart']:continue
        key=identity(cart);control=lookup.get((*key[:-1],cart['condition'][:-5]))
        if not control:
            waiting.append(dict(runId=cart['runId'],reason='Control absent from plan'));continue
        if strip(cart)!=strip(control) or flags(cart)!=flags(control):
            raise ValueError('CART and control differ beyond CART.')
        a,b=manifests.get(cart['runId']),manifests.get(control['runId'])
        if not a or not b or not curves.get(cart['runId']) or not curves.get(control['runId']):
            waiting.append(dict(runId=cart['runId'],reason='Waiting for both saved round histories'));continue
        for cfg,row in ((cart,a),(control,b)):
            if (row.get('config')!=cfg or row.get('source')!=plan['source'] or
                row.get('data')!=plan['data'][cfg['dataset']] or row.get('partition')!=plan['partitions'][cfg['runId']]):
                raise ValueError('Live manifest differs from frozen plan: '+cfg['runId'])
        if any(a[k]!=b[k] for k in ('initialHash','data','partition')):
            raise ValueError('CART/control initialization or data are not paired.')
        if any(a['runtime'].get(k)!=b['runtime'].get(k) for k in ('tensorflow','numpy','device','tf32')):
            raise ValueError('CART/control numerical runtime differs.')
        count=min(len(curves[cart['runId']]),len(curves[control['runId']]))
        pairs.append(dict(dataset=cart['dataset'],partition=cart['partition'],seed=cart['seed'],sigma=cart['sigma'],
            cartRunId=cart['runId'],controlRunId=control['runId'],latestCommonRound=count))
    def at(pair,rounds):
        a=curves[pair['cartRunId']][rounds-1];b=curves[pair['controlRunId']][rounds-1]
        clean=lookup.get((pair['dataset'],pair['partition'],pair['seed'],0.,'clean'))
        clean_rows=curves.get(clean['runId'],[]) if clean else []
        if clean_rows:
            cm=manifests[clean['runId']];reference=manifests[pair['cartRunId']]
            if any(cm[k]!=reference[k] for k in ('initialHash','data','partition')):
                raise ValueError('Clean reference initialization/data are not paired.')
        clean_mean=clean_rows[rounds-1]['meanAccuracy'] if len(clean_rows)>=rounds else None
        return dict(pair,round=rounds,cartMean=a['meanAccuracy'],controlMean=b['meanAccuracy'],
            deltaPercentagePoints=100*(a['meanAccuracy']-b['meanAccuracy']),
            cartWorst=min(a['nodeAccuracy']),controlWorst=min(b['nodeAccuracy']),cleanMean=clean_mean)
    groups=[]
    for key in sorted({(p['dataset'],p['partition'],p['sigma']) for p in pairs}):
        chosen=sorted([p for p in pairs if (p['dataset'],p['partition'],p['sigma'])==key],key=lambda p:p['seed'])
        if [p['seed'] for p in chosen]!=[2025,2026,2027]:continue
        common=min(p['latestCommonRound'] for p in chosen);rows=[at(p,common) for p in chosen]
        values=[r['deltaPercentagePoints'] for r in rows]
        groups.append(dict(dataset=key[0],partition=key[1],sigma=key[2],round=common,seeds=rows,
            cartMean=statistics.mean(r['cartMean'] for r in rows),controlMean=statistics.mean(r['controlMean'] for r in rows),
            deltaPercentagePoints=statistics.mean(values),sampleStdPercentagePoints=statistics.stdev(values),
            cleanMean=statistics.mean(r['cleanMean'] for r in rows) if all(r['cleanMean'] is not None for r in rows) else None))
    return [at(p,p['latestCommonRound']) for p in pairs],groups,waiting


def inspect(root):
    from basil_core.sharp_monitor import snapshot
    from basil_core.sharp_cart_v2_protocol import check_source,validate_config
    root=Path(root).resolve();plan=read_json(root/'plan.json');check_source(plan['source'])
    for cfg in plan['configs']:validate_config(cfg)
    state=retry_io(snapshot,root);campaign=optional(root/'campaign.json',{})
    managed=optional(root/'managed_status.json',{});manifests={};curves={};warnings=[]
    needed={c['runId'] for c in plan['configs'] if c['cart'] or c['condition'] in ('attack_noise_defense_rla','clean')}
    for entry in state['entries']:
        cfg=entry['config'];row=optional(root/cfg['runId']/'run.json',{})
        manifests[cfg['runId']]=row
        adapter=row.get('runtimeAdapter',{});entry['sinceSwitchActivations']=entry['cursor']-adapter.get('startedCursor',0) if adapter else None
        entry['backendDeclared']=adapter.get('name','queued' if entry['status']=='PENDING' else 'original')
        entry['checkpointFilePresent']=(root/cfg['runId']/'checkpoint.npz').exists()
        entry['worst']=row.get('worstNodeAccuracy')
        if cfg['runId'] in needed:
            try:
                curves[cfg['runId']]=history(root,cfg,row)
                if curves[cfg['runId']] and (row.get('config')!=cfg or row.get('source')!=plan['source'] or
                        row.get('data')!=plan['data'][cfg['dataset']] or row.get('partition')!=plan['partitions'][cfg['runId']]):
                    raise ValueError('Saved history identity differs from plan: '+cfg['runId'])
            except OSError as error:warnings.append(f'{cfg["runId"]}: {error}')
    comparisons,groups,waiting=paired(plan,manifests,curves)
    lanes={}
    for entry in state['entries']:
        lane=str(entry['lane']);row=lanes.setdefault(lane,dict(gpu=entry['gpu'],totalRemainingSeconds=0.,priorRemainingSeconds=0.,jobs=0))
        if entry['status'] not in ('PENDING','RUNNING'):continue
        row['jobs']+=1;row['totalRemainingSeconds']+=entry['eta'] or 0.
        if entry.get('etaSource')=='prior':row['priorRemainingSeconds']+=entry['eta'] or 0.
    manager=health(root/'managed_status.json',managed,campaign.get('host'))
    supervisor=health(root/'campaign.json',campaign,campaign.get('host'))
    if state['status']=='RUNNING' and manager['status'] in ('FAILED','INTERRUPTED'):
        warnings.append('Manager is stopped while campaign metadata says RUNNING; check supervisor PID and logs. Notifications/postprocessing may lack a live manager.')
    for name,row in [('manager',manager),('supervisor',supervisor)]:
        if row['status']=='RUNNING' and row.get('pidExistsOnThisHost') is False:
            warnings.append(name+' metadata says RUNNING but its recorded PID is absent on this host.')
    return dict(generatedUtc=datetime.now(timezone.utc).isoformat(),output=str(root),
        campaign=state,manager=manager,supervisor=supervisor,
        criticalLane=max(lanes,key=lambda k:lanes[k]['totalRemainingSeconds']) if any(r['jobs'] for r in lanes.values()) else None,lanes=lanes,
        matchedSeedPairs=comparisons,matchedThreeSeedGroups=groups,waitingPairs=waiting,warnings=warnings,
        officialTestUsed=False,independentInferencePerformed=False,
        interpretation='Exploratory live comparison at the latest common saved round, not the prespecified final/last-ten-round endpoint. Three-seed means align all seeds to one common round; sigmas stay separate. Files can advance during this non-atomic snapshot.')


def render(report):
    from basil_core.sharp_monitor import duration
    state=report['campaign'];lines=[f'Campaign {state["status"]}: {state["completed"]}/{state["total"]} complete, '
        f'{state["failed"]} failed, {state["pending"]} pending. Snapshot {report["generatedUtc"]}']
    for name in ('manager','supervisor'):
        row=report[name];lines.append(f'{name}: {row["status"]}; PID {row.get("pid")}; PID exists={row.get("pidExistsOnThisHost")}; '
            f'updated {duration(row.get("ageSeconds"))} ago ({row.get("modifiedUtc")})')
        if row.get('error'):lines.append('  error: '+row['error'])
    lane=report['criticalLane']
    if lane:
        row=report['lanes'][lane];lines.append(f'ETA limiting slot {lane}: {duration(row["totalRemainingSeconds"])} remaining, '
            f'of which {duration(row["priorRemainingSeconds"])} uses priors. Planning estimate.')
    lines.append('\nALL CONFIGURATIONS (s/activation is measured since runtime switch when declared)')
    for row in state['entries']:
        c=row['config'];rate=f'{row["rate"]:.3f}' if row['rate'] is not None else '--'
        mean=f'{row["mean"]:.2%}' if row['mean'] is not None else '--'
        worst=f'{row["worst"]:.2%}' if row['worst'] is not None else '--'
        lines.append(f'G{row["gpu"]} {row["status"]:11} {c["dataset"]}/{c["partition"]} S{c["seed"]} '
            f's={c["sigma"]:g} {c["condition"]} | R{row["rounds"]}/{c["rounds"]} A{row["cursor"]}/{row["total"]} '
            f'mean={mean} worst={worst} s/A={rate} backend={row["backendDeclared"]} checkpoint={row["checkpointFilePresent"]} ETA={duration(row["eta"])} '
            f'({row.get("etaSource","complete")})')
        if row['error']:lines.append('  error: '+row['error'])
    lines.append('\nMATCHED CART / CONTROL (same seed, noise and completed round; exploratory)')
    for row in report['matchedSeedPairs']:
        lines.append(f'{row["dataset"]}/{row["partition"]} S{row["seed"]} s={row["sigma"]:g} R{row["round"]}: '
            f'CART {row["cartMean"]:.2%}, control {row["controlMean"]:.2%}, delta {row["deltaPercentagePoints"]:+.3f} pp; '
            f'worst CART/control {row["cartWorst"]:.2%}/{row["controlWorst"]:.2%}')
    lines.append('\nTHREE SEEDS AT ONE COMMON ROUND (sample SD, not a confidence interval)')
    for row in report['matchedThreeSeedGroups']:
        clean=f'{row["cleanMean"]:.2%}' if row['cleanMean'] is not None else 'waiting for matched clean rounds'
        lines.append(f'{row["dataset"]}/{row["partition"]} s={row["sigma"]:g} R{row["round"]}: '
            f'CART {row["cartMean"]:.2%}, control {row["controlMean"]:.2%}, '
            f'delta {row["deltaPercentagePoints"]:+.3f} +/- {row["sampleStdPercentagePoints"]:.3f} pp; clean {clean}')
    lines.append(f'Waiting pairs: {len(report["waitingPairs"])}. TRAIN holdout only; final endpoints and independent inference are separate.')
    lines.extend(report['warnings']);return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--json',action='store_true');args=p.parse_args()
    result=inspect(args.output)
    print(json.dumps(result,indent=2,allow_nan=False) if args.json else render(result),end='\n' if args.json else '')
