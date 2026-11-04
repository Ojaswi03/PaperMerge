"""Explicit, audited checkpoint forks; never relax the original checkpoint loader."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from basil_core import basil_sharp_thesis as study

ROOT=Path(__file__).resolve().parents[1]
PRESET=ROOT/'configs/research/all_500r_from_100r.json'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def extended_view(view,rounds):
    """Rebuild all nested identities, while proving numerical settings unchanged."""
    cfg=study.effective_config(view);case=view['comparisonCase']
    if type(rounds) is not int or rounds<=cfg['rounds']:raise ValueError('Continuation must extend the horizon.')
    args=SimpleNamespace(datasets=[cfg['dataset']],partitions=[cfg['partition']],seeds=[cfg['seed']],
        families=[case['family']],environments=[case['environment']],rounds=rounds,nodes=cfg['nodes'],
        local_steps=cfg['localSteps'],alpha=cfg['dirichletAlpha'],sigma=case['protocolConfig']['sigma'],
        attack_start=cfg['attackStart'],attack_type=cfg['attackType'],gaussian_std=cfg.get('gaussianAttackStd',1.))
    target=next(v for v in study.selected(args) if v['comparisonCase']['arm']==case['arm'])
    a,b=dict(cfg),study.effective_config(target)
    for value in (a,b):
        for key in ('rounds','runId','baseRunId','guardedRunId'):value.pop(key,None)
    if a!=b:raise ValueError('Continuation changed scientific settings other than the horizon.')
    return target


def catalog(preset=PRESET):
    spec=read(preset)
    if spec['schema']!=1:raise ValueError('Unsupported continuation schema.')
    items=[]
    for entry in spec['archives']:
        root=(ROOT/entry['path']).resolve()
        if not root.is_relative_to(ROOT/'docs/results') or sha(root/'SHA256.json')!=entry['manifest_sha256']:
            raise ValueError('Wrong source export: '+entry['path'])
        manifest=read(root/'SHA256.json')
        for name,expected in manifest.items():
            path=root/name
            if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path)!=expected:
                raise ValueError('Invalid source export hash/path: '+name)
        plan=read(root/'plan.json');receipt=read(root/'independent_validation.json')
        verified={r['runId'] for r in receipt['verified']}
        if (len(plan['configs'])!=entry['configurations'] or len(verified)!=len(receipt['verified'])
                or verified!={c['runId'] for c in plan['configs']}
                or receipt['status']!='ALL_COMPARISON_MODELS_VERIFIED'
                or receipt['source']!=plan['source'] or receipt['officialTestUsed']):
            raise ValueError('Source export lacks matching independent verification.')
        for index,view in enumerate(plan['configs']):
            record=read(root/view['runId']/'run.json');cfg=study.effective_config(view)
            cursor=cfg['nodes']*cfg['rounds']
            if (cfg['rounds']!=spec['source_rounds'] or record['status']!='COMPLETED'
                    or record['completedActivations']!=cursor or record['optimizerSteps']!=cursor*cfg['localSteps']
                    or record['config']!=view or record['source']!=plan['source']):
                raise ValueError('Source run is incomplete or mismatched.')
            target=extended_view(view,spec['target_rounds'])
            source_root=Path(plan['profiles'][index%len(plan['profiles'])]['output_root'])
            items.append(dict(config=target,sourceArchive=entry['path'],sourceRunId=view['runId'],
                sourceRunSha256=manifest[view['runId']+'/run.json'],
                sourceMetricsSha256=manifest[view['runId']+'/metrics.npz'],
                sourcePlanSha256=manifest['plan.json'],sourceManifestSha256=entry['manifest_sha256'],
                sourceHost=record['runtime']['host'],
                originalCheckpoint=str(source_root/view['runId']/'checkpoint.npz'),
                sourceCursor=cursor,sourceRounds=cfg['rounds'],sourceElapsedSeconds=record['elapsedSeconds']))
    if len(items)!=spec['expected_configurations'] or len({i['config']['runId'] for i in items})!=len(items):
        raise ValueError('Incomplete or duplicated continuation catalog.')
    # Interleave datasets/methods rather than leaving an entire slow dataset last.
    items.sort(key=lambda i:(i['config']['seed'],i['config']['comparisonCase']['arm'],i['config']['partition'],
        i['config']['dataset'],i['config']['attackType']))
    return spec,items


def source_record(item):
    root=(ROOT/item['sourceArchive']).resolve()
    if not root.is_relative_to(ROOT/'docs/results'):raise ValueError('Invalid source archive.')
    path=root/item['sourceRunId']/'run.json'
    if sha(path)!=item['sourceRunSha256']:raise ValueError('Source run record changed.')
    return read(path)


def description(spec,items):
    remaining=sum((spec['target_rounds']-i['sourceRounds'])*i['config']['nodes'] for i in items)
    return dict(configurations=len(items),sourceRounds=spec['source_rounds'],targetRounds=spec['target_rounds'],
        nextDisplayedRound=spec['source_rounds']+1,additionalActivations=remaining,
        additionalOptimizerSteps=sum((spec['target_rounds']-i['sourceRounds'])*i['config']['nodes']*
            i['config']['localSteps'] for i in items),
        datasets=dict(Counter(i['config']['dataset'] for i in items)),
        fedSharpPlus=sum(i['config']['comparisonCase']['addon']=='cart' for i in items),
        sourceArchives=dict(Counter(i['sourceArchive'] for i in items)),trainingStarted=False)


def copy_seed(source,target,expected=None):
    source,target=Path(source),Path(target)
    if source.is_symlink() or not source.is_file():raise ValueError('Missing/unsafe seed checkpoint: '+str(source))
    digest=sha(source)
    if expected and digest!=expected:raise ValueError('Seed checkpoint checksum differs.')
    if source.resolve()==target.resolve():raise ValueError('Seed destination must be a separate copy.')
    if target.exists():
        if target.is_symlink() or sha(target)!=digest:raise ValueError('Conflicting seed checkpoint copy.')
        return digest
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
    try:
        shutil.copyfile(source,temporary)
        if sha(temporary)!=digest or sha(source)!=digest:raise ValueError('Seed changed during copy.')
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    return digest


def bootstrap(plan,view,folder,worker,contract,vx,vy):
    """Validate the complete old state and fork it into a new, explicit contract."""
    import numpy as np
    from basil_core.cart_campaign_io import write_json
    from basil_core.sharp_protocol import load_state,save_state,vector_hash,evaluate,attacked_outbound
    from basil_core.research_protocol import resolve_attackers
    item=plan['continuations'][view['runId']];old=source_record(item)
    if old['initialHash']!=contract['initialHash'] or old['data']!=contract['data'] or old['partition']!=contract['partition']:
        raise ValueError('Continuation initialization/data/partition differs from the source.')
    if extended_view(old['config'],contract['config']['rounds'])!=view:
        raise ValueError('Continuation target differs from its recorded parent.')
    for key in ('tensorflow','numpy','tf32','device'):
        if old['runtime'][key]!=contract['runtime'][key]:raise ValueError('Numerical runtime changed: '+key)
    old_profile=old['runtime'].get('profile',{});new_profile=contract['runtime'].get('profile',{})
    for key in ('threads','data_workers','memory_mib'):
        if old_profile.get(key)!=new_profile.get(key):raise ValueError('Numerical worker profile changed: '+key)
    source=Path(plan['sharedQueue'])/item['seedRelativePath']
    cached=Path(plan['profiles'][0]['output_root'])/'seed_cache'/view['runId']/'checkpoint.npz'
    digest=copy_seed(source,cached,item['seedSha256'])
    state=load_state(cached,old['contract'],worker.dimension)
    cfg=old['contract']['config'];cursor=cfg['nodes']*cfg['rounds']
    if state['cursor']!=cursor or len(state['roundMetrics'])!=cfg['rounds']:
        raise ValueError('Parent checkpoint is not complete at its declared boundary.')
    if sum(r['training']['optimizerSteps'] for r in state['telemetry'])!=old['optimizerSteps']:
        raise ValueError('Parent optimizer-step budget differs.')
    if any(r['training']['optimizerSteps']!=cfg['localSteps'] for r in state['telemetry']):
        raise ValueError('Parent local SGD budget differs.')
    metrics=ROOT/item['sourceArchive']/item['sourceRunId']/'metrics.npz'
    if sha(metrics)!=item['sourceMetricsSha256']:raise ValueError('Parent metric file changed.')
    with np.load(metrics,allow_pickle=False) as saved:
        for key,field in (('roundNodeAccuracy','nodeAccuracy'),('roundNodePerClassAccuracy','perClass'),
                ('roundHonestMeanAccuracy','honestMeanAccuracy')):
            np.testing.assert_array_equal(saved[key],np.asarray([r[field] for r in state['roundMetrics']]))
        accuracy=saved['roundNodeAccuracy'][-1].copy();classes=saved['roundNodePerClassAccuracy'][-1].copy()
    attackers=resolve_attackers(cfg['seed'],cfg['actualAttackers'],cfg['nodes'])
    for node,params in enumerate(state['nodeParams']):
        expected=attacked_outbound(cfg,worker,params,cfg['rounds']-1,node) if node in attackers and cfg['rounds']>cfg['attackStart'] else params
        np.testing.assert_array_equal(expected,state['outbound'][node])
        event=state['telemetry'][cursor-cfg['nodes']+node]
        if event['trainedHash']!=vector_hash(params) or event['outboundHash']!=vector_hash(state['outbound'][node]):
            raise ValueError('Parent final model hashes differ from activation telemetry.')
    fresh=[evaluate(worker,p,vx,vy,cfg['classes']) for p in state['nodeParams']]
    error=float(np.max(abs(np.asarray([v for v,_ in fresh])-accuracy)))
    counts=np.bincount(vy,minlength=cfg['classes'])
    if error>1/len(vy)+1e-9 or np.any(abs(np.asarray([v for _,v in fresh])-classes)>1/counts[None,:]+1e-9):
        raise ValueError('Continuation-device inference disagrees with the verified boundary.')
    if sha(cached)!=digest:raise ValueError('Parent checkpoint changed during validation.')
    receipt=dict(sourceRunId=item['sourceRunId'],sourceArchive=item['sourceArchive'],sourceCheckpointSha256=digest,
        sourceContractSha256=hashlib.sha256(json.dumps(old['contract'],sort_keys=True).encode()).hexdigest(),
        sourceCursor=cursor,sourceRounds=cfg['rounds'],sourceElapsedSeconds=state['elapsedSeconds'],
        targetRunId=view['runId'],targetRounds=contract['config']['rounds'],
        sourceRuntime=old['runtime'],targetRuntime=contract['runtime'],
        maximumBoundaryInferenceDifference=error,prefixMetricsExactlyPreserved=True,
        retainedModelsOutgoingVersionsReferencesAndTelemetryPreserved=True,sourceCheckpointUnmodified=True)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    temporary=folder/'bootstrap.npz'
    save_state(temporary,state,contract)
    restored=load_state(temporary,contract,worker.dimension)
    for key in ('nodeParams','outbound','senderRounds'):np.testing.assert_array_equal(restored[key],state[key])
    if restored['telemetry']!=state['telemetry'] or restored['roundMetrics']!=state['roundMetrics']:
        raise ValueError('Checkpoint fork lost its historical records.')
    if set(restored['references'])!=set(state['references']):raise ValueError('Checkpoint fork lost classifier references.')
    for node,reference in state['references'].items():
        other=dict(restored['references'][node]);params=other.pop('params');meta=dict(reference);original=meta.pop('params')
        np.testing.assert_array_equal(params,original)
        if other!=meta:raise ValueError('Checkpoint fork changed classifier reference metadata.')
    write_json(folder/'continuation.json',receipt)
    temporary.replace(folder/'checkpoint.npz')
    return restored,receipt
