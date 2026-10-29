"""Isolated, source-frozen causal comparisons; existing studies stay immutable."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REVISION='cart_mechanism_comparison_20261029_v1'
ARMS=('sharp','cart','macro','ingress_only','commit_only','local_only','fresh_peer',
      'no_calibration','no_repair','acds')
EXPERIMENTAL_ARMS=('balanced_commit','prior_cart','prior_macro','prior_wide')
ALL_ARMS=ARMS+EXPERIMENTAL_ARMS
LABELS=dict(sharp='SHARP',cart='SHARP + CART',macro='CART macro only',
    ingress_only='CART ingress only',commit_only='CART commit only',
    local_only='Local only',fresh_peer='Fresh peer',no_calibration='CART no calibration',
    no_repair='CART no head repair',acds='SHARP + ACDS (5% data)',
    balanced_commit='CART balanced-loss check',prior_cart='CART local-prior training',
    prior_macro='CART local prior, macro only',prior_wide='CART local prior, full peer window')
ENVIRONMENTS=('joint','clean','noise_only','attack_only')
PRESETS=dict(screen=dict(arms=ARMS,rounds=10,attack_start=3,seeds=[2025],environments=['joint']),
    isolation=dict(arms=('sharp','cart','macro','acds'),rounds=10,attack_start=3,seeds=[2025],environments=ENVIRONMENTS),
    confirm=dict(arms=('cart','acds'),rounds=100,attack_start=20,seeds=[2025,2026,2027],environments=['joint']),
    easy=dict(arms=('cart','prior_cart','prior_macro','prior_wide'),rounds=5,attack_start=3,
        seeds=[2025],environments=['clean','joint']))


def revision(_=None):return REVISION


def source():
    from basil_core.cart_revision_study import source as base
    names=('basil_core/acds.py','basil_core/cart_comparison_study.py',
        'basil_core/cart_mechanism_audit.py','basil_core/cart_balanced_commit.py','basil_core/cart_local_prior.py',
        'scripts/verify_cart_local_prior_gpu.py','scripts/run_cart_comparison.py',
        'scripts/cart_comparison_worker.py','scripts/verify_cart_comparison_gpu.py',
        'reporting/cart_comparison.py')
    return {**base(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def check_source(expected):
    if source()!=expected:raise ValueError('Comparison source changed; use a fresh output.')


def selected(args):
    from basil_core.cart_noise_selection import NoiseSelection
    from basil_core.sharp_study import configuration
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration as cart_config
    noise=NoiseSelection.create(args.noise_levels,args.noise_values)
    if not 0<=args.share_fraction<=1:raise ValueError('Share fraction must be in [0,1].')
    if args.acds_batches<1:raise ValueError('Positive ACDS batch count required.')
    for name,values in [('arms',args.arms),('environments',args.environments),('datasets',args.datasets),
                        ('partitions',args.partitions),('seeds',args.seeds)]:
        if not values or len(set(values))!=len(values):raise ValueError(name+' must be nonempty and distinct.')
    configs=[]
    for dataset in args.datasets:
        for partition in args.partitions:
            for seed in args.seeds:
                for env in args.environments:
                    for sigma in (noise.values if env in ('joint','noise_only') else (0.,)):
                        for arm in args.arms:
                            # Keep the full defense/optimizer stack fixed when isolating
                            # physical impairments. Attack/noise-only here are controlled
                            # environment ablations, not untreated legacy conditions.
                            physical=configuration(dataset,partition,seed,'attack_noise_defense_rla',
                                sigma or noise.values[0],nodes=args.nodes,rounds=args.rounds,
                                local_steps=args.local_steps,model=args.model,alpha=args.alpha,
                                attack_start=args.attack_start,train_per_class=args.train_per_class,
                                validation_per_class=args.validation_per_class)
                            physical.update(sigma=sigma,actualAttackers=physical['byzantineBudget'] if env in ('joint','attack_only') else 0)
                            physical['switches']=dict(physical['switches'],attack=env in ('joint','attack_only'),noise=bool(sigma))
                            # The original protocol validator remains strict. Store a
                            # standard, valid inner config; environment overrides are
                            # passed through a separately validated execution adapter.
                            inner=cart_config(guarded_configuration(configuration(dataset,partition,seed,
                                'attack_noise_defense_rla_cart' if arm not in ('sharp','acds','local_only','fresh_peer') else 'attack_noise_defense_rla',
                                noise.values[0] if not sigma else sigma,nodes=args.nodes,rounds=args.rounds,
                                local_steps=args.local_steps,model=args.model,alpha=args.alpha,
                                attack_start=args.attack_start,train_per_class=args.train_per_class,
                                validation_per_class=args.validation_per_class)))
                            case=dict(arm=arm,environment=env,protocolConfig=inner,
                                effectiveSigma=sigma,effectiveAttackers=physical['actualAttackers'],
                                shareFraction=args.share_fraction if arm=='acds' else 0.,acdsBatches=args.acds_batches)
                            key=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()[:12]
                            view=dict(inner,comparisonCase=case,comparisonLabel=arm,
                                sigma=sigma,runId=f'{dataset}-{partition}-{seed}-{env}-{arm}-{key}')
                            configs.append(view)
    return noise,configs


def effective_config(view):
    """Explicitly versioned physical switch overrides with unchanged inner schema."""
    from basil_core.sharp_cart_v2_protocol import validate_config
    case=view['comparisonCase'];cfg=case['protocolConfig'];validate_config(cfg)
    expected_sigma=case['effectiveSigma'];env=case['environment']
    if case['arm'] not in ALL_ARMS or env not in ENVIRONMENTS:raise ValueError('Unknown comparison case.')
    if (env in ('joint','noise_only')) != (expected_sigma>0):raise ValueError('Environment/noise mismatch.')
    expected_attackers=cfg['byzantineBudget'] if env in ('joint','attack_only') else 0
    if case['effectiveAttackers']!=expected_attackers:raise ValueError('Environment/attacker mismatch.')
    cfg=dict(cfg,sigma=expected_sigma,actualAttackers=expected_attackers)
    cfg['switches']=dict(cfg['switches'],attack=bool(expected_attackers),noise=bool(expected_sigma))
    return cfg


def partitions_for(view,labels):
    from basil_core.sharp_data import prepare_partition
    cfg=view['comparisonCase']['protocolConfig'];chunks,held,audit=prepare_partition(labels,cfg)
    if view['comparisonCase']['arm']=='acds':
        from basil_core.acds import cyclic_training_share
        chunks,sharing=cyclic_training_share(chunks,labels,fraction=view['comparisonCase']['shareFraction'],
            batches=view['comparisonCase']['acdsBatches'],seed=cfg['seed'])
        audit=dict(audit,acds=sharing)
    return chunks,held,audit


def data_contract(configs):
    from basil_core.sharp_data import training_data,digest
    data={};audits={};cache={}
    for view in configs:
        ds=view['dataset']
        if ds not in cache:
            x,y=training_data(ds);cache[ds]=(x,y)
            data[ds]=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
        _,_,audits[view['runId']]=partitions_for(view,cache[ds][1])
    return data,audits


def configure_worker(view,device='GPU'):
    from basil_core.cart_batched_scoring import BatchedAdmittedWorker
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    cfg=effective_config(view);arm=view['comparisonCase']['arm']
    if arm in ('prior_cart','prior_macro','prior_wide'):
        from basil_core.cart_local_prior import LocalPriorWorker
        return LocalPriorWorker(cfg,device)
    if arm in ('sharp','acds'):
        class PooledWorker(BatchedAdmittedWorker):
            # Reuse exact candidate scoring but retain ordinary SHARP SGD,
            # including its original bootstrap coefficient and anchor.
            train=CompiledTransferWorker.train
        return PooledWorker(cfg,device)
    return BatchedAdmittedWorker(cfg,device)


def _commit_context(cfg,state,worker,x,y,node,incoming,base):
    """Commit-only: keep SHARP ingress; score the full local update only."""
    import numpy as np
    from basil_core.sharp_cart import probes
    from basil_core.cart_class_skill import class_skill
    from basil_core.cart_quality_routing_candidate import utility
    from basil_core.research_protocol import keyed_rng
    worker.quality_context=None
    if state['senderRounds'][node]<0:return
    fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass'])
    classes=np.intersect1d(np.unique(y[fit]),np.unique(y[check]))
    if len(classes)<2:return
    fit=fit[np.isin(y[fit],classes)];check=check[np.isin(y[check],classes)];idx=np.concatenate((fit,check));split=len(fit)
    def evidence(flat):
        z=worker.outputs(flat,x[idx])
        return class_skill(z[:split],y[fit],classes),class_skill(z[split:],y[check],classes)
    own=state['nodeParams'][node];f,k=evidence(own)
    if min(f.mean(),k.mean())<=.02:return
    target=np.minimum(f,k);qualified=np.asarray([sum(y[fit]==c)>=8 and sum(y[check]==c)>=8 for c in classes])&(target>=.2)
    allowance=np.zeros(len(classes));draws=0;pf,pk=evidence(incoming)
    if qualified.any() and cfg['sigma'] and np.any((target-np.minimum(pf,pk))[qualified]>0):
        for draw in range(cfg['calibrationDraws']):
            noise=keyed_rng(cfg['seed'],'quality_route_channel',node,state['cursor'],draw).normal(0,cfg['sigma'],size=own.shape).astype(np.float32)
            nf,nk=evidence(own+noise);allowance=np.maximum(allowance,np.maximum(target-np.minimum(nf,nk),0));draws+=1
    score=utility(pf,pk,target,qualified,allowance)
    worker.quality_context=dict(ingress=incoming.copy(),indices=idx,split=split,classes=classes,
        target=target,qualified=qualified,allowance=allowance,penalty=True,baseline=score)
    base['continuity'].update(qualifiedClasses=classes[qualified].tolist(),calibrationDraws=draws,
        qualityScoreBoundApplied=False,commitOnly=True,selectedScore=score)


@contextmanager
def intervention(view,worker):
    """Process-local adapter; never modifies any archived scientific source."""
    import basil_core.sharp_cart_v2_protocol as protocol
    import basil_core.sharp_cart_v2 as cart
    from basil_core.cart_batched_scoring import select_input as pooled
    from basil_core.cart_prior_neutral_commit import selector,transport,commit
    from basil_core.sharp_protocol import link_vector
    arm=view['comparisonCase']['arm'];cfg=effective_config(view)
    original_select=protocol.select_input;original_transport=cart.transport;original_validate=protocol.validate_config
    inner=view['comparisonCase']['protocolConfig']
    def validate(actual):
        original_validate(inner)
        if actual!=cfg:raise ValueError('Execution overrides differ from declared comparison case.')
    choose=selector(pooled,'macro' if arm in ('macro','prior_macro') else 'class',
        peer_budget=cfg['window'] if arm=='prior_wide' else 8)
    def select(c,state,w,x,y,node,r,batch):
        w.quality_context=None
        if arm in ('cart','macro','ingress_only','no_calibration','no_repair','balanced_commit','prior_cart','prior_macro','prior_wide'):
            # Only the allowance intervention changes calibration draws.
            return choose(dict(c,calibrationDraws=0) if arm=='no_calibration' else c,state,w,x,y,node,r,batch)
        if arm=='local_only':
            w.admitted_route='own';w.admitted_has_channel=False
            available=sum(bool(state['senderRounds'][(node-d)%c['nodes']]>=0) for d in range(1,c['window']+1))
            value=state['nodeParams'][node].copy()
            return value,dict(selectedSenders=[],selectedCount=1,availableCopies=available,bootstrap=not available,
                screenApplied=False,screenAllFailFallback=False,ranking=[],continuity=dict(actualRoute='own',localOnly=True))
        value,row=pooled(c,state,w,x,y,node,r,batch)
        if arm=='fresh_peer' and row['availableCopies']:
            senders=[(node-d)%c['nodes'] for d in range(1,c['window']+1) if state['senderRounds'][(node-d)%c['nodes']]>=0]
            passed=[s for j,s in enumerate(senders) if not row.get('screenApplied') or row['peakRatios'][j]<=row['peakThreshold']]
            s=min(passed or senders,key=lambda s:(-int(state['senderRounds'][s]),(node-s)%c['nodes']))
            value=link_vector(state['outbound'][s],c['seed'],int(state['senderRounds'][s]),s,node,c['sigma'])
            row.update(selectedSenders=[s],selectedCount=1)
        if arm not in ('sharp','acds'):
            w.admitted_route='pool' if arm=='commit_only' else 'neighbor'
            w.admitted_has_channel=bool(row['availableCopies']);w.continuity_anchor=value.copy()
            row['continuity']=dict(actualRoute=w.admitted_route)
        if arm=='commit_only':_commit_context(c,state,w,x,y,node,value,row)
        return value,row
    def repair(w,proposal,previous,x,y,node,activation):
        if arm=='balanced_commit':
            from basil_core.cart_balanced_commit import transport as balanced_transport
            return balanced_transport(original_transport)(w,proposal,previous,x,y,node,activation)
        if arm=='no_repair':
            context=getattr(w,'quality_context',None)
            chosen,qc=commit(w,proposal,context,x,y) if context else (proposal,dict(applied=False))
            w.load(chosen)
            return chosen,previous,dict(referenceAdmitted=False,headRepairDisabled=True,qualityCommit=qc)
        return transport(original_transport)(w,proposal,previous,x,y,node,activation)
    # Commit-only needs a repair callback even though its inner cfg is CART-on.
    protocol.select_input=select;protocol.validate_config=validate
    if cfg['cart'] and arm!='ingress_only':cart.transport=repair
    try:yield cfg
    finally:
        protocol.select_input=original_select;protocol.validate_config=original_validate;cart.transport=original_transport
