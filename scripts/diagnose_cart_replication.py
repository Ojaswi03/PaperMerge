#!/usr/bin/env python3
"""Post-hoc TRAIN-only diagnosis of saved CART comparisons; never update weights."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def records(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def phase_statistics(history,config):
    import numpy as np
    from basil_core.dirichlet_diagnosis import actually_corrupted
    steps=[s for r in history for s in r['optimizerTelemetry']]
    if not steps:raise ValueError('Measured optimizer telemetry is required.')
    ratios=[s['cartBarrierGradientNorm']/(s['baseGradientNorm']+1e-12) for s in steps]
    return dict(activations=len(history),optimizerSteps=len(steps),
        outboundAttacks=sum(r['attackActive'] for r in history),
        corruptedSelections=sum(actually_corrupted(r['inputSourceNode'],r['inputSnapshotRound'],config) for r in history),
        meanRetentionBaseRatio=float(np.mean(ratios)),
        retentionActiveStepFraction=float(np.mean([s['cartBarrierGradientNorm']>0 for s in steps])),
        capSaturatedStepFraction=float(np.mean([v>=.099 for v in ratios])),
        meanViolatedSupportedClassFraction=float(np.mean([s['cartActiveClassFraction'] for s in steps])),
        meanEbmCorrectionRatio=float(np.mean([s['ebmCorrectionRatio'] for s in steps])),
        meanRawTrainingBatchAccuracy=float(np.mean([s['trainingBatchAccuracy'] for s in steps])))


def summarize(root):
    import numpy as np
    review=json.loads((root/'review.json').read_text())
    if review['status']!='all_saved_pairs_verified':raise ValueError('Verify the complete saved study before diagnosis.')
    rows=[];warmup=[]
    for seed in (2025,2026,2027):
        curves={}
        for letter in 'ABCD':
            for method in ('prior','budgeted'):
                folder=root/f'seed-{seed}'/f'pair-{letter}'/f'{method}_{letter}'
                config=json.loads((folder/'run.json').read_text())['config']
                with np.load(folder/'validation_metrics.npz',allow_pickle=False) as values:
                    mean=values['mean'].copy();worst=values['worst'].copy()
                if mean.shape!=(30,) or worst.shape!=(30,):raise ValueError('Complete thirty-round metrics required.')
                curves[method,letter]=mean
                history=records(folder/'activation_telemetry.json.gz')
                if [(r['round'],r['nodeId']) for r in history]!=[(r,n) for r in range(30) for n in range(10)]:
                    raise ValueError('Incomplete/unordered activation history.')
                rows.append(dict(seed=seed,condition=letter,method=method,
                    validationMean=mean.tolist(),validationWorst=worst.tolist(),
                    preAttack=phase_statistics(history[:200],config),postAttack=phase_statistics(history[200:],config)))
        for method in ('prior','budgeted'):
            warmup.append(dict(seed=seed,method=method,
                exactABWarmup=np.array_equal(curves[method,'A'][:20],curves[method,'B'][:20])))
    phases=[]
    for letter in 'ABCD':
        for method in ('prior','budgeted'):
            selected=[r for r in rows if r['condition']==letter and r['method']==method]
            combined=dict(condition=letter,method=method)
            for phase in ('preAttack','postAttack'):
                combined[phase]={key:float(np.mean([r[phase][key] for r in selected])) for key in selected[0][phase]}
            combined['postAttackMeanAccuracy']=float(np.mean([np.mean(r['validationMean'][20:]) for r in selected]))
            phases.append(combined)
    return dict(status='post_hoc_diagnosis',officialTestUsed=False,trainingStarted=False,
        sourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        explanation='Post-hoc diagnostic statistics; no causal improvement or new benchmark approval is implied.',
        primaryResults=review['conditions'],warmupChecks=warmup,phaseSummaries=phases,arms=rows)


def gradient_probes(root):
    """CPU replay of one saved first minibatch per seed for clean A and noisy D.

    Uses the last activation's saved input weights, frozen per-class ceilings,
    local data order and augmentation key. TensorFlow versions/devices differ
    from the archived server, so compare scalar diagnostics within tolerance;
    this is not an exact server replay or a new training experiment.
    """
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    prepare_environment('CPU')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,'CPU')
    from basil_core.cart_local_memory import LocalClassMemory,MAX_PROBES
    from basil_core.cart_validation import partition
    from basil_core.dirichlet_calibration import calibration_offsets,calibrated_anchor_gradients
    from basil_core.cart_barrier import barrier_gradients
    from basil_core.dirichlet_diagnosis import array_identity_hash
    from basil_core.research_audit import augment_batch
    from basil_core.research_protocol import build_model,params_hash,keyed_rng
    cache=ROOT/'experiments/cache/cifar10'
    images=np.load(cache/'x_train_float32.npy',mmap_mode='r',allow_pickle=False)
    labels=np.load(cache/'y_train_int32.npy',mmap_mode='r',allow_pickle=False)
    split=json.loads((root/'seed-2025/pair-A/split.json').read_text())
    fit,validation=np.array(split['fitIndices']),np.array(split['validationIndices'])
    x,y,vx,vy=images[fit],labels[fit],images[validation],labels[validation]
    actual={k:array_identity_hash(v) for k,v in zip(('fitImages','fitLabels','validationImages','validationLabels'),(x,y,vx,vy))}
    results=[]
    def flat(values):return np.concatenate([v.numpy().ravel().astype(np.float64) for v in values])
    def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
    for seed in (2025,2026,2027):
        digest,chunks=partition(y,seed=seed)
        for letter in ('A','D'):
            folder=root/f'seed-{seed}'/f'pair-{letter}'/f'budgeted_{letter}'
            run=json.loads((folder/'run.json').read_text());cfg=run['config']
            if actual!=run['dataSha256'] or digest!=cfg['partitionHash']:raise ValueError('Replay data identity differs.')
            record=records(folder/'activation_telemetry.json.gz')[-1]
            with np.load(folder/'checkpoint.npz',allow_pickle=False) as saved:
                meta=json.loads(str(saved['metadata']));key=record['inputHash']
                params=[saved[f'{key}_{i}'].copy() for i in range(meta['weights'][key])]
            if params_hash(params)!=key:raise ValueError('Saved selected input model is corrupt.')
            round_id,node=record['round'],record['nodeId'];local=chunks[node]
            lx,ly=x[local],y[local]
            probe=LocalClassMemory(seed).prepare(node,lx,ly)
            px,py=lx[probe],ly[probe]
            controls=record['cartMemory']['epochControls'][0]['classes']
            padded=np.zeros((MAX_PROBES,32,32,3),np.float32);padded[:len(px)]=px
            target=np.zeros(MAX_PROBES,np.int32);target[:len(py)]=py
            average=np.zeros((10,MAX_PROBES),np.float32);best=np.zeros(10,np.float32);support=np.zeros(10,np.float32)
            for row in controls:
                c=row['classId'];ids=np.flatnonzero(py==c);average[c,ids]=1./len(ids)
                best[c]=row['bestLoss'];support[c]=1.
            model=build_model('basil_paper_cnn',seed);model.set_weights(params)
            order=keyed_rng(seed,'data_order',round_id,node,0).permutation(len(ly))[:512]
            key=keyed_rng(seed,'augmentation',round_id,node,0,0).integers(0,2**30,size=2,dtype=np.int32)
            batch=augment_batch(lx[order],key)
            offset=calibration_offsets(np.bincount(ly,minlength=10),method='prior_adjustment')
            coefficient=.0001 if letter=='D' else 0.
            base,base_values=calibrated_anchor_gradients(model,batch,ly[order],params,offset,coefficient=coefficient)
            complete,values=barrier_gradients(model,batch,ly[order],params,offset,padded,target,average,best,support,
                coefficient=coefficient,correction_ratio_cap=.1)
            b,g=flat(base),flat(complete);correction=g-b
            # This validation derivative is diagnostic only. It is never supplied
            # to a worker, used as an update, or used as an admission criterion.
            validation_gradient=np.zeros_like(b)
            for start in range(0,len(vy),512):
                with tf.GradientTape() as tape:
                    ce=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=vy[start:start+512],logits=model(vx[start:start+512],training=False)))/len(vy)
                validation_gradient+=flat(tape.gradient(ce,model.trainable_variables))
            saved_step=record['optimizerTelemetry'][0]
            checks={name:dict(saved=saved_step[name],cpu=float(value)) for name,value in (
                ('baseCrossEntropy',values[0]),('baseGradientNorm',values[4]),
                ('cartBarrierLoss',values[15]),('cartBarrierGradientNorm',values[16]))}
            for name,row in checks.items():
                if not np.isclose(row['saved'],row['cpu'],rtol=.01,atol=1e-5):
                    raise ValueError(f'CPU reconstruction differs from saved {name}; interpret no gradient result.')
            results.append(dict(seed=seed,condition=letter,node=node,roundIndex=round_id,
                cpuTensorflow=tf.__version__,savedTensorflow=run['runtime']['tensorflow'],savedScalarChecks=checks,
                correctionBaseCosine=cosine(correction,b),correctionValidationCosine=cosine(correction,validation_gradient),
                correctionValidationDot=float(np.dot(correction,validation_gradient)),
                note='Negative correction/validation dot predicts increased holdout CE to first order for the added descent correction.'))
            print(json.dumps(results[-1]),flush=True)
    return results


def plot_curves(result,path):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained',sharex=True,sharey=True)
    for letter,ax in zip('ABCD',axes.flat):
        for method,color in [('prior','tab:blue'),('budgeted','tab:orange')]:
            values=np.array([r['validationMean'] for r in result['arms'] if r['condition']==letter and r['method']==method])*100
            ax.plot(range(1,31),values.mean(axis=0),label='Control' if method=='prior' else 'CART',color=color)
            ax.fill_between(range(1,31),values.min(axis=0),values.max(axis=0),color=color,alpha=.13)
        ax.axvline(20.5,color='gray',linestyle='--',linewidth=1)
        ax.set(title=f'Condition {letter}',xlabel='Completed round',ylabel='Held-out TRAIN accuracy (%)')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('All three seeds: mean line, min/max band; attack starts at round index 20 in B/C/D')
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--gradient-probes',action='store_true');args=parser.parse_args();root=args.directory.resolve()
    if not root.is_relative_to(ROOT/'.papermerge/local-tests'):parser.error('Use the isolated local evidence tree.')
    result=summarize(root)
    if args.gradient_probes:result['gradientProbes']=gradient_probes(root)
    output=root/'diagnostics';output.mkdir(exist_ok=True)
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot_curves(result,output/'three_seed_curves.png')
    print(json.dumps(dict(status=result['status'],output=str(output),warmup=result['warmupChecks']),indent=2))


if __name__=='__main__':main()
