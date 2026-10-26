#!/usr/bin/env python3
"""Verify saved TRAIN-only comparisons and plot all fixed-horizon outcomes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def review(root,*,plot=False):
    import numpy as np
    from basil_core.cart_local_memory import state_hash
    from basil_core.cart_memory_study import validate_memory_checkpoint
    from basil_core.cart_validation import selected_configurations,validate_config
    from basil_core.research_protocol import params_hash
    split=json.loads((root/'split.json').read_text())
    fit,validation=np.array(split['fitIndices']),np.array(split['validationIndices'])
    if (len(fit)!=49000 or len(validation)!=1000 or np.intersect1d(fit,validation).size or
            not np.array_equal(np.sort(np.concatenate((fit,validation))),np.arange(50000))):
        raise ValueError('TRAIN/validation split overlaps, omits or duplicates examples.')
    summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='completed':raise ValueError('The complete fixed comparison is required.')
    seed=summary.get('seed',2025)
    expected=selected_configurations('a'*64,'b'*64,'CPU',summary['stage'],seed=seed,pair=summary.get('pair'))
    if (set(summary['outcomes'])!={c['runId'] for c in expected} or
            summary['horizon']!=expected[0]['nRounds']):
        raise ValueError('The complete predeclared set of comparison arms is required.')
    rounds=summary['horizon']
    identities=[];rows=[]
    for name in summary['outcomes']:
        output=root/name;run=json.loads((output/'run.json').read_text())
        validate_config(run['config'])
        if (run['config']['runId']!=name or run['config']['cartValidationStage']!=summary['stage'] or
                run['config']['seed']!=seed):
            raise ValueError('Run identity differs from its comparison arm.')
        if run['status']!='completed' or run['completedRounds']!=rounds or run['completedActivations']!=rounds*10:
            raise ValueError('All fixed-horizon conditions must complete before review.')
        if run['officialTestLoaded'] is not False or run['evaluationOrigin']!='heldout_TRAIN':
            raise ValueError('TRAIN-only evaluation identity is missing.')
        identities.append((run['implementationSha256'],run['dataSha256'],run['runtime']))
        for filename,digest in run['implementationSha256'].items():
            if hashlib.sha256((output/'source_snapshots'/digest).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Executed source archive differs: {filename}')
        telemetry=json.loads(gzip.decompress((output/'activation_telemetry.json.gz').read_bytes()))
        if [(r['round'],r['nodeId']) for r in telemetry]!=[(r,n) for r in range(rounds) for n in range(10)]:
            raise ValueError('Activation history is not contiguous.')
        steps=[]
        for record in telemetry:
            cfg=run['config']
            attack=bool(cfg['attackHidden'] and record['configuredByzantine'] and record['round']>=cfg['attackStartRound'])
            coefficient=.0001 if cfg['ebmMode']=='gradient_norm_objective' else 0.
            if (record['attackActive']!=attack or record['channelNoiseActive']!=cfg['useChannelNoise']
                    or abs(record['ebmCoefficient']-coefficient)>1e-10):
                raise ValueError('Attack/channel/EBM behavior differs from the declared condition.')
            if record['inputHash']!=record['snapshotAnchorReferenceHash']:
                raise ValueError('Snapshot anchor differs from the complete selected model.')
            expected=5*((record['localSampleCount']+511)//512)
            if record['optimizerSteps']!=expected or len(record['optimizerTelemetry'])!=expected:
                raise ValueError('Optimizer work differs from five full local epochs.')
            if any(e['samplesSeen']!=record['localSampleCount'] or sum(e['batchSizes'])!=record['localSampleCount']
                   for e in record['epochs']):raise ValueError('Epoch coverage differs.')
            steps.extend(record['optimizerTelemetry'])
        with np.load(output/'validation_metrics.npz',allow_pickle=False) as metrics:
            shapes=dict(mean=(rounds,),worst=(rounds,),perNode=(rounds,10),
                perClass=(rounds,10,10),selectedSenders=(rounds,10))
            if set(metrics.files)!=set(shapes) or any(metrics[k].shape!=shape for k,shape in shapes.items()):
                raise ValueError('Saved validation metric dimensions differ from the declared horizon.')
            for field in metrics.files:
                if not np.isfinite(metrics[field]).all():raise ValueError('Non-finite measured metrics.')
            np.testing.assert_array_equal(metrics['mean'],metrics['perNode'].mean(axis=1))
            np.testing.assert_array_equal(metrics['worst'],metrics['perNode'].min(axis=1))
            np.testing.assert_array_equal(metrics['mean'],run['validationMean'])
            means,worst=metrics['mean'].tolist(),metrics['worst'].tolist()
        with np.load(output/'checkpoint.npz',allow_pickle=False) as checkpoint:
            metadata=json.loads(str(checkpoint['metadata']))
            if (metadata['config']!=run['config'] or metadata['implementation']!=run['implementationSha256'] or
                    metadata['initialHash']!=run['config']['initialModelHash'] or
                    metadata['partitionHash']!=run['config']['partitionHash']):
                raise ValueError('Checkpoint identity differs from its executed run.')
            for digest,count in metadata['weights'].items():
                if params_hash([checkpoint[f'{digest}_{i}'] for i in range(count)])!=digest:
                    raise ValueError('Checkpoint model weights are corrupt.')
            memory=validate_memory_checkpoint(metadata['cartMemory'],rounds*10,seed)
            if state_hash(memory)!=metadata['cartMemoryHash']:raise ValueError('Local cache identity differs.')
            if metadata.get('version')==2:
                if len(metadata['telemetryRefs'])!=len(telemetry):raise ValueError('Incomplete checkpoint telemetry.')
                for digest,record in zip(metadata['telemetryRefs'],telemetry):
                    raw=gzip.decompress((output/'checkpoint_telemetry'/f'{digest}.json.gz').read_bytes())
                    if hashlib.sha256(raw).hexdigest()!=digest or json.loads(raw)!=record:
                        raise ValueError('Checkpoint and exported telemetry differ.')
        gradient_key='cartRetentionGradientNorm' if run['config']['cartMemoryMode']=='kl' else 'cartBarrierGradientNorm'
        if run['optimizerSteps']!=len(steps):raise ValueError('Saved optimizer-step total differs from telemetry.')
        correction=[s[gradient_key] for s in steps]
        ratios=[s[gradient_key]/(s['baseGradientNorm']+1e-12) for s in steps]
        if run['config'].get('cartCorrectionRatioCap')==.1 and any(v>.10001 for v in ratios):
            raise ValueError('The predeclared retention-gradient budget was exceeded.')
        row=dict(method=name,validationMean=means,validationWorst=worst,runtimeSeconds=run['runtimeSeconds'],
            optimizerSteps=len(steps),activeRetentionSteps=sum(v>0 for v in correction),
            meanRetentionBaseGradientRatio=float(np.mean(ratios)),
            peakGpuMiB=run['memorySafety']['maximumGpuPeakMiB'])
        rows.append(row)
    if any(identity!=identities[0] for identity in identities):raise ValueError('Conditions differ in source, data or runtime.')
    by_name={r['method']:r for r in rows}
    for row in rows:
        suffix=row['method'].rsplit('_',1)[-1]
        reference=by_name['prior_'+suffix] if suffix in 'ABCD' and len(suffix)==1 else by_name['prior']
        row['finalMeanDifferencePercentagePoints']=100*(row['validationMean'][-1]-reference['validationMean'][-1])
        row['finalWorstDifferencePercentagePoints']=100*(row['validationWorst'][-1]-reference['validationWorst'][-1])
    result=dict(status='saved_data_verified',officialTestLoaded=False,seed=seed,stage=summary['stage'],methods=rows,
        interpretation='One seed, TRAIN holdout only: exploratory evidence, not a final TEST or robustness claim.')
    (root/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Fixed TRAIN-only CART comparison','',result['interpretation'],'',
        '| Method | Round 1 mean | Round 5 mean | Final mean | Final worst | Mean difference vs paired prior |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[row['validationMean'][r] for r in (0,4,rounds-1)]
        lines.append(f'| {row["method"]} | '+ ' | '.join(f'{v:.2%}' for v in values)+
            f' | {row["validationWorst"][-1]:.2%} | {row["finalMeanDifferencePercentagePoints"]:+.3f} pp |')
    (root/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for row in rows:
            for ax,key in zip(axes,('validationMean','validationWorst')):
                ax.plot(range(1,rounds+1),np.array(row[key])*100,marker='o',label=row['method'])
        for ax,title in zip(axes,('Mean across nodes','Worst node')):
            ax.set(xlabel='Completed ring rounds',ylabel='Held-out TRAIN accuracy (%)',title=title)
            ax.grid(alpha=.25);ax.legend()
        fig.suptitle(f'CART: fixed {rounds}-round TRAIN-only comparison, seed {seed}')
        fig.savefig(root/'validation_trends.png',dpi=160);plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--plot',action='store_true');args=parser.parse_args();root=args.directory.resolve()
    if not root.is_relative_to(ROOT/'.papermerge/local-tests'):parser.error('Use an isolated local-test directory.')
    print(json.dumps(review(root,plot=args.plot),indent=2))


if __name__=='__main__':main()
