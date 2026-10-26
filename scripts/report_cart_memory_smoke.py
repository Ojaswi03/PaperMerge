#!/usr/bin/env python3
"""Read saved CART diagnostic artifacts, verify reductions, and plot measured trends."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def review(root, *, plot=False):
    import numpy as np
    from basil_core.cart_local_memory import state_hash
    from basil_core.cart_memory_study import validate_memory_checkpoint
    from basil_core.research_protocol import params_hash
    rows=[]
    identities=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/'run.json').is_file():
            continue
        run=json.loads((directory/'run.json').read_text())
        if run['status']!='completed' or run['completedActivations']!=30 or run['completedRounds']!=3:
            raise ValueError(f'{directory.name}: completed three-round artifacts required.')
        cfg=run['config']
        if cfg['researchValid'] is not False:
            raise ValueError('These are diagnostic reports, not production claims.')
        for filename,digest in run['implementationSha256'].items():
            if hashlib.sha256((directory/'source_snapshots'/digest).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Corrupt executed source archive: {filename}')
        identities.append((run['implementationSha256'],run['dataSha256'],run['runtime']))
        records=json.loads(gzip.decompress((directory/'activation_telemetry.json.gz').read_bytes()))
        if [(r['round'],r['nodeId']) for r in records]!=[(r,n) for r in range(3) for n in range(10)]:
            raise ValueError('Noncontiguous diagnostic activations.')
        with np.load(directory/'metrics.npz',allow_pickle=False) as metrics:
            for key in metrics.files:
                if not np.isfinite(metrics[key]).all():
                    raise ValueError(f'Non-finite measured metric: {key}')
            accuracy=metrics['roundNodeAccuracy']
            np.testing.assert_array_equal(metrics['averageAccuracy'],accuracy.mean(axis=1))
            np.testing.assert_array_equal(metrics['worstNodeAccuracy'],accuracy.min(axis=1))
            mean,worst=metrics['averageAccuracy'].tolist(),metrics['worstNodeAccuracy'].tolist()
        with (directory/'round_metrics.csv').open() as handle:
            saved=list(csv.DictReader(handle))
        if [int(r['round']) for r in saved]!=list(range(3)):
            raise ValueError('CSV round cursor mismatch.')
        np.testing.assert_array_equal([float(r['mean']) for r in saved],mean)
        np.testing.assert_array_equal([float(r['worst']) for r in saved],worst)
        np.testing.assert_array_equal(mean,run['meanAccuracy'])
        steps=[s for r in records for s in r['optimizerTelemetry']]
        for r in records:
            if r['inputHash']!=r['snapshotAnchorReferenceHash']:
                raise ValueError('Selected snapshot/anchor mismatch.')
            expected_steps=5*((r['localSampleCount']+511)//512)
            if r['optimizerSteps']!=expected_steps or len(r['optimizerTelemetry'])!=expected_steps:
                raise ValueError('Full five-epoch sample accounting failed.')
            expected_attack=bool(cfg['attackHidden'] and r['configuredByzantine'] and r['round']>=cfg['attackStartRound'])
            if r['attackActive']!=expected_attack or r['channelNoiseActive']!=cfg['useChannelNoise']:
                raise ValueError('Attack/channel branch differs from the declared diagnostic.')
            if len(r['cartMemory']['epochControls'])!=6:
                raise ValueError('Five epoch controls plus final telemetry required.')
            for step in r['optimizerTelemetry']:
                controls=r['cartMemory']['epochControls'][step['epoch']-1]['classes']
                expected_weight=np.mean([c['weight'] for c in controls])
                np.testing.assert_allclose(step['cartMeanRegressionWeight'],expected_weight,atol=1e-7)
                expected_c=.0001 if cfg['ebmMode']=='gradient_norm_objective' else 0.
                np.testing.assert_allclose(step['ebmCoefficient'],expected_c,atol=1e-10)
                if cfg['cartBeta']==0 and step['cartRetentionGradientNorm']!=0:
                    raise ValueError('Disabled CART applied a retention correction.')
        with np.load(directory/'checkpoint.npz',allow_pickle=False) as checkpoint:
            metadata=json.loads(str(checkpoint['metadata']))
            memory=validate_memory_checkpoint(metadata['cartMemory'],30,cfg['seed'])
            if metadata['cartMemoryHash']!=state_hash(memory):
                raise ValueError('Checkpoint memory hash differs.')
            for digest,count in metadata['weights'].items():
                if params_hash([checkpoint[f'{digest}_{i}'] for i in range(count)])!=digest:
                    raise ValueError('Checkpoint model weights changed.')
        ratio=[s['cartRetentionGradientNorm']/(s['baseGradientNorm']+1e-12) for s in steps]
        row=dict(condition=directory.name,meanAccuracy=mean,worstAccuracy=worst,
            steps=len(steps),retentionActiveSteps=sum(s['cartRetentionGradientNorm']>0 for s in steps),
            meanRetentionToBaseGradientRatio=float(np.mean(ratio)),
            maxRetentionGradientNorm=max(s['cartRetentionGradientNorm'] for s in steps),
            attackActivations=sum(r['attackActive'] for r in records),
            noisyActivations=sum(r['channelNoiseActive'] for r in records),
            runtimeSeconds=run['runtimeSeconds'],peakGpuMiB=run['memorySafety']['maximumGpuPeakMiB'])
        rows.append(row)
    if not rows or any(identity!=identities[0] for identity in identities):
        raise ValueError('Conditions must share executed sources, data and runtime.')
    comparison=None
    by_name={r['condition']:r for r in rows}
    if {'A','A_beta0'} <= by_name.keys():
        comparison=dict(finalMeanDifferencePercentagePoints=100*(by_name['A']['meanAccuracy'][-1]-by_name['A_beta0']['meanAccuracy'][-1]),
            finalWorstDifferencePercentagePoints=100*(by_name['A']['worstAccuracy'][-1]-by_name['A_beta0']['worstAccuracy'][-1]),
            interpretation='Single seed, three rounds: an early diagnostic, not an established retention or generalization benefit.')
    result=dict(status='verified',productionRunStarted=False,conditions=rows,matchedCleanComparison=comparison)
    (root/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# CART local diagnostic results','',
        'Measured early results only. No production readiness or statistical benefit is established.','',
        '| Condition | Round 1 mean | Round 2 mean | Round 3 mean | Final worst | Active retention steps |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        values=' | '.join(f'{v:.2%}' for v in row['meanAccuracy'])
        lines.append(f'| {row["condition"]} | {values} | {row["worstAccuracy"][-1]:.2%} | {row["retentionActiveSteps"]}/{row["steps"]} |')
    if comparison:
        lines+=['',f'CART minus matched control: {comparison["finalMeanDifferencePercentagePoints"]:+.3f} percentage points final mean.',
            comparison['interpretation']]
    (root/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for row in rows:
            axes[0].plot([1,2,3],np.array(row['meanAccuracy'])*100,marker='o',label=row['condition'])
            axes[1].plot([1,2,3],np.array(row['worstAccuracy'])*100,marker='o',label=row['condition'])
        for ax,title in zip(axes,('Mean node accuracy','Worst node accuracy')):
            ax.set(title=title,xlabel='Completed ring rounds',ylabel='Evaluation accuracy (%)',xticks=[1,2,3])
            ax.grid(alpha=.25);ax.legend()
        fig.suptitle('CART local diagnostic: one seed, three rounds')
        fig.savefig(root/'learning_trends.png',dpi=160)
        plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    root=args.directory.resolve()
    if not root.is_relative_to(ROOT/'.papermerge/local-tests'):
        parser.error('Reports must stay with isolated .papermerge/local-tests artifacts.')
    print(json.dumps(review(root,plot=args.plot),indent=2))


if __name__=='__main__':main()
