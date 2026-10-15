"""Read-only review of the predeclared three-round calibration pair."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path


def review(source, *, names=('baseline','calibrated')):
    import numpy as np
    from basil_core.dirichlet_diagnosis import summarize_activations
    source=Path(source)
    summaries={}; histories={}
    manifests={}
    allowed=(('baseline','calibrated'),('baseline','calibrated','prior_adjusted'),('baseline','prior_adjusted'),('noise_ce','noise_ebm'),tuple('BCD'),tuple('ABCD'))
    if tuple(names) not in allowed:raise ValueError('Declared diagnostic comparison order required.')
    for name in names:
        path=source/name
        manifest=json.loads((path/'run.json').read_text())
        manifests[name]=manifest
        rounds=manifest.get('config',{}).get('nRounds',3)
        if manifest['status']!='completed' or manifest['completedRounds']!=rounds or manifest['completedActivations']!=10*rounds:
            raise ValueError('All completed declared conditions required; no partial conclusions.')
        partition=json.loads((path/'partition_audit.json').read_text())
        records=json.loads(gzip.decompress((path/'activation_telemetry.json.gz').read_bytes()))
        contracts=summarize_activations(records,partition,manifest['config'])
        with (path/'round_metrics.csv').open() as handle:rows=list(csv.DictReader(handle))
        if [int(r['round']) for r in rows]!=list(range(rounds)):
            raise ValueError('CSV rounds missing or duplicated.')
        with np.load(path/'metrics.npz',allow_pickle=False) as metrics:
            full=metrics['fullTestAccuracy'].copy()
            mean=metrics['averageAccuracy'].copy();worst=metrics['worstNodeAccuracy'].copy()
            node_class=metrics['roundNodePerClassAccuracy'][-1].copy()
            if any(not np.isfinite(metrics[k]).all() for k in metrics.files if metrics[k].dtype.kind in 'fci'):
                raise ValueError('Non-finite completed result array.')
            np.testing.assert_allclose(full,metrics['roundNodeAccuracy'][:,9],atol=1e-8,rtol=0)
            np.testing.assert_allclose(mean,metrics['roundNodeAccuracy'].mean(axis=1),atol=1e-8,rtol=0)
            np.testing.assert_allclose(worst,metrics['roundNodeAccuracy'].min(axis=1),atol=1e-8,rtol=0)
            np.testing.assert_allclose(metrics['fullTestPerClassAccuracy'],metrics['roundNodePerClassAccuracy'][:,9],atol=1e-8,rtol=0)
            for field,array in [('mean_node_accuracy',mean),('worst_node_accuracy',worst),('full_test_loss',metrics['fullTestLoss'])]:
                np.testing.assert_allclose(array,[float(r[field]) for r in rows],atol=1e-8,rtol=0)
            np.testing.assert_allclose(metrics['fullTestPerClassAccuracy'],
                [[float(r[f'class_{c}_accuracy']) for c in range(10)] for r in rows],atol=1e-8,rtol=0)
        if not np.allclose(full,[float(r['full_test_accuracy']) for r in rows],atol=1e-8,rtol=0):
            raise ValueError('CSV/NPZ metric mismatch.')
        steps=[s for r in records for s in r['optimizerTelemetry']]
        cfg=manifest['config']
        expected_sigma=cfg['channelNoiseSigmaAbsolute'] if cfg['useChannelNoise'] else 0.
        expected_coefficient=expected_sigma**2 if cfg['ebmMode']=='gradient_norm_objective' else 0.
        np.testing.assert_allclose([s['ebmCoefficient'] for s in steps],expected_coefficient,rtol=1e-6,atol=1e-12)
        np.testing.assert_allclose([s['effectiveCoordinateSigma'] for s in steps],expected_sigma,rtol=1e-6,atol=1e-12)
        for s in steps:
            np.testing.assert_allclose(s['ebmPenalty'],s['ebmCoefficient']*s['gradientNormSquared'],atol=1e-8,rtol=1e-6)
            np.testing.assert_allclose(s['snapshotAnchorPenalty'],.005*s['snapshotDistanceSquared'],atol=1e-8,rtol=1e-6)
            np.testing.assert_allclose(s['ebmObjective'],s['baseCrossEntropy']+s['ebmPenalty']+s['snapshotAnchorPenalty'],atol=1e-6,rtol=1e-6)
        from basil_core.dirichlet_calibration import calibration_offsets
        for r in records:
            counts=np.asarray(partition['nodes'][r['nodeId']]['classCounts'])
            np.testing.assert_array_equal(r['localClassCounts'],counts)
            expected_offsets=calibration_offsets(counts,method=cfg.get('logitCalibrationMethod','fedlc')) if cfg['logitCalibrationEnabled'] else np.zeros(10)
            np.testing.assert_allclose(r['logitCalibrationOffsets'],expected_offsets,atol=1e-7,rtol=1e-6)
        ordinary=sum(s['ordinaryCrossEntropy']*s['samples'] for s in steps)/sum(s['samples'] for s in steps)
        before=np.array([r['beforeTraining']['perClassGlobalTestAccuracy'] for r in records])
        after=np.array([r['perClassGlobalTestAccuracy'] for r in records])
        absent=np.array([np.asarray(partition['nodes'][r['nodeId']]['classCounts'])==0 for r in records])
        missing_forgetting=(before-after)[absent]
        summaries[name]=dict(finalFullTestAccuracy=float(full[-1]),finalMeanNodeAccuracy=float(mean[-1]),
            finalWorstNodeAccuracy=float(worst[-1]),fullTestAccuracyByRound=full.tolist(),
            finalPerClassAccuracy=node_class[9].tolist(),finalNodePerClassAccuracy=node_class.tolist(),
            meanActivationClassForgettingPp=100*float(np.mean(before-after)),
            meanAbsentClassForgettingPp=100*float(missing_forgetting.mean()),
            sampleWeightedOrdinaryTrainingCE=float(ordinary),
            allLocalAccuracyScopes={r['localTrainingAccuracyScope'] for r in records},
            runtimeSeconds=manifest['runtimeSeconds'],memory=manifest['memorySafety'],
            contracts=contracts,partitionHash=manifest['partitionHash'],initialModelHash=manifest['initialModelHash'])
        with np.load(path/'metrics.npz',allow_pickle=False) as metrics:
            summaries[name]['finalNodeAccuracy']=metrics['roundNodeAccuracy'][-1].tolist()
        summaries[name].update(optimizerSteps=len(steps),rounds=rounds,
            meanEbmPenalty=float(np.mean([s['ebmPenalty'] for s in steps])),
            meanEbmCorrectionRatio=float(np.mean([s['ebmCorrectionRatio'] for s in steps])),
            meanOrdinaryGradientNorm=float(np.mean([s['baseGradientNorm'] for s in steps])),
            meanNoiseNorm=float(np.mean([link['noiseL2Norm'] for r in records for link in r['outgoingLinks']])),
            meanNoiseToModelRatio=float(np.mean([link['relativeNoiseL2'] for r in records for link in r['outgoingLinks']])),
            selectedCorruptByRound=[sum(1 for r in records if r['round']==i and
                manifest['config']['attackHidden'] and r['inputSourceNode'] in manifest['config']['resolvedAttackerIds'] and
                r['inputSnapshotRound']>=manifest['config']['attackStartRound']) for i in range(rounds)])
        summaries[name]['allLocalAccuracyScopes']=sorted(summaries[name]['allLocalAccuracyScopes'])
        histories[name]=dict(full=full,mean=mean,worst=worst)
    for key in ('partitionHash','initialModelHash'):
        if any(summaries[names[0]][key]!=summaries[name][key] for name in names):raise ValueError('Unpaired result identity.')
    if len({c['config']['nRounds'] for c in manifests.values()})!=1:
        raise ValueError('Cannot compare different diagnostic horizons.')
    return dict(status='completed',summaries=summaries,
        finalFullTestDifferencePp={name:100*(summaries[name]['finalFullTestAccuracy']-summaries[names[0]]['finalFullTestAccuracy'])
            for name in names[1:]},
        researchValid=False,
        attackRobustnessTested=any(c['config']['attackHidden'] and c['config']['nRounds']>20 for c in manifests.values()),
        noiseMitigationTested=any(c['config']['useChannelNoise'] for c in manifests.values())),histories


def generate_plots(result,histories,directory):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    if directory.exists():raise ValueError('Refusing to overwrite completed diagnostic plots.')
    directory.mkdir(parents=True)
    names=list(result['summaries'])
    colors=dict(baseline='#1565c0',calibrated='#d84315',prior_adjusted='#7b1fa2',
        noise_ce='#1565c0',noise_ebm='#d84315',A='#1565c0',B='#d84315',C='#00897b',D='#7b1fa2')
    rounds=len(histories[names[0]]['full'])
    for field,title in [('full','Node-9 full-test accuracy'),('mean','Mean-node full-test accuracy'),('worst','Worst-node full-test accuracy')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for name in names:
            ax.plot(range(rounds),100*histories[name][field],marker='o',color=colors[name],label=name)
        ax.set(xlabel='Zero-based full ring round',ylabel='Accuracy (%)',ylim=(0,100),
            title=f'{rounds} full-data diagnostic rounds — {title}')
        if rounds>20:ax.axvline(20,color='#444',linestyle='--',label='Attack starts (B/C/D)')
        ax.legend();fig.tight_layout();fig.savefig(directory/f'{field}_accuracy_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,len(names),figsize=(6*len(names),4),sharey=True,squeeze=False)
    axes=axes[0]
    for ax,name in zip(axes,names):
        image=ax.imshow(100*np.asarray(result['summaries'][name]['finalNodePerClassAccuracy']),vmin=0,vmax=100,cmap='viridis')
        ax.set(title=f'{name}: final node/class recall',xlabel='CIFAR class ID',xticks=range(10),yticks=range(10))
    axes[0].set_ylabel('Node');fig.colorbar(image,ax=axes,label='Class accuracy (%)',shrink=.8)
    fig.savefig(directory/'final_node_class_recall.png',dpi=150,bbox_inches='tight');plt.close(fig)
    return sorted(p.name for p in directory.glob('*.png'))


def verify_final_transmissions(source, names):
    """Replay last-ring attack/noise and inspect checkpoint memories, no SGD.

    Keyed streams require no mutable RNG restoration. This checks the actual
    saved model parameters, not merely boolean configuration fields.
    """
    import numpy as np
    import hashlib
    from basil_core.protocol_checkpoint import _load_telemetry
    from basil_core.research_protocol import applyAttack,apply_channel_noise,keyed_rng,params_hash,copy_params
    outcomes={}
    for name in names:
        directory=Path(source)/name
        manifest=json.loads((directory/'run.json').read_text())
        if manifest['status']!='completed':raise ValueError('Final transmission audit requires completed evidence.')
        cfg=manifest['config']
        # Read-only decoding is distinct from permission to resume. Operational
        # resumes still reject any changed source. Archived source bytes permit
        # reviewing a completed run after diagnostic-only refactoring.
        for filename,digest in manifest['implementationSha256'].items():
            archive=directory/'source_snapshots'/digest
            if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:
                raise ValueError('Executed-source archive missing or changed.')
        root=Path(__file__).resolve().parents[1]
        for filename in ('basil_core/attacks.py','basil_core/research_protocol.py'):
            if hashlib.sha256((root/filename).read_bytes()).hexdigest()!=manifest['implementationSha256'][filename]:
                raise ValueError('Transmission mathematics changed; current replay is not permitted.')
        with np.load(directory/'checkpoint.npz',allow_pickle=False) as packed:
            metadata=json.loads(str(packed['metadata']))
            if (metadata['config']!=cfg or metadata['implementation']!=manifest['implementationSha256'] or
                    metadata['partitionHash']!=manifest['partitionHash'] or metadata['initialHash']!=manifest['initialModelHash'] or
                    metadata['completedRounds']!=cfg['nRounds'] or metadata['nextActivation']!=cfg['nRounds']*10):
                raise ValueError('Final checkpoint identity/cursor is incomplete or inconsistent.')
            def unpack(digest):
                values=[packed[f'{digest}_{i}'].copy() for i in range(metadata['weights'][digest])]
                if params_hash(values)!=digest:raise ValueError('Saved checkpoint parameter hash differs.')
                return values
            nodes=[unpack(h) for h in metadata['nodeParams']]
            memories=[{s['sender']:(s['round'],unpack(s['params'])) for s in memory} for memory in metadata['memories']]
        records=_load_telemetry(directory/'checkpoint.npz',metadata)
        round_id=cfg['nRounds']-1
        for node,record in enumerate(records[-10:]):
            params=nodes[node]
            if params_hash(params)!=record['trainedOutputHash']:raise ValueError('Saved final trained model hash differs.')
            active=cfg['attackHidden'] and node in cfg['resolvedAttackerIds'] and round_id>=20
            outbound=applyAttack(params,'hidden',rng=keyed_rng(cfg['seed'],'hidden_attack',round_id,node)) if active else copy_params(params)
            if params_hash(outbound)!=record['outboundAttackHash']:raise ValueError('Attack replay differs.')
            link_hashes=[]
            for link in record['outgoingLinks']:
                receiver=link['receiverId']
                if cfg['useChannelNoise']:
                    received,stats=apply_channel_noise(outbound,semantics='paper_absolute_gaussian',sigma=.01,
                        rng=keyed_rng(cfg['seed'],'channel_noise',round_id,node,receiver))
                    for key in ('coordinateSigma','noiseL2Norm','relativeNoiseL2'):
                        np.testing.assert_allclose(stats[key],link[key],rtol=1e-12,atol=1e-12)
                else:received=outbound
                expected=params_hash(received)
                saved_round,snapshot=memories[receiver][node]
                if expected!=link['transmittedHash'] or expected!=params_hash(snapshot) or saved_round!=round_id:
                    raise ValueError('Per-link received snapshot / rolling memory differs.')
                link_hashes.append(expected)
            if cfg['useChannelNoise'] and len(set(link_hashes))!=5:
                raise ValueError('Final outgoing links did not have independent noise realizations.')
        outcomes[name]=dict(finalDirectedLinksVerified=50,exactSavedMemory=True,attackBeforeNoise=True,
            independentNoisyLinks=bool(cfg['useChannelNoise']),zeroTrainingUpdates=True,
            completedRounds=metadata['completedRounds'])
    return outcomes
