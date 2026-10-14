"""Four-way trajectories and prespecified contrasts; no automatic extension."""
import json
from pathlib import Path
from basil_core.iid_campaign import ROOT,CONDITIONS,COMPARISON,study_layout
from reporting.iid_study_plots import load_run,plot

CONTRASTS={
    'attack_effect':('A','B','Attacks while BASIL is active; not the cost of BASIL'),
    'noise_effect':('B','C','Absolute Gaussian channel effect under CE'),
    'ebm_effect':('C','D','Source EBM effect under the matched noisy channel'),
    'total_robustness_gap':('A','D','Combined attacker/noise/EBM condition versus clean BASIL'),
}


def convergence(metrics,*,attack_applicable,expected_rounds=None):
    import numpy as np
    accuracy=np.asarray(metrics['fullTestAccuracy'])
    if accuracy.ndim!=1 or len(accuracy) not in (100,500) or not np.isfinite(accuracy).all():raise ValueError('A completed condition requires 100 or 500 actual finite rounds.')
    horizon=len(accuracy)
    if expected_rounds is not None and horizon!=expected_rounds:
        raise ValueError('Measured rounds do not match the requested study horizon.')
    return dict(final_accuracy=float(accuracy[-1]),best_accuracy=float(accuracy.max()),best_round=int(accuracy.argmax()),
        mean_rounds_0_19=float(accuracy[:20].mean()),mean_rounds_20_99=float(accuracy[20:100].mean()),
        mean_post_attack=float(accuracy[20:].mean()),post_attack_rounds=[20,horizon-1],
        attack_applicable=attack_applicable,mean_rounds_80_89=float(accuracy[80:90].mean()),
        horizon=horizon,late_window_previous=[horizon-20,horizon-11],late_window_final=[horizon-10,horizon-1],
        late_round_mean=float(accuracy[-10:].mean()),late_round_change=float(accuracy[-10:].mean()-accuracy[-20:-10].mean()),
        final_worst_node_accuracy=float(metrics['worstNodeAccuracy'][-1]))


def scientific_contrasts(stats):
    result={}
    for name,(left,right,meaning) in CONTRASTS.items():
        if left not in stats or right not in stats:continue
        result[name]=dict(comparison=f'{right} minus {left}',meaning=meaning,
            differences={key:stats[right][key]-stats[left][key] for key in ('final_accuracy','best_accuracy','late_round_mean','final_worst_node_accuracy')})
    return result


def comparison_plots(loaded,output):
    import numpy as np
    if set(loaded)!=set('ABCD'):raise ValueError('All four completed conditions are required.')
    identity=[(m['nRounds'],m['iidPartitionHash'],m['initialModelHash']) for m,_,_ in loaded.values()]
    if len(set(identity))!=1 or identity[0][0] not in (100,500):raise ValueError('Comparison requires paired 100- or 500-round runs.')
    horizon=identity[0][0]
    if len({m.get('device','CPU') for m,_,_ in loaded.values()})!=1:
        raise ValueError('Do not mix CPU and GPU trajectories in a paired study.')
    for name,key in (('ABCD_full_test_accuracy_comparison','fullTestAccuracy'),
        ('ABCD_mean_node_accuracy_comparison','averageAccuracy'),('ABCD_worst_node_accuracy_comparison','worstNodeAccuracy')):
        if any(np.asarray(v[key]).shape!=(horizon,) for _,v,_ in loaded.values()):raise ValueError('Do not plot fabricated or truncated horizons.')
        plot(output,name,{f'{letter} — {CONDITIONS[letter][5]}':loaded[letter][1][key] for letter in 'ABCD'},title=f'{horizon}-round full-data IID BASIL — fixed A/B/C/D conditions')
    plot(output,'ABCD_final_per_class_comparison',{letter:loaded[letter][1]['fullTestPerClassAccuracy'][-1] for letter in 'ABCD'},xlabel='Class ID',title='Final per-class accuracy — fixed Node 9 reference')


def report_campaign(root=ROOT, *, device='CPU', rounds=100):
    _,conditions,comparison=study_layout(rounds)
    from basil_core.iid_runtime import execution_device
    execution_device({'executionDevice':device})
    suffix='_gpu' if device=='GPU' else ''
    root=Path(root);output=root/'newResults/IID'/(comparison+suffix);output.mkdir(parents=True,exist_ok=True)
    loaded={};status={};stats={};transitions={}
    for letter,item in conditions.items():
        directory=root/'newResults/IID'/(item[1]+suffix);path=directory/'run.json'
        status[letter]='not_started' if not path.exists() else json.loads(path.read_text())['status']
        if path.exists():
            transitions[letter]=json.loads(path.read_text()).get('deviceTransition')
        if status[letter]=='completed':
            loaded[letter]=load_run(directory)
            if loaded[letter][0]['nRounds']!=rounds:raise ValueError('Manifest horizon differs from the requested study.')
            stats[letter]=convergence(loaded[letter][1],attack_applicable=letter!='A',expected_rounds=rounds)
    evidence=dict(status=status,conditions=stats,contrasts=scientific_contrasts(stats),
        signConvention='right minus left; positive means higher accuracy, not guaranteed benefit or statistical significance',
        horizon=rounds,automaticExtension=False,testUsage='evaluation_only',device=device,deviceTransitions=transitions)
    (output/'analysis.json').write_text(json.dumps(evidence,indent=2)+'\n')
    lines=[f'# Four-condition {rounds}-round IID BASIL','',
        'All runs use BASIL. A/B measures attacks under BASIL; B/C measures channel noise; C/D is the matched EBM comparison; A/D is the total robustness gap.','',
        'Differences are right minus left in accuracy units; multiply by 100 for percentage points. No ordering or improvement is assumed. One paired seed is not evidence of statistical significance.','']
    for letter in 'ABCD':lines += [f"## {letter} — {status[letter]}",'',json.dumps(stats.get(letter,{'metrics':'unavailable until completed'}),indent=2),'']
    lines += ['## Scientific contrasts','',json.dumps(evidence['contrasts'],indent=2),'',
        f'Round means {rounds-20}–{rounds-11} and {rounds-10}–{rounds-1} describe late trends only. They never extend or stop the fixed {rounds}-round horizon.']
    if any(transitions.values()):lines += ['', '## Execution-device transitions','',json.dumps(transitions,indent=2),'',
        'Recorded CPU prefixes are retained; subsequent updates use GPU. These are mixed-device trajectories, not fresh all-GPU paired runs. Backend rounding can affect trajectories and is an additional interpretation limitation.']
    (output/f'ABCD_{rounds}R_REPORT.md').write_text('\n'.join(lines)+'\n')
    if len(loaded)==4:comparison_plots(loaded,root/'newPlots/IID'/(comparison+suffix))
    return evidence
