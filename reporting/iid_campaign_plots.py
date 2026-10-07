"""Four-way trajectories and prespecified contrasts; no automatic extension."""
import json
from pathlib import Path
import numpy as np
from basil_core.iid_campaign import ROOT,CONDITIONS,COMPARISON
from reporting.iid_study_plots import load_run,plot

CONTRASTS={
    'attack_effect':('A','B','Attacks while BASIL is active; not the cost of BASIL'),
    'noise_effect':('B','C','Absolute Gaussian channel effect under CE'),
    'ebm_effect':('C','D','Source EBM effect under the matched noisy channel'),
    'total_robustness_gap':('A','D','Combined attacker/noise/EBM condition versus clean BASIL'),
}


def convergence(metrics,*,attack_applicable):
    accuracy=np.asarray(metrics['fullTestAccuracy'])
    if accuracy.shape!=(100,) or not np.isfinite(accuracy).all():raise ValueError('A completed condition requires 100 actual finite rounds.')
    return dict(final_accuracy=float(accuracy[-1]),best_accuracy=float(accuracy.max()),best_round=int(accuracy.argmax()),
        mean_rounds_0_19=float(accuracy[:20].mean()),mean_rounds_20_99=float(accuracy[20:].mean()),
        attack_applicable=attack_applicable,mean_rounds_80_89=float(accuracy[80:90].mean()),
        late_round_mean=float(accuracy[90:100].mean()),late_round_change=float(accuracy[90:100].mean()-accuracy[80:90].mean()),
        final_worst_node_accuracy=float(metrics['worstNodeAccuracy'][-1]))


def scientific_contrasts(stats):
    result={}
    for name,(left,right,meaning) in CONTRASTS.items():
        if left not in stats or right not in stats:continue
        result[name]=dict(comparison=f'{right} minus {left}',meaning=meaning,
            differences={key:stats[right][key]-stats[left][key] for key in ('final_accuracy','best_accuracy','late_round_mean','final_worst_node_accuracy')})
    return result


def comparison_plots(loaded,output):
    if set(loaded)!=set('ABCD'):raise ValueError('All four completed conditions are required.')
    identity=[(m['nRounds'],m['iidPartitionHash'],m['initialModelHash']) for m,_,_ in loaded.values()]
    if len(set(identity))!=1 or identity[0][0]!=100:raise ValueError('Comparison requires paired 100-round runs.')
    if len({m.get('device','CPU') for m,_,_ in loaded.values()})!=1:
        raise ValueError('Do not mix CPU and GPU trajectories in a paired study.')
    for name,key in (('ABCD_full_test_accuracy_comparison','fullTestAccuracy'),
        ('ABCD_mean_node_accuracy_comparison','averageAccuracy'),('ABCD_worst_node_accuracy_comparison','worstNodeAccuracy')):
        if any(np.asarray(v[key]).shape!=(100,) for _,v,_ in loaded.values()):raise ValueError('Do not plot fabricated or truncated horizons.')
        plot(output,name,{f'{letter} — {CONDITIONS[letter][5]}':loaded[letter][1][key] for letter in 'ABCD'},title='100-round full-data IID BASIL — fixed A/B/C/D conditions')
    plot(output,'ABCD_final_per_class_comparison',{letter:loaded[letter][1]['fullTestPerClassAccuracy'][-1] for letter in 'ABCD'},xlabel='Class ID',title='Final per-class accuracy — fixed Node 9 reference')


def report_campaign(root=ROOT, *, device='CPU'):
    from basil_core.iid_runtime import execution_device
    execution_device({'executionDevice':device})
    suffix='_gpu' if device=='GPU' else ''
    root=Path(root);output=root/'newResults/IID'/(COMPARISON+suffix);output.mkdir(parents=True,exist_ok=True)
    loaded={};status={};stats={};transitions={}
    for letter,item in CONDITIONS.items():
        directory=root/'newResults/IID'/(item[1]+suffix);path=directory/'run.json'
        status[letter]='not_started' if not path.exists() else json.loads(path.read_text())['status']
        if path.exists():
            transitions[letter]=json.loads(path.read_text()).get('deviceTransition')
        if status[letter]=='completed':
            loaded[letter]=load_run(directory)
            stats[letter]=convergence(loaded[letter][1],attack_applicable=letter!='A')
    evidence=dict(status=status,conditions=stats,contrasts=scientific_contrasts(stats),
        signConvention='right minus left; positive means higher accuracy, not guaranteed benefit or statistical significance',
        horizon=100,automaticExtension=False,testUsage='evaluation_only',device=device,deviceTransitions=transitions)
    (output/'analysis.json').write_text(json.dumps(evidence,indent=2)+'\n')
    lines=['# Four-condition 100-round IID BASIL','',
        'All runs use BASIL. A/B measures attacks under BASIL; B/C measures channel noise; C/D is the matched EBM comparison; A/D is the total robustness gap.','',
        'Differences are right minus left in accuracy units; multiply by 100 for percentage points. No ordering or improvement is assumed. One paired seed is not evidence of statistical significance.','']
    for letter in 'ABCD':lines += [f"## {letter} — {status[letter]}",'',json.dumps(stats.get(letter,{'metrics':'unavailable until completed'}),indent=2),'']
    lines += ['## Scientific contrasts','',json.dumps(evidence['contrasts'],indent=2),'',
        'Round means 80–89 and 90–99 describe late trends only. They never extend or stop the fixed 100-round horizon.']
    if any(transitions.values()):lines += ['', '## Execution-device transitions','',json.dumps(transitions,indent=2),'',
        'Recorded CPU prefixes are retained; subsequent updates use GPU. These are mixed-device trajectories, not fresh all-GPU paired runs. Backend rounding can affect trajectories and is an additional interpretation limitation.']
    (output/'ABCD_100R_REPORT.md').write_text('\n'.join(lines)+'\n')
    if len(loaded)==4:comparison_plots(loaded,root/'newPlots/IID'/(COMPARISON+suffix))
    return evidence
