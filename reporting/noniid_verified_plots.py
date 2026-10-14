"""Saved-result plots for the fixed thirty-round verified CART study."""
from __future__ import annotations

import json
from pathlib import Path


def line_plot(output, name, series, *, title, ylabel, attack=False, xlabel='Round (zero-based)'):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    colors=('#0072B2','#D55E00','#009E73','#CC79A7')
    for index,(label,values) in enumerate(series.items()):
        values=np.asarray(values,dtype=float)
        if values.ndim!=1 or not np.isfinite(values).all():
            raise ValueError('Only actual finite measured values may be plotted.')
        ax.plot(range(len(values)),values,label=label,color=colors[index % len(colors)],linewidth=1.8)
    if attack:ax.axvline(20,color='#555555',linestyle='--',label='Attack activates R20')
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    if 'accuracy' in ylabel.lower() or 'rate' in ylabel.lower():ax.set_ylim(0,1)
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/f'{name}.png',dpi=150);plt.close(fig)


def report_condition(directory, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory,output=Path(directory),Path(output)
    run=json.loads((directory/'run.json').read_text())
    rows=run['roundMetrics'];cfg=run['config'];letter=cfg['conditionId']
    if run['status']!='completed' or len(rows)!=30 or [r['round'] for r in rows]!=list(range(30)):
        raise ValueError('Thirty measured completed rounds required; do not use partial evidence as final.')
    output.mkdir(parents=True,exist_ok=True)
    fields={'full_test_accuracy_vs_round':('full_test_accuracy','Node 9 full-TEST accuracy'),
        'mean_node_accuracy_vs_round':('mean_node_accuracy','Mean node full-TEST accuracy'),
        'worst_node_accuracy_vs_round':('worst_node_accuracy','Worst node full-TEST accuracy'),
        'full_test_loss_vs_round':('full_test_loss','Node 9 full-TEST CE'),
        'corrupted_snapshot_selection_rate_vs_round':('corrupted_selection_rate','Actually corrupted selection rate'),
        'attacker_designated_selection_rate_vs_round':('attacker_designated_selection_rate','Attacker-designated selection rate'),
        'selected_snapshot_age_vs_round':('selected_snapshot_age','Snapshot age (rounds)')}
    if letter in 'CD':
        fields.update(channel_noise_norm_vs_round=('channel_noise_norm','Mean link noise L2 norm'),
            noise_to_model_norm_ratio_vs_round=('noise_to_model_norm_ratio','Mean link noise/model L2 ratio'))
    if letter=='D':
        fields.update(ebm_penalty_vs_round=('ebm_penalty','Mean EBM penalty'),
            ebm_correction_norm_vs_round=('ebm_correction_norm','Mean Hessian-gradient correction norm'),
            ebm_correction_ratio_vs_round=('ebm_correction_ratio','Mean correction/local CE gradient norm ratio'))
    for name,(key,ylabel) in fields.items():
        line_plot(output,name,{letter:[r[key] for r in rows]},title=f'{letter}: 30-round full-data non-IID verified CART',ylabel=ylabel,attack=letter!='A')
    fig,ax=plt.subplots(figsize=(11,6))
    classes=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
    for i,label in enumerate(classes):
        ax.plot(range(30),[r[f'class_{i}_accuracy'] for r in rows],color=plt.get_cmap('tab10')(i),label=label)
    if letter!='A':ax.axvline(20,color='#555555',linestyle='--',label='Attack R20')
    ax.set(title=f'{letter}: Node 9 per-class test accuracy',xlabel='Round',ylabel='Accuracy',ylim=(0,1))
    ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output/'per_class_accuracy_vs_round.png',dpi=150);plt.close(fig)
    matrix=np.asarray(rows[-1]['per_node_per_class_accuracy'])
    fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
    ax.set(xlabel='CIFAR-10 class',ylabel='Trained node',xticks=range(10),yticks=range(10),title=f'{letter}: final node-by-class accuracy')
    fig.colorbar(im,ax=ax,label='Accuracy');fig.tight_layout();fig.savefig(output/'final_node_class_heatmap.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.bar(classes,[rows[-1][f'class_{i}_accuracy'] for i in range(10)],color='#0072B2')
    ax.set(ylim=(0,1),ylabel='Accuracy',title=f'{letter}: final Node 9 class accuracy');ax.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(output/'final_per_class_accuracy.png',dpi=150);plt.close(fig)
    final_epochs=json.loads((directory/'activation_0299.json').read_text())['epochs']
    fig,ax=plt.subplots(figsize=(10,5))
    for i,label in enumerate(classes):ax.plot(range(6),[e['perClass'][i] for e in final_epochs],label=label,color=plt.get_cmap('tab10')(i))
    ax.set(xlabel='Local epoch (0 = received selected snapshot)',ylabel='Accuracy',ylim=(0,1),title=f'{letter}: R29 Node 9 five-epoch retention')
    ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(output/'five_epoch_class_retention.png',dpi=150);plt.close(fig)
    for e in final_epochs:
        fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(e['confusion'],cmap='Blues')
        ax.set(xlabel='Predicted class',ylabel='True class',title=f'{letter}: R29 Node 9 epoch {e["epoch"]}',xticks=range(10),yticks=range(10))
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(output/f'final_node9_epoch_{e["epoch"]}_confusion.png',dpi=120);plt.close(fig)
    return dict(rounds=30,plots=len(list(output.glob('*.png'))))


def report_comparison(root, config):
    from basil_core.noniid_verified_study import configurations, verify_pairing
    root=Path(root);configs=configurations(config['partitionHash'],config['initialModelHash']);verify_pairing(configs)
    loaded={}
    for cfg in configs:
        path=root/cfg['resultDirectory']/'run.json'
        if not path.exists():return dict(status='awaiting_conditions',complete=list(loaded))
        run=json.loads(path.read_text())
        if run['status']!='completed':return dict(status='awaiting_conditions',complete=list(loaded))
        if run['config']!=cfg or len(run['roundMetrics'])!=30:
            raise ValueError('Comparison configuration/horizon mismatch.')
        loaded[cfg['conditionId']]=run['roundMetrics']
    output=root/config['comparisonPlotsDirectory']
    for key in ('full_test_accuracy','mean_node_accuracy','worst_node_accuracy','full_test_loss'):
        line_plot(output,f'ABCD_{key}_comparison',{l:[r[key] for r in rows] for l,rows in loaded.items()},
            title='30-round non-IID verified CART — paired A/B/C/D',ylabel=key.replace('_',' '),attack=True)
    final={l:r[-1]['full_test_accuracy'] for l,r in loaded.items()}
    contrasts={'attack_B_minus_A':100*(final['B']-final['A']),
        'noise_C_minus_B':100*(final['C']-final['B']),
        'EBM_D_minus_C':100*(final['D']-final['C']),
        'combined_D_minus_A':100*(final['D']-final['A'])}
    directory=root/config['comparisonResultsDirectory'];directory.mkdir(parents=True,exist_ok=True)
    (directory/'comparison.json').write_text(json.dumps(dict(finalAccuracy=final,contrastsPercentagePoints=contrasts),indent=2)+'\n')
    lines=['# Thirty-round non-IID verified CART A/B/C/D','',
        'Trusted coordinator and replicated TRAIN. One paired seed; no significance or universal robustness claim.','',
        '| Condition | Final Node 9 | Mean node | Worst node | Actually corrupted selections |',
        '|---|---:|---:|---:|---:|']
    for l,rows in loaded.items():
        r=rows[-1]
        lines.append(f"| {l} | {100*r['full_test_accuracy']:.2f}% | {100*r['mean_node_accuracy']:.2f}% | {100*r['worst_node_accuracy']:.2f}% | {sum(round(v['corrupted_selection_rate']*10) for v in rows)} / 300 |")
    lines += ['', '| Contrast | Difference (pp) |','|---|---:|',*[f'| {k} | {v:+.3f} |' for k,v in contrasts.items()],
        '', 'A/B measures the combined model-attack and adversarial-helper environment under certified verification. B/C measures snapshot-channel noise. C/D measures the local-CE EBM term under that matched channel.',
        '', 'No automatic extension or test-set tuning was performed.']
    (directory/'ABCD_30R_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return dict(status='completed',finalAccuracy=final,contrastsPercentagePoints=contrasts)
