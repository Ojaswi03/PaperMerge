"""Analysis-only figures from immutable saved results; no training dependencies."""
from __future__ import annotations


def generate_plots(diagnosis, directory):
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(directory)
    if output.exists():
        raise ValueError('Refusing to replace existing diagnostic plots.')
    output.mkdir(parents=True)
    colors = dict(A='#1565c0', B='#d84315', C='#7b1fa2', D='#00897b')
    rows = diagnosis['conditions']
    fig, ax = plt.subplots(figsize=(8, 4))
    for letter, row in rows.items():
        ax.plot(range(len(row['node9NormByRound'])), row['node9NormByRound'], label=letter, color=colors[letter])
    ax.axvline(20, color='black', ls='--', label='Attack starts R20 (B/C/D)')
    ax.set(xlabel='Zero-based full ring round', ylabel='Honest trained Node-9 parameter L2 norm', yscale='log')
    ax.legend(); fig.tight_layout(); fig.savefig(output / 'parameter_norm_collapse.png', dpi=150); plt.close(fig)

    matrix = np.asarray([n['meanPerClassChangePp'] for n in rows['A']['nodes']])
    limit = max(float(np.abs(matrix).max()), 1.)
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap='RdBu')
    ax.set(xticks=range(10), yticks=range(10), xlabel='CIFAR class ID', ylabel='Active node',
           title='Clean A: mean per-class accuracy change during five local epochs')
    for node in range(10):
        for cls in range(10):
            ax.text(cls, node, f'{matrix[node, cls]:.0f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(matrix[node, cls]) > limit*.55 else 'black')
    fig.colorbar(image, ax=ax, label='After − before (percentage points); negative = forgetting')
    fig.tight_layout(); fig.savefig(output / 'clean_local_class_retention.png', dpi=150); plt.close(fig)

    first = rows['B']['firstCorruptedSelection']
    if first:
        candidates = first['candidates']
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"N{c['senderId']} R{c['snapshotRound']}" for c in candidates]
        corrupt = [c['senderByzantine'] and c['snapshotRound'] >= 20 for c in candidates]
        ax.bar(range(5), [c['receiverLocalLoss'] for c in candidates],
               color=['#d84315' if bad else '#1565c0' for bad in corrupt])
        ax.axhline(np.log(10), color='black', ls=':', label='Uniform ten-class CE: ln(10)')
        ax.set(xticks=range(5), xticklabels=labels, ylabel='Receiver-local TRAIN cross-entropy',
            title=f"B: first corrupted selection, R{first['round']} receiver {first['node']}")
        ax.legend(); fig.tight_layout(); fig.savefig(output / 'first_corrupted_selection.png', dpi=150); plt.close(fig)
    return sorted(p.name for p in output.glob('*.png'))
