"""Analysis-only head/backbone interventions; never an SGD or BASIL rule.

Numerical libraries are loaded on use. Balanced audit probes are not supplied to
training, selection, CART control or parameter selection.
"""
from __future__ import annotations


def classification_metrics(logits, labels, *, classes=10):
    import numpy as np
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    if z.shape != (len(y), classes) or not len(y) or not np.isfinite(z).all():
        raise ValueError('Finite, nonempty ten-logit audit predictions required.')
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= classes)):
        raise ValueError('Valid integer audit labels required.')
    shifted = z - z.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = z.argmax(axis=1)
    confusion = np.zeros((classes, classes), np.int64)
    np.add.at(confusion, (y, prediction), 1)
    support = confusion.sum(axis=1)
    acc = np.divide(np.diag(confusion), support, out=np.full(classes, np.nan), where=support > 0)
    loss = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]
    bias_gradient = probabilities.mean(axis=0) - np.bincount(y, minlength=classes) / len(y)
    return dict(accuracy=float(np.trace(confusion) / len(y)), crossEntropy=float(loss.mean()),
        perClassAccuracy=[None if not np.isfinite(v) else float(v) for v in acc],
        classSupport=support.tolist(), predictedCounts=confusion.sum(axis=0).tolist(),
        confusion=confusion.tolist(), meanLogits=z.mean(axis=0).tolist(),
        meanProbabilities=probabilities.mean(axis=0).tolist(),
        outputBiasCEGradient=bias_gradient.tolist())


def split_parameters(current, reference, intervention):
    """Last dense kernel+bias is the head; copies only, never weight averaging."""
    import numpy as np
    if len(current) < 2 or len(current) != len(reference):
        raise ValueError('Matching parameter sequences including a head required.')
    if any(np.asarray(a).shape != np.asarray(b).shape for a, b in zip(current, reference)):
        raise ValueError('Parameter shape mismatch.')
    if np.asarray(current[-2]).ndim != 2 or np.asarray(current[-1]).ndim != 1:
        raise ValueError('Expected final dense kernel and bias.')
    if np.asarray(current[-2]).shape[-1] != np.asarray(current[-1]).size:
        raise ValueError('Head width mismatch.')
    if intervention == 'trained':
        values = current
    elif intervention == 'selected':
        values = reference
    elif intervention == 'trained_backbone_selected_head':
        values = list(current[:-2]) + list(reference[-2:])
    elif intervention == 'selected_backbone_trained_head':
        values = list(reference[:-2]) + list(current[-2:])
    elif intervention == 'trained_zero_output_bias':
        values = list(current[:-1]) + [np.zeros_like(current[-1])]
    else:
        raise ValueError(f'Unknown analysis-only intervention: {intervention}')
    return [np.asarray(v).copy() for v in values]


def parameter_drift(current, reference):
    import numpy as np
    split_parameters(current, reference, 'trained')  # validate the split
    rows = []
    for index, (a, b) in enumerate(zip(current, reference)):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Non-finite audited parameters.')
        rows.append(dict(index=index, shape=list(a.shape), role='head' if index >= len(current)-2 else 'backbone',
            driftL2=float(np.linalg.norm(a-b)), referenceL2=float(np.linalg.norm(b)),
            parameterL2=float(np.linalg.norm(a))))
    totals = {role: float(np.sqrt(sum(r['driftL2']**2 for r in rows if r['role'] == role)))
              for role in ('head', 'backbone')}
    return dict(tensors=rows, **totals,
        outputBias=[float(v) for v in current[-1]], selectedOutputBias=[float(v) for v in reference[-1]])


def representation_metrics(features, reference, labels):
    import numpy as np
    a, b, y = np.asarray(features, np.float64), np.asarray(reference, np.float64), np.asarray(labels)
    if a.shape != b.shape or a.ndim != 2 or len(a) != len(y) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Matching finite audit representations required.')
    centroids = np.stack([a[y == c].mean(axis=0) for c in range(10)])
    if not np.isfinite(centroids).all():
        raise ValueError('Balanced ten-class feature support required.')
    pairs = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominator > 0
    cosine = np.sum(a[valid]*b[valid], axis=1) / denominator[valid]
    return dict(meanSampleFeatureDrift=float(np.linalg.norm(a-b, axis=1).mean()),
        meanFeatureNorm=float(np.linalg.norm(a, axis=1).mean()),
        meanReferenceCosine=float(cosine.mean()) if valid.any() else None,
        zeroNormFeaturePairs=int((~valid).sum()), meanFeatureVariance=float(a.var(axis=0).mean()),
        meanClassCentroidSeparation=float(pairs[np.triu_indices(10, 1)].mean()),
        withinClassVariance=float(np.mean([a[y == c].var(axis=0).mean() for c in range(10)])))


INTERVENTIONS = ('trained', 'selected', 'trained_backbone_selected_head',
                 'selected_backbone_trained_head', 'trained_zero_output_bias')
