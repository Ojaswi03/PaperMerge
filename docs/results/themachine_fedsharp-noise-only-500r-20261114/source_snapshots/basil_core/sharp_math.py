"""SHARP equations 7--18, isolated from historical BASIL experiments.

The variance reduction and hop-curvature coefficient are modeled benchmarks,
not guarantees against adaptive selection, correlated histories or attackers.
"""
from __future__ import annotations

import math
from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')


def averaging_count(window_count, byzantine_budget):
    if type(window_count) is not int or window_count < 0 or type(byzantine_budget) is not int or byzantine_budget < 1:
        raise ValueError('A nonnegative window count and positive design budget are required.')
    return min(byzantine_budget, max(1, window_count - byzantine_budget))


def peak_screen(vectors, *, full_window, enabled=True):
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('A nonempty matrix of finite model copies is required.')
    d = values.shape[1]
    threshold = 2.5 * math.sqrt(2 * math.log(max(d, 2)))
    ratios = np.zeros(len(values), np.float64)
    if enabled and full_window:
        median = np.median(values, axis=0)
        differences = values.astype(np.float64) - median
        rms = np.sqrt(np.mean(differences ** 2, axis=1))
        ratios = np.divide(np.max(np.abs(differences), axis=1), rms,
                           out=np.zeros(len(values), np.float64), where=rms > 0)
    passed = np.flatnonzero(ratios <= threshold)
    fallback = not len(passed)
    if fallback:
        passed = np.arange(len(values))  # Algorithm 1's explicitly reported all-fail policy.
    return passed, dict(peakRatios=ratios.tolist(), peakThreshold=threshold,
        screenApplied=bool(enabled and full_window), screenAllFailFallback=fallback)


def ranked_indices(losses, senders, snapshot_rounds, receiver, node_count, count):
    """Raw CE; newer round first, then nearest predecessor, for exact ties."""
    if not (len(losses) == len(senders) == len(snapshot_rounds)) or count < 1:
        raise ValueError('Aligned ranking inputs and a positive count required.')
    order = sorted(range(len(losses)), key=lambda j: (
        float(losses[j]) if math.isfinite(float(losses[j])) else math.inf,
        -int(snapshot_rounds[j]), (receiver - int(senders[j])) % node_count))
    finite = [j for j in order if math.isfinite(float(losses[j]))]
    if not finite:
        raise FloatingPointError('All candidate receiver losses are non-finite; no update started.')
    return np.asarray(finite[:count], np.int64)


def hop_coefficient(sigma, averaged_count, learning_rate, curvature=1.):
    if (not all(math.isfinite(float(x)) for x in (sigma, learning_rate, curvature))
            or sigma < 0 or learning_rate <= 0 or curvature <= 0
            or type(averaged_count) is not int or averaged_count < 1):
        raise ValueError('Finite nonnegative sigma and positive q, eta and h required.')
    return float(sigma) ** 2 / (2 * averaged_count * learning_rate * curvature)


def class_ce(logits, labels):
    z = np.asarray(logits, np.float64)
    y = np.asarray(labels)
    if (z.ndim != 2 or z.shape[1] < 2 or y.shape != (len(z),) or not len(y)
            or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= z.shape[1])
            or not np.isfinite(z).all()):
        raise ValueError('Finite class logits and matching integer labels required.')
    shifted = z - z.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]
    classes = np.unique(y)
    return classes, np.asarray([losses[y == c].mean() for c in classes])


def ridge_transport(features, proposal_logits, reference_logits, labels, gates, fraction=.01):
    h = np.asarray(features, np.float64)
    p, r = np.asarray(proposal_logits, np.float64), np.asarray(reference_logits, np.float64)
    y, g = np.asarray(labels), np.asarray(gates, np.float64)
    class_ce(p, y)
    if (h.ndim != 2 or len(h) != len(y) or p.shape != r.shape or g.shape != (p.shape[1],)
            or not all(np.isfinite(v).all() for v in (h, r, g)) or np.any(g < 0) or np.any(g > 1)
            or not math.isfinite(fraction) or fraction <= 0):
        raise ValueError('Aligned finite transport arrays and bounded gates required.')
    a = np.column_stack((h, np.ones(len(h))))
    counts = np.bincount(y, minlength=p.shape[1])
    weights = 1. / (np.count_nonzero(counts) * counts[y])
    residual = (r - r.mean(axis=1, keepdims=True)) - (p - p.mean(axis=1, keepdims=True))
    target = g[y, None] * residual
    gram = a.T @ (weights[:, None] * a)
    penalty = float(fraction * np.trace(gram) / len(gram))
    correction = np.linalg.solve(gram + penalty * np.eye(len(gram)), a.T @ (weights[:, None] * target))
    initial = float(np.sum(weights[:, None] * target ** 2))
    fitted = float(np.sum(weights[:, None] * (a @ correction - target) ** 2) + penalty * np.sum(correction ** 2))
    if not np.isfinite(correction).all() or fitted > initial + 1e-8 * max(1., initial):
        raise FloatingPointError('Invalid ridge solution.')
    return correction, dict(ridgePenalty=penalty, zeroObjective=initial, fittedObjective=fitted)
