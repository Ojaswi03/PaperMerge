"""
Worst-Case Model (WCM) scaffold for noisy communication.
Implements:
  - boundary noise sampling: ||Δw|| = sigma  (L2 sphere)
  - SAA: average over S samples
  - SCA-like surrogate: convex combo of current loss and linearization around previous iterate

This file is a LIGHT scaffold to help you extend Section V of the paper.
You can integrate these pieces into your training loop similarly to EBM.
"""

import numpy as np
import tensorflow as tf

# ----- noise sampling on the L2 boundary -----
def _sampleBoundaryNoiseLike(weight, sigma):
    # draw random direction then scale to L2 norm = sigma
    g = tf.random.normal(shape=tf.shape(weight), dtype=weight.dtype)
    gNorm = tf.norm(g)
    # handle rare zero norm by fallback to unit vector-like
    scale = tf.where(gNorm > 0, sigma / gNorm, tf.constant(0.0, dtype=weight.dtype))
    return g * scale

def sampleBoundaryPayload(params, sigma):
    # return zero deltas when no noise requested
    if not sigma or sigma <= 0:
        return [np.zeros_like(p) for p in params]
    deltas = []
    for p in params:
        w = tf.convert_to_tensor(p, dtype=tf.float32)
        d = _sampleBoundaryNoiseLike(w, tf.constant(float(sigma), tf.float32))
        deltas.append(d.numpy().astype(np.float32))
    return deltas

def applyDelta(params, deltas, alpha=1.0):
    # add scaled deltas to each layer of params
    out = []
    for p, d in zip(params, deltas):
        out.append((p + alpha * d).astype(np.float32))
    return out

# ----- SAA + SCA surrogate -----
def scaSurrogateLoss(lossFn, model, x, y, deltaList, rho, lam):
    # prepare tensors and snapshot current weights before applying delta
    xT = tf.convert_to_tensor(x, dtype=tf.float32)
    yT = tf.convert_to_tensor(y, dtype=tf.int32)

    # snapshot original weights
    w0 = [w.numpy().copy() for w in model.trainable_variables]

    # apply offsets (w + Δ)
    for var, delta in zip(model.trainable_variables, deltaList):
        var.assign_add(tf.convert_to_tensor(delta, dtype=var.dtype))

    with tf.GradientTape() as tape:
        logits = model(xT, training=True)
        base = lossFn(yT, logits)

    # remove offsets (restore)
    for var, old in zip(model.trainable_variables, w0):
        var.assign(old)

    # surrogate: rho * F(w + Δ) + lam * ||w - w_prev||^2
    # caller should add (1 - rho) * <w - w_prev, G_prev> externally if they maintain G_prev and w_prev
    lossTerm = rho * base
    # regularizer term must be added by caller when w_prev is known:
    # e.g., loss_total = loss_term + lam * ||w - w_prev||^2 + (1 - rho) * <w - w_prev, G_prev>

    return lossTerm, w0

def wcmStep(model, optimizer, lossFn, x, y, sigma, S, rho, lam,
            wPrev=None, gPrev=None, betaForG=0.9):
    # prepare delta samples
    currentParams = [w.numpy().copy() for w in model.trainable_variables]
    saLosses = []
    with tf.GradientTape() as tapeOuter:
        totalSurrogate = 0.0
        for _ in range(int(S)):
            deltas = sampleBoundaryPayload(currentParams, sigma)
            lossDelta, _ = scaSurrogateLoss(lossFn, model, x, y, deltas, rho, lam)
            totalSurrogate = totalSurrogate + lossDelta
        totalSurrogate = totalSurrogate / float(max(1, int(S)))

        # add quadratic proximity if wPrev is given
        if wPrev is not None and lam > 0.0:
            prox = 0.0
            for var, prev in zip(model.trainable_variables, wPrev):
                diff = var - tf.convert_to_tensor(prev, dtype=var.dtype)
                prox = prox + tf.reduce_sum(tf.square(diff))
            totalSurrogate = totalSurrogate + lam * prox

        # (optional) linearization term (1 - rho) * <w - wPrev, gPrev>
        if wPrev is not None and gPrev is not None and rho < 1.0:
            lin = 0.0
            for var, prev, g in zip(model.trainable_variables, wPrev, gPrev):
                diff = var - tf.convert_to_tensor(prev, dtype=var.dtype)
                lin = lin + tf.reduce_sum(diff * tf.convert_to_tensor(g, dtype=var.dtype))
            totalSurrogate = totalSurrogate + (1.0 - rho) * lin

    grads = tapeOuter.gradient(totalSurrogate, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # update moving-average gradient G_t
    newG = None
    if gPrev is not None:
        # recompute gradient at current point for update of G
        with tf.GradientTape() as tapeG:
            logits = model(tf.convert_to_tensor(x, dtype=tf.float32), training=True)
            base = lossFn(tf.convert_to_tensor(y, dtype=tf.int32), logits)
        gNow = tapeG.gradient(base, model.trainable_variables)
        newG = []
        for gPrevItem, gCur in zip(gPrev, gNow):
            gPrevT = tf.convert_to_tensor(gPrevItem, dtype=gCur.dtype)
            gCurT  = tf.convert_to_tensor(0.0, dtype=gCur.dtype) if gCur is None else gCur
            newG.append(betaForG * gPrevT + (1.0 - betaForG) * gCurT)

    result = {
        "loss": float(totalSurrogate.numpy()),
        "wPrev": [w.numpy().copy() for w in model.trainable_variables],
        "gPrev": newG if newG is not None else gPrev
    }
    return result
