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
def _sample_boundary_noise_like(weight, sigma):
    """
    Sample Δ of same shape as weight such that ||Δ||_2 = sigma (approximately).
    Uses normalized Gaussian direction scaled to sigma.
    """
    g = tf.random.normal(shape=tf.shape(weight), dtype=weight.dtype)
    g_norm = tf.norm(g)
    # handle rare zero norm by fallback to unit vector-like
    scale = tf.where(g_norm > 0, sigma / g_norm, tf.constant(0.0, dtype=weight.dtype))
    return g * scale

def sample_boundary_payload(params, sigma):
    """
    Return a NEW list of numpy arrays Δ with ||Δ|| layer-wise scaled so that
    the concatenated vector approximates global norm sigma. For simplicity,
    we enforce per-layer boundary (common approximation).
    """
    if not sigma or sigma <= 0:
        return [np.zeros_like(p) for p in params]
    deltas = []
    for p in params:
        w = tf.convert_to_tensor(p, dtype=tf.float32)
        d = _sample_boundary_noise_like(w, tf.constant(float(sigma), tf.float32))
        deltas.append(d.numpy().astype(np.float32))
    return deltas

def apply_delta(params, deltas, alpha=1.0):
    """
    Return params + alpha * deltas (both lists of numpy arrays).
    """
    out = []
    for p, d in zip(params, deltas):
        out.append((p + alpha * d).astype(np.float32))
    return out

# ----- SAA + SCA surrogate -----
def sca_surrogate_loss(loss_fn, model, x, y, delta_list, rho, lam):
    """
    One SCA-like surrogate:
      F_w(w) = rho * F(w + Δ) + (1 - rho) * < w - w_prev, G_prev > + lam * ||w - w_prev||^2
    We implement a practical analog where:
      - We use the first-order term via a moving average of gradients (Gt) if provided by the caller.
      - Caller is responsible for maintaining (w_prev, G_prev) across steps; this function returns
        base terms to update those structures.

    For simplicity here, we compute only rho * F(w + Δ) + lam * ||w - w_prev||^2,
    and expose hooks for caller to add the linearization term when available.

    Args:
      loss_fn: callable(y_true, logits) -> scalar
      model: tf.keras.Model
      x, y: batch tensors/arrays
      delta_list: list of numpy arrays (same shapes as model weights) to offset weights
      rho: in (0,1]
      lam: >= 0

    Returns: (loss_scalar_tensor, w_current_list) where w_current_list is current weights (numpy).
    """
    x_t = tf.convert_to_tensor(x, dtype=tf.float32)
    y_t = tf.convert_to_tensor(y, dtype=tf.int32)

    # snapshot original weights
    w0 = [w.numpy().copy() for w in model.trainable_variables]

    # apply offsets (w + Δ)
    for var, delta in zip(model.trainable_variables, delta_list):
        var.assign_add(tf.convert_to_tensor(delta, dtype=var.dtype))

    with tf.GradientTape() as tape:
        logits = model(x_t, training=True)
        base = loss_fn(y_t, logits)

    # remove offsets (restore)
    for var, old in zip(model.trainable_variables, w0):
        var.assign(old)

    # surrogate: rho * F(w + Δ) + lam * ||w - w_prev||^2
    # caller should add (1 - rho) * <w - w_prev, G_prev> externally if they maintain G_prev and w_prev
    loss_term = rho * base
    # regularizer term must be added by caller when w_prev is known:
    # e.g., loss_total = loss_term + lam * ||w - w_prev||^2 + (1 - rho) * <w - w_prev, G_prev>

    return loss_term, w0

def wcm_step(model, optimizer, loss_fn, x, y, sigma, S, rho, lam,
             w_prev=None, G_prev=None, beta_for_G=0.9):
    """
    A single WCM-flavored local step:
      - Sample S boundary deltas, average their surrogate losses (SAA)
      - Add stabilization term lam * ||w - w_prev||^2 when w_prev is provided
      - Maintain moving-average gradient G_t if G_prev provided
      - Perform one gradient step on the surrogate

    Returns: dict with updated (optionally) w_prev, G_prev and scalar loss.
    """
    # prepare delta samples
    current_params = [w.numpy().copy() for w in model.trainable_variables]
    sa_losses = []
    with tf.GradientTape() as tape_outer:
        total_surrogate = 0.0
        for _ in range(int(S)):
            deltas = sample_boundary_payload(current_params, sigma)
            loss_delta, _ = sca_surrogate_loss(loss_fn, model, x, y, deltas, rho, lam)
            total_surrogate = total_surrogate + loss_delta
        total_surrogate = total_surrogate / float(max(1, int(S)))

        # add quadratic proximity if w_prev is given
        if w_prev is not None and lam > 0.0:
            prox = 0.0
            for var, prev in zip(model.trainable_variables, w_prev):
                diff = var - tf.convert_to_tensor(prev, dtype=var.dtype)
                prox = prox + tf.reduce_sum(tf.square(diff))
            total_surrogate = total_surrogate + lam * prox

        # (optional) linearization term (1 - rho) * <w - w_prev, G_prev>
        if w_prev is not None and G_prev is not None and rho < 1.0:
            lin = 0.0
            for var, prev, g in zip(model.trainable_variables, w_prev, G_prev):
                diff = var - tf.convert_to_tensor(prev, dtype=var.dtype)
                lin = lin + tf.reduce_sum(diff * tf.convert_to_tensor(g, dtype=var.dtype))
            total_surrogate = total_surrogate + (1.0 - rho) * lin

    grads = tape_outer.gradient(total_surrogate, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # update moving-average gradient G_t
    new_G = None
    if G_prev is not None:
        # recompute gradient at current point for update of G
        with tf.GradientTape() as tape_G:
            logits = model(tf.convert_to_tensor(x, dtype=tf.float32), training=True)
            base = loss_fn(tf.convert_to_tensor(y, dtype=tf.int32), logits)
        g_now = tape_G.gradient(base, model.trainable_variables)
        new_G = []
        for g_prev, g_cur in zip(G_prev, g_now):
            g_prev_t = tf.convert_to_tensor(g_prev, dtype=g_cur.dtype)
            g_cur_t  = tf.convert_to_tensor(0.0, dtype=g_cur.dtype) if g_cur is None else g_cur
            new_G.append(beta_for_G * g_prev_t + (1.0 - beta_for_G) * g_cur_t)

    result = {
        "loss": float(total_surrogate.numpy()),
        "w_prev": [w.numpy().copy() for w in model.trainable_variables],
        "G_prev": new_G if new_G is not None else G_prev
    }
    return result
