import numpy as np
import tensorflow as tf

def add_gaussian_noise_weights(model, sigma):
    """
    In-place: add N(0, sigma^2) to all trainable weights of a TF/Keras model.
    No-op if sigma <= 0.
    """
    if not sigma or sigma <= 0:
        return
    for w in model.trainable_variables:
        noise = tf.random.normal(shape=tf.shape(w), mean=0.0, stddev=sigma, dtype=w.dtype)
        w.assign_add(noise)

def add_gaussian_noise_numpy(params, sigma):
    """
    Return NEW list of numpy arrays with N(0, sigma^2) added.
    Compatible with Basil param payloads (list of layer arrays).
    """
    if not sigma or sigma <= 0:
        return [p.copy() for p in params]
    noisy = []
    for p in params:
        noise = np.random.normal(0.0, sigma, size=p.shape).astype(np.float32)
        noisy.append((p + noise).astype(np.float32))
    return noisy

def ebm_regularized_loss(loss_fn, model, x, y, sigma):
    """
    Compute: base_loss + sigma^2 * ||∇ base_loss||^2
    - loss_fn: callable(y_true, logits) -> scalar loss
    - model: tf.keras.Model
    - x, y: batches (np or tf tensors)
    Returns scalar tf.Tensor loss.
    """
    x_t = tf.convert_to_tensor(x, dtype=tf.float32)
    y_t = tf.convert_to_tensor(y, dtype=tf.int32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logits = model(x_t, training=True)
        base_loss = loss_fn(y_t, logits)

    if not sigma or sigma <= 0:
        return base_loss

    grads = tape.gradient(base_loss, model.trainable_variables)
    total = None
    for g in grads:
        if g is None:
            continue
        term = tf.reduce_sum(tf.square(g))
        total = term if total is None else (total + term)
    reg = tf.cast(0.0, tf.float32) if total is None else tf.cast(total, tf.float32)
    return base_loss + (sigma ** 2) * reg
