import numpy as np
import tensorflow as tf

def add_gaussian_noise_weights(model, sigma):
    # no-op when sigma is zero or negative
    if not sigma or sigma <= 0:
        return
    for w in model.trainable_variables:
        noise = tf.random.normal(shape=tf.shape(w), mean=0.0, stddev=sigma, dtype=w.dtype)
        w.assign_add(noise)

def add_gaussian_noise_numpy(params, sigma):
    # return copies without noise when sigma is not positive
    if not sigma or sigma <= 0:
        return [p.copy() for p in params]
    noisy = []
    for p in params:
        noise = np.random.normal(0.0, sigma, size=p.shape).astype(np.float32)
        noisy.append((p + noise).astype(np.float32))
    return noisy

def ebm_regularized_loss(loss_fn, model, x, y, sigma):
    # compute base loss and add sigma^2 * grad_norm_sq regularizer
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
