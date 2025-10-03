import numpy as np
import tensorflow as tf

# Cross-entropy on logits (as in Basil)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ===== EBM helpers (Expectation-Based Model) =====
def _grad_norm_sq(loss_tensor, variables):
    # ||∇ loss||^2 over trainable variables
    with tf.GradientTape() as tape2:
        # re-create a zero op to keep shape; we just need grads of loss wrt vars
        pass
    # We can compute gradients outside the dummy tape using tf.gradients-like API:
    grads = tf.gradients(ys=loss_tensor, xs=variables)
    total = None
    for g in grads:
        if g is None:
            continue
        term = tf.reduce_sum(tf.square(g))
        total = term if total is None else (total + term)
    if total is None:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.cast(total, tf.float32)

def add_channel_noise_to_params(params, sigma):
    """Sender-side noisy channel: add N(0, sigma^2) to each layer (numpy arrays)."""
    if not sigma or sigma <= 0:
        return params
    noisy = []
    for w in params:
        noise = np.random.normal(0.0, sigma, size=w.shape).astype(np.float32)
        noisy.append((w + noise).astype(np.float32))
    return noisy

# ===== Local training / eval =====
def local_update(model, data_loader, epochs, lr, noise_model="none", sigma=0.0):
    """
    Local SGD update, optionally with EBM regularizer (loss + sigma^2 * ||∇loss||^2).
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    for _ in range(epochs):
        for X_batch, y_batch in data_loader:
            Xb = tf.convert_to_tensor(X_batch, dtype=tf.float32)
            yb = tf.convert_to_tensor(y_batch, dtype=tf.int32)

            with tf.GradientTape() as tape:
                logits = model(Xb, training=True)
                base = loss_fn(yb, logits)
                loss = base
                if noise_model == "ebm" and sigma > 0.0:
                    # σ²‖∇F‖²
                    gns = _grad_norm_sq(base, model.trainable_variables)
                    loss = base + (sigma ** 2) * gns

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

def evaluate(model, data_loader):
    """Accuracy on loader."""
    correct = 0
    total = 0
    for X_batch, y_batch in data_loader:
        Xb = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        yb = tf.convert_to_tensor(y_batch, dtype=tf.int32)
        logits = model(Xb, training=False)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, yb), tf.int32)).numpy())
        total += yb.shape[0]
    return (correct / total) if total > 0 else 0.0

def evaluate_batch_loss(model, data_loader):
    """
    Evaluate CE loss on the FIRST batch of a loader (used for Basil snapshot selection).
    """
    for X_batch, y_batch in data_loader:
        Xb = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        yb = tf.convert_to_tensor(y_batch, dtype=tf.int32)
        logits = model(Xb, training=False)
        return float(loss_fn(yb, logits).numpy())
    return 0.0

def evaluate_all(nodes, test_loader):
    """Return avg_acc, worst_acc, per_node_acc list."""
    accs = [evaluate(node.model, test_loader) for node in nodes]
    if not accs:
        return 0.0, 0.0, []
    avg = float(np.mean(accs))
    worst = float(np.min(accs))
    return avg, worst, accs
