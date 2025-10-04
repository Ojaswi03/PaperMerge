# basil_core/trainer.py
import numpy as np
import tensorflow as tf

# Cross-entropy on logits (as in Basil)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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
    Local SGD update, optionally with EBM regularizer (loss + sigma^2 * ||∇loss||^2),
    implemented with a persistent GradientTape (no manual tape.watch needed).
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    for _ in range(epochs):
        for X_batch, y_batch in data_loader:
            Xb = tf.convert_to_tensor(X_batch, dtype=tf.float32)
            yb = tf.convert_to_tensor(y_batch, dtype=tf.int32)

            # Use a persistent tape so we can:
            #  1) get grads of base loss (for grad-norm regularizer)
            #  2) get grads of final loss (base + reg) for the update
            with tf.GradientTape(persistent=True) as tape:
                logits = model(Xb, training=True)
                base = loss_fn(yb, logits)

                if noise_model == "ebm" and sigma > 0.0:
                    # ||∇ base||^2 using the same tape
                    grads_base = tape.gradient(base, model.trainable_variables)
                    terms = []
                    for g in grads_base:
                        if g is not None:
                            terms.append(tf.reduce_sum(tf.square(g)))
                    grad_norm_sq = tf.add_n(terms) if terms else tf.constant(0.0, dtype=tf.float32)
                    loss = base + (sigma ** 2) * tf.cast(grad_norm_sq, tf.float32)
                else:
                    loss = base

            grads = tape.gradient(loss, model.trainable_variables)
            del tape  # free the persistent tape ASAP

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
