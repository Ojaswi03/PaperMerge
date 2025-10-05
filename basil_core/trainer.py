# basil_core/trainer.py
import numpy as np
import tensorflow as tf

__all__ = [
    "loss_fn",
    "add_channel_noise_to_params",
    "get_params",
    "set_params",
    "evaluate",
    "evaluate_batch_loss",
    "evaluate_all",
    "make_lr_scheduler",
    "local_update",
]

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

def get_params(model):
    """Return a list of numpy arrays for model.trainable_weights (float32)."""
    return [w.numpy().astype(np.float32) for w in model.trainable_weights]

def set_params(model, params):
    """Assign a list of numpy arrays to model.trainable_weights."""
    for var, new in zip(model.trainable_weights, params):
        var.assign(tf.convert_to_tensor(new, dtype=var.dtype))

def _iter_limited(ds, max_batches=None):
    """Yield at most max_batches batches from ds. If max_batches is None, iterate fully."""
    if max_batches is None:
        for batch in ds:
            yield batch
    else:
        for i, batch in enumerate(ds):
            if i >= max_batches:
                break
            yield batch

def evaluate(model, data_loader, max_batches=100):
    """
    Accuracy in [0,1] over at most `max_batches` batches.
    Bounds eval when data_loader is an infinite .repeat().
    """
    total = 0
    correct = 0
    for X_batch, y_batch in _iter_limited(data_loader, max_batches=max_batches):
        Xb = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        logits = model(Xb, training=False)
        preds = tf.argmax(logits, axis=1).numpy()
        # y_batch may be tensor or numpy
        y_np = y_batch.numpy() if hasattr(y_batch, "numpy") else y_batch
        correct += (preds == y_np).sum()
        total += len(y_np)
    return float(correct) / float(total) if total else 0.0

def evaluate_batch_loss(model, data_loader):
    """Loss on the first batch (used for Basil snapshot selection)."""
    for X_batch, y_batch in _iter_limited(data_loader, max_batches=1):
        Xb = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        yb = tf.convert_to_tensor(y_batch, dtype=tf.int32)
        logits = model(Xb, training=False)
        return float(loss_fn(yb, logits).numpy())
    return 0.0

def evaluate_all(nodes, test_loader):
    """Return (avg_acc, worst_acc, list_per_node)."""
    accs = [evaluate(node.model, test_loader) for node in nodes]
    if not accs:
        return 0.0, 0.0, []
    avg = float(np.mean(accs))
    worst = float(np.min(accs))
    return avg, worst, accs

def make_lr_scheduler(lr0, alpha=0.6, min_lr=1e-4):
    """Polynomial decay: lr_t = max(min_lr, lr0 * (t+1)^(-alpha))."""
    def lr(t):
        return float(max(min_lr, lr0 * (t + 1) ** (-alpha)))
    return lr

def local_update(
    model,
    data_loader,
    epochs,
    lr,
    noise_model="none",
    sigma=0.0,
    ebm_lambda=1.0,
    steps_per_epoch=100,   # <-- NEW: bound for infinite datasets
):
    """
    Local SGD step. If noise_model=='ebm', add sigma^2 * ||grad||^2 regularizer (EBM).
    Bounded by `steps_per_epoch` to avoid hangs when data_loader repeats indefinitely.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)
    for _ in range(epochs):
        for X_batch, y_batch in _iter_limited(data_loader, max_batches=steps_per_epoch):
            Xb = tf.convert_to_tensor(X_batch, dtype=tf.float32)
            yb = tf.convert_to_tensor(y_batch, dtype=tf.int32)
            with tf.GradientTape(persistent=True) as tape:
                logits = model(Xb, training=True)
                base_loss = loss_fn(yb, logits)
                if noise_model == "ebm" and sigma > 0:
                    grads_base = tape.gradient(base_loss, model.trainable_weights)
                    reg = 0.0
                    for g in grads_base:
                        reg += tf.reduce_sum(tf.square(g))
                    final_loss = base_loss + ebm_lambda * (sigma ** 2) * reg
                else:
                    final_loss = base_loss
            grads = tape.gradient(final_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
