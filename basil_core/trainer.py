# basil_core/trainer.py
import numpy as np
import tensorflow as tf

__all__ = [
    "lossFn",
    "addChannelNoiseToParams",
    "getParams",
    "setParams",
    "evaluate",
    "evaluateBatchLoss",
    "evaluateAll",
    "makeLrScheduler",
    "localUpdate",
]

# Cross-entropy on logits (as in Basil)
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def addChannelNoiseToParams(params, sigma):
    """Sender-side noisy channel: add N(0, sigma^2) to each layer (numpy arrays)."""
    if not sigma or sigma <= 0:
        return params
    noisy = []
    for w in params:
        noise = np.random.normal(0.0, sigma, size=w.shape).astype(np.float32)
        noisy.append((w + noise).astype(np.float32))
    return noisy

def getParams(model):
    """Return a list of numpy arrays for model.trainable_weights (float32)."""
    return [w.numpy().astype(np.float32) for w in model.trainable_weights]

def setParams(model, params):
    """Assign a list of numpy arrays to model.trainable_weights."""
    for var, new in zip(model.trainable_weights, params):
        var.assign(tf.convert_to_tensor(new, dtype=var.dtype))

def _iterLimited(ds, maxBatches=None):
    """Yield at most maxBatches batches from ds. If maxBatches is None, iterate fully."""
    if maxBatches is None:
        for batch in ds:
            yield batch
    else:
        for i, batch in enumerate(ds):
            if i >= maxBatches:
                break
            yield batch

def evaluate(model, dataLoader, maxBatches=100):
    """
    Accuracy in [0,1] over at most `maxBatches` batches.
    Bounds eval when dataLoader is an infinite .repeat().
    """
    total = 0
    correct = 0
    for xBatch, yBatch in _iterLimited(dataLoader, maxBatches=maxBatches):
        xb = tf.convert_to_tensor(xBatch, dtype=tf.float32)
        logits = model(xb, training=False)
        preds = tf.argmax(logits, axis=1).numpy()
        # yBatch may be tensor or numpy
        yNp = yBatch.numpy() if hasattr(yBatch, "numpy") else yBatch
        correct += (preds == yNp).sum()
        total += len(yNp)
    return float(correct) / float(total) if total else 0.0

def evaluateBatchLoss(model, dataLoader):
    """Loss on the first batch (used for Basil snapshot selection)."""
    for xBatch, yBatch in _iterLimited(dataLoader, maxBatches=1):
        xb = tf.convert_to_tensor(xBatch, dtype=tf.float32)
        yb = tf.convert_to_tensor(yBatch, dtype=tf.int32)
        logits = model(xb, training=False)
        return float(lossFn(yb, logits).numpy())
    return 0.0

def evaluateAll(nodes, testLoader):
    """Return (avgAcc, worstAcc, listPerNode)."""
    accs = [evaluate(node.model, testLoader) for node in nodes]
    if not accs:
        return 0.0, 0.0, []
    avg = float(np.mean(accs))
    worst = float(np.min(accs))
    return avg, worst, accs

def makeLrScheduler(lr0, alpha=0.6, minLr=1e-4):
    """Polynomial decay: lr_t = max(minLr, lr0 * (t+1)^(-alpha))."""
    def lr(t):
        return float(max(minLr, lr0 * (t + 1) ** (-alpha)))
    return lr

def localUpdate(
    model,
    dataLoader,
    epochs,
    lr,
    noiseModel="none",
    sigma=0.0,
    ebmLambda=1.0,
    wcmLambda=0.1,
    wcmSamples=5,
    wcmRho=0.5,
    stepsPerEpoch=100,
):
    """
    Local SGD step with support for different noise models:
    - 'none' or 'clean': standard SGD
    - 'noisy': standard SGD (noise added at communication, not here)
    - 'ebm': add sigma^2 * ||grad||^2 regularizer (Equation 13 from Noisy Channel paper)
    - 'wcm': Worst-Case Model with boundary noise sampling and SCA surrogate

    Bounded by `stepsPerEpoch` to avoid hangs when dataLoader repeats indefinitely.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)

    # WCM initialization
    wcmAvailable = False
    if noiseModel == "wcm":
        try:
            from noise_comm.wcm import wcmStep
            wPrev = None
            gPrev = None
            wcmAvailable = True
        except ImportError as e:
            print(f"Warning: WCM module not available ({e}). Falling back to standard training.")
            wcmAvailable = False

    for _ in range(epochs):
        for xBatch, yBatch in _iterLimited(dataLoader, maxBatches=stepsPerEpoch):
            xb = tf.convert_to_tensor(xBatch, dtype=tf.float32)
            yb = tf.convert_to_tensor(yBatch, dtype=tf.int32)

            if noiseModel == "wcm" and sigma > 0 and wcmAvailable:
                # Use WCM training step
                try:
                    result = wcmStep(
                        model=model,
                        optimizer=optimizer,
                        lossFn=lossFn,
                        x=xb,
                        y=yb,
                        sigma=sigma,
                        S=wcmSamples,
                        rho=wcmRho,
                        lam=wcmLambda,
                        wPrev=wPrev,
                        gPrev=gPrev,
                        betaForG=0.9
                    )
                    # Update state for next iteration
                    wPrev = result["wPrev"]
                    gPrev = result["gPrev"]
                except Exception as e:
                    print(f"Warning: WCM step failed ({e}). Using standard training for this batch.")
                    # Fall through to standard training below
                    pass

            if noiseModel != "wcm" or sigma <= 0 or not wcmAvailable:
                # Standard training or EBM
                with tf.GradientTape(persistent=True) as tape:  # Might be an issue
                    logits = model(xb, training=True)
                    baseLoss = lossFn(yb, logits)
                    if noiseModel == "ebm" and sigma > 0:
                        gradsBase = tape.gradient(baseLoss, model.trainable_weights)
                        reg = 0.0
                        for g in gradsBase:
                            reg += tf.reduce_sum(tf.square(g))
                        finalLoss = baseLoss + ebmLambda * (sigma ** 2) * reg  # Equation 13
                    else:
                        finalLoss = baseLoss
                grads = tape.gradient(finalLoss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
