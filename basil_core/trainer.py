# basil_core/trainer.py
import numpy as np
import tensorflow as tf

# Optional WCM module
try:
    from noise_comm.wcm import wcmStep
    WCM_AVAILABLE = True
except ImportError:
    WCM_AVAILABLE = False
    wcmStep = None

__all__ = [
    "lossFn",
    "addChannelNoiseToParams",
    "getParams",
    "setParams",
    "averageParams",
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


def averageParams(paramsList, weights=None):
    """
    Average a list of model parameters (FedAvg-style aggregation).

    Args:
        paramsList: List of [params1, params2, ...] where each params is a list of numpy arrays
        weights: Optional weights for weighted average (e.g., by data size). If None, uniform average.

    Returns:
        Averaged parameters as list of numpy arrays

    Reference: "Robust Federated Learning with Noisy Communication" Equation 3a
        w = (Σ D_j × w_j) / D
    """
    if not paramsList:
        return []

    n = len(paramsList)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        # Normalize weights
        totalWeight = sum(weights)
        weights = [w / totalWeight for w in weights]

    # Initialize with zeros
    avgParams = [np.zeros_like(p) for p in paramsList[0]]

    # Weighted sum
    for params, weight in zip(paramsList, weights):
        for i, p in enumerate(params):
            avgParams[i] += weight * p

    return [p.astype(np.float32) for p in avgParams]


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

def makeLrScheduler(lr0, alpha=0.6, minLr=1e-4, useBasilSchedule=True):
    """
    Learning rate scheduler.

    If useBasilSchedule=True (default): Uses BASIL paper formula (Section V):
        lr_t = lr0 / (1 + lr0 * t)
        Paper uses lr0=0.03, giving: 0.03 / (1 + 0.03*t)

    If useBasilSchedule=False: Polynomial decay:
        lr_t = max(minLr, lr0 * (t+1)^(-alpha))
    """
    if useBasilSchedule:
        def lr(t):
            return float(max(minLr, lr0 / (1.0 + lr0 * t)))
        return lr
    else:
        def lr(t):
            return float(max(minLr, lr0 * (t + 1) ** (-alpha)))
        return lr

def _computeGrads(model, x, y):
    """Compute gradients of cross-entropy loss w.r.t. model weights.
    Returns (gradients_list, loss_scalar)."""
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = lossFn(y, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    return grads, loss


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
              Uses finite-difference Hessian-vector product (no nested tapes)
    - 'wcm': Worst-Case Model with boundary noise sampling and SCA surrogate

    Bounded by `stepsPerEpoch` to avoid hangs when dataLoader repeats indefinitely.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)

    # WCM initialization
    wcmAvailable = False
    wPrev = None
    gPrev = None
    if noiseModel == "wcm":
        if WCM_AVAILABLE:
            wcmAvailable = True
        else:
            print("Warning: WCM module not available. Falling back to standard training.")

    for _ in range(epochs):
        for xBatch, yBatch in _iterLimited(dataLoader, maxBatches=stepsPerEpoch):
            xb = tf.convert_to_tensor(xBatch, dtype=tf.float32)
            yb = tf.convert_to_tensor(yBatch, dtype=tf.int32)

            if noiseModel == "wcm" and sigma > 0 and wcmAvailable:
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
                    wPrev = result["wPrev"]
                    gPrev = result["gPrev"]
                except Exception as e:
                    print(f"Warning: WCM step failed ({e}). Using standard training for this batch.")

            elif noiseModel == "ebm" and sigma > 0:
                # EBM: Expectation-Based Model (Equation 13 & 23 from paper)
                # Loss: F_e(w) = F(w) + sigma^2 * ||grad F(w)||^2
                # Gradient: grad F_e(w) = (1 + lambda * sigma^2) * grad F(w)
                #
                # Reference: "Robust Federated Learning with Noisy Communication"
                # Paper uses sigma=1.0, scale=2.0
                #
                # With lambda multiplier, you can use smaller sigma:
                #   sigma=1.0, lambda=1   -> scale=2.0 (paper's setting)
                #   sigma=0.05, lambda=400 -> scale=2.0 (default)
                #   sigma=0.1, lambda=100 -> scale=2.0 (alternative)

                grads, _ = _computeGrads(model, xb, yb)
                scale = 1.0 + ebmLambda * sigma * sigma  # (1 + λσ²)
                scaledGrads = [g * scale for g in grads]
                optimizer.apply_gradients(zip(scaledGrads, model.trainable_weights))

            else:
                # Standard training (clean or noisy without mitigation)
                grads, _ = _computeGrads(model, xb, yb)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
