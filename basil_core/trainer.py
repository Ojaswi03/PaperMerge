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
    "evaluatePerClass",
    "makeLrScheduler",
    "localUpdate",
    "_iterLimited",
]

# Cross-entropy on logits (as in Basil)
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def addChannelNoiseToParams(params, sigma, rng=None):
    """Add Gaussian channel noise using sigma as a model-level relative L2 budget.

    This project interprets sigma as a communication perturbation budget for
    the transmitted model, not an absolute stddev applied independently to every
    parameter coordinate. For CIFAR-scale CNNs, absolute or per-tensor stddev
    noise destroys the model because millions of coordinates are perturbed.

    We sample isotropic Gaussian noise and scale it so the expected full-model
    L2 norm is sigma * ||w||_2:

        noise_coord_std = sigma * ||w||_2 / sqrt(num_params)

    This keeps sigma comparable across MNIST/CIFAR and lets EBM target channel
    noise instead of recovering from a completely randomized model.
    """
    if not sigma or sigma <= 0:
        return params
    total_sq = 0.0
    total_dim = 0
    for w in params:
        wf = w.astype(np.float32, copy=False)
        total_sq += float(np.sum(wf * wf))
        total_dim += int(wf.size)
    if total_dim <= 0:
        return params

    model_norm = float(np.sqrt(max(total_sq, 1e-12)))
    noise_std = float(sigma) * model_norm / float(np.sqrt(total_dim))

    normal = rng.normal if rng is not None else np.random.normal
    noisy = []
    for w in params:
        noise = normal(0.0, noise_std, size=w.shape).astype(np.float32)
        noisy.append((w + noise).astype(np.float32))
    return noisy

def getParams(model):
    # extract trainable weights as float32 numpy arrays
    return [w.numpy().astype(np.float32) for w in model.trainable_weights]

def setParams(model, params):
    # assign numpy arrays back into model trainable weights
    for var, new in zip(model.trainable_weights, params):
        var.assign(tf.convert_to_tensor(new, dtype=var.dtype))


def averageParams(paramsList, weights=None):
    # return empty list if nothing to average
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
    # yield all batches when no limit is set
    if maxBatches is None:
        for batch in ds:
            yield batch
    else:
        for i, batch in enumerate(ds):
            if i >= maxBatches:
                break
            yield batch

def evaluate(model, dataLoader, maxBatches=None):
    # count correct predictions; maxBatches=None uses full test set
    total = 0
    correct = 0
    for xBatch, yBatch in _iterLimited(dataLoader, maxBatches=maxBatches):
        xb = tf.cast(xBatch, tf.float32)
        logits = _forwardPass(model, xb)
        preds = tf.argmax(logits, axis=1).numpy()
        yNp = yBatch.numpy() if hasattr(yBatch, "numpy") else yBatch
        correct += (preds == yNp).sum()
        total += len(yNp)
    return float(correct) / float(total) if total else 0.0

def evaluateBatchLoss(model, dataLoader):
    # compute loss on a single batch for BASIL snapshot selection
    for xBatch, yBatch in _iterLimited(dataLoader, maxBatches=1):
        xb = tf.cast(xBatch, tf.float32)
        yb = tf.cast(yBatch, tf.int32)
        logits = _forwardPass(model, xb)
        return float(lossFn(yb, logits).numpy())
    return 0.0

def evaluatePerClass(model, dataLoader, nClasses=10, maxBatches=None):
    """Return per-class accuracy as np.float32 array of shape (nClasses,).

    Uses at most maxBatches batches (None = full loader). Suitable for CART
    class registry updates and class-aware snapshot selection.
    """
    correct = np.zeros(nClasses, dtype=np.int64)
    total = np.zeros(nClasses, dtype=np.int64)
    for xBatch, yBatch in _iterLimited(dataLoader, maxBatches=maxBatches):
        xb = tf.cast(xBatch, tf.float32)
        logits = _forwardPass(model, xb)
        preds = tf.argmax(logits, axis=1).numpy()
        yNp = yBatch.numpy() if hasattr(yBatch, "numpy") else np.array(yBatch)
        for c in range(nClasses):
            mask = yNp == c
            total[c] += mask.sum()
            correct[c] += (preds[mask] == c).sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(total > 0, correct / total, 0.0)
    return acc.astype(np.float32)

def evaluateAll(nodes, testLoader, maxBatches=None):
    # evaluate every node and aggregate into avg/worst
    accs = [evaluate(node.model, testLoader, maxBatches=maxBatches) for node in nodes]
    if not accs:
        return 0.0, 0.0, []
    avg = float(np.mean(accs))
    worst = float(np.min(accs))
    return avg, worst, accs

def makeLrScheduler(lr0, alpha=0.6, minLr=1e-4, useBasilSchedule=True, useLrDecay=True):
    # return a callable lr(t) based on selected schedule
    if not useLrDecay:
        # Fixed learning rate (no decay)
        def lr(t):
            return float(lr0)
        return lr
    elif useBasilSchedule:
        def lr(t):
            return float(max(minLr, lr0 / (1.0 + lr0 * t)))
        return lr
    else:
        def lr(t):
            return float(max(minLr, lr0 * (t + 1) ** (-alpha)))
        return lr

@tf.function(reduce_retracing=True)
def _computeGrads(model, x, y):
    # compute cross-entropy gradients and loss in one tape pass
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = lossFn(y, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    return grads, loss


@tf.function(reduce_retracing=True)
def _forwardPass(model, x):
    return model(x, training=False)


def localUpdate(
    model,
    dataLoader,
    epochs,
    lr,
    noiseModel="none",
    sigma=0.0,
    ebmLambda=25.0,
    wcmLambda=0.1,
    wcmSamples=5,
    wcmRho=0.5,
    stepsPerEpoch=100,
    momentum=0.0,
):
    # create optimizer once per local update call
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

    # WCM initialization
    wcmAvailable = False
    wPrev = None
    gPrev = None
    if noiseModel == "wcm":
        if WCM_AVAILABLE:
            wcmAvailable = True
        else:
            print("Warning: WCM module not available. Falling back to standard training.")

    totalSteps = epochs * stepsPerEpoch
    for xBatch, yBatch in _iterLimited(dataLoader, maxBatches=totalSteps):
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
            # Lambda values that give scale=2.0 for different sigma:
            #   sigma=1.0,  lambda=1   -> scale=2.0 (paper's original setting)
            #   sigma=0.2,  lambda=25  -> scale=2.0 (default for configs)
            #   sigma=0.1,  lambda=100 -> scale=2.0
            #   sigma=0.05, lambda=400 -> scale=2.0

            grads, _ = _computeGrads(model, xb, yb)
            scale = 1.0 + ebmLambda * sigma * sigma  # (1 + λσ²)
            clipped, _ = tf.clip_by_global_norm(grads, 5.0)
            scaledGrads = [g * scale for g in clipped]
            optimizer.apply_gradients(zip(scaledGrads, model.trainable_weights))

        else:
            # Standard training (clean or noisy without mitigation)
            grads, _ = _computeGrads(model, xb, yb)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
