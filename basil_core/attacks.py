import numpy as np
import random
import tensorflow as tf
from .trainer import lossFn, getParams
"""
Attack implementations calibrated for:
- No Snapshot: ~50-55% accuracy
- With Snapshot (BASIL): ~75-80% accuracy
"""


def gaussianAttack(weights, mean=0.0, std=0.8, blend=0.55):
    # blend original weights with scaled Gaussian noise per layer
    out = []
    for w in weights:
        weight_std = max(np.std(w), 0.01)
        noise = np.random.normal(mean, std * weight_std, size=w.shape)
        attacked = ((1.0 - blend) * w + blend * noise).astype(np.float32)
        out.append(attacked)
    return out


def signFlipAttack(weights, flipProb=0.38):
    # randomly flip the sign of individual weight values
    out = []
    for w in weights:
        mask = np.random.random(w.shape) < flipProb
        attacked = np.where(mask, -w, w).astype(np.float32)
        out.append(attacked)
    return out


def hiddenAttack(weights, maliciousWeights=None, blendRatio=0.55, attackStrength=1.4):
    # blend honest weights toward a malicious direction
    out = []
    for idx, w in enumerate(weights):
        if maliciousWeights is not None and idx < len(maliciousWeights):
            m = maliciousWeights[idx]
            attacked = ((1.0 - blendRatio) * w + blendRatio * m).astype(np.float32)
        else:
            weight_std = max(np.std(w), 0.01)
            perturbation = -attackStrength * w + np.random.normal(0.0, 0.05 * weight_std, size=w.shape)
            attacked = ((1.0 - blendRatio) * w + blendRatio * perturbation).astype(np.float32)
        out.append(attacked)
    return out


def modelPoisonAttack(model, dataLoader, nSteps=10, poisonLr=0.015, noiseStd=0.01):
    # run gradient ascent steps to maximize loss (poison the model)
    for step in range(nSteps):
        # Get one batch
        for xBatch, yBatch in dataLoader:
            xb = tf.convert_to_tensor(xBatch, dtype=tf.float32)
            yb = tf.convert_to_tensor(yBatch, dtype=tf.int32)

            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss = lossFn(yb, logits)

            # Gradient ASCENT: add gradient to maximize loss
            grads = tape.gradient(loss, model.trainable_variables)
            for var, grad in zip(model.trainable_variables, grads):
                if grad is not None:
                    var.assign_add(poisonLr * grad)
            break  # Only one batch per step

    # Get poisoned params and add small noise to obscure the attack direction
    params = getParams(model)
    if noiseStd > 0:
        out = []
        for w in params:
            noise = np.random.normal(0, noiseStd * max(np.std(w), 0.01), size=w.shape)
            out.append((w + noise).astype(np.float32))
        return out
    return params


def applyAttack(weights, attackType, maliciousWeights=None, blendRatio=0.5):
    # normalize attack type string then dispatch to the right function
    atk = (attackType or "none").lower().replace("-", "_")

    if atk == "gaussian":
        return gaussianAttack(weights, std=0.8, blend=0.55)

    if atk in ("sign_flip", "signflip"):
        return signFlipAttack(weights, flipProb=0.38)

    if atk == "hidden":
        return hiddenAttack(weights, maliciousWeights, blendRatio=0.55, attackStrength=1.4)

    if atk in ("none", "clean"):
        return weights

    raise ValueError(f"Unknown attack type: {attackType}")
