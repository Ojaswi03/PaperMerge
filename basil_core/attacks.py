import numpy as np
import random

def gaussianAttack(weights, mean=0.0, std=1.0):
    """Replace each layer with Gaussian noise of same shape."""
    out = []
    for w in weights:
        noise = np.random.normal(mean, std, size=w.shape).astype(np.float32)
        out.append(noise)
    return out

def signFlipAttack(weights):
    """Layer-wise random sign flip."""
    out = []
    for w in weights:
        if random.random() < 0.5:
            out.append((-w).astype(np.float32))
        else:
            out.append(w.astype(np.float32))
    return out

def hiddenAttack(weights, maliciousWeights=None, blendRatio=0.5):
    """
    Blend in a malicious vector (same shapes) with ratio.
    If none provided, small random Gaussian bias.
    """
    out = []
    for idx, w in enumerate(weights):
        if maliciousWeights is not None and idx < len(maliciousWeights):
            m = maliciousWeights[idx]
        else:
            m = np.random.normal(0.0, 0.1, size=w.shape).astype(np.float32)
        out.append(((1.0 - blendRatio) * w + blendRatio * m).astype(np.float32))
    return out

def applyAttack(weights, attackType, maliciousWeights=None, blendRatio=0.5):
    """Dispatch with aliases: 'sign-flip' == 'sign_flip'."""
    atk = (attackType or "none").lower().replace("-", "_")
    if atk == "gaussian":
        return gaussianAttack(weights)
    if atk == "sign_flip":
        return signFlipAttack(weights)
    if atk == "hidden":
        return hiddenAttack(weights, maliciousWeights, blendRatio)
    if atk == "none":
        return weights
    raise ValueError(f"Unknown attack type: {attackType}")
