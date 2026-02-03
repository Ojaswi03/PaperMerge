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

def hiddenAttack(weights, maliciousWeights=None, blendRatio=0.3, attackStrength=0.5):
    """
    Hidden/omniscient Byzantine attack (Paper Section V-A).

    Paper description: Byzantine nodes are omniscient - they collect all benign
    models and design models that are statistically indistinguishable from
    honest ones but push the global model in a malicious direction.

    Implementation: Since we can't be truly omniscient without access to all
    other models, we approximate by:
    1. Keeping most weights similar (small perturbations)
    2. Pushing specific layers in a bad direction (sign reversal scaled down)

    Args:
        weights: Current model weights
        maliciousWeights: Optional target weights (if None, use scaled sign reversal)
        blendRatio: How much to blend toward malicious direction (default 0.3)
        attackStrength: Strength of perturbation (default 0.5)

    Returns:
        Attacked weights that look similar but push model in bad direction
    """
    out = []
    for idx, w in enumerate(weights):
        if maliciousWeights is not None and idx < len(maliciousWeights):
            # Blend toward provided malicious weights
            m = maliciousWeights[idx]
            attacked = ((1.0 - blendRatio) * w + blendRatio * m).astype(np.float32)
        else:
            # Omniscient-style attack: subtly push weights in wrong direction
            # Small-magnitude sign reversal that's hard to detect
            perturbation = -attackStrength * w + np.random.normal(0.0, 0.01, size=w.shape)
            attacked = (w + blendRatio * perturbation).astype(np.float32)
        out.append(attacked)
    return out

def applyAttack(weights, attackType, maliciousWeights=None, blendRatio=0.5):
    """Dispatch with aliases: 'sign-flip' == 'sign_flip' == 'signFlip' == 'signflip'."""
    atk = (attackType or "none").lower().replace("-", "_")
    if atk == "gaussian":
        return gaussianAttack(weights)
    # Handle all variations: sign_flip, signflip, sign-flip, signFlip
    if atk in ("sign_flip", "signflip"):
        return signFlipAttack(weights)
    if atk == "hidden":
        return hiddenAttack(weights, maliciousWeights, blendRatio)
    if atk in ("none", "clean"):
        return weights
    raise ValueError(f"Unknown attack type: {attackType}")
