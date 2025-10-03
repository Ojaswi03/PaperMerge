import numpy as np
import random

def gaussian_attack(weights, mean=0.0, std=1.0):
    """Replace each layer with Gaussian noise of same shape."""
    out = []
    for w in weights:
        noise = np.random.normal(mean, std, size=w.shape).astype(np.float32)
        out.append(noise)
    return out

def sign_flip_attack(weights):
    """Layer-wise random sign flip."""
    out = []
    for w in weights:
        if random.random() < 0.5:
            out.append((-w).astype(np.float32))
        else:
            out.append(w.astype(np.float32))
    return out

def hidden_attack(weights, malicious_weights=None, blend_ratio=0.5):
    """
    Blend in a malicious vector (same shapes) with ratio.
    If none provided, just small random bias.
    """
    out = []
    for idx, w in enumerate(weights):
        if malicious_weights is not None and idx < len(malicious_weights):
            m = malicious_weights[idx]
        else:
            m = np.random.normal(0.0, 0.1, size=w.shape).astype(np.float32)
        out.append(((1.0 - blend_ratio) * w + blend_ratio * m).astype(np.float32))
    return out

def apply_attack(weights, attack_type, malicious_weights=None, blend_ratio=0.5):
    """Dispatch."""
    if attack_type == 'gaussian':
        return gaussian_attack(weights)
    if attack_type == 'sign_flip':
        return sign_flip_attack(weights)
    if attack_type == 'hidden':
        return hidden_attack(weights, malicious_weights, blend_ratio)
    if attack_type == 'none':
        return weights
    raise ValueError(f"Unknown attack type: {attack_type}")
