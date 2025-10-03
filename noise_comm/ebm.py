import copy
import torch

@torch.no_grad()
def add_gaussian_noise_state_dict(state_dict, sigma, device=None):
    """
    Return a NEW state_dict with N(0, sigma^2) noise added to floating tensors.
    Safe to call with sigma <= 0 (returns original).
    """
    if not sigma or sigma <= 0:
        return state_dict
    noisy = {k: v.clone() for k, v in state_dict.items()}
    for k, v in noisy.items():
        if torch.is_floating_point(v):
            d = device if device is not None else v.device
            noise = torch.normal(0.0, sigma, size=v.shape, device=d)
            noisy[k] = v + noise
    return noisy

def ebm_grad_norm_sq(loss, model, create_graph=True):
    """
    Compute ||∇ loss||^2 over model parameters (trainable).
    Returns a scalar tensor (0.0 if no grads).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params,
                                create_graph=create_graph,
                                retain_graph=True,
                                allow_unused=False)
    total = None
    for g in grads:
        term = (g ** 2).sum()
        total = term if total is None else (total + term)
    return torch.tensor(0.0, device=params[0].device) if total is None else total
