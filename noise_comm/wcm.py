"""Worst-case communication model from Ang et al. (2019).

This module implements the optimization structure in Sections III-B and V of
"Robust Federated Learning with Noisy Communication":

* bounded full-model uncertainty, ``||delta||_2 <= radius`` (Eq. 8-10);
* one or more samples on the uncertainty boundary (Eq. 27);
* the recursive gradient estimate ``G_t`` (Eq. 32/34);
* the SCA surrogate in Eq. 31; and
* the conditional update in Eq. 36b.

The paper assumes an absolute uncertainty radius. This repository transmits
noise using a relative full-model L2 budget, so ``relative_radius=True`` maps
``sigma`` to ``sigma * ||w||_2``. The pilot code records this adaptation.

The finite inner SGD loop approximates the ``arg min`` in Eq. 30. It is not
claimed to be an exact solution of the paper's non-convex subproblem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf


ArrayList = list[np.ndarray]


@dataclass
class WcmState:
    """Per-client state for the recursive estimate in Equation 32."""

    gradient_estimate: ArrayList | None = None
    iteration: int = 0

    def clone(self) -> "WcmState":
        return WcmState(
            gradient_estimate=(
                None
                if self.gradient_estimate is None
                else [
                    np.asarray(value, dtype=np.float32).copy()
                    for value in self.gradient_estimate
                ]
            ),
            iteration=int(self.iteration),
        )


def _as_float32_params(params: Iterable) -> ArrayList:
    return [
        np.asarray(value, dtype=np.float32)
        for value in params
    ]


def fullModelL2Norm(params: Iterable) -> float:
    """Return the L2 norm of the flattened full model."""
    squared = 0.0
    for value in params:
        array = np.asarray(value, dtype=np.float32)
        squared += float(np.sum(array * array, dtype=np.float64))
    return float(math.sqrt(max(0.0, squared)))


def sampleBoundaryPayload(
    params,
    sigma,
    rng=None,
    *,
    relative_radius=True,
) -> ArrayList:
    """Sample one isotropic direction on the full-model L2 boundary.

    ``sigma`` is the absolute radius when ``relative_radius=False``. With the
    project adaptation ``relative_radius=True``, the target boundary norm is
    ``sigma * ||params||_2``.
    """
    arrays = _as_float32_params(params)
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("WCM uncertainty radius must be non-negative.")
    if not arrays or sigma == 0.0:
        return [np.zeros_like(value) for value in arrays]

    generator = rng if rng is not None else np.random.default_rng()
    directions = [
        np.asarray(
            generator.normal(0.0, 1.0, size=value.shape),
            dtype=np.float32,
        )
        for value in arrays
    ]
    direction_norm = fullModelL2Norm(directions)
    if direction_norm <= 1e-12:
        directions = [np.zeros_like(value) for value in arrays]
        first_nonempty = next(
            (value for value in directions if value.size),
            None,
        )
        if first_nonempty is None:
            return directions
        first_nonempty.flat[0] = 1.0
        direction_norm = 1.0

    target_norm = sigma
    if relative_radius:
        target_norm *= fullModelL2Norm(arrays)
    scale = target_norm / direction_norm
    return [
        np.asarray(direction * scale, dtype=np.float32)
        for direction in directions
    ]


def applyDelta(params, deltas, alpha=1.0) -> ArrayList:
    """Apply a structured parameter perturbation."""
    params = _as_float32_params(params)
    deltas = _as_float32_params(deltas)
    if len(params) != len(deltas):
        raise ValueError("WCM parameter and delta structures do not match.")
    return [
        np.asarray(param + float(alpha) * delta, dtype=np.float32)
        for param, delta in zip(params, deltas)
    ]


def paperSequence(iteration: int, exponent: float) -> float:
    """Return ``(t + 1)^(-exponent)`` for the paper's step sequences."""
    if int(iteration) < 0:
        raise ValueError("WCM iteration cannot be negative.")
    if not 0.0 < float(exponent) < 1.0:
        raise ValueError("WCM sequence exponent must be between 0 and 1.")
    return float((int(iteration) + 1) ** (-float(exponent)))


def validatePaperExponents(rho_exponent: float, gamma_exponent: float) -> None:
    """Validate the convergence condition ``0.5 < beta < alpha < 1``."""
    beta = float(rho_exponent)
    alpha = float(gamma_exponent)
    if not 0.5 < beta < alpha < 1.0:
        raise ValueError(
            "WCM requires 0.5 < rho_exponent < gamma_exponent < 1 "
            "(paper Lemma 7)."
        )


def updateGradientEstimate(
    previous,
    sampled,
    rho,
) -> ArrayList:
    """Apply ``G_t = (1-rho)G_{t-1} + rho*grad F(w+delta)``."""
    sampled = _as_float32_params(sampled)
    if previous is None:
        previous = [np.zeros_like(value) for value in sampled]
    previous = _as_float32_params(previous)
    if len(previous) != len(sampled):
        raise ValueError("WCM gradient-estimate structure does not match.")
    rho = float(rho)
    if not 0.0 < rho <= 1.0:
        raise ValueError("WCM rho must be in (0, 1].")
    return [
        np.asarray((1.0 - rho) * old + rho * current, dtype=np.float32)
        for old, current in zip(previous, sampled)
    ]


def _batch_slices(batch_size: int, micro_batch_size: int):
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        yield start, end, float(end - start) / float(batch_size)


def lossAndGradientsAtDelta(
    model,
    lossFn,
    x,
    y,
    deltas,
    *,
    micro_batch_size=128,
) -> tuple[float, ArrayList]:
    """Evaluate loss and gradient at ``w + delta`` with bounded activations."""
    variables = list(model.trainable_variables)
    deltas = _as_float32_params(deltas)
    if len(variables) != len(deltas):
        raise ValueError("WCM delta structure does not match model variables.")

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    batch_size = x.shape[0]
    if batch_size is None:
        batch_size = int(tf.shape(x)[0].numpy())
    else:
        batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("WCM requires a non-empty training batch.")
    micro_batch_size = max(1, int(micro_batch_size))

    original = [tf.identity(variable) for variable in variables]
    for variable, value, delta in zip(variables, original, deltas):
        variable.assign(value + tf.convert_to_tensor(delta, dtype=variable.dtype))

    total_loss = 0.0
    gradient_sums = [tf.zeros_like(variable) for variable in variables]
    try:
        for start, end, factor in _batch_slices(batch_size, micro_batch_size):
            with tf.GradientTape() as tape:
                logits = model(x[start:end], training=True)
                loss = lossFn(y[start:end], logits)
            gradients = tape.gradient(
                loss,
                variables,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            total_loss += float(loss.numpy()) * factor
            gradient_sums = [
                total + gradient * factor
                for total, gradient in zip(gradient_sums, gradients)
            ]
    finally:
        for variable, value in zip(variables, original):
            variable.assign(value)

    return (
        float(total_loss),
        [
            np.asarray(gradient.numpy(), dtype=np.float32)
            for gradient in gradient_sums
        ],
    )


def scaSurrogateGradients(
    *,
    noisy_gradients,
    candidate,
    reference,
    previous_gradient_estimate,
    rho,
    penalty,
    extra_prox_mu=0.0,
) -> ArrayList:
    """Differentiate the Equation 31 surrogate at a candidate model."""
    noisy_gradients = _as_float32_params(noisy_gradients)
    candidate = _as_float32_params(candidate)
    reference = _as_float32_params(reference)
    if previous_gradient_estimate is None:
        previous_gradient_estimate = [
            np.zeros_like(value) for value in noisy_gradients
        ]
    previous_gradient_estimate = _as_float32_params(
        previous_gradient_estimate
    )
    structures = (
        noisy_gradients,
        candidate,
        reference,
        previous_gradient_estimate,
    )
    if len({len(values) for values in structures}) != 1:
        raise ValueError("WCM surrogate structures do not match.")

    rho = float(rho)
    penalty = float(penalty)
    extra_prox_mu = float(extra_prox_mu)
    if not 0.0 < rho <= 1.0:
        raise ValueError("WCM rho must be in (0, 1].")
    if penalty <= 0.0:
        raise ValueError("WCM SCA penalty must be positive.")
    if extra_prox_mu < 0.0:
        raise ValueError("Additional proximal strength cannot be negative.")

    return [
        np.asarray(
            rho * noisy
            + (1.0 - rho) * history
            + (2.0 * penalty + extra_prox_mu) * (current - anchor),
            dtype=np.float32,
        )
        for noisy, history, current, anchor in zip(
            noisy_gradients,
            previous_gradient_estimate,
            candidate,
            reference,
        )
    ]


def _clip_gradients(gradients, clip_norm):
    tensors = [
        tf.convert_to_tensor(value, dtype=tf.float32)
        for value in gradients
    ]
    unclipped_norm = float(tf.linalg.global_norm(tensors).numpy())
    if clip_norm is not None and float(clip_norm) > 0.0:
        tensors, _ = tf.clip_by_global_norm(tensors, float(clip_norm))
    return (
        [
            np.asarray(value.numpy(), dtype=np.float32)
            for value in tensors
        ],
        unclipped_norm,
    )


def wcmLocalUpdate(
    *,
    model,
    lossFn,
    batches: Sequence[tuple],
    state: WcmState | None,
    uncertainty_radius,
    penalty,
    inner_lr,
    rho_exponent=0.6,
    gamma_exponent=0.8,
    boundary_samples=1,
    relative_radius=True,
    clip_norm=5.0,
    extra_prox_mu=0.0,
    rng=None,
    rho_override=None,
    gamma_override=None,
) -> tuple[WcmState, dict]:
    """Perform one sampled-SCA local update.

    Each item in ``batches`` is one finite SGD step toward the Equation 30
    subproblem. The resulting candidate is blended with the reference using
    Equation 36b.
    """
    validatePaperExponents(rho_exponent, gamma_exponent)
    if not batches:
        raise ValueError("WCM requires at least one local batch.")
    if int(boundary_samples) < 1:
        raise ValueError("WCM boundary_samples must be at least one.")
    if float(inner_lr) <= 0.0:
        raise ValueError("WCM inner learning rate must be positive.")
    if float(uncertainty_radius) < 0.0:
        raise ValueError("WCM uncertainty radius cannot be negative.")

    state = WcmState() if state is None else state.clone()
    reference = [
        np.asarray(variable.numpy(), dtype=np.float32)
        for variable in model.trainable_variables
    ]
    previous_gradient = (
        [np.zeros_like(value) for value in reference]
        if state.gradient_estimate is None
        else _as_float32_params(state.gradient_estimate)
    )
    if len(previous_gradient) != len(reference):
        raise ValueError("WCM state does not match model variables.")

    rho = (
        paperSequence(state.iteration, rho_exponent)
        if rho_override is None
        else float(rho_override)
    )
    gamma = (
        paperSequence(state.iteration, gamma_exponent)
        if gamma_override is None
        else float(gamma_override)
    )
    if not 0.0 < rho <= 1.0 or not 0.0 < gamma <= 1.0:
        raise ValueError("WCM rho and gamma must be in (0, 1].")

    generator = rng if rng is not None else np.random.default_rng()
    deltas = [
        sampleBoundaryPayload(
            reference,
            uncertainty_radius,
            generator,
            relative_radius=relative_radius,
        )
        for _ in range(int(boundary_samples))
    ]
    state_gradient = None
    losses = []
    last_surrogate_norm = 0.0

    for x, y in batches:
        sampled_losses = []
        sampled_gradients = []
        for delta in deltas:
            loss, gradients = lossAndGradientsAtDelta(
                model,
                lossFn,
                x,
                y,
                delta,
            )
            sampled_losses.append(loss)
            sampled_gradients.append(gradients)
        mean_gradients = [
            np.mean(
                np.stack(
                    [sample[index] for sample in sampled_gradients],
                    axis=0,
                ),
                axis=0,
            ).astype(np.float32)
            for index in range(len(reference))
        ]
        if state_gradient is None:
            # The first local step is evaluated exactly at the round reference,
            # matching the gradient sample in Equation 32.
            state_gradient = [
                value.copy() for value in mean_gradients
            ]
        candidate = [
            np.asarray(variable.numpy(), dtype=np.float32)
            for variable in model.trainable_variables
        ]
        surrogate = scaSurrogateGradients(
            noisy_gradients=mean_gradients,
            candidate=candidate,
            reference=reference,
            previous_gradient_estimate=previous_gradient,
            rho=rho,
            penalty=penalty,
            extra_prox_mu=extra_prox_mu,
        )
        surrogate, last_surrogate_norm = _clip_gradients(
            surrogate,
            clip_norm,
        )
        for variable, gradient in zip(model.trainable_variables, surrogate):
            variable.assign_sub(
                tf.convert_to_tensor(
                    float(inner_lr) * gradient,
                    dtype=variable.dtype,
                )
            )
        losses.append(float(np.mean(sampled_losses)))

    candidate = [
        np.asarray(variable.numpy(), dtype=np.float32)
        for variable in model.trainable_variables
    ]
    updated = [
        np.asarray(
            anchor + gamma * (value - anchor),
            dtype=np.float32,
        )
        for anchor, value in zip(reference, candidate)
    ]
    for variable, value in zip(model.trainable_variables, updated):
        variable.assign(tf.convert_to_tensor(value, dtype=variable.dtype))

    new_state = WcmState(
        gradient_estimate=updateGradientEstimate(
            previous_gradient,
            state_gradient,
            rho,
        ),
        iteration=state.iteration + 1,
    )
    boundary_norms = [
        fullModelL2Norm(delta)
        for delta in deltas
    ]
    diagnostics = {
        "loss": float(np.mean(losses)),
        "rho": float(rho),
        "gamma": float(gamma),
        "boundary_norm": float(np.mean(boundary_norms)),
        "reference_norm": fullModelL2Norm(reference),
        "candidate_distance": fullModelL2Norm(
            [
                value - anchor
                for value, anchor in zip(candidate, reference)
            ]
        ),
        "update_distance": fullModelL2Norm(
            [
                value - anchor
                for value, anchor in zip(updated, reference)
            ]
        ),
        "surrogate_gradient_norm": float(last_surrogate_norm),
        "boundary_samples": int(boundary_samples),
        "inner_steps": len(batches),
    }
    return new_state, diagnostics


def scaSurrogateLoss(
    lossFn,
    model,
    x,
    y,
    deltaList,
    rho,
    lam,
    wPrev=None,
    gPrev=None,
):
    """Compatibility helper returning the Equation 31 surrogate value."""
    variables = list(model.trainable_variables)
    candidate = [
        np.asarray(variable.numpy(), dtype=np.float32)
        for variable in variables
    ]
    reference = candidate if wPrev is None else _as_float32_params(wPrev)
    history = (
        [np.zeros_like(value) for value in candidate]
        if gPrev is None
        else _as_float32_params(gPrev)
    )
    original = [tf.identity(variable) for variable in variables]
    for variable, value, delta in zip(
        variables,
        original,
        _as_float32_params(deltaList),
    ):
        variable.assign(value + tf.convert_to_tensor(delta, dtype=variable.dtype))
    try:
        logits = model(tf.convert_to_tensor(x, dtype=tf.float32), training=True)
        base = lossFn(tf.convert_to_tensor(y, dtype=tf.int32), logits)
    finally:
        for variable, value in zip(variables, original):
            variable.assign(value)

    prox = sum(
        tf.reduce_sum(
            tf.square(
                tf.convert_to_tensor(current - anchor, dtype=tf.float32)
            )
        )
        for current, anchor in zip(candidate, reference)
    )
    linear = sum(
        tf.reduce_sum(
            tf.convert_to_tensor(current - anchor, dtype=tf.float32)
            * tf.convert_to_tensor(gradient, dtype=tf.float32)
        )
        for current, anchor, gradient in zip(candidate, reference, history)
    )
    surrogate = (
        float(rho) * base
        + float(lam) * prox
        + (1.0 - float(rho)) * linear
    )
    return surrogate, [value.copy() for value in candidate]


def _optimizer_learning_rate(optimizer) -> float:
    value = optimizer.learning_rate
    if callable(value):
        value = value(optimizer.iterations)
    return float(tf.keras.backend.get_value(value))


def wcmStep(
    model,
    optimizer,
    lossFn,
    x,
    y,
    sigma,
    S,
    rho,
    lam,
    wPrev=None,
    gPrev=None,
    betaForG=0.9,
):
    """Backward-compatible one-batch WCM call for legacy training paths.

    New experiments should use :func:`wcmLocalUpdate` and preserve
    :class:`WcmState` per logical client. ``betaForG`` remains accepted for
    API compatibility; Equation 32 uses ``rho`` for the recursive update.
    """
    del betaForG
    state = WcmState(
        gradient_estimate=(
            None if gPrev is None else _as_float32_params(gPrev)
        ),
        iteration=0,
    )
    new_state, diagnostics = wcmLocalUpdate(
        model=model,
        lossFn=lossFn,
        batches=[(x, y)],
        state=state,
        uncertainty_radius=float(sigma),
        penalty=float(lam),
        inner_lr=_optimizer_learning_rate(optimizer),
        boundary_samples=max(1, int(S)),
        relative_radius=True,
        rho_override=float(rho),
        gamma_override=1.0,
    )
    return {
        **diagnostics,
        "wPrev": [
            np.asarray(variable.numpy(), dtype=np.float32)
            for variable in model.trainable_variables
        ],
        "gPrev": [
            value.copy()
            for value in new_state.gradient_estimate
        ],
    }


__all__ = [
    "WcmState",
    "applyDelta",
    "fullModelL2Norm",
    "lossAndGradientsAtDelta",
    "paperSequence",
    "sampleBoundaryPayload",
    "scaSurrogateGradients",
    "scaSurrogateLoss",
    "updateGradientEstimate",
    "validatePaperExponents",
    "wcmLocalUpdate",
    "wcmStep",
]
