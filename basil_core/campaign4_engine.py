"""GPU-resident, telemetry-rich ring engine for Campaign 4.

Campaign 4 is intentionally separate from the Campaign 3 engine. It keeps the
same research components available as controls while adding protocol-matched
clean runs, adaptive EBM, explicit optimizer-state modes, and node-level
telemetry. Logical states are tensors on the active device; only bounded
metrics cross back to the host during a run.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Callable

import numpy as np
import tensorflow as tf

from .basil import _MutableLR
from .class_registry import ClassRegistry
from .trainer import evaluate, lossFn, makeLrScheduler


TensorParams = list[tf.Tensor]
EPSILON = 1e-12


def _device_copy(params) -> TensorParams:
    return [tf.identity(tf.cast(value, tf.float32)) for value in params]


def _params_to_numpy(params) -> list[np.ndarray]:
    return [np.asarray(value.numpy(), dtype=np.float32) for value in params]


def params_hash(params) -> str:
    digest = hashlib.sha256()
    for value in params:
        array = np.asarray(value.numpy() if tf.is_tensor(value) else value, dtype=np.float32)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_diagnostics(params) -> tuple[str, float]:
    """Return a compact signature and norm with one device synchronization."""
    sums = [tf.reduce_sum(tf.cast(value, tf.float32)) for value in params]
    squared_sums = [
        tf.reduce_sum(tf.square(tf.cast(value, tf.float32))) for value in params
    ]
    values = tf.stack(sums + squared_sums).numpy().astype(np.float64, copy=False)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    model_norm = float(np.sqrt(np.maximum(np.sum(values[len(sums) :]), 0.0)))
    return digest[:16], model_norm


def _state_signature(params) -> str:
    return _state_diagnostics(params)[0]


def _seed_pair(base_seed: int, *parts) -> tuple[int, int]:
    text = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    first = int.from_bytes(digest[:4], "little", signed=False) & 0x7FFFFFFF
    second = int.from_bytes(digest[4:8], "little", signed=False) & 0x7FFFFFFF
    return first, second


def _global_norm(params) -> tf.Tensor:
    return tf.linalg.global_norm([tf.cast(value, tf.float32) for value in params])


def _relative_parameter_distance(left, right) -> tf.Tensor:
    differences = [
        tf.cast(right_value, tf.float32) - tf.cast(left_value, tf.float32)
        for left_value, right_value in zip(left, right)
    ]
    return _global_norm(differences) / tf.maximum(
        _global_norm(left), tf.constant(EPSILON, tf.float32)
    )


def _average_params(left, right, left_weight: float = 0.5) -> TensorParams:
    left_weight = tf.constant(float(left_weight), dtype=tf.float32)
    right_weight = tf.constant(1.0, dtype=tf.float32) - left_weight
    return [
        left_weight * tf.cast(left_value, tf.float32)
        + right_weight * tf.cast(right_value, tf.float32)
        for left_value, right_value in zip(left, right)
    ]


def _average_many(params_list: list[TensorParams]) -> TensorParams:
    if not params_list:
        return []
    count = tf.constant(float(len(params_list)), dtype=tf.float32)
    return [
        tf.add_n([tf.cast(params[layer], tf.float32) for params in params_list])
        / count
        for layer in range(len(params_list[0]))
    ]


def _relative_l2_channel_noise(
    params,
    *,
    sigma: float,
    base_seed: int,
    seed_parts: tuple,
) -> tuple[TensorParams, float, float]:
    """Apply deterministic stateless relative-L2 Gaussian link noise on-device."""
    if float(sigma) <= 0.0:
        zero = tf.constant(0.0, tf.float32)
        return _device_copy(params), zero, zero
    model_norm = _global_norm(params)
    total_dimension = sum(int(np.prod(value.shape)) for value in params)
    coordinate_std = (
        tf.constant(float(sigma), tf.float32)
        * model_norm
        / tf.sqrt(tf.constant(float(total_dimension), tf.float32))
    )
    transmitted = []
    noises = []
    for layer_id, value in enumerate(params):
        seed = tf.constant(
            _seed_pair(base_seed, *seed_parts, layer_id), dtype=tf.int32
        )
        noise = tf.random.stateless_normal(
            tf.shape(value), seed=seed, stddev=coordinate_std, dtype=tf.float32
        )
        noises.append(noise)
        transmitted.append(tf.cast(value, tf.float32) + noise)
    noise_norm = _global_norm(noises)
    relative = noise_norm / tf.maximum(model_norm, tf.constant(EPSILON, tf.float32))
    return transmitted, noise_norm, relative


def _hidden_attack(
    params,
    *,
    base_seed: int,
    round_id: int,
    node_id: int,
) -> tuple[TensorParams, float]:
    """Campaign 3 hidden-attack distribution implemented with stateless TF RNG."""
    attacked = []
    for layer_id, value in enumerate(params):
        value = tf.cast(value, tf.float32)
        weight_std = tf.maximum(tf.math.reduce_std(value), 0.01)
        seed = tf.constant(
            _seed_pair(base_seed, "hidden", round_id, node_id, layer_id),
            dtype=tf.int32,
        )
        perturbation_noise = tf.random.stateless_normal(
            tf.shape(value),
            seed=seed,
            stddev=0.05 * weight_std,
            dtype=tf.float32,
        )
        malicious_direction = -1.4 * value + perturbation_noise
        attacked.append(0.45 * value + 0.55 * malicious_direction)
    relative = float(_relative_parameter_distance(params, attacked).numpy())
    return attacked, relative


def _microbatch_slices(batch_size, micro_batch_size: int):
    if batch_size is None:
        return ((0, None, 1.0),)
    batch_size = int(batch_size)
    return tuple(
        (
            start,
            min(start + int(micro_batch_size), batch_size),
            float(min(start + int(micro_batch_size), batch_size) - start)
            / float(batch_size),
        )
        for start in range(0, batch_size, int(micro_batch_size))
    )


def _base_gradients(
    model_inner,
    weights,
    x,
    y,
    *,
    micro_batch_size: int,
):
    sums = [tf.zeros_like(weight, dtype=tf.float32) for weight in weights]
    for start, end, factor in _microbatch_slices(x.shape[0], micro_batch_size):
        with tf.GradientTape() as tape:
            logits = model_inner(x[start:end], training=True)
            loss = lossFn(y[start:end], logits)
        gradients = tape.gradient(
            loss, weights, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        sums = [
            total + tf.cast(gradient, tf.float32) * factor
            for total, gradient in zip(sums, gradients)
        ]
    return sums


def _base_and_regularizer_gradients(
    model_inner,
    weights,
    x,
    y,
    *,
    micro_batch_size: int,
):
    """Return base and gradient-norm gradients for one training batch.

    The declared EBM objective is exact when ``micro_batch_size`` covers the
    full configured batch. Smaller slices average per-slice regularizers and
    are therefore restricted to explicitly labeled diagnostics.
    """
    base_sums = [tf.zeros_like(weight, dtype=tf.float32) for weight in weights]
    regularizer_sums = [tf.zeros_like(weight, dtype=tf.float32) for weight in weights]
    for start, end, factor in _microbatch_slices(x.shape[0], micro_batch_size):
        batch_x = x[start:end]
        batch_y = y[start:end]
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                logits = model_inner(batch_x, training=True)
                loss = lossFn(batch_y, logits)
            base = inner_tape.gradient(
                loss, weights, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
            base = [tf.cast(value, tf.float32) for value in base]
            norm_squared = tf.add_n(
                [tf.reduce_sum(tf.square(value)) for value in base]
            )
        regularizer = outer_tape.gradient(
            norm_squared,
            weights,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        base_sums = [
            total + value * factor for total, value in zip(base_sums, base)
        ]
        regularizer_sums = [
            total + tf.cast(value, tf.float32) * factor
            for total, value in zip(regularizer_sums, regularizer)
        ]
    return base_sums, regularizer_sums


def _adaptive_requested_coefficient(
    *,
    previous,
    stress,
    base_norm,
    regularizer_norm,
    target_ratio_base,
    stress_gain,
    ratio_min,
    ratio_max,
    coefficient_min,
    coefficient_max,
):
    """Return the finite bounded coefficient requested by current stress."""
    desired_ratio = tf.clip_by_value(
        target_ratio_base + stress_gain * stress,
        ratio_min,
        ratio_max,
    )
    raw = desired_ratio * base_norm / tf.maximum(regularizer_norm, 1e-12)
    raw = tf.clip_by_value(raw, coefficient_min, coefficient_max)
    previous = tf.clip_by_value(previous, coefficient_min, coefficient_max)
    finite = (
        tf.math.is_finite(raw)
        & tf.math.is_finite(base_norm)
        & tf.math.is_finite(regularizer_norm)
        & tf.math.is_finite(stress)
    )
    return tf.where(finite, raw, previous)


def _bounded_adaptive_coefficient(
    *,
    previous,
    stress,
    base_norm,
    regularizer_norm,
    target_ratio_base,
    stress_gain,
    ratio_min,
    ratio_max,
    coefficient_min,
    coefficient_max,
    beta,
    max_change_factor,
):
    """Compute one bounded adaptive coefficient using receiver-observable state."""
    requested = _adaptive_requested_coefficient(
        previous=previous,
        stress=stress,
        base_norm=base_norm,
        regularizer_norm=regularizer_norm,
        target_ratio_base=target_ratio_base,
        stress_gain=stress_gain,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        coefficient_min=coefficient_min,
        coefficient_max=coefficient_max,
    )
    previous = tf.clip_by_value(previous, coefficient_min, coefficient_max)
    smoothed = beta * previous + (1.0 - beta) * requested
    lower = previous / tf.maximum(max_change_factor, 1.0)
    upper = previous * tf.maximum(max_change_factor, 1.0)
    return tf.clip_by_value(smoothed, lower, upper)


@dataclass
class RingSnapshot:
    sender_id: int
    round_id: int
    params: TensorParams
    registry: ClassRegistry | None = None
    noise_norm: float = 0.0
    relative_noise: float = 0.0
    attacked: bool = False
    state_signature: str = ""


@dataclass
class SelectionDecision:
    snapshot: RingSnapshot
    senders: list[int]
    distances: list[float]
    losses: list[float]
    plausible: list[bool]
    fallback: bool


@dataclass
class LogicalNode:
    node_id: int
    params: TensorParams
    optimizer_state: TensorParams
    data_loader: object
    probe_x: tf.Tensor
    probe_y: tf.Tensor
    probe_y_numpy: np.ndarray
    memory_size: int
    registry: ClassRegistry = field(default_factory=lambda: ClassRegistry(10))
    class_acc: np.ndarray = field(
        default_factory=lambda: np.full(10, np.nan, dtype=np.float32)
    )
    class_support: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.int32)
    )
    ema_gap: float = 0.0
    stress_ema: float = 0.0
    adaptive_coefficient: float = 0.0
    memory: OrderedDict = field(default_factory=OrderedDict)

    def receive(self, snapshot: RingSnapshot) -> None:
        self.memory[snapshot.sender_id] = snapshot
        self.memory.move_to_end(snapshot.sender_id)
        while len(self.memory) > self.memory_size:
            self.memory.popitem(last=False)


class SharedDeviceWorker:
    def __init__(
        self,
        model,
        *,
        lr0: float,
        momentum: float,
        micro_batch_size: int,
        jit_compile: bool,
        adaptive_config: dict,
        weight_decay: float = 0.0,
    ):
        self.model = model
        self.micro_batch_size = int(micro_batch_size)
        self._lr = _MutableLR(float(lr0))
        self._optimizer = tf.keras.optimizers.SGD(
            learning_rate=self._lr,
            momentum=float(momentum),
            weight_decay=(float(weight_decay) if float(weight_decay) > 0.0 else None),
        )
        self._optimizer.build(self.model.trainable_weights)
        self._optimizer_slots = [
            variable
            for variable in self._optimizer.variables
            if variable is not self._optimizer.iterations and variable.shape.rank != 0
        ]
        self._prox_mu = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._coefficient = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._requested_coefficient = tf.Variable(
            0.0, trainable=False, dtype=tf.float32
        )
        self._stress = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._mode = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._ref_params = [
            tf.Variable(tf.cast(weight, tf.float32), trainable=False)
            for weight in self.model.trainable_weights
        ]

        self._diag_steps = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._diag_base = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._diag_regularizer = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._diag_preclip = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._diag_ratio = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._diag_clipped = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        cfg = adaptive_config
        self._target_ratio_base = tf.constant(
            float(cfg["adaptiveEbmTargetRatioBase"]), tf.float32
        )
        self._stress_gain = tf.constant(float(cfg["adaptiveEbmStressGain"]), tf.float32)
        self._ratio_min = tf.constant(float(cfg["adaptiveEbmRatioMin"]), tf.float32)
        self._ratio_max = tf.constant(float(cfg["adaptiveEbmRatioMax"]), tf.float32)
        self._coefficient_min = tf.constant(
            float(cfg["adaptiveEbmCoefficientMin"]), tf.float32
        )
        self._coefficient_max = tf.constant(
            float(cfg["adaptiveEbmCoefficientMax"]), tf.float32
        )
        self._adaptive_beta = tf.constant(float(cfg["adaptiveEbmBeta"]), tf.float32)
        self._max_change_factor = tf.constant(
            float(cfg["adaptiveEbmMaxChangeFactor"]), tf.float32
        )

        model_inner = self.model.model
        weights = model_inner.trainable_weights
        optimizer = self._optimizer
        prox_mu = self._prox_mu
        ref_params = self._ref_params

        def apply_gradients(base, regularizer):
            base_norm = _global_norm(base)
            regularizer_norm = _global_norm(regularizer)
            coefficient = self._coefficient
            if_mode_adaptive = tf.equal(self._mode, 2)
            requested = _adaptive_requested_coefficient(
                previous=coefficient,
                stress=self._stress,
                base_norm=base_norm,
                regularizer_norm=regularizer_norm,
                target_ratio_base=self._target_ratio_base,
                stress_gain=self._stress_gain,
                ratio_min=self._ratio_min,
                ratio_max=self._ratio_max,
                coefficient_min=self._coefficient_min,
                coefficient_max=self._coefficient_max,
            )
            adapted = _bounded_adaptive_coefficient(
                previous=coefficient,
                stress=self._stress,
                base_norm=base_norm,
                regularizer_norm=regularizer_norm,
                target_ratio_base=self._target_ratio_base,
                stress_gain=self._stress_gain,
                ratio_min=self._ratio_min,
                ratio_max=self._ratio_max,
                coefficient_min=self._coefficient_min,
                coefficient_max=self._coefficient_max,
                beta=self._adaptive_beta,
                max_change_factor=self._max_change_factor,
            )
            coefficient = tf.where(if_mode_adaptive, adapted, coefficient)
            requested = tf.where(if_mode_adaptive, requested, coefficient)
            self._coefficient.assign(coefficient)
            self._requested_coefficient.assign(requested)
            robust = [
                base_value + coefficient * regularizer_value
                for base_value, regularizer_value in zip(base, regularizer)
            ]
            preclip_norm = _global_norm(robust)
            clipped, _ = tf.clip_by_global_norm(robust, 5.0)
            proximal = [
                prox_mu * (tf.cast(weight, tf.float32) - reference)
                for weight, reference in zip(weights, ref_params)
            ]
            final = [value + prox for value, prox in zip(clipped, proximal)]
            tf.debugging.assert_all_finite(
                _global_norm(final), "Campaign 4 produced a non-finite training gradient."
            )
            optimizer.apply_gradients(zip(final, weights))

            active_ratio = coefficient * regularizer_norm / tf.maximum(base_norm, 1e-12)
            self._diag_steps.assign_add(1.0)
            self._diag_base.assign_add(base_norm)
            self._diag_regularizer.assign_add(regularizer_norm)
            self._diag_preclip.assign_add(preclip_norm)
            self._diag_ratio.assign_add(active_ratio)
            self._diag_clipped.assign_add(tf.cast(preclip_norm > 5.0, tf.float32))

        def standard_step(x, y):
            base = _base_gradients(
                model_inner,
                weights,
                x,
                y,
                micro_batch_size=self.micro_batch_size,
            )
            zeros = [tf.zeros_like(value) for value in base]
            apply_gradients(base, zeros)

        def ebm_step(x, y):
            base, regularizer = _base_and_regularizer_gradients(
                model_inner,
                weights,
                x,
                y,
                micro_batch_size=self.micro_batch_size,
            )
            apply_gradients(base, regularizer)

        function_options = {
            "reduce_retracing": True,
            "jit_compile": bool(jit_compile),
        }
        self._standard_step = tf.function(standard_step, **function_options)
        self._ebm_step = tf.function(ebm_step, **function_options)
        self._predict = tf.function(
            lambda x: model_inner(x, training=False), **function_options
        )

    def load(self, params) -> None:
        for variable, value in zip(self.model.trainable_weights, params):
            variable.assign(tf.cast(value, variable.dtype))

    def export(self) -> TensorParams:
        return [tf.identity(tf.cast(value, tf.float32)) for value in self.model.trainable_weights]

    def zero_optimizer_state(self) -> TensorParams:
        return [tf.zeros(variable.shape, tf.float32) for variable in self._optimizer_slots]

    def _load_optimizer_state(self, state) -> None:
        if len(state) != len(self._optimizer_slots):
            raise ValueError("Logical-node optimizer state does not match worker slots.")
        for variable, value in zip(self._optimizer_slots, state):
            variable.assign(tf.cast(value, variable.dtype))

    def _export_optimizer_state(self) -> TensorParams:
        return [tf.identity(tf.cast(variable, tf.float32)) for variable in self._optimizer_slots]

    def _reset_diagnostics(self) -> None:
        for variable in (
            self._diag_steps,
            self._diag_base,
            self._diag_regularizer,
            self._diag_preclip,
            self._diag_ratio,
            self._diag_clipped,
        ):
            variable.assign(0.0)

    def batch_loss_tensor(self, params, batch) -> tf.Tensor:
        self.load(params)
        x, y = batch
        logits = self._predict(tf.cast(x, tf.float32))
        return tf.cast(lossFn(tf.cast(y, tf.int32), logits), tf.float32)

    def batch_loss(self, params, batch) -> float:
        return float(self.batch_loss_tensor(params, batch).numpy())

    def train(
        self,
        params,
        *,
        optimizer_state,
        data_iterator,
        first_batch,
        total_steps: int,
        lr: float,
        ebm_mode: str,
        initial_coefficient: float,
        stress: float,
        prox_mu: float,
    ):
        self.load(params)
        for reference, weight in zip(self._ref_params, self.model.trainable_weights):
            reference.assign(tf.cast(weight, tf.float32))
        self._lr.assign(float(lr))
        self._coefficient.assign(float(initial_coefficient))
        self._requested_coefficient.assign(float(initial_coefficient))
        self._stress.assign(float(stress))
        self._mode.assign({"none": 0, "static": 1, "adaptive": 2}[ebm_mode])
        self._prox_mu.assign(float(prox_mu))
        self._load_optimizer_state(optimizer_state)
        self._reset_diagnostics()
        step = self._ebm_step if ebm_mode != "none" else self._standard_step

        x, y = first_batch
        step(tf.cast(x, tf.float32), tf.cast(y, tf.int32))
        for _ in range(max(0, int(total_steps) - 1)):
            x, y = next(data_iterator)
            step(tf.cast(x, tf.float32), tf.cast(y, tf.int32))

        diagnostics_values = tf.stack(
            [
                self._diag_steps,
                self._coefficient,
                self._requested_coefficient,
                self._diag_base,
                self._diag_regularizer,
                self._diag_preclip,
                self._diag_ratio,
                self._diag_clipped,
                _global_norm(self._optimizer_slots),
            ]
        ).numpy()
        count = max(float(diagnostics_values[0]), 1.0)
        diagnostics = {
            "coefficient": float(diagnostics_values[1]) if ebm_mode != "none" else 0.0,
            "requestedCoefficient": (
                float(diagnostics_values[2]) if ebm_mode != "none" else 0.0
            ),
            "baseGradientNorm": float(diagnostics_values[3]) / count,
            "regularizerGradientNorm": float(diagnostics_values[4]) / count,
            "preclipGradientNorm": float(diagnostics_values[5]) / count,
            "activeEbmRatio": float(diagnostics_values[6]) / count,
            "clipFraction": float(diagnostics_values[7]) / count,
            "momentumNorm": float(diagnostics_values[8]),
        }
        return self.export(), self._export_optimizer_state(), diagnostics

    def probe_metrics(self, params, x, y, y_numpy):
        support = np.bincount(y_numpy, minlength=10).astype(np.int32)
        accuracy = np.full(10, np.nan, dtype=np.float32)
        if len(y_numpy) == 0:
            return accuracy, support
        self.load(params)
        predictions = tf.argmax(self._predict(tf.cast(x, tf.float32)), axis=1).numpy()
        for class_id in range(10):
            mask = y_numpy == class_id
            if np.any(mask):
                accuracy[class_id] = float(np.mean(predictions[mask] == class_id))
        return accuracy, support

    def confusion(self, params, data_loader):
        self.load(params)
        matrix = np.zeros((10, 10), dtype=np.int64)
        for x, y in data_loader:
            predictions = tf.argmax(
                self._predict(tf.cast(x, tf.float32)), axis=1
            ).numpy()
            labels = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
            np.add.at(matrix, (labels.astype(np.int64), predictions.astype(np.int64)), 1)
        return matrix


def _gpu_peak_bytes() -> int:
    try:
        return int(tf.config.experimental.get_memory_info("GPU:0").get("peak", 0))
    except Exception:
        return 0


def _class_accuracy_from_confusion(matrix):
    support = matrix.sum(axis=1)
    result = np.full(10, np.nan, dtype=np.float32)
    present = support > 0
    result[present] = matrix.diagonal()[present] / support[present]
    return result


def _accuracy_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    totals = confusion.sum(axis=(1, 2))
    correct = np.trace(confusion, axis1=1, axis2=2)
    per_node = np.divide(
        correct,
        totals,
        out=np.zeros_like(correct, dtype=np.float64),
        where=totals > 0,
    )
    return float(np.mean(per_node)), float(np.min(per_node)), per_node.astype(np.float32)


def _mean_reference_supported_gap(
    registry,
    current_acc,
    current_support,
    reference_acc,
    reference_support,
) -> float:
    reliable = (
        registry.reliableMask(np.asarray(current_support, dtype=np.int32))
        & (np.asarray(reference_support, dtype=np.int32) > 0)
        & np.isfinite(current_acc)
        & np.isfinite(reference_acc)
    )
    if not np.any(reliable):
        return 0.0
    target = np.minimum(
        registry.class_best_acc[reliable], np.asarray(reference_acc)[reliable]
    )
    return float(np.mean(np.maximum(0.0, target - np.asarray(current_acc)[reliable])))


def _select_snapshot(
    node: LogicalNode,
    worker: SharedDeviceWorker,
    batch,
    *,
    use_snapshots: bool,
    node_count: int,
    max_relative_distance: float | None,
) -> SelectionDecision:
    if not node.memory:
        snapshot = RingSnapshot(
            sender_id=node.node_id,
            round_id=-1,
            params=node.params,
            registry=node.registry.clone(),
        )
        return SelectionDecision(snapshot, [node.node_id], [0.0], [math.nan], [True], False)

    candidates = list(node.memory.values())
    if not use_snapshots:
        predecessor = (node.node_id - 1) % node_count
        snapshot = node.memory.get(predecessor, candidates[-1])
        distance = float(_relative_parameter_distance(node.params, snapshot.params).numpy())
        return SelectionDecision(snapshot, [snapshot.sender_id], [distance], [math.nan], [True], False)

    distances = tf.stack(
        [
            _relative_parameter_distance(node.params, snapshot.params)
            for snapshot in candidates
        ]
    ).numpy().astype(np.float64, copy=False).tolist()
    plausible = [
        max_relative_distance is None or distance <= float(max_relative_distance)
        for distance in distances
    ]
    fallback = not any(plausible)
    eligible_indices = (
        [index for index, value in enumerate(plausible) if value]
        if not fallback
        else [int(np.argmin(distances))]
    )
    losses = [math.nan] * len(candidates)
    eligible_losses = tf.stack(
        [
            worker.batch_loss_tensor(candidates[index].params, batch)
            for index in eligible_indices
        ]
    ).numpy()
    for index, loss in zip(eligible_indices, eligible_losses):
        losses[index] = float(loss)
    selected_index = min(eligible_indices, key=lambda index: losses[index])
    return SelectionDecision(
        snapshot=candidates[selected_index],
        senders=[snapshot.sender_id for snapshot in candidates],
        distances=distances,
        losses=losses,
        plausible=plausible,
        fallback=fallback,
    )


def _evaluate_states(worker, nodes, test_loader, max_batches=5):
    values = []
    for node in nodes:
        worker.load(node.params)
        values.append(evaluate(worker.model, test_loader, maxBatches=max_batches))
    values = np.asarray(values, dtype=np.float32)
    return float(np.mean(values)), float(np.min(values)), values


def _telemetry_arrays(rounds: int, nodes: int, memory: int):
    shape = (rounds, nodes)
    candidate_shape = (rounds, nodes, memory)
    return {
        "selected_sources": np.full(shape, -1, dtype=np.int32),
        "selected_distance": np.full(shape, np.nan, dtype=np.float32),
        "selected_noise_norm": np.zeros(shape, dtype=np.float32),
        "selected_relative_noise": np.zeros(shape, dtype=np.float32),
        "selected_attacked": np.zeros(shape, dtype=np.bool_),
        "selection_fallback": np.zeros(shape, dtype=np.bool_),
        "candidate_senders": np.full(candidate_shape, -1, dtype=np.int32),
        "candidate_distances": np.full(candidate_shape, np.nan, dtype=np.float32),
        "candidate_losses": np.full(candidate_shape, np.nan, dtype=np.float32),
        "candidate_plausible": np.zeros(candidate_shape, dtype=np.bool_),
        "model_norm": np.zeros(shape, dtype=np.float32),
        "consensus_innovation": np.zeros(shape, dtype=np.float32),
        "stress_ema": np.zeros(shape, dtype=np.float32),
        "ebm_coefficient": np.zeros(shape, dtype=np.float32),
        "ebm_requested_coefficient": np.zeros(shape, dtype=np.float32),
        "base_gradient_norm": np.zeros(shape, dtype=np.float32),
        "regularizer_gradient_norm": np.zeros(shape, dtype=np.float32),
        "active_ebm_ratio": np.zeros(shape, dtype=np.float32),
        "preclip_gradient_norm": np.zeros(shape, dtype=np.float32),
        "clip_fraction": np.zeros(shape, dtype=np.float32),
        "momentum_norm": np.zeros(shape, dtype=np.float32),
        "cart_gap": np.zeros(shape, dtype=np.float32),
        "cart_mu": np.zeros(shape, dtype=np.float32),
        "registry_accepted": np.zeros(shape, dtype=np.int32),
        "registry_rejected": np.zeros(shape, dtype=np.int32),
        "registry_coverage_node": np.zeros(shape, dtype=np.float32),
        "attack_active": np.zeros(shape, dtype=np.bool_),
        "attack_relative_norm": np.zeros(shape, dtype=np.float32),
        "outgoing_relative_noise": np.zeros(candidate_shape, dtype=np.float32),
        "outgoing_noise_norm": np.zeros(candidate_shape, dtype=np.float32),
        "selection_seconds": np.zeros(shape, dtype=np.float32),
        "cart_seconds": np.zeros(shape, dtype=np.float32),
        "training_seconds": np.zeros(shape, dtype=np.float32),
        "transmit_seconds": np.zeros(shape, dtype=np.float32),
        "state_signature": np.full(shape, b"", dtype="S16"),
    }


def _store_candidates(telemetry, round_id, node_id, decision, memory_size):
    count = min(memory_size, len(decision.senders))
    telemetry["candidate_senders"][round_id, node_id, :count] = decision.senders[:count]
    telemetry["candidate_distances"][round_id, node_id, :count] = decision.distances[:count]
    telemetry["candidate_losses"][round_id, node_id, :count] = decision.losses[:count]
    telemetry["candidate_plausible"][round_id, node_id, :count] = decision.plausible[:count]


def _node_event(
    *,
    config,
    round_id,
    node,
    decision,
    diagnostics,
    mu,
    gap,
    accepted,
    rejected,
    attack_active,
    attack_norm,
    outgoing,
    signature,
):
    candidates = []
    for sender, distance, loss, plausible in zip(
        decision.senders,
        decision.distances,
        decision.losses,
        decision.plausible,
    ):
        candidates.append(
            {
                "senderId": int(sender),
                "distance": float(distance),
                "loss": None if not math.isfinite(loss) else float(loss),
                "plausible": bool(plausible),
                "selected": int(sender) == int(decision.snapshot.sender_id),
            }
        )
    return {
        "round": int(round_id + 1),
        "nodeId": int(node.node_id),
        "environment": config["environment"],
        "attack": {
            "configured": node.node_id in {
                int(value) for value in str(config.get("attackerIds", "")).split(",") if value
            },
            "active": bool(attack_active),
            "relativeNorm": float(attack_norm),
        },
        "incoming": {
            "senderId": int(decision.snapshot.sender_id),
            "sourceRound": int(decision.snapshot.round_id),
            "noiseNorm": float(decision.snapshot.noise_norm),
            "relativeNoise": float(decision.snapshot.relative_noise),
            "attacked": bool(decision.snapshot.attacked),
            "stateSignature": str(decision.snapshot.state_signature),
        },
        "selection": {
            "fallback": bool(decision.fallback),
            "candidates": candidates,
        },
        "training": {
            "stateSignature": signature,
            "momentumMode": config["optimizerStateMode"],
            "momentumNorm": float(diagnostics["momentumNorm"]),
            "baseGradientNorm": float(diagnostics["baseGradientNorm"]),
            "preclipGradientNorm": float(diagnostics["preclipGradientNorm"]),
            "clipFraction": float(diagnostics["clipFraction"]),
        },
        "ebm": {
            "mode": config["ebmMode"],
            "stress": float(node.stress_ema),
            "coefficient": float(diagnostics["coefficient"]),
            "requestedCoefficient": float(diagnostics["requestedCoefficient"]),
            "regularizerGradientNorm": float(diagnostics["regularizerGradientNorm"]),
            "activeRatio": float(diagnostics["activeEbmRatio"]),
        },
        "cart": {
            "gap": float(gap),
            "mu": float(mu),
            "coverage": float(np.mean(node.registry.class_best_support > 0)),
            "accepted": int(accepted),
            "rejected": int(rejected),
        },
        "outgoing": outgoing,
    }


def run_campaign_four(
    *,
    config: dict,
    model_class,
    train_loaders,
    test_loader,
    data_metadata: dict,
    stop_callback: Callable[[], bool] | None = None,
    round_callback=None,
    node_callback=None,
):
    """Run one Campaign 4 Merged or CART configuration."""
    uses_ebm = (
        bool(config.get("useChannelNoise"))
        and str(config.get("noiseMitigation", "none")) == "ebm"
    )
    phase = str(config.get("phase", "confirmation"))
    if (
        uses_ebm
        and int(config.get("internalMicroBatchSize", 0))
        != int(config.get("batchSize", 512))
        and phase != "performance_benchmark"
    ):
        raise ValueError(
            "Official Campaign 4 EBM runs require a full configured batch for "
            "the exact gradient-norm objective."
        )
    seed = int(config["seed"])
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    np.random.seed(seed)

    model = model_class()
    worker = SharedDeviceWorker(
        model,
        lr0=float(config["learningRate"]),
        momentum=float(config.get("momentum", 0.0)),
        micro_batch_size=int(config.get("internalMicroBatchSize", 128)),
        jit_compile=bool(config.get("jitCompile", False)),
        adaptive_config=config,
        weight_decay=float(config.get("weightDecayCoefficient", 0.0)),
    )
    initial_params = worker.export()
    initial_optimizer = worker.zero_optimizer_state()
    initial_hash = params_hash(initial_params)
    node_count = int(config["nNodes"])
    memory_size = int(config.get("basilMemorySize", 5))
    probe_batches = data_metadata["probeBatches"]
    initial_coefficient = float(config.get("ebmInitialCoefficient", 0.0))
    nodes = []
    for node_id in range(node_count):
        probe_x, probe_y = probe_batches[node_id]
        nodes.append(
            LogicalNode(
                node_id=node_id,
                params=_device_copy(initial_params),
                optimizer_state=_device_copy(initial_optimizer),
                data_loader=train_loaders[node_id],
                probe_x=tf.convert_to_tensor(probe_x, dtype=tf.float32),
                probe_y=tf.convert_to_tensor(probe_y, dtype=tf.int32),
                probe_y_numpy=np.asarray(probe_y, dtype=np.int32),
                memory_size=memory_size,
                adaptive_coefficient=initial_coefficient,
            )
        )

    rounds = int(config["nRounds"])
    telemetry = _telemetry_arrays(rounds, node_count, memory_size)
    is_cart = config.get("approach") == "cart"
    use_snapshots = bool(config.get("snapshotSelection", False))
    use_noise = bool(config.get("useChannelNoise", False))
    sigma = float(config.get("channelNoiseSigma", 0.0)) if use_noise else 0.0
    use_ebm = config.get("noiseMitigation") == "ebm" and use_noise
    ebm_mode = str(config.get("ebmMode", "none")) if use_ebm else "none"
    gamma = float(config.get("distillStrength", 0.0)) if is_cart else 0.0
    cart_beta = float(config.get("cartEmaBeta", 0.85))
    stress_beta = float(config.get("adaptiveEbmBeta", 0.90))
    attackers = {
        int(value)
        for value in str(config.get("attackerIds", "")).split(",")
        if value.strip()
    }
    hidden_start = int(config.get("attackHiddenStart", 0))
    hidden_enabled = bool(config.get("attackHidden", False))
    full_consensus = config.get("aggregationMode") == "full_consensus"
    total_steps = int(config["localEpochs"]) * int(config.get("stepsPerEpoch", 5))
    scheduler = makeLrScheduler(
        float(config["learningRate"]),
        useBasilSchedule=True,
        useLrDecay=bool(config.get("useLrDecay", True)),
    )

    avg_history = []
    worst_history = []
    per_node_history = []
    mu_history = []
    accepted_history = []
    rejected_history = []
    coverage_history = []
    coefficient_history = []
    stage_history = []
    started = time.perf_counter()

    for round_id in range(rounds):
        if stop_callback is not None and stop_callback():
            break
        lr = scheduler(round_id)
        round_mus = []
        round_accepted = 0
        round_rejected = 0
        round_coefficients = []

        if full_consensus:
            trained_params = []
            registries = []
            for node in nodes:
                iterator = iter(node.data_loader)
                first_batch = next(iterator)
                optimizer_state = (
                    worker.zero_optimizer_state()
                    if config["optimizerStateMode"] == "visit_reset"
                    else node.optimizer_state
                )
                trained, node.optimizer_state, diagnostics = worker.train(
                    node.params,
                    optimizer_state=optimizer_state,
                    data_iterator=iterator,
                    first_batch=first_batch,
                    total_steps=total_steps,
                    lr=lr,
                    ebm_mode="none",
                    initial_coefficient=0.0,
                    stress=0.0,
                    prox_mu=0.0,
                )
                trained_params.append(trained)
                if is_cart:
                    node.class_acc, node.class_support = worker.probe_metrics(
                        trained, node.probe_x, node.probe_y, node.probe_y_numpy
                    )
                    node.registry.update(
                        node.node_id,
                        node.class_acc,
                        support=node.class_support,
                        roundId=round_id,
                    )
                    registries.append(node.registry.clone())
            consensus = _average_many(trained_params)
            merged_registry = ClassRegistry(10)
            for registry in registries:
                merged_registry.merge(registry)
            for node in nodes:
                node.params = _device_copy(consensus)
                if is_cart:
                    node.registry = merged_registry.clone()
            round_mus = [0.0] * node_count
        else:
            for node_id, node in enumerate(nodes):
                iterator = iter(node.data_loader)
                first_batch = next(iterator)

                selected_started = time.perf_counter()
                decision = _select_snapshot(
                    node,
                    worker,
                    first_batch,
                    use_snapshots=use_snapshots,
                    node_count=node_count,
                    max_relative_distance=(
                        float(config["snapshotMaxRelativeDistance"])
                        if use_snapshots
                        and config.get("snapshotPlausibilityGuard")
                        == "relative_l2_channel_budget"
                        else None
                    ),
                )
                telemetry["selection_seconds"][round_id, node_id] = (
                    time.perf_counter() - selected_started
                )
                selected = decision.snapshot
                selected_index = decision.senders.index(selected.sender_id)
                selected_distance = float(decision.distances[selected_index])
                consensus = _average_params(node.params, selected.params)
                consensus_innovation = 0.5 * selected_distance
                node.stress_ema = (
                    stress_beta * node.stress_ema
                    + (1.0 - stress_beta) * selected_distance
                )

                cart_started = time.perf_counter()
                mu = 0.0
                gap = 0.0
                accepted = 0
                rejected = 0
                if is_cart:
                    if not np.any(np.isfinite(node.class_acc)):
                        node.class_acc, node.class_support = worker.probe_metrics(
                            node.params,
                            node.probe_x,
                            node.probe_y,
                            node.probe_y_numpy,
                        )
                    selected_acc, selected_support = worker.probe_metrics(
                        selected.params,
                        node.probe_x,
                        node.probe_y,
                        node.probe_y_numpy,
                    )
                    if selected.registry is not None:
                        accepted, rejected = node.registry.verifyAndMerge(
                            selected.registry,
                            selected_acc,
                            float(config.get("verifyThreshold", 0.05)),
                            receivedSupport=selected_support,
                        )
                    reference_acc, reference_support = worker.probe_metrics(
                        consensus,
                        node.probe_x,
                        node.probe_y,
                        node.probe_y_numpy,
                    )
                    gap = _mean_reference_supported_gap(
                        node.registry,
                        node.class_acc,
                        node.class_support,
                        reference_acc,
                        reference_support,
                    )
                    node.ema_gap = cart_beta * node.ema_gap + (1.0 - cart_beta) * gap
                    mu = float(np.clip(gamma * node.ema_gap, 0.0, gamma))
                telemetry["cart_seconds"][round_id, node_id] = (
                    time.perf_counter() - cart_started
                )

                optimizer_state = (
                    worker.zero_optimizer_state()
                    if config["optimizerStateMode"] == "visit_reset"
                    else node.optimizer_state
                )
                training_started = time.perf_counter()
                node.params, node.optimizer_state, diagnostics = worker.train(
                    consensus,
                    optimizer_state=optimizer_state,
                    data_iterator=iterator,
                    first_batch=first_batch,
                    total_steps=total_steps,
                    lr=lr,
                    ebm_mode=ebm_mode,
                    initial_coefficient=(
                        node.adaptive_coefficient
                        if ebm_mode == "adaptive"
                        else initial_coefficient
                    ),
                    stress=node.stress_ema,
                    prox_mu=mu,
                )
                telemetry["training_seconds"][round_id, node_id] = (
                    time.perf_counter() - training_started
                )
                node.adaptive_coefficient = diagnostics["coefficient"]
                round_coefficients.append(diagnostics["coefficient"])

                if is_cart:
                    node.class_acc, node.class_support = worker.probe_metrics(
                        node.params,
                        node.probe_x,
                        node.probe_y,
                        node.probe_y_numpy,
                    )
                    node.registry.update(
                        node.node_id,
                        node.class_acc,
                        support=node.class_support,
                        roundId=round_id,
                    )

                signature, model_norm = _state_diagnostics(node.params)

                attack_active = bool(
                    hidden_enabled and round_id >= hidden_start and node_id in attackers
                )
                outbound = node.params
                attack_norm = 0.0
                if attack_active:
                    outbound, attack_norm = _hidden_attack(
                        outbound,
                        base_seed=seed,
                        round_id=round_id,
                        node_id=node_id,
                    )
                outbound_signature = (
                    _state_signature(outbound) if attack_active else signature
                )

                transmit_started = time.perf_counter()
                outgoing = []
                pending_transmissions = []
                for offset in range(1, memory_size + 1):
                    receiver_id = (node_id + offset) % node_count
                    transmitted, noise_norm, relative_noise = _relative_l2_channel_noise(
                        outbound,
                        sigma=sigma,
                        base_seed=seed,
                        seed_parts=("channel", round_id, node_id, receiver_id),
                    )
                    pending_transmissions.append(
                        (receiver_id, transmitted, noise_norm, relative_noise)
                    )
                link_diagnostics = tf.stack(
                    [
                        tf.stack([noise_norm, relative_noise])
                        for _, _, noise_norm, relative_noise in pending_transmissions
                    ]
                ).numpy()
                for offset, (
                    receiver_id,
                    transmitted,
                    _,
                    _,
                ) in enumerate(pending_transmissions, start=1):
                    noise_norm = float(link_diagnostics[offset - 1, 0])
                    relative_noise = float(link_diagnostics[offset - 1, 1])
                    nodes[receiver_id].receive(
                        RingSnapshot(
                            sender_id=node_id,
                            round_id=round_id,
                            params=transmitted,
                            registry=node.registry.clone() if is_cart else None,
                            noise_norm=noise_norm,
                            relative_noise=relative_noise,
                            attacked=attack_active,
                            state_signature=outbound_signature,
                        )
                    )
                    telemetry["outgoing_relative_noise"][round_id, node_id, offset - 1] = relative_noise
                    telemetry["outgoing_noise_norm"][round_id, node_id, offset - 1] = noise_norm
                    outgoing.append(
                        {
                            "receiverId": int(receiver_id),
                            "noiseNorm": float(noise_norm),
                            "relativeNoise": float(relative_noise),
                            "stateSignature": outbound_signature,
                        }
                    )
                telemetry["transmit_seconds"][round_id, node_id] = (
                    time.perf_counter() - transmit_started
                )

                telemetry["selected_sources"][round_id, node_id] = selected.sender_id
                telemetry["selected_distance"][round_id, node_id] = selected_distance
                telemetry["selected_noise_norm"][round_id, node_id] = selected.noise_norm
                telemetry["selected_relative_noise"][round_id, node_id] = selected.relative_noise
                telemetry["selected_attacked"][round_id, node_id] = selected.attacked
                telemetry["selection_fallback"][round_id, node_id] = decision.fallback
                telemetry["model_norm"][round_id, node_id] = model_norm
                telemetry["consensus_innovation"][round_id, node_id] = consensus_innovation
                telemetry["stress_ema"][round_id, node_id] = node.stress_ema
                telemetry["ebm_coefficient"][round_id, node_id] = diagnostics["coefficient"]
                telemetry["ebm_requested_coefficient"][round_id, node_id] = diagnostics[
                    "requestedCoefficient"
                ]
                telemetry["base_gradient_norm"][round_id, node_id] = diagnostics["baseGradientNorm"]
                telemetry["regularizer_gradient_norm"][round_id, node_id] = diagnostics["regularizerGradientNorm"]
                telemetry["active_ebm_ratio"][round_id, node_id] = diagnostics["activeEbmRatio"]
                telemetry["preclip_gradient_norm"][round_id, node_id] = diagnostics["preclipGradientNorm"]
                telemetry["clip_fraction"][round_id, node_id] = diagnostics["clipFraction"]
                telemetry["momentum_norm"][round_id, node_id] = diagnostics["momentumNorm"]
                telemetry["cart_gap"][round_id, node_id] = gap
                telemetry["cart_mu"][round_id, node_id] = mu
                telemetry["registry_accepted"][round_id, node_id] = accepted
                telemetry["registry_rejected"][round_id, node_id] = rejected
                telemetry["registry_coverage_node"][round_id, node_id] = float(
                    np.mean(node.registry.class_best_support > 0)
                )
                telemetry["attack_active"][round_id, node_id] = attack_active
                telemetry["attack_relative_norm"][round_id, node_id] = attack_norm
                telemetry["state_signature"][round_id, node_id] = signature.encode("ascii")
                _store_candidates(telemetry, round_id, node_id, decision, memory_size)

                round_mus.append(mu)
                round_accepted += accepted
                round_rejected += rejected
                if node_callback is not None:
                    node_callback(
                        _node_event(
                            config=config,
                            round_id=round_id,
                            node=node,
                            decision=decision,
                            diagnostics=diagnostics,
                            mu=mu,
                            gap=gap,
                            accepted=accepted,
                            rejected=rejected,
                            attack_active=attack_active,
                            attack_norm=attack_norm,
                            outgoing=outgoing,
                            signature=signature,
                        )
                    )

        evaluation_started = time.perf_counter()
        avg, worst, per_node = _evaluate_states(worker, nodes, test_loader, max_batches=5)
        evaluation_seconds = time.perf_counter() - evaluation_started
        avg_history.append(avg)
        worst_history.append(worst)
        per_node_history.append(per_node)
        mu_history.append(float(np.mean(round_mus)) if round_mus else 0.0)
        accepted_history.append(round_accepted)
        rejected_history.append(round_rejected)
        coverage_history.append(
            float(np.mean([np.mean(node.registry.class_best_support > 0) for node in nodes]))
            if is_cart
            else 0.0
        )
        coefficient_history.append(
            float(np.mean(round_coefficients)) if round_coefficients else 0.0
        )
        stage_history.append(
            [
                float(np.sum(telemetry["selection_seconds"][round_id])),
                float(np.sum(telemetry["cart_seconds"][round_id])),
                float(np.sum(telemetry["training_seconds"][round_id])),
                float(np.sum(telemetry["transmit_seconds"][round_id])),
                float(evaluation_seconds),
            ]
        )
        print(f"[campaign4 round {round_id}] eval avg={avg:.4f}", flush=True)
        if round_callback is not None:
            round_callback(round_id + 1, avg, worst, rounds, per_node)

    completed_rounds = len(avg_history)
    if completed_rounds < rounds:
        partial = {key: value[:completed_rounds] for key, value in telemetry.items()}
        return {
            "stopped": True,
            "metrics": {
                "avg_history": np.asarray(avg_history, dtype=np.float32),
                "worst_history": np.asarray(worst_history, dtype=np.float32),
            },
            "telemetry": partial,
            "initialization_hash": initial_hash,
        }

    confusion = np.stack([worker.confusion(node.params, test_loader) for node in nodes])
    final_avg, final_worst, final_nodes = _accuracy_from_confusion(confusion)
    final_class_accuracy = np.stack(
        [_class_accuracy_from_confusion(matrix) for matrix in confusion]
    )
    elapsed = time.perf_counter() - started
    metrics = {
        "avg_history": np.asarray(avg_history, dtype=np.float32),
        "worst_history": np.asarray(worst_history, dtype=np.float32),
        "per_node_history": np.asarray(per_node_history, dtype=np.float32),
        "final_avg": np.float32(final_avg),
        "final_worst": np.float32(final_worst),
        "final_node_accuracy": final_nodes.astype(np.float32),
        "final_class_accuracy": final_class_accuracy.astype(np.float32),
        "confusion": confusion.astype(np.int64),
        "class_counts": np.asarray(data_metadata["classCounts"], dtype=np.int32),
        "mu_history": np.asarray(mu_history, dtype=np.float32),
        "accepted_claims": np.asarray(accepted_history, dtype=np.int32),
        "rejected_claims": np.asarray(rejected_history, dtype=np.int32),
        "registry_coverage": np.asarray(coverage_history, dtype=np.float32),
        "selected_sources": telemetry["selected_sources"].astype(np.int32),
        "ebm_coefficient_history": np.asarray(coefficient_history, dtype=np.float32),
        "stage_runtime_seconds": np.asarray(stage_history, dtype=np.float32),
        "runtime_seconds": np.float64(elapsed),
        "peak_gpu_bytes": np.int64(_gpu_peak_bytes()),
        "learning_curve_auc": np.float32(
            np.trapz(np.asarray(avg_history, dtype=np.float32))
            / max(1, len(avg_history) - 1)
        ),
    }
    return {
        "stopped": False,
        "metrics": metrics,
        "telemetry": telemetry,
        "initialization_hash": initial_hash,
    }


__all__ = [
    "LogicalNode",
    "RingSnapshot",
    "SelectionDecision",
    "SharedDeviceWorker",
    "_accuracy_from_confusion",
    "_base_and_regularizer_gradients",
    "_adaptive_requested_coefficient",
    "_bounded_adaptive_coefficient",
    "_relative_l2_channel_noise",
    "_select_snapshot",
    "params_hash",
    "run_campaign_four",
]
