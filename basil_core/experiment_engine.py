"""Deterministic single-worker ring engine for campaign three.

The legacy BASIL/CART functions keep one Keras model per logical node.  The
campaign-three protocol is sequential, so only one model is active at a time.
This engine stores logical node state as NumPy arrays and reuses one compiled
GPU worker, reducing both graph compilation and peak GPU memory.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import time

import numpy as np
import tensorflow as tf

from .attacks import applyAttack
from .basil import MAX_TRAIN_MICRO_BATCH_SIZE, _MutableLR, _batchGradients
from .class_registry import ClassRegistry
from .trainer import (
    addChannelNoiseToParams,
    averageParams,
    evaluate,
    getParams,
    lossFn,
    makeLrScheduler,
    setParams,
)


def _copy_params(params):
    return [np.asarray(value, dtype=np.float32).copy() for value in params]


def _keyed_rng(base_seed: int, *parts) -> np.random.Generator:
    text = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def params_hash(params) -> str:
    digest = hashlib.sha256()
    for value in params:
        array = np.asarray(value, dtype=np.float32)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _relative_parameter_distance(left, right) -> float:
    difference_squared = 0.0
    reference_squared = 0.0
    for left_value, right_value in zip(left, right):
        left_array = np.asarray(left_value, dtype=np.float32)
        right_array = np.asarray(right_value, dtype=np.float32)
        difference = right_array - left_array
        difference_squared += float(np.sum(difference * difference))
        reference_squared += float(np.sum(left_array * left_array))
    return float(
        np.sqrt(difference_squared)
        / max(np.sqrt(reference_squared), 1e-12)
    )


def _gradient_norm_regularized_gradients(
    model_inner,
    weights,
    x,
    y,
    coefficient,
    micro_batch_size=MAX_TRAIN_MICRO_BATCH_SIZE,
):
    """Differentiate F + coefficient * ||grad F||^2 in bounded slices.

    The configured batch remains the effective optimizer batch. Slicing only
    limits activation memory; slice gradients are weighted by sample count.
    """
    batch_size = x.shape[0]
    if batch_size is None:
        slices = ((0, None, 1.0),)
    else:
        batch_size = int(batch_size)
        slices = tuple(
            (
                start,
                min(start + int(micro_batch_size), batch_size),
                float(min(start + int(micro_batch_size), batch_size) - start)
                / float(batch_size),
            )
            for start in range(0, batch_size, int(micro_batch_size))
        )

    gradient_sums = [tf.zeros_like(weight) for weight in weights]
    for start, end, factor in slices:
        batch_x = x[start:end]
        batch_y = y[start:end]
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                logits = model_inner(batch_x, training=True)
                loss = lossFn(batch_y, logits)
            loss_gradients = inner_tape.gradient(
                loss,
                weights,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            gradient_norm_squared = tf.add_n(
                [tf.reduce_sum(tf.square(gradient)) for gradient in loss_gradients]
            )
            robust_objective = loss + coefficient * gradient_norm_squared
        robust_gradients = outer_tape.gradient(
            robust_objective,
            weights,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        gradient_sums = [
            total + gradient * factor
            for total, gradient in zip(gradient_sums, robust_gradients)
        ]
    return gradient_sums


@dataclass
class RingSnapshot:
    sender_id: int
    round_id: int
    params: list[np.ndarray]
    registry: ClassRegistry | None = None


@dataclass
class LogicalNode:
    node_id: int
    params: list[np.ndarray]
    optimizer_state: list[np.ndarray]
    data_loader: object
    probe_x: np.ndarray
    probe_y: np.ndarray
    memory_size: int
    registry: ClassRegistry = field(default_factory=lambda: ClassRegistry(10))
    class_acc: np.ndarray = field(
        default_factory=lambda: np.full(10, np.nan, dtype=np.float32)
    )
    class_support: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.int32)
    )
    ema_gap: float = 0.0
    memory: OrderedDict = field(default_factory=OrderedDict)

    def receive(self, snapshot: RingSnapshot) -> None:
        self.memory[snapshot.sender_id] = snapshot
        self.memory.move_to_end(snapshot.sender_id)
        while len(self.memory) > self.memory_size:
            self.memory.popitem(last=False)


class SharedModelWorker:
    def __init__(self, model, *, lr0: float, momentum: float):
        self.model = model
        self._lr = _MutableLR(float(lr0))
        self._optimizer = tf.keras.optimizers.SGD(
            learning_rate=self._lr,
            momentum=float(momentum),
        )
        self._optimizer.build(self.model.trainable_weights)
        self._optimizer_slots = [
            variable
            for variable in self._optimizer.variables
            if variable is not self._optimizer.iterations
        ]
        self._ebm_coefficient = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._prox_mu = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._ref_params = [
            tf.Variable(weight.numpy(), trainable=False, dtype=tf.float32)
            for weight in self.model.trainable_weights
        ]
        model_inner = self.model.model
        optimizer = self._optimizer
        ebm_coefficient = self._ebm_coefficient
        prox_mu = self._prox_mu
        ref_params = self._ref_params

        def apply_step(x, y, gradients):
            weights = model_inner.trainable_weights
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            prox_grads = [
                prox_mu * (weight - reference)
                for weight, reference in zip(weights, ref_params)
            ]
            grads = [
                gradient + proximal
                for gradient, proximal in zip(clipped_gradients, prox_grads)
            ]
            optimizer.apply_gradients(zip(grads, weights))

        def standard_step(x, y):
            gradients = _batchGradients(model_inner, model_inner.trainable_weights, x, y)
            apply_step(x, y, gradients)

        def ebm_step(x, y):
            gradients = _gradient_norm_regularized_gradients(
                model_inner,
                model_inner.trainable_weights,
                x,
                y,
                ebm_coefficient,
            )
            apply_step(x, y, gradients)

        self._standard_step = tf.function(standard_step, reduce_retracing=True)
        self._ebm_step = tf.function(ebm_step, reduce_retracing=True)
        self._predict = tf.function(
            lambda x: model_inner(x, training=False),
            reduce_retracing=True,
        )

    def load(self, params) -> None:
        setParams(self.model, params)

    def export(self):
        return getParams(self.model)

    def zero_optimizer_state(self):
        return [
            np.zeros(variable.shape, dtype=np.float32)
            for variable in self._optimizer_slots
        ]

    def _load_optimizer_state(self, state) -> None:
        if len(state) != len(self._optimizer_slots):
            raise ValueError("Logical-node optimizer state does not match the worker.")
        for variable, value in zip(self._optimizer_slots, state):
            variable.assign(tf.convert_to_tensor(value, dtype=variable.dtype))

    def _export_optimizer_state(self):
        return [
            variable.numpy().astype(np.float32, copy=True)
            for variable in self._optimizer_slots
        ]

    def batch_loss(self, params, batch) -> float:
        self.load(params)
        x, y = batch
        logits = self._predict(tf.cast(x, tf.float32))
        return float(lossFn(tf.cast(y, tf.int32), logits).numpy())

    def train(
        self,
        params,
        *,
        optimizer_state,
        data_iterator,
        first_batch,
        total_steps: int,
        lr: float,
        ebm_coefficient: float,
        prox_mu: float,
    ):
        self.load(params)
        for reference, weight in zip(self._ref_params, self.model.trainable_weights):
            reference.assign(weight)
        self._lr.assign(lr)
        self._ebm_coefficient.assign(float(ebm_coefficient))
        self._prox_mu.assign(float(prox_mu))
        self._load_optimizer_state(optimizer_state)
        step = self._ebm_step if float(ebm_coefficient) > 0.0 else self._standard_step

        x, y = first_batch
        step(tf.cast(x, tf.float32), tf.cast(y, tf.int32))
        for _ in range(max(0, int(total_steps) - 1)):
            x, y = next(data_iterator)
            step(tf.cast(x, tf.float32), tf.cast(y, tf.int32))
        return self.export(), self._export_optimizer_state()

    def probe_metrics(self, params, x, y):
        support = np.bincount(y, minlength=10).astype(np.int32)
        accuracy = np.full(10, np.nan, dtype=np.float32)
        if len(y) == 0:
            return accuracy, support
        self.load(params)
        logits = self._predict(tf.convert_to_tensor(x, dtype=tf.float32))
        predictions = tf.argmax(logits, axis=1).numpy()
        for class_id in range(10):
            mask = y == class_id
            if np.any(mask):
                accuracy[class_id] = float(np.mean(predictions[mask] == class_id))
        return accuracy, support

    def confusion(self, params, data_loader):
        self.load(params)
        matrix = np.zeros((10, 10), dtype=np.int64)
        for x, y in data_loader:
            logits = self._predict(tf.cast(x, tf.float32))
            predictions = tf.argmax(logits, axis=1).numpy()
            labels = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
            np.add.at(matrix, (labels.astype(np.int64), predictions.astype(np.int64)), 1)
        return matrix


def _gpu_peak_bytes() -> int:
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return int(info.get("peak", 0))
    except Exception:
        return 0


def _class_accuracy_from_confusion(matrix):
    support = matrix.sum(axis=1)
    result = np.full(10, np.nan, dtype=np.float32)
    present = support > 0
    result[present] = matrix.diagonal()[present] / support[present]
    return result


def _accuracy_from_confusion(confusion):
    """Return average, worst, and per-node accuracy from confusion matrices."""
    confusion = np.asarray(confusion, dtype=np.int64)
    totals = confusion.sum(axis=(1, 2))
    correct = np.trace(confusion, axis1=1, axis2=2)
    per_node = np.divide(
        correct,
        totals,
        out=np.zeros_like(correct, dtype=np.float64),
        where=totals > 0,
    )
    return (
        float(np.mean(per_node)) if len(per_node) else 0.0,
        float(np.min(per_node)) if len(per_node) else 0.0,
        per_node.astype(np.float32),
    )


def _mean_reference_supported_gap(
    registry,
    current_acc,
    current_support,
    reference_acc,
    reference_support,
) -> float:
    """Return only registry gaps reproduced by the proximal reference model.

    Registry entries are historical maxima. A later snapshot can carry a valid
    old claim without still embodying that class performance. Anchoring local
    training to such a snapshot causes drift instead of preventing it. Cap each
    registry target by the accuracy reproduced by the exact consensus model
    used as the proximal reference.
    """
    current_acc = np.asarray(current_acc, dtype=np.float32)
    current_support = np.asarray(current_support, dtype=np.int32)
    reference_acc = np.asarray(reference_acc, dtype=np.float32)
    reference_support = np.asarray(reference_support, dtype=np.int32)
    reliable = (
        registry.reliableMask(current_support)
        & (reference_support > 0)
        & np.isfinite(current_acc)
        & np.isfinite(reference_acc)
    )
    if not np.any(reliable):
        return 0.0
    verified_target = np.minimum(
        registry.class_best_acc[reliable],
        reference_acc[reliable],
    )
    gaps = np.maximum(0.0, verified_target - current_acc[reliable])
    return float(np.mean(gaps))


def _selected_snapshot(
    node: LogicalNode,
    worker: SharedModelWorker,
    batch,
    *,
    use_snapshots: bool,
    node_count: int,
    max_relative_distance: float | None = None,
):
    if not node.memory:
        return RingSnapshot(
            sender_id=node.node_id,
            round_id=-1,
            params=node.params,
            registry=node.registry.clone(),
        )
    if use_snapshots:
        # BASIL Definition 1: select only among stored models received from
        # counterclockwise neighbors, scoring each received model itself. The
        # project-level plausibility guard removes transmissions outside the
        # declared channel-plus-drift budget before BASIL's loss ranking.
        candidates = list(node.memory.values())
        if max_relative_distance is not None:
            distances = [
                _relative_parameter_distance(node.params, snapshot.params)
                for snapshot in candidates
            ]
            plausible = [
                snapshot
                for snapshot, distance in zip(candidates, distances)
                if distance <= float(max_relative_distance)
            ]
            candidates = (
                plausible
                if plausible
                else [candidates[int(np.argmin(distances))]]
            )
        losses = [
            worker.batch_loss(snapshot.params, batch)
            for snapshot in candidates
        ]
        return candidates[int(np.argmin(losses))]

    predecessor = (node.node_id - 1) % node_count
    if predecessor in node.memory:
        return node.memory[predecessor]
    return next(reversed(node.memory.values()))


def _evaluate_states(worker, nodes, test_loader, max_batches=5):
    accuracies = []
    for node in nodes:
        worker.load(node.params)
        accuracies.append(evaluate(worker.model, test_loader, maxBatches=max_batches))
    return (
        float(np.mean(accuracies)),
        float(np.min(accuracies)),
        np.asarray(accuracies, dtype=np.float32),
    )


def run_campaign_three(
    *,
    config: dict,
    model_class,
    train_loaders,
    test_loader,
    data_metadata: dict,
    stop_callback=None,
    round_callback=None,
):
    """Run one campaign-three Merged or CART configuration."""
    seed = int(config["seed"])
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    np.random.seed(seed)

    model = model_class()
    worker = SharedModelWorker(
        model,
        lr0=float(config["learningRate"]),
        momentum=float(config.get("momentum", 0.0)),
    )
    initial_params = worker.export()
    initial_optimizer_state = worker.zero_optimizer_state()
    initial_hash = params_hash(initial_params)
    memory_size = int(config.get("basilMemorySize", 5))
    probe_batches = data_metadata["probeBatches"]
    nodes = [
        LogicalNode(
            node_id=node_id,
            params=_copy_params(initial_params),
            optimizer_state=_copy_params(initial_optimizer_state),
            data_loader=train_loaders[node_id],
            probe_x=probe_batches[node_id][0],
            probe_y=probe_batches[node_id][1],
            memory_size=memory_size,
        )
        for node_id in range(int(config["nNodes"]))
    ]

    is_cart = config.get("approach") == "cart"
    use_snapshots = bool(
        config.get("snapshotSelection", config.get("useBasil", False))
    )
    use_ebm = (
        config.get("noiseMitigation") == "ebm"
        and config.get("useChannelNoise", False)
    )
    sigma = (
        float(config.get("channelNoiseSigma", 0.0))
        if config.get("useChannelNoise", False)
        else 0.0
    )
    ebm_coefficient = (
        float(config.get("ebmLambda", 0.0)) * sigma * sigma
        if use_ebm
        else 0.0
    )
    gamma = float(config.get("distillStrength", 0.0)) if is_cart else 0.0
    ema_beta = float(config.get("cartEmaBeta", 0.85))
    attackers = {
        int(value.strip())
        for value in str(config.get("attackerIds", "")).split(",")
        if value.strip()
    }
    hidden_active = bool(config.get("attackHidden", False))
    hidden_start = int(config.get("attackHiddenStart", 0))
    clean_consensus = config.get("environment") == "clean"
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
    selected_sources = []
    started = time.perf_counter()

    for round_id in range(int(config["nRounds"])):
        if stop_callback is not None and stop_callback():
            break
        lr = scheduler(round_id)
        round_mus = []
        round_accepted = 0
        round_rejected = 0
        round_sources = np.full(len(nodes), -1, dtype=np.int32)

        if clean_consensus:
            trained_params = []
            round_registries = []
            for node in nodes:
                iterator = iter(node.data_loader)
                first_batch = next(iterator)
                mu = 0.0
                if is_cart:
                    node.class_acc, node.class_support = worker.probe_metrics(
                        node.params, node.probe_x, node.probe_y
                    )
                    # All nodes already start from the same full-consensus
                    # reference in the clean arm. There is no independently
                    # verified incoming advantage for CART to preserve.
                    node.ema_gap = ema_beta * node.ema_gap
                trained, node.optimizer_state = worker.train(
                    node.params,
                    optimizer_state=node.optimizer_state,
                    data_iterator=iterator,
                    first_batch=first_batch,
                    total_steps=total_steps,
                    lr=lr,
                    ebm_coefficient=0.0,
                    prox_mu=mu,
                )
                trained_params.append(trained)
                round_mus.append(mu)
                if is_cart:
                    class_acc, support = worker.probe_metrics(
                        trained, node.probe_x, node.probe_y
                    )
                    node.registry.update(
                        node.node_id,
                        class_acc,
                        support=support,
                        roundId=round_id,
                    )
                    round_registries.append(node.registry.clone())

            consensus = averageParams(trained_params)
            merged_registry = ClassRegistry(10)
            for registry in round_registries:
                merged_registry.merge(registry)
            for node in nodes:
                node.params = _copy_params(consensus)
                if is_cart:
                    node.registry = merged_registry.clone()
                    node.class_acc, node.class_support = worker.probe_metrics(
                        node.params, node.probe_x, node.probe_y
                    )
        else:
            for node_id, node in enumerate(nodes):
                iterator = iter(node.data_loader)
                first_batch = next(iterator)
                selected = _selected_snapshot(
                    node,
                    worker,
                    first_batch,
                    use_snapshots=use_snapshots,
                    node_count=len(nodes),
                    max_relative_distance=(
                        float(config["snapshotMaxRelativeDistance"])
                        if use_snapshots
                        and config.get("snapshotPlausibilityGuard")
                        == "relative_l2_channel_budget"
                        else None
                    ),
                )
                round_sources[node_id] = selected.sender_id
                # The predecessor snapshot is the communication input.  Mixing
                # it with the node's current state is the ordinary pairwise
                # consensus step that prevents non-IID one-node replacement.
                # SS only decides which predecessor snapshot is safe enough to
                # use; it does not replace this aggregation operation.
                consensus_params = averageParams([node.params, selected.params])
                mu = 0.0

                if is_cart:
                    node.class_acc, node.class_support = worker.probe_metrics(
                        node.params, node.probe_x, node.probe_y
                    )
                    selected_acc, selected_support = worker.probe_metrics(
                        selected.params, node.probe_x, node.probe_y
                    )
                    if selected.registry is not None:
                        accepted, rejected = node.registry.verifyAndMerge(
                            selected.registry,
                            selected_acc,
                            float(config.get("verifyThreshold", 0.05)),
                            receivedSupport=selected_support,
                        )
                        round_accepted += accepted
                        round_rejected += rejected
                    reference_acc, reference_support = worker.probe_metrics(
                        consensus_params, node.probe_x, node.probe_y
                    )
                    gap = _mean_reference_supported_gap(
                        node.registry,
                        node.class_acc,
                        node.class_support,
                        reference_acc,
                        reference_support,
                    )
                    node.ema_gap = ema_beta * node.ema_gap + (1.0 - ema_beta) * gap
                    mu = float(np.clip(gamma * node.ema_gap, 0.0, gamma))

                node.params, node.optimizer_state = worker.train(
                    consensus_params,
                    optimizer_state=node.optimizer_state,
                    data_iterator=iterator,
                    first_batch=first_batch,
                    total_steps=total_steps,
                    lr=lr,
                    ebm_coefficient=ebm_coefficient,
                    prox_mu=mu,
                )
                round_mus.append(mu)

                if is_cart:
                    node.class_acc, node.class_support = worker.probe_metrics(
                        node.params, node.probe_x, node.probe_y
                    )
                    node.registry.update(
                        node.node_id,
                        node.class_acc,
                        support=node.class_support,
                        roundId=round_id,
                    )

                outbound = node.params
                if (
                    hidden_active
                    and round_id >= hidden_start
                    and node_id in attackers
                ):
                    outbound = applyAttack(
                        outbound,
                        "hidden",
                        rng=_keyed_rng(seed, "hidden", round_id, node_id),
                    )

                for offset in range(1, memory_size + 1):
                    receiver_id = (node_id + offset) % len(nodes)
                    transmitted = addChannelNoiseToParams(
                        outbound,
                        sigma=sigma,
                        rng=_keyed_rng(
                            seed,
                            "channel",
                            round_id,
                            node_id,
                            receiver_id,
                        ),
                    )
                    nodes[receiver_id].receive(
                        RingSnapshot(
                            sender_id=node_id,
                            round_id=round_id,
                            params=transmitted,
                            registry=node.registry.clone() if is_cart else None,
                        )
                    )

        avg, worst, per_node = _evaluate_states(
            worker, nodes, test_loader, max_batches=5
        )
        avg_history.append(avg)
        worst_history.append(worst)
        per_node_history.append(per_node)
        mu_history.append(float(np.mean(round_mus)) if round_mus else 0.0)
        accepted_history.append(round_accepted)
        rejected_history.append(round_rejected)
        coverage_history.append(
            float(
                np.mean(
                    [
                        np.mean(node.registry.class_best_support > 0)
                        for node in nodes
                    ]
                )
            )
            if is_cart
            else 0.0
        )
        selected_sources.append(round_sources)
        print(f"[campaign3 round {round_id}] eval avg={avg:.4f}", flush=True)
        if round_callback is not None:
            round_callback(round_id + 1, avg, worst, int(config["nRounds"]))

    stopped = len(avg_history) < int(config["nRounds"])
    if stopped:
        return {
            "stopped": True,
            "avg_history": np.asarray(avg_history, dtype=np.float32),
            "worst_history": np.asarray(worst_history, dtype=np.float32),
        }

    confusion = np.stack(
        [worker.confusion(node.params, test_loader) for node in nodes]
    )
    final_avg, final_worst, final_nodes = _accuracy_from_confusion(confusion)
    final_class_accuracy = np.stack(
        [_class_accuracy_from_confusion(matrix) for matrix in confusion]
    )
    elapsed = time.perf_counter() - started

    return {
        "stopped": False,
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
        "selected_sources": np.asarray(selected_sources, dtype=np.int32),
        "runtime_seconds": np.float64(elapsed),
        "peak_gpu_bytes": np.int64(_gpu_peak_bytes()),
        "ebm_coefficient": np.float32(ebm_coefficient),
        "learning_curve_auc": np.float32(
            np.trapz(np.asarray(avg_history, dtype=np.float32))
            / max(1, len(avg_history) - 1)
        ),
        "initialization_hash": initial_hash,
    }
