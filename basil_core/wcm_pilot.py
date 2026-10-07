"""Isolated ring pilot for the paper's worst-case communication model.

This module intentionally is not imported by the Campaign 3 worker or GUI.
It lets WCM be evaluated without changing EBM, SS, CART, queue configs, or
``experiments/results3``.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from noise_comm.wcm import WcmState, wcmLocalUpdate

from .attacks import applyAttack
from .experiment_engine import (
    LogicalNode,
    RingSnapshot,
    _accuracy_from_confusion,
    _class_accuracy_from_confusion,
    _copy_params,
    _evaluate_states,
    _gpu_peak_bytes,
    _keyed_rng,
    _mean_reference_supported_gap,
    _selected_snapshot,
    params_hash,
)
from .class_registry import ClassRegistry
from .trainer import (
    addChannelNoiseToParams,
    averageParams,
    getParams,
    lossFn,
    makeLrScheduler,
    setParams,
)


class WcmPilotWorker:
    """One shared model without unused SGD/momentum GPU slots."""

    def __init__(self, model, *, lr0: float, momentum: float):
        del lr0, momentum
        self.model = model
        model_inner = self.model.model
        self._predict = tf.function(
            lambda x: model_inner(x, training=False),
            reduce_retracing=True,
        )

    def load(self, params) -> None:
        setParams(self.model, params)

    def export(self):
        return getParams(self.model)

    def zero_optimizer_state(self):
        return []

    def batch_loss(self, params, batch) -> float:
        self.load(params)
        x, y = batch
        logits = self._predict(tf.cast(x, tf.float32))
        return float(lossFn(tf.cast(y, tf.int32), logits).numpy())

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
                accuracy[class_id] = float(
                    np.mean(predictions[mask] == class_id)
                )
        return accuracy, support

    def confusion(self, params, data_loader):
        self.load(params)
        matrix = np.zeros((10, 10), dtype=np.int64)
        for x, y in data_loader:
            logits = self._predict(tf.cast(x, tf.float32))
            predictions = tf.argmax(logits, axis=1).numpy()
            labels = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
            np.add.at(
                matrix,
                (
                    labels.astype(np.int64),
                    predictions.astype(np.int64),
                ),
                1,
            )
        return matrix

    def train_wcm(
        self,
        params,
        *,
        optimizer_state,
        wcm_state,
        data_iterator,
        first_batch,
        total_steps,
        lr,
        uncertainty_radius,
        penalty,
        rho_exponent,
        gamma_exponent,
        boundary_samples,
        radius_multiplier,
        prox_mu,
        rng,
    ):
        self.load(params)
        batches = [first_batch]
        for _ in range(max(0, int(total_steps) - 1)):
            batches.append(next(data_iterator))
        new_state, diagnostics = wcmLocalUpdate(
            model=self.model,
            lossFn=lossFn,
            batches=batches,
            state=wcm_state,
            uncertainty_radius=(
                float(uncertainty_radius) * float(radius_multiplier)
            ),
            penalty=float(penalty),
            inner_lr=float(lr),
            rho_exponent=float(rho_exponent),
            gamma_exponent=float(gamma_exponent),
            boundary_samples=int(boundary_samples),
            relative_radius=True,
            clip_norm=5.0,
            extra_prox_mu=float(prox_mu),
            rng=rng,
        )
        # WCM follows the paper's conditional update rather than the Campaign 3
        # SGD optimizer. Keep the logical optimizer slots unchanged so the
        # state structure remains compatible with the shared worker.
        return (
            self.export(),
            _copy_params(optimizer_state),
            new_state,
            diagnostics,
        )


def _validate_pilot_config(config):
    if config.get("noiseMitigation") != "wcm":
        raise ValueError("The isolated WCM pilot requires noiseMitigation='wcm'.")
    if config.get("environment") not in ("noise", "hidden_noise"):
        raise ValueError("The WCM pilot supports noise or hidden_noise only.")
    if not config.get("useChannelNoise", False):
        raise ValueError("WCM requires an active channel-noise environment.")
    if float(config.get("channelNoiseSigma", 0.0)) <= 0.0:
        raise ValueError("WCM requires channelNoiseSigma > 0.")
    if (
        config.get("snapshotSelection", config.get("useBasil", False))
        and not config.get("attackHidden", False)
    ):
        raise ValueError("SS is valid only with the hidden Byzantine attack.")
    if float(config.get("wcmPenalty", 0.0)) <= 0.0:
        raise ValueError("wcmPenalty must be positive.")
    if int(config.get("wcmBoundarySamples", 0)) < 1:
        raise ValueError("wcmBoundarySamples must be at least one.")


def run_wcm_pilot(
    *,
    config,
    model_class,
    train_loaders,
    test_loader,
    data_metadata,
    stop_callback=None,
    round_callback=None,
):
    """Run one isolated Merged/CART WCM experiment.

    The ring, attack, SS, CART, and channel paths match Campaign 3. Only the
    local EBM/standard optimizer is replaced by the paper-WCM SCA update.
    """
    _validate_pilot_config(config)
    seed = int(config["seed"])
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    np.random.seed(seed)

    model = model_class()
    worker = WcmPilotWorker(
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
    wcm_states = [WcmState() for _ in nodes]

    is_cart = config.get("approach") == "cart"
    use_snapshots = bool(
        config.get("snapshotSelection", config.get("useBasil", False))
    )
    sigma = float(config["channelNoiseSigma"])
    cart_gamma = float(config.get("distillStrength", 0.0)) if is_cart else 0.0
    ema_beta = float(config.get("cartEmaBeta", 0.85))
    attackers = {
        int(value.strip())
        for value in str(config.get("attackerIds", "")).split(",")
        if value.strip()
    }
    hidden_active = bool(config.get("attackHidden", False))
    hidden_start = int(config.get("attackHiddenStart", 0))
    total_steps = int(config["localEpochs"]) * int(
        config.get("stepsPerEpoch", 5)
    )
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
    wcm_rho_history = []
    wcm_gamma_history = []
    wcm_boundary_history = []
    wcm_candidate_history = []
    wcm_update_history = []
    wcm_gradient_history = []
    started = time.perf_counter()

    for round_id in range(int(config["nRounds"])):
        if stop_callback is not None and stop_callback():
            break
        lr = scheduler(round_id)
        round_mus = []
        round_accepted = 0
        round_rejected = 0
        round_sources = np.full(len(nodes), -1, dtype=np.int32)
        round_diagnostics = []

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
            consensus_params = averageParams([node.params, selected.params])
            mu = 0.0

            if is_cart:
                node.class_acc, node.class_support = worker.probe_metrics(
                    node.params,
                    node.probe_x,
                    node.probe_y,
                )
                selected_acc, selected_support = worker.probe_metrics(
                    selected.params,
                    node.probe_x,
                    node.probe_y,
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
                    consensus_params,
                    node.probe_x,
                    node.probe_y,
                )
                gap = _mean_reference_supported_gap(
                    node.registry,
                    node.class_acc,
                    node.class_support,
                    reference_acc,
                    reference_support,
                )
                node.ema_gap = (
                    ema_beta * node.ema_gap
                    + (1.0 - ema_beta) * gap
                )
                mu = float(
                    np.clip(cart_gamma * node.ema_gap, 0.0, cart_gamma)
                )

            (
                node.params,
                node.optimizer_state,
                wcm_states[node_id],
                diagnostics,
            ) = worker.train_wcm(
                consensus_params,
                optimizer_state=node.optimizer_state,
                wcm_state=wcm_states[node_id],
                data_iterator=iterator,
                first_batch=first_batch,
                total_steps=total_steps,
                lr=lr,
                uncertainty_radius=sigma,
                penalty=float(config["wcmPenalty"]),
                rho_exponent=float(config["wcmRhoExponent"]),
                gamma_exponent=float(config["wcmGammaExponent"]),
                boundary_samples=int(config["wcmBoundarySamples"]),
                radius_multiplier=float(
                    config.get("wcmRadiusMultiplier", 1.0)
                ),
                prox_mu=mu,
                rng=_keyed_rng(seed, "wcm", round_id, node_id),
            )
            round_diagnostics.append(diagnostics)
            round_mus.append(mu)

            if is_cart:
                node.class_acc, node.class_support = worker.probe_metrics(
                    node.params,
                    node.probe_x,
                    node.probe_y,
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
                        registry=(
                            node.registry.clone()
                            if is_cart
                            else None
                        ),
                    )
                )

        avg, worst, per_node = _evaluate_states(
            worker,
            nodes,
            test_loader,
            max_batches=5,
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
        wcm_rho_history.append(
            float(np.mean([value["rho"] for value in round_diagnostics]))
        )
        wcm_gamma_history.append(
            float(np.mean([value["gamma"] for value in round_diagnostics]))
        )
        wcm_boundary_history.append(
            float(
                np.mean(
                    [value["boundary_norm"] for value in round_diagnostics]
                )
            )
        )
        wcm_candidate_history.append(
            float(
                np.mean(
                    [
                        value["candidate_distance"]
                        for value in round_diagnostics
                    ]
                )
            )
        )
        wcm_update_history.append(
            float(
                np.mean(
                    [value["update_distance"] for value in round_diagnostics]
                )
            )
        )
        wcm_gradient_history.append(
            float(
                np.mean(
                    [
                        value["surrogate_gradient_norm"]
                        for value in round_diagnostics
                    ]
                )
            )
        )
        print(f"[wcm pilot round {round_id}] eval avg={avg:.4f}", flush=True)
        if round_callback is not None:
            round_callback(
                round_id + 1,
                avg,
                worst,
                int(config["nRounds"]),
            )

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
        "class_counts": np.asarray(
            data_metadata["classCounts"],
            dtype=np.int32,
        ),
        "mu_history": np.asarray(mu_history, dtype=np.float32),
        "accepted_claims": np.asarray(accepted_history, dtype=np.int32),
        "rejected_claims": np.asarray(rejected_history, dtype=np.int32),
        "registry_coverage": np.asarray(
            coverage_history,
            dtype=np.float32,
        ),
        "selected_sources": np.asarray(
            selected_sources,
            dtype=np.int32,
        ),
        "wcm_rho_history": np.asarray(
            wcm_rho_history,
            dtype=np.float32,
        ),
        "wcm_gamma_history": np.asarray(
            wcm_gamma_history,
            dtype=np.float32,
        ),
        "wcm_boundary_norm_history": np.asarray(
            wcm_boundary_history,
            dtype=np.float32,
        ),
        "wcm_candidate_distance_history": np.asarray(
            wcm_candidate_history,
            dtype=np.float32,
        ),
        "wcm_update_distance_history": np.asarray(
            wcm_update_history,
            dtype=np.float32,
        ),
        "wcm_surrogate_gradient_norm_history": np.asarray(
            wcm_gradient_history,
            dtype=np.float32,
        ),
        "runtime_seconds": np.float64(elapsed),
        "peak_gpu_bytes": np.int64(_gpu_peak_bytes()),
        "learning_curve_auc": np.float32(
            np.trapz(np.asarray(avg_history, dtype=np.float32))
            / max(1, len(avg_history) - 1)
        ),
        "initialization_hash": initial_hash,
    }


__all__ = ["WcmPilotWorker", "run_wcm_pilot"]
