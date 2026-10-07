"""Opt-in, evaluation-only observations for bounded protocol audits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from basil_core.artifact_paths import writable_output

from basil_core.research_protocol import (
    evaluate_params, gradient_norm_objective_gradients, params_hash,
    tensor_noise_variance,
)


def array_hash(array):
    value = np.asarray(array)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def tensor_stats(value):
    array = np.asarray(value)
    finite = array[np.isfinite(array)].astype(np.float64)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
        "nan": int(np.isnan(array).sum()), "positiveInf": int(np.isposinf(array).sum()),
        "negativeInf": int(np.isneginf(array).sum()),
        "finiteMin": float(finite.min()) if finite.size else None,
        "finiteMax": float(finite.max()) if finite.size else None,
        "finiteL2": float(np.linalg.norm(finite))}


@tf.function(reduce_retracing=True)
def augment_batch(x, key):
    def augment(item):
        image, index = item
        sample_key = tf.random.experimental.stateless_fold_in(key, index)
        image = tf.image.stateless_random_flip_left_right(image, sample_key)
        image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
        return tf.image.stateless_random_crop(image, [32, 32, 3], sample_key + [0, 1])
    return tf.map_fn(augment, (x, tf.range(tf.shape(x)[0])), fn_output_signature=tf.float32)


class ProtocolAudit:
    def __init__(self, directory, *, candidate_evaluation=False):
        self.directory = writable_output(Path(directory))
        self.directory.mkdir(parents=True, exist_ok=False)
        self.trace = (self.directory / "batch_trace.jsonl").open("w", buffering=1)
        self.candidate_evaluation = candidate_evaluation
        self.confusions, self.candidates = [], []
        self.last_confusion = None
        self.context = {}
        self.pending = None
        self.last_training_batch = None

    def write(self, name, payload):
        (self.directory / name).write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    def initial(self, params, train_x, train_y, test_x, test_y):
        self.write("initial.json", {"parameterHash": params_hash(params),
            "parameters": [tensor_stats(p) for p in params],
            "trainImagesHash": array_hash(train_x), "trainLabelsHash": array_hash(train_y),
            "testImagesHash": array_hash(test_x), "testLabelsHash": array_hash(test_y),
            "testClassCounts": np.bincount(test_y, minlength=10).tolist()})

    def activation(self, round_id, node_id):
        self.context = {"round": round_id, "node": node_id}
        self.evaluation_index = 0

    def evaluation(self):
        self.confusions.append({**self.context, "stage": self.evaluation_index,
            "confusion": self.last_confusion.tolist()})
        self.evaluation_index += 1

    def selection(self, worker, node, candidates, selected, diagnostics, attackers,
                  config, test_x, test_y):
        if not self.candidate_evaluation or not config.get("snapshotSelection"):
            return
        # Selection has already finished. These global metrics cannot feed back.
        for snapshot, local in zip(candidates, diagnostics):
            accuracy, per_class = evaluate_params(worker, snapshot.params, test_x, test_y)
            self.candidates.append({**self.context, **local,
                "selected": snapshot is selected,
                "configuredByzantine": snapshot.sender_id in attackers,
                "attackActiveAtSource": bool(config.get("attackHidden")) and
                    snapshot.sender_id in attackers and snapshot.round_id >= config["attackStartRound"],
                "globalAccuracyEvaluationOnly": accuracy,
                "perClassEvaluationOnly": per_class.tolist()})
        worker.load(selected.params)

    def begin_batch(self, worker, x, y, key, indices, **context):
        self.pending = {"params": worker.export(), "x": x, "y": y, "key": key,
            "context": context}
        self.current = {**context, "parameterHashBefore": params_hash(self.pending["params"]),
            "parameterStatsBefore": [tensor_stats(p) for p in self.pending["params"]],
            "imagesHash": array_hash(x), "labelsHash": array_hash(y),
            "indicesHash": array_hash(indices), "augmentationKey": key.tolist(),
            "augmentedImagesHash": array_hash(augment_batch(tf.convert_to_tensor(x), key).numpy())}
        if "receivedNoiseStats" in self.context:
            self.current.update(self.context)

    def gradients(self, gradients, values):
        names = ("crossEntropy", "gradientNormSquared", "objective", "coefficient",
                 "baseGradientNorm", "robustGradientNorm")
        self.current.update(zip(names, map(float, values)))
        if len(values)>6:
            self.current.update(ebmCorrectionNorm=float(values[6]),trainingBatchAccuracy=float(values[7]),
                ebmPenalty=float(values[3])*float(values[1]))
        self.current["gradients"] = [tensor_stats(g.numpy()) for g in gradients]
        self.current["gradientHash"] = params_hash([g.numpy() for g in gradients])

    def end_batch(self, worker):
        params = worker.export()
        self.current.update(parameterHashAfter=params_hash(params),
                            parameterStatsAfter=[tensor_stats(p) for p in params])
        self.trace.write(json.dumps(self.current) + "\n")
        if any(not np.isfinite(p).all() for p in params):
            raise FloatingPointError("First non-finite parameter detected after SGD update")
        self.last_training_batch = self.pending
        self.pending = None

    def failure(self, worker, error):
        if self.pending is None:
            return
        pending = self.pending
        self.write("failure.json", {**self.current, "exception": str(error),
            "parameterStatsAtFailure": [tensor_stats(p) for p in worker.export()]})
        np.savez_compressed(self.directory / "failure_batch.npz", x=pending["x"],
            y=pending["y"], key=pending["key"],
            **{f"weight_{i}": p for i, p in enumerate(pending["params"])})
        # Retrace only the saved batch with op-level checks, never resume training.
        worker.load(pending["params"])
        context = pending["context"]
        if context["mode"] == "evaluation":
            self.dump_failed_graph(worker, pending)
            self.replay_values(worker, pending)
            return
        tf.debugging.enable_check_numerics(stack_height_limit=12, path_length_limit=100)
        try:
            checked = tf.function(worker._batch_gradients)
            checked(tf.convert_to_tensor(pending["x"]), tf.convert_to_tensor(pending["y"]),
                tf.convert_to_tensor(pending["key"]), context["mode"], context["semantics"],
                tf.constant(context["sigma"], tf.float32),
                tf.constant(context["legacy_lambda"], tf.float32))
            self.write("check_numerics_probe.json", {**context,
                "replay": "Op-checked replay did not reproduce the original failure"})
        except Exception as replay_error:
            self.write("check_numerics_probe.json", {**context, "replayException": str(replay_error),
                "scope": "First op-check failure in saved-batch graph replay; instrumentation may change fusion"})
        finally:
            tf.debugging.disable_check_numerics()
        self.dump_failed_graph(worker, pending)
        self.replay_values(worker, pending)

    def dump_failed_graph(self, worker, pending):
        worker.load(pending["params"])
        context = pending["context"]
        # CheckNumerics alone sees TF's intentional NaN constant for invalid
        # labels. Dump executed arithmetic instead; exclude literal constants.
        dump_path = self.directory / "failed_graph"
        writer = tf.debugging.experimental.enable_dump_debug_info(str(dump_path),
            tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1,
            op_regex=r"^(Conv2D.*|MatMul|Sum|Mean|Mul|AddN|AddV2|Sub|RealDiv|Square|L2Loss|Relu.*|BiasAdd.*|MaxPool.*|SparseSoftmaxCrossEntropyWithLogits)$")
        replay_error = None
        try:
            if context["mode"] == "evaluation":
                traced = tf.function(lambda images: worker.model(images, training=False))
                traced(tf.convert_to_tensor(pending["x"]))
            else:
                traced = tf.function(worker._batch_gradients)
                traced(tf.convert_to_tensor(pending["x"]), tf.convert_to_tensor(pending["y"]),
                    tf.convert_to_tensor(pending["key"]), context["mode"], context["semantics"],
                    tf.constant(context["sigma"], tf.float32),
                    tf.constant(context["legacy_lambda"], tf.float32))
        except Exception as error:
            replay_error = str(error)
        finally:
            tf.debugging.experimental.disable_dump_debug_info()
            writer.FlushNonExecutionFiles()
            writer.FlushExecutionFiles()
        from tensorflow.python.debug.lib.debug_events_reader import DebugDataReader
        reader = DebugDataReader(str(dump_path))
        reader.update()
        operations = []
        for trace in reader.graph_execution_traces():
            health = trace.debug_tensor_value
            if health is not None and len(health) == 11 and sum(health[5:8]) > 0:
                operations.append({"op": trace.op_name, "type": trace.op_type,
                    "outputSlot": trace.output_slot, "elementCount": health[4],
                    "negativeInf": health[5], "positiveInf": health[6], "nan": health[7]})
        self.write("graph_nonfinite_operations.json", {**context,
            "scope": "Executed arithmetic in instrumented saved-batch replay; literal sentinel constants excluded",
            "nonfiniteOperationsInExecutionOrder": operations, "exception": replay_error})
        if operations:
            self.write("first_nonfinite.json", {**context, "firstArithmeticOperation": operations[0],
                "scope": "Saved-batch graph replay; excluded TensorFlow's intentional NaN label sentinel"})
            self.capture_first_tensor(worker, pending, operations[0]["type"])

    def capture_first_tensor(self, worker, pending, op_type):
        from tensorflow.python.debug.lib.debug_events_reader import DebugDataReader
        context = pending["context"]
        worker.load(pending["params"])
        path = self.directory / "first_tensor_dump"
        writer = tf.debugging.experimental.enable_dump_debug_info(str(path),
            tensor_debug_mode="FULL_TENSOR", circular_buffer_size=-1, op_regex=f"^{op_type}$")
        try:
            if context["mode"] == "evaluation":
                tf.function(lambda x: worker.model(x, training=False))(tf.convert_to_tensor(pending["x"]))
            else:
                tf.function(worker._batch_gradients)(tf.convert_to_tensor(pending["x"]),
                    tf.convert_to_tensor(pending["y"]), tf.convert_to_tensor(pending["key"]),
                    context["mode"], context["semantics"], tf.constant(context["sigma"], tf.float32),
                    tf.constant(context["legacy_lambda"], tf.float32))
        except tf.errors.InvalidArgumentError:
            pass
        finally:
            tf.debugging.experimental.disable_dump_debug_info()
            writer.FlushNonExecutionFiles(); writer.FlushExecutionFiles()
        reader = DebugDataReader(str(path)); reader.update()
        for trace in reader.graph_execution_traces():
            value = reader.graph_execution_trace_to_tensor_value(trace)
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                np.save(self.directory / "first_nonfinite_tensor.npy", value, allow_pickle=False)
                self.write("first_nonfinite_tensor_statistics.json", {**context,
                    "op": trace.op_name, "type": trace.op_type, **tensor_stats(value),
                    "firstNonfiniteIndices": np.argwhere(~np.isfinite(value))[:10].tolist()})
                break

    def replay_values(self, worker, pending, name="failure_tensor_values.json"):
        worker.load(pending["params"])
        context = pending["context"]
        x = tf.convert_to_tensor(pending["x"])
        if context["mode"] != "evaluation":
            x = augment_batch(x, pending["key"])
        layers = []
        value = x
        for layer in worker.model.model.layers:
            value = layer(value)
            layers.append({"layer": layer.name, **tensor_stats(value.numpy())})
        if context["mode"] == "evaluation":
            self.write(name, {**context, "layers": layers,
                "scope": "Evaluation-only eager replay at saved parameters; no training or augmentation"})
            return
        coefficient = tensor_noise_variance(worker.model, context["semantics"], context["sigma"])
        from basil_core.trainer import lossFn
        with tf.GradientTape() as tape:
            logits = worker.model(x, training=True)
            base = lossFn(pending["y"], logits)
        base_grads = tape.gradient(base, worker.model.trainable_variables)
        coordinate_variance = coefficient
        if context["mode"] == "gradient_norm_objective":
            robust, loss, norm_sq, base_norm, robust_norm = gradient_norm_objective_gradients(
                worker.model, x, pending["y"], coefficient)
            objective = loss + coefficient * norm_sq
            corrections = [r - b for r, b in zip(robust, base_grads)]
        else:
            loss = base
            base_norm = tf.linalg.global_norm(base_grads)
            norm_sq = base_norm ** 2
            coefficient = context["legacy_lambda"] * context["sigma"] ** 2 if context["mode"] == "legacy_gradient_scale" else 0.
            robust = [g * (1 + coefficient) for g in base_grads]
            robust_norm = tf.linalg.global_norm(robust)
            objective, corrections = loss, []
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pending["y"], logits=logits)
        self.write(name, {**context,
            "scope": "Eager replay at identical saved parameters/data; not claimed bit-identical to fused graph",
            "layers": layers, "perExampleCrossEntropy": tensor_stats(ce.numpy()),
            "crossEntropy": float(loss), "coefficient": float(coefficient),
            "gradientNormSquared": float(norm_sq), "objective": float(objective),
            "channelCoordinateVariance": float(coordinate_variance),
            "baseGradientNorm": float(base_norm), "robustGradientNorm": float(robust_norm),
            "baseGradients": [tensor_stats(g.numpy()) for g in base_grads],
            "fullObjectiveGradients": [tensor_stats(g.numpy()) for g in robust],
            "secondOrderCorrections": [tensor_stats(g.numpy()) for g in corrections]})

    def evaluation_failure(self, worker, x, y, start, error):
        preceding = self.last_training_batch
        context = {"round_id": self.context["round"], "node_id": self.context["node"],
            "epoch": self.evaluation_index, "batch": start // 512,
            "mode": "evaluation", "stage": "before_training" if self.evaluation_index == 0 else "after_epoch",
            "semantics": self.current.get("semantics", "unknown") if hasattr(self, "current") else "unknown",
            "sigma": self.current.get("sigma", 0.) if hasattr(self, "current") else 0.,
            "legacy_lambda": 0., "learning_rate": self.current.get("learning_rate", 0.) if hasattr(self, "current") else 0.}
        self.pending = {"context": context, "params": worker.export(), "x": x, "y": y,
                        "key": np.zeros(2, np.int32)}
        self.current = {**context, "parameterHashBefore": params_hash(self.pending["params"])}
        self.failure(worker, error)
        if preceding and preceding["context"]["node_id"] == self.context["node"]:
            np.savez_compressed(self.directory / "preceding_optimizer_batch.npz",
                x=preceding["x"], y=preceding["y"], key=preceding["key"],
                **{f"weight_{i}": p for i, p in enumerate(preceding["params"])})
            self.replay_values(worker, preceding, name="preceding_optimizer_tensor_values.json")
        self.pending = None

    def record_activation(self, record):
        with (self.directory / "activation_trace.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def close(self):
        self.trace.close()
        self.write("confusions.json", self.confusions)
        self.write("candidate_evaluation.json", self.candidates)
