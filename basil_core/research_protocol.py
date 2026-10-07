"""Strict sequential BASIL/CIFAR-10 ring research protocol.

This module is intentionally separate from historical consensus and campaign
engines. Only model parameters travel; optimizer state is reset for every node
activation. One ring traversal is one round.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
from typing import Callable, Iterable

import numpy as np
import tensorflow as tf

from basil_core.attacks import applyAttack
from basil_core.models import BasilPaperCifarModel, CIFARModel
from basil_core.trainer import lossFn
from basil_core.protocol_compatibility import PROTOCOL, IID_PROTOCOL, normalize_config


CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def keyed_rng(seed: int, *parts: object) -> np.random.Generator:
    key = "|".join((str(int(seed)), *(str(part) for part in parts)))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def copy_params(params: Iterable[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(value, dtype=np.float32).copy() for value in params]


def params_hash(params: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in params:
        array = np.asarray(value, dtype=np.float32)
        digest.update(str(array.shape).encode("ascii")); digest.update(array.tobytes())
    return digest.hexdigest()


def parameter_stats(params: Iterable[np.ndarray]) -> tuple[int, float]:
    arrays = [np.asarray(value, dtype=np.float32) for value in params]
    # Accumulate norms in float64: finite float32 weights can overflow w*w.
    return sum(value.size for value in arrays), math.sqrt(sum(float(np.sum(np.square(value, dtype=np.float64))) for value in arrays))


def resolve_attackers(seed: int, count: int = 4, node_count: int = 10) -> tuple[int, ...]:
    if count < 0 or count > node_count: raise ValueError("attacker count must be within the node count")
    return tuple(sorted(keyed_rng(seed, "attacker_selection").choice(node_count, size=count, replace=False).tolist()))


def basil_round_learning_rate(initial: float, round_id: int) -> float:
    return float(initial) / (1.0 + 0.05 * int(round_id))


@dataclass
class Snapshot:
    sender_id: int
    round_id: int
    params: list[np.ndarray]
    sender_loss: float | None = None
    sender_accuracy: float | None = None
    channel_metadata: dict | None = None


@dataclass
class RollingMemory:
    receiver_id: int
    node_count: int = 10
    size: int = 5
    snapshots: OrderedDict[int, Snapshot] = field(default_factory=OrderedDict)

    @property
    def allowed_senders(self) -> tuple[int, ...]:
        return tuple((self.receiver_id - distance) % self.node_count for distance in range(self.size, 0, -1))

    def receive(self, snapshot: Snapshot) -> None:
        if snapshot.sender_id not in self.allowed_senders:
            raise ValueError(f"sender {snapshot.sender_id} is not counterclockwise of receiver {self.receiver_id}")
        self.snapshots[snapshot.sender_id] = snapshot
        # Sender identity, not insertion history, defines memory membership.
        self.snapshots = OrderedDict((sender, self.snapshots[sender]) for sender in self.allowed_senders if sender in self.snapshots)

    def candidates(self) -> list[Snapshot]:
        return list(self.snapshots.values())


def initialize_memories(initial_params, node_count=10, size=5) -> list[RollingMemory]:
    memories=[]
    for receiver in range(node_count):
        memory=RollingMemory(receiver,node_count,size)
        for sender in memory.allowed_senders: memory.receive(Snapshot(sender,-1,copy_params(initial_params)))
        memories.append(memory)
    return memories


def broadcast(memories: list[RollingMemory], snapshot: Snapshot, transform: Callable[[int], list[np.ndarray]] | None=None) -> None:
    count=len(memories)
    for distance in range(1,memories[0].size+1):
        receiver=(snapshot.sender_id+distance)%count
        params=transform(receiver) if transform else copy_params(snapshot.params)
        memories[receiver].receive(Snapshot(snapshot.sender_id,snapshot.round_id,params,snapshot.sender_loss,snapshot.sender_accuracy))


def nearest_counterclockwise_distance(receiver: int, sender: int, node_count: int) -> int:
    return (receiver-sender)%node_count


def select_snapshot(
    receiver_id: int,
    candidates: list[Snapshot],
    evaluator: Callable[[list[np.ndarray]], tuple[float,float]],
    *,
    node_count: int = 10,
    tie_tolerance: float = 1e-8,
) -> tuple[Snapshot,list[dict]]:
    if not candidates: raise ValueError("Snapshot Selection requires at least one candidate.")
    diagnostics=[]
    for candidate in candidates:
        local_loss,local_accuracy=evaluator(candidate.params)
        diagnostics.append({"senderId":candidate.sender_id,"snapshotRound":candidate.round_id,"receiverLocalLoss":float(local_loss),"receiverLocalAccuracy":float(local_accuracy),"senderReportedLoss":candidate.sender_loss,"senderReportedAccuracy":candidate.sender_accuracy})
    minimum=min(item["receiverLocalLoss"] for item in diagnostics)
    tied=[(candidate,item) for candidate,item in zip(candidates,diagnostics) if item["receiverLocalLoss"]<=minimum+tie_tolerance]
    selected,_=min(tied,key=lambda pair:nearest_counterclockwise_distance(receiver_id,pair[0].sender_id,node_count))
    return selected,diagnostics


def apply_channel_noise(params, *, semantics: str, sigma: float, rng: np.random.Generator):
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Channel sigma must be finite and nonnegative.")
    dimension,model_norm=parameter_stats(params)
    if semantics=="relative_l2_gaussian": coordinate_sigma=float(sigma)*model_norm/math.sqrt(dimension)
    elif semantics in {"paper_absolute_gaussian", "absolute_coordinate_gaussian"}: coordinate_sigma=float(sigma)
    else: raise ValueError(f"Unsupported channelNoiseSemantics: {semantics}")
    noises=[rng.normal(0.0,coordinate_sigma,size=np.asarray(value).shape).astype(np.float32) for value in params]
    noisy=[(np.asarray(value,dtype=np.float32)+noise).astype(np.float32) for value,noise in zip(params,noises)]
    noise_norm=parameter_stats(noises)[1]
    return noisy,{"channelNoiseSemantics":semantics,"configuredSigma":float(sigma),"effectiveCoordinateSigma":coordinate_sigma,"coordinateSigma":coordinate_sigma,"noiseL2Norm":noise_norm,"relativeNoiseL2":noise_norm/max(model_norm,1e-12),"modelL2Norm":model_norm,"parameterCount":dimension,"noiseHash":params_hash(noises)}


def ebm_coefficient(params, *, semantics: str, sigma: float) -> float:
    if semantics in {"paper_absolute_gaussian", "absolute_coordinate_gaussian"}: return float(sigma)**2
    if semantics=="relative_l2_gaussian":
        dimension,norm=parameter_stats(params); coordinate_sigma=float(sigma)*norm/math.sqrt(dimension); return coordinate_sigma**2
    raise ValueError(f"Unsupported channelNoiseSemantics: {semantics}")


def gradient_norm_objective_gradients(model, x, y, coefficient, *, details=False):
    """Differentiate F + c||grad F||²; c is a stopped noise variance.

    Unlike legacy scalar gradient scaling, this includes 2c H_F grad(F).
    """
    coefficient=tf.stop_gradient(tf.cast(coefficient,tf.float32))
    weights=model.trainable_variables
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            logits=model(x,training=True)
            if details:
                tf.debugging.assert_all_finite(logits,"Non-finite training logits")
            base_loss=lossFn(y,logits)
        base_grads=inner.gradient(base_loss,weights,unconnected_gradients=(tf.UnconnectedGradients.NONE if details else tf.UnconnectedGradients.ZERO))
        if details and any(g is None for g in base_grads):
            raise ValueError("Unexpected disconnected CE gradient in the IID objective.")
        norm_sq=tf.add_n([tf.reduce_sum(tf.square(gradient)) for gradient in base_grads])
        objective=base_loss+coefficient*norm_sq
    robust=outer.gradient(objective,weights,unconnected_gradients=(tf.UnconnectedGradients.NONE if details else tf.UnconnectedGradients.ZERO))
    if details and any(g is None for g in robust):
        raise ValueError("Unexpected disconnected robust gradient in the IID objective.")
    base_norm=tf.linalg.global_norm(base_grads); robust_norm=tf.linalg.global_norm(robust)
    if details:
        if any(g is None for g in base_grads + robust):
            raise ValueError("Unexpected disconnected gradient in the IID objective.")
        correction=[r-g for r,g in zip(robust,base_grads)]
        correction_norm=tf.sqrt(tf.add_n([tf.reduce_sum(tf.cast(g,tf.float64)**2) for g in correction]))
        accuracy=tf.reduce_mean(tf.cast(tf.argmax(logits,axis=1,output_type=tf.int32)==tf.cast(y,tf.int32),tf.float32))
        return robust,(base_loss,norm_sq,objective,coefficient,base_norm,robust_norm,correction_norm,accuracy)
    return robust,base_loss,norm_sq,base_norm,robust_norm


def build_model(architecture: str, seed: int = 2025):
    initialization_seed = int(keyed_rng(seed, "model_initialization").integers(0, 2**30))
    if architecture=="basil_paper_cnn": return BasilPaperCifarModel(seed=initialization_seed)
    tf.keras.utils.set_random_seed(initialization_seed)
    if architecture=="legacy_vgg_cnn": return CIFARModel()
    raise ValueError(f"Unsupported modelArchitecture: {architecture}")


def validate_config(config: dict) -> None:
    config = normalize_config(config)
    iid_study=config.get("experimentProtocol")==IID_PROTOCOL
    required={"schemaVersion":5,"experimentProtocol":PROTOCOL,"dataset":"cifar10","nNodes":10,"localEpochs":5,"batchSize":512,"aggregationMode":"strict_sequential_handoff","epochSemantics":"full_local_dataset","optimizer":"sgd","optimizerStateMode":"reset_each_activation","learningRateSchedule":"basil_round_decay","learningRate":0.05,"momentum":0.0,"weightDecayCoefficient":0.0,"adaptiveWeightDecayMode":"off","gradientClipNorm":None,"basilMemorySize":5,"assumedByzantineCount":4,"attackerCount":4,"snapshotPlausibilityGuard":"none"}
    if iid_study:
        required["experimentProtocol"]=IID_PROTOCOL
        count=config.get("attackerCount",4)
        if count not in {0,4}:raise ValueError("IID actual attackerCount must be 0 or 4; memory always remains S=5.")
        required["attackerCount"]=count
        required.update(partitionStrategy="iid",modelArchitecture="basil_paper_cnn",snapshotSelection=True,
                        attackHidden=bool(count),channelNoiseSemantics="paper_absolute_gaussian",
                        attackerSelection="seeded_random_without_replacement",
                        attackerSelectionSeed=config.get("seed"),augmentation="seeded_flip_pad_crop")
        objective=config.get("localObjective")
        expected_mode={"cross_entropy":"none","source_ebm":"gradient_norm_objective"}.get(objective)
        if expected_mode is None or config.get("ebmMode")!=expected_mode:
            raise ValueError("IID tests require cross_entropy/none or source_ebm/gradient_norm_objective.")
        if objective=="source_ebm" and not config.get("useChannelNoise"):
            raise ValueError("Source EBM requires the explicitly configured Gaussian channel.")
        if any(config.get(k,0) for k in ("snapshotAnchorMu","snapshotAnchorMuMax","monteCarloNoiseSamples")):
            raise ValueError("No anchor or Monte Carlo objective belongs in the IID pair.")
        if config.get("channelNoiseSigmaAbsolute") is None:
            raise ValueError("Test 2 requires an explicitly approved absolute sigma_e; it is unresolved.")
    for key,expected in required.items():
        if config.get(key)!=expected: raise ValueError(f"{key} must be {expected!r} for {required['experimentProtocol']}.")
    if config.get("partitionStrategy") not in {"iid","dirichlet","one_class_per_node"}:raise ValueError("Unsupported partitionStrategy")
    if config.get("ringOrder","fixed_0_to_9")!="fixed_0_to_9":raise ValueError("Research ring order is fixed 0 to 9.")
    if config.get("ebmMode") not in {"none","gradient_norm_objective","legacy_gradient_scale"}: raise ValueError("Unsupported ebmMode")
    if config.get("snapshotSelection") and config.get("snapshotSelectionRule")!="lowest_receiver_local_loss": raise ValueError("Research Snapshot Selection must use receiver-local loss")
    if config.get("channelNoiseSemantics") not in {"relative_l2_gaussian", "paper_absolute_gaussian", "absolute_coordinate_gaussian"}:
        raise ValueError("Unsupported channelNoiseSemantics")
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise ValueError("An explicit integer seed is required.")
    if int(config.get("nRounds", 0)) < 1:
        raise ValueError("nRounds must be positive.")
    if config.get("resolvedAttackerIds") != list(resolve_attackers(config["seed"],int(config.get("attackerCount",4)))):
        raise ValueError("resolvedAttackerIds must match keyed seeded attacker selection.")
    if config.get("researchValid", True):
        horizons={50,100} if iid_study else {100}
        if config.get("nRounds") not in horizons or config.get("attackStartRound") != 20:
            raise ValueError(f"Production requires rounds in {sorted(horizons)} and attack start round 20.")
        if any(key.startswith("diagnostic") for key in config):
            raise ValueError("Production cannot use diagnostic data subsets.")
    elif config.get("attackStartRound", 20) != 20 and config.get("purpose") != "implementation_smoke_test":
        raise ValueError("Only implementation smoke tests may shorten attack warm-up.")
    if config.get("attackStartRound") not in {1,20}:raise ValueError("Attack starts at round 20, or round 1 for smoke only.")
    if config.get("channelNoiseStartRound", config.get("channelNoiseStart", 0)) != 0:
        raise ValueError("Research channel noise starts at round 0.")
    if config.get("attackType")!="delayed_hidden_parameter_attack_v1":raise ValueError("Unsupported research attack transformation.")
    if not iid_study and config.get("ebmMode")=="gradient_norm_objective" and any(k in config for k in ("legacyEbmLambda",)):
        raise ValueError("Full gradient-norm EBM cannot use legacy lambda.")
    sigma = configured_sigma(config)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Channel sigma must be finite and nonnegative.")
    if config.get("useChannelNoise") and sigma==0:
        raise ValueError("Enabled channel noise requires a positive explicitly labeled sigma.")
    if config.get("iidCampaignId"):
        from basil_core.iid_campaign import validate_campaign_condition
        validate_campaign_condition(config)


def configured_sigma(config: dict) -> float:
    if config["channelNoiseSemantics"] == "relative_l2_gaussian":
        return float(config.get("channelNoiseSigmaRelative", config.get("channelNoiseSigmaRel", 0.0)))
    return float(config.get("channelNoiseSigmaAbsolute", 0.0))


def tensor_noise_variance(model, semantics: str, sigma):
    sigma = tf.cast(sigma, tf.float32)
    if semantics == "relative_l2_gaussian":
        weights = model.trainable_variables
        dimension = sum(int(np.prod(variable.shape)) for variable in weights)
        squared_norm = tf.add_n([tf.reduce_sum(tf.cast(v, tf.float64) ** 2) for v in weights])
        variance = tf.cast(tf.cast(sigma, tf.float64) ** 2 * squared_norm / dimension, tf.float32)
    else:
        variance = sigma ** 2
    return tf.stop_gradient(variance)


class SequentialWorker:
    def __init__(self, model):
        self.model = model
        self.optimizer = None
        self.audit = None
        self.predict = tf.function(lambda x: self.model(x, training=False), reduce_retracing=True)
        self._gradients = tf.function(self._batch_gradients, reduce_retracing=True)

    def load(self, params):
        if len(params) != len(self.model.trainable_variables):
            raise ValueError("Checkpoint parameter count does not match the model.")
        for variable, value in zip(self.model.trainable_variables, params):
            variable.assign(value)

    def export(self):
        return [variable.numpy().astype(np.float32) for variable in self.model.trainable_variables]

    def probe(self, params, x, y):
        self.load(params)
        logits = self.predict(tf.convert_to_tensor(x))
        loss = float(lossFn(y, logits).numpy())
        if not np.isfinite(loss):
            raise FloatingPointError("Snapshot receiver-local loss is non-finite.")
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        return loss, float(tf.reduce_mean(tf.cast(prediction == tf.cast(y, tf.int32), tf.float32)))

    def _batch_gradients(self, x, y, augmentation_key, mode, semantics, sigma, legacy_lambda):
        def augment(item):
            image, index = item
            key = tf.random.experimental.stateless_fold_in(augmentation_key, index)
            image = tf.image.stateless_random_flip_left_right(image, key)
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
            return tf.image.stateless_random_crop(image, [32, 32, 3], key + [0, 1])

        x = tf.map_fn(augment, (x, tf.range(tf.shape(x)[0])), fn_output_signature=tf.float32)
        coefficient = tensor_noise_variance(self.model, semantics, sigma)
        if mode == "gradient_norm_objective":
            grads, loss, norm_sq, base_norm, robust_norm = gradient_norm_objective_gradients(self.model, x, y, coefficient)
            objective = loss + coefficient * norm_sq
        else:
            with tf.GradientTape() as tape:
                loss = lossFn(y, self.model(x, training=True))
            grads = tape.gradient(loss, self.model.trainable_variables)
            base_norm = tf.linalg.global_norm(grads)
            norm_sq = base_norm ** 2
            coefficient = legacy_lambda * sigma ** 2 if mode == "legacy_gradient_scale" else tf.constant(0.0)
            if mode == "legacy_gradient_scale":
                grads = [g * (1.0 + coefficient) for g in grads]
            robust_norm = tf.linalg.global_norm(grads)
            # Scaling is not the gradient-norm objective; do not report it as one.
            objective = loss
        tf.debugging.assert_all_finite(loss, "Non-finite cross-entropy")
        tf.debugging.assert_all_finite(objective, "Non-finite robust objective")
        for gradient in grads:
            tf.debugging.assert_all_finite(gradient, "Non-finite training gradient")
        return grads, (loss, norm_sq, objective, coefficient, base_norm, robust_norm)

    def train(self, params, x, y, *, seed, round_id, node_id, epochs, batch_size,
              learning_rate, ebm_mode, semantics="relative_l2_gaussian", sigma=0.0,
              legacy_lambda=25.0, evaluation=None, should_stop=None):
        if epochs != 5:
            raise ValueError("Research activations require five full local epochs.")
        self.load(params)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0)
        before = evaluation() if evaluation else None
        epoch_records, step_records = [], []
        for epoch in range(epochs):
            order = keyed_rng(seed, "data_order", round_id, node_id, epoch).permutation(len(y))
            weighted_loss, seen = 0.0, 0
            batch_sizes = []
            for visit, start in enumerate(range(0, len(order), batch_size)):
                if should_stop and should_stop():
                    raise InterruptedError("Graceful stop requested between optimizer steps.")
                indices = order[start:start + batch_size]
                key = keyed_rng(seed, "augmentation", round_id, node_id, epoch, visit).integers(0, 2**30, size=2, dtype=np.int32)
                batch_x, batch_y = x[indices], y[indices]
                if self.audit:
                    self.audit.begin_batch(self, batch_x, batch_y, key, indices,
                        round_id=round_id, node_id=node_id, epoch=epoch + 1, batch=visit,
                        mode=ebm_mode, semantics=semantics, sigma=sigma,
                        legacy_lambda=legacy_lambda, learning_rate=learning_rate)
                try:
                    grads, values = self._gradients(
                        tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y, tf.int32),
                        tf.convert_to_tensor(key), ebm_mode, semantics,
                        tf.constant(sigma, tf.float32), tf.constant(legacy_lambda, tf.float32))
                    if self.audit:
                        self.audit.gradients(grads, values)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    if self.audit:
                        self.audit.end_batch(self)
                except Exception as error:
                    if self.audit:
                        self.audit.failure(self, error)
                    raise
                loss, norm_sq, objective, coefficient, base_norm, robust_norm = map(float, values[:6])
                step_records.append({"epoch": epoch + 1, "batch": visit, "samples": len(indices),
                    "baseCrossEntropy": loss, "gradientNormSquared": norm_sq,
                    "ebmObjective": objective, "ebmCoefficient": coefficient,
                    "effectiveCoordinateSigma": math.sqrt(float(tensor_noise_variance(self.model, semantics, sigma))),
                    "baseGradientNorm": base_norm, "robustObjectiveGradientNorm": robust_norm})
                step_records[-1].update(zip(getattr(self,"extra_metric_names",()),map(float,values[6:])))
                if getattr(self,"extra_metric_names",()):
                    step_records[-1].update(parameterNorm=parameter_stats(self.export())[1],
                        ebmPenalty=coefficient*norm_sq if ebm_mode=="gradient_norm_objective" else 0.,
                        ebmCorrectionRatio=step_records[-1].get("ebmCorrectionNorm",0.)/(base_norm+1e-12),
                        configuredSigma=float(sigma), sigmaSquared=float(sigma)**2)
                # Effective sigma for EBM must describe pre-update weights.
                if ebm_mode == "gradient_norm_objective":
                    step_records[-1]["effectiveCoordinateSigma"] = math.sqrt(coefficient)
                weighted_loss += loss * len(indices)
                seen += len(indices)
                batch_sizes.append(len(indices))
            epoch_records.append({"epoch": epoch + 1, "samplesSeen": seen, "batchSizes": batch_sizes,
                "localTrainingLoss": weighted_loss / seen,
                **(evaluation() if evaluation else {})})
        final = self.export()
        _, accuracy = self.probe(final, x[:512], y[:512])
        return final, {"loss": float(np.mean([e["localTrainingLoss"] for e in epoch_records])),
            "accuracy": accuracy, "steps": len(step_records), "beforeTraining": before,
            "epochs": epoch_records, "optimizerTelemetry": step_records,
            "baseGradientNorm": float(np.mean([s["baseGradientNorm"] for s in step_records])),
            "robustGradientNorm": float(np.mean([s["robustObjectiveGradientNorm"] for s in step_records]))}


def evaluate_params(worker: SequentialWorker,params,test_x,test_y,batch_size=512,*,details=False):
    worker.load(params); correct=np.zeros(10,np.int64);total=np.zeros(10,np.int64)
    weighted_ce=0.
    confusion = np.zeros((10, 10), np.int64)
    for start in range(0,len(test_y),batch_size):
        y=np.asarray(test_y[start:start+batch_size],np.int32); logits=worker.predict(tf.convert_to_tensor(test_x[start:start+batch_size]))
        try:
            tf.debugging.assert_all_finite(logits, "Non-finite evaluation logits")
            if details:
                ce=lossFn(y,logits)
                tf.debugging.assert_all_finite(ce,"Non-finite evaluation CE")
                weighted_ce+=float(ce)*len(y)
        except tf.errors.InvalidArgumentError as error:
            if getattr(worker, "audit", None):
                worker.audit.evaluation_failure(worker, test_x[start:start+batch_size], y, start, error)
            raise
        pred=tf.argmax(logits,axis=1).numpy()
        np.add.at(confusion, (y, pred), 1)
        for class_id in range(10): mask=y==class_id; total[class_id]+=int(mask.sum()); correct[class_id]+=int((pred[mask]==y[mask]).sum())
    if getattr(worker, "audit", None):
        worker.audit.last_confusion = confusion
    per_class=np.divide(correct,total,out=np.zeros(10,np.float32),where=total>0)
    values=(float(correct.sum()/max(total.sum(),1)),per_class)
    return (*values,weighted_ce/max(len(test_y),1)) if details else values


def iid_partition_indices(labels, seed, node_count=10):
    from basil_core.data.cifar import _loadOrCreatePartition
    return _loadOrCreatePartition(labels,iid=True,nClients=node_count,alpha=.2,seed=seed,
                                  rng=keyed_rng(seed,"iid_partition"),cacheDir=None)


def run_research_protocol(config: dict, train_x, train_y, test_x, test_y, *, activation_callback=None, round_callback=None, should_stop=None, audit=None, worker_factory=None, resume_state=None, checkpoint_callback=None):
    """Run the strict ring and return metrics/telemetry without writing files."""
    config = normalize_config(config)
    validate_config(config); seed=int(config["seed"]); n=10; size=5; model=build_model(config["modelArchitecture"], seed); worker=(worker_factory or SequentialWorker)(model)
    initial=worker.export(); node_params=[copy_params(initial) for _ in range(n)]; memories=initialize_memories(initial,n,size); attackers=tuple(config.get("resolvedAttackerIds") or resolve_attackers(seed,int(config.get("attackerCount",4)),n))
    if audit:
        worker.audit = audit
        audit.initial(initial, train_x, train_y, test_x, test_y)
    rounds=int(config["nRounds"]); test_count=len(test_y); activation_accuracy=np.zeros((rounds,n),np.float32); activation_per_class=np.zeros((rounds,n,10),np.float32); round_node_accuracy=np.zeros((rounds,n),np.float32); round_node_per_class=np.zeros((rounds,n,10),np.float32); selected_sources=np.full((rounds,n),-1,np.int16); selected_rounds=np.full((rounds,n),-1,np.int16); telemetry=[]
    detailed_rounds=config.get("experimentProtocol")==IID_PROTOCOL and config.get("roundEvaluationMetrics",False)
    round_node_loss=np.zeros((rounds,n),np.float64)
    next_activation=0; completed_rounds=0
    if resume_state is not None:
        node_params=resume_state['node_params']; memories=resume_state['memories']
        telemetry=list(resume_state['telemetry'])
        next_activation=resume_state['next_activation']; completed_rounds=resume_state['completed_rounds']
        activation_accuracy=resume_state['activation_accuracy']; activation_per_class=resume_state['activation_per_class']
        round_node_accuracy=resume_state['round_node_accuracy']; round_node_per_class=resume_state['round_node_per_class']
        selected_sources=resume_state['selected_sources']; selected_rounds=resume_state['selected_rounds']
        round_node_loss=resume_state['round_node_loss']
    def checkpoint():
        if checkpoint_callback:
            checkpoint_callback(dict(next_activation=next_activation, completed_rounds=completed_rounds,
                node_params=node_params, memories=memories, telemetry=telemetry,
                activation_accuracy=activation_accuracy, activation_per_class=activation_per_class,
                round_node_accuracy=round_node_accuracy, round_node_per_class=round_node_per_class,
                selected_sources=selected_sources, selected_rounds=selected_rounds,
                round_node_loss=round_node_loss))
    predecessor=Snapshot(n-1,-1,copy_params(initial)); partition=config["partitionStrategy"]
    labels=np.asarray(train_y,np.int32)
    if not config.get("diagnosticTestSamplesPerClass"):
        if len(test_y)!=10000 or not np.array_equal(np.bincount(test_y,minlength=10),np.full(10,1000)):
            raise ValueError("Full-data evaluation requires the complete balanced 10,000-image CIFAR test set.")
    eligible = np.arange(len(labels))
    if config.get("diagnosticSamplesPerClass"):
        amount = int(config["diagnosticSamplesPerClass"])
        eligible = np.concatenate([keyed_rng(seed, "diagnostic_class", c).choice(np.flatnonzero(labels == c), size=amount, replace=False) for c in range(10)])
    if partition=="one_class_per_node": indices=[eligible[labels[eligible]==node] for node in range(n)]
    elif partition=="iid": indices=[eligible[chunk] for chunk in iid_partition_indices(labels[eligible],seed,n)]
    elif partition=="dirichlet":
        from basil_core.data.cifar import _dirichletPartition
        indices=[eligible[chunk] for chunk in _dirichletPartition(labels[eligible],n,float(config.get("dirichletAlpha",0.2)),keyed_rng(seed,"dirichlet_partition"))]
    else: raise ValueError(f"Unsupported partitionStrategy: {partition}")
    limit=config.get("diagnosticSamplesPerNode")
    if limit:
        indices=[keyed_rng(seed,"diagnostic_subset",node_id).choice(chunk,size=min(int(limit),len(chunk)),replace=False) for node_id,chunk in enumerate(indices)]
    probe_indices=[chunk[:min(512,len(chunk))] for chunk in indices]
    if any(len(chunk) == 0 for chunk in indices):
        raise ValueError("Partition produced an empty local training dataset.")
    if partition == "one_class_per_node" and not limit and not config.get("diagnosticSamplesPerClass"):
        if [len(chunk) for chunk in indices] != [5000] * 10:
            raise ValueError("Full-data one-class protocol requires exactly 5,000 samples per class.")
    sigma = configured_sigma(config) if config.get("useChannelNoise") else 0.0
    semantics = config["channelNoiseSemantics"]
    def evaluation():
        accuracy, per_class = evaluate_params(worker, worker.export(), test_x, test_y, int(config.get("evaluationBatchSize", 512)))
        if audit:
            audit.evaluation()
        return {"globalTestAccuracy": accuracy, "perClassGlobalTestAccuracy": per_class.tolist()}
    for round_id in range(completed_rounds,rounds):
        learning_rate=basil_round_learning_rate(float(config["learningRate"]),round_id)
        for node_id in range(n):
            if round_id*n+node_id < next_activation:
                continue
            if audit:
                audit.activation(round_id, node_id)
            if should_stop and should_stop():
                checkpoint()
                raise InterruptedError("Graceful stop requested before activation.")
            candidates=memories[node_id].candidates()
            if config.get("snapshotSelection"):
                probe=probe_indices[node_id]; selected,diagnostics=select_snapshot(node_id,candidates,lambda params,probe=probe:worker.probe(params,train_x[probe],train_y[probe]),node_count=n)
            else:
                selected=next((item for item in candidates if item.sender_id==(node_id-1)%n),predecessor); diagnostics=[]
            selected_sources[round_id,node_id]=selected.sender_id; selected_rounds[round_id,node_id]=selected.round_id; input_hash=params_hash(selected.params)
            if config.get("experimentProtocol")==IID_PROTOCOL:
                for candidate,diagnostic in zip(candidates,diagnostics):
                    link=candidate.channel_metadata or {}
                    diagnostic.update(senderByzantine=candidate.sender_id in attackers,
                        snapshotAgeRounds=round_id-candidate.round_id if candidate.round_id>=0 else None,
                        receivedThroughNoise=bool(link.get("configuredSigma",0)),
                        receivedModelNorm=parameter_stats(candidate.params)[1],receivedNoiseNorm=link.get("noiseL2Norm",0.))
            if audit:
                if config.get("experimentProtocol")==IID_PROTOCOL:
                    audit.context.update(selectedSource=selected.sender_id,selectedSnapshotRound=selected.round_id,
                        receivedNoiseStats=selected.channel_metadata or {})
                audit.selection(worker, node_id, candidates, selected, diagnostics, attackers,
                    config, test_x, test_y)
            legacy_lambda=0. if config.get("experimentProtocol")==IID_PROTOCOL else float(config.get("legacyEbmLambda",25.0))
            local=indices[node_id]
            try:
                trained,training=worker.train(selected.params,train_x[local],train_y[local],seed=seed,round_id=round_id,node_id=node_id,epochs=int(config["localEpochs"]),batch_size=int(config["batchSize"]),learning_rate=learning_rate,ebm_mode=config.get("ebmMode","none"),semantics=semantics,sigma=sigma,legacy_lambda=legacy_lambda,evaluation=evaluation,should_stop=should_stop)
            except InterruptedError:
                # Optimizer state resets per activation: discard the incomplete
                # activation, never a completed node or its memory deliveries.
                checkpoint()
                raise
            coefficient = float(np.mean([step["ebmCoefficient"] for step in training["optimizerTelemetry"]]))
            node_params[node_id]=copy_params(trained); global_accuracy,per_class=evaluate_params(worker,trained,test_x,test_y,int(config.get("evaluationBatchSize",512))); activation_accuracy[round_id,node_id]=global_accuracy; activation_per_class[round_id,node_id]=per_class
            is_byzantine=node_id in attackers; attack_active=bool(config.get("attackHidden")) and is_byzantine and round_id>=int(config.get("attackStartRound",20)); outbound=applyAttack(trained,"hidden",rng=keyed_rng(seed,"hidden_attack",round_id,node_id)) if attack_active else copy_params(trained)
            _,trained_norm=parameter_stats(trained); attack_change=math.sqrt(sum(float(np.sum((a-b)**2)) for a,b in zip(outbound,trained)))/max(trained_norm,1e-12)
            link_noise=[]; noise_active=bool(config.get("useChannelNoise")) and round_id>=int(config.get("channelNoiseStartRound",config.get("channelNoiseStart",0)))
            def link_params(receiver):
                if not noise_active:
                    dimension,norm=parameter_stats(outbound); stats={"coordinateSigma":0.0,"noiseL2Norm":0.0,"relativeNoiseL2":0.0,"modelL2Norm":norm,"parameterCount":dimension}; noisy=copy_params(outbound)
                else:noisy,stats=apply_channel_noise(outbound,semantics=config["channelNoiseSemantics"],sigma=sigma,rng=keyed_rng(seed,"channel_noise",round_id,node_id,receiver))
                link_noise.append({"receiverId":receiver,"transmittedHash":params_hash(noisy),"channelNoiseSemantics":semantics,"configuredSigma":sigma,**stats}); return noisy
            snapshot=Snapshot(node_id,round_id,outbound,training["loss"],global_accuracy); broadcast(memories,snapshot,link_params); predecessor=next(item for item in memories[(node_id+1)%n].candidates() if item.sender_id==node_id)
            if config.get("experimentProtocol")==IID_PROTOCOL:
                for link in link_noise:
                    memories[link["receiverId"]].snapshots[node_id].channel_metadata=link
            record={"round":round_id,"nodeId":node_id,"inputSourceNode":selected.sender_id,"inputSnapshotRound":selected.round_id,"inputHash":input_hash,"trainedOutputHash":params_hash(trained),"candidates":diagnostics,"selectedCandidateLoss":next((item["receiverLocalLoss"] for item in diagnostics if item["senderId"]==selected.sender_id),None),"selectedCandidateAccuracy":next((item["receiverLocalAccuracy"] for item in diagnostics if item["senderId"]==selected.sender_id),None),"localTrainingLoss":training["loss"],"localTrainingAccuracy":training["accuracy"],"globalTestAccuracy":global_accuracy,"perClassGlobalTestAccuracy":per_class.tolist(),"learningRate":learning_rate,"optimizerSteps":training["steps"],"modelL2Norm":parameter_stats(trained)[1],"configuredByzantine":is_byzantine,"attackActive":attack_active,"attackRelativeParameterChange":attack_change,"channelNoiseActive":noise_active,"outgoingLinks":link_noise,"ebmMode":config.get("ebmMode","none"),"ebmCoefficient":coefficient,"baseGradientNorm":training["baseGradientNorm"],"robustObjectiveGradientNorm":training["robustGradientNorm"]}
            telemetry.append(record)
            record.update(beforeTraining=training["beforeTraining"], epochs=training["epochs"],
                optimizerTelemetry=training["optimizerTelemetry"], localSampleCount=len(local),
                memorySenderIds=[s.sender_id for s in candidates], memorySnapshotRounds=[s.round_id for s in candidates],
                outboundAttackHash=params_hash(outbound), channelNoiseSemantics=semantics,
                configuredSigma=sigma, ebmCoefficientSemantics=config["ebmCoefficientSemantics"],
                effectiveCoordinateSigma=float(np.mean([s["effectiveCoordinateSigma"] for s in training["optimizerTelemetry"]])))
            if config.get("experimentProtocol")==IID_PROTOCOL:
                record.update(selectedSenderByzantine=selected.sender_id in attackers,
                    selectedSnapshotAgeRounds=round_id-selected.round_id if selected.round_id>=0 else None,
                    localObjective=config["localObjective"],legacyLambdaIgnored=True)
            if activation_callback: activation_callback(record)
            if audit:
                audit.record_activation(record)
            next_activation=round_id*n+node_id+1
            checkpoint()
        for node_id in range(n):
            if detailed_rounds:
                if audit: audit.context.update(round=round_id,node=node_id,phase="round_evaluation")
                accuracy,per_class,ce=evaluate_params(worker,node_params[node_id],test_x,test_y,int(config.get("evaluationBatchSize",512)),details=True)
                round_node_accuracy[round_id,node_id]=accuracy;round_node_per_class[round_id,node_id]=per_class;round_node_loss[round_id,node_id]=ce
            else:
                round_node_accuracy[round_id,node_id],round_node_per_class[round_id,node_id]=evaluate_params(worker,node_params[node_id],test_x,test_y,int(config.get("evaluationBatchSize",512)))
        if round_callback:
            event={"round":round_id,"averageAccuracy":float(round_node_accuracy[round_id].mean()),"worstNodeAccuracy":float(round_node_accuracy[round_id].min())}
            if detailed_rounds:
                event.update(fullTestAccuracy=float(round_node_accuracy[round_id,9]),fullTestLoss=float(round_node_loss[round_id,9]),perClassAccuracy=round_node_per_class[round_id,9].tolist())
            round_callback(event)
        completed_rounds=round_id+1
        checkpoint()
    result = {"resolvedAttackerIds":np.asarray(attackers,np.int16),"activationOverallAccuracy":activation_accuracy,"activationPerClassAccuracy":activation_per_class,"roundNodeAccuracy":round_node_accuracy,"roundNodePerClassAccuracy":round_node_per_class,"averageAccuracy":np.mean(round_node_accuracy,axis=1),"worstNodeAccuracy":np.min(round_node_accuracy,axis=1),"bestNodeAccuracy":np.max(round_node_accuracy,axis=1),"selectedSources":selected_sources,"selectedSnapshotRounds":selected_rounds,"finalNodeParams":node_params,"telemetry":telemetry}
    result["beforeTrainingAccuracy"] = np.asarray([r["beforeTraining"]["globalTestAccuracy"] for r in telemetry],np.float32).reshape(rounds,n)
    result["beforeTrainingPerClassAccuracy"] = np.asarray([r["beforeTraining"]["perClassGlobalTestAccuracy"] for r in telemetry],np.float32).reshape(rounds,n,10)
    result["epochOverallAccuracy"] = np.asarray([[e["globalTestAccuracy"] for e in r["epochs"]] for r in telemetry],np.float32).reshape(rounds,n,5)
    result["epochPerClassAccuracy"] = np.asarray([[e["perClassGlobalTestAccuracy"] for e in r["epochs"]] for r in telemetry],np.float32).reshape(rounds,n,5,10)
    if config.get("experimentProtocol")==IID_PROTOCOL:
        result["partitionIndices"] = indices
        if detailed_rounds:
            result.update(fullTestAccuracy=round_node_accuracy[:,9].copy(),fullTestLoss=round_node_loss[:,9].copy(),
                fullTestPerClassAccuracy=round_node_per_class[:,9].copy(),roundNodeLoss=round_node_loss,
                roundMeanTestLoss=round_node_loss.mean(axis=1))
    return result


__all__=["CLASS_NAMES","PROTOCOL","RollingMemory","SequentialWorker","Snapshot","apply_channel_noise","basil_round_learning_rate","broadcast","build_model","copy_params","ebm_coefficient","gradient_norm_objective_gradients","initialize_memories","keyed_rng","parameter_stats","params_hash","resolve_attackers","run_research_protocol","select_snapshot","validate_config"]
