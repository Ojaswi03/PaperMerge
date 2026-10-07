# IID BASIL — code changes and training mathematics

Updated 2026-10-07. **Actual source code, exact current line ranges, and plain-language training explanations.**

[Completed results and comparisons](../newResults/IID/ABCD_100r_comparison_gpu/FINAL_ABCD_100R_ANALYSIS.md) · [Machine-readable verification](../newResults/IID/ABCD_100r_comparison_gpu/POST_RUN_VERIFICATION.json)

## Reading guide

| If you want to understand… | Start here |
|---|---|
| Model and IID correctness | [Data and model](#1-data-and-model) |
| What one activation actually trains | [Deterministic training and handoff](#2-deterministic-training-and-strict-handoff) |
| Why BASIL is not averaging | [Memory and selection](#3-basil-memory-and-receiver-local-selection) |
| Noise, CE and the no-lambda EBM derivative | [Channel and objective](#4-channel-noise-and-full-source-grounded-ebm) |
| GPU safety and exact recovery | [Execution and checkpoints](#6-paired-configurations-gpu-policy-and-checkpoints) |
| Dashboard, ETA and queue fixes | [GUI changes](#7-gui-changes-live-updates-identity-clocks-and-queue) |
| Evidence for the mathematical claims | [Tests](#9-mathematical-and-analysis-tests) |

**How to use line references:** in VS Code, press **Ctrl+P**, type a repository path followed by `:line` (for example `basil_core/models.py:112`), and press Enter. Ranges are one-based, inclusive, and refer to the current file—not the Markdown line. Long excerpts are expandable; all Python blocks copy real source verbatim, not pseudocode.

## Protocol at a glance

| Setting | All four conditions |
|---|---|
| Dataset / partition | CIFAR-10; random disjoint IID; 5,000 samples per node |
| Logical network | 10-node fixed ring; five predecessor snapshots |
| Local training | Five full epochs; batch 512; remainder 392 |
| Model / optimizer | 117,706-parameter CNN; reset momentum-free SGD |
| Learning rate | 0.05 / (1 + 0.05r) |
| Snapshot Selection | Minimum receiver-local CE; complete weight replacement |
| A / B | Clean communication + CE; B has delayed attackers |
| C / D | Absolute σₑ=0.010; C uses CE, D uses full source EBM |

## Execution map

```text
Local training-only SS batch
          │
          ▼
Five latest predecessor snapshots
          │ receiver-local minimum CE
          ▼
ONE selected snapshot ──► complete parameter replacement
          │
          ▼
Five full local epochs ──► CE (A/B/C) or CE + σₑ²‖∇CE‖² (D)
          │
          ├─► honest trained state ──► evaluation-only metrics
          ▼
Outbound copy ──► delayed attack, if active ──► per-link noise, if enabled
          │
          ▼
Refresh five successor memories; only the immediate successor trains next
```

## Source-code index

<details>
<summary>Expand the complete file-and-line index (38 excerpts)</summary>

| Topic | Source | Current lines |
|---|---|---|
| CNN construction | [basil_core/models.py](../basil_core/models.py) | 112–129 |
| IID adapter | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 423–426 |
| IID audit | [basil_core/iid_study.py](../basil_core/iid_study.py) | 75–99 |
| Full-epoch batch sizes | [basil_core/data/cifar.py](../basil_core/data/cifar.py) | 217–225 |
| Keyed RNG | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 28–31 |
| Learning-rate schedule | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 57–58 |
| Strict weight load | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 280–284 |
| Full local training and SGD | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 329–366 |
| Stateless augmentation | [basil_core/research_audit.py](../basil_core/research_audit.py) | 35–42 |
| Rolling memory | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 72–90 |
| BASIL selection | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 114–130 |
| Fixed selection subset | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 475–476 |
| Shared local evaluator | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 496–500 |
| Gaussian channel | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 133–143 |
| Nested-tape EBM | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 153–182 |
| GPU CE versus EBM | [basil_core/iid_gpu_worker.py](../basil_core/iid_gpu_worker.py) | 11–37 |
| Attack and transmission order | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 528–536 |
| Allowed config differences | [basil_core/iid_campaign.py](../basil_core/iid_campaign.py) | 16–20 |
| Pairing validator | [basil_core/iid_campaign.py](../basil_core/iid_campaign.py) | 84–94 |
| GPU policy | [basil_core/iid_runtime.py](../basil_core/iid_runtime.py) | 40–53 |
| Atomic complete-state save | [basil_core/protocol_checkpoint.py](../basil_core/protocol_checkpoint.py) | 62–81 |
| Checkpoint validation and memory restoration | [basil_core/protocol_checkpoint.py](../basil_core/protocol_checkpoint.py) | 84–109 |
| Legacy GUI import shim | [gui/experiment_app.py](../gui/experiment_app.py) | 1–7 |
| Main-thread polling | [gui/app.py](../gui/app.py) | 125–135 |
| Activation versus round events | [gui/services/iid_execution_service.py](../gui/services/iid_execution_service.py) | 65–79 |
| Experiment identity and ETAs | [gui/views/dashboard_view.py](../gui/views/dashboard_view.py) | 59–78 |
| Live curve styles | [gui/views/dashboard_view.py](../gui/views/dashboard_view.py) | 6–10 |
| Queue column definitions | [gui/views/queue_view.py](../gui/views/queue_view.py) | 15–16 |
| Incremental queue rows | [gui/views/queue_view.py](../gui/views/queue_view.py) | 41–46 |
| Entry contrast | [gui/theme.py](../gui/theme.py) | 113–126 |
| Round evaluation | [basil_core/research_protocol.py](../basil_core/research_protocol.py) | 557–568 |
| Convergence windows | [reporting/iid_campaign_plots.py](../reporting/iid_campaign_plots.py) | 16–23 |
| Scientific contrasts | [reporting/iid_campaign_plots.py](../reporting/iid_campaign_plots.py) | 26–32 |
| Completed-result statistics | [scripts/analyze_completed_iid.py](../scripts/analyze_completed_iid.py) | 35–46 |
| Actual corruption classification | [scripts/analyze_completed_iid.py](../scripts/analyze_completed_iid.py) | 57–59 |
| EBM analytical and finite-difference checks | [tests/test_iid_study.py](../tests/test_iid_study.py) | 99–121 |
| No-lambda and SGD checks | [tests/test_iid_study.py](../tests/test_iid_study.py) | 124–143 |
| Read-only analysis regressions | [tests/test_completed_iid_analysis.py](../tests/test_completed_iid_analysis.py) | 8–33 |

</details>

## 1. Data and model

### 1.1 Added small BASIL CNN

Source: [basil_core/models.py](../basil_core/models.py) — **lines 112–129**.

```python
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, seed=2025):
        def dense_initializer(fan_in, offset):
            bound = fan_in ** -0.5
            return tf.keras.initializers.RandomUniform(-bound, bound, seed=seed + offset)

        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, activation="relu", padding="valid",
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
            layers.MaxPool2D(pool_size=3, strides=3),
            layers.Conv2D(64, 4, activation="relu", padding="valid",
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1)),
            layers.MaxPool2D(pool_size=4, strides=4),
            layers.Flatten(),
            layers.Dense(384, activation="relu", kernel_initializer=dense_initializer(64, 2)),
            layers.Dense(192, activation="relu", kernel_initializer=dense_initializer(384, 3)),
            layers.Dense(num_classes, kernel_initializer=dense_initializer(192, 4)),
        ])
```

The added `BasilPaperCifarModel` is separate from the retained historical `CIFARModel`. It outputs **logits**, not a training softmax. Conv kernels use seeded Glorot initialization; dense kernels use bounds ±1/√fan-in; biases default to zero.

| Stage | Output | Trainable parameters |
|---|---|---:|
| Input | 32 × 32 × 3 | 0 |
| Conv: 16 filters, 3 × 3, valid | 30 × 30 × 16 | 448 |
| Pool: 3 × 3, stride 3 | 10 × 10 × 16 | 0 |
| Conv: 64 filters, 4 × 4, valid | 7 × 7 × 64 | 16,448 |
| Pool: 4 × 4, stride 4 | 1 × 1 × 64 | 0 |
| Flatten → dense 384 | 384 | 24,960 |
| Dense 192 | 192 | 73,920 |
| Dense 10 logits | 10 | 1,930 |
| **Total** | | **117,706** |

The second pool reduces spatial features to 64 values. That is a plausible representation constraint—not proof of a 58% ceiling.

### 1.2 Reuse the existing IID partitioner

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 423–426**.

```python
def iid_partition_indices(labels, seed, node_count=10):
    from basil_core.data.cifar import _loadOrCreatePartition
    return _loadOrCreatePartition(labels,iid=True,nClients=node_count,alpha=.2,seed=seed,
                                  rng=keyed_rng(seed,"iid_partition"),cacheDir=None)
```

**What changed:** the new protocol calls the existing partitioner with an explicit keyed seed. It does **not** introduce a second IID splitting algorithm.

Training indices alone are shuffled and split into ten disjoint sets of 5,000. Test examples are not arguments to this adapter. Random IID does not mean exactly 500 examples of every class per node.

### 1.3 Added full-data partition audit

Source: [basil_core/iid_study.py](../basil_core/iid_study.py) — **lines 75–99**.

<details>
<summary>Show actual code: IID audit</summary>

```python
def partition_audit(train_labels, test_labels, *, seed=2025):
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    if len(train_labels) != 50000 or not np.array_equal(np.bincount(train_labels,minlength=10),[5000]*10):
        raise ValueError("Expected the complete CIFAR-10 training set.")
    if len(test_labels) != 10000 or not np.array_equal(np.bincount(test_labels,minlength=10),[1000]*10):
        raise ValueError("Expected the complete evaluation-only CIFAR-10 test set.")
    indices = iid_partition_indices(train_labels, seed)
    repeat = iid_partition_indices(train_labels, seed)
    joined = np.concatenate(indices)
    digest = hashlib.sha256()
    rows = []
    for node, chunk in enumerate(indices):
        counts = np.bincount(train_labels[chunk], minlength=10)
        if len(chunk) != 5000 or np.any(counts == 0) or not np.array_equal(chunk, repeat[node]):
            raise ValueError("IID cardinality, class coverage or determinism failed.")
        digest.update(np.asarray(chunk,dtype="<i8").tobytes())
        rows.append({"nodeId":node,"samples":len(chunk),"classCounts":counts.tolist()})
    if not np.array_equal(np.sort(joined),np.arange(50000)):
        raise ValueError("IID training assignments overlap or omit samples.")
    return {"strategy":"existing_seeded_random_disjoint_equal_split", "partitionHash":digest.hexdigest(),
        "seed":seed,"rngStream":"iid_partition", "assignedSamples":len(joined),
        "uniqueSamples":len(np.unique(joined)),"nodes":rows,"deterministic":True,
        "trainClassCounts":np.bincount(train_labels,minlength=10).tolist(),
        "testClassCounts":np.bincount(test_labels,minlength=10).tolist(),
        "testUsage":"evaluation_only; partition and selection receive training indices only"}, indices
```

</details>

**Checks:** 50,000 training samples; 10,000 evaluation samples; all classes per node; 5,000 samples per node; reproducibility; no overlap or omission. The audit hashes the ordered node-index arrays.

| Property | Executed A/B/C/D evidence |
|---|---|
| Assigned / unique training samples | 50,000 / 50,000 |
| Local datasets | 10 × 5,000 |
| Class counts per node | 450–545; all ten classes represented |
| Full test set | 10,000; 1,000 per class; evaluation only |
| Same partition in all four | Yes; complete SHA-256 in the results report |

The test-label counts are checked for evaluation integrity; they do not influence training assignment.

### 1.4 Added full-epoch batch helper

Source: [basil_core/data/cifar.py](../basil_core/data/cifar.py) — **lines 217–225**.

```python
def fullLocalEpochBatches(sampleCount, batchSize, localEpochs):
    """Return finite batch sizes; the final partial batch is never discarded."""
    sampleCount, batchSize, localEpochs = map(int, (sampleCount, batchSize, localEpochs))
    if sampleCount <= 0 or batchSize <= 0 or localEpochs <= 0:
        raise ValueError("sampleCount, batchSize, and localEpochs must be positive.")
    oneEpoch = [batchSize] * (sampleCount // batchSize)
    if sampleCount % batchSize:
        oneEpoch.append(sampleCount % batchSize)
    return oneEpoch * localEpochs
```

For 5,000 samples and batch size 512:

```text
One epoch:  [512 × 9] + [392] = 5,000 samples
Activation: 5 epochs × 10 batches = 50 optimizer updates
Condition:  100 rounds × 10 nodes × 50 updates = 50,000 updates
```

This helper expresses the contract; the actual training loop below performs the visits. The separate `oneClassPerNodePartition` addition at lines 206–214 remains available but was **not used** in these IID runs.

## 2. Deterministic training and strict handoff

### 2.1 Keyed random streams

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 28–31**.

```python
def keyed_rng(seed: int, *parts: object) -> np.random.Generator:
    key = "|".join((str(int(seed)), *(str(part) for part in parts)))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))
```

A stream key includes the master seed and logical context. Separate keys are used for initialization, partitioning, sample order, augmentation, attacker selection, attack, and sender→receiver channel noise. Enabling EBM cannot advance an unrelated mutable noise RNG.

### 2.2 Unchanged approved round-decay equation, now explicit

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 57–58**.

```python
def basil_round_learning_rate(initial: float, round_id: int) -> float:
    return float(initial) / (1.0 + 0.05 * int(round_id))
```

The round is a complete ten-node traversal:

\[
\eta_r=\frac{0.05}{1+0.05r}.
\]

| Round | Learning rate |
|---:|---:|
| 0 | 0.050000 |
| 49 | 0.014493 |
| 99 | 0.008403 |

The test set does not adjust this schedule.

### 2.3 Complete parameter replacement

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 280–284**.

```python
    def load(self, params):
        if len(params) != len(self.model.trainable_variables):
            raise ValueError("Checkpoint parameter count does not match the model.")
        for variable, value in zip(self.model.trainable_variables, params):
            variable.assign(value)
```

For every tensor, assignment replaces its value:

\[
\theta_{\text{start},i}=\theta_{\text{selected},i}.
\]

There is no averaging with the receiver's stale model, no FedAvg, and no pairwise consensus. Saved input hashes were checked against first-batch parameter hashes.

### 2.4 Five complete epochs and reset SGD

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 329–366**.

<details>
<summary>Show actual code: Full local training and SGD</summary>

```python
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
```

</details>

**Training effect:** reload the selected snapshot; create momentum-free SGD; visit one complete keyed permutation each epoch; keep the last partial batch; apply gradients; stop safely between steps; save evidence on errors.

| Code | Meaning |
|---|---|
| `if epochs != 5` | Refuse an unapproved epoch count |
| `self.load(params)` | Start from the complete selected model |
| New `SGD(..., momentum=0.0)` | No optimizer history transmitted across activations |
| `range(0, len(order), batch_size)` | Includes the remainder, not `drop_remainder=True` |
| `apply_gradients` | Update θ ← θ − ηᵣg; D uses the full EBM gradient |
| `should_stop` / audit exception | Graceful interruption and finite-value evidence |

The general method signature retains `legacy_lambda` for historical compatibility. The IID caller passes zero and the IID/GPU gradient overrides never read it.

### 2.5 Reproducible training-only augmentation

Source: [basil_core/research_audit.py](../basil_core/research_audit.py) — **lines 35–42**.

```python
def augment_batch(x, key):
    def augment(item):
        image, index = item
        sample_key = tf.random.experimental.stateless_fold_in(key, index)
        image = tf.image.stateless_random_flip_left_right(image, sample_key)
        image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
        return tf.image.stateless_random_crop(image, [32, 32, 3], sample_key + [0, 1])
    return tf.map_fn(augment, (x, tf.range(tf.shape(x)[0])), fn_output_signature=tf.float32)
```

Each training image receives a keyed horizontal flip, four-pixel pad and 32 × 32 crop. Snapshot Selection uses normalized **unaugmented** local training images. Test images are never augmented for training.

## 3. BASIL memory and receiver-local selection

### 3.1 Latest snapshot from each distinct predecessor

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 72–90**.

```python
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
```

Dictionary keys are sender IDs. A new snapshot replaces that sender's previous entry. This is not five historical versions from one sender.

```text
Before Node 0, Round 11:
memory[0] = {5:R10, 6:R10, 7:R10, 8:R10, 9:R10}

Node 0 trains and refreshes five clockwise memories:
Node 0 ──► Node 1   next active trainer
       ├─► Node 2   cache only
       ├─► Node 3   cache only
       ├─► Node 4   cache only
       └─► Node 5   cache only

Before Node 1:
memory[1] = {6:R10, 7:R10, 8:R10, 9:R10, 0:R11}
```

### 3.2 Select one minimum-local-loss model

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 114–130**.

```python
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
```

Every candidate is passed to the same receiver evaluator. Sender-reported metrics are stored for diagnostics but are not used by the minimum-loss or tie-breaking expression.

\[
j^*=\arg\min_{j\in\mathcal M_i} CE(\theta_j;B_i^{SS}).
\]

Losses within 10⁻⁸ are treated as tied; the nearest counterclockwise sender wins.

### 3.3 Selection batch comes from training data

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 475–476**.

```python
    probe_indices=[chunk[:min(512,len(chunk))] for chunk in indices]
    if any(len(chunk) == 0 for chunk in indices):
```

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 496–500**.

```python
                checkpoint()
                raise InterruptedError("Graceful stop requested before activation.")
            candidates=memories[node_id].candidates()
            if config.get("snapshotSelection"):
                probe=probe_indices[node_id]; selected,diagnostics=select_snapshot(node_id,candidates,lambda params,probe=probe:worker.probe(params,train_x[probe],train_y[probe]),node_count=n)
```

The same first up-to-512 indices of the already randomized local partition are used for all candidates. This batch is reused across activations. Global test accuracy is **not** a selection input.

**Qualification:** reusing a small fixed local subset could bias selection; that is a hypothesis, not demonstrated test leakage or a confirmed protocol defect.

## 4. Channel noise and full source-grounded EBM

### 4.1 Explicit channel semantics

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 133–143**.

```python
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
```

Only the absolute branch is used by C/D:

| Branch | Coordinate standard deviation | Used by current A/B/C/D? |
|---|---|---|
| `paper_absolute_gaussian` | σₑ = 0.010, independent of model norm | C/D |
| `relative_l2_gaussian` | σrel‖θ‖/√d | No; retained separate branch |
| Noise disabled | No parameter perturbation | A/B |

For C/D:

\[
\tilde\theta=\theta^{out}+\xi,\qquad
\xi\sim\mathcal N(0,0.0001I_d).
\]

Standard deviation is **0.010**; variance is **0.0001**. The norm ratio is telemetry, not a control variable.

### 4.2 Differentiate the actual modified loss

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 153–182**.

<details>
<summary>Show actual code: Nested-tape EBM</summary>

```python
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
```

</details>

The inner tape computes CE gradients; the outer tape differentiates their squared norm. It does not allocate a full Hessian.

| Quantity | Mathematical definition | Code |
|---|---|---|
| Base loss | F = CE | `base_loss` |
| Base gradient | g = ∇F | `base_grads` |
| Penalty | c‖g‖² | `coefficient * norm_sq` |
| Total objective | L = F + c‖g‖² | `objective` |
| Full gradient | ∇L = g + 2cHg | `robust` |
| Correction | ∇L − g | `correction` |

For D, **c = σₑ² = 0.0001**, so the correction is 0.0002Hg. `tf.stop_gradient` treats the coefficient as fixed. Unexpected missing gradients are rejected in the IID detailed path.

### 4.3 GPU worker keeps CE and EBM separate

Source: [basil_core/iid_gpu_worker.py](../basil_core/iid_gpu_worker.py) — **lines 11–37**.

<details>
<summary>Show actual code: GPU CE versus EBM</summary>

```python
    def _batch_gradients(self,x,y,key,mode,semantics,sigma,legacy_lambda):
        if semantics!='paper_absolute_gaussian' or mode not in {'none','gradient_norm_objective'}:
            raise ValueError('GPU IID execution supports CE and source EBM only.')
        # Per-image map/seed/crop operations are expensive tiny CUDA launches.
        # The original keyed transform stays byte-identical on the CPU.
        with tf.device('/CPU:0'):
            images=augment_batch(x,key)
        with tf.device('/GPU:0'):
            for variable in self.model.trainable_variables:
                tf.debugging.assert_all_finite(variable,'Non-finite incoming parameter')
            if mode=='gradient_norm_objective':
                gradients,values=gradient_norm_objective_gradients(self.model,images,y,sigma**2,details=True)
            else:
                with tf.GradientTape() as tape:
                    logits=self.model(images,training=True)
                    tf.debugging.assert_all_finite(logits,'Non-finite training logits')
                    ce=lossFn(y,logits)
                gradients=tape.gradient(ce,self.model.trainable_variables)
                if any(g is None for g in gradients):raise ValueError('Disconnected CE gradient.')
                norm=tf.linalg.global_norm(gradients)
                accuracy=tf.reduce_mean(tf.cast(tf.argmax(logits,axis=1,output_type=tf.int32)==y,tf.float32))
                values=(ce,norm**2,ce,tf.constant(0.),norm,norm,tf.constant(0.,tf.float64),accuracy)
            for name,value in zip(('CE','gradient norm squared','objective','coefficient',
                'base norm','applied norm','EBM correction norm','accuracy'),values):
                tf.debugging.assert_all_finite(value,f'Non-finite {name}')
            for gradient in gradients:tf.debugging.assert_all_finite(gradient,'Non-finite applied gradient')
            return gradients,values
```

</details>

The worker calls the EBM helper with exactly `sigma**2`. It never reads `legacy_lambda`, even though the inherited interface accepts the argument. Its CE branch reports zero coefficient and correction.

| Condition | Training objective | Outgoing channel |
|---|---|---|
| A/B | CE | Clean |
| C | CE | Absolute Gaussian |
| D | CE + 0.0001‖∇CE‖² | Same absolute Gaussian |

**Source distinction:** the noisy-communication paper's Eq. 14 concerns a squared-loss approximation. We adapt its gradient-norm regularization concept to multiclass CE; this is not an identity for exact expected noisy CNN CE. Eq. 23 scalar gradient scaling is not equivalent to g + 2cHg for a general CNN. See [source equations](SOURCE_EQUATIONS.md).

**Not active:** snapshot anchoring, μ, adaptive μ, CART, class registry, Monte Carlo loss, weight decay, clipping, or legacy scaling. Proposed non-IID objectives must not be inferred from unused config field names.

## 5. Attack before per-link noise

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 528–536**.

```python
            is_byzantine=node_id in attackers; attack_active=bool(config.get("attackHidden")) and is_byzantine and round_id>=int(config.get("attackStartRound",20)); outbound=applyAttack(trained,"hidden",rng=keyed_rng(seed,"hidden_attack",round_id,node_id)) if attack_active else copy_params(trained)
            _,trained_norm=parameter_stats(trained); attack_change=math.sqrt(sum(float(np.sum((a-b)**2)) for a,b in zip(outbound,trained)))/max(trained_norm,1e-12)
            link_noise=[]; noise_active=bool(config.get("useChannelNoise")) and round_id>=int(config.get("channelNoiseStartRound",config.get("channelNoiseStart",0)))
            def link_params(receiver):
                if not noise_active:
                    dimension,norm=parameter_stats(outbound); stats={"coordinateSigma":0.0,"noiseL2Norm":0.0,"relativeNoiseL2":0.0,"modelL2Norm":norm,"parameterCount":dimension}; noisy=copy_params(outbound)
                else:noisy,stats=apply_channel_noise(outbound,semantics=config["channelNoiseSemantics"],sigma=sigma,rng=keyed_rng(seed,"channel_noise",round_id,node_id,receiver))
                link_noise.append({"receiverId":receiver,"transmittedHash":params_hash(noisy),"channelNoiseSemantics":semantics,"configuredSigma":sigma,**stats}); return noisy
            snapshot=Snapshot(node_id,round_id,outbound,training["loss"],global_accuracy); broadcast(memories,snapshot,link_params); predecessor=next(item for item in memories[(node_id+1)%n].candidates() if item.sender_id==node_id)
```

The existing `applyAttack(..., "hidden")` implementation is reused from [basil_core/attacks.py](../basil_core/attacks.py); that supporting file was not rewritten for this analysis.

| Step | State affected |
|---|---|
| Local training and evaluation | Honest trained node parameters |
| Hidden attack, designated node and R ≥ 20 | Outbound copy only |
| Gaussian corruption | Independent copy for each directed link |
| Delivery | One latest snapshot per sender in five successor memories |

With the project's strength/blend defaults, the outgoing transform is approximately −0.32θ plus a small attack perturbation. That can raise noise/model ratios by shrinking the denominator even though coordinate σ stays fixed. It is not claimed to reproduce another attack paper exactly.

The link RNG key includes **seed, round, sender, receiver**. Corresponding C/D additive noise samples match; different links receive independent samples.

## 6. Paired configurations, GPU policy and checkpoints

### 6.1 Enforce only the intended scientific differences

Source: [basil_core/iid_campaign.py](../basil_core/iid_campaign.py) — **lines 16–20**.

```python
CONTRAST_FIELDS = {
    'A/B': {'attackerCount','actualAttackerCount','resolvedAttackerIds','attackerIds','attackHidden'},
    'B/C': {'useChannelNoise','channelNoiseSigmaAbsolute'},
    'C/D': {'localObjective','ebmMode'},
}
```

Source: [basil_core/iid_campaign.py](../basil_core/iid_campaign.py) — **lines 84–94**.

```python
def verify_campaign(configs):
    from basil_core.research_protocol import validate_config
    if [c.get('conditionId') for c in configs]!=list(CONDITIONS):raise ValueError('Queue order must be A, B, C, D.')
    for config in configs:validate_config(config)
    differences={}
    for label,(left,right) in zip(CONTRAST_FIELDS,((0,1),(1,2),(2,3))):
        a,b=configs[left],configs[right]
        diff={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        if diff-METADATA_KEYS!=CONTRAST_FIELDS[label]:raise ValueError(f'Unexpected scientific differences for {label}: {sorted(diff-METADATA_KEYS)}')
        differences[label]=sorted(diff-METADATA_KEYS)
    return differences
```

Run names and output paths are metadata exceptions; other differences must equal the listed scientific fields.

| Contrast | Changed | Kept the same |
|---|---|---|
| A → B | Actual attackers enabled | BASIL S=5, IID data, initialization, training |
| B → C | Channel enabled; σₑ=0.010 | Attack, CE, partition, training |
| C → D | `localObjective` / `ebmMode` | Channel, sigma, keyed draws, attackers, training |

A's actual attacker count is zero while its assumed bound remains four; its BASIL memory does not shrink to one.

### 6.2 GPU configuration and fail-closed behavior

Source: [basil_core/iid_runtime.py](../basil_core/iid_runtime.py) — **lines 40–53**.

```python
def configure_tensorflow(tf, device: str) -> None:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    if device == 'GPU':
        acquire_gpu_lease()
        free_gpu_memory()
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError('GPU was requested but TensorFlow cannot access CUDA. No CPU fallback was started.')
        tf.config.set_logical_device_configuration(gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_MB)])
        # Preserve float32 arithmetic rather than using Ampere/Ada TF32 kernels.
        tf.config.experimental.enable_tensor_float_32_execution(False)
```

Constants are 4,096 MiB cap and 1,024 MiB headroom ([lines 13–14](../basil_core/iid_runtime.py)). Before launch, free-memory checks require at least 5,120 MiB. An OS lock permits one IID GPU process at a time ([lines 56–80](../basil_core/iid_runtime.py)).

**Training consequence:** float32 GPU kernels with deterministic ops and TF32 disabled; no silent CPU fallback. CPU keyed augmentation avoids many tiny CUDA launches. This is an execution change, not a new loss or learning-rate rule.

### 6.3 Persist full logical state, not final weights alone

Source: [basil_core/protocol_checkpoint.py](../basil_core/protocol_checkpoint.py) — **lines 62–81**.

```python
    memories = []
    for memory in state['memories']:
        memories.append([dict(sender=s.sender_id, round=s.round_id, params=pack_params(s.params),
            loss=s.sender_loss, accuracy=s.sender_accuracy, channel=s.channel_metadata)
            for s in memory.candidates()])
    metadata = dict(version=1, config=config, nextActivation=state['next_activation'],
        implementation=implementation_fingerprint(),
        completedRounds=state['completed_rounds'], partitionHash=partition_hash,
        initialHash=initial_hash, nodeParams=[pack_params(p) for p in state['node_params']],
        memories=memories, weights=weights, telemetry=state['telemetry'])
    arrays['metadata'] = np.asarray(json.dumps(metadata, allow_nan=False))
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    # Retain the previous committed checkpoint if the newest file is damaged.
    if path.exists():
        os.replace(path, path.with_name('checkpoint.previous.npz'))
    os.replace(temporary, path)
```

The preceding packer deduplicates parameter arrays by hash. The checkpoint retains ten node states, all sender/round memberships, metrics, activation cursor and telemetry. Writes are flushed, fsynced and atomically committed, retaining the previous checkpoint.

Source: [basil_core/protocol_checkpoint.py](../basil_core/protocol_checkpoint.py) — **lines 84–109**.

<details>
<summary>Show actual code: Checkpoint validation and memory restoration</summary>

```python
def load_checkpoint(path, config, *, partition_hash, initial_hash):
    from basil_core.research_protocol import RollingMemory, Snapshot, params_hash
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        if metadata['version'] != 1 or metadata['config'] != config:
            raise ValueError('Checkpoint configuration does not match this experiment.')
        if metadata['implementation'] != implementation_fingerprint():
            raise ValueError('Scientific implementation changed since checkpoint creation.')
        if (metadata['partitionHash'], metadata['initialHash']) != (partition_hash, initial_hash):
            raise ValueError('Checkpoint partition or initialization hash differs.')

        def unpack_params(digest):
            params = [saved[f'{digest}_{i}'].copy() for i in range(metadata['weights'][digest])]
            if params_hash(params) != digest or any(not np.isfinite(p).all() for p in params):
                raise ValueError('Checkpoint parameters are corrupt or non-finite.')
            return params

        memories = []
        for receiver, snapshots in enumerate(metadata['memories']):
            memory = RollingMemory(receiver)
            for s in snapshots:
                memory.receive(Snapshot(s['sender'], s['round'], unpack_params(s['params']),
                    s['loss'], s['accuracy'], s['channel']))
            if len(memory.candidates()) != 5:
                raise ValueError('Checkpoint must contain all five predecessor snapshots.')
            memories.append(memory)
```

</details>

Resume rejects mismatched configuration, scientific source, partition, initialization, corrupt weights, or missing five-snapshot memories. Keyed RNGs can be reconstructed from logical indices. Optimizer state is reset at activation boundaries; partial activations are not treated as completed.

**Current results:** A/B/C/D are fresh GPU runs, not CPU→GPU continuations. Historical reconstruction/transfer support is separate; exact state restoration would not prove bit-identical CPU/GPU arithmetic.

## 7. GUI changes: live updates, identity, clocks and queue

### 7.1 Replace the old monolith with a compatibility shim

Source: [gui/experiment_app.py](../gui/experiment_app.py) — **lines 1–7**.

```python
"""Compatibility import for the pre-workspace GUI module."""

from gui.app import PaperMergeApp, main

ExperimentGUI = PaperMergeApp

__all__ = ["ExperimentGUI", "PaperMergeApp", "main"]
```

The launcher now imports the new application in [run_gui.py](../run_gui.py), **line 26**. The shell has five persistent workspaces; state and services own configuration, execution and persistence. This changes orchestration, not gradients.

### 7.2 Poll and dispatch on the Tk main thread

Source: [gui/app.py](../gui/app.py) — **lines 125–135**.

```python
    def _poll(self):
        self.state.tick()
        if self._worker_handle and self._worker_handle.active:self._worker_handle.poll()
        self.execution.dispatch_pending()
        if self._active_queue_entry:
            done=max(self._active_queue_entry.completed_rounds,self.state.completed_rounds)
            if done!=self._active_queue_entry.completed_rounds:
                self._active_queue_entry.completed_rounds=done
                if hasattr(self,'shell'):self.shell.views[Workspace.QUEUE].refresh()
        if hasattr(self,'shell') and time.monotonic()-getattr(self,'_last_clock_refresh',0)>=1:
            self.shell.views[Workspace.DASHBOARD].refresh(); self._last_clock_refresh=time.monotonic()
```

A reader thread queues worker messages; Tk polling drains them into state. Completed-round counts advance only from measured complete rounds, not from partial activations.

Source: [gui/services/iid_execution_service.py](../gui/services/iid_execution_service.py) — **lines 65–79**.

```python
        elif 'roundComplete' in payload:
            r=payload['roundComplete']
            self.sink(ExecutionEvent('progress_updated',{'round':r['round']+1,'completedRounds':r['round']+1,'progress':(r['round']+1)/rounds}))
            self.sink(ExecutionEvent('accuracy_updated',{'average':r['averageAccuracy'],'worst':r['worstNodeAccuracy']}))
            self.sink(ExecutionEvent('network_round_updated',{**context,'payload':{**r,'round':r['round']+1}}))
        elif 'node' in payload and 'round' in payload:
            position=payload['round']+(payload['node']+1)/nodes
            self.sink(ExecutionEvent('progress_updated',{'round':payload['round']+1,'progress':position/rounds}))
            if payload.get('accuracy') is not None:
                self.sink(ExecutionEvent('node_accuracy_updated',{'node':payload['node'],'accuracy':payload['accuracy'],'round_position':position}))
            update=payload.get('networkUpdate')
            if update is None:
                # Older workers publish accuracy only, not candidate/link details.
                update={'round':payload['round']+1,'nodeId':payload['node'],'globalTestAccuracy':payload.get('accuracy'),'telemetryComplete':False}
            self.sink(ExecutionEvent('network_updated',{**context,'payload':update}))
```

**Why the distinction matters:** an activation event updates the latest-node curve/network immediately; complete-round events update node mean and worst. These are different quantities.

### 7.3 Active experiment name and separate clocks

Source: [gui/views/dashboard_view.py](../gui/views/dashboard_view.py) — **lines 59–78**.

<details>
<summary>Show actual code: Experiment identity and ETAs</summary>

```python
        name=state.active_experiment_name or state.experiment.experiment_name
        running=state.execution.value in {'preparing','running','stopping'}
        self._text(self.identity,f'{"Running experiment" if running else "Experiment"}: {name}')
        recovering=state.recovery_completed<state.recovery_total
        context=(f'Rebuilding saved state: {state.recovery_completed}/{state.recovery_total} activations verified. '
            'This replays recorded training; it is not a new scientific run.' if recovering else
            'Experiment elapsed includes prior attempts. Queue elapsed is this launch session; ETAs are estimates.')
        if state.execution_device:context+=f' Device: {state.execution_device} — {state.device_name}.'
        self._text(self.context,context)
        elapsed=state.elapsed_seconds; eta=state.experiment_eta_seconds
        pending=[e for e in self.queue_state.entries if e.status.value=='pending']
        estimates=[e.estimated_remaining_seconds for e in pending]
        queue_eta=eta+sum(estimates) if state.queue_active and eta is not None and all(v is not None for v in estimates) else None
        rounds=state.active_experiment_rounds or state.experiment.rounds
        values=(state.execution.value.title(),f"{state.progress:.1%}",f"{state.current_round} / {rounds}",
            duration(elapsed),"Estimating…" if eta is None and running else "—" if eta is None else duration(eta),str(state.worker_count),
            "—" if state.average_accuracy is None else f"{state.average_accuracy:.2%}",
            "—" if state.worst_accuracy is None else f"{state.worst_accuracy:.2%}",f'{len(pending)} pending',
            duration(state.queue_elapsed_seconds) if state.queue_started_monotonic is not None else 'Not started',
            duration(queue_eta) if queue_eta is not None else 'Estimating…' if state.queue_active else 'Paused / not started')
```

</details>

| Display | Meaning |
|---|---|
| Experiment elapsed | Current condition, including prior attempts if applicable |
| Experiment ETA | Estimated remaining work in that condition |
| Queue elapsed | This queue-launch session |
| Queue ETA | Current remaining time plus pending estimates, only when all are available |
| “Estimating…” | Insufficient measured/known timing—not proof of a stalled worker |

Labels are changed only when text changes; the view does not repeatedly recreate every widget.

### 7.4 Distinguishable live curves

Source: [gui/views/dashboard_view.py](../gui/views/dashboard_view.py) — **lines 6–10**.

```python
ACCURACY_SERIES = (
    ('#0072B2', (2, 3), 2, 'Node just trained'),
    ('#00875A', (), 3, 'Round mean'),
    ('#D55E00', (7, 3), 2, 'Round worst'),
)
```

| Curve | Color | Pattern | Update frequency |
|---|---|---|---|
| Node just trained | Blue, #0072B2 | Short dashes | Every activation |
| Round mean | Green, #00875A | Solid | Complete round |
| Round worst | Orange, #D55E00 | Long dashes | Complete round |

Color and line pattern both distinguish series. The x-axis uses fractional completed rounds for activation points.

### 7.5 Queue Done / total and incremental rows

Source: [gui/views/queue_view.py](../gui/views/queue_view.py) — **lines 15–16**.

```python
        self.tree=ttk.Treeview(self,columns=("status","name","dataset","rounds","done","device","recovery"),show="headings",selectmode="extended")
        for key,label,width in (("status","Status",90),("name","Experiment",280),("dataset","Dataset",90),("rounds","Rounds",65),("done","Done / total",95),("device","Device",60),("recovery","Start / recovery",260)): self.tree.heading(key,text=label); self.tree.column(key,width=width,stretch=key in {"name","recovery"})
```

Source: [gui/views/queue_view.py](../gui/views/queue_view.py) — **lines 41–46**.

```python
        for index,entry in enumerate(visible):
            name=entry.config.extra.get('displayName',entry.config.experiment_name).split(' — Fresh')[0]
            values=tuple(map(str,(entry.status.value.title(),name,f'{entry.config.dataset}/{entry.config.split}',entry.config.rounds,f'{entry.completed_rounds}/{entry.config.rounds}',entry.config.extra.get('executionDevice','CPU'),entry.execution_hint)))
            if not self.tree.exists(entry.entry_id):self.tree.insert('',index,iid=entry.entry_id,values=values)
            elif tuple(self.tree.item(entry.entry_id,'values'))!=values:self.tree.item(entry.entry_id,values=values)
            if self.tree.index(entry.entry_id)!=index:self.tree.move(entry.entry_id,'',index)
```

Actual evaluated rounds appear as `completed_rounds / total`. Device and fresh/recovery mode remain visible; unchanged rows are not cleared and rebuilt.

### 7.6 Readable input colors

Source: [gui/theme.py](../gui/theme.py) — **lines 113–126**.

```python
    # Explicit field colors prevent platform themes from producing white-on-white text.
    style.configure(
        "TEntry",
        padding=7,
        fieldbackground=COLORS["input"],
        foreground=COLORS["text"],
        insertcolor=COLORS["text"],
        bordercolor=COLORS["border_strong"],
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", COLORS["disabled_bg"]), ("readonly", COLORS["input"])],
        foreground=[("disabled", COLORS["disabled_text"]), ("readonly", COLORS["text"])],
        bordercolor=[("focus", COLORS["accent"])],
```

The combobox also receives explicit normal/readonly text and background colors ([lines 128–146](../gui/theme.py)). The slate theme uses dark text on light fields, preventing platform defaults from producing white-on-white inputs.

**Operational boundary:** these GUI changes do not alter sigma, optimizer, partition, attack timing, or scientific stopping criteria.

## 8. Metric storage and comparison reporting

### 8.1 Evaluate the fixed Node-9 reference correctly

Source: [basil_core/research_protocol.py](../basil_core/research_protocol.py) — **lines 557–568**.

```python
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
```

| Saved metric | Definition |
|---|---|
| `fullTestAccuracy` | Node 9 honest trained model on all 10,000 test images |
| `averageAccuracy` | Mean of the ten node test accuracies |
| `worstNodeAccuracy` | Minimum of the ten node test accuracies |
| `fullTestLoss` | Node 9 full-test cross-entropy |
| `fullTestPerClassAccuracy` | Ten Node-9 class accuracies |

No ensemble or averaged model is evaluated under the full-test label. Test metrics are observational; selection and training remain local-training-only.

### 8.2 Require real 100-round data and signed contrasts

Source: [reporting/iid_campaign_plots.py](../reporting/iid_campaign_plots.py) — **lines 16–23**.

```python
def convergence(metrics,*,attack_applicable):
    accuracy=np.asarray(metrics['fullTestAccuracy'])
    if accuracy.shape!=(100,) or not np.isfinite(accuracy).all():raise ValueError('A completed condition requires 100 actual finite rounds.')
    return dict(final_accuracy=float(accuracy[-1]),best_accuracy=float(accuracy.max()),best_round=int(accuracy.argmax()),
        mean_rounds_0_19=float(accuracy[:20].mean()),mean_rounds_20_99=float(accuracy[20:].mean()),
        attack_applicable=attack_applicable,mean_rounds_80_89=float(accuracy[80:90].mean()),
        late_round_mean=float(accuracy[90:100].mean()),late_round_change=float(accuracy[90:100].mean()-accuracy[80:90].mean()),
        final_worst_node_accuracy=float(metrics['worstNodeAccuracy'][-1]))
```

Source: [reporting/iid_campaign_plots.py](../reporting/iid_campaign_plots.py) — **lines 26–32**.

```python
def scientific_contrasts(stats):
    result={}
    for name,(left,right,meaning) in CONTRASTS.items():
        if left not in stats or right not in stats:continue
        result[name]=dict(comparison=f'{right} minus {left}',meaning=meaning,
            differences={key:stats[right][key]-stats[left][key] for key in ('final_accuracy','best_accuracy','late_round_mean','final_worst_node_accuracy')})
    return result
```

Plots reject fabricated/truncated horizons. Late windows describe the fixed run; they do not automatically extend it. Positive D−C means higher measured D accuracy, not guaranteed benefit or statistical significance.

### 8.3 Added read-only post-run verifier

Source: [scripts/analyze_completed_iid.py](../scripts/analyze_completed_iid.py) — **lines 35–46**.

```python
def accuracy_summary(a):
    a = np.asarray(a, dtype=float)
    if a.shape != (100,) or not np.isfinite(a).all():
        raise ValueError('Expected 100 actual finite round measurements')
    return dict(final=float(a[-1]), best=float(a.max()), bestRound=int(a.argmax()),
                round49=float(a[49]), mean0_19=float(a[:20].mean()),
                mean20_79=float(a[20:80].mean()), mean20_99=float(a[20:].mean()),
                mean80_89=float(a[80:90].mean()), mean90_99=float(a[90:].mean()),
                lateChange=float(a[90:].mean()-a[80:90].mean()),
                lateSlope=float(np.polyfit(np.arange(80, 100), a[80:], 1)[0]),
                round19=float(a[19]), round20=float(a[20]), round21=float(a[21]),
                maxPostDropFromRound19=float(a[19]-a[20:].min()))
```

Source: [scripts/analyze_completed_iid.py](../scripts/analyze_completed_iid.py) — **lines 57–59**.

```python
def corrupted_selection(record):
    # Sender identity alone is insufficient: source round determines activation.
    return bool(record['selectedSenderByzantine'] and record['inputSnapshotRound'] >= 20)
```

This analysis module imports no TensorFlow, performs no SGD and launches no workers. It distinguishes a designated attacker from an **actually corrupted source snapshot**: source round, not receiver round, determines whether the delayed attack was active.

## 9. Mathematical and analysis tests

### 9.1 Full-objective gradient contract

Source: [tests/test_iid_study.py](../tests/test_iid_study.py) — **lines 99–121**.

<details>
<summary>Show actual code: EBM analytical and finite-difference checks</summary>

```python
def test_ebm_matches_hessian_vector_product_and_objective():
    model=tiny_model(); x=tf.constant([[.3,-.2],[.1,.6]]); y=tf.constant([0,1]); c=.02**2
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce=lossFn(y,model(x))
        g=inner.gradient(ce,model.trainable_variables)
        norm=tf.add_n([tf.reduce_sum(v*v) for v in g])
    correction=outer.gradient(norm,model.trainable_variables)
    robust,ce2,norm2,_,_=gradient_norm_objective_gradients(model,x,y,c)
    for expected,base,actual in zip(correction,g,robust):
        np.testing.assert_allclose(actual,base+c*expected,rtol=2e-6,atol=2e-7)
    assert float(ce2+c*norm2)==pytest.approx(float(ce+c*norm),rel=1e-6)
    # The correction is not a scalar learning-rate multiplier.
    assert any(not np.allclose(r,(1+c)*base,rtol=1e-6,atol=1e-7) for r,base in zip(robust,g))
    # Independent finite-difference check of CE + c||grad CE||².
    variable=model.trainable_variables[0]; original=variable.numpy(); delta=.001
    def objective():
        _,ce,norm,_,_=gradient_norm_objective_gradients(model,x,y,c)
        return float(ce+c*norm)
    changed=original.copy(); changed[0,0]+=delta; variable.assign(changed); plus=objective()
    changed=original.copy(); changed[0,0]-=delta; variable.assign(changed); minus=objective()
    variable.assign(original)
    assert float(robust[0][0,0])==pytest.approx((plus-minus)/(2*delta),rel=.005,abs=2e-4)
```

</details>

The test compares full EBM gradients against g + c∇‖g‖² and independently checks one gradient component by central finite difference. It also detects accidental replacement by (1+c)g.

### 9.2 No-lambda, strict loading and optimizer contract

Source: [tests/test_iid_study.py](../tests/test_iid_study.py) — **lines 124–143**.

<details>
<summary>Show actual code: No-lambda and SGD checks</summary>

```python
def test_no_lambda_loss_gradients_updates_and_strict_load():
    model=tiny_model((32,32,3)); worker=IidWorker(model)
    params=worker.export(); x=tf.ones([2,32,32,3])*.2; y=tf.constant([0,1]); key=tf.constant([1,2])
    outcomes=[]
    for legacy in (1.,999999.):
        worker.load(params)
        grads,values=worker._gradients(x,y,key,"gradient_norm_objective","paper_absolute_gaussian",tf.constant(.01),tf.constant(legacy))
        optimizer=tf.keras.optimizers.SGD(.05,momentum=0.)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        outcomes.append(([g.numpy() for g in grads],[float(v) for v in values],worker.export()))
    for a,b in zip(outcomes[0][0]+outcomes[0][2],outcomes[1][0]+outcomes[1][2]): np.testing.assert_array_equal(a,b)
    assert outcomes[0][1]==outcomes[1][1]
    worker.load(params)
    for a,b in zip(worker.export(),params): np.testing.assert_array_equal(a,b)
    final,record=worker.train(params,x.numpy(),y.numpy(),seed=2025,round_id=0,node_id=0,
        epochs=5,batch_size=512,learning_rate=.05,ebm_mode="none",semantics="paper_absolute_gaussian")
    assert record["steps"]==5 and all(e["samplesSeen"]==2 for e in record["epochs"])
    assert all(s["ebmCoefficient"]==s["ebmPenalty"]==s["ebmCorrectionNorm"]==0 for s in record["optimizerTelemetry"])
    assert int(worker.optimizer.iterations)==5 and float(worker.optimizer.momentum)==0
    assert worker.optimizer.weight_decay is None and worker.optimizer.clipnorm is None
```

</details>

Changing legacy lambda from **1** to **999999** must leave loss, gradients and parameter updates identical. The fixture also checks complete loading, five epochs, zero momentum, and no clipping/weight decay. Its two-image training fixture tests correctness; it is not a new scientific experiment.

### 9.3 New analysis-only regression tests

Source: [tests/test_completed_iid_analysis.py](../tests/test_completed_iid_analysis.py) — **lines 8–33**.

<details>
<summary>Show actual code: Read-only analysis regressions</summary>

```python
def test_round_windows_and_zero_based_round_50():
    a = np.arange(100, dtype=float)/100
    stats = accuracy_summary(a)
    assert stats['round49'] == .49
    assert stats['bestRound'] == 99
    assert stats['mean80_89'] == pytest.approx(.845)
    assert stats['mean90_99'] == pytest.approx(.945)
    assert stats['lateChange'] == pytest.approx(.1)
    assert stats['lateSlope'] == pytest.approx(.01)


@pytest.mark.parametrize('values', [np.zeros(99), np.full(100, np.nan)])
def test_incomplete_or_nonfinite_runs_cannot_be_reported_as_complete(values):
    with pytest.raises(ValueError):
        accuracy_summary(values)


def test_corruption_depends_on_source_round_not_receiver_round():
    assert not corrupted_selection(dict(selectedSenderByzantine=True, inputSnapshotRound=19, round=20))
    assert corrupted_selection(dict(selectedSenderByzantine=True, inputSnapshotRound=20, round=20))
    assert not corrupted_selection(dict(selectedSenderByzantine=False, inputSnapshotRound=20, round=20))


def test_absent_metric_is_not_fabricated_as_zero():
    assert summary([]) is None
    assert summary([0, 1])['mean'] == .5
```

</details>

These five collected tests cover actual 100-point requirements, zero-based round windows, missing/non-finite metrics, and source-round corruption classification.

| Verification | Previously executed result |
|---|---|
| Focused protocol / GUI-state / recovery / analysis suite | 158 passed; 2 display-dependent skips |
| Analysis-only suite | 5 passed |
| Compile check | Passed |
| Live GUI integration in this environment | Not verified; display connection unavailable |
| Entire legacy suite | Not claimed passed |
| GPU behavior | Representative existing verification read; no new benchmark in this documentation update |

Exact earlier commands and environment qualifications are in the [results report](../newResults/IID/ABCD_100r_comparison_gpu/FINAL_ABCD_100R_ANALYSIS.md#verification-commands-and-actual-outcomes).

## 10. Other files and boundaries

The source excerpts above cover the executable changes most relevant to training and the requested GUI fixes. They are **not a complete Git chronology** and do not reproduce every line of every new module.

| File / directory | Role | Mathematical effect |
|---|---|---|
| `scripts/run_iid_condition.py` | Validates by default; execution requires explicit `--execute` | None until authorized execution |
| `scripts/run_iid_basil_pair.py` | Data/hash audit, worker, manifests, CSV, final weights | Records the executed protocol |
| `basil_core/research_audit.py` | Batch/data/gradient/parameter hashes; finite failures; epoch evaluations | Instrumentation, not a selection criterion |
| `basil_core/iid_device_transition.py` | Verified historical reconstruction and transfer provenance | Logical state restoration; no backend-equivalence claim |
| `basil_core/artifact_paths.py` | Rejects protected historical output destinations | Safety only |
| `basil_core/protocol_compatibility.py` | Explicit deprecated aliases; preserve compatible unknown fields | Naming migration, not metadata rewriting |
| `gui/state/`, `gui/services/` | Validation, persistence, events, lazy discovery | No research mathematics in widgets |
| `gui/views/`, `gui/network_view.py` | Workspace rendering and network replay | Display only |
| `gui/worker_pool.py` | Extra worker argument routing for retained legacy workers | Not the primary serial IID runner |
| `reporting/iid_study_plots.py` | Real trajectories and selection/noise/gradient diagnostics | Reads measured data |
| `README.md`, architecture docs | Current launch/workflow instructions | Documentation only |

### Added versus reused code

| Classification | Evidence |
|---|---|
| Added inside tracked files | `BasilPaperCifarModel`; one-class/full-batch helpers in `cifar.py` |
| Modified tracked GUI code | Compatibility shim, launcher import, network display, worker argument routing |
| New modules currently untracked | Explicit protocol, IID/GPU/runtime/checkpoint modules; state/services/views; reporting and tests |
| Reused, not newly rewritten | Existing IID algorithm, `trainer.py` CE, `attacks.py` Hidden transform, legacy algorithms |
| Added for the completed-results analysis | Read-only verifier, its five tests, reports and verification JSON |
| This follow-up | Documentation formatting and exact source excerpts only |

Git commit alone is insufficient because executed runs recorded a dirty worktree. Source SHA-256 fingerprints are the stronger implementation evidence. Current line numbers describe this working tree; dates/authorship of earlier uncommitted edits cannot be reconstructed reliably.

## 11. What this code says about the ~58% result

| Observation | What it supports | What it does not prove |
|---|---|---|
| Correct IID class coverage | No one-class partition mistake | Guaranteed 80–90% accuracy |
| A: 55.04% at R49 → 58.83% at R99 | Longer training helped | A hard 58% ceiling |
| Small pooled CNN | Plausible feature limitation | Architecture is the sole cause |
| Reset SGD and approved decay | Plausible slower optimization | A different optimizer would solve it |
| BASIL can select a non-latest branch | Some computed updates need not remain in the final lineage | Every non-nearest selection is harmful |
| D correction mean ≈1.84% of CE gradient | EBM is active but modest | Universal noise mitigation |
| D−C final +0.17 pp; worst-node −2.63 pp | Mixed EBM outcome across endpoints | Significant or universal improvement |

No protocol was retuned after viewing the results. The paused non-IID snapshot-anchor study is not part of this IID evidence.

### Integrity and documentation-only scope

The completed audit recorded **5,363 historical files unchanged**, and an expanded **5,813-file** raw-result/plot/archive scan unchanged. Its machine-readable evidence is [POST_RUN_VERIFICATION.json](../newResults/IID/ABCD_100r_comparison_gpu/POST_RUN_VERIFICATION.json).

This follow-up edits only this report and the companion results Markdown. It does not edit source code, queues, saved configurations, raw metrics, checkpoints, PNGs, or protected artifacts.

### Checks performed for this documentation update

| Check | Result |
|---|---|
| Python excerpts versus current source | 38 of 38 match verbatim, including line ranges |
| Local file/image links across both reports | 115 checked; no missing targets |
| Tables, code fences and expandable sections | Structurally checked; no errors |
| Inspected source files | 28 unchanged |
| Analysis-only regression tests | 5 passed in 0.15 s; no research training |
| Protected `results4` / `plots4` | 2,153 / 737 files; before/after SHA-256 manifests match |

Command rerun for this update:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q -p no:cacheprovider tests/test_completed_iid_analysis.py
```
