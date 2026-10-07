# Research sequential CIFAR-10 protocol

The five-epoch amendment is authoritative. No one- or three-epoch configuration
is part of this protocol. Earlier reduced-epoch smoke outputs, where present,
are historical implementation artifacts and are not overwritten or used in the
amended report.

## Fixed training contract

Ten logical nodes start from identical keyed initial weights. Node `i` owns all
5,000 training images of class `i`: airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, truck. Training index sets are disjoint and cover 50,000
images. The balanced 10,000-image global test set is evaluation-only.

One communication round is the complete traversal `0 → 1 → … → 9 → 0`.
`strict_sequential_handoff` completely replaces the active node's training
weights with its selected snapshot; it never averages with the old local model.
With SS off, Node 1's incoming hash matches Node 0's transmitted hash, and Node
0 next round matches Node 9's last transmitted hash. In a clean run these also
match the predecessor's honest trained-output hashes.

Each activation uses **five complete local epochs**, batch 512. Each full epoch
has nine 512-image batches and one 392-image batch. Thus there are 50 optimizer
steps/activation and 50,000 steps/100-ring-round production run. SGD is freshly
created per activation, with zero momentum, no clipping and no weight decay.
Only model weights travel. Learning rate for ring round `r` is
`0.05 / (1 + 0.05*r)`. Data order and individual-image flip/pad/crop augmentation
are keyed by experiment, round, node, epoch and batch/sample visit. Global test
and local candidate-selection images are normalized but unaugmented.

The BASIL paper CNN has 117,706 parameters; source dimensions and initialization
are documented in [the equation note](SOURCE_EQUATIONS.md). Historical
VGG, fixed-step loaders, consensus and CART engines retain their meanings.

## Rolling memory, not five snapshots from one sender

With assumed Byzantine count 4, memory size `S=5`. Every sender transmits to its
next five clockwise receivers, independently applying channel noise to each
link. Only the immediate receiver becomes the next active trainer.

```text
                             next active trainer
                                     |
Node 0 -- trained snapshot -------> Node 1
        +-- memory refresh -------> Node 2
        +-- memory refresh -------> Node 3
        +-- memory refresh -------> Node 4
        +-- memory refresh -------> Node 5

Before Node 0 in Round 11:   [5:R10, 6:R10, 7:R10, 8:R10, 9:R10]
After Node 0 sends,
before Node 1 in Round 11:   [6:R10, 7:R10, 8:R10, 9:R10, 0:R11]
```

Each memory holds the latest received snapshot from each distinct permitted
sender. Initial memory entries contain W0 with snapshot round -1.
`lowest_receiver_local_loss` evaluates all five candidates on the **same**
deterministic local batch, at most 512 examples. Minimum CE wins; ties within
`1e-8` go to the nearest counterclockwise sender. Sender-reported metrics are
telemetry only. No global test metric controls this selection.

## Channel and EBM are distinct

`paper_absolute_gaussian` transmits `w + epsilon` with independent coordinate
standard deviation `channelNoiseSigmaAbsolute`. No model normalization is used.
Its full EBM coefficient is exactly `sigma_e²`.

`relative_l2_gaussian` uses `channelNoiseSigmaRelative` and actual coordinate
standard deviation `s = sigma_rel * ||w|| / sqrt(d)`. Its full EBM coefficient
is `stop_gradient(s²)`, recalculated from the current weights **each optimizer
step**, not once per activation. The model L2 norm calibrates the channel; it is
not the regularization objective.

Full EBM is `CE + c*||grad CE||²`. Nested tapes differentiate this complete loss,
including Hessian-gradient information. `legacy_gradient_scale` instead scales
ordinary gradients by `1 + legacyEbmLambda*sigma²`; the compatibility configs
explicitly use lambda 25. Research full-objective configs contain no legacy
lambda. Exact source equations 14, 15b and 23 and their mathematical distinction
are in [SOURCE_EQUATIONS.md](SOURCE_EQUATIONS.md).

At keyed initialization, `d=117706`, model norm is approximately 15.0794.
Twenty calibration draws give relative ratios 0.200003, 0.400073, 0.599158.
Absolute values 0.005/0.01/0.02 give ratios 0.113792/0.227329/0.454846. These
span moderate perturbation magnitudes without selecting on final accuracy.
Absolute 0.2/0.4/0.6 instead give 4.548/9.098/13.649 times model norm, and are
not production sweeps. Calibration `--evaluate` reports full-test before/after
accuracy; these initialization scores are not evidence of trained robustness.
`--checkpoint node_0_weights.npz` supports an actual trained checkpoint.

## Threat order and paired reproducibility

Receive/cache → receiver-local SS (if enabled) → load complete weights → reset
SGD → train five epochs → honest evaluation → optional outbound hidden attack
→ independent per-link channel noise → next-five memory delivery.

`delayed_hidden_parameter_attack_v1` calls the existing repository hidden
parameter transformation without a malicious reference model. It is honestly
labeled as repository behavior, not claimed to reproduce another attack paper.
Four IDs are chosen without replacement once per seed. Seed 2025 resolves
`[0,1,5,7]`; all paired configs persist these same IDs. Attack starts at production
round 20; channels start at round 0. Separate SHA-256-derived streams cover
initialization, partition/order, augmentation, attacker choice, attack and
channel. Channel keys include seed, round, sender and receiver; mitigation
switches do not advance a shared random stream. Paired equal upstream weights
therefore receive the same underlying noise draw. Different upstream norms can
scale that draw differently in the relative branch.

The old `absolute_coordinate_gaussian` and `channelNoiseSigmaRel` names remain
accepted read adapters. New configs use the amendment's explicit names; old
result provenance is not normalized.

## Workflow and scientific gate

1. Reconstruct the faithful protocol and verify contracts, including finite
   differences for second-order EBM and measured handoff/memory invariants.
2. Run all 47 three-round five-epoch smoke configs, using 512 samples/class and
   100 evaluation images/class. Only smoke configs shorten attack start to 1.
3. Run clean IID, Dirichlet alpha 0.2 and one-class **full-data five-round**
   preflights. Every one-class activation still performs 50 updates.
4. Review the per-class activation and epoch heatmaps before approving the
   100-round threat matrix. No production run starts automatically.
5. Consider a separately labeled CART/non-IID extension after this baseline is
   understood. Existing CART experiments are not silently mixed into the plain
   research baseline.

The measured five-round controls are IID 33.74% average / 31.87% worst,
Dirichlet 15.18% / 10.77%, and one-class 10.00% / 10.00%. The clean one-class
node-by-class states are own-class-only: five epochs overwrite previously learned
classes. This supports catastrophic forgetting under this partition, not a
broken learning harness. Threat runs near chance cannot independently establish
SS/EBM effectiveness. Four high-relative-noise smoke conditions diverged; their
failed status and full traceback are retained. No rescue clipping, fewer epochs,
new loss or test-driven tuning was introduced.

An isolated reference invocation of relative sigma 0.4/full EBM completed near
chance, while a subsequent normal-worker recheck failed again. All observations
remain recorded. Invocation/build sensitivity is observed, but its exact cause
is not isolated; production reproducibility/stability needs review, not an
assertion that this condition always diverges. New manifests record implementation SHA-256,
TensorFlow/NumPy versions, CPU thread limits and available device types. Older
non-research smoke manifests are not rewritten to invent missing source hashes.

All smoke and preflight outputs have `researchValid=false`. They are kept out
of production comparison groups. Failed runs do not get fabricated final scores.
Review [all conditions and measured results](RESEARCH_DIAGNOSTIC_REPORT.md).

## Results, telemetry and inference

New data writes only to `experiments/research_protocol_results/`; new plots go
to `plots/research_protocol/<revision>/<family>/`. All historical roots reject
new research writes. Existing runs are never overwritten. Queue/worker state
is separate from protected research artifacts. SIGTERM requests a stop between
optimizer batches, producing a stopped manifest; retries start a fresh run,
not an unsupported mid-activation scientific resume.

Every activation records input/trained/transmitted hashes, local samples and
batch sizes, five epoch losses, global and ten-class accuracies before and after
each epoch, candidate losses/rounds, attack state and relative change, per-link
noise norms/sigmas, and stepwise EBM coefficient/CE/gradient norm/objective.
`partial_activation_telemetry.jsonl` journals completed activations in new runs;
full completed traces are compressed JSON. Exceptions retain a technical log.

`metrics.npz` contains:

- `activationOverallAccuracy[round,node]`;
- `activationPerClassAccuracy[round,node,class]`, production shape `[100,10,10]`;
- `beforeTrainingAccuracy` and `beforeTrainingPerClassAccuracy`;
- `epochOverallAccuracy[round,node,epoch]`;
- `epochPerClassAccuracy[round,node,epoch,class]`, epoch axis length 5;
- round-average/worst/best, per-node and node-by-class arrays;
- selected sender and snapshot-round arrays.

Plots show average/worst, per-node curves, final/selected-round node-by-class
heatmaps, activation and five-epoch class trajectories, source selection,
candidate losses, model norm, realized noise, EBM coefficients/objectives,
attack-start markers, and paired mitigation comparison grouped by channel
semantics, sigma, partition, seed and revision. The 50% line is a research target.

The GUI exposes research settings, explicit noise labels, progressive sigma
controls and full-vs-legacy EBM. Results discovery is metadata-only; selecting a
run loads metrics. The existing ring renderer receives read-only research live
events and replay records through an adapter. Production Run remains gated.

```bash
python scripts/inference_demo.py PATH_TO_RUN --cifar-index 8
python scripts/inference_demo.py PATH_TO_RUN --image ship.png --label ship
```

The same evaluation-only image is sent through all ten final node checkpoints.
External images are RGB-resized to 32×32 and normalized to [0,1]. Without a
known label, correctness is not asserted. No demo image enters training.

## Verification

```bash
python -m compileall -q gui basil_core reporting scripts tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --ignore=tests/test_convergence.py
python scripts/verify_research_results.py PATH_TO_COMPLETED_RUN
xvfb-run -a python scripts/smoke_research_gui.py
```

The convergence script is excluded because it trains during collection and may
download MNIST. TensorFlow CPU diagnostics use cached CIFAR data; no large GPU
stack installation or 100-round matrix is required for these checks.
