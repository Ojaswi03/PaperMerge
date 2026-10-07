# Paired IID BASIL reference

`sequential_basil_iid_v1` is separate from `sequential_basil_one_class_v1`.
Only two research conditions are authorized:

| Setting | Test 1 | Test 2 |
|---|---|---|
| Channel | Clean | Absolute Gaussian, fixed coordinate sigma_e |
| Objective | CE | CE + sigma_e² ||grad CE||² |
| Legacy lambda | Unused | Unused, including if present in a loaded config |

Everything else is paired: seed 2025, original CIFAR partition, initialization,
four attacker IDs `[0, 1, 5, 7]`, Hidden transformation, augmentation and data-order
streams, five-epoch training, and ring/selection semantics. The existing IID
partitioner was already correct; it is reused, not replaced. The shared engine's
prior `permutation` split and the reused partitioner's `shuffle` split are
byte-identical with the same keyed stream.

Each node has 5,000 disjoint training examples representing all ten classes.
Each activation performs five complete epochs, ten batches per epoch, including
392 examples in the final batch. The approved 117,706-parameter CNN and reset
SGD use initial LR 0.05, round decay 0.05/(1+0.05r), no momentum, clipping or
weight decay. The current explicit authorization is 50 full rounds,
researchValid=true. The separate one-class protocol remains 100 rounds.

## BASIL and source interpretation

The BASIL source's **Equation (3)** selects the minimum expected receiver-local
loss; it is not an “Algorithm 3.” Our implementation approximates this with one
deterministic, unaugmented receiver training batch, shared across all five
candidates. Sender telemetry and global test metrics do not participate.
Ties within 1e-8 select the nearest counterclockwise sender. No averaging occurs.

The memory rollover is:

```text
Before Node 0, round 11: 5:R10 6:R10 7:R10 8:R10 9:R10
After Node 0's delivery: 6:R10 7:R10 8:R10 9:R10 0:R11 (Node 1)
```

The selected weights entirely initialize local training. Outbound Hidden attack
begins at round 20; independent link noise, when enabled, begins at round 0 and
is applied **after** attack. Senders refresh five clockwise memories; only the
immediate successor becomes the next active trainer. Our approved Hidden
simulation trains the Byzantine participant honestly before corrupting outbound
parameters; this is a project attack model, not a claim of exact source attack reproduction.

We adapt Ang et al.'s proposed gradient-norm regularization to multiclass
logits-based CE, using nested tapes to compute `g + 2 sigma_e² H g` without a
full Hessian. Source Eq. (14) is a squared-loss approximation, not a general
expected-CE identity. Its Eq. (23) scalar simplification is not a CNN identity.
See [the source equations](SOURCE_EQUATIONS.md),
[BASIL](../papers/001-Basil%20A%20Fast%20and%20Byzantine-Resilient%20Approach%20for%20Decentralized%20Training.pdf),
and [noisy communication](../papers/002-Robust%20Federated%20Learning%20with%20Noisy%20Communication.pdf).
No handwritten scan was supplied separately; the task's transcribed expression
was cross-checked against BASIL Eq. (3).

## Execution and outputs

```bash
python scripts/run_iid_basil_pair.py --run --rounds 50 --sigma-absolute 0.010
```

Test 2's sigma_e=0.010, variance=0.0001 is approved a priori from calibration,
not selected by test accuracy. `--run` checks pairing and finite first updates,
then executes Test 1 followed by Test 2 without another three-round preflight.
Existing results require explicit `--replace-iid-outputs`, which archives only
the six named IID directories in `/tmp` before replacement.

Full-test accuracy/loss evaluate the fixed end-of-ring Node 9 honest trained
model on all 10,000 test images before outbound noise. Mean/worst-node accuracy
summarize all ten individual node models; there is no parameter averaging.
Per-class trajectories use Node 9; complete node-by-class metrics are retained.
Round CSVs retain full-test CE and EBM/channel aggregates. Plots use zero-based
rounds 0–49, with attack activation marked at 20.

Results use `newResults/IID/<test>/`; plots use `newPlots/IID/<test>/`.
`preflight/` and `smoke/` children are marked researchValid=false. Full-data
preflights retain onset 20 and hence do not exercise attack-period training in
three rounds. Reduced-data smoke explicitly uses onset 1 to exercise attacks.
The `nonIID/` trees remain empty.

Primary execution is single-process CPU, oneDNN off, deterministic operations,
one intra/inter-op thread. Unresolved multiprocessing gradient discrepancies are
not used as scientific evidence. GPU numerical behavior is unverified.
Batch traces and checkpoints preserve numerical failures; non-finite logits
never become an accuracy number. Diagnostic-only norms do not control channel
sigma or training. No anchors, CART, MC objective or new non-IID loss is used.

The two conditions change both noise and loss. A future matched noisy-channel
CE-only control is necessary to isolate EBM's causal contribution; it is not
automatically run. Read the generated
[study summary](../newResults/IID/IID_TWO_TEST_SUMMARY.md) for measured outcomes,
pairing evidence, exact test commands and artifact integrity.

## Completed 50-round pair

Both full-data runs completed with finite training and 120 attack-active node
activations each. The fixed Node 9 reference finished at 54.35% for clean CE
and 50.57% for noisy EBM; mean-node accuracy was 54.146% and 50.667%, and
worst-node accuracy was 53.24% and 49.34%, respectively. These are descriptive
single-seed results, not an isolated estimate of EBM benefit.

Each run contains 500 activations and 25,000 optimizer batches. All 25,000
paired data/augmentation hashes match, as do initialization and partition
hashes. Neither run selected an actually attack-corrupted snapshot. Across
the six historical result/plot trees, all 5,363 file hashes remain unchanged.

Recheck the completed research artifacts without training:

```bash
python scripts/verify_iid_research_outputs.py --write-evidence
```

The checker verifies full epochs, strict-handoff parameter hashes, the actual
Round-11 memory rollover, partition coverage, threat schedules, EBM variance,
round metrics, recorded source hashes and paired batch/augmentation streams.
