# Local-only Dirichlet BASIL: protocol and thesis attribution

## Completed 30-round feasibility study

The study comprises **four fresh 30-round feasibility runs**, using
the existing CIFAR-10 Dirichlet partitioner with **α=0.2 and seed 2025**.
They are not extensions of one-class or IID runs. All four subsequently completed
30 rounds and 300 activations on GPU; the queue is now idle. No research run was
launched during configuration preparation. W&B was disabled. `researchValid=false` means a bounded
feasibility study, not publication-ready validation or a promised accuracy.

| Condition | Actual attackers | Channel | Local objective |
|---|---|---|---|
| A | None | Clean | CE + frozen snapshot anchor |
| B | `[0,1,5,7]`, Hidden from R20 | Clean | CE + frozen snapshot anchor |
| C | Same as B | Absolute Gaussian σₑ=0.010 | CE + frozen snapshot anchor |
| D | Same as B | Same as C | CE + full EBM + frozen snapshot anchor |

All four retain the 117,706-parameter CNN, strict ring handoff, S=5 memory,
receiver-local minimum-CE Snapshot Selection, five full local epochs, batch 512,
reset SGD with momentum zero, initial LR 0.05 and the existing
`ηᵣ=0.05/(1+0.05r)` schedule. No clipping, weight decay, CART helper gradients,
prototype replay, ACDS, raw-data sharing or replicated TRAIN coordinator is used.
The older CART implementation remains available as a **separate method**.

| Condition | Final Node-9 accuracy | Best Node-9 accuracy | Final node mean | Final worst node | Recorded runtime |
|---|---:|---:|---:|---:|---:|
| A | 24.34% | 26.41% | 22.79% | 16.53% | 21.1 min |
| B | 10.00% | 22.84% | 10.00% | 10.00% | 22.5 min |
| C | 10.00% | 31.06% | 10.00% | 10.00% | 23.1 min |
| D | 31.56% | 33.47% | 21.16% | 10.00% | 22.5 min |

Values were cross-checked against saved CSV/NPZ data. Every condition has 30
contiguous measured round points and 45 generated PNGs; no worker exception was
recorded. Final Node-9 confusion matrices show B/C predicted bird on all 10,000
test images. D retained some learning but three node models finished at 10%.
These outcomes are not sufficient to claim that non-IID retention or Byzantine
filtering is solved. No new horizon, model or objective was selected to improve
these completed results. A later diagnosis must preserve this evidence.

## Exact selection and loss

For active receiver i, use one deterministic, unaugmented local TRAIN batch
`Bᵢˢˢ` (up to 512 images), identical for every candidate:

The selection batch samples without replacement from the receiver's partition,
keyed by `(seed, snapshot_selection_batch, node)`. It is not the first 512
class-grouped indices returned by the Dirichlet allocator. All candidate
evaluations still share exactly that one local batch, without augmentation.

$$
\theta_{\mathrm{sel},i}=\arg\min_{\theta_j\in\mathcal M_i}
CE_i(\theta_j;B_i^{SS}).
$$

Ties choose the nearest counterclockwise sender. Sender telemetry and the global
test set never choose the winner. Fully load that one received snapshot; do not
average it with other candidates or a stale local model. Freeze its parameters
as the reference throughout all five epochs.

At **each optimizer step**, with current trainable parameters θ:

$$
\boxed{L_i(\theta)=
\underbrace{CE_i(\theta;B_i)}_{\text{receiver-local learning}}+
\underbrace{\sigma_e^2\|\nabla_\theta CE_i(\theta;B_i)\|_2^2}_{\text{EBM: D only}}+
\underbrace{\frac{\mu}{2}\|\theta-\theta_{\mathrm{sel},i}\|_2^2}_{\text{snapshot anchor: A/B/C/D}}.}
$$

Here μ=0.01 is fixed before results. The EBM coefficient in D is exactly
σₑ²=0.0001, with **no lambda**. C has actual noise but **no EBM**. A/B use
zero EBM coefficient. `θ` is the current state during training, not a detached
final-after-training state: the anchor must affect every SGD update.

Writing g=∇CE and H=∇²CE, D uses:

$$\nabla L_i=g+2\sigma_e^2Hg+\mu(\theta-\theta_{\mathrm{sel},i}).$$

Nested TensorFlow tapes calculate the EBM correction without materializing H.
The anchor reference is detached and never updated during the activation.
Neither term guarantees preservation of all classes. Anchoring limits parameter
drift; it does not supply missing examples or certify a globally good snapshot.

```text
latest five predecessors → receiver-local CE selection → full snapshot load
                                                        ↓
                                       freeze selected snapshot as anchor
                                                        ↓
                              five full local epochs → honest trained model
                                                        ↓
                                 Hidden outbound corruption, if active
                                                        ↓
                             independent Gaussian draw per directed link
                                                        ↓
                      refresh next five memories; only next node trains
```

For C/D the channel is `θ̃ᵢ→ⱼ=θᵢ+ξᵢ→ⱼ`, with
`ξᵢ→ⱼ ∼ N(0, σₑ² I)` independently keyed by seed/round/sender/receiver.
σₑ=0.010 is a **coordinate standard deviation, not a percentage**. No model-L2
normalization controls this channel. Model/noise norms are telemetry only.
The existing relative-L2 branch remains available outside this study.

## Who deserves credit

| Component | Primary source | Exact relationship to this implementation |
|---|---|---|
| Ring, rolling predecessor memory and minimum receiver-local expected loss | [Elkordy, Prakash & Avestimehr, BASIL](https://arxiv.org/abs/2109.07706), Eq. (3) and Algorithm 1; local `papers/001-…pdf` | Reuses the selection mechanism with a local minibatch approximation and strict handoff. S=b+1=5 is the fixed study bound. |
| Gaussian-channel gradient-norm regularizer | [Ang et al., Robust Federated Learning with Noisy Communication](https://arxiv.org/abs/1911.00251), Eqs. (13), (14), (15b), (22), (23); local `papers/002-…pdf` | Adapts the gradient-norm objective to multiclass CNN cross-entropy and a decentralized ring; not the paper’s original objective, aggregation or CNN. |
| Quadratic proximal penalty | [Li et al., Federated Optimization in Heterogeneous Networks](https://proceedings.mlsys.org/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf), Eq. (2); local `papers/018-…pdf` | Same proximal form as FedProx; our reference is the **selected received snapshot**, not the central server model. This is an adaptation, not a claim that proximal regularization is new. |
| Per-class Dirichlet allocation, unequal sizes and absent labels | [Yurochkin et al., Bayesian Nonparametric Federated Learning of Neural Networks](https://proceedings.mlr.press/v97/yurochkin19a/yurochkin19a.pdf), §4.1, PDF p.6 | Same kind of label-wise allocation. Their example uses α=0.5; this study fixes α=0.2 and deterministic integer splits. Their neural matching algorithm is **not** used. |
| Additional heterogeneous CIFAR partition context | [Hsu, Qi & Brown, Measuring the Effects of Non-Identical Data Distribution](https://arxiv.org/abs/1909.06335) | Relevant Dirichlet non-IID study to cite; not proof that our exact split, method or results reproduce theirs. |

Primary PDFs already supplied were read alongside online author/publisher
sources; secondary search snippets were not used as authority for equations.
Existing source PDFs were not changed.

### Source equations versus our adaptation

The supplied Ang PDF states:

| Equation | Mathematical content | What must not be inferred |
|---|---|---|
| (13), repeated as (22) | `Fⱼᵉ(w)=Fⱼ(w)+σₑ²‖∇Fⱼ(w)‖²` | Substituting CIFAR CE is our adaptation. |
| (14) | `E‖Fⱼ(w+w̃ⱼ)‖² ≈ E‖Fⱼ(w)‖² + σₑ²‖∇Fⱼ(w)‖²`, through first-order Taylor approximation | This is a **squared-loss expression**, not a general identity for expected multiclass CE. |
| (15b) | `wⱼᵗ⁺¹=wᵗ−η∇Fⱼᵉ(wᵗ)` | The decentralized strict ring is not the paper’s server aggregation. |
| (23) | The source simplifies the derivative to `(1+σₑ²)∇Fⱼ` | This simplification is **not** the general derivative of a nonlinear CNN gradient-norm objective. Legacy scaling stays separate. |

For differentiable CE, the actual gradient-norm derivative is `2H g`.
Furthermore, a second-order Taylor approximation of **expected noisy CE** is
`E[CE(θ+ξ)]≈CE(θ)+(σₑ²/2)tr(H)`, under the necessary smoothness assumptions.
It is not generally `CE+σₑ²‖g‖²`. We retain the requested gradient-norm loss,
but call it **paper-inspired gradient-norm regularization**, not an exact
evaluation of expected noisy CNN CE. ReLU/max-pool kinks also qualify smooth
Hessian identities; the software contracts test differentiable toy cases.

Suggested thesis wording:

> We adapt the expectation-based gradient-norm regularization of Ang et al.
> to multiclass CIFAR-10 cross-entropy. We use a FedProx-style quadratic
> proximal penalty referenced to the BASIL-selected received model to
> investigate local drift under Dirichlet label heterogeneity. The combined
> objective is a study adaptation, not a claimed new proximal mechanism or
> a proof of robustness or class retention.

BASIL’s non-IID extension uses **ACDS**; this local-only study does not. Its
IID/convex guarantees cannot simply be transferred to this Dirichlet CNN.
The current project Hidden transform is retained by explicit choice and is
**not claimed as an exact reproduction of the source attack**. Gaussian
communication noise is not the Gaussian Byzantine attack.

## Actual full-data split

All 50,000 TRAIN images are assigned exactly once; the 10,000 TEST images
remain evaluation-only. α=0.2 does **not** guarantee all labels at each node.
The existing partitioner was reused unchanged; no balancing/resampling was
introduced to improve accuracy.

| Node | Samples | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | Batches/epoch | Steps/activation | Last batch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2938 | 3 | 0 | 0 | 1147 | 3 | 3 | 16 | 0 | 1 | 1765 | 6 | 30 | 378 |
| 1 | 4407 | 289 | 0 | 49 | 158 | 2356 | 0 | 719 | 127 | 0 | 709 | 9 | 45 | 311 |
| 2 | 4768 | 733 | 1340 | 0 | 73 | 634 | 550 | 2 | 538 | 3 | 895 | 10 | 50 | 160 |
| 3 | 989 | 28 | 177 | 1 | 198 | 20 | 17 | 0 | 104 | 262 | 182 | 2 | 10 | 477 |
| 4 | 9660 | 4 | 1370 | 1659 | 1226 | 1 | 1740 | 179 | 3468 | 9 | 4 | 19 | 95 | 444 |
| 5 | 6693 | 1332 | 387 | 659 | 135 | 1247 | 2216 | 276 | 205 | 92 | 144 | 14 | 70 | 37 |
| 6 | 2945 | 670 | 1297 | 70 | 53 | 264 | 1 | 47 | 0 | 450 | 93 | 6 | 30 | 385 |
| 7 | 5622 | 652 | 78 | 7 | 1360 | 298 | 398 | 2349 | 1 | 0 | 479 | 11 | 55 | 502 |
| 8 | 4115 | 299 | 261 | 199 | 0 | 0 | 29 | 9 | 177 | 3141 | 0 | 9 | 45 | 19 |
| 9 | 7863 | 990 | 90 | 2356 | 650 | 177 | 46 | 1403 | 380 | 1042 | 729 | 16 | 80 | 183 |

Each ring traversal now contains **510 optimizer steps**, not 500. Five epochs
and every partial batch remain intact. All four conditions have the same
partition and canonical initialization hashes:

```text
partition: 2c792027992625ff914827842fc41e27a88411c0ae0fcc4ef900160ec5463f1e
initial:   7c717dbbd9bd1f2f230b1ff3d288bd8fec33dcd3ec730eb43e8a14e496cdc2df
```

## Running and recovery

```bash
python run_gui.py
```

Open Queue, confirm four **Dirichlet α=0.2, 30 rounds, GPU** entries in A/B/C/D
order, then click **Run Queue**. If needed, **Load queue** selects
`gui/queues/noniid_dirichlet_alpha02_abcd_30r_gpu.json`.
Do not load the older verified-CART/one-class preset for this study.

Results go to `newResults/nonIID/dirichlet_alpha02_local_anchor_30r_v1/`;
plots go to the corresponding `newPlots/nonIID/` namespace. Existing outputs
are never overwritten. Same-device checkpoints restore all models, rolling
memories, telemetry and cursor; an interrupted activation is discarded and
recomputed from its selected input (optimizer state resets per activation).
This is **activation-boundary**, not mid-minibatch, resume. Changed code/config
or corrupt evidence blocks resume instead of guessing. Older checkpoints are
not migrated to Dirichlet and their historical source fingerprints remain intact.

The GPU cap is 6 GiB with a startup free-memory gate. Tests cannot promise that
unrelated desktop/GPU processes never cause OOM. No 30-round learning success
has been claimed from synthetic compatibility checks.

## Thesis bibliography

```bibtex
@misc{elkordy2021basil,
  title={Basil: A Fast and Byzantine-Resilient Approach for Decentralized Training},
  author={Elkordy, Ahmed R. and Prakash, Saurav and Avestimehr, Salman},
  year={2021}, eprint={2109.07706}, archivePrefix={arXiv}
}
@article{ang2020noisy,
  title={Robust Federated Learning With Noisy Communication},
  author={Ang, Fan and Chen, Li and Zhao, Nan and Chen, Yunfei and Wang, Weidong and Yu, F. Richard},
  journal={IEEE Transactions on Communications}, year={2020}, volume={68},
  number={6}, pages={3452--3464}, doi={10.1109/TCOMM.2020.2979149}
}
@inproceedings{li2020fedprox,
  title={Federated Optimization in Heterogeneous Networks},
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  booktitle={Proceedings of Machine Learning and Systems}, year={2020}
}
@inproceedings{yurochkin2019bnp,
  title={Bayesian Nonparametric Federated Learning of Neural Networks},
  author={Yurochkin, Mikhail and Agarwal, Mayank and Ghosh, Soumya and Greenewald, Kristjan and Hoang, Trong Nghia and Khazaeni, Yasaman},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  year={2019}, volume={97}, pages={7252--7261}, series={Proceedings of Machine Learning Research}
}
@misc{hsu2019heterogeneity,
  title={Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification},
  author={Hsu, Tzu-Ming Harry and Qi, Hang and Brown, Matthew},
  year={2019}, eprint={1909.06335}, archivePrefix={arXiv}
}
```

These credits identify mechanism antecedents, not an exhaustive priority/novelty
review. No claim is made that this exact combination has never appeared before.

## Previous CART mechanisms are separate

The downloaded prototype/continual-learning papers and their detailed
[attribution review](CART_LITERATURE_AND_ATTRIBUTION.md) remain intact.
FedProto informs prototype alignment; PASS/FeTrIL/FedProK inform feature
rehearsal; Learning without Forgetting informs teacher-output distillation.
Those mechanisms must be credited if reintroduced. They are **not terms in the
current local-only loss**, and no performance from the earlier helper/replicated
TRAIN implementation is evidence for this new protocol. A registry alone does
not create missing-class supervision or verify Byzantine honesty.

## Completed engineering checks — 13 October 2026

| Check | Actual command / scope | Observed result |
|---|---|---|
| Compilation | `environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests` | Exit 0 |
| Full CPU + Xvfb suite | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/papermerge-test-mpl xvfb-run -a environment/basil-noise-env/bin/python -m pytest -q tests` | **700 passed**, **644 subtests passed**, 3 warnings, 148.26 s; no failures/skips |
| Final targeted changes | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 environment/basil-noise-env/bin/python -m pytest -q tests/test_live_tracking.py tests/test_dirichlet_study.py` | **44 passed**, 2 warnings, 7.56 s; overlapping tests, not 44 additional independent tests |
| Lazy dependencies | `environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py` | 270 source files; no movable unconditional heavy imports |
| CPU/GPU objective + update contracts | `environment/basil-noise-env/bin/python scripts/verify_iid_gpu.py` | Passed on RTX 4070 Ti, TF 2.18.0, 6 GiB cap, TF32 off; analytic gradient errors below 5×10⁻⁸ |
| Frozen anchor + variable partial batches | `environment/basil-noise-env/bin/python scripts/verify_snapshot_anchor_gpu.py` | Passed CE/anchor and EBM/anchor; all actual Dirichlet partial sizes plus 392/512 finite; λ=1 vs 999999 identical; reference unchanged |
| Full cached data + GPU startup | `environment/basil-noise-env/bin/python scripts/run_dirichlet_condition.py --config gui/configs/nonIID/dirichlet_alpha02_30r/<A/B/C/D>_dirichlet_alpha02_local_anchor_30r_gpu.json --validate-runtime` (executed separately for each condition) | A/B/C/D passed, **zero optimizer updates**, current source hashes match; real receiver-local TRAIN probes finite; D coefficient 9.999999747378752×10⁻⁵ (float32 representation of 0.0001) |
| GUI startup | Actual Tkinter app under Xvfb, execution hook patched to reject starts | Four pending 0/30 entries; D preset loads; queue idle; no worker started |
| Logical ring + resume | New `test_local_selection_handoff_memory_attack_and_stateful_resume`, using a deterministic fake worker | Round-11 rolling memory correct; no TEST selection; full input hashes match; attack before noise at R20; interrupted/restored logical history identical |
| Evidence SHA-256 | Reused `scripts/verify_saved_evidence.py` inventory with the five trees recorded in `gui/worker_state/dirichlet_alpha02_before.json` | **24,722 before / 24,722 after / 24,722 matched / 0 changed** |

The first unrestricted pytest command did not collect tests because an external
ROS plugin lacked `lark`; disabling external plugin autoload resolved that
environment issue. The first full run had one dry-run test failure because it
read the real stopped one-class A directory instead of a temporary root. The
test was isolated; saved research evidence and recovery checks were not weakened.
Final results above are the completed reruns, not invented counts.

GPU synthetic anchor checks peaked at **363,078,144 bytes**; real-data D startup
peaked at **336,274,432 bytes**. These are bounded checks, not a guarantee of
500-round memory behavior, successful learning or CPU/GPU bitwise equivalence.
No 30-/100-/500-round research process was started during preparation, and W&B remained disabled.
Validation JSON and logs live under ignored `gui/worker_state/`, not research
output folders or thesis result tables. The subsequent measured 30-round outcomes
are recorded at the top of this guide; correctness does not establish adequate learning.

## GUI startup-event correction

The worker's `configuration_valid` message contains a recovery **plan**
(`mode`, `startActivation`, `completedRounds`), not replay-progress counts
(`completed`, `total`). The GUI formerly confused these and raised
`KeyError: 'completed'`, terminating its polling callback while the independent
GPU worker continued. The adapter now distinguishes the two payloads, the
consumer rejects malformed progress messages, and polling reschedules even
after a display-update exception. The targeted GUI/recovery suite completed
**64 passed**, 2 warnings, in 10.91 seconds. Compilation passed. No scientific
worker source, saved checkpoint, hyperparameter or active process was changed.

An already open GUI does not hot-reload Python source. If its polling stopped
at this exception, let its active worker finish before restarting the GUI to
load the fix. The completed condition is recognized from its saved manifest;
remaining queued conditions must be started using **Run Queue**. Do not close
or restart the GUI mid-training just to refresh the display.

## Cleanup

Repository-root ZIPs and local context prompts were removed from the working
tree. No source PDF or current saved research artifact was deleted during this
protocol change. Publication hooks reject context/catch-up prompts and session
handoff documents, including force-added files; scientific protocol guides
remain allowed. Generated outputs, credentials and machine-local queue/runtime
state remain outside the source publication.
