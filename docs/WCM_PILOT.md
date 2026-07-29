# Worst-Case Model Pilot

## Status

WCM is an isolated research pilot. It is not connected to the GUI, Campaign 3,
the active queue, `experiments/results3`, or `plots3`.

No existing paper-derived component was changed:

- Snapshot Selection (SS) is unchanged.
- Expectation-Based Mitigation (EBM) is unchanged.
- CART is unchanged.
- WCM is tested as an alternative channel-noise mitigation, not as a renamed
  EBM implementation.

The implementation is based only on the local paper:

`Papers/002-Robust Federated Learning with Noisy Communication.pdf`

## Paper Mapping

The implementation follows Sections III-B and V of the paper:

| Paper item | Implementation |
|---|---|
| Bounded uncertainty and min-max objective, Eqs. (8)-(10) | One bounded uncertainty set over the flattened full model |
| Worst condition sampled on the boundary, Eq. (27) | `sampleBoundaryPayload()` |
| SCA surrogate, Eq. (31) | `scaSurrogateGradients()` |
| Recursive gradient estimate, Eqs. (32) and (34) | `updateGradientEstimate()` and per-node `WcmState` |
| Conditional model update, Eq. (36b) | Final reference-to-candidate blend in `wcmLocalUpdate()` |
| Step conditions in Lemma 7 | `rho_t=(t+1)^-beta`, `gamma_t=(t+1)^-alpha`, with `0.5 < beta < alpha < 1` |

The paper says to sample noise on the uncertainty boundary but does not specify
a unique directional distribution. The pilot uses a seeded isotropic direction
over the flattened full model.

## Explicit Adaptations

This repository is not the paper's original experimental system. The pilot
records these differences in every `run.json`:

| Paper | This pilot |
|---|---|
| Central server and weighted global aggregation | Decentralized sequential ring with the existing pairwise consensus |
| IID MNIST, binary SVM experiment | Non-IID CIFAR-10, 10-class CNN |
| Absolute spherical uncertainty radius | Existing project convention: `sigma * ||w||_2` full-model radius |
| Mathematical `arg min` of the SCA subproblem | Finite local SGD approximation using the configured local steps |
| Conditional-gradient update without SGD momentum | Plain inner gradient descent; Campaign3 momentum slots are not allocated |

These are adaptations, so a successful pilot supports WCM in this ring system;
it is not a reproduction of the paper's numerical experiment.

## Code

- `noise_comm/wcm.py`: paper equations and backward-compatible legacy API.
- `basil_core/wcm_pilot.py`: isolated ring integration using the existing
  consensus, SS, hidden attack, channel-noise, and CART paths.
- `scripts/run_wcm_pilot.py`: one-process runner and result writer.
- `tests/test_wcm.py`: equation, runner-safety, and tiny-ring tests.

## GPU And Memory Controls

- GPU mode refuses to start if `nvidia-smi` reports any active compute process.
- GPU mode also refuses to start if GPU occupancy cannot be verified.
- The default TensorFlow GPU limit is 3800 MB; the runner accepts at most
  4500 MB.
- Only one TensorFlow model is resident. Ten logical-node parameter and WCM
  gradient states are stored as CPU NumPy arrays.
- Activation work is microbatched at 128 while the experiment batch size
  remains 512.
- TensorFlow state is cleared and Python garbage collection runs three times
  when the process exits.
- Each parameterized run has a distinct hash directory, preventing overwrite.

Do not start a GPU pilot while the GUI queue is running. There is deliberately
no force or bypass option for the GPU occupancy check.

## CPU-Only Validation

This does not use the GPU:

```bash
MPLCONFIGDIR=/tmp/papermerge-mpl \
CUDA_VISIBLE_DEVICES=-1 \
TF_CPP_MIN_LOG_LEVEL=2 \
PYTHONDONTWRITEBYTECODE=1 \
environment/basil-noise-env/bin/python -m unittest tests.test_wcm
```

## Calibration After The Queue Finishes

First verify that no compute process remains:

```bash
nvidia-smi
```

Run matched 30-round, seed-2025 pilots at the difficult `sigma=0.6` condition:

```bash
environment/basil-noise-env/bin/python scripts/run_wcm_pilot.py \
  --device gpu --approach merged --environment noise \
  --sigma 0.6 --seed 2025 --rounds 30

environment/basil-noise-env/bin/python scripts/run_wcm_pilot.py \
  --device gpu --approach merged --environment hidden_noise \
  --snapshot-selection --sigma 0.6 --seed 2025 --rounds 30

environment/basil-noise-env/bin/python scripts/run_wcm_pilot.py \
  --device gpu --approach cart --environment noise \
  --sigma 0.6 --seed 2025 --rounds 30 --cart-gamma 0.0005

environment/basil-noise-env/bin/python scripts/run_wcm_pilot.py \
  --device gpu --approach cart --environment hidden_noise \
  --snapshot-selection --sigma 0.6 --seed 2025 --rounds 30 \
  --cart-gamma 0.0005
```

Outputs are written only under:

```text
experiments/wcm_pilot/r1/
```

For hidden attack plus noise, compare `SS+WCM` with both matching `SS` and
`SS+EBM`. For noise only, compare WCM with matching no mitigation and EBM.
Comparisons must use the same approach, sigma, seed, round count, data
partition hash, and initialization hash.

## Decision Rule

1. Tune WCM penalty, radius multiplier, or boundary-sample count only with the
   seed-2025 calibration runs.
2. Continue only if WCM improves noise-only accuracy and SS+WCM improves the
   hidden-noise result without reducing worst-node accuracy materially.
3. Freeze the chosen settings before confirmation.
4. Confirm at `sigma=0.2`, `0.4`, and `0.6`, 100 rounds, seeds 2026-2028.
5. Compare paired per-seed deltas and report mean, standard deviation, final
   average accuracy, final worst-node accuracy, and learning-curve AUC.
6. Integrate WCM into the GUI or paper only if the confirmation results support
   it. A favorable single seed is not enough.

WCM is not expected to solve Byzantine corruption by itself. Under the joint
environment, SS remains the Byzantine defense and WCM addresses bounded
channel uncertainty.
