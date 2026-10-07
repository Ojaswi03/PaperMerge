# Research reproducibility audit

Status: **not fully resolved**. Successful process exit and equal final accuracy
do not establish identical trajectories. GPU behavior is unverified.

## Controlled comparisons

The same persisted `smoke_noise_rel_0.4_ebm.json` was used without changing
its seed, partition, rounds, five epochs, batch size, LR, channel or EBM mode.
Each completed run performs 150 optimizer batches. The comparison includes:

1. direct engine execution;
2. the actual `scripts/run_research_protocol.py` worker, including its GUI
   telemetry callback and persistence;
3. `multiprocessing.get_context("spawn")`;
4. the actual GUI `CampaignWorkerPool` launching the standard worker.

The pool gained optional argument forwarding solely to route diagnostic
output to a new namespace. Existing callers, shutdown behavior and queue
formats remain unchanged. Forking an initialized TensorFlow runtime was avoided.

Initial parameters have SHA-256
`7c717dbbd9bd1f2f230b1ff3d288bd8fec33dcd3ec730eb43e8a14e496cdc2df`.
Train/test image and label hashes match. Every batch records local index order,
raw data hashes, stateless augmentation keys and augmented-image hashes.
Channel streams remain keyed by seed/round/sender/receiver; attacker selection
is resolved once to `[0,1,5,7]`. Sender telemetry does not influence training RNG.

## Earliest divergence

| Comparison | Earliest differing gradient batch | First differing updated weights |
|---|---|---|
| Raw direct vs standard worker | batch 10: round 0/node 2/epoch 1 | batch 17: round 0/node 3/epoch 3 |
| Deterministic-op flag, oneDNN still enabled | batch 1: round 0/node 0/epoch 2 | batch 8: round 0/node 1/epoch 4 |
| oneDNN off, four-thread corrected direct vs worker | batch 88: round 1/node 7/epoch 4 | batch 88 |
| First serial-CPU comparison | no difference in all 150 batches, all four modes | no difference |
| Repeated serial comparison, direct vs spawn | batch 0: round 0/node 0/epoch 1 | batch 0 |
| Fixed hash/BLAS launch environment, direct vs spawn | batch 0 | batch 0 |

At each first divergence, incoming weights, data, shuffle and augmentation
still match. The initial repeated-spawn discrepancy has identical CE
`2.305204391479492`, coefficient `0.00030909254564903677`, base gradient norm
`1.1378867626190186`, and robust gradient norm `1.1380165815353394`, but different
gradient bytes. For example the first convolution gradient's float64-measured
L2 is `0.052040734086190935` versus `0.05204073404273498`.

This isolates the observed divergence to floating-point gradient computation,
not different model initialization, data or random draws. Tiny discrepancies
can grow sharply in the high-noise/second-order regime. The exact TensorFlow
kernel/gradient-aggregation ordering responsible has **not** been isolated.
Do not label it a proven missing-seed, augmentation or worker-initialization bug.

## What was corrected, and what was not

The worker now sets its runtime policy before TensorFlow import: oneDNN off
by default, deterministic operations enabled, CPU intra/inter-op threads 1/1.
GPU host intra-op threads remain 4, but that path is not validated. Runtime
and implementation hashes are recorded. Direct API users must configure their
runtime before initializing TensorFlow; an already-initialized runtime cannot
be made equivalent by changing environment strings afterward.

The validation launcher also fixes `PYTHONHASHSEED=0`, OpenBLAS/MKL/OMP threads
to 1. This does **not** eliminate the repeated spawn discrepancy. The first
fully matched comparison is retained alongside the non-matching repeats;
neither observation is hidden or substituted for the other.

`final_cpu/repro` and `keyed_runtime/repro` each have **4 completed invocations**,
all ending at 10% average/worst/best. Direct, worker and GUI pool hashes match
in these comparisons. Spawn does not match despite completing. Reference-mode
reproducibility is therefore a remaining launch gate, not a passed assertion.

## Evidence and rerun

Machine-readable comparisons, including full first-divergence records, are in
`docs/validation_evidence.json`. Batch/activation traces remain in
`experiments/professor_validation_results/`; no old worker result was overwritten.

```bash
python scripts/run_research_validation.py --suite repro --output experiments/research_validation_results/new_repro_check
python scripts/report_research_validation.py
```

The first command launches only the three-round non-research condition in four
execution modes. Its exit status checks execution, not bitwise reproducibility;
the comparison report must also be inspected. Production remains unauthorized.

