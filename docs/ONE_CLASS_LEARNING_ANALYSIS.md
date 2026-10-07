# Five-epoch one-class learning analysis

Conclusion: **both constant-class prediction collapse and catastrophic
prediction forgetting are observed**. This is a scientific baseline finding,
not evidence that five epochs should be reduced or the method silently changed.

## Full-data evidence

Three new five-round preflights used all 50,000 training and 10,000 evaluation
images, unchanged five full epochs and batch 512. They are explicitly
`researchValid=false`, `purpose=full_data_preflight`, not production runs.

| Partition | Final average | Worst | Best | Constant-class states after epoch 5 |
|---|---:|---:|---:|---:|
| IID | 33.697% | 31.880% | 36.270% | 0/50 |
| Dirichlet alpha 0.2 | 14.873% | 10.710% | 19.100% | 21/50 |
| One class per node | 10.000% | 10.000% | 10.000% | 50/50 |

IID/Dirichlet captures used the earlier four-thread CPU instrumentation; the
one-class capture used the single-thread CPU reference. These results establish
observed trends, not bitwise cross-runtime equality. They are close to, but do
not overwrite, the previously recorded preflights. GPU behavior is unverified.

The one-class training partition contains exactly 5,000 disjoint examples for
each class/node; the test set remains balanced at 1,000/class. All 50 one-class
activations performed 50 SGD batches: five epochs of nine 512-image batches
and one 392-image batch. Evaluation is never used for training or selection.

## Collapse versus forgetting

The final node-by-class matrix is the identity. That alone might be misread as
each model learning useful class-specific representations. Confusion matrices
show a stronger failure: **every input is assigned the active node's class**.
For node `i`, every true-class row puts all 1,000 test examples in prediction
column `i`. Own-class recall 100% does not establish discriminatory knowledge;
it is compatible with a constant-output classifier.

Temporal evidence confirms forgetting as well. At round 0, node 1 receives
node 0's model:

| Stage at node 1 | Airplane recall | Automobile recall | Other eight recalls |
|---|---:|---:|---:|
| Before local training | 100% | 0% | 0% |
| After epoch 1 | 0% | 100% | 0% |
| After epochs 2–5 | 0% | 100% | 0% |

The classifier overwrites the predecessor's prediction capability while
fitting its sole local label. Overall balanced-test accuracy stays at 10%
through this transition, so the aggregate curve alone conceals it. Across
the one-class preflight, 50 activation/class pairs lose over 50 percentage
points between before-training and epoch 5. This describes prediction-level
forgetting; these measurements do not prove all internal features were erased.

IID improves and retains multiple predicted classes. Dirichlet shows mixed
retention and substantial intermittent collapse, consistent with skewed local
data. Node ID is not a class assignment in those controls. These observations
support the extreme partition/local fine-tuning hypothesis rather than a
completely nonfunctional CIFAR harness, without proving a universal causal theorem.

## Recorded matrices and plots

For each preflight, `trace/confusions.json` contains 300 matrices: five rounds
× ten nodes × six stages (before plus epochs 1–5). Each matrix totals 10,000
evaluation images. Across the three controls, 900 matrices are available.
`metrics.npz` retains `beforeTrainingPerClassAccuracy`, `epochPerClassAccuracy`,
activation and round metrics. Confusion axes are true class × predicted class.

Sixty PNG figures are under `plots/professor_validation/`, with 20 per control:
ten per-node grids covering every round/stage and ten per-node per-class
trajectory figures. Each confusion grid is row-normalized on `[0,1]`.
Example: `preflight_clean_one_class_per_node_direct/node_1_confusions.png`.

```bash
python scripts/run_research_validation.py --suite preflight --output experiments/research_validation_results/new_preflight_check
MPLCONFIGDIR=/tmp/papermerge_mpl python scripts/report_research_validation.py --plots
```

The five local epochs remain fixed. No local-epoch ablation, test-driven
optimization, raw-data sharing or CART mechanism was added. The 50% target is
not a correctness assertion. CART can later be evaluated as a separate,
explicitly authorized extension, not mixed into these baseline findings.

