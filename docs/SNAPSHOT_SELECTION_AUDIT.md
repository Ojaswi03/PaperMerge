# BASIL Snapshot Selection audit

Software contracts pass. The smoke evidence does **not** demonstrate that
receiver-local selection restores ten-class learning or reliably excludes
Byzantine candidates under the one-class partition.

## Memory, activation and handoff contracts

Each receiver stores one latest snapshot from each of its five nearest distinct
counterclockwise senders. Sending replaces that sender's entry; it does not
append another historical copy. One activation broadcasts independently to
the next five clockwise receivers; only the immediate receiver trains next.

```text
Before node 0, round 11: [5:r10, 6:r10, 7:r10, 8:r10, 9:r10]
node 0 trains five epochs, then sends to 1, 2, 3, 4, 5
Before node 1, round 11: [6:r10, 7:r10, 8:r10, 9:r10, 0:r11]
                          ^ cached predecessors          ^ next active input
```

`test_rolling_memory_round_10_round_11_invariant` and the actual-engine
`test_real_engine_handoff_memory_and_attack_order` check this example, strict
weight hash chains, attack warm-up through round 19, and attack-before-noise.
No receiving-node weights are averaged with the selected input.

Selection evaluates all five candidates on the same deterministic, normalized,
unaugmented receiver-local batch of at most 512 training images. Minimum local
CE wins; differences within `1e-8` tie toward the nearest counterclockwise
sender. Sender-reported loss/accuracy are telemetry only. Synthetic contracts
verify malicious sender reports cannot alter the decision.

## Measured selections

Final serial smoke runs have 30 activations and 150 candidate evaluations each.
Their attack starts at round 1 solely to exercise the smoke path; production
still starts at 20. The four seeded attacker IDs are `[0,1,5,7]`.

| Condition | Correct local-minimum/tie selections | Honest identity selected | Byzantine identity selected | Actively attacked snapshot selected |
|---|---:|---:|---:|---:|
| Hidden + SS | 30/30 | 18/30 | 12/30 | 10/30 |
| Hidden + relative 0.4 + SS | 30/30 | 17/30 | 13/30 | 11/30 |
| Hidden + relative 0.4 + SS + full EBM | 30/30 | 17/30 | 13/30 | 11/30 |

Configured Byzantine identity is not equivalent to a corrupted snapshot:
initial round -1 models and warm-up round 0 outputs are uncorrupted. The active
column requires a configured Byzantine sender and snapshot round at least 1.
Honest identities may still carry noisy channels. A snapshot's historical
corruption is measured at its source round, not the current receiver round.

## Receiver-local loss versus global accuracy

Global metrics were measured **after the selection decision**, as a read-only
diagnostic. The selected weights are restored before training. Global labels,
accuracy and confusion matrices are never supplied to the selection criterion.

All three conditions finish at 10% average/worst/best global accuracy. Candidate
mean global accuracy is approximately 9.920%, 9.927%, 9.926%; selected-input
mean is 9.867% for each. There is no demonstrated global advantage from the
lower receiver-local loss here.

Only 4/30 (hidden) or 5/30 (joint) activations have varying global accuracy among
candidates, mostly around initial/warm-up transitions. Within those groups,
mean local-loss/global-accuracy Pearson correlations are approximately +0.951,
+0.630 and +0.630. A positive sign does not mean lower loss improves global
accuracy; most groups have no global variation and their correlation is
undefined. These small, dependent smoke samples are not a causal defense
comparison or evidence of statistically significant improvement.

The one-class criterion can favor a model fitting the receiver's sole class
without preserving the other nine. Selection correctness and research efficacy
are distinct; near-chance clean learning precludes a strong independent SS/EBM
conclusion from these threat smokes.

Full per-candidate observations are in
`final_cpu/smoke/<condition>_direct/trace/candidate_evaluation.json` under
`experiments/professor_validation_results/`. The machine summary is
`docs/validation_evidence.json`; historical telemetry is untouched.

