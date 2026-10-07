# PaperMerge Documentation

- [Scientific naming migration](RESEARCH_NAMING_MIGRATION.md): full inventory,
  canonical identifiers, deprecated aliases and numerical-equivalence evidence.

Start with:

- [Fresh GPU-only A rerun](IID_FRESH_A_GPU_QUEUE.md): current A-only idle queue,
  unchanged completed B/C/D, and archived mixed-device A reference.
- [Fresh B/C/D GPU queue](IID_FRESH_GPU_QUEUE.md): current idle preset, preserved
  completed A, archived recovery evidence, and manual fresh-run instructions.
- [IID GPU execution](IID_GPU_EXECUTION.md): remaining-work queue, bounded memory,
  measured synthetic speed, CPU-state reconstruction and device-transition provenance.
- [IID queue recovery](IID_QUEUE_RECOVERY.md): checkpoint continuation, verified
  legacy reconstruction, A/B/C/D ordering, timing labels and plot styles.
- [Live GUI corrections](GUI_LIVE_UPDATES.md): stable widgets, activation-level
  accuracy, IID network event wiring and simulated-event verification.
- [Prepared four-condition IID queue](IID_ABCD_100R_PREPARATION.md): idle
  A/B/C/D 100-round configurations, original preparation policy, pairing evidence and
  exact GUI launch steps. No research execution was started during preparation.
- [Production readiness](RESEARCH_PRODUCTION_READINESS.md): latest scientific
  validation verdict, executed tests, and remaining launch gates.
- [Numerical stability](NUMERICAL_STABILITY_AUDIT.md): first non-finite
  operations, saved-batch evidence, channel norm amplification, and confirmed fixes.
- [Reproducibility](REPRODUCIBILITY_AUDIT.md): direct, worker, spawn,
  and worker-pool comparisons, including residual gradient mismatches.
- [One-class learning](ONE_CLASS_LEARNING_ANALYSIS.md): before/after
  each epoch confusion matrices, prediction collapse, forgetting, and controls.
- [Snapshot Selection](SNAPSHOT_SELECTION_AUDIT.md): rolling memory,
  receiver-local selection contracts and evaluation-only global comparisons.
- [Research sequential protocol](SEQUENTIAL_BASIL_PROTOCOL.md): authoritative five-epoch
  handoff, rolling BASIL memory, separate channel models and full second-order EBM.
- [Source equations](SOURCE_EQUATIONS.md): checked expressions 14, 15b
  and 23, their limitations, and the explicit CIFAR adaptation.
- [Research diagnostic evidence](RESEARCH_DIAGNOSTIC_REPORT.md): five-epoch smoke
  matrix, full-data clean controls, failures, runtime estimates and prepared configs.
- [Research verification](RESEARCH_PROTOCOL_VERIFICATION.md): executed commands, actual
  test counts, protected-artifact hash checks and environment boundaries.
- [Get To Know PaperMerge](GetToKnow.md): advisor-facing repository map,
  current test status, diagrams, end-to-end execution, and function reference.
- [Campaign 3 R2 Guide](BASELINE_STUDY_GUIDE.md): exact protocol, equations,
  calibration, result interpretation, and paper-facing boundaries.
- [Campaign 4 Engineering And Evaluation Plan](ADAPTIVE_STUDY_PLAN.md): implemented
  isolated `results4`/`plots4` protocol, sigma 0.4-0.6 root-cause diagnosis,
  adaptive-EBM boundary, run order, live ring telemetry, plots, and measured
  GPU runtime. Its explicit method-freeze gate prevents confirmation runs until
  all Merged+CART diagnostics and the predeclared decision are recorded.
  Official Campaign 4 evidence is still pending.
- [Gamma Explained](gammaExplained.md): CART `gamma`, class gaps, and active
  proximal strength `mu`.
- [WCM Pilot](WCM_PILOT.md): isolated Worst-Case Model implementation and
  evaluation plan.

Historical retained evidence remains under `experiments/results4/` and `plots4/`.
Obsolete results3/plots3 links are intentionally not recreated. New research
smoke/preflight evidence is separate and explicitly not production research.

The root [README](../README.md) contains setup and normal run commands.
