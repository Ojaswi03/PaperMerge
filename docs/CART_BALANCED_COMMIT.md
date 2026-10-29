# CART balanced-loss commitment experiment

The completed 10-round local mechanism screen found that full CART rejected
82 of 95 checked complete updates in its final Dirichlet round. Removing the
commit check reduced final accuracy from 22.361% to 16.321%. Macro-only scoring
reached 22.395%, so these diagnostics do not establish an independent benefit
from the class-retention penalty. They motivate changing the commitment score
while retaining a check, rather than removing all checks.

The experimental `balanced_commit` arm keeps the original class-MCC ingress
selection, channel allowance, actual admitted proximal anchor, five local SGD
steps, and classifier repair. It changes only the score used to accept the
complete locally trained model. Default screen, isolation, and confirmation
presets retain their existing arms; the new arm must be explicitly requested.

For each private probe P and its supported class intersection C, define

    L(P, theta) = mean over c in C of mean CE(theta; x, c) over examples of c in P.

Both fitting and checking probes must have no higher L than the admitted model.
The only allowance is 1e-10 for numerical comparisons. Try the unchanged update
scales 1, 1/2, 1/4, 1/8, and 0. The zero scale restores the admitted model exactly.
Class-balanced means give each locally supported class equal weight rather
than giving majority classes more weight through their probe counts. Stable
float64 log-sum-exp evaluates the CE directly from logits. A nonfinite candidate
is rejected, and a nonfinite baseline stops the activation with an error.

This continuous confidence-sensitive score is a response to the discontinuity
of argmax-based MCC. It does not guarantee that every class improves, and a
small loss improvement can still change a class prediction unfavorably.
Classification accuracy and MCC therefore remain diagnostic outcomes. A partial
update retains the previous reference metadata, as in the original commit path.

The reused private probes, data availability, and global evaluation procedure
are unchanged. This is not independent validation at every activation or a
generalization guarantee. Balanced CE, validation-based update screening, and
backtracking are established techniques; their use here is not a standalone
novelty claim.

## Evaluation sequence

1. Unit checks for per-class weighting, confidence changes, both-probe rejection,
   nonfinite proposals, zero fallback, and partial-update reference consistency.
2. CPU protocol equivalence for the unchanged original CART and exact interrupted
   recovery for the new arm.
3. GPU objective/scoring equivalence gate for the frozen new producer source.
4. Matched 10-round Dirichlet pilot, seed 2025, attack beginning at zero-based
   round 3; compare with the archived completed local screen.
5. One 50-round Dirichlet CART run, seed 2025 and attack beginning at zero-based
   round 20, if the short pilot warrants continuation. Report the changed onset
   explicitly. A single trajectory cannot establish across-seed improvement.

Earlier results and source snapshots remain preserved. Source changes produce
new frozen plans, and completed old runs must be read with their own snapshots.
Do not silently replace old figure series or treat this experiment as completed
until its saved status and final inference checks establish completion.
