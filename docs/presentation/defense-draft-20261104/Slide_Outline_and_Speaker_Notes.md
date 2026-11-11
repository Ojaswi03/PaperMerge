# Ojaswi Sinha: thesis defense draft

Working format: 16:9. Estimated talk: about 30 minutes, excluding questions.
This is a review draft. Defense date and any university slide template were not supplied.
The PPTX is editable. Charts and equations are images; diagram shapes and prose are native PowerPoint objects.
The PDF is rendered from the same layout description, not by PowerPoint. Check the PPTX in the software used for the defense.
Selected equations are reproduced from the supplied thesis source without mathematical edits. No thesis equations were changed.
Current numerical evidence comes from 150 configurations / 50 groups, using three seeds per group.
All numerical endpoint claims refer to development-holdout accuracy. No significance claim is made.

## 01. On the Robustness of Decentralized Federated Learning against Byzantine Attacks and Channel Noise
Suggested time: 30 seconds.

Introduce yourself as Ojaswi Sinha, an MS Computer Science student at Ohio University, advised by Dr. Animesh Yadav. The title is copied from the latest supplied thesis.tex. No defense date has been invented.

Sources: thesis.tex: title and advisor fields

Visible text:

MASTER OF SCIENCE THESIS DEFENSE
On the Robustness of Decentralized Federated Learning against Byzantine Attacks and Channel Noise
Ojaswi Sinha
Advisor: Dr. Animesh Yadav
MS Computer Science | Ohio University

## 02. The thesis in one slide
Suggested time: 60 seconds.

Read the four quadrants briefly. The question is joint robustness under Byzantine attacks, link noise and label skew. The methods are FedSharp and its class-aware extension, FedSharp+. The result is conditional: the extension helps CIFAR-10 non-IID joint cases, but does not improve MNIST relative to base FedSharp. These are means across three seeds, not significance tests. The two core references establish federated learning and the BASIL baseline.

Sources: Verified 50-group summary; references.bib: mcmahan_fedavg, basil

Visible text:

OVERVIEW
The thesis in one slide
Ojaswi Sinha | MS Computer Science | Ohio University
02
Research question
How can a receiver learn reliably from peers under attacks, channel noise and label skew?
Methodology
FedSharp: guarded exchange + noise-aware training. FedSharp+: class-aware admission + update checks.
Result
FedSharp+: 28.98% on CIFAR-10 non-IID joint cases. No gain over FedSharp on MNIST.
Core references
[1] McMahan et al., AISTATS 2017 Federated averaging [2] Elkordy et al., IEEE JSAC 2022 BASIL

## 03. Collaborative learning has three practical constraints
Suggested time: 50 seconds.

The motivating setting is a collection of institutions or devices that retain training data locally. A central coordinator may be unavailable or undesirable. Even without an aggregator, a peer can transmit a corrupted model and communication can perturb parameters. The experiment is a simulation of parameter noise, not a measured wireless deployment.

Sources: Thesis scope and verified experiment documentation

Visible text:

MOTIVATION
Collaborative learning has three practical constraints
Ojaswi Sinha | MS Computer Science | Ohio University
03
Local data
Clients keep their fitting examples.
No central server
Peers exchange model parameters.
Unreliable exchange
Malicious updates and noisy links.
Removing the server does not remove the need to decide which model to trust.

## 04. Federated learning keeps training data at the clients
Suggested time: 55 seconds.

Define a client, local dataset and model parameters. Each client uses local examples to update a common model structure. Model exchange can still reveal information; this thesis does not prove differential privacy or cryptographic privacy. The diagram is conceptual and does not imply a particular aggregation algorithm.

Sources: [1] McMahan et al., 2017; thesis background

Visible text:

BACKGROUND
Federated learning keeps training data at the clients
Ojaswi Sinha | MS Computer Science | Ohio University
04
Client 1
Data
Local training
Client 2
Data
Local training
Client 3
Data
Local training
Exchange model parameters
The data remain local; the models connect the clients.

## 05. Centralized federated learning uses a coordinating server
Suggested time: 55 seconds.

Contrast federated learning with centralized data collection. In centralized federated learning, the server coordinates and aggregates model updates while data stay at clients. Do not call it centralized training on pooled data. This establishes the server dependency that DFL removes.

Sources: [1] McMahan et al., 2017

Visible text:

BACKGROUND
Centralized federated learning uses a coordinating server
Ojaswi Sinha | MS Computer Science | Ohio University
05
Server
Client 1
Client 2
Client 3
Client 4
Aggregate updates
Distribute the model
Clients train locally, but coordination still depends on one server.

## 06. Decentralized FL moves model exchange to the peers
Suggested time: 55 seconds.

General DFL includes several graph structures and gossip or consensus approaches. A directed ring is one topology, not the definition of DFL. The thesis uses sequential client activation in a directed ring, with predecessor memory rather than a central aggregator.

Sources: [2] BASIL; [3] Hsieh et al., 2020

Visible text:

BACKGROUND
Decentralized FL moves model exchange to the peers
Ojaswi Sinha | MS Computer Science | Ohio University
06
1
2
3
4
5
6
7
8
Peer-to-peer topology
What changes?
No central aggregator. A receiver makes its own decision. Local data may differ across clients.
This thesis studies a sequential directed ring within the broader DFL setting.

## 07. BASIL provides the starting point for the ring study
Suggested time: 65 seconds.

Summarize selected prior work rather than claim a complete survey. BASIL supplies sequential ring learning and local performance-based model selection. Hsieh and colleagues explain difficulties caused by non-IID data. Ang and colleagues motivate noisy communication in federated learning. BASIL with ACDS adds anonymous cyclic data sharing; in our experiment it shares fitting examples and has a different data-access budget. This thesis investigates the combination of threats with receiver-local safeguards.

Sources: [2] BASIL; [3] Hsieh et al.; [4] noisy communication

Visible text:

PRIOR WORK
BASIL provides the starting point for the ring study
Ojaswi Sinha | MS Computer Science | Ohio University
07
Byzantine resilience
BASIL evaluates received models on the receiver's local data. [2]
Statistical heterogeneity
Non-IID data can make local model preferences disagree. [3]
Communication noise
Link perturbations can accumulate during repeated exchange. [4]
Study attacks and channel noise together, then address receiver-level class recognition.

## 08. One model can carry several kinds of error
Suggested time: 60 seconds.

Use this slide to separate Byzantine corruption, fresh link noise and inherited error. The Hidden attack in the experiment is its implemented blend, not necessarily the single-coordinate theoretical example. IID and non-IID refer to the data partition. These are distinct sources of difficulty.

Sources: Thesis scope and verified experiment documentation

Visible text:

PROBLEM SETTING
One model can carry several kinds of error
Ojaswi Sinha | MS Computer Science | Ohio University
08
Sender
Trained model
Attack
Outgoing corruption
Channel
Gaussian link noise
Receiver
Label-skewed data
Byzantine model replacement
Fresh noise + inherited error
The receiver sees the combined effect, not a label identifying the source of error.

## 09. Three research questions guide the evaluation
Suggested time: 55 seconds.

RQ1 asks about guarded exchange and noise-aware training under the combined threats. RQ2 asks whether private class-recognition evidence improves non-IID decisions without sharing fitting examples. RQ3 asks where those choices help or hurt relative to BASIL and ACDS. These are empirical questions for the tested models and conditions, not a universal robustness claim.

Sources: Thesis scope and verified experiment documentation

Visible text:

RESEARCH QUESTIONS
Three research questions guide the evaluation
Ojaswi Sinha | MS Computer Science | Ohio University
09
1
RQ1
Can guarded exchange and noise-aware training improve joint attack and noise performance?
2
RQ2
Can receiver-local class evidence improve learning on non-IID data without sharing examples?
3
RQ3
Where do these methods help, and what performance costs remain?

## 10. Each receiver uses a memory of predecessor models
Suggested time: 65 seconds.

There are 100 clients in a fixed directed ring. One round activates all 100 clients, and each activation uses five SGD steps. In impaired cases, BASIL considers up to 34 predecessors, while FedSharp uses up to 66 with a Byzantine design budget of 33. The diagram shows eight clients only for readability. The window is not 66 independent fresh training runs. Noise is keyed by receiver, sender and model version; re-reading the same copy does not draw fresh noise.

Sources: research_protocol.py; sharp_protocol.py; basil_sharp_thesis.py

Visible text:

SYSTEM MODEL
Each receiver uses a memory of predecessor models
Ojaswi Sinha | MS Computer Science | Ohio University
10
1
2
3
4
5
6
7
8
Schematic: 8 of 100 clients
Memory
Received versions
Decision
Select or admit
Update
Five SGD steps
BASIL: up to 34 copies FedSharp: up to 66 copies

## 11. The evaluation follows a fixed, auditable pipeline
Suggested time: 55 seconds.

Show the evaluation process. Both datasets use a common 1000-image development holdout from original training data, with 100 images per class. Official test sets are not loaded. Partitions, initialization and keyed batches are matched where controls are compared. Results include saved fresh CPU inference receipts and source snapshots. This is not a claim that a final independent test set was used.

Sources: Thesis scope and verified experiment documentation

Visible text:

METHODOLOGY
The evaluation follows a fixed, auditable pipeline
Ojaswi Sinha | MS Computer Science | Ohio University
11
Data
Training archives
Partition
IID or non-IID
Ring
500 rounds
Evaluate
Common holdout
Verify
Sources + inference
Match the data and training budget before comparing the methods.

## 12. BASIL forwards one locally preferred candidate
Suggested time: 60 seconds.

For the impaired BASIL baseline, the receiver evaluates up to 34 predecessor candidates and copies the single candidate with the lowest local cross-entropy. It does not average the three lowest models or the 33 lowest models. The current clean controls follow a separate defense-disabled immediate-predecessor transfer rule, explained later.

Sources: basil_sharp_thesis.py; research_protocol.py; [2]

Visible text:

BASELINE
BASIL forwards one locally preferred candidate
Ojaswi Sinha | MS Computer Science | Ohio University
12
Window
Up to 34 copies
Score
Local mini-batch loss
Choose
One lowest-loss model
Train
Five local steps
A low local loss may not reflect recognition across all classes.

## 13. FedSharp adds safeguards before and during training
Suggested time: 70 seconds.

Explain every block briefly: coordinate-wise median and peak-ratio screening, local loss ranking, checked averaging, and robust proximal training. The evaluated selector uses survivor-dependent budgets and checks the realized pool loss. It can reduce the pool or choose a single model. Do not describe the evaluated implementation as always averaging 33 models. The word pool refers to parameter vectors, not averaging loss values.

Sources: sharp_math.py; sharp_checked_pool.py; guarded selection and worker sources

Visible text:

FEDSHARP
FedSharp adds safeguards before and during training
Ojaswi Sinha | MS Computer Science | Ohio University
13
Screen
Peak deviations
Rank
Local loss
Check pool
Accept or reduce q
Train
Noise + drift penalties
What enters local training?
An accepted parameter vector, after the receiver checks the realized average.

## 14. FedSharp averages model parameters, not the lowest losses
Suggested time: 75 seconds.

Correct a frequent interpretation issue. Loss values rank complete parameter vectors. The actual checked pool averages selected parameter vectors. The count q is at most 33 and can be much smaller. The thesis aggregate equation is reproduced unchanged. Code adds a realized-pool loss check and fallback behavior beyond the older Algorithm 1 description. Current measured joint-case mean q values are approximately 1.94 to 9.75 across dataset/split/attack groups over rounds 21 to 500, not a fixed 33.

Sources: sharp_math.py: peak_screen, ranked_indices; sharp_checked_pool.py: selector

Visible text:

FEDSHARP SELECTION
FedSharp averages model parameters, not the lowest losses
Ojaswi Sinha | MS Computer Science | Ohio University
14
Rank parameters
Use local loss values
Test their average
Check the realized loss
Reduce if needed
Actual count q can be 1
The maximum budget is 33; the accepted count is a measured outcome.

## 15. The training objective addresses noise and local drift
Suggested time: 85 seconds.

Reproduce the thesis equations without editing them. The robust loss adds the squared gradient norm, and the proximal objective anchors training at the admitted initialization. The gradient penalty requires Hessian-vector products through automatic differentiation. In the measured impaired implementation, the lambda denominator uses the realized q, not a fixed 33. The impaired runs use mu=1; the new defense-disabled clean controls use mu=0. In FedSharp+ the same objective structure uses adjusted training CE. The idealized fresh-noise covariance argument does not prove exact post-selection noise reduction or inherited-noise removal.

Sources: thesis.tex: eq:method_robust_loss, eq:method_proximal_objective, eq:method_lambda

Visible text:

FEDSHARP OBJECTIVE
The training objective addresses noise and local drift
Ojaswi Sinha | MS Computer Science | Ohio University
15
Gradient penalty
Reduce local loss sensitivity.
Proximal anchor
Limit movement from initialization.
q is the realized count in code.
Thesis robust and proximal objectives, unchanged. Proximal training: Li et al. [5].

## 16. Under label skew, one loss value may hide class failures
Suggested time: 65 seconds.

This schematic motivates the extension; it is not measured data. Local class imbalance can make low average loss compatible with weak recognition of less frequent classes. FedSharp+ compares complete models using class recognition on two receiver-private probes. It does not combine per-class pieces from different models. No raw fitting examples are shared by this extension. Model sharing alone does not establish formal privacy.

Sources: Thesis scope and verified experiment documentation

Visible text:

WHY FEDSHARP+?
Under label skew, one loss value may hide class failures
Ojaswi Sinha | MS Computer Science | Ohio University
16
Illustrative local label distribution
0
1
2
3
4
Class
Receiver-local question
Does this candidate preserve the classes this receiver can already recognize?
Schematic only. Class counts are illustrative, not experimental measurements.

## 17. FedSharp+ admits models using local class evidence
Suggested time: 80 seconds.

Candidates are the checked FedSharp pool, up to eight eligible individual peers, and the owned model when available. Fit and checking probes are fixed disjoint samples from the receiver fitting data, up to 16 examples per class per probe. Recognition is one-vs-rest MCC from raw logits, not F1. Qualification and informativeness are different checks. With fewer than two supported classes, use the checked pool. With supported classes but uninformative ownership, use the nearest-priority candidate. Otherwise admit the highest-priority candidate within 0.01 of best utility. The priority order puts nearest peers before pool before own. Perturbation allowances use 19 keyed draws when conditions hold; this is an empirical local allowance, not a statistical confidence interval.

Sources: cart_prior_neutral_commit.py; cart_peer_candidates.py; cart_class_skill.py

Visible text:

FEDSHARP+ ADMISSION
FedSharp+ admits models using local class evidence
Ojaswi Sinha | MS Computer Science | Ohio University
17
Candidates
Pool + peers + owned
Two probes
Fit and check, local
Class evidence
Recognition + allowance
Admission
Complete model + anchor
Little class support
Use the checked pool
Uninformative owned model
Prefer the nearest candidate
Informative owned model
Compare class-aware utility

## 18. FedSharp+ checks the whole update before sending it
Suggested time: 90 seconds.

After admission, perform five SGD steps with training-only local class-prior adjustment and the robust proximal objective. Ranking, class probes, repair, commitment and final evaluation use raw logits. An inherited repair stage can adjust only the final affine classifier using two local probes and a stored repair reference. The later commitment step checks the entire model, including backbone layers. It tests scales 1, 1/2, 1/4, 1/8, 0 in order and takes the first utility check that passes. Zero returns to the admitted model, which need not be the owned model. There are no extra SGD steps in this scale check. The unchanged equations are reproduced from the commented FedSharp+ section in supplied thesis.tex for presentation; the thesis itself is not edited. The utility guarantee is finite-probe and single-activation, not population accuracy or every-class preservation.

Sources: thesis.tex: commented eq:commit_scales, eq:commit_gate; cart_local_prior.py; sharp_cart_v2.py; cart_prior_neutral_commit.py

Visible text:

FEDSHARP+ TRAINING AND COMMITMENT
FedSharp+ checks the whole update before sending it
Ojaswi Sinha | MS Computer Science | Ohio University
18
Train
Local prior + five SGD steps
Repair
Classifier, when eligible
Commit
Whole-model scale check
Accept the full update, a smaller update, or the admitted model.
Training-time class-prior adjustment follows Menon et al. [6]. Utility checks are local, not population guarantees.

## 19. The study covers two datasets and three independent seeds
Suggested time: 65 seconds.

There are 150 current configurations grouped into 50 condition groups, with seeds 2025, 2026, 2027. Use three independent seeds per curve; do not pool historical repetitions as six seeds. Non-IID uses Dirichlet alpha 0.5. The MNIST model is 784-64-32-10 MLP, the CIFAR-10 model is the BASIL paper CNN. One common 1000-image training-derived holdout per dataset has 100 examples per class. Holdout is development evaluation, not official independent test data. Mean accuracy includes all 100 retained trained client models, including those with attacked outgoing copies. Report the mean over rounds 491 through 500 per seed, then seed mean and sample SD.

Sources: Thesis scope and verified experiment documentation

Visible text:

EXPERIMENTAL DESIGN
The study covers two datasets and three independent seeds
Ojaswi Sinha | MS Computer Science | Ohio University
19
100
clients
500
rounds
5
SGD steps / activation
3
seeds
Data and partitions
MNIST + CIFAR-10 IID + Dirichlet non-IID (alpha = 0.5)
Evaluation
1,000-image development holdout Endpoint: rounds 491 to 500, mean ± sample SD

## 20. Clean controls establish the transfer baseline
Suggested time: 65 seconds.

The latest BASIL and FedSharp clean controls both use immediate-predecessor transfer, no attack, no channel noise, mu=0, window=1 and no screening, ranking or pooling. Their entire measured 500-round accuracy and class arrays are identical across all 12 paired runs. Do not claim checkpoint parameter-byte identity because checkpoints are not exported. Impaired cases retain selection and regularization, so clean versus impaired comparisons change both environment and policy. Joint conditions use sigma=0.01 and 33 outgoing attacker identities after 20 warm-up rounds. The Hidden implementation is a blend; avoid equating it with a single-coordinate illustrative attack. FedSharp+ and ACDS are only present for joint non-IID cases. ACDS shares 5% of fitting examples, so data access differs.

Sources: Thesis scope and verified experiment documentation

Visible text:

CONTROLS AND COMPARISON SCOPE
Clean controls establish the transfer baseline
Ojaswi Sinha | MS Computer Science | Ohio University
20
Clean environment
No attack or channel noise. Accept the immediate predecessor. No selection or regularization.
Impaired environment
Attack only, or attack + noise. Selection and defenses remain enabled. FedSharp+ is tested on non-IID joint cases.
Joint tests: 33 attackers | Noise SD = 0.01 | Attack onset: round 21
BASIL and FedSharp clean accuracy arrays match exactly in all 12 pairs.

## 21. MNIST IID: accuracy through 500 rounds
Suggested time: 70 seconds.

Clean control: 97.60%. Attack-only BASIL is about 94.09%, higher than FedSharp about 92.56 to 92.60%. Joint FedSharp is about 87.81%, higher than BASIL about 73.37 to 74.18%. These are endpoint means; no significance claim. FedSharp+ was not tested here. The separate BASIL noise-only control, not plotted on this presentation panel, is 73.645%. Noise-only exists only for BASIL on MNIST. Solid/dotted styles distinguish families. Bands are sample SD over three seeds, not confidence intervals. Plot curves are the raw unsmoothed means over all retained client models. The dashed vertical line marks attack onset at displayed round 21. One combined clean curve represents the two exactly matching control curves. See companion chart audit for run IDs and arrays.

Sources: Verified 150-run current matrix; current 50-group endpoint summary

Visible text:

RESULTS
MNIST IID: accuracy through 500 rounds
Ojaswi Sinha | MS Computer Science | Ohio University
21
FedSharp improves the joint-impaired result.
Joint = attack + noise. Three seeds; bands = sample SD. Clean curves coincide. Development-holdout accuracy.

## 22. MNIST non-IID: accuracy through 500 rounds
Suggested time: 70 seconds.

Clean control: 97.63%. Joint base FedSharp is 84.902% Gaussian and 84.841% Hidden. FedSharp+ is 84.264% and 84.296%, respectively, slightly lower. Both exceed BASIL and ACDS here. Attack-only BASIL remains higher than FedSharp. FedSharp+ is not a universal improvement. ACDS has additional fitting-data access. Solid/dotted styles distinguish families. Bands are sample SD over three seeds, not confidence intervals. Plot curves are the raw unsmoothed means over all retained client models. The dashed vertical line marks attack onset at displayed round 21. One combined clean curve represents the two exactly matching control curves. See companion chart audit for run IDs and arrays.

Sources: Verified 150-run current matrix; current 50-group endpoint summary

Visible text:

RESULTS
MNIST non-IID: accuracy through 500 rounds
Ojaswi Sinha | MS Computer Science | Ohio University
22
FedSharp+ does not improve the MNIST mean over FedSharp.
Joint = attack + noise. Three seeds; bands = sample SD. Clean curves coincide. Development-holdout accuracy.

## 23. CIFAR-10 IID: accuracy through 500 rounds
Suggested time: 70 seconds.

Clean control is 61.754%. Joint FedSharp reaches 44.757%, above BASIL 15.960% Gaussian and 16.201% Hidden. Attack-only BASIL is 41.488%, higher than FedSharp 34.855%. Joint FedSharp exceeding attack-only does not isolate a benefit of noise because the gradient penalty and other policy effects differ. FedSharp+ was not tested on IID. Solid/dotted styles distinguish families. Bands are sample SD over three seeds, not confidence intervals. Plot curves are the raw unsmoothed means over all retained client models. The dashed vertical line marks attack onset at displayed round 21. One combined clean curve represents the two exactly matching control curves. See companion chart audit for run IDs and arrays.

Sources: Verified 150-run current matrix; current 50-group endpoint summary

Visible text:

RESULTS
CIFAR-10 IID: accuracy through 500 rounds
Ojaswi Sinha | MS Computer Science | Ohio University
23
FedSharp helps under joint impairments; clean remains highest.
Joint = attack + noise. Three seeds; bands = sample SD. Clean curves coincide. Development-holdout accuracy.

## 24. CIFAR-10 non-IID: accuracy through 500 rounds
Suggested time: 70 seconds.

Clean control is 59.980%. FedSharp+ is 28.983% in both attack types. Base FedSharp is 24.680% Gaussian and 14.626% Hidden; BASIL is 11.488% and 14.770%. The Hidden base difference versus BASIL is small and does not support a superiority claim. ACDS is 19.081% Gaussian and 18.475% Hidden. Equal Gaussian and Hidden FedSharp+ curves are observed, not proof that communication never matters. Solid/dotted styles distinguish families. Bands are sample SD over three seeds, not confidence intervals. Plot curves are the raw unsmoothed means over all retained client models. The dashed vertical line marks attack onset at displayed round 21. One combined clean curve represents the two exactly matching control curves. See companion chart audit for run IDs and arrays.

Sources: Verified 150-run current matrix; current 50-group endpoint summary

Visible text:

RESULTS
CIFAR-10 non-IID: accuracy through 500 rounds
Ojaswi Sinha | MS Computer Science | Ohio University
24
FedSharp+ has the highest joint-impaired mean in this setting.
Joint = attack + noise. Three seeds; bands = sample SD. Clean curves coincide. Development-holdout accuracy.

## 25. FedSharp+ helps CIFAR-10, but the MNIST gain is absent
Suggested time: 65 seconds.

The bars report the mean endpoint over rounds 491 to 500, with sample SD across three seeds. CIFAR-10 uses a different vertical scale from MNIST; both axes are explicitly labeled. Relative to base FedSharp, FedSharp+ changes CIFAR-10 by +4.302 percentage points Gaussian and +14.357 Hidden; MNIST by -0.638 and -0.545. ACDS is a fitting-data-sharing baseline. No formal significance tests have been performed.

Sources: condition_summary.csv, joint non-IID rows

Visible text:

RESULTS COMPARISON
FedSharp+ helps CIFAR-10, but the MNIST gain is absent
Ojaswi Sinha | MS Computer Science | Ohio University
25
The extension is setting-dependent, rather than a consistent improvement.
Joint attack + noise only. Endpoint means ± sample SD, three seeds. Note the different dataset scales.

## 26. Mechanism records help explain the remaining tradeoff
Suggested time: 70 seconds.

These are descriptive mechanism records across rounds 21 to 500, averaged across seeds. On CIFAR-10 non-IID, FedSharp+ admits the owned model in about 99.619% of activations for each attack type, while roughly 28.909% of active commitment checks choose scale zero. Owned admission sets the fresh-link coefficient to zero but does not remove historical error. On MNIST non-IID, owned admission is about 94.758 to 95.103%, and zero commitment is about 12.660 to 12.938%. These associations help interpret outcomes; they do not isolate causation or show all classes are protected. Final worst-client MNIST mean is lower for the extension than the base method.

Sources: mechanism_summary.csv; per_run_mechanisms.csv; rounds 21 to 500

Visible text:

MECHANISM ANALYSIS
Mechanism records help explain the remaining tradeoff
Ojaswi Sinha | MS Computer Science | Ohio University
26
CIFAR-10 non-IID
99.6% owned-model admission
CIFAR-10 non-IID
28.9% zero-scale commitment
Receiver checks often retain a locally established model.
The checks limit change on local probes; they do not guarantee every class improves.

## 27. The answer depends on the condition being tested
Suggested time: 60 seconds.

Answer the research questions without claiming dominance. Guarded exchange with the robust objective helps many joint-impaired cases, while BASIL has higher attack-only means. The class-aware extension improves tested CIFAR-10 non-IID joint means but not MNIST relative to base FedSharp. Current matched clean controls are highest in each group. Longer training is not uniformly better; for example CIFAR-10 non-IID Hidden base FedSharp declines from the 100-round endpoint to the 500-round endpoint. These conclusions apply to the tested experimental matrix.

Sources: Thesis scope and verified experiment documentation

Visible text:

CONCLUSIONS
The answer depends on the condition being tested
Ojaswi Sinha | MS Computer Science | Ohio University
27
FedSharp
Higher means in many joint-impaired cases; BASIL remains stronger in attack-only cases.
FedSharp+
CIFAR-10 non-IID gains, with no consistent gain over FedSharp on MNIST.
Clean controls
Highest accuracy in each group, with identical BASIL and FedSharp trajectories.

## 28. Test generalization and isolate each component
Suggested time: 65 seconds.

Separate completed evidence from proposed work. The current study uses a development holdout, three seeds, two small benchmark models, a fixed ring and fixed attacker identities. Future work should use a final untouched test split, more seeds and uncertainty analysis; ablate admission, prior adjustment, repair and commitment separately; evaluate adaptive attacks, dynamic topologies and realistic channels; and measure per-activation runtime and memory. No differential privacy, population accuracy, convergence, or universal Byzantine guarantee is established by the local utility checks.

Sources: Thesis scope and verified experiment documentation

Visible text:

LIMITATIONS AND FUTURE WORK
Test generalization and isolate each component
Ojaswi Sinha | MS Computer Science | Ohio University
28
Generalization
Untouched final test data and more seeds.
Attribution
Separate admission, prior adjustment, repair and commitment.
Deployment
Adaptive threats, changing topologies and real channel measurements.
Keep local probe guarantees separate from claims about population accuracy.

## 29. Selected references
Suggested time: 30 seconds.

The references are taken from the supplied bibliography. Citation keys are mcmahan_fedavg, basil, hsieh_noniid_quagmire_2020, NoiseComm, fedprox and logit_adjustment. The core references also appear on slide 2. These references cover centralized FL, the BASIL ring baseline, statistical heterogeneity, noisy communication, proximal training and training-time logit adjustment.

Sources: Entries reproduced from supplied references.bib; keys retained in companion notes

Visible text:

REFERENCES
Selected references
Ojaswi Sinha | MS Computer Science | Ohio University
29
[1] H. B. McMahan et al. Communication-Efficient Learning of Deep Networks From Decentralized Data. AISTATS, 2017.
[2] A. R. Elkordy, S. Prakash and S. Avestimehr. Basil: A Fast and Byzantine-Resilient Approach for Decentralized Training. IEEE JSAC, 2022.
[3] K. Hsieh et al. The Non-IID Data Quagmire of Decentralized Machine Learning. ICML, 2020.
[4] F. Ang et al. Robust Federated Learning With Noisy Communication. IEEE Transactions on Communications, 2020.
[5] T. Li et al. Federated Optimization in Heterogeneous Networks. MLSys, 2020.
[6] A. K. Menon et al. Long-Tail Learning via Logit Adjustment. ICLR, 2021.

## 30. Thank you
Suggested time: 10 seconds.

Invite questions. If asked for details, use the separate plot appendix and the slide-by-slide notes. The deck is a draft for review with your advisor.

Sources: Thesis scope and verified experiment documentation

Visible text:

Thank you
Questions and discussion
Ojaswi Sinha
Advisor: Dr. Animesh Yadav
