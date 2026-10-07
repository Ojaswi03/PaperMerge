"""Plots for the two IID tests only; descriptive, not hyperparameter selection."""
from __future__ import annotations

import gzip
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from basil_core.iid_study import iid_output, RESULT_ROOT, TEST_NAMES


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"run.json").read_text())
    if manifest["status"] != "completed": raise ValueError("Metrics require a completed run.")
    with np.load(directory/"metrics.npz",allow_pickle=False) as saved:
        metrics = {k:saved[k] for k in saved.files}
    with gzip.open(directory/"activation_telemetry.json.gz","rt") as handle:
        return manifest,metrics,json.load(handle)


def plot(output, name, series, *, xlabel="Round", ylabel="Accuracy", bar=False, title="", attack_start=20):
    output = iid_output(output,plots=True); output.mkdir(parents=True,exist_ok=True)
    fig,ax = plt.subplots(figsize=(9,4))
    for label,values in series.items():
        if bar: ax.bar(np.arange(len(values)),values,label=label,alpha=.75)
        else: ax.plot(values,label=label)
    if not bar and xlabel=="Round" and attack_start is not None and attack_start<len(next(iter(series.values()))):
        ax.axvline(attack_start,ls=":",color="red",label=f"Attack activation (round {attack_start})")
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    if ylabel=="Accuracy": ax.set_ylim(0,1)
    ax.legend(); fig.tight_layout()
    try: fig.savefig(output/f"{name}.png",dpi=150)
    finally: plt.close(fig)


def round_means(telemetry, getter):
    rounds = sorted({r["round"] for r in telemetry})
    return [float(np.mean([v for r in telemetry if r["round"]==i for v in getter(r)])) for i in rounds]


def generate_run_plots(directory, output):
    manifest,metrics,records = load_run(directory)
    if "fullTestAccuracy" in metrics:
        return generate_research_plots(directory,output,manifest,metrics,records)
    label = "research" if manifest["researchValid"] else manifest["config"]["purpose"]
    options = dict(title=f"{manifest['experimentName']} — {label}",attack_start=manifest["attackStartRound"])
    for name,key in (("full_test_accuracy","averageAccuracy"),("average_node_accuracy","averageAccuracy"),
                     ("worst_node_accuracy","worstNodeAccuracy")):
        plot(output,name,{name:metrics[key]},**options)
    plot(output,"local_loss",{"Local CE":round_means(records,lambda r:[r["localTrainingLoss"]])},ylabel="Cross-entropy",**options)
    plot(output,"final_node_accuracy",{"Final":metrics["roundNodeAccuracy"][-1]},bar=True,xlabel="Node",**options)
    plot(output,"final_class_accuracy",{"Node mean":metrics["roundNodePerClassAccuracy"][-1].mean(axis=0)},bar=True,xlabel="Class ID",**options)
    counts = np.bincount([r["inputSourceNode"] for r in records],minlength=10)
    plot(output,"selected_sender_frequency",{"Selections":counts},bar=True,xlabel="Sender",ylabel="Count",**options)
    plot(output,"honest_byzantine_selections",{"Honest / Byzantine":[manifest["selectedHonest"],manifest["selectedByzantine"]]},bar=True,xlabel="Sender category (0=honest, 1=Byzantine)",ylabel="Count",**options)
    ages = [r["selectedSnapshotAgeRounds"] for r in records if r["selectedSnapshotAgeRounds"] is not None]
    plot(output,"snapshot_staleness",{"Selections":np.bincount(ages) if ages else [0]},bar=True,xlabel="Age in rounds (initial snapshots excluded)",ylabel="Count",**options)
    if manifest["config"]["useChannelNoise"]:
        for key,name,ylabel in (("coordinateSigma","coordinate_sigma","Coordinate standard deviation"),
            ("noiseL2Norm","channel_noise_norm","Noise L2 norm"),("relativeNoiseL2","noise_model_ratio","Noise/model L2 (telemetry only)")):
            plot(output,name,{ylabel:round_means(records,lambda r:[l[key] for l in r["outgoingLinks"]])},ylabel=ylabel,**options)
        for key,name,ylabel in (("ebmPenalty","ebm_penalty","EBM penalty"),("baseGradientNorm","base_gradient_norm","Gradient L2 norm"),
            ("ebmCorrectionNorm","ebm_correction_norm","Hessian-gradient correction norm"),("ebmCorrectionRatio","ebm_correction_ratio","Correction/base norm")):
            plot(output,name,{ylabel:round_means(records,lambda r:[s[key] for s in r["optimizerTelemetry"]])},ylabel=ylabel,**options)
    return len(list(Path(output).glob("*.png")))


def generate_comparison(directories, output):
    loaded = [load_run(d) for d in directories]
    a,b = (r[0] for r in loaded)
    if (a["researchValid"],a["nRounds"],a["iidPartitionHash"],a["initialModelHash"]) != (b["researchValid"],b["nRounds"],b["iidPartitionHash"],b["initialModelHash"]):
        raise ValueError("Comparison requires matched run families, horizons, partition and initialization hashes.")
    if "fullTestAccuracy" in loaded[0][1]:
        for name,key in (("full_test_accuracy_comparison","fullTestAccuracy"),("mean_node_accuracy_comparison","averageAccuracy"),("worst_node_accuracy_comparison","worstNodeAccuracy")):
            plot(output,name,{m["experimentName"]:v[key] for m,v,_ in loaded},title="50-round full-data paired IID BASIL")
        plot(output,"final_per_class_comparison",{m["experimentName"]:v["fullTestPerClassAccuracy"][-1] for m,v,_ in loaded},xlabel="Class ID")
        plot(output,"full_test_loss_comparison",{m["experimentName"]:v["fullTestLoss"] for m,v,_ in loaded},ylabel="Full-test CE (end-of-ring Node 9)")
        rows=[]
        for directory in directories:
            with (Path(directory)/"round_metrics.csv").open() as handle:
                rows.append(list(csv.DictReader(handle)))
        for name,key,label in (("honest_selection_rate_comparison","honest_selection_rate","Non-attacker-designated sender fraction"),
            ("attacked_source_selection_rate_comparison","attacked_source_selection_rate","Actually corrupted source snapshot fraction")):
            plot(output,name,{m["experimentName"]:[float(r[key]) for r in data] for (m,_,_),data in zip(loaded,rows)},ylabel=label,title="50-round full-data paired IID BASIL")
        return
    for name,key in (("full_test_accuracy","averageAccuracy"),("average_node_accuracy","averageAccuracy"),("worst_node_accuracy","worstNodeAccuracy")):
        plot(output,name,{m["experimentName"]:v[key] for m,v,_ in loaded})
    plot(output,"local_loss",{m["experimentName"]:round_means(r,lambda v:[v["localTrainingLoss"]]) for m,_,r in loaded},ylabel="Cross-entropy")
    plot(output,"final_class_accuracy",{m["experimentName"]:v["roundNodePerClassAccuracy"][-1].mean(axis=0) for m,v,_ in loaded},xlabel="Class ID")
    plot(output,"byzantine_selection_rate",{m["experimentName"]:round_means(r,lambda v:[float(v["selectedSenderByzantine"])]) for m,_,r in loaded},ylabel="Byzantine selection fraction")


def write_study_summary():
    config=json.loads((RESULT_ROOT/TEST_NAMES[0]/"config.json").read_text())
    if config.get("roundEvaluationMetrics"):
        return write_research_summary()
    lines = ["# Paired IID BASIL study", "",
        "The research horizon is **100 rounds**. Diagnostics are not research-valid runs.", "",
        "## Exact configurations", "",
        "- [Test 1 config](test_01_basil_iid_no_noise_ce/config.json)",
        "- [Test 2 config](test_02_basil_iid_noise_ebm/config.json)", "",
        "Both: `sequential_basil_iid_v1`, CIFAR-10, ten nodes, four seeded attackers,",
        "S=5 rolling distinct-predecessor memory, receiver-local minimum-CE Snapshot Selection,",
        "strict handoff, five full epochs, batch 512, SGD with initial LR 0.05 and",
        "approved round decay 0.05/(1+0.05r), no momentum/weight decay/clipping.", "",
        "Hidden attack: `delayed_hidden_parameter_attack_v1`, production onset round 20.",
        "Each node trains honestly before outbound corruption; attack precedes per-link noise.",
        "No anchoring, CART, class registry, MC objective, or legacy scaling is used.", "",
        "Test 1: clean channel, CE only. Test 2: absolute Gaussian channel and",
        "CE + sigma_e² ||grad CE||², differentiated with nested tapes; no lambda.", "",
        "## Outcomes", ""]
    for name in TEST_NAMES:
        config = json.loads((RESULT_ROOT/name/"config.json").read_text())
        lines += [f"### {name}", "", f"Attacker IDs: `{config['resolvedAttackerIds']}`.",
            f"sigma_e: `{config['channelNoiseSigmaAbsolute']}`; approval: `{config['sigmaApproval']}`.", ""]
        for family in ("research","preflight","smoke"):
            directory = RESULT_ROOT/name/(family if family!="research" else "")
            path = directory/"run.json"
            if not path.exists():
                lines += [f"- {family}: **not run**."]
                continue
            manifest = json.loads(path.read_text())
            lines += [f"- {family}: **{manifest['status']}**, {manifest['nRounds']} rounds; researchValid={manifest['researchValid']}."]
            if manifest["status"] != "completed": continue
            lines += [f"  Average: {manifest['finalAverageAccuracy']:.4%}; worst: {manifest['finalWorstAccuracy']:.4%}.",
                f"  Per-node: `{manifest['finalPerNodeAccuracy']}`.",
                f"  Per-class (mean over ten node models): `{manifest['finalPerClassAccuracy']}`.",
                f"  Best average: {manifest['bestAverageAccuracy']:.4%}, zero-based round {manifest['bestRound']}.",
                f"  Selections: honest={manifest['selectedHonest']}, Byzantine identity={manifest['selectedByzantine']}.",
                f"  Attack-active activations: {manifest['attackActiveActivations']}.",
                f"  Gate checks: `{manifest['preflightChecks']}`.",
                f"  Runtime: {manifest['runtimeSeconds']:.1f}s." + (f" Linear full-data 100-round estimate: {manifest['runtimeSeconds']*100/manifest['nRounds']/3600:.2f}h." if family!='smoke' else " Reduced-data smoke timing is not a production runtime estimate."),
                f"  Partition hash: `{manifest['iidPartitionHash']}`.",
                f"  Initialization hash: `{manifest['initialModelHash']}`."]
        lines += [""]
    audit_path = RESULT_ROOT/"comparison"/"iid_partition_audit.json"
    lines += ["## IID partition verification", ""]
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines += [f"Assigned/unique training samples: {audit['assignedSamples']}/{audit['uniqueSamples']}; deterministic seed 2025.",
            "Full test set: 10,000 examples, 1,000/class, evaluation-only.",
            "The existing random disjoint equal-sized partitioner was correct and reused unchanged.", "",
            "| Node | Samples | " + " | ".join(str(i) for i in range(10)) + " |",
            "|---|---|"+"---|"*10]
        lines += ["| "+" | ".join(map(str,[r['nodeId'],r['samples'],*r['classCounts']]))+" |" for r in audit['nodes']]
    lines += ["", "## Pairing, interpretation and limitations", "",
        "Test 2 sigma remains unresolved unless explicitly supplied by the operator.",
        "Calibrated candidates: 0.005, 0.010, 0.020 (coordinate standard deviations, not percentages).",
        "Full runs require both full-data preflights to pass; numerical divergence aborts rather than tuning.",
        "Three-round full-data preflights exercise warm-up only; they do not validate attack-period learning.",
        "Attack onset/order is covered separately by deterministic engine contracts and optional labelled smoke.",
        "If Test 2 is not run, accuracy differences, noise effects, and paired selection differences are unavailable.",
        "Global accuracy is the mean of ten logical node models evaluated on the complete test set, not a FedAvg model.",
        "Honest-vs-Byzantine selection counts identify senders; a Byzantine sender is not necessarily attack-active during warm-up.",
        "Pre/post-channel accuracy is not measured in this path; channel norms and hashes are available per transmission.",
        "The two tests change both channel and objective, so **they do not isolate EBM's causal benefit**.",
        "A third matched IID + noise + CE-only control is scientifically necessary, but was not run.", "",
        "## Source attribution", "",
        "BASIL Eq. (3): sequential logical ring, bounded memory, receiver-local performance selection, Byzantine filtering.",
        "Ang et al.: Gaussian channel and expectation-based gradient-norm regularization.",
        "Our multiclass CNN adaptation replaces the source objective with logits-based CIFAR-10 CE.",
        "Eq. (14) approximates a squared-loss quantity; it is not a general expected-CE identity.",
        "Eq. (23)'s scalar simplification is not the CNN identity g + 2 sigma_e² H g.",
        "See [source equations](../../docs/SOURCE_EQUATIONS.md) and the two PDFs under `papers/`.", "",
        "## Integrity and verification", "",
        "Single-process deterministic CPU, oneDNN off, one intra/inter-op lane; GPU behavior unverified.",
        "Existing multiprocessing discrepancies remain unresolved; primary IID execution does not use that path.",
        "See `comparison/integrity.json` and `comparison/verification.json` for hashes and exact commands/results.", ""]
    (iid_output(RESULT_ROOT/"IID_TWO_TEST_SUMMARY.md")).write_text("\n".join(lines))


def attack_analysis(values):
    values=np.asarray(values,dtype=float)
    if len(values)<=20: return {"completedRounds":len(values),"postAttack":"unavailable"}
    before=values[:20];after=values[20:]; peak=np.maximum.accumulate(after)
    return dict(preAttackMean=float(before.mean()),postAttackMean=float(after.mean()),
        preAttackBest=float(before.max()),postAttackBest=float(after.max()),
        round19=float(values[19]),round20=float(values[20]),final=float(values[-1]),
        maxDropFromRound19=float(max(0.,values[19]-after.min())),
        maxPostAttackPeakToLaterDrop=float((peak-after).max()))


def generate_research_plots(directory,output,manifest,metrics,records=()):
    title=("Test 1 clean IID BASIL" if not manifest["config"]["useChannelNoise"] else "Test 2 noisy IID BASIL + EBM")+f" — {manifest['nRounds']}-round full-data run"
    campaign=bool(manifest['config'].get('iidCampaignId'))
    if campaign:title=manifest['config']['displayName'].split(' — Fresh')[0]+f" — {manifest['nRounds']}-round full-data run"
    options=dict(title=title,attack_start=manifest['attackStartRound'] if not campaign or manifest['config']['attackHidden'] else None)
    for name,key,ylabel in (("full_test_accuracy_vs_round","fullTestAccuracy","Accuracy"),
        ("mean_node_accuracy_vs_round","averageAccuracy","Accuracy"),
        ("worst_node_accuracy_vs_round","worstNodeAccuracy","Accuracy"),
        ("loss_vs_round","fullTestLoss","Full-test CE (end-of-ring Node 9)")):
        plot(output,name,{name:metrics[key]},ylabel=ylabel,**options)
    if campaign:plot(output,'full_test_loss_vs_round',{'Full-test CE':metrics['fullTestLoss']},ylabel='Full-test CE (end-of-ring Node 9)',**options)
    classes=metrics["fullTestPerClassAccuracy"]
    plot(output,"per_class_accuracy_vs_round",{f"Class {c}":classes[:,c] for c in range(10)},**options)
    for c in range(10): plot(output,f"class_{c}_accuracy_vs_round",{f"Class {c}":classes[:,c]},**options)
    plot(output,"final_per_class_accuracy",{"Node 9":classes[-1]},bar=True,xlabel="Class ID",**options)
    plot(output,"final_per_node_accuracy",{"Final":metrics["roundNodeAccuracy"][-1]},bar=True,xlabel="Node ID",**options)
    with (Path(directory)/"round_metrics.csv").open() as handle: rows=list(csv.DictReader(handle))
    for name,key,label in (("honest_snapshot_selection_rate_vs_round","honest_selection_rate","Non-attacker-designated sender fraction"),
        ("byzantine_snapshot_selection_rate_vs_round","byzantine_selection_rate","Attacker-designated sender fraction; attack starts at 20"),
        ("attacked_source_selection_rate_vs_round","attacked_source_selection_rate","Actually corrupted source snapshot fraction"),
        ("selected_snapshot_age_vs_round","mean_snapshot_age","Snapshot age (rounds)")):
        plot(output,name,{label:[float(r[key]) if r[key] else np.nan for r in rows]},ylabel=label,**options)
    plot(output,"local_ce_vs_round",{"Mean local training CE":[float(r["mean_local_ce"]) for r in rows]},ylabel="Mean local training CE",**options)
    if records:
        counts=np.bincount([r["inputSourceNode"] for r in records],minlength=10)
        plot(output,"selected_sender_frequency",{"Selections":counts},bar=True,xlabel="Sender ID",ylabel="Count",**options)
        honest=sum(not r["selectedSenderByzantine"] for r in records)
        plot(output,"honest_byzantine_selections",{"Selections":[honest,len(records)-honest]},bar=True,xlabel="Sender identity: 0=honest, 1=attacker-designated",ylabel="Count",**options)
        ages=[r["selectedSnapshotAgeRounds"] for r in records if r["selectedSnapshotAgeRounds"] is not None]
        if ages:
            plot(output,"snapshot_staleness",{"Selections":np.bincount(ages)},bar=True,xlabel="Age in rounds (initial snapshots excluded)",ylabel="Count",**options)
    if manifest["config"]["useChannelNoise"]:
        for key,label in (("ebm_penalty","EBM penalty"),("ordinary_gradient_norm","Ordinary CE gradient L2"),
            ("ebm_correction_norm","EBM correction L2"),("ebm_correction_ratio","Correction / CE gradient norm"),
            ("channel_noise_norm","Sampled channel noise L2"),("noise_to_model_norm_ratio","Noise / post-attack model L2 (telemetry only)")):
            if campaign and manifest['config']['localObjective']!='source_ebm' and key.startswith(('ebm_','ordinary_')):continue
            plot(output,key+"_vs_round",{label:[float(r[key]) for r in rows]},ylabel=label,**options)
    return len(list(Path(output).glob("*.png")))


def write_research_summary():
    configs=[json.loads((RESULT_ROOT/n/"config.json").read_text()) for n in TEST_NAMES]
    lines=["# Paired full-data IID BASIL research", "",f"Authorized horizon: **{configs[0]['nRounds']} rounds**, zero-based rounds 0–49.","",
        "Test 2 sigma_e=0.010, variance=0.0001: a-priori moderate calibration-based selection, not selected from final test accuracy.",
        "Calibration at initialization: approximately 11%, 23%, 45% noise/model-L2 for coordinate sigmas 0.005, 0.010, 0.020. These ratios never control the channel.","",
        "## Metric definitions","",
        "- full_test_accuracy: end-of-ring **Node 9 honest trained model**, before outbound channel noise, evaluated on all 10,000 test images. This reference node is fixed before training, never selected by test performance.",
        "- mean_node_accuracy: arithmetic mean of full-test accuracy across all ten stored logical node trained models; no parameter averaging or ensemble.",
        "- worst_node_accuracy: minimum full-test accuracy across those ten models.",
        "- full_test_loss: mean logits-based CE of the fixed Node 9 model on the complete test set.",
        "- per-class trajectory/final bars: Node 9 test accuracy within each class; node-by-class metrics are also retained.",
        "- Attacker-designated sender rates include warm-up; source snapshots are genuinely attacked only if their source round is >=20.","",
        "## Paired protocol","",
        "`sequential_basil_iid_v1`; seed 2025; attackers [0,1,5,7]; delayed Hidden attack starts round 20; N=10, b=4, S=5.",
        "5,000 IID samples/node, all ten classes; five full epochs, batch 512, last batch 392, 50 optimizer steps/activation.",
        "Approved 117,706-parameter CNN; reset SGD, eta_r=0.05/(1+0.05r), no momentum/clipping/weight decay.",
        "Receiver-local minimum-CE selection on one identical training batch for all five candidates; strict handoff, no averaging.",
        "Hidden outbound corruption precedes independent keyed absolute Gaussian noise. No lambda, anchor, CART, proximal or MC loss.","",
        "## Outcomes",""]
    completed=[]
    for name in TEST_NAMES:
        directory=RESULT_ROOT/name;cfg=json.loads((directory/"config.json").read_text())
        lines += [f"### {name}","",f"Exact configuration: [{name}/config.json]({name}/config.json).",""]
        path=directory/"run.json"
        if not path.exists(): lines += ["Not started.",""];continue
        meta=json.loads(path.read_text());lines += [f"Status: **{meta['status']}**; completed rounds: {meta.get('completedRounds',0)}; researchValid={meta['researchValid']}.",
            f"Initial fixed-model accuracy: {meta['initialGlobalAccuracy']:.4%}; device {meta['device']}; {meta['executionMode']}.",
            f"Partition hash: `{meta['iidPartitionHash']}`; initial hash: `{meta['initialModelHash']}`.",""]
        if meta["status"]!="completed":
            lines += [meta.get("error","Measured partial rounds are available in round_metrics.csv; no unexecuted rounds fabricated."),""];continue
        _,metrics,records=load_run(directory);completed.append((meta,metrics,records))
        lines += [f"Runtime: {meta['runtimeSeconds']:.1f}s; final full-test {metrics['fullTestAccuracy'][-1]:.4%}; best {metrics['fullTestAccuracy'].max():.4%} at round {metrics['fullTestAccuracy'].argmax()}.",
            f"Final node mean {metrics['averageAccuracy'][-1]:.4%}; worst {metrics['worstNodeAccuracy'][-1]:.4%}.",
            f"Final Node 9 per-class accuracies: `{metrics['fullTestPerClassAccuracy'][-1].tolist()}`.",
            f"Pre/post attack analysis (full-test reference): `{attack_analysis(metrics['fullTestAccuracy'])}`.",
            f"Pre/post attack analysis (node mean): `{attack_analysis(metrics['averageAccuracy'])}`.",
            f"Selections: honest identity {meta['selectedHonest']}, attacker identity {meta['selectedByzantine']}; attack-active activations {meta['attackActiveActivations']}.",
            f"Round aggregates: `{meta.get('roundAggregateStatistics',{})}`.",""]
    if len(completed)==2:
        a,b=(v[1] for v in completed)
        lines += ["## Comparison","",f"Final full-test difference (Test 2 minus Test 1): {100*float(b['fullTestAccuracy'][-1]-a['fullTestAccuracy'][-1]):+.4f} percentage points.",
            f"Pre-attack mean difference: {100*float(b['fullTestAccuracy'][:20].mean()-a['fullTestAccuracy'][:20].mean()):+.4f} percentage points.",
            f"Post-attack mean difference: {100*float(b['fullTestAccuracy'][20:].mean()-a['fullTestAccuracy'][20:].mean()):+.4f} percentage points.",
            f"Final per-class differences: `{(b['fullTestPerClassAccuracy'][-1]-a['fullTestPerClassAccuracy'][-1]).tolist()}`.",""]
        lines += ["### Final per-node full-test accuracy","","| Node | Test 1 | Test 2 |","|---|---:|---:|"]
        lines += [f"| {i} | {a['roundNodeAccuracy'][-1,i]:.2%} | {b['roundNodeAccuracy'][-1,i]:.2%} |" for i in range(10)]
        classes=("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
        lines += ["","### Final per-class accuracy of the fixed Node 9 reference","","| Class | Test 1 | Test 2 | Difference (percentage points) |","|---|---:|---:|---:|"]
        lines += [f"| {label} | {a['fullTestPerClassAccuracy'][-1,i]:.2%} | {b['fullTestPerClassAccuracy'][-1,i]:.2%} | {100*(b['fullTestPerClassAccuracy'][-1,i]-a['fullTestPerClassAccuracy'][-1,i]):+.2f} |" for i,label in enumerate(classes)]
        lines += ["","### Snapshot Selection by phase","","| Condition | Phase | Selections | Honest-designated sender | Attacker-designated sender | Actually attacked snapshot |","|---|---|---:|---:|---:|---:|"]
        for index,(_,_,records) in enumerate(completed,1):
            for phase,chosen in (("Pre-attack",[r for r in records if r['round']<20]),("Post-activation",[r for r in records if r['round']>=20])):
                designated=sum(r['selectedSenderByzantine'] for r in chosen)
                attacked=sum(r['selectedSenderByzantine'] and r['inputSnapshotRound']>=20 for r in chosen)
                lines += [f"| Test {index} | {phase} | {len(chosen)} | {len(chosen)-designated} | {designated} | {attacked} |"]
        lines += ["","An attacker-designated sender can still supply a clean round-19 snapshot at attack onset. Increased post-attack accuracy also reflects continued training; it does not mean attacks improve learning.","",
            "### Execution and pairing verification","",
            "[Artifact verification](comparison/artifact_verification.json): 500 activations, 25,000 optimizer batches, 2,500 directed transmissions and 120 attack-active activations per run.",
            "All 25,000 paired batch data/augmentation hashes match. Selected-to-loaded parameter hashes match at all 500 activations in each run. The actual Round-11 memory example matches in both runs.","",
            "### Main plots","",
            "- [Full-test accuracy comparison](../../newPlots/IID/comparison/full_test_accuracy_comparison.png)",
            "- [Mean-node comparison](../../newPlots/IID/comparison/mean_node_accuracy_comparison.png)",
            "- [Worst-node comparison](../../newPlots/IID/comparison/worst_node_accuracy_comparison.png)",
            "- [Final per-class comparison](../../newPlots/IID/comparison/final_per_class_comparison.png)",
            "- [Honest selection comparison](../../newPlots/IID/comparison/honest_selection_rate_comparison.png)",
            "- [Actually attacked selection comparison](../../newPlots/IID/comparison/attacked_source_selection_rate_comparison.png)",
            "Per-run directories also contain full-test CE, mean local CE, ten per-class trajectories, snapshot ages/frequencies, and Test-2 EBM/noise trajectories.",""]
    lines += ["## Integrity and limitations","",
        "Complete CIFAR test set is evaluation-only: no test-driven selection, training, schedule, noise choice or early stopping.",
        "Single-process deterministic CPU; GPU equivalence unverified; unresolved multiprocessing differences are not used.",
        "Two conditions change both noise and objective: **this comparison does not isolate EBM's causal benefit**.",
        "A future IID + sigma_e=0.010 noise + CE-only control is required, but is not run in this task.",
        "BASIL Eq. (3) supplies receiver-local performance-based selection. Ang et al. supplies noise-aware gradient-norm regularization; multiclass CNN CE is our adaptation.",
        "Source Eq. (14) concerns a squared-loss approximation; Eq. (23) scalar scaling is not a general CNN identity. The applied objective gradient is g+2 sigma_e² H g.",
        "See comparison/verification.json and comparison/integrity.json for commands, tests and protected SHA-256 verification.",""]
    iid_output(RESULT_ROOT/"IID_TWO_TEST_SUMMARY.md").write_text("\n".join(lines))
