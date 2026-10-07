"""Plots for research-protocol results; never writes beneath plots4/."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from basil_core.research_protocol import CLASS_NAMES
from gui.research_protocol import PLOT_ROOT, RESULT_ROOT
from basil_core.artifact_paths import writable_output
from basil_core.protocol_compatibility import normalize_config


def _save(fig,path):
    path=writable_output(Path(path)); path.parent.mkdir(parents=True,exist_ok=True); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def _line(ax,values,label,**kwargs): ax.plot(np.arange(len(values)),values,label=label,**kwargs); ax.axhline(.5,color="#777",ls="--",lw=1,label="50% research target"); ax.set(xlabel="Full ring round",ylabel="Global test accuracy",ylim=(0,1)); ax.legend(fontsize=8)


def generate_run_plots(run_dir,plot_root=PLOT_ROOT):
    run_dir=Path(run_dir); metadata=json.loads((run_dir/"run.json").read_text())
    if metadata["status"]!="completed":raise ValueError("Only completed runs have plot-ready metrics.")
    family="production" if metadata["researchValid"] else "preflight" if metadata["config"].get("purpose")=="full_data_preflight" else "smoke"
    if "reference_rechecks" in run_dir.parts:family="reference_rechecks"
    elif "rechecks" in run_dir.parts:family="rechecks"
    output=writable_output(Path(plot_root)/metadata["config"].get("protocolRevision","legacy_diagnostic")/family/metadata["runId"])
    with np.load(run_dir/"metrics.npz",allow_pickle=False) as saved: metrics={key:saved[key] for key in saved.files}
    with gzip.open(run_dir/"activation_telemetry.json.gz","rt",encoding="utf-8") as handle: telemetry=json.load(handle)
    attack_start=int(metadata["config"].get("attackStartRound",20)); generated=[]
    def mark_attack(ax,scale=1):
        if metadata["config"].get("attackHidden"):
            ax.axvline(attack_start*scale,color="#c53d50",ls=":",label="attack start")
    fig,ax=plt.subplots(figsize=(8,4)); _line(ax,metrics["averageAccuracy"],"average"); ax.plot(metrics["worstNodeAccuracy"],label="worst"); mark_attack(ax); ax.legend(); path=output/"average_worst_accuracy.png"; _save(fig,path); generated.append(path)
    fig,ax=plt.subplots(figsize=(8,5));
    for node in range(10): ax.plot(metrics["roundNodeAccuracy"][:,node],label=f"Node {node}")
    ax.axhline(.5,color="#777",ls="--"); mark_attack(ax); ax.set(xlabel="Round",ylabel="Accuracy",ylim=(0,1)); ax.legend(ncol=2,fontsize=7); path=output/"per_node_accuracy.png"; _save(fig,path); generated.append(path)
    rounds=len(metrics["roundNodePerClassAccuracy"])
    selected=sorted(set([index for index in (0,1,5,10,20,50,99) if index<rounds]+[rounds-1]))
    for round_id in selected:
        fig,ax=plt.subplots(figsize=(8,6)); image=ax.imshow(metrics["roundNodePerClassAccuracy"][round_id],vmin=0,vmax=1,cmap="viridis"); ax.set(xticks=range(10),xticklabels=CLASS_NAMES,yticks=range(10),ylabel="Node",title=f"Node-by-class accuracy — round {round_id}"); plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); fig.colorbar(image,ax=ax); path=output/f"node_class_round_{round_id:03d}.png"; _save(fig,path); generated.append(path)
    activation=metrics["activationPerClassAccuracy"].reshape(-1,10); fig,ax=plt.subplots(figsize=(10,6)); image=ax.imshow(activation.T,aspect="auto",vmin=0,vmax=1,cmap="magma"); ax.set(xlabel="Sequential node activation",ylabel="CIFAR-10 class",yticks=range(10),yticklabels=CLASS_NAMES); fig.colorbar(image,ax=ax); path=output/"activation_class_trajectory.png"; _save(fig,path); generated.append(path)
    fig,ax=plt.subplots(figsize=(9,5)); image=ax.imshow(metrics["selectedSources"].T,aspect="auto",cmap="tab10",vmin=0,vmax=9); ax.set(xlabel="Round",ylabel="Receiver node",title="Selected sender"); fig.colorbar(image,ax=ax); path=output/"selected_source_heatmap.png"; _save(fig,path); generated.append(path)
    activations=np.arange(len(telemetry));
    for key,label,name in (("modelL2Norm","Model L2 norm","model_l2_norm.png"),("ebmCoefficient","EBM coefficient","ebm_coefficient.png"),("baseGradientNorm","Base gradient norm","gradient_norms.png")):
        fig,ax=plt.subplots(figsize=(9,4)); ax.plot(activations,[item[key] for item in telemetry],label=label)
        if key=="baseGradientNorm": ax.plot(activations,[item["robustObjectiveGradientNorm"] for item in telemetry],label="Robust-objective gradient norm")
        mark_attack(ax,10); ax.set(xlabel="Sequential activation",ylabel=label); ax.legend(); path=output/name; _save(fig,path); generated.append(path)
    relative=[link["relativeNoiseL2"] for item in telemetry for link in item["outgoingLinks"]]; coordinate=[link["coordinateSigma"] for item in telemetry for link in item["outgoingLinks"]]
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True); axes[0].plot(relative); axes[0].set_ylabel("Relative noise L2"); axes[1].plot(coordinate); axes[1].set(ylabel="Coordinate sigma",xlabel="Outgoing link transmission"); path=output/"realized_channel_noise.png"; _save(fig,path); generated.append(path)
    candidate_records=[item for item in telemetry if item["candidates"]]
    if candidate_records:
        fig,ax=plt.subplots(figsize=(9,4)); ax.boxplot([[candidate["receiverLocalLoss"] for candidate in item["candidates"]] for item in candidate_records],showfliers=False); ax.set(xlabel="SS activation",ylabel="Receiver-local CE",title="Snapshot candidate losses"); path=output/"candidate_loss_diagnostics.png"; _save(fig,path); generated.append(path)
    if "epochPerClassAccuracy" in metrics:
        # Before training followed by all five epochs reveals intra-activation forgetting.
        states=np.concatenate([metrics["beforeTrainingPerClassAccuracy"][:,:,None,:],metrics["epochPerClassAccuracy"]],axis=2).reshape(-1,10)
        fig,ax=plt.subplots(figsize=(11,6));image=ax.imshow(states.T,aspect="auto",vmin=0,vmax=1,cmap="magma");ax.set(xlabel="Activation × (before, epoch 1, 2, 3, 4, 5)",yticks=range(10),yticklabels=CLASS_NAMES,title="Five-epoch class retention");fig.colorbar(image,ax=ax);path=output/"five_epoch_class_retention.png";_save(fig,path);generated.append(path)
        fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
        steps=[s for record in telemetry for s in record["optimizerTelemetry"]]
        axes[0].plot([s["baseCrossEntropy"] for s in steps],label="Base CE");axes[0].plot([s["ebmObjective"] for s in steps],label="Reported objective (full EBM only where configured)");axes[0].legend(fontsize=8)
        axes[1].plot([s["ebmCoefficient"] for s in steps],label="Actual step coefficient");axes[1].set(xlabel="Optimizer step",ylabel="Coefficient");axes[1].legend();path=output/"optimizer_objective_diagnostics.png";_save(fig,path);generated.append(path)
    return {"runId":metadata["runId"],"generated":[str(path) for path in generated],"plotRoot":str(output)}


def generate_comparison(result_root=RESULT_ROOT,plot_root=PLOT_ROOT,*,purpose="research_protocol_production",revision=None):
    records=[]
    for path in Path(result_root).rglob("run.json"):
        if "rechecks" in path.parts or "reference_rechecks" in path.parts:continue
        data=json.loads(path.read_text());data["config"]=normalize_config(data.get("config",{}))
        if data.get("status")=="completed" and data.get("config",{}).get("purpose")==purpose and (revision is None or data["config"].get("protocolRevision")==revision): records.append((path,data))
    if not records:return {"generated":[],"records":0}
    output=writable_output(Path(plot_root)/(revision or "all_revisions")/purpose/"comparison");generated=[]
    groups={}
    for source,item in records:
        cfg=item["config"];sigma=cfg.get("channelNoiseSigmaRelative",cfg.get("channelNoiseSigmaRel",0)) if cfg["channelNoiseSemantics"]=="relative_l2_gaussian" else cfg.get("channelNoiseSigmaAbsolute",0)
        key=(cfg["channelNoiseSemantics"],sigma,cfg["partitionStrategy"],cfg.get("seed"),cfg.get("protocolRevision"))
        groups.setdefault(key,[]).append((source,item))
    for group_index,(key,group) in enumerate(groups.items()):
        labels=[item["config"]["conditionId"] for _,item in group];positions=np.arange(len(group))
        fig,axes=plt.subplots(1,2,figsize=(max(12,len(group)*1.4),5));width=.4
        axes[0].bar(positions-width/2,[item["finalAverageAccuracy"] for _,item in group],width,label="average");axes[0].bar(positions+width/2,[item["finalWorstAccuracy"] for _,item in group],width,label="worst");axes[0].set(xticks=positions,xticklabels=labels,ylabel="Final accuracy",ylim=(0,1));plt.setp(axes[0].get_xticklabels(),rotation=65,ha="right");axes[0].legend(fontsize=7)
        for source,item in group:
            with np.load(source.with_name("metrics.npz"),allow_pickle=False) as metrics:
                axes[1].plot(metrics["averageAccuracy"],label=item["config"]["conditionId"])
        for ax in axes:ax.axhline(.5,color="#777",ls="--",lw=1)
        axes[1].set(xlabel="Full ring round",ylabel="Average accuracy",ylim=(0,1));axes[1].legend(fontsize=6)
        fig.suptitle(f"{purpose} — {key[0]} sigma={key[1]} — {key[2]} — seed {key[3]}")
        path=output/f"paired_group_{group_index:02d}.png";_save(fig,path);generated.append(str(path))
    return {"generated":generated,"records":len(records)}


__all__=["generate_comparison","generate_run_plots"]
