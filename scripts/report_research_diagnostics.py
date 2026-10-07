#!/usr/bin/env python3
"""Generate an evidence-only diagnostic report and separate non-research plots."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from gui.research_protocol import RESULT_ROOT, PLOT_ROOT, REVISION, production_configs
from basil_core.protocol_compatibility import LEGACY_RESULT_ROOTS, normalize_config
from basil_core.artifact_paths import writable_output

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plots",action="store_true")
    parser.add_argument("--result-root", action="append", help="Read-only input root; defaults to new and retained protocol results")
    parser.add_argument("--output", default="docs/RESEARCH_DIAGNOSTIC_REPORT.md")
    parser.add_argument("--plot-root", default=str(PLOT_ROOT))
    args=parser.parse_args()
    roots=[Path(p) for p in args.result_root] if args.result_root else [PROJECT_ROOT/RESULT_ROOT, PROJECT_ROOT/LEGACY_RESULT_ROOTS[0]]
    rows=[];completed=[];runtime=[]
    for path in sorted({p for root in roots for p in root.rglob("run.json")}):
        data=json.loads(path.read_text());cfg=normalize_config(data.get("config",{}))
        if cfg.get("protocolRevision")!=REVISION:continue
        sigma=cfg["channelNoiseSigmaRelative"] if cfg["channelNoiseSemantics"]=="relative_l2_gaussian" else cfg["channelNoiseSigmaAbsolute"]
        noise=f"{cfg['channelNoiseSemantics']} {sigma:g}" if cfg["useChannelNoise"] else "off"
        trend=retention="unavailable";numbers=["—"]*3;note=data["status"]
        if data["status"]=="completed":
            completed.append(path.parent)
            with np.load(path.with_name("metrics.npz"),allow_pickle=False) as metrics:
                values=metrics["averageAccuracy"];trend="up" if values[-1]>values[0]+.005 else "down" if values[-1]<values[0]-.005 else "flat"
                matrix=metrics["roundNodePerClassAccuracy"][-1];own=np.trace(matrix)/10;other=(matrix.sum()-np.trace(matrix))/90
                retention=f"own {own:.1%}; other {other:.1%}"
            numbers=[f"{data[key]:.2%}" for key in ("finalAverageAccuracy","finalWorstAccuracy","finalBestAccuracy")]
            if cfg["purpose"]=="full_data_preflight":runtime.append((cfg["conditionId"],data["runtimeSeconds"]/5*100))
        elif data["status"]=="failed":note="non-finite loss/gradient; see execution.log" if "finite" in data.get("error","") else data.get("error","failed").splitlines()[0]
        suffix=" (worker recheck)" if "rechecks" in path.parts else " (isolated reference recheck)" if "reference_rechecks" in path.parts else ""
        condition=cfg["conditionId"]+suffix
        rows.append([condition,str(cfg["nRounds"]),str(cfg["localEpochs"]),str(cfg.get("diagnosticSamplesPerClass",5000)),str(cfg["attackStartRound"]),noise,"on" if cfg["snapshotSelection"] else "off",cfg["ebmMode"],*numbers,trend,retention,note])
    lines=["# Five-epoch diagnostic evidence", "", f"Protocol revision: `{REVISION}`. All rows use batch size 512 and five complete local epochs.","", "Smoke runs use 512 examples/class and 100 test images/class; full-data preflights use all 50,000 training and 10,000 test images. Neither family is production research evidence.","", "| Condition | Rounds | Epochs | Samples/class | Attack start | Channel | SS | EBM | Final avg | Worst | Best | Trend | Class retention | Status/notes |", "|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|---|---|"]
    lines.extend("| "+" | ".join(row)+" |" for row in sorted(rows))
    lines += ["", "## Interpretation", "", "Compare the clean controls before interpreting threat defenses. An own-class-dominated node-by-class matrix measures catastrophic forgetting, not a software acceptance failure. Numerical divergence is recorded explicitly; no clipping, weight decay, epoch reduction or test-driven tuning was added to rescue a curve.","", "The initial matrix recorded four non-finite relative-noise failures. An isolated sigma_rel=0.4/full-EBM reference invocation completed near chance, but a subsequent normal-worker recheck failed again. All three observations are retained, not replaced. Invocation/build sensitivity is observed; its exact numerical cause is not isolated. Further reproducibility/stability review is required before production. New manifests record source hashes and runtime versions so future comparisons can distinguish implementation changes.","", "## Runtime planning", "", "Linear CPU extrapolations from five-round clean full-data runs (same epoch-level evaluation frequency; not GPU promises):"]
    lines.extend(f"- {condition}: approximately {seconds/3600:.2f} hours for 100 rounds." for condition,seconds in runtime)
    lines += ["", "Second-order EBM and five candidate evaluations add cost. Use these as baseline estimates, not guarantees for the entire matrix. Production remains explicitly gated.","", "## Prepared production configurations", "", "All are generated, not launched. Legacy scaling is a separately labeled compatibility ablation."]
    lines.extend(f"- `gui/configs/sequential_basil/production/{cfg['conditionId']}.json`" for cfg in production_configs())
    output=writable_output(Path(args.output));output.parent.mkdir(parents=True,exist_ok=True);output.write_text("\n".join(lines)+"\n")
    print(f"Report: {len(rows)} runs, {len(completed)} completed; {output}",flush=True)
    if args.plots:
        from reporting.sequential_basil_plots import generate_run_plots,generate_comparison
        count=0
        for root in completed:
            generated=generate_run_plots(root,plot_root=args.plot_root);count+=len(generated["generated"]);print(f"Plots: {root.name}",flush=True)
        for purpose in ("implementation_smoke_test","full_data_preflight"):
            for root in roots:
                count+=len(generate_comparison(root,plot_root=args.plot_root,purpose=purpose,revision=REVISION)["generated"])
        print(f"Generated {count} plots outside protected trees.",flush=True)

if __name__=="__main__":main()
