#!/usr/bin/env python3
"""Run the bounded five-epoch diagnostic gate, never production configs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from gui.research_protocol import CONFIG_ROOT, RESULT_ROOT, smoke_configs, preflight_configs
from scripts.verify_research_results import verify_run

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--family",choices=("smoke","preflight"),required=True);parser.add_argument("--jobs",type=int,default=1);args=parser.parse_args()
    configs=smoke_configs() if args.family=="smoke" else preflight_configs()
    if not 1<=args.jobs<=2:parser.error("Use one or two bounded CPU workers.")
    logs=PROJECT_ROOT/RESULT_ROOT/"diagnostic_logs";logs.mkdir(parents=True,exist_ok=True)
    def run(config):
        output=PROJECT_ROOT/RESULT_ROOT/args.family/config["runId"]
        if not (output/"run.json").exists():
            with (logs/f"{config['conditionId']}.log").open("w") as log:
                env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1","TF_CPP_MIN_LOG_LEVEL":"3","OMP_NUM_THREADS":"4"}
                subprocess.run([sys.executable,str(PROJECT_ROOT/"scripts/run_research_protocol.py"),"--config",str(CONFIG_ROOT/args.family/f"{config['conditionId']}.json")],cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify_run(output)
        metadata=json.loads((output/"run.json").read_text())
        return config["conditionId"],metadata
    failures=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run,c):c for c in configs}
        for future in as_completed(futures):
            config=futures[future]
            try:
                condition,data=future.result()
                print(f"{condition}: avg={data['finalAverageAccuracy']:.4f} worst={data['finalWorstAccuracy']:.4f} best={data['finalBestAccuracy']:.4f} {data['runtimeSeconds']:.1f}s",flush=True)
            except Exception as error:
                failures.append(config["conditionId"]);print(f"FAILED {config['conditionId']}: {error}",flush=True)
    print(f"{args.family}: {len(configs)-len(failures)} completed and verified; {len(failures)} failed",flush=True)
    raise SystemExit(bool(failures))

if __name__=="__main__":main()
