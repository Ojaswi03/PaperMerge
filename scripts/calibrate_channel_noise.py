#!/usr/bin/env python3
"""Calibrate research-protocol channel noise without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))

from basil_core.research_protocol import apply_channel_noise, build_model, keyed_rng, parameter_stats, SequentialWorker, evaluate_params
from basil_core.data.cifar import loadCifar10, _datasetArrays


def parse_args():
    parser=argparse.ArgumentParser(); parser.add_argument("--semantics",choices=("relative_l2_gaussian","paper_absolute_gaussian","absolute_coordinate_gaussian"),default="relative_l2_gaussian"); parser.add_argument("--sigma",type=float,nargs="+"); parser.add_argument("--trials",type=int,default=20); parser.add_argument("--checkpoint",default=""); parser.add_argument("--evaluate",action="store_true"); parser.add_argument("--seed",type=int,default=2025); parser.add_argument("--output",help="Optional JSON calibration report"); parser.add_argument("--model-architecture",choices=("basil_paper_cnn","legacy_vgg_cnn"),default="basil_paper_cnn"); return parser.parse_args()


def main():
    args=parse_args(); model=build_model(args.model_architecture,args.seed)
    if args.trials<1:raise SystemExit("trials must be positive")
    sigmas=args.sigma or ((0.2,0.4,0.6) if args.semantics=="relative_l2_gaussian" else (0.005,0.01,0.02))
    if args.checkpoint:
        with np.load(args.checkpoint,allow_pickle=False) as saved:
            params=[saved[key] for key in sorted(saved.files,key=lambda name:int(name.split("_")[-1]))]
    else: params=[value.numpy().astype(np.float32) for value in model.trainable_variables]
    dimension,norm=parameter_stats(params); rows=[]; before=None
    worker=SequentialWorker(model)
    if args.evaluate or args.checkpoint:
        _,test=loadCifar10(cacheDir=PROJECT_ROOT/"experiments/cache/cifar10");test_x,test_y=_datasetArrays(test)
        before=evaluate_params(worker,params,test_x,test_y)[0]
    for sigma in sigmas:
        trials=[]
        for trial in range(args.trials):
            noisy,stats=apply_channel_noise(params,semantics=args.semantics,sigma=sigma,rng=keyed_rng(args.seed,"calibration",args.semantics,sigma,trial)); trials.append(stats)
            if trial==0:after=evaluate_params(worker,noisy,test_x,test_y)[0] if before is not None else None
        rows.append({"channelNoiseSemantics":args.semantics,"checkpointSupplied":bool(args.checkpoint),"requestedSigma":sigma,"parameterCount":dimension,"modelL2Norm":norm,"perCoordinateNoiseStd":float(np.mean([item["coordinateSigma"] for item in trials])),"realizedNoiseL2NormMean":float(np.mean([item["noiseL2Norm"] for item in trials])),"realizedNoiseModelRatioMean":float(np.mean([item["relativeNoiseL2"] for item in trials])),"realizedNoiseModelRatioStd":float(np.std([item["relativeNoiseL2"] for item in trials])),"accuracyBefore":before,"accuracyAfterOnePerturbation":after})
    if args.output:
        from basil_core.artifact_paths import writable_output
        output=writable_output(Path(args.output));output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":main()
