#!/usr/bin/env python3
"""Evaluate one image with all ten final research-protocol node models."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from basil_core.data.cifar import _datasetArrays,loadCifar10
from basil_core.research_protocol import CLASS_NAMES,build_model

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("run_dir"); source=parser.add_mutually_exclusive_group(required=True); source.add_argument("--cifar-index",type=int); source.add_argument("--image"); parser.add_argument("--label",choices=CLASS_NAMES); args=parser.parse_args()
    if args.cifar_index is not None:
        _,test=loadCifar10(cacheDir=Path(__file__).resolve().parents[1]/"experiments/cache/cifar10"); images,labels=_datasetArrays(test)
        if not 0<=args.cifar_index<len(labels):parser.error("CIFAR test index must be between 0 and 9999.")
        image=images[args.cifar_index]; true_label=int(labels[args.cifar_index])
    else:
        image=np.asarray(Image.open(args.image).convert("RGB").resize((32,32)),np.float32)/255.0; true_label=CLASS_NAMES.index(args.label) if args.label else None
    print("True class:",CLASS_NAMES[true_label] if true_label is not None else "unknown")
    metadata=json.loads((Path(args.run_dir)/"run.json").read_text())
    model=build_model(metadata["config"]["modelArchitecture"])
    for node in range(10):
        path=Path(args.run_dir)/f"node_{node}_weights.npz"
        with np.load(path,allow_pickle=False) as saved: params=[saved[f"weight_{index}"] for index in range(len(saved.files))]
        for variable,value in zip(model.trainable_variables,params):variable.assign(value)
        probabilities=__import__("tensorflow").nn.softmax(model(image[None,...],training=False))[0].numpy(); prediction=int(np.argmax(probabilities)); correctness="" if true_label is None else "correct" if prediction==true_label else "incorrect"
        print(f"Node {node} -> {CLASS_NAMES[prediction]:10s} confidence {probabilities[prediction]:.3f} {correctness}")

if __name__=="__main__":main()
