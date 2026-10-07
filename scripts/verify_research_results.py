#!/usr/bin/env python3
"""Check measured protocol contracts, not scientific accuracy targets."""
from pathlib import Path
import gzip
import json
import sys
import argparse
import numpy as np


def verify_run(run_dir):
    root=Path(run_dir);metadata=json.loads((root/"run.json").read_text());config=metadata["config"]
    if metadata["status"]!="completed":raise ValueError(f"Run did not complete: {root}")
    with gzip.open(root/"activation_telemetry.json.gz","rt") as source:records=json.load(source)
    assert len(records)==config["nRounds"]*10
    previous=None
    for record in records:
        assert len(record["epochs"])==5
        expected=5*int(np.ceil(record["localSampleCount"]/512))
        assert record["optimizerSteps"]==expected
        for epoch in record["epochs"]:
            assert epoch["samplesSeen"]==record["localSampleCount"]
            assert sum(epoch["batchSizes"])==record["localSampleCount"]
            assert np.isfinite(epoch["localTrainingLoss"])
            assert len(epoch["perClassGlobalTestAccuracy"])==10
        assert record["inputHash"]!=record["trainedOutputHash"]
        if not config["snapshotSelection"] and previous:
            link=next(link for link in previous["outgoingLinks"] if link["receiverId"]==record["nodeId"])
            assert record["inputHash"]==link["transmittedHash"]
            if not config["useChannelNoise"] and not previous["attackActive"]:
                assert record["inputHash"]==previous["trainedOutputHash"]
        assert len(record["memorySenderIds"])==5 and len(set(record["memorySenderIds"]))==5
        if record["candidates"]:
            losses=[c["receiverLocalLoss"] for c in record["candidates"]]
            assert record["selectedCandidateLoss"]<=min(losses)+1e-8
        active=config["attackHidden"] and record["nodeId"] in config["resolvedAttackerIds"] and record["round"]>=config["attackStartRound"]
        assert record["attackActive"]==active
        assert record["channelNoiseActive"]==config["useChannelNoise"]
        if config["useChannelNoise"]:
            assert all(link["noiseL2Norm"]>0 for link in record["outgoingLinks"])
            assert len({link["transmittedHash"] for link in record["outgoingLinks"]})==5
        for step in record["optimizerTelemetry"]:
            assert all(np.isfinite(step[key]) for key in ("baseCrossEntropy","gradientNormSquared","ebmObjective","ebmCoefficient"))
            if config["ebmMode"]=="gradient_norm_objective":
                assert np.isclose(step["ebmObjective"],step["baseCrossEntropy"]+step["ebmCoefficient"]*step["gradientNormSquared"],rtol=1e-5,atol=1e-6)
                assert np.isclose(step["ebmCoefficient"],step["effectiveCoordinateSigma"]**2)
        previous=record
    with np.load(root/"metrics.npz",allow_pickle=False) as saved:
        assert saved["activationPerClassAccuracy"].shape==(config["nRounds"],10,10)
        assert saved["epochPerClassAccuracy"].shape==(config["nRounds"],10,5,10)
    return len(records)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs",nargs="+")
    for directory in parser.parse_args().run_dirs:print(f"{directory}: {verify_run(directory)} verified activations")
