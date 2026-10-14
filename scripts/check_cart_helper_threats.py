#!/usr/bin/env python3
"""No-training helper threat audit: bounded outliers are NOT full-node safety."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def audit():
    import numpy as np
    from basil_core.cart_helper_robustness import trimmed_coordinates, full_node_readiness
    # An honest one-class direction can be a minority, indistinguishable by
    # coordinate magnitude alone from a malicious outlier.
    rare=np.zeros((10,1));rare[0,0]=1.
    honest_scores=np.array([[0.,1.]]*3+[[10.,11.]]*3)
    false_scores=np.array([[100.,0.]]*4)
    ordinary=honest_scores.mean(axis=0)
    scores=trimmed_coordinates(np.concatenate([honest_scores,false_scores]),bound=4)
    rng=np.random.default_rng(2025)
    trials=100
    for _ in range(trials):
        honest=rng.normal(size=(6,128))
        bad=rng.uniform(-1e10,1e10,size=(4,128))
        result=trimmed_coordinates(np.concatenate([honest,bad]),bound=4)
        if not (np.all(result>=honest.min(axis=0)) and np.all(result<=honest.max(axis=0))):
            raise AssertionError('Honest coordinate-range bound failed.')
    if not (np.argmin(ordinary)==0 and np.argmin(scores)==1):
        raise AssertionError('Expected forged-score counterexample missing.')
    paths=('basil_core/cart_helper_robustness.py','scripts/check_cart_helper_threats.py',
           'tests/test_cart_helper_robustness.py')
    return dict(full_node_readiness(), mathematicalChecksPassed=True,
        sourceHashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        adversarialCoordinateRangeTrials=trials, testCoordinatesPerTrial=128,
        rareHonestClassDirectionBefore=float(rare.mean()),
        rareHonestClassDirectionAfter=float(trimmed_coordinates(rare,bound=4)[0]),
        honestMeanCandidateScores=ordinary.tolist(), poisonedTrimmedCandidateScores=scores.tolist(),
        honestPreferredCandidate=0, poisonedSelectedCandidate=1,
        scoreRankingReversalObserved=True, gradientClassSignalRemovalObserved=True,
        researchExperimentsStarted=0, optimizerUpdates=0, wandbEnabled=False,
        snapshotNoiseModel='unchanged: absolute coordinate sigma=0.010',
        helperNoiseModel='not defined; no silent reuse of weight sigma',
        conclusion='NOT READY: bounded-payload contracts pass; full-node guarantees do not follow')


def main():
    from basil_core.cart_helper_robustness import full_node_readiness
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Synthetic array checks; no CNN/data/training.')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit() if args.check else full_node_readiness()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
