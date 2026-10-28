"""Workload-specific ETA priors; replaced by matching measured timings.

These are planning estimates from the user's RTX PRO 6000 HPC pilot, not
guaranteed completion times or statistical confidence intervals.
"""
from __future__ import annotations


def prior_rate(cfg):
    condition=cfg['condition'];flags=cfg.get('switches',{})
    defense=flags.get('byzantine_mitigation','defense' in condition)
    rla=flags.get('noise_mitigation','rla' in condition)
    dataset=cfg['dataset']
    if dataset=='mnist':rate=.28 if defense else (.065 if rla else .06)
    else:rate=.95 if defense else (.62 if rla else .31)
    if dataset=='cifar100':rate*=1.5
    if cfg.get('model','paper')!='paper' and dataset!='mnist':rate*=4
    if defense:
        window=cfg.get('window',2*((cfg['nodes']-1)//3))
        rate*=.25+.75*max(1,window)/66
    rate*=max(1,cfg.get('localSteps',5))/5
    return max(rate,.001)
