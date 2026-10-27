#!/usr/bin/env python3
"""Predeclared bounded comparisons; no automatic novelty or robustness claims.

The default prints a plan. Explicit execution computes CPU initialization and
TRAIN-partition identities, then starts one isolated GPU worker per config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def configurations(args,*,partitions=None,initials=None):
    from basil_core.cart_coherent_study import configuration
    return [configuration(p,s,sigma,v,stage=args.stage,case=case,
        partition_hash=partitions[p,s] if partitions else None,initial_hash=initials[s] if initials else None,
        training_objective=objective,compiled_optimizer=getattr(args,'compiled_optimizer',False))
        for p in args.partitions for s in args.seeds for case in args.cases
        for sigma in (args.sigmas if case=='joint' else (0.,))
        for objective in getattr(args,'training_objectives',['raw_ce'])
        for v in (['none'] if case=='clean' and getattr(args,'clean_controls_only',False) else args.variants)]


def partition_identity(labels,strategy,seed,stage):
    import numpy as np
    from basil_core.research_protocol import keyed_rng,iid_partition_indices
    from basil_core.data.cifar import _dirichletPartition
    eligible=np.arange(len(labels))
    if stage=='smoke':
        eligible=np.concatenate([keyed_rng(seed,'diagnostic_class',c).choice(np.flatnonzero(labels==c),64,replace=False) for c in range(10)])
    local=(iid_partition_indices(labels[eligible],seed) if strategy=='iid' else
        _dirichletPartition(labels[eligible],10,.2,keyed_rng(seed,'dirichlet_partition')))
    chunks=[eligible[c] for c in local]
    if (len(chunks)!=10 or any(not len(c) for c in chunks)
            or not np.array_equal(np.sort(np.concatenate(chunks)),np.sort(eligible))):
        raise ValueError('Partition must cover the selected TRAIN examples exactly once.')
    return hashlib.sha256(b''.join(np.asarray(c,dtype='<i8').tobytes() for c in chunks)).hexdigest()


def execute(args):
    from scripts.run_factorial_study import data,execute as execute_configs
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    from basil_core.research_protocol import build_model,params_hash
    from basil_core.factorial_study import MODEL
    arrays,hashes=data()
    y=arrays[1]
    partitions={(p,s):partition_identity(y,p,s,args.stage) for p in args.partitions for s in args.seeds}
    print('Preflight: computing seeded initial model identities on CPU; training will use fresh GPU workers.',flush=True)
    prepare_environment('CPU')
    import tensorflow as tf
    configure_tensorflow(tf,'CPU')
    with tf.device('/CPU:0'):
        initials={s:params_hash(build_model(MODEL,s).get_weights()) for s in args.seeds}
    configs=configurations(args,partitions=partitions,initials=initials)
    # GPU workers receive fresh startup environments; the parent's CPU-only
    # visibility must never leak into their execution profile.
    del arrays
    return execute_configs(args,configs=configs,prepared_hashes=hashes)


def main():
    from basil_core.cart_coherent_study import SEEDS,SIGMAS,VARIANTS
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group()
    action.add_argument('--execute',action='store_true')
    action.add_argument('--monitor',action='store_true')
    p.add_argument('--stage',choices=('smoke','pilot','mini','confirmation'),default='pilot')
    p.add_argument('--partitions',nargs='+',choices=('iid','dirichlet'),default=['dirichlet'])
    p.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=[2025])
    p.add_argument('--sigmas',nargs='+',type=float,choices=SIGMAS,default=[.01])
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['none','budgeted','coherent'])
    p.add_argument('--training-objectives',nargs='+',choices=('raw_ce','local_prior'),default=['raw_ce'])
    p.add_argument('--compiled-optimizer',action='store_true',help='Execute the same SGD update in a TensorFlow graph for every selected arm.')
    p.add_argument('--cases',nargs='+',choices=('clean','joint'),default=['joint'])
    p.add_argument('--clean-controls-only',action='store_true',
        help='For the clean case, include only the no-CART reference; keep all requested joint variants.')
    p.add_argument('--profile',choices=('local','hpc'),default='local')
    p.add_argument('--gpus',default='0')
    p.add_argument('--recover',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'.papermerge/local-tests/cart-coherent-pilot-v1')
    args=p.parse_args()
    if args.monitor:
        from basil_core.factorial_dashboard import monitor
        monitor(args.output)
    elif args.execute:execute(args)
    else:
        rows=configurations(args)
        print(json.dumps(dict(trainingStarted=False,configs=rows,runs=len(rows),
            researchValid=False,noveltyEstablished=False,performanceValidated=False),indent=2))


if __name__=='__main__':main()
