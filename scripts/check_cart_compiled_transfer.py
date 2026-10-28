#!/usr/bin/env python3
"""Compare original/compiled transfers, full ring trajectories and checkpoint pickup."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def compare(left,right):
    import numpy as np
    if isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            if key!='elapsedSeconds':compare(left[key],right[key])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right):compare(a,b)
    else:assert left==right,(left,right)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import numpy as np
    from scripts.run_cart_v2_study import setup_gpu,fingerprint
    setup_gpu()
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    from basil_core.sharp_study import configuration
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration as cart_config,run
    from basil_core.sharp_protocol import save_state,load_state
    from basil_core.sharp_data import training_data,prepare_partition
    reports=[]
    for dataset in ('mnist','cifar10'):
        x,y=training_data(dataset)
        for partition in ('iid','dirichlet'):
            for seed in (2025,2026,2027):
                cfg=cart_config(guarded_configuration(configuration(dataset,partition,seed,
                    'attack_noise_defense_rla_cart',.01,nodes=4,rounds=2,attack_start=0,
                    train_per_class=20,validation_per_class=5)))
                chunks,held,_=prepare_partition(y,cfg)
                old=SharpWorker(cfg,'GPU');fast=CompiledTransferWorker(cfg,'GPU')
                initial=old.export();np.testing.assert_array_equal(initial,fast.export())
                baseline=run(cfg,old,x,y,chunks,x[held],y[held])
                result=run(cfg,fast,x,y,chunks,x[held],y[held]);compare(baseline,result)
                old.load(initial)
                partial=run(cfg,old,x,y,chunks,x[held],y[held],max_activations=3)
                contract=dict(config=cfg,runtimeAdapterCheck=True);path=args.output/'pickup.npz'
                save_state(path,partial,contract);restored=load_state(path,contract,old.dimension)
                resumed=run(cfg,fast,x,y,chunks,x[held],y[held],state=restored)
                compare(baseline,resumed)
                accepted=sum(bool(r.get('cart') and r['cart'].get('scale',0)>0) for r in baseline['telemetry'])
                row=dict(dataset=dataset,partition=partition,seed=seed,allStateExact=True,
                    checkpointPickupExact=True,acceptedCorrections=accepted)
                reports.append(row);print(row,flush=True)
        # Full receiver window at N=100, real TRAIN partitions, attack active.
        cfg=cart_config(guarded_configuration(configuration(dataset,'iid',2025,
            'attack_noise_defense_rla_cart',.01,nodes=100,rounds=2,attack_start=0)))
        chunks,held,_=prepare_partition(y,cfg);timings={};states={}
        for label,cls in [('original',SharpWorker),('compiled',CompiledTransferWorker)]:
            worker=cls(cfg,'GPU');start=time.perf_counter()
            states[label]=run(cfg,worker,x,y,chunks,x[held],y[held],max_activations=120)
            timings[label]=time.perf_counter()-start
        compare(states['original'],states['compiled'])
        row=dict(dataset=dataset,nodes=100,activations=120,fullWindow=66,allStateExact=True,
            wallSeconds=timings,speedup=timings['original']/timings['compiled'])
        reports.append(row);print(row,flush=True)
        (args.output/'validation.json').write_text(json.dumps(dict(checks=reports,source=fingerprint(),officialTestUsed=False),indent=2)+'\n')
    print('ALL COMPILED TRANSFER CHECKS PASSED.',flush=True)


if __name__=='__main__':main()
