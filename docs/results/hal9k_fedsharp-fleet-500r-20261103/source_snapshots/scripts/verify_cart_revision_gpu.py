#!/usr/bin/env python3
"""Verify exact transfer, RLA and admitted-anchor objectives on a bound GPU."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(root,gpu,lease_fd):
    from scripts.cart_v2_slot_worker import adopt_lease
    adopt_lease(lease_fd)
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core.cart_revision_study import check_source
    from basil_core.sharp_cart_v2_protocol import fingerprint
    plan=read_json(root/'plan.json');check_source(plan['source'])
    base=root/'base_gpu_checks';base.mkdir(exist_ok=True)
    write_json(base/'plan.json',dict(configs=plan['configs'],source=fingerprint()))
    from scripts.verify_cart_v2_transfer import verify as transfer
    transfer(base,gpu)
    import numpy as np
    from scripts.run_cart_v2_study import runtime
    import tensorflow as tf
    from basil_core.sharp_worker import SharpWorker
    from basil_core.cart_batched_scoring import BatchedAdmittedWorker
    from basil_core.sharp_data import training_data,prepare_partition
    checks=[]
    for dataset in dict.fromkeys(c['dataset'] for c in plan['configs']):
        cfg=next(c for c in plan['configs'] if c['dataset']==dataset)
        old=SharpWorker(cfg,'GPU');new=BatchedAdmittedWorker(cfg,'GPU');flat=old.export()
        np.testing.assert_array_equal(flat,new.export());x,y=training_data(dataset);chunks,_,_=prepare_partition(y,cfg)
        subset=chunks[0][:cfg['batchSize']];images=x[subset];labels=y[subset];batch=np.arange(len(subset))
        copies=flat[None]+np.random.default_rng(291).normal(0,.001,size=(66,len(flat))).astype(np.float32)
        expected_scores=np.asarray([old.probe(v,images,labels) for v in copies])
        actual_scores=new.score_many(copies,images,labels)
        np.testing.assert_array_equal(actual_scores,expected_scores)
        checks.append(dict(dataset=dataset,sequentialCandidateScoresExact=True,candidateCount=66))
        for route,effective in (('fresh',.001),('own',0.)):
            new.admitted_route=route;new.admitted_has_channel=route!='own'
            expected,old_record=old.train(flat,images,labels,0,0,effective,batch)
            actual,new_record=new.train(flat,images,labels,0,0,.001,batch)
            np.testing.assert_array_equal(expected,actual)
            if new_record['steps']!=old_record['steps']:raise AssertionError('Admitted objective differs from explicit SGD.')
            checks.append(dict(dataset=dataset,route=route,weightsExact=True,objectiveTelemetryExact=True))
    write_json(root/f'revision_gpu_{gpu}.json',dict(status='PASSED',source=plan['source'],runtime=runtime(tf),
        checks=checks,baseGate=str(base/f'compiled_transfer_verification_{gpu}.json'),officialTestUsed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--lease-fd',type=int,required=True)
    a=p.parse_args();verify(a.output.resolve(),a.gpu,a.lease_fd)
