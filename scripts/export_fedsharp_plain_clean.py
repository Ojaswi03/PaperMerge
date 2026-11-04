#!/usr/bin/env python3
"""Audit the plain-ring activation chain, freshly infer and export twelve runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import fedsharp_plain_clean as study
from scripts import export_fedsharp_clean_controls as common
from scripts.export_cart_campaign import write_matching


def check_routing(root,view,source):
    import numpy as np
    cfg=study.effective_config(view);folder=root/view['runId']
    row=json.loads((folder/'run.json').read_text())
    if row['status']!='COMPLETED' or row['config']!=view or row['source']!=source:
        raise ValueError('Incomplete/mismatched plain ring run: '+view['runId'])
    path=folder/'checkpoint.npz';before=path.stat()
    with np.load(path,allow_pickle=False) as archive:meta=json.loads(archive['metadata'].tobytes())
    after=path.stat()
    signature=lambda s:(s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if signature(before)!=signature(after):raise ValueError('Checkpoint changed while auditing routing.')
    records=meta['telemetry'];total=cfg['nodes']*cfg['rounds']
    if (meta['contract']!=row['contract'] or row['contract']['config']!=cfg or
            meta['cursor']!=total or len(records)!=total):
        raise ValueError('Plain ring checkpoint contract or activation count differs.')
    previous=row['initialHash']
    for cursor,r in enumerate(records):
        round_id,node=divmod(cursor,cfg['nodes']);selection=r['selection'];training=r['training']
        if ((r['round'],r['node'],r['activation'])!=(round_id,node,cursor+1) or
                r['inputHash']!=previous or r['outboundHash']!=r['trainedHash'] or
                r['attackActive'] or r['configuredByzantine'] or r['selectedConfiguredAttackers'] or r['cart'] is not None):
            raise ValueError('Broken predecessor model chain or unexpected attack/CART.')
        expected=[] if cursor==0 else [(node-1)%cfg['nodes']]
        if (selection['selectedSenders']!=expected or selection['selectedCount']!=1 or
                selection['targetCount']!=1 or selection['availableCopies']!=len(expected) or
                selection['bootstrap']!=(cursor==0) or selection['screenApplied'] or
                selection['screenAllFailFallback'] or selection['screenGuardFallback'] or
                selection['ranking'] or selection['transferRule']!='immediate_predecessor' or
                selection['senderRound']!=(None if cursor==0 else round_id if node else round_id-1)):
            raise ValueError('Unexpected screening, ranking, averaging or predecessor version.')
        steps=training['steps']
        if (training['optimizerSteps']!=cfg['localSteps'] or len(steps)!=cfg['localSteps'] or
                training['proximalMu']!=0 or training['rlaCoefficient']!=0 or
                any(t['proximalPenalty']!=0 or t['gradientSquared']!=0 or t['objective']!=t['ce'] for t in steps)):
            raise ValueError('Plain ring training differs from ordinary cross-entropy SGD.')
        previous=r['trainedHash']
    return dict(runId=view['runId'],activationsChecked=total,optimizerStepsChecked=total*cfg['localSteps'],
                predecessorModelChainVerified=True,ordinarySGDVerified=True)


def export(output,destination):
    root=Path(output).resolve();dest=Path(destination).resolve()
    if root==dest or root in dest.parents or dest in root.parents:raise ValueError('Separate export required.')
    plan=json.loads((root/'plan.json').read_text())
    if plan['revision']!=study.REVISION:raise ValueError('Expected the plain clean ring study.')
    study.check_source(plan['source'])
    verified=[check_routing(root,v,plan['source']) for v in plan['configs']]
    audit=dict(status='PLAIN_RING_CHAIN_AND_SGD_VERIFIED',source=plan['source'],verified=verified,
               verifierSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_matching(root/'plain_ring_verification.json',audit)
    result=common.export(root,dest,study_adapter=study,verifier_script='run_fedsharp_plain_clean.py',
                         extra_files=('plain_ring_verification.json',))
    return dict(result,plainRingRoutingVerified=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export(args.output,args.destination),indent=2))
