#!/usr/bin/env python3
"""Freshly verify and export a fleet fragment including explicit repair evidence."""
import argparse
from pathlib import Path
import shutil
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import fleet_queue
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic,host_name
from basil_core.round_continuation import sha
from scripts.export_fedsharp_fleet import export


def recovered_export(queue,destination):
    queue=Path(queue);destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Use a fresh export destination.')
    temporary=destination.with_name(destination.name+'.recovery-'+uuid.uuid4().hex)
    try:
        count=export(queue,temporary,verify=True)
        registry=fleet_queue.state(queue);plan=read_json(temporary/'plan.json');repairs=0
        for cfg in plan['configs']:
            run=cfg['runId'];entry=registry['runs'][run]
            if entry['owner']!=host_name() or entry['state']!='DONE':raise ValueError('Result ownership changed.')
            root=Path(entry['location']['output']);path=root/run/'checkpoint_recovery.json'
            if not path.exists():continue
            receipt=read_json(path)
            if (receipt.get('status')!='RECOVERED' or receipt.get('runId')!=run or receipt.get('source')!=plan['source']
                    or receipt.get('recipe')!='deterministic_outbound_from_trained_models' or receipt.get('optimizerUpdatesAdded')!=0
                    or receipt.get('outboundHashesMatched')!=cfg['nodes'] or receipt.get('trainedModelHashesMatched')!=cfg['nodes']
                    or receipt.get('repairedCheckpointSha256')!=sha(root/run/'checkpoint.npz')
                    or receipt.get('metricsSha256')!=sha(temporary/run/'metrics.npz')
                    or receipt.get('runRecordSha256')!=sha(temporary/run/'run.json')):
                raise ValueError('Checkpoint recovery provenance differs.')
            digest=receipt['repairScriptSha256'];name='scripts/recover_fedsharp_outbound.py'
            if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid recovery source identity.')
            source=root/'recovery_source_snapshots'/digest/name
            if source.is_symlink() or not source.resolve().is_relative_to((root/'recovery_source_snapshots').resolve()) or sha(source)!=digest:
                raise ValueError('Recovery source snapshot differs.')
            target=temporary/'recovery_source_snapshots'/digest/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
            shutil.copyfile(path,temporary/run/'checkpoint_recovery.json');repairs+=1
        if not repairs:raise ValueError('Expected an explicitly recovered checkpoint in this fragment.')
        atomic(temporary/'SHA256.json',{str(p.relative_to(temporary)):sha(p) for p in temporary.rglob('*')
            if p.is_file() and p!=temporary/'SHA256.json'})
        temporary.rename(destination)
        return dict(configurations=count,recoveryReceipts=repairs,freshInferenceVerified=True)
    finally:
        if temporary.exists():shutil.rmtree(temporary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True,type=Path);parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args();print(recovered_export(args.queue,args.destination))
