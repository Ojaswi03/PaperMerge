#!/usr/bin/env python3
"""Export this host's completed fleet fragment, preserving original run bytes."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import fleet_queue
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic,host_name
from basil_core.round_continuation import sha,source_record


def export(queue,destination):
    queue,destination=Path(queue).resolve(),Path(destination).resolve()
    if destination.exists():raise FileExistsError('Use a fresh export directory.')
    current=fleet_queue.state(queue);master=read_json(queue/'continuation_plan.json');host=host_name()
    if any(entry['state']!='DONE' for entry in current['runs'].values()):
        raise ValueError('All configurations must complete successfully before exporting final fragments.')
    items=[item for item in master['items'] if current['runs'][item['config']['runId']]['owner']==host]
    if not items:raise ValueError('This server owns no completed fleet results.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.tmp-'+uuid.uuid4().hex);temporary.mkdir()
    try:
        data={};partitions={};profiles=[];provenance={}
        for item in items:
            run=item['config']['runId'];entry=current['runs'][run];root=Path(entry['location']['output'])
            plan_path=root/'plan.json';execution=read_json(plan_path)
            if (execution['host']!=socket.gethostname() or execution['source']!=master['source']
                    or execution['configs'][entry['location']['index']]!=item['config']
                    or execution['continuationPlanSha256']!=sha(queue/'continuation_plan.json')):
                raise ValueError('Result location differs from its immutable execution plan.')
            plan_sha=sha(plan_path);target=temporary/'execution_plans'/(plan_sha+'.json')
            target.parent.mkdir(exist_ok=True)
            if not target.exists():shutil.copyfile(plan_path,target)
            parent=source_record(item);dataset=item['config']['dataset']
            data[dataset]=parent['data'];partitions[run]=parent['partition']
            record=read_json(root/run/'run.json');profiles.append(record['runtime']['profile'])
            for name in ('run.json','metrics.npz','mechanisms.json','continuation.json'):
                source=root/run/name;target=temporary/run/name;target.parent.mkdir(exist_ok=True)
                if source.is_symlink():raise ValueError('Symlink result exports are refused.')
                shutil.copyfile(source,target)
            provenance[run]=dict(host=host,executionPlanSha256=plan_sha,
                originalAssignedHost=entry['originHost'],transferred=entry['originHost']!=host,
                sourceRunId=item['sourceRunId'],sourceCheckpointSha256=item['seedSha256'])
        fragment=dict(revision=master['revision'],host=socket.gethostname(),source=master['source'],
            configs=[item['config'] for item in items],continuations={item['config']['runId']:item for item in items},
            data=data,partitions=partitions,profiles=profiles,continuationQueueId=master['queueId'],fleetFragment=True)
        atomic(temporary/'plan.json',fragment)
        atomic(temporary/'execution_provenance.json',provenance)
        for name,expected in master['source'].items():
            source=queue/'source_snapshots'/name;target=temporary/'source_snapshots'/name
            if sha(source)!=expected:raise ValueError('Numerical source snapshot differs.')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        from reporting.round_continuation import generate
        generate(temporary,temporary/'thesis_figures')
        names=[path for path in temporary.rglob('*') if path.is_file()]
        atomic(temporary/'SHA256.json',{str(path.relative_to(temporary)):sha(path) for path in names})
        temporary.rename(destination)
        return len(items)
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();print(f'Exported {export(args.queue,args.destination)} completed configurations from this server.')


if __name__=='__main__':main()
