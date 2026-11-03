#!/usr/bin/env python3
"""Export this host's completed fleet fragment, preserving original run bytes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import subprocess
import tempfile
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import fleet_queue
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic,host_name
from basil_core.round_continuation import sha,source_record


def verify_owned(current,items,destination,host):
    """Use the existing fresh CPU verifier on only this host's completed runs."""
    groups={};verified=[];sources=None
    for item in items:
        run=item['config']['runId'];root=Path(current['runs'][run]['location']['output'])
        groups.setdefault(root,[]).append(run)
    for root,runs in groups.items():
        plan=read_json(root/'plan.json')
        if sources is not None and sources!=plan['source']:raise ValueError('Verification source differs across roots.')
        sources=plan['source']
        with tempfile.TemporaryDirectory(prefix='papermerge-final-verification-',dir=root.parent) as name:
            view=Path(name);selected=set(runs)
            subset=dict(plan,configs=[cfg for cfg in plan['configs'] if cfg['runId'] in selected])
            if {cfg['runId'] for cfg in subset['configs']}!=selected:raise ValueError('Verification plan coverage differs.')
            atomic(view/'plan.json',subset)
            (view/'source_snapshots').symlink_to(root/'source_snapshots',target_is_directory=True)
            before={}
            for run in runs:
                (view/run).symlink_to(root/run,target_is_directory=True)
                stat=(root/run/'checkpoint.npz').stat();before[run]=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
            subprocess.run([sys.executable,str(ROOT/'scripts/verify_basil_sharp_thesis_results.py'),
                '--output',str(view)],cwd=ROOT,check=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            receipt=read_json(view/'independent_validation.json')
            if (receipt.get('status')!='ALL_COMPARISON_MODELS_VERIFIED' or receipt.get('officialTestUsed') is not False
                    or receipt.get('source')!=sources or len(receipt['verified'])!=len(runs)
                    or {v['runId'] for v in receipt['verified']}!=selected):
                raise ValueError('Fresh verification receipt is incomplete or mismatched.')
            for row in receipt['verified']:
                run=row['runId'];stat=(root/run/'checkpoint.npz').stat()
                if (stat.st_ino,stat.st_size,stat.st_mtime_ns)!=before[run]:raise ValueError('Checkpoint changed during verification.')
                hashes={file:sha(destination/run/file) for file in ('run.json','metrics.npz')}
                if any(sha(root/run/file)!=digest for file,digest in hashes.items()):raise ValueError('Exported result changed during verification.')
                verified.append(dict(row,resultSha256=hashes,executionPlanSha256=sha(root/'plan.json')))
    atomic(destination/'independent_validation.json',dict(status='ALL_COMPARISON_MODELS_VERIFIED',
        host=host,fleetFragment=True,source=sources,officialTestUsed=False,verified=verified,
        verifierSha256=sha(ROOT/'scripts/verify_cart_comparison_results.py'),fleetExporterSha256=sha(Path(__file__)),
        interpretation='Fresh CPU inference of every final client model in this host fragment; original TRAIN holdout, no official TEST data.'))


def export(queue,destination,verify=False):
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
            optimized=record.get('runtimeAdapter',{}).get('executionOptimization')
            if optimized:
                for name,digest in optimized['source'].items():
                    if Path(name).is_absolute() or '..' in Path(name).parts or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                        raise ValueError('Unsafe optimization snapshot identity.')
                    source=root/'optimization_snapshots'/digest/name
                    if source.is_symlink() or not source.resolve().is_relative_to((root/'optimization_snapshots').resolve()) or sha(source)!=digest:
                        raise ValueError('Optimization source snapshot differs.')
                    target=temporary/'optimization_snapshots'/digest/name;target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():shutil.copyfile(source,target)
                sidecar=read_json(root/run/'optimization.json')
                if sidecar!=optimized:raise ValueError('Optimization receipt differs from its run record.')
            for name in ('run.json','metrics.npz','mechanisms.json','continuation.json'):
                source=root/run/name;target=temporary/run/name;target.parent.mkdir(exist_ok=True)
                if source.is_symlink():raise ValueError('Symlink result exports are refused.')
                shutil.copyfile(source,target)
            provenance[run]=dict(host=host,executionPlanSha256=plan_sha,
                originalAssignedHost=entry['originHost'],transferred=entry['originHost']!=host,
                sourceRunId=item['sourceRunId'],sourceCheckpointSha256=item['seedSha256'])
            if optimized:
                shutil.copyfile(root/run/'optimization.json',temporary/run/'optimization.json')
                provenance[run]['executionOptimization']={key:optimized[key] for key in ('revision','source','features','zeroCoefficientQualification')}
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
        if verify:verify_owned(current,items,temporary,host)
        names=[path for path in temporary.rglob('*') if path.is_file()]
        atomic(temporary/'SHA256.json',{str(path.relative_to(temporary)):sha(path) for path in names})
        temporary.rename(destination)
        return len(items)
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--verify',action='store_true',help='Run fresh final-model CPU inference before completing the export.')
    args=parser.parse_args();print(f'Exported {export(args.queue,args.destination,args.verify)} completed configurations from this server.')


if __name__=='__main__':main()
