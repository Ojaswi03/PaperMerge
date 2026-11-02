#!/usr/bin/env python3
"""Explain current GPU/RAM admission without launching or stopping processes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import live_gpu_scheduler as live
from basil_core.cart_campaign_io import read_json
from basil_core.cart_worker_admission import Resident,admission


def worker_memory(pid):
    values={}
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key,_,value=line.partition(':')
            if key in ('VmRSS','VmHWM','VmSwap'):values[key]=int(value.split()[0])//1024
    except (OSError,ValueError):pass
    return dict(rssMiB=values.get('VmRSS',0),peakRssMiB=values.get('VmHWM'),swapMiB=values.get('VmSwap',0))


def probe(queue):
    from basil_core.continuation_campaign import host_name
    host=host_name();output=Path(read_json(queue/'host_bindings'/host/'owner.json')['output']).resolve()
    plan,hardware,assigned,digest=live.verified(output)
    inventory=live.gpu_inventory();processes=live.gpu_processes()
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('MemTotal','MemAvailable','SwapTotal','SwapFree'):memory[key+'MiB']=int(value.split()[0])//1024
    from basil_core import fleet_queue
    fleet_state=fleet_queue.state(queue) if fleet_queue.enabled(queue) else None
    if fleet_state and host in fleet_state['hosts']:
        from basil_core.fleet_runner import discover,discover_gates
        fleet_output=Path(fleet_state['hosts'][host]['fleet']);fleet_plan=read_json(fleet_output/'plan.json')
        found=discover(output,fleet_output,plan,fleet_plan)
        specs=list(found.items())
        from dataclasses import replace
        gate_profiles={gpu:replace(p,output_root=str(fleet_output)) for gpu,p in hardware.items()}
        for gpu,item in discover_gates(fleet_output,gate_profiles).items():
            specs.append((f'GPU {gpu} verification',dict(item,profile=dict(gpu=gpu))))
    else:
        specs=[(plan['configs'][index]['runId'],item) for index,item in live.discover(output,len(plan['configs'])).items()]
    workers=[];residents=[]
    for run,item in specs:
        usage=worker_memory(item['pid']);gpu=item['profile']['gpu']
        workers.append(dict(pid=item['pid'],gpu=gpu,runId=run,**usage))
        residents.append(Resident(item['pid'],gpu,usage['rssMiB']+usage['swapMiB']))
    budget=plan['resourceBudget'];maximum=plan['maximumWorkersPerGpu']
    def decisions(current):
        return {gpu:admission(profile,current,inventory,processes,memory['MemAvailableMiB'],
            context_mib=budget['context_overhead_mib'],ram_headroom_mib=budget['ram_headroom_mib'],maximum=maximum)
            for gpu,profile in hardware.items()}
    current=decisions(residents);simulated=list(residents);extra=Counter()
    for decision in current.values():
        decision['limitingResources']=[]
        if 'gpuFreeMiB' in decision and decision['gpuFreeMiB']<decision['gpuRequiredMiB']:
            decision['limitingResources'].append('GPU VRAM')
        if 'ramAvailableMiB' in decision and decision['ramAvailableMiB']<decision['ramRequiredMiB']:
            decision['limitingResources'].append('host RAM')
        if decision['reason']=='worker ceiling':decision['limitingResources'].append('worker ceiling')
    # Reserve a complete future allocation for every simulated startup worker.
    for step in range(maximum*len(hardware)):
        checks=decisions(simulated);fits=[gpu for gpu,row in checks.items() if row['admit']]
        if not fits:break
        counts=Counter(r.gpu for r in simulated);gpu=min(fits,key=lambda g:(counts[g],g))
        simulated.append(Resident(-step-1,gpu,0));extra[gpu]+=1
    campaign=read_json(output/'campaign.json')
    if fleet_state and host in fleet_state['hosts']:
        report=read_json(queue/'fleet/reports'/f'{host}.json')
        campaign=dict(pending=sum(e['state']=='WAITING' and host in e['eligibleHosts'] for e in fleet_state['runs'].values()),
            failed=sum(e['state']=='FAILED' and e['owner']==host for e in fleet_state['runs'].values()),
            scheduler=dict(blocked=report['blocked']))
    return dict(host=host,trainingModified=False,hostMemory=memory,workers=workers,
        workersPerGpu=dict(Counter(r.gpu for r in residents)),
        gpuInventory=[{k:row[k] for k in ('index','memory.total','memory.free','utilization.gpu')} for row in inventory],
        workerGpuBudgetMiB=next(iter(hardware.values())).memory_mib+budget['context_overhead_mib'],
        workerRamBudgetMiB=next(iter(hardware.values())).ram_mib,hostRamHeadroomMiB=budget['ram_headroom_mib'],
        nextWorkerAdmission=current,additionalWorkersWithinCurrentBudgets=dict(extra),
        schedulerBlocked=campaign.get('scheduler',{}).get('blocked'),
        pending=campaign.get('pending'),failed=campaign.get('failed'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(probe(args.queue.resolve()),indent=2),flush=True)


if __name__=='__main__':main()
