#!/usr/bin/env python3
"""Adopt existing thesis workers and balance future jobs without restarting them."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from basil_core import live_gpu_scheduler as live
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--background',action='store_true')
    p.add_argument('--maximum',type=int,choices=range(1,25),default=24)
    p.add_argument('--poll-seconds',type=int,choices=range(1,31),default=3)
    p.add_argument('--ntfy-topic',default='papermerge-ojaswi')
    args=p.parse_args(); args.output=args.output.resolve()
    if args.ntfy_topic:
        from basil_core.ntfy_notifications import validate_destination
        validate_destination(args.ntfy_topic,'https://ntfy.sh')
    plan,hardware,assigned,digest=live.verified(args.output)
    workers=live.discover(args.output,len(assigned))
    print(json.dumps(dict(verifiedGpus=list(hardware),activeWorkers=len(workers),
        activePerGpu=dict(Counter(v['profile']['gpu'] for v in workers.values())),
        workerGpuBudgetMiB=assigned[0].memory_mib,
        contextReserveMiB=plan['resourceBudget']['context_overhead_mib'],
        hostRamPerWorkerMiB=assigned[0].ram_mib,
        hostHeadroomMiB=plan['resourceBudget']['ram_headroom_mib'],
        maximumPerGpu=args.maximum,planPreserved=True),indent=2),flush=True)
    if not args.execute:
        print('Read-only check. Add --execute --background for a live handover.');return 0
    if args.background:
        forwarded=[s for s in sys.argv[1:] if s!='--background']
        with (args.output/'live_scheduler.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*forwarded],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'Live balancing launcher PID {process.pid}. Log: {args.output}/live_scheduler.log')
        print('Training workers are not restarted. The existing monitor link is unchanged.')
        return 0
    return live.execute(args)


if __name__=='__main__':raise SystemExit(main())
