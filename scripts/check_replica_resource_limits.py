#!/usr/bin/env python3
"""Bounded kernel memory-limit proof; no TensorFlow, training or GPU usage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import selectors
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def check():
    from basil_core.replica_resources import ResourceScope,MIB,require_host_headroom
    budget=require_host_headroom()
    scope=ResourceScope.create(memory_bytes=64*MIB,tasks=16)
    code="print('ready',flush=True);input();data=bytearray(128*1024*1024);print('unexpected-allocation',flush=True)"
    p=subprocess.Popen(scope.command([sys.executable,'-I','-u','-c',code]),
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with selectors.DefaultSelector() as s:
            s.register(p.stdout,selectors.EVENT_READ)
            if not s.select(15):raise TimeoutError('Resource probe startup failed.')
        first=p.stdout.readline()
        if first.strip()!=b'ready':
            _,err=p.communicate(timeout=5)
            raise RuntimeError('Resource probe did not start: '+err.decode(errors='replace')[-500:])
        limits=scope.verify()
        out,err=p.communicate(b'go\n',timeout=15)
        if p.returncode==0 or b'unexpected-allocation' in out:
            raise AssertionError('Kernel RAM ceiling failed.')
        return dict(status='passed',kernelLimits=limits,hostBudget=budget,
            requestedAllocationBytes=128*MIB,exitCode=p.returncode,
            allocationCompleted=False,stderr=err.decode(errors='replace')[-500:],
            researchExperimentsStarted=0,scopeStopped=True)
    finally:
        scope.stop()
        if p.poll() is None:p.kill()
        p.wait(timeout=10)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();result=check() if a.check else dict(researchExperimentsStarted=0,checkRequested=False)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
