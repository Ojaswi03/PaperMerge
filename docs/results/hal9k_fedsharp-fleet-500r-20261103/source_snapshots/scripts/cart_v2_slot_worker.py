#!/usr/bin/env python3
"""Adopt an inherited, exclusively locked physical-GPU lease for one pool slot."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def adopt_lease(fd):
    from basil_core.execution_profile import current_profile,gpu_inventory
    from basil_core import iid_runtime
    profile=current_profile().bind(gpu_inventory())
    path=Path('/tmp')/f'papermerge-{os.getuid()}-{profile.gpu_uuid}.lock'
    actual=os.fstat(fd);expected=path.stat()
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
        raise RuntimeError('The inherited lease is not for this physical GPU.')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # dup retains the same open-file description and exclusive lock. The
    # original policy accepts this existing lease; no lock check is disabled.
    iid_runtime._gpu_lease=[os.fdopen(os.dup(fd),'a')]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=int,required=True);p.add_argument('--lease-fd',type=int,required=True)
    args=p.parse_args();adopt_lease(args.lease_fd)
    from scripts.run_cart_v2_study import run_worker
    run_worker(args)
