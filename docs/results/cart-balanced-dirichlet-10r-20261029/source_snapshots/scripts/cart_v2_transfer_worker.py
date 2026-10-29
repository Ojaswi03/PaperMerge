#!/usr/bin/env python3
"""Run the frozen worker with an explicitly fingerprinted transfer adapter."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=int,required=True);p.add_argument('--lease-fd',type=int,required=True)
    args=p.parse_args()
    from scripts.cart_v2_slot_worker import adopt_lease
    adopt_lease(args.lease_fd)
    from scripts.run_cart_v2_transfer import program_hashes
    from basil_core.cart_campaign_io import read_json
    import hashlib
    execution=read_json(args.output/'compiled_transfer_execution.json')
    if (execution['programSha256']!=program_hashes() or
        execution['planSha256']!=hashlib.sha256((args.output/'plan.json').read_bytes()).hexdigest()):
        raise ValueError('Compiled-transfer execution differs from verified provenance.')
    from basil_core.execution_profile import current_profile
    receipt=read_json(args.output/f'compiled_transfer_verification_{current_profile().gpu}.json')
    from scripts import run_cart_v2_study as frozen
    setup=frozen.setup_gpu;tf=setup()
    if (receipt['status']!='PASSED' or receipt['programSha256']!=program_hashes()
            or receipt['runtime']!=frozen.runtime(tf)):
        raise ValueError('Device/runtime differs from compiled-transfer verification.')
    # This is a process-local factory substitution, not a source-file edit.
    # Its complete implementation/launcher hashes are recorded separately from
    # the unchanged mathematical/checkpoint contract and checked above.
    import basil_core.sharp_worker as factory
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    original=factory.SharpWorker;write=frozen.write_json
    plan=read_json(args.output/'plan.json');cfg=plan['configs'][args.worker]
    checkpoint=args.output/cfg['runId']/'checkpoint.npz'
    started_cursor=0;started_elapsed=0.
    if checkpoint.exists():
        import numpy as np,json
        with np.load(checkpoint,allow_pickle=False) as arrays:metadata=json.loads(arrays['metadata'].tobytes())
        started_cursor=metadata['cursor'];started_elapsed=metadata['elapsedSeconds']
    def tagged_write(path,value):
        if Path(path).name=='run.json':value=dict(value,runtimeAdapter=dict(
            name='compiled_transfer_v1',programSha256=execution['programSha256'],
            startedCursor=started_cursor,startedElapsedSeconds=started_elapsed))
        return write(path,value)
    try:
        factory.SharpWorker=CompiledTransferWorker;frozen.write_json=tagged_write
        frozen.setup_gpu=lambda:tf
        frozen.run_worker(args)
    finally:factory.SharpWorker=original;frozen.write_json=write;frozen.setup_gpu=setup


if __name__=='__main__':main()
