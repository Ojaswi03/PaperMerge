#!/usr/bin/env python3
"""Validate one prepared IID condition; training requires explicit --execute."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='Start this single research condition (manual authorization only)')
    parser.add_argument('--recover',action='store_true',help='Restore a checkpoint or verify/rebuild saved activations before continuing')
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text())
    from basil_core.iid_runtime import execution_device, prepare_environment, configure_tensorflow, runtime_metadata
    device=execution_device(config)
    prepare_environment(device)
    from basil_core.iid_campaign import gui_approved
    if not gui_approved(config):raise ValueError('Not a valid fixed A/B/C/D configuration.')
    if not args.execute:
        result=dict(status='valid',condition=config['conditionId'],rounds=100,executionStarted=False,startMode='fresh')
        if device=='GPU':result.update(device='GPU',startMode='verified_cpu_state_transition' if config.get('recoverySourceDirectory') else 'fresh')
        print(json.dumps(result))
        return
    # This branch is reached only by the user's explicit Run Queue action.
    from scripts.run_iid_basil_pair import execute,tf
    from basil_core.data.cifar import loadCifar10,_datasetArrays
    from basil_core.iid_study import partition_audit
    configure_tensorflow(tf,device)
    runtime_metadata(tf,config)  # Fail before loading data if GPU verification is stale.
    train,test=loadCifar10(cacheDir=ROOT/'experiments/cache/cifar10')
    data=(*_datasetArrays(train),*_datasetArrays(test))
    audit,_=partition_audit(data[1],data[3],seed=config['seed'])
    execute(config,data,audit,family='research',recover=args.recover)
    print(json.dumps(dict(event='completed',condition=config['conditionId'])),flush=True)


if __name__=='__main__':
    try:main()
    except InterruptedError as error:
        print(json.dumps(dict(event='stopped',error=str(error))),flush=True)
        raise SystemExit(130)
    except Exception as error:
        traceback.print_exc()
        print(json.dumps(dict(event='failed',error=str(error))),flush=True)
        raise SystemExit(1)
