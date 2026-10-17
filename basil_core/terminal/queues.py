"""Read-only adapters for existing experiment validators and recovery readers."""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

from basil_core.execution_profile import ROOT, current_profile, profile_path


def adapter(config):
    if config.get('iidCampaignId'):
        return ('basil_core.iid_campaign','basil_core.protocol_checkpoint','scripts/run_iid_condition.py')
    if config.get('experimentProtocol')=='sequential_basil_dirichlet_v1':
        name=('dirichlet_horizon_study' if config.get('horizonStudyRevision') else
              'dirichlet_prior_study' if config.get('lossStudyRevision')=='dirichlet_prior_readiness_diagnostic_v1'
              else 'dirichlet_study')
        script={'dirichlet_horizon_study':'run_dirichlet_horizon_condition.py',
                'dirichlet_prior_study':'run_dirichlet_prior_condition.py',
                'dirichlet_study':'run_dirichlet_condition.py'}[name]
        return ('basil_core.'+name,'basil_core.'+name,'scripts/'+script)
    if config.get('experimentProtocol')=='cart_verified_one_class_v1':
        return ('basil_core.noniid_verified_study','basil_core.noniid_verified_study',
                'scripts/run_noniid_verified_condition.py')
    raise ValueError('No approved checkpoint-aware terminal adapter exists for this protocol.')


def load_queue(path):
    from gui.services.queue_service import QueueService
    from gui.state.experiment_state import to_persisted
    state=QueueService(path).load()
    configs=[to_persisted(entry.config) for entry in state.entries]
    if not configs:raise ValueError('The queue is empty.')
    paths=set()
    for cfg in configs:
        # Credentials belong in the SDK's local authentication store, never a queue.
        if credential_field(cfg):
            raise ValueError('Queue configuration contains a credential field; remove it before use.')
        output=(ROOT/cfg.get('resultDirectory','')).resolve()
        if not output.is_relative_to(ROOT/'newResults'):
            raise ValueError('Research output must be a versioned path under newResults.')
        target=profile_path(output)
        if target in paths:raise ValueError('Duplicate output directory in queue; conflicting writes are forbidden.')
        paths.add(target)
    return configs


def credential_field(value):
    if isinstance(value,dict):
        return any(any(word in str(k).lower() for word in
            ('apikey','api_key','password','access_token','private_key')) or credential_field(v)
            for k,v in value.items())
    return isinstance(value,list) and any(credential_field(v) for v in value)


def tracking_availability(config):
    from importlib.util import find_spec
    return dict(required=bool(config.get('liveTrackingEnabled')),
        sdkAvailable=find_spec('wandb') is not None,
        environmentCredentialPresent='WANDB_API_KEY' in __import__('os').environ,
        authentication='Not probed; required online initialization must succeed before updates.')


def work_config(config):
    copy=dict(config)
    if config.get('experimentProtocol') in ('sequential_basil_iid_v1','cart_verified_one_class_v1'):
        # These validated protocols own 5,000 examples per node, retain partial
        # batches and complete five epochs. Dirichlet uses its explicit weights.
        copy.setdefault('optimizerStepsPerNode',[50]*10)
    return copy


def completed(config):
    """Recorded completion is observable without approving new execution."""
    output=profile_path(ROOT/config['resultDirectory'])
    path=output/'run.json'
    if config.get('experimentProtocol')=='cart_verified_one_class_v1':path=output/'progress.json'
    if not path.exists():return False
    manifest=json.loads(path.read_text())
    if manifest.get('status')!='completed':return False
    if manifest.get('config')!=config:raise ValueError('Completed output has a different configuration.')
    if 'cursor' in manifest:
        if manifest['cursor']!=[config['nRounds'],0,0]:raise ValueError('Incomplete completed CART cursor.')
    elif manifest.get('completedRounds')!=config['nRounds']:
        raise ValueError('Incomplete completed round cursor.')
    with (output/'round_metrics.csv').open() as stream:rows=list(csv.DictReader(stream))
    if [int(r['round']) for r in rows]!=list(range(config['nRounds'])):
        raise ValueError('Completed result lacks contiguous measured rounds.')
    return True


def science(config):
    approval,_,_=adapter(config)
    module=importlib.import_module(approval)
    if approval=='basil_core.iid_campaign':
        module.validate_campaign_condition(config)
        from basil_core.research_protocol import validate_config
        validate_config(config)
    else:module.validate_configuration(config)
    if hasattr(module,'validate_evidence'):module.validate_evidence(config)
    if config.get('experimentProtocol')=='cart_verified_one_class_v1':
        from scripts.run_verified_cart_learning import require_proof
        require_proof();module.require_transport_proof()
        gate=profile_path(ROOT/'gui/worker_state/cart_full_batch_threat_v1/summary.json')
        from basil_core.verified_cart_runner import fingerprints
        saved=json.loads(gate.read_text())
        if (saved.get('status')!='passed' or saved.get('sourceHashes')!=fingerprints()
                or not saved.get('allFiftyUpdatesExactlyMatch') or not saved.get('incorrectProposalRecovered')):
            raise ValueError('Current CART full-batch threat proof required.')
        if current_profile().name=='hpc' and config['gpuMemoryLimitMb']!=current_profile().memory_mib:
            raise ValueError('This CART protocol requires its declared GPU memory cap; choose the matching resource policy.')


def recovery(config):
    _,reader,_=adapter(config)
    output=profile_path(ROOT/config['resultDirectory'])
    manifest='progress.json' if config.get('experimentProtocol')=='cart_verified_one_class_v1' else 'run.json'
    if output.exists() and not (output/manifest).is_file():
        raise ValueError('Existing output without its manifest requires investigation; overwriting is forbidden.')
    # Generic IID reconstruction may replay scientific work; terminal resume
    # intentionally requires a real complete-state checkpoint instead.
    plan=importlib.import_module(reader).recovery_plan(output,config)
    if plan['mode']=='verified_replay':raise ValueError('No compatible stateful checkpoint; terminal resume will not restart from round zero.')
    if plan['mode']=='checkpoint' and current_profile().name=='hpc':
        from basil_core.execution_profile import resource_identity
        saved=json.loads((output/'run.json').read_text())
        numerical=saved.get('numericalExecution',saved.get('runtime',{}).get('numericalExecution',{}))
        if numerical.get('resources')!=resource_identity():
            raise ValueError('Checkpoint runtime belongs to different hardware, software or resources; exact HPC resume is blocked.')
    return plan


def prerequisites(config, include_runtime=True):
    missing=[]
    if current_profile().name=='hpc':
        from importlib.util import find_spec
        if find_spec('psutil') is None:missing.append('HPC worker RSS supervision requires psutil; rerun setup_hpc.sh.')
    if config.get('liveTrackingEnabled'):
        from importlib.util import find_spec
        if find_spec('wandb') is None:missing.append('Required W&B SDK is unavailable; install the tracking dependencies.')
    cache=ROOT/'experiments/cache/cifar10'
    for name in ('x_train_float32.npy','y_train_int32.npy','x_test_float32.npy','y_test_int32.npy'):
        if not (cache/name).is_file():missing.append('Missing CIFAR cache: '+str(cache/name))
    if include_runtime and config.get('executionDevice','CPU')=='GPU':
        try:
            from basil_core.execution_profile import verification_path, resource_identity
            from basil_core.protocol_checkpoint import implementation_fingerprint
            proof=json.loads(verification_path().read_text())
            if proof.get('status')!='passed' or proof.get('implementationSha256')!=implementation_fingerprint():
                raise ValueError('GPU verification source identity is stale.')
            initial=config.get('expectedInitialModelHash',config.get('initialModelHash'))
            if initial and proof.get('initialModelHash')!=initial:
                raise ValueError('Canonical model initialization differs from this queue; fresh scientific approval is required.')
            if current_profile().name=='hpc' and proof['numericalExecution'].get('resources')!=resource_identity():
                raise ValueError('GPU verification belongs to different resources or hardware.')
            from importlib.metadata import version
            if proof['numericalExecution']['tensorflowVersion']!=version('tensorflow'):
                raise ValueError('GPU verification TensorFlow version is stale.')
        except (OSError,ValueError,KeyError,ImportError) as error:
            missing.append('GPU verification required: '+str(error))
        receipt=None
        if config.get('lossStudyRevision')=='dirichlet_prior_readiness_diagnostic_v1':
            receipt='DIRICHLET_READINESS_GPU_VERIFICATION.json';source='basil_core/dirichlet_calibration.py'
        elif config.get('experimentProtocol')=='sequential_basil_dirichlet_v1':
            receipt='SNAPSHOT_ANCHOR_GPU_VERIFICATION.json';source='basil_core/noniid_study.py'
        if receipt:
            import hashlib
            try:
                proof=json.loads(profile_path(ROOT/'gui/worker_state'/receipt).read_text())
                if proof.get('status')!='passed' or proof.get('objectiveSha256')!=hashlib.sha256((ROOT/source).read_bytes()).hexdigest():
                    raise ValueError('Objective verification is stale.')
            except (OSError,ValueError) as error:missing.append(receipt+': '+str(error))
    return missing


def inspect_config(config, include_runtime=True):
    result=dict(name=config.get('experimentName'),condition=config.get('conditionId'),
        protocol=config.get('experimentProtocol'),dataset=config.get('dataset'),
        nonIID=config.get('nonIID'),nodes=config.get('nNodes'),rounds=config.get('nRounds'),
        epochs=config.get('localEpochs'),batchSize=config.get('batchSize'),
        device=config.get('executionDevice','CPU'),output=str(profile_path(ROOT/config['resultDirectory'])),
        trackingRequired=bool(config.get('liveTrackingEnabled')),status='BLOCKED',problems=[])
    result['tracking']=tracking_availability(config)
    try:
        adapter(config)
        if completed(config):
            result.update(status='COMPLETED',recovery=dict(mode='completed',startActivation=config['nRounds']*config['nNodes']))
            return result
        science(config)
        plan=recovery(config)
        result['recovery']=plan
        result['problems']=prerequisites(config,include_runtime)
        if config.get('recoverySourceDirectory') and current_profile().name=='hpc':
            result['problems'].append('Local-to-HPC device transitions are not approved; use a separate fresh HPC condition.')
        result['status']='BLOCKED' if result['problems'] else 'STOPPED' if plan['mode']=='checkpoint' else 'PENDING'
    except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
        result['problems'].append(str(error))
    return result


def inspect_queue(path, include_runtime=True):
    configs=load_queue(path)
    groups={}
    for config in configs:
        groups.setdefault((config.get('experimentProtocol'),config.get('nRounds')),[]).append(config)
    for group in groups.values():
        if len(group)==4 and {c.get('conditionId') for c in group}==set('ABCD'):
            approval,_,_=adapter(group[0])
            module=importlib.import_module(approval)
            if hasattr(module,'verify_campaign'):module.verify_campaign(group)
            elif hasattr(module,'verify_pairing'):module.verify_pairing(group)
    return configs,[inspect_config(c,include_runtime) for c in configs]
