"""Persistent queue controller using existing scientific worker adapters."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import time
import uuid

from basil_core.execution_profile import ROOT, profile_path, worker_environment, use_profile
from basil_core.terminal import queues
from basil_core.terminal.state import alive, atomic_json, event, identity, lease, logger, read_json, safe_text


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def deferred(config):
    if config.get('horizonStudyRevision') and config.get('nRounds')==500:
        from gui.services.staged_queue_service import dependency_state
        return dependency_state(config)['state'] in ('waiting','needs_validation')
    return False


def validate_plan(configs, plans):
    """Allow only the declared combined queue to defer its 500-round gate."""
    for index,(cfg,plan) in enumerate(zip(configs,plans)):
        if plan['status']!='BLOCKED':continue
        preceding={c.get('conditionId') for c in configs[:index]
            if c.get('nRounds')==100 and c.get('horizonStudyRevision')==cfg.get('horizonStudyRevision')
            and c.get('partitionHash')==cfg.get('partitionHash')
            and c.get('initialModelHash')==cfg.get('initialModelHash')}
        if preceding==set('ABCD') and deferred(cfg):
            # Hardware/data remain immediate requirements even for later stages.
            missing=queues.prerequisites(cfg)
            if missing:raise ValueError('; '.join(missing))
            continue
        raise ValueError(f"{plan['name']} is blocked: {'; '.join(plan['problems'])}")


def entry_state(config, plan, index):
    from gui.work_progress import completed_work
    work=completed_work(queues.work_config(config),plan.get('recovery',{}).get('startActivation',0))
    return dict(id=str(index),name=config['experimentName'],condition=config.get('conditionId'),
        dataset=config.get('dataset'),nonIID=config.get('nonIID'),nodes=config['nNodes'],rounds=config['nRounds'],
        epochs=config['localEpochs'],batchSize=config['batchSize'],device=config.get('executionDevice','CPU'),
        status=plan['status'],completedRounds=plan.get('recovery',{}).get('completedRounds',
            config['nRounds'] if plan['status']=='COMPLETED' else 0),
        output=plan['output'],checkpoint=(dict(activation=work['completedActivations'],
            checkpoint=plan['recovery'].get('checkpoint')) if plan.get('recovery',{}).get('mode')=='checkpoint' else None),
        metrics={},elapsed_seconds=0.,eta_seconds=None,
        tracking=dict(enabled=bool(config.get('liveTrackingEnabled')),status='not_initialized'),
        problems=plan['problems'],**work)


def totals(entries):
    steps=all('totalOptimizerSteps' in e for e in entries)
    denominator=sum(e['totalOptimizerSteps'] if steps else e['totalActivations'] for e in entries)
    numerator=sum(e.get('completedOptimizerSteps',0) if steps else e['completedActivations'] for e in entries)
    return dict(total=len(entries),completed=sum(e['status']=='COMPLETED' for e in entries),
        pending=sum(e['status']=='PENDING' for e in entries),failed=sum(e['status']=='FAILED' for e in entries),
        blocked=sum(e['status']=='BLOCKED' for e in entries),progress=numerator/denominator if denominator else 0.,
        completedActivations=sum(e['completedActivations'] for e in entries),
        totalActivations=sum(e['totalActivations'] for e in entries),
        completedOptimizerSteps=sum(e.get('completedOptimizerSteps',0) for e in entries) if steps else None,
        totalOptimizerSteps=sum(e['totalOptimizerSteps'] for e in entries) if steps else None)


def make_handle(config, sink, policy, directory):
    from gui.services.iid_execution_service import IidExecutionHandle
    approval,_,script=queues.adapter(config)
    module=importlib.import_module(approval)
    with use_profile(policy):
        return IidExecutionHandle(config,sink,_approved=module.gui_approved,
            _plan=queues.recovery(config),_script=ROOT/script,_work_root=directory,
            _environment=worker_environment(config.get('executionDevice','CPU'),
                dict(os.environ,**policy.environment(),PAPERMERGE_SKIP_PLOTS='1'),policy),_headless=True)


class Controller:
    def __init__(self, directory, configs, plans, policies, queue_path, *, resume=False,
                 independent_approved=False, handle_factory=make_handle, wait=time.sleep):
        self.directory=Path(directory);self.configs=configs;self.policies=policies
        self.factory=handle_factory;self.wait=wait;self.active={};self.stopping=False
        self.resume=resume;self.queue_path=Path(queue_path).resolve();self.plans=plans
        self.parallel=len(policies)>1;self.independent_approved=independent_approved
        self.snapshot=dict(version=1,run_id=uuid.uuid4().hex,queue=str(self.queue_path),
            queue_hash=digest(configs),profile=policies[0].name,resources=[asdict(p) for p in policies],
            host=socket.gethostname(),status='PENDING',entries=[entry_state(c,p,i)
                for i,(c,p) in enumerate(zip(configs,plans))],activity=[],elapsed_seconds=0.,eta_seconds=None)

    def record(self, message):
        event(self.snapshot,message)
        self.log.info(safe_text(message))

    def persist(self, force=False):
        now=time.monotonic()
        if not force and now-getattr(self,'last_persist',0.)<1.:return
        self.last_persist=now
        self.snapshot.update(totals=totals(self.snapshot['entries']),updated_wall=time.time())
        atomic_json(self.directory/'status.json',self.snapshot)

    def stopped(self):
        request=read_json(self.directory/'stop.json',{})
        return self.stopping or request.get('run_id')==self.snapshot['run_id']

    def stop(self, *_):
        if self.stopping:return
        self.stopping=True;self.snapshot['status']='STOPPING'
        self.record('Graceful stop requested; waiting for scientific checkpoint boundaries.')
        for index,job in self.active.items():
            entry=self.snapshot['entries'][index]
            if entry['status']=='RUNNING' and job['handle'].active:
                entry['status']='STOPPING';job['handle'].stop()
        self.persist(force=True)

    def validate_parallel(self):
        if not self.parallel:return
        if not self.independent_approved:
            raise ValueError('Parallel scheduling requires --independent-approved after administrator permission and protocol review.')
        if any(not c.get('iidCampaignId') or c.get('recoverySourceDirectory') for c in self.configs):
            raise ValueError('Independent scheduling is currently validated only for fresh IID conditions without stage dependencies.')
        proof=read_json(Path(self.policies[0].output_root)/'verification/single_gpu_execution.json',{})
        if (proof.get('host')!=socket.gethostname() or proof.get('protocol')!='sequential_basil_iid_v1'
                or proof.get('status')!='completed' or not proof.get('config')):
            raise ValueError('Complete a verified single-GPU IID execution in this namespace before independent scheduling.')
        cfg=proof['config']
        from basil_core.protocol_checkpoint import checkpoint_implementation
        if proof.get('implementation')!=checkpoint_implementation(cfg) or not queues.completed(cfg):
            raise ValueError('Single-GPU execution evidence is missing, changed or incompatible.')
        from basil_core.execution_profile import resource_identity
        prior=proof.get('resources',{})
        matching=[p for p in self.policies if p.gpu==prior.get('physicalGpuId')]
        if not matching or prior!=resource_identity(matching[0]):
            raise ValueError('Single-GPU execution evidence must match the current runtime on one selected GPU.')

    def stage(self, config):
        if not deferred(config):return True
        from gui.services.staged_queue_service import dependency_state,validate_completed_100
        if dependency_state(config)['state']=='waiting':raise ValueError('The 500-round prerequisites are incomplete.')
        self.snapshot['status']='VALIDATING';self.record('Stage validation: read-only audit of all completed 100-round prerequisites.')
        with ThreadPoolExecutor(max_workers=1) as pool:
            task=pool.submit(validate_completed_100)
            while not task.done():
                if self.stopped():self.stopping=True
                self.snapshot['elapsed_seconds']=self.prior_elapsed+time.monotonic()-self.started
                self.persist();self.wait(.2)
            task.result()
        self.record('Stage validation passed; prerequisite evidence retained.')
        self.snapshot['status']='RUNNING'
        return not self.stopped()

    def launch(self, index, policy):
        from gui.state.application_state import ApplicationState,ExecutionState
        from gui.state.experiment_state import from_persisted
        from gui.services.execution_service import ExecutionService
        config=self.configs[index];entry=self.snapshot['entries'][index]
        with use_profile(policy):
            queues.science(config);plan=queues.recovery(config)
        if plan['mode']=='completed':
            from gui.work_progress import completed_work
            entry.update(completed_work(queues.work_config(config),entry['totalActivations']))
            entry.update(status='COMPLETED',completedRounds=entry['rounds'],eta_seconds=0.)
            return
        prior=entry['elapsed_seconds']
        state=ApplicationState(experiment=from_persisted(queues.work_config(config)),
            execution=ExecutionState.RUNNING,started_monotonic=time.monotonic(),elapsed_offset=prior,
            total_activations=config['nRounds']*config['nNodes'])
        service=ExecutionService(state)
        terminal=[]
        def sink(item):
            payload=item.payload;kind=item.kind
            if kind.startswith('execution_') and kind in ('execution_completed','execution_failed','execution_stopped'):
                terminal.append(item)
            if kind=='checkpoint_saved':
                entry['checkpoint']=payload
                self.record(f"{entry['name']}: checkpoint saved at activation {payload['activation']}")
            elif kind=='tracking_status':entry['tracking'].update(payload)
            elif kind=='network_updated':
                node=payload.get('payload',{})
                loss=node.get('training',{}).get('localLoss')
                if loss is not None:entry['metrics']['local_training_loss']=loss
                self.record(f"{entry['name']}: node {node.get('nodeId','?')} activation completed")
            elif kind=='network_round_updated':
                row=payload['payload']
                if row.get('fullTestLoss') is not None:entry['metrics']['full_test_loss']=row['fullTestLoss']
                self.record(f"{entry['name']}: round {row['round']} completed; mean accuracy updated")
            elif kind=='execution_restored':
                self.record(f"{entry['name']}: checkpoint restored; {payload['round']} rounds; interrupted activation repeats from its start")
            elif kind=='recovery_updated':self.record(f"{entry['name']}: recovery verification {payload}")
            elif kind=='status_updated':self.record(payload.get('message','Worker status updated'))
            service.post(item)
        entry.update(status='RUNNING',gpu=policy.gpu,gpu_uuid=policy.gpu_uuid,problems=[])
        self.record(f"Experiment started: {entry['name']} on physical GPU {policy.gpu}")
        handle=self.factory(config,sink,policy,self.directory/'workers'/entry['id'])
        self.active[index]=dict(handle=handle,state=state,service=service,terminal=terminal,
                                policy=policy,start=time.monotonic())
        entry['worker']=identity(handle.process.pid) if hasattr(handle,'process') else None
        self.persist(force=True)

    def update_jobs(self):
        for index,job in list(self.active.items()):
            entry=self.snapshot['entries'][index];state=job['state']
            job['handle'].poll();job['service'].dispatch_pending();state.tick()
            if (hasattr(job['handle'],'process') and job['policy'].name=='hpc'
                    and time.monotonic()-job.get('last_memory_check',0.)>=1.):
                job['last_memory_check']=time.monotonic()
                try:
                    import psutil
                    process=psutil.Process(job['handle'].process.pid)
                    processes=[process,*process.children(recursive=True)]
                    rss=sum(p.memory_info().rss for p in processes if p.is_running())/1024**2
                    entry['hostRssMiB']=rss
                    if rss>job['policy'].ram_mib:
                        entry['resource_error']=f'Worker tree RSS {rss:.0f} MiB exceeds budget {job["policy"].ram_mib} MiB.'
                        self.record(entry['resource_error']);self.stop()
                except ImportError:raise RuntimeError('HPC RAM supervision requires psutil; rerun setup_hpc.sh.')
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if state.completed_activations or state.progress:
                from gui.work_progress import completed_work
                entry.update(completed_work(queues.work_config(self.configs[index]),state.completed_activations))
                entry['completedRounds']=state.completed_rounds
            entry.update(elapsed_seconds=state.elapsed_seconds,eta_seconds=state.experiment_eta_seconds)
            for field,key in (('latest_node_accuracy','latest_node_accuracy'),('average_accuracy','mean_node_accuracy'),
                               ('worst_accuracy','worst_node_accuracy')):
                if getattr(state,field) is not None:entry['metrics'][key]=getattr(state,field)
            # The terminal snapshot contains latest measurements, never history arrays.
            state.network_events.clear();del state.accuracy_history[:-20];del state.activation_accuracy_history[:-20]
            if job['handle'].active:continue
            terminal=job['terminal'][-1] if job['terminal'] else None
            if terminal and terminal.kind in ('execution_completed','execution_stopped') and queues.completed(self.configs[index]):
                from gui.work_progress import completed_work
                entry.update(completed_work(queues.work_config(self.configs[index]),entry['totalActivations']))
                entry.update(status='COMPLETED',completedRounds=entry['rounds'],eta_seconds=0.)
                self.record(f"Experiment completed: {entry['name']}")
                if not self.parallel and self.policies[0].name=='hpc':
                    from basil_core.protocol_checkpoint import checkpoint_implementation
                    from basil_core.execution_profile import resource_identity
                    atomic_json(Path(self.policies[0].output_root)/'verification/single_gpu_execution.json',
                        dict(status='completed',host=socket.gethostname(),protocol=self.configs[index]['experimentProtocol'],
                             config=self.configs[index],implementation=checkpoint_implementation(self.configs[index]),
                             resources=resource_identity(job['policy'])))
            else:
                entry['status']='FAILED' if entry.get('resource_error') else 'STOPPED' if self.stopped() or terminal and terminal.kind=='execution_stopped' else 'FAILED'
                entry['error']=safe_text(terminal.payload.get('error') or str(terminal.payload)) if terminal else 'Worker exited without a terminal event.'
                if terminal and terminal.kind=='execution_completed':entry['error']='Worker exited successfully without complete measured result evidence.'
                if entry.get('resource_error'):entry['error']=entry['resource_error']
                self.record(f"{entry['status']}: {entry['name']}: {entry['error']}")
                if entry['status']=='FAILED':self.snapshot['error']=entry['error']
                self.stop()
            del self.active[index]

    def run(self):
        validate_plan(self.configs,self.plans);self.validate_parallel()
        self.directory.mkdir(parents=True,exist_ok=True)
        with lease(self.directory/'queue.lock'), lease(profile_path(ROOT/'gui/worker_state/iid/queue.lock')):
            if self.policies[0].name=='local':
                # Older GUI markers lack process-start identities. A live PID is
                # conservatively a conflict; it is never signaled or removed.
                marker=read_json(ROOT/'gui/.queue_active.json',{})
                if marker.get('pid') and Path(f"/proc/{marker['pid']}").exists():
                    raise ValueError('A GUI or legacy terminal queue is active; stop it before starting this local queue.')
            old=read_json(self.directory/'status.json')
            if old:
                if alive(old.get('owner')) is not False and old.get('status') in ('RUNNING','STOPPING','VALIDATING'):
                    raise ValueError('A queue owner is still alive or unverifiable.')
                if any(alive(e.get('worker')) is not False for e in old.get('entries',[]) if e.get('worker')):
                    raise ValueError('A prior worker is alive or unverifiable; refusing duplicate execution.')
                if not self.resume:raise ValueError('Saved terminal state exists; use resume or a new --state-dir.')
                if old['queue_hash']!=digest(self.configs) or old['resources']!=self.snapshot['resources']:
                    raise ValueError('Saved queue or execution profile changed; exact resume blocked.')
                for previous,plan in zip(old['entries'],self.plans):
                    if previous['status']=='COMPLETED' and plan.get('recovery',{}).get('mode')!='completed':
                        raise ValueError('Completed queue entry lacks matching completed evidence; refusing to repeat it.')
                    if previous['status'] in ('RUNNING','STOPPING','STOPPED','FAILED') and plan.get('recovery',{}).get('mode') not in ('checkpoint','completed','startup_retry'):
                        raise ValueError('Interrupted experiment has no compatible checkpoint; no restart from round zero.')
                for entry,previous in zip(self.snapshot['entries'],old['entries']):
                    entry['elapsed_seconds']=previous.get('elapsed_seconds',0.)
                    if previous.get('gpu') is not None:
                        if previous['gpu'] not in {p.gpu for p in self.policies}:
                            raise ValueError('Saved worker GPU is absent from the execution policy.')
                        entry['gpu']=previous['gpu'];entry['gpu_uuid']=previous.get('gpu_uuid')
                self.snapshot['run_id']=old['run_id']
            elif self.resume:raise ValueError('No saved terminal queue exists to resume.')
            self.log=logger(self.directory/'queue.log')
            self.prior_elapsed=old.get('elapsed_seconds',0.) if old else 0.
            self.started=time.monotonic();self.snapshot.update(status='RUNNING',owner=identity(),started_wall=time.time())
            if self.snapshot['owner'] is None:raise RuntimeError('Cannot establish a safe Linux process identity.')
            atomic_json(self.directory/'plan.json',dict(configs=self.configs,queue=str(self.queue_path),
                policies=[asdict(p) for p in self.policies]))
            (self.directory/'stop.json').unlink(missing_ok=True)
            original={sig:signal.signal(sig,self.stop) for sig in (signal.SIGINT,signal.SIGTERM)}
            self.record('Queue resumed after validation' if self.resume else 'Queue started after validation')
            self.persist(force=True);next_index=0
            try:
                while next_index<len(self.configs) or self.active:
                    if self.stopped() and not self.stopping:self.stop()
                    while not self.stopped() and len(self.active)<len(self.policies) and next_index<len(self.configs):
                        index=next_index;entry=self.snapshot['entries'][index]
                        if entry['status']=='COMPLETED':
                            self.record(f"Already completed; skipped: {entry['name']}");next_index+=1;continue
                        try:
                            if not self.stage(self.configs[index]):break
                            used={j['policy'].gpu for j in self.active.values()}
                            available=[p for p in self.policies if p.gpu not in used
                                and (entry.get('gpu') is None or p.gpu==entry['gpu'])]
                            if not available:break  # A resumed checkpoint retains its actual physical GPU.
                            policy=available[0]
                            self.launch(index,policy)
                        except Exception as error:
                            entry['status']='BLOCKED';entry['problems']=[safe_text(error)]
                            raise
                        next_index+=1
                    self.update_jobs()
                    self.snapshot['elapsed_seconds']=self.prior_elapsed+time.monotonic()-self.started
                    remaining=[e for e in self.snapshot['entries'] if e['status']!='COMPLETED']
                    self.snapshot['eta_seconds']=sum(e['eta_seconds'] for e in remaining) if remaining and all(e['eta_seconds'] is not None for e in remaining) and not self.parallel else None
                    self.persist()
                    if self.stopped() and not self.active:break
                    self.wait(.2)
                failed=any(e['status']=='FAILED' for e in self.snapshot['entries'])
                complete=all(e['status']=='COMPLETED' for e in self.snapshot['entries'])
                self.snapshot['status']='FAILED' if failed else 'COMPLETED' if complete else 'STOPPED' if self.stopped() else 'COMPLETED'
                self.record('Queue '+self.snapshot['status'].lower());self.persist(force=True)
                return 1 if failed else 130 if self.snapshot['status']=='STOPPED' else 0
            except BaseException as error:
                self.snapshot['error']=safe_text(error);self.record('Queue failed: '+safe_text(error));self.stop()
                while self.active:self.update_jobs();self.wait(.2)
                self.snapshot['status']='FAILED';self.persist(force=True)
                raise
            finally:
                for sig,handler in original.items():signal.signal(sig,handler)
