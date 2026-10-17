"""GUI control of one explicitly selected serial research process."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque

from basil_core.iid_campaign import ROOT, gui_approved
from basil_core.protocol_checkpoint import recovery_plan
from basil_core.iid_runtime import execution_device
from .execution_service import ExecutionEvent

WORKER_ROOT=ROOT/'gui/worker_state/iid'


class IidExecutionHandle:
    def __init__(self,config,event_sink,*,_approved=None,_plan=None,_script=None,_work_root=None,
                 _environment=None,_headless=False):
        if not (_approved or gui_approved)(config):raise ValueError('Only validated A/B/C/D configurations may use this GUI runner.')
        output=ROOT/config['resultDirectory']
        plan=_plan if _plan is not None else recovery_plan(output,config)
        if plan['mode']=='completed':raise ValueError('This experiment already completed; do not run it again.')
        self._worker_root=Path(_work_root) if _work_root is not None else WORKER_ROOT
        self._worker_root.mkdir(parents=True,exist_ok=True)
        run_id=config['runId'];self.config_path=self._worker_root/f'{run_id}.json';self.pid_path=self._worker_root/f'{run_id}.pid.json'
        self.config=dict(config)
        self._headless=_headless;self._error_lines=deque(maxlen=12)
        self.config_path.write_text(json.dumps(config,indent=2)+'\n')
        self.sink=event_sink;self.events=queue.Queue();self._finished=False;self._stopping=False
        device=execution_device(config)
        from basil_core.execution_profile import worker_environment
        env=worker_environment(device,os.environ)
        env.update(_environment or {})
        script=str(_script or ROOT/'scripts/run_iid_condition.py')
        command=([sys.executable,str(ROOT/'scripts/hpc.py'),'_worker','--runner',script]
            if _headless else [sys.executable,script])
        command+=['--config',str(self.config_path),'--execute']
        if plan['mode']!='fresh' or config.get('recoverySourceDirectory'):command.append('--recover')
        self.process=subprocess.Popen(command,
            cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,
            start_new_session=True)
        self.pid_path.write_text(json.dumps(dict(pid=self.process.pid,runId=run_id,startedAt=time.time()))+'\n')
        self.reader=threading.Thread(target=self._read,name='iid-output-reader',daemon=True);self.reader.start()
        self.sink(ExecutionEvent('worker_status_changed',{'count':1}))
        self.sink(ExecutionEvent('network_started',{'lane':0,'config':self.config}))
    def _read(self):
        path=getattr(self,'_worker_root',WORKER_ROOT)/f"{self.config_path.stem}.log"
        if getattr(self,'_headless',False):
            from basil_core.terminal.state import logger,safe_text
            log=logger(path)
            for line in self.process.stdout:
                line=safe_text(line);log.info(line.rstrip())
                try:self.events.put(json.loads(line))
                except ValueError:self._error_lines.append(line.strip())
        else:
            with path.open('a',buffering=1) as log:
                for line in self.process.stdout:
                    log.write(line)
                    try:self.events.put(json.loads(line))
                    except ValueError:pass
    @property
    def active(self):return not self._finished
    def stop(self):
        self._stopping=True
        if self.process.poll() is None:self.process.send_signal(signal.SIGTERM)
    def _dispatch(self,payload):
        config=getattr(self,'config',{})
        rounds=int(config.get('nRounds',100)); nodes=int(config.get('nNodes',10))
        context={'lane':0,'config':config}
        if payload.get('event')=='device_ready':
            self.sink(ExecutionEvent('device_status_changed',payload))
        elif payload.get('event')=='tracking_status':
            self.sink(ExecutionEvent('tracking_status',payload))
            message=('Offline tracking: metrics stored locally for later sync' if payload.get('status')=='active' and payload.get('mode')=='offline' else
                f"Live plots: {payload['url']}" if payload.get('status')=='active' and payload.get('url') else
                f"Live tracking unavailable; local results continue: {payload.get('error','not configured')}")
            self.sink(ExecutionEvent('status_updated',{'message':message}))
        elif payload.get('event')=='checkpoint_saved':
            self.sink(ExecutionEvent('checkpoint_saved',payload))
        elif 'restored' in payload:
            restored=dict(payload['restored'])
            if config.get('optimizerStepsPerNode'):
                from gui.work_progress import completed_work
                cursor=len(restored['activationHistory'])
                restored.update(completed_work(config,cursor))
                restored['startProgress']=restored['progress']
            self.sink(ExecutionEvent('execution_restored',restored))
        elif isinstance(payload.get('recovery'),dict) and {'completed','total'} <= payload['recovery'].keys():
            # A CLI dry-run/startup also emits a recovery *plan* with fields
            # mode/startActivation/completedRounds. That is not replay progress.
            self.sink(ExecutionEvent('recovery_updated',payload['recovery']))
        elif payload.get('event') in {'failed','stopped'}:
            self.sink(ExecutionEvent('status_updated',{'message':payload.get('error','Execution stopped')}))
        elif 'roundComplete' in payload:
            r=payload['roundComplete']
            from gui.work_progress import completed_work
            self.sink(ExecutionEvent('progress_updated',{'round':r['round']+1,'completedRounds':r['round']+1,
                **completed_work(config,(r['round']+1)*nodes)}))
            self.sink(ExecutionEvent('accuracy_updated',{'average':r['averageAccuracy'],'worst':r['worstNodeAccuracy']}))
            self.sink(ExecutionEvent('network_round_updated',{**context,'payload':{**r,'round':r['round']+1}}))
        elif 'node' in payload and 'round' in payload:
            position=payload['round']+(payload['node']+1)/nodes
            from gui.work_progress import completed_work
            self.sink(ExecutionEvent('progress_updated',{'round':payload['round']+1,
                **completed_work(config,payload['round']*nodes+payload['node']+1)}))
            if payload.get('accuracy') is not None:
                self.sink(ExecutionEvent('node_accuracy_updated',{'node':payload['node'],'accuracy':payload['accuracy'],'round_position':position}))
            update=payload.get('networkUpdate')
            if update is None:
                # Older workers publish accuracy only, not candidate/link details.
                update={'round':payload['round']+1,'nodeId':payload['node'],'globalTestAccuracy':payload.get('accuracy'),'telemetryComplete':False}
            self.sink(ExecutionEvent('network_updated',{**context,'payload':update}))
    def poll(self):
        if self._finished:return
        while not self.events.empty():
            payload=self.events.get_nowait()
            if not isinstance(payload,dict):continue
            self._dispatch(payload)
        code=self.process.poll()
        if code is not None and not self.reader.is_alive() and self.events.empty():
            self._finished=True;self.process.stdout.close();self.process.wait()
            self.pid_path.unlink(missing_ok=True)
            kind='execution_stopped' if self._stopping or code in {130,-signal.SIGTERM,-signal.SIGINT} else 'execution_completed' if code==0 else 'execution_failed'
            self.sink(ExecutionEvent(kind,{'returnCode':code,'logPath':str(getattr(self,'_worker_root',WORKER_ROOT)/f'{self.config_path.stem}.log'),
                'error':'\n'.join(getattr(self,'_error_lines',())) if code else None}))
            self.sink(ExecutionEvent('network_finished',{'lane':0,'status':kind.removeprefix('execution_')}))
            self.sink(ExecutionEvent('worker_status_changed',{'count':0}))
