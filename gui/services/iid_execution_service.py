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

from basil_core.iid_campaign import ROOT, gui_approved
from basil_core.protocol_checkpoint import recovery_plan
from basil_core.iid_runtime import execution_device
from .execution_service import ExecutionEvent

WORKER_ROOT=ROOT/'gui/worker_state/iid'


class IidExecutionHandle:
    def __init__(self,config,event_sink):
        if not gui_approved(config):raise ValueError('Only validated A/B/C/D configurations may use the IID GUI runner.')
        output=ROOT/config['resultDirectory']
        plan=recovery_plan(output,config)
        if plan['mode']=='completed':raise ValueError('This experiment already completed; do not run it again.')
        WORKER_ROOT.mkdir(parents=True,exist_ok=True)
        run_id=config['runId'];self.config_path=WORKER_ROOT/f'{run_id}.json';self.pid_path=WORKER_ROOT/f'{run_id}.pid.json'
        self.config=dict(config)
        self.config_path.write_text(json.dumps(config,indent=2)+'\n')
        self.sink=event_sink;self.events=queue.Queue();self._finished=False;self._stopping=False
        device=execution_device(config)
        env=dict(os.environ,PAPERMERGE_IID_DEVICE=device,CUDA_VISIBLE_DEVICES='0' if device=='GPU' else '-1',TF_ENABLE_ONEDNN_OPTS='0',TF_DETERMINISTIC_OPS='1',
            TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        command=[sys.executable,str(ROOT/'scripts/run_iid_condition.py'),'--config',str(self.config_path),'--execute']
        if plan['mode']!='fresh' or config.get('recoverySourceDirectory'):command.append('--recover')
        self.process=subprocess.Popen(command,
            cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        self.pid_path.write_text(json.dumps(dict(pid=self.process.pid,runId=run_id,startedAt=time.time()))+'\n')
        self.reader=threading.Thread(target=self._read,name='iid-output-reader',daemon=True);self.reader.start()
        self.sink(ExecutionEvent('worker_status_changed',{'count':1}))
        self.sink(ExecutionEvent('network_started',{'lane':0,'config':self.config}))
    def _read(self):
        with (WORKER_ROOT/f"{self.config_path.stem}.log").open('a',buffering=1) as log:
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
        elif 'restored' in payload:
            self.sink(ExecutionEvent('execution_restored',payload['restored']))
        elif 'recovery' in payload:
            self.sink(ExecutionEvent('recovery_updated',payload['recovery']))
        elif payload.get('event') in {'failed','stopped'}:
            self.sink(ExecutionEvent('status_updated',{'message':payload.get('error','Execution stopped')}))
        elif 'roundComplete' in payload:
            r=payload['roundComplete']
            self.sink(ExecutionEvent('progress_updated',{'round':r['round']+1,'completedRounds':r['round']+1,'progress':(r['round']+1)/rounds}))
            self.sink(ExecutionEvent('accuracy_updated',{'average':r['averageAccuracy'],'worst':r['worstNodeAccuracy']}))
            self.sink(ExecutionEvent('network_round_updated',{**context,'payload':{**r,'round':r['round']+1}}))
        elif 'node' in payload and 'round' in payload:
            position=payload['round']+(payload['node']+1)/nodes
            self.sink(ExecutionEvent('progress_updated',{'round':payload['round']+1,'progress':position/rounds}))
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
            self.sink(ExecutionEvent(kind,{'returnCode':code,'logPath':str(WORKER_ROOT/f'{self.config_path.stem}.log')}))
            self.sink(ExecutionEvent('network_finished',{'lane':0,'status':kind.removeprefix('execution_')}))
            self.sink(ExecutionEvent('worker_status_changed',{'count':0}))
