from __future__ import annotations
from dataclasses import dataclass, field
import queue, threading
from pathlib import Path
from typing import Any, Callable
from gui.state import ApplicationState, ExecutionState
from basil_core.protocol_compatibility import normalize_config

@dataclass(frozen=True)
class ExecutionEvent:
    kind: str
    payload: dict[str,Any]=field(default_factory=dict)

class ExecutionService:
    """Thread-safe execution boundary; Tk consumes events on its main thread."""
    def __init__(self,state: ApplicationState, launcher: Callable[[dict],Any] | None=None):
        self.state=state; self.launcher=launcher; self.events: queue.Queue[ExecutionEvent]=queue.Queue(); self._lock=threading.Lock(); self._worker=None
    def start(self, persisted_config: dict) -> None:
        with self._lock:
            if not self.state.can_run or self._worker is not None: raise RuntimeError("An experiment is already active or the configuration is invalid.")
            self.state.transition(ExecutionState.PREPARING)
            if self.launcher is None:
                self.state.transition(ExecutionState.FAILED); raise RuntimeError("No compatible execution launcher is configured.")
            try:self._worker=self.launcher(persisted_config)
            except Exception:
                self.state.transition(ExecutionState.FAILED)
                raise
            self.state.transition(ExecutionState.RUNNING)
    def request_stop(self) -> None:
        with self._lock:
            if not self.state.can_stop: return
            if self.state.execution is ExecutionState.RUNNING: self.state.transition(ExecutionState.STOPPING)
            worker=self._worker
            if worker and hasattr(worker,"stop"): worker.stop()
    def post(self,event: ExecutionEvent) -> None: self.events.put(event)
    def dispatch_pending(self) -> int:
        count=0
        while True:
            try: event=self.events.get_nowait()
            except queue.Empty: break
            count+=1; payload=event.payload
            if event.kind=='device_status_changed':
                self.state.execution_device=payload['device']; self.state.device_name=payload['deviceName']
                self.state.status=f"Using {payload['device']}: {payload['deviceName']} (serial execution)"
            elif event.kind=="execution_restored":
                self.state.progress=payload['progress']; self.state.current_round=payload['round']
                self.state.completed_rounds=payload['round']
                self.state.elapsed_offset=payload['priorRuntimeSeconds']
                self.state.session_start_progress=payload['startProgress']
                self.state.accuracy_history=[tuple(r) for r in payload['accuracyHistory']]
                self.state.activation_accuracy_history=[tuple(r) for r in payload['activationHistory']]
                self.state.recovery_total=payload['recoveryTotal']; self.state.recovery_completed=payload['recoveryCompleted']
                if self.state.accuracy_history:
                    _,self.state.average_accuracy,self.state.worst_accuracy=self.state.accuracy_history[-1]
                self.state.status='Rebuilding and verifying saved state' if self.state.recovery_completed<self.state.recovery_total else 'Checkpoint restored; continuing experiment'
            elif event.kind=="recovery_updated":
                self.state.recovery_completed=payload['completed']; self.state.recovery_total=payload['total']
                self.state.status=(f'Rebuilding saved state: {payload["completed"]}/{payload["total"]} activations verified'
                    if payload['completed']<payload['total'] else 'Saved state verified; continuing experiment')
            elif event.kind=="status_updated":self.state.status=payload['message']
            elif event.kind=="progress_updated":
                self.state.current_round=int(payload.get("round",self.state.current_round)); self.state.progress=float(payload.get("progress",self.state.progress))
                if 'completedRounds' in payload:self.state.completed_rounds=int(payload['completedRounds'])
            elif event.kind=="accuracy_updated":
                self.state.average_accuracy=payload.get("average"); self.state.worst_accuracy=payload.get("worst")
                if self.state.average_accuracy is not None and self.state.worst_accuracy is not None:
                    self.state.accuracy_history.append((self.state.current_round,float(self.state.average_accuracy),float(self.state.worst_accuracy)))
            elif event.kind=="node_accuracy_updated":
                self.state.latest_node=int(payload['node']); self.state.latest_node_accuracy=float(payload['accuracy'])
                self.state.activation_accuracy_history.append((float(payload['round_position']),self.state.latest_node_accuracy))
            elif event.kind in {"execution_completed","execution_failed","execution_stopped"}:
                target={"execution_completed":ExecutionState.COMPLETED,"execution_failed":ExecutionState.FAILED,"execution_stopped":ExecutionState.STOPPED}[event.kind]
                if target is ExecutionState.STOPPED and self.state.execution is ExecutionState.RUNNING:
                    self.state.transition(ExecutionState.STOPPING,notify=False)
                if target is not self.state.execution and target in ({ExecutionState.COMPLETED,ExecutionState.FAILED} if self.state.execution is ExecutionState.RUNNING else {ExecutionState.STOPPED,ExecutionState.FAILED}): self.state.transition(target,notify=False)
                self._worker=None
            elif event.kind=="worker_status_changed": self.state.worker_count=int(payload.get("count",0))
            elif event.kind in {"network_started","network_updated","network_round_updated","network_finished"}: self.state.network_events.append((event.kind,payload))
        if count:self.state.notify()
        return count


class IsolatedWorkerHandle:
    """Adapts the established worker pool tuples into structured GUI events."""
    def __init__(self, config: dict, event_sink: Callable[[ExecutionEvent], None]):
        from gui.worker_pool import CampaignWorkerPool
        config=normalize_config(config)
        version=int(config.get("campaignVersion",0)); research=config.get("experimentProtocol")=="sequential_basil_one_class_v1"
        if version not in {3,4} and not research: raise ValueError("No isolated worker is registered for this configuration.")
        if research:
            from basil_core.research_protocol import validate_config
            validate_config(config)
            if config.get("researchValid",True):raise ValueError("Production is gated; review diagnostics and explicitly approve via CLI.")
        root=Path("experiments")/("research_protocol_results/workers" if research else "worker_state/adaptive" if version==4 else "worker_state/baseline")
        script=Path("scripts")/("run_research_protocol.py" if research else "run_adaptive_worker.py" if version==4 else "run_baseline_worker.py")
        self.pool=CampaignWorkerPool(lanes=1,gpu_memory_limit_mb=int(config.get("gpuMemoryLimitMb",0)),work_dir=root,worker_script=script)
        self.event_sink=event_sink; self.pool.launch(config)
    def stop(self) -> None: self.pool.request_stop()
    @property
    def active(self) -> bool: return self.pool.has_active
    def poll(self) -> None:
        for kind,_lane,_config,payload in self.pool.drain_events():
            if kind!="event": continue
            event=payload.get("event")
            if event=="round_complete":
                total=max(1,int(payload.get("totalRounds",_config.get("nRounds",1)))); current=int(payload.get("round",0))+(1 if _config.get("experimentProtocol") else 0)
                self.event_sink(ExecutionEvent("progress_updated",{"round":current,"completedRounds":current,"progress":current/total}))
                worst=float(payload.get("worstAccuracy",0))
                if "worstNodeAccuracy" in payload:worst=float(payload["worstNodeAccuracy"])
                self.event_sink(ExecutionEvent("accuracy_updated",{"average":float(payload.get("averageAccuracy",0)),"worst":worst}))
            elif event=="activation_complete":
                total=max(1,int(_config.get("nRounds",1))*int(_config.get("nNodes",10))); current=int(payload.get("round",0))*10+int(payload.get("nodeId",0))+1
                self.event_sink(ExecutionEvent("progress_updated",{"round":int(payload.get("round",0))+1,"progress":current/total}))
            elif event=="node_update": self.event_sink(ExecutionEvent("network_updated",{"lane":_lane,"config":_config,"payload":payload}))
            elif event=="started":
                self.event_sink(ExecutionEvent("worker_status_changed",{"count":1})); self.event_sink(ExecutionEvent("network_started",{"lane":_lane,"config":_config}))
            elif event in {"failed","stopped","completed","skipped"}:
                mapped={"failed":"execution_failed","stopped":"execution_stopped","completed":"execution_completed","skipped":"execution_completed"}[event]
                self.event_sink(ExecutionEvent(mapped,payload)); self.event_sink(ExecutionEvent("network_finished",{"lane":_lane,"status":event}))
        completed=self.pool.poll_finished()
        if completed:
            self.event_sink(ExecutionEvent("worker_status_changed",{"count":0}))
            # Native failures may not have emitted a structured failure event.
            if any(item.return_code!=0 for item in completed): self.event_sink(ExecutionEvent("execution_failed",{"returnCode":completed[0].return_code}))
