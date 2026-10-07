from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
import time
from .experiment_state import ExperimentConfig, validate_experiment

class Workspace(str, Enum):
    DASHBOARD="dashboard"; BUILDER="builder"; QUEUE="queue"; RESULTS="results"; NETWORK="network"

class ExecutionState(str, Enum):
    IDLE="idle"; PREPARING="preparing"; RUNNING="running"; STOPPING="stopping"; STOPPED="stopped"; COMPLETED="completed"; FAILED="failed"

_TRANSITIONS = {
    ExecutionState.IDLE:{ExecutionState.PREPARING}, ExecutionState.PREPARING:{ExecutionState.RUNNING,ExecutionState.FAILED,ExecutionState.STOPPED},
    ExecutionState.RUNNING:{ExecutionState.STOPPING,ExecutionState.COMPLETED,ExecutionState.FAILED}, ExecutionState.STOPPING:{ExecutionState.STOPPED,ExecutionState.FAILED},
    ExecutionState.STOPPED:{ExecutionState.PREPARING,ExecutionState.IDLE}, ExecutionState.COMPLETED:{ExecutionState.PREPARING,ExecutionState.IDLE},
    ExecutionState.FAILED:{ExecutionState.PREPARING,ExecutionState.IDLE},
}

@dataclass
class ApplicationState:
    workspace: Workspace = Workspace.DASHBOARD
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    execution: ExecutionState = ExecutionState.IDLE
    dirty: bool = False
    status: str = "Ready"
    progress: float = 0.0
    current_round: int = 0
    completed_rounds: int = 0
    average_accuracy: float | None = None
    worst_accuracy: float | None = None
    accuracy_history: list[tuple[int, float, float]] = field(default_factory=list)
    activation_accuracy_history: list[tuple[float, float]] = field(default_factory=list)
    latest_node: int | None = None
    latest_node_accuracy: float | None = None
    network_events: list[tuple[str, dict]] = field(default_factory=list)
    worker_count: int = 0
    execution_device: str = ''
    device_name: str = ''
    started_monotonic: float | None = None
    elapsed_seconds: float = 0.0
    elapsed_offset: float = 0.0
    session_start_progress: float = 0.0
    active_experiment_name: str = ""
    active_experiment_rounds: int | None = None
    recovery_completed: int = 0
    recovery_total: int = 0
    queue_started_monotonic: float | None = None
    queue_elapsed_seconds: float = 0.0
    queue_active: bool = False
    listeners: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def subscribe(self, listener: Callable[[], None]) -> None: self.listeners.append(listener)
    def notify(self) -> None:
        for listener in tuple(self.listeners): listener()
    def navigate(self, workspace: Workspace) -> None: self.workspace=Workspace(workspace); self.notify()
    def transition(self, target: ExecutionState, *, notify: bool=True) -> None:
        target=ExecutionState(target)
        if target not in _TRANSITIONS[self.execution]: raise ValueError(f"Illegal execution transition: {self.execution.value} -> {target.value}")
        self.execution=target
        if target is ExecutionState.PREPARING:
            self.progress=0.; self.current_round=0; self.average_accuracy=None; self.worst_accuracy=None
            self.completed_rounds=0
            self.latest_node=None; self.latest_node_accuracy=None
            self.accuracy_history.clear(); self.activation_accuracy_history.clear()
            self.elapsed_offset=0.; self.session_start_progress=0.
            self.recovery_completed=0; self.recovery_total=0
            self.execution_device=''; self.device_name=''
            self.active_experiment_name=self.experiment.experiment_name
            self.active_experiment_rounds=self.experiment.rounds
        if target is ExecutionState.RUNNING:self.started_monotonic=time.monotonic(); self.elapsed_seconds=0.0
        elif target in {ExecutionState.STOPPED,ExecutionState.COMPLETED,ExecutionState.FAILED} and self.started_monotonic is not None:self.elapsed_seconds=self.elapsed_offset+time.monotonic()-self.started_monotonic
        if notify:self.notify()
    def tick(self) -> None:
        if self.execution in {ExecutionState.RUNNING,ExecutionState.STOPPING} and self.started_monotonic is not None:self.elapsed_seconds=self.elapsed_offset+time.monotonic()-self.started_monotonic
        if self.queue_active and self.queue_started_monotonic is not None:self.queue_elapsed_seconds=time.monotonic()-self.queue_started_monotonic
    def begin_queue(self) -> None:
        self.queue_started_monotonic=time.monotonic(); self.queue_elapsed_seconds=0.; self.queue_active=True
    def end_queue(self) -> None:
        self.tick(); self.queue_active=False
    @property
    def experiment_eta_seconds(self) -> float | None:
        work=self.progress-self.session_start_progress
        if work <= 0 or self.execution not in {ExecutionState.RUNNING,ExecutionState.STOPPING}:
            return None
        session_elapsed=max(0.,self.elapsed_seconds-self.elapsed_offset)
        return session_elapsed/work*max(0.,1.-self.progress)
    @property
    def can_run(self) -> bool:
        from basil_core.protocol_compatibility import is_research_protocol
        from basil_core.iid_campaign import gui_approved
        from .experiment_state import to_persisted
        gated=is_research_protocol(self.experiment.extra.get("experimentProtocol")) and self.experiment.extra.get("researchValid",True) and not gui_approved(to_persisted(self.experiment))
        return not gated and not validate_experiment(self.experiment) and self.execution in {ExecutionState.IDLE,ExecutionState.STOPPED,ExecutionState.COMPLETED,ExecutionState.FAILED}
    @property
    def can_stop(self) -> bool: return self.execution in {ExecutionState.PREPARING,ExecutionState.RUNNING}
