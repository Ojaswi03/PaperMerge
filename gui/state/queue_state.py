from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from .experiment_state import ExperimentConfig

class QueueStatus(str, Enum):
    PENDING="pending"; RUNNING="running"; COMPLETED="completed"; STOPPED="stopped"; FAILED="failed"

@dataclass
class QueueEntry:
    config: ExperimentConfig
    status: QueueStatus = QueueStatus.PENDING
    entry_id: str = field(default_factory=lambda: uuid4().hex)
    error: str = ""
    execution_hint: str = ""
    estimated_remaining_seconds: float | None = None
    completed_rounds: int = 0

@dataclass
class QueueState:
    entries: list[QueueEntry] = field(default_factory=list)
    def add(self, config: ExperimentConfig) -> QueueEntry:
        entry=QueueEntry(config=config); self.entries.append(entry); return entry
    def move(self, entry_id: str, offset: int) -> None:
        index=next(i for i,e in enumerate(self.entries) if e.entry_id==entry_id); target=max(0,min(len(self.entries)-1,index+offset))
        self.entries.insert(target,self.entries.pop(index))
    def remove(self, ids: set[str]) -> None: self.entries[:]=[e for e in self.entries if e.entry_id not in ids]
    def clear_pending(self) -> None: self.entries[:]=[e for e in self.entries if e.status is not QueueStatus.PENDING]
    def retry(self, entry_id: str) -> None:
        entry=next(e for e in self.entries if e.entry_id==entry_id)
        if entry.status not in {QueueStatus.FAILED,QueueStatus.STOPPED}: raise ValueError("Only failed or stopped entries can be retried.")
        entry.status=QueueStatus.PENDING; entry.error=""
    def transition(self, entry_id: str, status: QueueStatus) -> None:
        entry=next(e for e in self.entries if e.entry_id==entry_id); target=QueueStatus(status)
        legal={QueueStatus.PENDING:{QueueStatus.RUNNING},QueueStatus.RUNNING:{QueueStatus.COMPLETED,QueueStatus.STOPPED,QueueStatus.FAILED}}
        if target not in legal.get(entry.status,set()): raise ValueError(f"Illegal queue transition: {entry.status.value} -> {target.value}")
        entry.status=target
