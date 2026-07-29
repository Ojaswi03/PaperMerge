"""Subprocess pool for isolated Campaign 3 queue execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from gui.campaign3 import write_json_atomic


EVENT_PREFIX = "@@CAMPAIGN_EVENT@@"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path("experiments") / "results3" / "r2" / "workers"


@dataclass
class ActiveWorker:
    lane: int
    config: dict
    process: subprocess.Popen
    config_path: Path
    started_monotonic: float
    reader: threading.Thread


@dataclass(frozen=True)
class WorkerCompletion:
    lane: int
    config: dict
    return_code: int
    elapsed_seconds: float


class CampaignWorkerPool:
    def __init__(
        self,
        *,
        lanes: int,
        gpu_memory_limit_mb: int,
        work_dir: Path | str = WORK_DIR,
        worker_script: Path | str | None = None,
        python_executable: str | None = None,
    ):
        self.lanes = max(1, int(lanes))
        self.gpu_memory_limit_mb = max(0, int(gpu_memory_limit_mb))
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.worker_script = Path(worker_script) if worker_script else (
            PROJECT_ROOT / "scripts" / "run_campaign_worker.py"
        )
        self.python_executable = python_executable or sys.executable
        self.active: dict[int, ActiveWorker] = {}
        self.events: queue.Queue = queue.Queue()
        self._lock = threading.RLock()

    @property
    def available_lanes(self):
        with self._lock:
            return [lane for lane in range(self.lanes) if lane not in self.active]

    @property
    def active_count(self):
        with self._lock:
            return len(self.active)

    @property
    def has_active(self):
        return self.active_count > 0

    def active_configs(self):
        with self._lock:
            return [active.config for active in self.active.values()]

    def launch(self, config: dict, lane: int | None = None) -> ActiveWorker:
        if lane is None:
            available = self.available_lanes
            if not available:
                raise RuntimeError("No isolated worker lane is available.")
            lane = available[0]
        with self._lock:
            if lane in self.active:
                raise RuntimeError(f"Worker lane {lane} is already active.")

        run_id = str(config.get("runId", f"lane-{lane}"))
        config_path = self.work_dir / f"{run_id}.lane-{lane}.json"
        write_json_atomic(config_path, config)
        command = [
            self.python_executable,
            str(self.worker_script),
            "--config",
            str(config_path),
            "--gpu-memory-mb",
            str(self.gpu_memory_limit_mb),
        ]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        active = ActiveWorker(
            lane=lane,
            config=config,
            process=process,
            config_path=config_path,
            started_monotonic=time.monotonic(),
            reader=None,
        )
        reader = threading.Thread(
            target=self._read_output,
            args=(active,),
            name=f"campaign-worker-output-{lane}",
            daemon=True,
        )
        active.reader = reader
        with self._lock:
            self.active[lane] = active
        reader.start()
        return active

    def _read_output(self, active: ActiveWorker):
        if active.process.stdout is None:
            return
        for raw_line in active.process.stdout:
            line = raw_line.rstrip()
            if line.startswith(EVENT_PREFIX):
                try:
                    event = json.loads(line[len(EVENT_PREFIX) :])
                except ValueError:
                    event = {"event": "malformed", "line": line}
                self.events.put(("event", active.lane, active.config, event))
            elif line:
                self.events.put(("line", active.lane, active.config, line))

    def drain_events(self):
        drained = []
        while True:
            try:
                drained.append(self.events.get_nowait())
            except queue.Empty:
                return drained

    def poll_finished(self) -> list[WorkerCompletion]:
        finished = []
        with self._lock:
            active_snapshot = list(self.active.items())
        for lane, active in active_snapshot:
            return_code = active.process.poll()
            if return_code is None:
                continue
            active.reader.join(timeout=2.0)
            if active.process.stdout is not None:
                active.process.stdout.close()
            active.process.wait(timeout=1.0)
            finished.append(
                WorkerCompletion(
                    lane=lane,
                    config=active.config,
                    return_code=int(return_code),
                    elapsed_seconds=time.monotonic() - active.started_monotonic,
                )
            )
            with self._lock:
                self.active.pop(lane, None)
            try:
                active.config_path.unlink()
            except OSError:
                pass
        return finished

    def active_elapsed_by_run_id(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            active_snapshot = list(self.active.values())
        return {
            str(active.config.get("runId", "")): now - active.started_monotonic
            for active in active_snapshot
            if active.config.get("runId")
        }

    def request_stop(self) -> None:
        with self._lock:
            active_snapshot = list(self.active.values())
        for active in active_snapshot:
            if active.process.poll() is None:
                active.process.terminate()

    def kill_remaining(self) -> None:
        with self._lock:
            active_snapshot = list(self.active.values())
        for active in active_snapshot:
            if active.process.poll() is None:
                active.process.kill()

    def wait(self, timeout: float = 30.0) -> list[WorkerCompletion]:
        deadline = time.monotonic() + max(0.0, timeout)
        completed = []
        while self.has_active and time.monotonic() < deadline:
            completed.extend(self.poll_finished())
            if self.has_active:
                time.sleep(0.1)
        return completed
