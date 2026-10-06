"""Subprocess pool for isolated Campaign 3 and Campaign 4 execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from gui.baseline_study import write_json_atomic


EVENT_PREFIX = "@@CAMPAIGN_EVENT@@"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path("experiments") / "results3" / "r2" / "workers"
PID_SUFFIX = ".pid.json"
LOG_SUFFIX = ".log"


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def find_orphan_workers(work_dir: Path | str) -> list[dict]:
    """Return records for tracked worker PIDs that are still alive.

    Stale sidecar files whose process has already exited are removed.
    Each returned record includes the sidecar path so the orphan can be
    terminated and its record cleaned afterwards.
    """
    orphans = []
    directory = Path(work_dir)
    if not directory.is_dir():
        return orphans
    for pid_path in sorted(directory.glob(f"*{PID_SUFFIX}")):
        try:
            record = json.loads(pid_path.read_text())
            pid = int(record["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            try:
                pid_path.unlink()
            except OSError:
                pass
            continue
        if not _pid_is_alive(pid):
            try:
                pid_path.unlink()
            except OSError:
                pass
            continue
        record["pidPath"] = pid_path
        orphans.append(record)
    return orphans


def terminate_orphan_workers(
    orphans: list[dict], *, term_timeout: float = 15.0
) -> int:
    """Terminate tracked orphan workers: SIGTERM, wait, then SIGKILL.

    Workers trap SIGTERM as a graceful stop request, so a hard kill follows
    for any process still alive after ``term_timeout`` seconds. Returns the
    number of processes that were signaled.
    """
    signaled = 0
    for record in orphans:
        pid = int(record["pid"])
        try:
            os.kill(pid, signal.SIGTERM)
            signaled += 1
        except OSError:
            continue
    deadline = time.monotonic() + max(0.0, term_timeout)
    remaining = [int(record["pid"]) for record in orphans]
    while remaining and time.monotonic() < deadline:
        remaining = [pid for pid in remaining if _pid_is_alive(pid)]
        if remaining:
            time.sleep(0.1)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for record in orphans:
        pid_path = record.get("pidPath")
        if pid_path is None:
            continue
        try:
            Path(pid_path).unlink()
        except OSError:
            pass
    return signaled


@dataclass
class ActiveWorker:
    lane: int
    config: dict
    process: subprocess.Popen
    config_path: Path
    log_path: Path
    gpu_memory_limit_mb: int
    started_monotonic: float
    reader: threading.Thread


@dataclass(frozen=True)
class WorkerCompletion:
    lane: int
    config: dict
    return_code: int
    elapsed_seconds: float
    gpu_memory_limit_mb: int


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
            PROJECT_ROOT / "scripts" / "run_baseline_worker.py"
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

    def launch(
        self,
        config: dict,
        lane: int | None = None,
        *,
        gpu_memory_limit_mb: int | None = None,
    ) -> ActiveWorker:
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
        log_path = self.work_dir / f"{run_id}.lane-{lane}{LOG_SUFFIX}"
        write_json_atomic(config_path, config)
        memory_limit = (
            self.gpu_memory_limit_mb
            if gpu_memory_limit_mb is None
            else max(0, int(gpu_memory_limit_mb))
        )
        command = [
            self.python_executable,
            str(self.worker_script),
            "--config",
            str(config_path),
            "--gpu-memory-mb",
            str(memory_limit),
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

        pid_path = self.work_dir / f"{run_id}.lane-{lane}{PID_SUFFIX}"
        write_json_atomic(
            pid_path,
            {
                "pid": process.pid,
                "runId": run_id,
                "lane": lane,
                "script": str(self.worker_script),
                "startedAt": time.time(),
            },
        )

        active = ActiveWorker(
            lane=lane,
            config=config,
            process=process,
            config_path=config_path,
            log_path=log_path,
            gpu_memory_limit_mb=memory_limit,
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
        # Line-buffered and flushed after every write: if the worker is
        # killed by a signal (OOM, native segfault/abort in TF/CUDA) rather
        # than exiting through Python, nothing else persists what it printed
        # before dying - the GUI's own log widget dies with it. This file is
        # the only place that output survives for post-mortem diagnosis.
        with open(active.log_path, "w", buffering=1) as log_file:
            log_file.write(
                f"# pid={active.process.pid} lane={active.lane} "
                f"startedAt={time.time()}\n"
            )
            for raw_line in active.process.stdout:
                log_file.write(raw_line)
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
                    gpu_memory_limit_mb=active.gpu_memory_limit_mb,
                )
            )
            with self._lock:
                self.active.pop(lane, None)
            cleanup_paths = [
                active.config_path,
                active.config_path.with_name(
                    active.config_path.stem + PID_SUFFIX
                ),
            ]
            if return_code == 0:
                # Successful run: the log duplicates what the GUI already
                # showed live, so drop it like the other per-run sidecars.
                cleanup_paths.append(active.log_path)
            else:
                # Non-zero or signal-killed exit: keep the log on disk -
                # it may hold the only surviving record of a native crash
                # that never reached run_adaptive_worker.py's own
                # exception handler (which writes status=failed itself).
                self.events.put(
                    (
                        "line",
                        lane,
                        active.config,
                        f"Worker exited with code {return_code}; "
                        f"output preserved at {active.log_path}",
                    )
                )
            for path in cleanup_paths:
                try:
                    path.unlink()
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
