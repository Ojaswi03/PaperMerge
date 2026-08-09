import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from gui.campaign_workers import (
    CampaignWorkerPool,
    find_orphan_workers,
    terminate_orphan_workers,
)


def _write_fake_worker(directory, *, sleep_seconds=0.05):
    path = Path(directory) / "fake_worker.py"
    path.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import json
            import time

            parser = argparse.ArgumentParser()
            parser.add_argument("--config")
            parser.add_argument("--gpu-memory-mb")
            args = parser.parse_args()
            config = json.load(open(args.config))
            prefix = "@@CAMPAIGN_EVENT@@"
            print(prefix + json.dumps({{"event": "started", "runId": config["runId"]}}), flush=True)
            time.sleep({float(sleep_seconds)})
            print(prefix + json.dumps({{"event": "completed", "runId": config["runId"]}}), flush=True)
            """
        ),
        encoding="utf-8",
    )
    return path


class CampaignWorkerPoolTests(unittest.TestCase):
    def _fake_worker(self, directory, *, sleep_seconds=0.05):
        return _write_fake_worker(directory, sleep_seconds=sleep_seconds)

    def test_two_workers_emit_events_and_remove_temporary_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self._fake_worker(directory)
            work_dir = Path(directory) / "work"
            pool = CampaignWorkerPool(
                lanes=2,
                gpu_memory_limit_mb=4200,
                work_dir=work_dir,
                worker_script=script,
            )
            pool.launch({"runId": "first", "experimentName": "First"})
            pool.launch({"runId": "second", "experimentName": "Second"})
            completions = []
            deadline = time.monotonic() + 5.0
            while pool.has_active and time.monotonic() < deadline:
                completions.extend(pool.poll_finished())
                time.sleep(0.01)
            events = pool.drain_events()

            self.assertEqual({item.config["runId"] for item in completions}, {"first", "second"})
            self.assertTrue(all(item.return_code == 0 for item in completions))
            self.assertEqual(
                {
                    payload["runId"]
                    for kind, _, _, payload in events
                    if kind == "event" and payload["event"] == "completed"
                },
                {"first", "second"},
            )
            self.assertEqual(list(work_dir.glob("*.json")), [])

    def test_stop_terminates_active_worker_without_losing_config_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self._fake_worker(directory, sleep_seconds=5.0)
            pool = CampaignWorkerPool(
                lanes=1,
                gpu_memory_limit_mb=4200,
                work_dir=Path(directory) / "work",
                worker_script=script,
            )
            config = {"runId": "stopped", "experimentName": "Stopped"}
            pool.launch(config)
            pool.request_stop()
            completions = []
            deadline = time.monotonic() + 5.0
            while pool.has_active and time.monotonic() < deadline:
                completions.extend(pool.poll_finished())
                time.sleep(0.01)
            self.assertEqual(len(completions), 1)
            self.assertIs(completions[0].config, config)
            self.assertNotEqual(completions[0].return_code, 0)

    def test_launch_can_use_a_resource_class_specific_memory_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self._fake_worker(directory)
            pool = CampaignWorkerPool(
                lanes=1,
                gpu_memory_limit_mb=7600,
                work_dir=Path(directory) / "work",
                worker_script=script,
            )
            active = pool.launch(
                {"runId": "capped", "experimentName": "Capped"},
                gpu_memory_limit_mb=3120,
            )
            self.assertEqual(active.gpu_memory_limit_mb, 3120)
            completions = []
            deadline = time.monotonic() + 5.0
            while pool.has_active and time.monotonic() < deadline:
                completions.extend(pool.poll_finished())
                time.sleep(0.01)
            self.assertEqual(len(completions), 1)
            self.assertEqual(completions[0].gpu_memory_limit_mb, 3120)


class OrphanWorkerTests(unittest.TestCase):
    def test_launch_writes_pid_sidecar_and_poll_cleans_it(self):
        with tempfile.TemporaryDirectory() as directory:
            script = _write_fake_worker(directory, sleep_seconds=0.5)
            work_dir = Path(directory) / "work"
            pool = CampaignWorkerPool(
                lanes=1,
                gpu_memory_limit_mb=4200,
                work_dir=work_dir,
                worker_script=script,
            )
            active = pool.launch({"runId": "tracked", "experimentName": "Tracked"})
            pid_files = list(work_dir.glob("*.pid.json"))
            self.assertEqual(len(pid_files), 1)
            record = json.loads(pid_files[0].read_text())
            self.assertEqual(record["pid"], active.process.pid)
            self.assertEqual(record["runId"], "tracked")
            deadline = time.monotonic() + 5.0
            while pool.has_active and time.monotonic() < deadline:
                pool.poll_finished()
                time.sleep(0.01)
            self.assertEqual(list(work_dir.glob("*.pid.json")), [])

    def test_find_orphan_workers_detects_live_and_cleans_dead(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory) / "work"
            work_dir.mkdir(parents=True)
            live = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"]
            )
            try:
                (work_dir / "live.lane-0.pid.json").write_text(
                    json.dumps(
                        {
                            "pid": live.pid,
                            "runId": "live",
                            "lane": 0,
                            "script": sys.executable,
                        }
                    )
                )
                (work_dir / "dead.lane-1.pid.json").write_text(
                    json.dumps(
                        {
                            "pid": 2**22 + 12345,
                            "runId": "dead",
                            "lane": 1,
                            "script": sys.executable,
                        }
                    )
                )
                orphans = find_orphan_workers(work_dir)
                self.assertEqual([entry["runId"] for entry in orphans], ["live"])
                self.assertFalse((work_dir / "dead.lane-1.pid.json").exists())
                self.assertTrue((work_dir / "live.lane-0.pid.json").exists())
            finally:
                live.kill()
                live.wait()

    def test_terminate_orphan_workers_kills_and_cleans(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory) / "work"
            work_dir.mkdir(parents=True)
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import signal, time\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "time.sleep(60)",
                ]
            )
            (work_dir / "stubborn.lane-0.pid.json").write_text(
                json.dumps(
                    {
                        "pid": proc.pid,
                        "runId": "stubborn",
                        "lane": 0,
                        "script": sys.executable,
                    }
                )
            )
            orphans = find_orphan_workers(work_dir)
            self.assertEqual(len(orphans), 1)
            killed = terminate_orphan_workers(orphans, term_timeout=0.5)
            self.assertEqual(killed, 1)
            deadline = time.monotonic() + 5.0
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIsNotNone(proc.poll())
            self.assertEqual(list(work_dir.glob("*.pid.json")), [])
            proc.wait()


if __name__ == "__main__":
    unittest.main()
