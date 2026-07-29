from pathlib import Path
import tempfile
import textwrap
import time
import unittest

from gui.campaign_workers import CampaignWorkerPool


class CampaignWorkerPoolTests(unittest.TestCase):
    def _fake_worker(self, directory, *, sleep_seconds=0.05):
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


if __name__ == "__main__":
    unittest.main()
