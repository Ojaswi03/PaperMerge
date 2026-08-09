import unittest

import numpy as np

from scripts.run_campaign4_worker import _scientific_fingerprint


class Campaign4WorkerFingerprintTests(unittest.TestCase):
    def test_scientific_fingerprint_excludes_only_runtime_measurements(self):
        metrics = {
            "avg_history": np.asarray([0.1, 0.2], dtype=np.float32),
            "runtime_seconds": np.float64(10.0),
            "peak_gpu_bytes": np.int64(100),
            "stage_runtime_seconds": np.asarray([[1.0]], dtype=np.float32),
        }
        telemetry = {
            "selected_sources": np.asarray([[1]], dtype=np.int32),
            "training_seconds": np.asarray([[2.0]], dtype=np.float32),
        }
        first = _scientific_fingerprint(metrics, telemetry)

        changed_timing = dict(metrics, runtime_seconds=np.float64(99.0))
        changed_telemetry_timing = dict(
            telemetry,
            training_seconds=np.asarray([[8.0]], dtype=np.float32),
        )
        self.assertEqual(
            first,
            _scientific_fingerprint(changed_timing, changed_telemetry_timing),
        )

        changed_science = dict(
            metrics,
            avg_history=np.asarray([0.1, 0.3], dtype=np.float32),
        )
        self.assertNotEqual(
            first,
            _scientific_fingerprint(changed_science, telemetry),
        )


if __name__ == "__main__":
    unittest.main()
