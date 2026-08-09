import unittest

from scripts.benchmark_campaign4 import _profile_candidates, _profile_id


class Campaign4PerformanceProfileTests(unittest.TestCase):
    def test_full_candidate_set_keeps_batch_512_and_explicit_backends(self):
        candidates = _profile_candidates(
            include_xla=True,
            include_cuda_malloc_async=True,
            include_mixed_bfloat16=True,
        )
        self.assertTrue(candidates)
        self.assertTrue(
            all(candidate["internalMicroBatchSize"] == 512 for candidate in candidates)
        )
        self.assertIn(
            "mb512_float32_no_xla_cuda_malloc_async",
            {_profile_id(candidate) for candidate in candidates},
        )
        self.assertIn(
            "mb512_mixed_bfloat16_no_xla_cuda_malloc_async",
            {_profile_id(candidate) for candidate in candidates},
        )

    def test_bfloat16_only_candidate_does_not_enable_xla_or_float16(self):
        candidates = _profile_candidates(
            include_xla=False,
            include_cuda_malloc_async=False,
            only_mixed_bfloat16=True,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["precisionProfile"], "mixed_bfloat16")
        self.assertEqual(candidates[0]["gpuAllocator"], "cuda_malloc_async")
        self.assertFalse(candidates[0]["jitCompile"])


if __name__ == "__main__":
    unittest.main()
