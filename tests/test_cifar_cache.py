import tempfile
import unittest

import numpy as np

from basil_core.data.cifar import CifarArrayDataset, makeLoaders


class CifarCacheTests(unittest.TestCase):
    def _data(self):
        rng = np.random.default_rng(17)
        train_images = rng.random((100, 32, 32, 3), dtype=np.float32)
        train_labels = np.tile(np.arange(10, dtype=np.int32), 10)
        test_images = rng.random((20, 32, 32, 3), dtype=np.float32)
        test_labels = np.tile(np.arange(10, dtype=np.int32), 2)
        return (
            CifarArrayDataset(train_images, train_labels),
            CifarArrayDataset(test_images, test_labels),
        )

    def test_partition_cache_is_reused_without_changing_the_split(self):
        train, test = self._data()
        with tempfile.TemporaryDirectory() as directory:
            first_loaders, _, first_metadata = makeLoaders(
                train,
                test,
                batchSize=4,
                iid=False,
                nClients=3,
                dirichletAlpha=0.2,
                seed=2026,
                returnMetadata=True,
                cacheDir=directory,
            )
            second_loaders, _, second_metadata = makeLoaders(
                train,
                test,
                batchSize=4,
                iid=False,
                nClients=3,
                dirichletAlpha=0.2,
                seed=2026,
                returnMetadata=True,
                cacheDir=directory,
            )
            for left, right in zip(
                first_metadata["clientIndices"],
                second_metadata["clientIndices"],
            ):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(
                first_metadata["classCounts"],
                second_metadata["classCounts"],
            )
            first_batch = next(iter(first_loaders[0]))
            second_batch = next(iter(second_loaders[0]))
            np.testing.assert_allclose(
                first_batch[0].numpy(),
                second_batch[0].numpy(),
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_array_equal(
                first_batch[1].numpy(),
                second_batch[1].numpy(),
            )


if __name__ == "__main__":
    unittest.main()
