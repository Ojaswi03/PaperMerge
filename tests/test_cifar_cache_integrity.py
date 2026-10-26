"""Cached partitions must be complete/disjoint and bound to their label order."""


def test_duplicate_cached_indices_are_not_accepted_as_complete_coverage(tmp_path):
    import numpy as np
    from basil_core.data.cifar import _loadOrCreatePartition
    labels=np.repeat(np.arange(10,dtype=np.int32),10)
    def load(cache):
        return _loadOrCreatePartition(labels,iid=False,nClients=3,alpha=.2,seed=2026,
            rng=np.random.default_rng(2026),cacheDir=cache)
    original=load(tmp_path)
    file=next((tmp_path/'partitions').glob('partition_*.npz'))
    broken=[c.copy() for c in original]
    joined=np.concatenate(broken)
    # Keep the same cardinality while duplicating an index across clients.
    broken[0][0]=joined[-1]
    np.savez(file,**{f'client_{i}':c for i,c in enumerate(broken)})
    actual=load(tmp_path)
    np.testing.assert_array_equal(np.sort(np.concatenate(actual)),np.arange(len(labels)))
    for a,b in zip(original,actual):np.testing.assert_array_equal(a,b)


def test_non_iid_partition_cache_identity_includes_label_order(tmp_path):
    import numpy as np
    from basil_core.data.cifar import _loadOrCreatePartition
    first=np.repeat(np.arange(10,dtype=np.int32),10)
    second=np.roll(first,7)
    def load(labels,cache):
        return _loadOrCreatePartition(labels,iid=False,nClients=3,alpha=.2,seed=2026,
            rng=np.random.default_rng(2026),cacheDir=cache)
    load(first,tmp_path)
    actual=load(second,tmp_path)
    expected=load(second,None)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
