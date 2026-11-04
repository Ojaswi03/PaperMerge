from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')
import random

def split_sensitive_non_sensitive(dataset, alpha=0.05, seed=42):
    """Split dataset into sensitive and non-sensitive parts based on alpha."""
    random.seed(seed)
    # For simplicity, we assume dataset is indexable and has a length.
    total_len = len(dataset)
    # Create a list of indices and shuffle it to randomize the split.
    indices = list(range(total_len))
    random.shuffle(indices)
    # Determine the split point based on alpha.
    split_point = int(alpha * total_len)
    # The first 'split_point' indices are considered sensitive, the rest non-sensitive.
    sensitive = indices[:split_point]
    # The remaining indices are non-sensitive.
    non_sensitive = indices[split_point:]
    # Return the two lists of indices.
    return non_sensitive, sensitive

def partition_batches(non_sensitive_indices, H):
    """
    Partition non-sensitive indices into H batches (as even as possible).
    """
    random.shuffle(non_sensitive_indices)
    n = len(non_sensitive_indices)
    base = n // H
    rem = n - base * H
    batches = []
    start = 0
    # Distribute the remainder among the first 'rem' batches.
    for i in range(H):
        # Add one more index to the first 'rem' batches to account for any leftover indices.
        add = 1 if i < rem else 0
        # Calculate the end index for the current batch.
        end = start + base + add
        # Append the current batch of indices to the list of batches.
        batches.append(non_sensitive_indices[start:end])
        # Update the start index for the next batch.
        start = end
    return batches

def acds_share(dataset, alpha, H, seed=42, group_map=None):
    """
    Return 'shared indices per node' after anonymous data sharing.
    If group_map is provided as dict {node_id: group_id}, sharing happens within group.
    Otherwise, one global pool.
    """
    non_sensitive, _ = split_sensitive_non_sensitive(dataset, alpha=alpha, seed=seed)
    n_nodes = max(group_map.keys()) + 1 if group_map else H
    # store indices per node id
    node_storage = {node: [] for node in range(n_nodes)}
    if group_map is None:
        batches = partition_batches(non_sensitive, H)
        for i, b in enumerate(batches):
            node_storage[i % n_nodes].extend(b)
    else:
        # within groups: collect & rebroadcast
        groups = {}
        for node, gid in group_map.items():
            groups.setdefault(gid, []).append(node)
        for gid, nodes in groups.items():
            # collect
            pool = []
            for node in nodes:
                pool.extend(non_sensitive[::max(1, len(nodes))])
            # rebroadcast
            for node in nodes:
                node_storage[node].extend(pool)
    return node_storage

def apply_acds(dataset, shared_indices):
    """Return list of dataset samples at shared indices."""
    out = []
    for idx in shared_indices:
        if 0 <= idx < len(dataset):
            out.append(dataset[idx])
    return out


def cyclic_training_share(partitions, labels, *, fraction=0.05, batches=5,
                          seed=2025, non_sensitive=None):
    """Simulate the training-data outcome of BASIL Algorithm 2 (ACDS).

    Select floor(fraction * local_size) examples per owner, circulate shuffled
    batches on one logical ring, then give each receiver the union of its own
    data and every shared example. Unequal sizes use explicit floor rounding.
    Only caller-supplied fitting indices are eligible. All are declared
    non-sensitive unless an explicit eligible-index set is supplied.

    This index-level simulator reproduces the final availability of shared
    examples, not a secure transport or a proven anonymity implementation.
    Do not substitute the historical acds_share helper: it distributes most of
    one global dataset rather than sharing a fraction of each owner's data.
    """
    import hashlib
    import math
    from basil_core.research_protocol import keyed_rng
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('ACDS share fraction must be between zero and one.')
    if type(batches) is not int or batches < 1:
        raise ValueError('ACDS batches must be a positive integer.')
    chunks=[np.asarray(p, dtype=np.int64) for p in partitions]
    y=np.asarray(labels)
    if not chunks or any(p.ndim != 1 or not len(p) for p in chunks):
        raise ValueError('Nonempty one-dimensional owner partitions required.')
    flat=np.concatenate(chunks)
    if len(np.unique(flat)) != len(flat) or np.any(flat < 0) or np.any(flat >= len(y)):
        raise ValueError('Owner fitting partitions must be disjoint valid indices.')
    eligible=set(flat.tolist()) if non_sensitive is None else set(map(int, non_sensitive))
    if not eligible.issubset(set(flat.tolist())):
        raise ValueError('Non-sensitive examples must belong to fitting partitions.')
    selected=[]
    for owner, local in enumerate(chunks):
        candidates=np.asarray([i for i in local if int(i) in eligible], dtype=np.int64)
        count=int(math.floor(fraction * len(local)))
        if count > len(candidates):
            raise ValueError('Insufficient declared non-sensitive data for requested fraction.')
        selected.append(keyed_rng(seed, 'acds_select', owner).permutation(candidates)[:count])
    pieces=[np.array_split(p, batches) for p in selected]
    stored=[set() for _ in chunks]
    circulating=[]
    transmitted=0
    # Each round replaces the sender's preceding batch. An extra drain round
    # delivers the last real batch to every receiver; its dummy batch is empty.
    for h in range(batches + 1):
        for owner in range(len(chunks)):
            previous=set(pieces[owner][h-1].tolist()) if h else set()
            circulating=[i for i in circulating if i not in previous]
            stored[owner].update(circulating)
            incoming=pieces[owner][h].tolist() if h < batches else []
            circulating=keyed_rng(seed, 'acds_shuffle', h, owner).permutation(
                np.asarray(circulating + incoming, dtype=np.int64)).tolist()
            transmitted+=len(circulating)
    shared=set(np.concatenate(selected).tolist())
    expanded=[]
    for owner, local in enumerate(chunks):
        if stored[owner] != shared.difference(selected[owner].tolist()):
            raise AssertionError('Cyclic sharing did not deliver every other owner batch.')
        expanded.append(np.concatenate((local, np.asarray(sorted(stored[owner] - set(local.tolist())), dtype=np.int64))))
    classes=int(y.max())+1
    h=lambda arrays:hashlib.sha256(b''.join(a.astype('<i8').tobytes() for a in arrays)).hexdigest()
    audit=dict(method='BASIL ACDS Algorithm 2 training-availability simulation',
        shareFraction=fraction,batches=batches,groups=1,seed=seed,
        sensitiveDataPolicy='all fitting examples declared non-sensitive' if non_sensitive is None else 'explicit fitting-index eligibility',
        originalCounts=[len(p) for p in chunks],sharedCounts=[len(p) for p in selected],
        sharedUniqueSamples=len(shared),expandedCounts=[len(p) for p in expanded],
        expandedClassCounts=[np.bincount(y[p],minlength=classes).tolist() for p in expanded],
        sharedIndicesSha256=h(selected),expandedPartitionSha256=h(expanded),
        transmittedExampleCopies=transmitted,trainingDataSharing=True,
        cryptographicAnonymityImplemented=False,officialTestUsed=False,
        interpretation='Additional raw fitting-data access; same optimizer-step budget. Parameter attacks begin after sharing; shared samples are not poisoned.')
    return expanded, audit
