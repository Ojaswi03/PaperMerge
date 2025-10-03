import numpy as np
import random

def split_sensitive_non_sensitive(dataset, alpha=0.05, seed=42):
    """
    Split dataset indices into (non_sensitive, sensitive) with ratio alpha on sensitive.
    Returns: non_sensitive_indices, sensitive_indices
    """
    random.seed(seed)
    total_len = len(dataset)
    indices = list(range(total_len))
    random.shuffle(indices)
    split_point = int(alpha * total_len)
    sensitive = indices[:split_point]
    non_sensitive = indices[split_point:]
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
    for i in range(H):
        add = 1 if i < rem else 0
        end = start + base + add
        batches.append(non_sensitive_indices[start:end])
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
