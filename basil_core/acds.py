import numpy as np
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
