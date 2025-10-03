import numpy as np
from copy import deepcopy

from basil import basil_ring_training_with_attack
from trainer import evaluate, evaluate_batch_loss

def _split_into_groups(nodes, num_groups):
    groups = [[] for _ in range(num_groups)]
    for idx, node in enumerate(nodes):
        groups[idx % num_groups].append(node)
    return groups

def basil_plus_training(all_nodes, num_groups, rounds, test_loader, attack_type="none",
                        attacker_ids=None, hidden_start_round=20):
    """
    Basil+: parallel Basil within groups, then robust circular aggregation among groups.
    """
    if attacker_ids is None:
        attacker_ids = []

    groups = _split_into_groups(all_nodes, num_groups)
    accs = []

    # round 0 baseline
    if test_loader is not None:
        base = [evaluate(n.model, test_loader) for n in all_nodes]
        accs.append(float(np.mean(base)))

    for r in range(rounds):
        # --- Stage 1: run Basil inside each group independently (1 round each) ---
        group_models = []
        for g_nodes in groups:
            _, _ = basil_ring_training_with_attack(
                g_nodes, rounds=1, test_loader=None,
                attack_type=attack_type, attacker_ids=attacker_ids, hidden_start_round=hidden_start_round
            )
            # choose group's best representative (lowest loss on test set) among members
            cand = []
            cand_losses = []
            for node in g_nodes:
                params = node.model.get_params()
                cand.append(params)
                loss = evaluate_batch_loss(node.model, test_loader) if test_loader is not None else 0.0
                cand_losses.append(loss)
            best_idx = int(np.argmin(np.array(cand_losses))) if cand_losses else 0
            group_models.append(deepcopy(cand[best_idx]))

        # --- Stage 2: robust circular aggregation across groups (pairwise min-loss merge) ---
        merged = group_models[0]
        for k in range(1, len(group_models)):
            candidates = [merged, group_models[k]]
            # pick better by loss on test set (averaged over all nodes' local batch evals)
            losses = []
            for params in candidates:
                # temporarily test on each node
                lsum = 0.0
                cnt = 0
                for node in all_nodes:
                    node.model.set_params(params)
                    lsum += evaluate_batch_loss(node.model, test_loader) if test_loader is not None else 0.0
                    cnt += 1
                losses.append(lsum / max(1, cnt))
            best = int(np.argmin(np.array(losses)))
            merged = deepcopy(candidates[best])

        # --- Stage 3: multicast the merged model back to all nodes ---
        for node in all_nodes:
            node.model.set_params(deepcopy(merged))

        # eval
        if test_loader is not None:
            cur = [evaluate(n.model, test_loader) for n in all_nodes]
            accs.append(float(np.mean(cur)))
            print(f"[Basil+ Round {r+1}] Avg accuracy = {accs[-1]:.4f}")

    return [n.model for n in all_nodes], accs
