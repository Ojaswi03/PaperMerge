"""
Run Basil with EBM (noisy channel + σ^2||∇loss||^2 regularizer in local training).
For each attack in `attacks`, run a separate 30-round experiment and save ONLY
the average accuracy history to experiments/results/<exp_name>__<attack>_avg.npy
"""

import os
import sys

# Ensure repo-root imports work when invoked as a module
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basil_core.data.mnist import load_mnist, make_loaders as make_mnist_loaders
from basil_core.data.cifar import load_cifar10, make_loaders as make_cifar_loaders
from basil_core.models import MNISTModel, CIFARModel
from basil_core.basil import BasilNode, basil_ring_training_with_attack
from scripts.common import ensure_dirs, save_curve


def _sanitize_attack_name(name: str) -> str:
    return (name or "none").lower().replace(" ", "_").replace("-", "_")


def run(config):
    """
    config keys used:
      exp_name, dataset, batch_size, iid, n_nodes, S, sigma, lr0, lr_alpha,
      local_epochs, rounds_per_attack (int, default 30),
      attacks (list[str]),
      attacker_ids (list[int]),
      hidden_start_round (int)
    """
    dataset = config.get("dataset", "mnist").lower()
    rounds_per_attack = int(config.get("rounds_per_attack", 30))

    # Load data + pick model
    if dataset == "mnist":
        train, test = load_mnist()
        train_loaders, test_loader = make_mnist_loaders(
            train,
            test,
            batch_size=config["batch_size"],
            iid=config["iid"],
            n_clients=config["n_nodes"],
        )
        Model = MNISTModel
    else:
        train, test = load_cifar10()
        train_loaders, test_loader = make_cifar_loaders(
            train,
            test,
            batch_size=config["batch_size"],
            iid=config["iid"],
            n_clients=config["n_nodes"],
        )
        Model = CIFARModel

    # Build nodes (EBM noise model)
    nodes_init = []
    for i in range(config["n_nodes"]):
        model = Model()
        node = BasilNode(
            node_id=i,
            model=model,
            data_loader=train_loaders[i],
            S=config["S"],
            noise_model="ebm",
            sigma=config["sigma"],
            lr0=config["lr0"],
            local_epochs=config["local_epochs"],
        )
        nodes_init.append(node)

    # Ensure output dir
    ensure_dirs()

    results = {}
    attacks = config.get("attacks", ["none"])
    if isinstance(attacks, str):
        attacks = [attacks]

    for attack in attacks:
        atk_name = _sanitize_attack_name(attack)
        # Deep-reset models for each attack run by re-instantiating nodes
        nodes = []
        for i in range(config["n_nodes"]):
            model = Model()
            nodes.append(BasilNode(
                node_id=i,
                model=model,
                data_loader=train_loaders[i],
                S=config["S"],
                noise_model="ebm",
                sigma=config["sigma"],
                lr0=config["lr0"],
                local_epochs=config["local_epochs"],
            ))

        print(f"\n=== Running EBM with attack: {attack} for {rounds_per_attack} rounds ===", flush=True)
        avg_hist, _ = basil_ring_training_with_attack(
            nodes=nodes,
            rounds=rounds_per_attack,
            test_loader=test_loader,
            attack_types=[attack],  # single attack for the whole run
            attacker_ids=config.get("attacker_ids", []),
            hidden_start_round=config.get("hidden_start_round", 10),
            sigma=config["sigma"],
            noise_model="ebm",
            lr0=config["lr0"],
            lr_alpha=config.get("lr_alpha", 0.6),
            steps_per_epoch=config.get("steps_per_epoch", 100),
        )

        # Save ONLY average accuracy
        out_path = f"experiments/results/{config['exp_name']}__{atk_name}_avg.npy"
        save_curve(avg_hist, out_path)
        print(f"[saved] {out_path}", flush=True)

        results[attack] = avg_hist

    # Summary
    print("\n=== Summary (avg acc last point) ===", flush=True)
    for attack, hist in results.items():
        last = hist[-1] if hist else float("nan")
        print(f"{attack:>12}: {last:.4f}", flush=True)

    return results


if __name__ == "__main__":
    cfg = dict(
        exp_name="mnist_ebm_looped",
        dataset="mnist",      # "mnist" or "cifar10"
        batch_size=128,
        iid=True,
        n_nodes=10,
        S=2,
        sigma=0.05,
        lr0=0.05,
        lr_alpha=0.6,
        local_epochs=1,
        rounds_per_attack=30,            # <-- 30 rounds per attack as requested
        attacks=["gaussian", "sign-flip", "hidden"],  # loop over these
        attacker_ids=[0, 3],
        hidden_start_round=10,
        steps_per_epoch=100,             # safety bound for infinite datasets
    )
    run(cfg)
