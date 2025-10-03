import numpy as np
import matplotlib.pyplot as plt

from data.mnist import load_mnist, make_loaders as make_mnist_loaders
from data.cifar import load_cifar10, make_loaders as make_cifar_loaders
from models import MNISTModel, CIFARModel
from basil import BasilNode, basil_ring_training_with_attack
from basil_plus import basil_plus_training
from trainer import evaluate_all

# ---------- NO ARGPARSE: set your run config here ----------
CONFIG = {
    "dataset": "mnist",          # "mnist" or "cifar10"
    "n_nodes": 10,
    "S": 2,                      # connectivity/ memory length
    "rounds": 30,
    "local_epochs": 1,
    "lr": 0.05,
    "batch_size": 32,
    "iid": True,
    "basil_plus": False,         # if True, run Basil+ (requires groups)
    "num_groups": 2,

    # Noise/robustness toggles
    "noise_model": "none",       # "none", "noisy", "ebm"
    "sigma": 0.1,

    # Attacks
    "attack_type": "none",       # "none","gaussian","sign_flip","hidden"
    "attacker_ids": [],          # e.g., [2,5]
    "hidden_start_round": 20,
}

def _build_nodes(train_loaders, dataset_name, cfg):
    nodes = []
    for i in range(cfg["n_nodes"]):
        if dataset_name == "mnist":
            model = MNISTModel()
        else:
            model = CIFARModel()
        node = BasilNode(
            node_id=i,
            model=model,
            data_loader=train_loaders[i],
            S=cfg["S"],
            noise_model=cfg["noise_model"],
            sigma=cfg["sigma"],
            lr=cfg["lr"],
            local_epochs=cfg["local_epochs"]
        )
        nodes.append(node)
    return nodes

def plot_curve(accs, title, out_png=None):
    xs = list(range(len(accs)))
    plt.figure(figsize=(7,4))
    plt.plot(xs, accs, marker="o")
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.grid(True, linestyle=":")
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()

def run_experiment():
    cfg = CONFIG.copy()

    # load data
    if cfg["dataset"] == "mnist":
        train, test = load_mnist()
        train_loaders, test_loader = make_mnist_loaders(
            train, test, batch_size=cfg["batch_size"], iid=cfg["iid"], n_clients=cfg["n_nodes"]
        )
    else:
        train, test = load_cifar10()
        train_loaders, test_loader = make_cifar_loaders(
            train, test, batch_size=cfg["batch_size"], iid=cfg["iid"], n_clients=cfg["n_nodes"]
        )

    # build Basil nodes
    nodes = _build_nodes(train_loaders, cfg["dataset"], cfg)

    # run Basil or Basil+
    if cfg["basil_plus"]:
        _, accs = basil_plus_training(
            nodes, num_groups=cfg["num_groups"],
            rounds=cfg["rounds"], test_loader=test_loader,
            attack_type=cfg["attack_type"],
            attacker_ids=cfg["attacker_ids"],
            hidden_start_round=cfg["hidden_start_round"]
        )
        title = f"Basil+ ({cfg['dataset']}) noise={cfg['noise_model']} σ={cfg['sigma']}"
    else:
        _, accs = basil_ring_training_with_attack(
            nodes, rounds=cfg["rounds"], test_loader=test_loader,
            attack_type=cfg["attack_type"],
            attacker_ids=cfg["attacker_ids"],
            hidden_start_round=cfg["hidden_start_round"]
        )
        title = f"Basil ({cfg['dataset']}) noise={cfg['noise_model']} σ={cfg['sigma']}"

    # summarize + plot
    avg, worst, per_node = evaluate_all(nodes, test_loader)
    print(f"Final AVG acc: {avg:.4f} | WORST acc: {worst:.4f}")
    plot_curve(accs, title)

if __name__ == "__main__":
    run_experiment()
