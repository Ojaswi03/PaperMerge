"""
Run Basil with EBM (noisy channel + σ²||∇loss||² regularizer in local training).
"""
from basil_core.data.mnist import load_mnist, make_loaders as make_mnist_loaders
from basil_core.data.cifar import load_cifar10, make_loaders as make_cifar_loaders
from basil_core.models import MNISTModel, CIFARModel
from basil_core.basil import BasilNode, basil_ring_training_with_attack
from basil_core.trainer import evaluate_all
from scripts.common import ensure_dirs, save_curve

def run(config):
    # data
    if config["dataset"] == "mnist":
        train, test = load_mnist()
        train_loaders, test_loader = make_mnist_loaders(
            train, test, batch_size=config["batch_size"], iid=config["iid"], n_clients=config["n_nodes"]
        )
    else:
        train, test = load_cifar10()
        train_loaders, test_loader = make_cifar_loaders(
            train, test, batch_size=config["batch_size"], iid=config["iid"], n_clients=config["n_nodes"]
        )
    # nodes
    nodes = []
    for i in range(config["n_nodes"]):
        model = MNISTModel() if config["dataset"] == "mnist" else CIFARModel()
        nodes.append(BasilNode(
            node_id=i, model=model, data_loader=train_loaders[i],
            S=config["S"], noise_model="ebm", sigma=config["sigma"],
            lr=config["lr"], local_epochs=config["local_epochs"]
        ))
    # train
    _, accs = basil_ring_training_with_attack(
        nodes, rounds=config["rounds"], test_loader=test_loader,
        attack_type="none", attacker_ids=[], hidden_start_round=999999
    )
    avg, worst, _ = evaluate_all(nodes, test_loader)
    return accs, avg, worst

if __name__ == "__main__":
    ensure_dirs()
    CONFIG = {
        "dataset": "mnist", "n_nodes": 10, "S": 2, "rounds": 30,
        "local_epochs": 1, "lr": 0.05, "batch_size": 32, "iid": True,
        "sigma": 0.10
    }
    accs, avg, worst = run(CONFIG)
    out = save_curve(accs, "experiments/results/acc_mnist_ebm.npy")
    print(f"[EBM] Final AVG={avg:.4f} WORST={worst:.4f} | saved: {out}")
