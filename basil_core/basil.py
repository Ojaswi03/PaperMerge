import sys
from copy import deepcopy

from .attacks import apply_attack
from .trainer import (
    local_update,
    evaluate_batch_loss,
    add_channel_noise_to_params,
    get_params,
    set_params,
    make_lr_scheduler,
    evaluate_all,
)


class BasilNode:
    """
    Node with memory (last S snapshots). Selects best snapshot via local batch loss,
    runs local update, then forwards current weights to next neighbor in ring.
    noise_model in {"none", "noisy", "ebm"}
    """
    def __init__(
        self,
        node_id,
        model,
        data_loader,
        S,
        noise_model="none",
        sigma=0.0,
        lr0=0.05,
        local_epochs=1,
        **kwargs,  # accept legacy/extra kwargs (e.g., lr)
    ):
        # Backward compatibility: map legacy 'lr' -> 'lr0'
        if "lr" in kwargs and kwargs["lr"] is not None:
            lr0 = kwargs["lr"]

        self.node_id = node_id
        self.model = model
        self.data_loader = data_loader
        self.S = int(S)
        self.noise_model = str(noise_model)
        self.sigma = float(sigma)
        self.lr0 = float(lr0)
        self.local_epochs = int(local_epochs)

        # Memory of parameter snapshots (list[list[np.ndarray]])
        self.memory = []
        self.round = 0

    def snapshot(self):
        """Store a copy of current params; keep only last S."""
        self.memory.append(get_params(self.model))
        if len(self.memory) > self.S:
            self.memory.pop(0)

    def select_best_snapshot(self):
        """
        Choose the params (from {current} ∪ memory) that minimize local batch loss.
        Restores the model to the best params.
        """
        candidates = [get_params(self.model)] + self.memory
        best_params = candidates[0]
        best_loss = evaluate_batch_loss(self.model, self.data_loader)  # current

        for p in self.memory:
            orig = get_params(self.model)
            set_params(self.model, p)
            loss_p = evaluate_batch_loss(self.model, self.data_loader)
            if loss_p < best_loss:
                best_loss = loss_p
                best_params = [x.copy() for x in p]
            # restore
            set_params(self.model, orig)

        set_params(self.model, best_params)

    def local_train(self, lr, steps_per_epoch=100):
        """
        Local training bounded by steps_per_epoch to avoid infinite loops when the
        tf.data pipeline uses `.repeat()` without a take/epoch cap.
        """
        local_update(
            self.model,
            self.data_loader,
            epochs=self.local_epochs,
            lr=lr,
            noise_model=self.noise_model,
            sigma=self.sigma,
            steps_per_epoch=steps_per_epoch,
        )


def basil_ring_training_with_attack(
    nodes,
    rounds,
    test_loader=None,
    attack_types=("none",),
    attacker_ids=None,
    hidden_start_round=20,
    sigma=0.0,
    noise_model="none",
    lr0=0.05,
    lr_alpha=0.6,
    steps_per_epoch=100,  # safe bound for local updates
    **kwargs,  # accept legacy kw like attack_type
):
    """
    Basil on a ring: each node forwards params to the next neighbor each round.
    - attack_types: list or str (e.g., ["gaussian", "sign-flip", "hidden"])
      rotated per round; legacy single 'attack_type' kw is mapped to [attack_type].
    - steps_per_epoch: limits local_update iterations to avoid hangs with infinite datasets.
    Returns: (avg_acc_history, worst_acc_history)
    """
    # Backward compatibility: allow single 'attack_type' kw
    if "attack_type" in kwargs and kwargs["attack_type"] is not None:
        attack_types = [kwargs["attack_type"]]

    # Normalize attack_types
    if isinstance(attack_types, str):
        attack_types = [attack_types]
    if not attack_types:
        attack_types = ["none"]

    n = len(nodes)
    # Push global noise / lr settings into nodes
    for nd in nodes:
        nd.noise_model = noise_model
        nd.sigma = float(sigma)
        nd.lr0 = float(lr0)

    # Dynamic LR: lr_t = max(min_lr, lr0 * (t+1)^(-alpha))
    lr_sched = make_lr_scheduler(lr0, alpha=lr_alpha)

    avg_acc_hist, worst_acc_hist = [], []

    # Pre-train evaluation (round -1)
    if test_loader is not None:
        avg, worst, _ = evaluate_all(nodes, test_loader)
        print(f"[round -1] pre-train avg={avg:.4f} worst={worst:.4f}", flush=True)
        avg_acc_hist.append(avg)
        worst_acc_hist.append(worst)

    for r in range(rounds):
        lr = lr_sched(r)
        print(f"[round {r}] lr={lr:.6f} training...", flush=True)

        # Local step per node
        for nd in nodes:
            nd.select_best_snapshot()
            nd.local_train(lr=lr, steps_per_epoch=steps_per_epoch)
            nd.snapshot()

        # Communication: add channel noise and apply attacks, then deliver to next neighbor
        out_params = []
        for nd in nodes:
            params = get_params(nd.model)
            # Add sender-side channel noise when using noisy/ebm models
            noisy_params = add_channel_noise_to_params(
                params, sigma=sigma if noise_model in ("noisy", "ebm") else 0.0
            )
            out_params.append(noisy_params)

        atk = attack_types[r % len(attack_types)]
        attackers = set(attacker_ids or [])

        new_params_after_comm = [None] * n
        for i in range(n):
            send = deepcopy(out_params[i])
            if i in attackers:
                if atk == "hidden" and r < hidden_start_round:
                    # delay hidden/backdoor until threshold
                    pass
                else:
                    send = apply_attack(send, atk)
            j = (i + 1) % n  # ring neighbor
            new_params_after_comm[j] = send

        # Apply received params
        for j in range(n):
            if new_params_after_comm[j] is not None:
                set_params(nodes[j].model, new_params_after_comm[j])

        # Evaluation/logging
        if test_loader is not None:
            avg, worst, _ = evaluate_all(nodes, test_loader)
            print(f"[round {r}] eval avg={avg:.4f} worst={worst:.4f}", flush=True)
            avg_acc_hist.append(avg)
            worst_acc_hist.append(worst)

    return avg_acc_hist, worst_acc_hist
