import numpy as np
from copy import deepcopy

from attacks import apply_attack
from trainer import local_update, evaluate_batch_loss, evaluate, add_channel_noise_to_params

class BasilNode:
    """
    Node with memory (last S snapshots). Selects best snapshot via local batch loss,
    runs local update, then forwards current weights to next S neighbors.
    """
    def __init__(self, node_id, model, data_loader, S, noise_model="none", sigma=0.0, lr=0.05, local_epochs=1):
        self.node_id = node_id
        self.model = model          # wrapper exposing get_params/set_params
        self.data_loader = data_loader
        self.S = S
        self.memory = []            # list of param snapshots (numpy arrays)
        self.noise_model = noise_model
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.local_epochs = int(local_epochs)

    def store_model(self, params):
        # FIFO memory of length S
        self.memory.append([p.copy() for p in params])
        if len(self.memory) > self.S:
            self.memory.pop(0)

    def select_model(self):
        """Pick snapshot with lowest batch loss; if none, keep current."""
        if not self.memory:
            return deepcopy(self.model.get_params())
        losses = []
        for params in self.memory:
            self.model.set_params(params)
            losses.append(evaluate_batch_loss(self.model, self.data_loader))
        best = int(np.argmin(np.array(losses)))
        return deepcopy(self.memory[best])

    def local_train(self):
        local_update(
            self.model,
            self.data_loader,
            epochs=self.local_epochs,
            lr=self.lr,
            noise_model=self.noise_model,
            sigma=self.sigma
        )

def basil_ring_training_with_attack(nodes, rounds, test_loader=None, attack_type="none",
                                    attacker_ids=None, hidden_start_round=20):
    """
    Basil on ring: each node forwards params to next S neighbors.
    Attack can alter the outgoing weights for attacker nodes.
    Also models channel noise (if enabled) at send time.
    """
    if attacker_ids is None:
        attacker_ids = []

    accs = []
    # baseline at round 0
    if test_loader is not None:
        base = [evaluate(n.model, test_loader) for n in nodes]
        accs.append(float(np.mean(base)))

    n = len(nodes)

    for r in range(rounds):
        # === 1) Selection + local training ===
        for node in nodes:
            chosen = node.select_model()
            node.model.set_params(chosen)
            node.local_train()

        # === 2) Forward to neighbors (with attack + channel noise if any) ===
        for i in range(n):
            params = nodes[i].model.get_params()

            # attackers modify outgoing payload
            if (i in attacker_ids) and (attack_type != "none") and (r >= hidden_start_round or attack_type != "hidden"):
                params = apply_attack(params, attack_type)

            # channel noise on the payload (noisy or ebm modes)
            if nodes[i].noise_model in ("noisy", "ebm") and nodes[i].sigma > 0.0:
                params = add_channel_noise_to_params(params, nodes[i].sigma)

            # multicast to next S neighbors
            for s in range(1, nodes[i].S + 1):
                j = (i + s) % n
                nodes[j].store_model(params)

        # === 3) eval ===
        if test_loader is not None:
            round_acc = [evaluate(n.model, test_loader) for n in nodes]
            accs.append(float(np.mean(round_acc)))
            print(f"[Round {r+1}] Avg accuracy = {accs[-1]:.4f}")

    return [n.model for n in nodes], accs
