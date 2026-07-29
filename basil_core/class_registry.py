# basil_core/class_registry.py
"""ClassRegistry: decentralized per-class knowledge tracker for CART.

The registry travels around the ring with the model. Each node:
  1. Evaluates its per-class accuracy after local training.
  2. Merges with the received registry (take max per class).
  3. Verifies Byzantine registry claims locally before accepting.
  4. Uses the registry gap to control its proximal regularization strength.
"""

import numpy as np

__all__ = ["ClassRegistry"]


class ClassRegistry:
    """Tracks the best known per-class accuracy seen across the ring so far.

    Attributes
    ----------
    class_best_acc : np.float32[nClasses]
        Best verified accuracy for each class across all nodes seen this round.
    class_best_source : int[nClasses]
        Node ID that achieved the best accuracy for each class (-1 = unknown).
    nClasses : int
    """

    def __init__(self, nClasses: int = 10):
        self.nClasses = nClasses
        self.class_best_acc = np.zeros(nClasses, dtype=np.float32)
        self.class_best_source = np.full(nClasses, -1, dtype=np.int32)
        self.class_best_round = np.full(nClasses, -1, dtype=np.int32)
        self.class_best_support = np.zeros(nClasses, dtype=np.int32)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def update(
        self,
        nodeId: int,
        myClassAcc: np.ndarray,
        support: np.ndarray | None = None,
        roundId: int = -1,
    ):
        """Update registry with this node's per-class accuracy.

        Only improves entries — never decreases best known accuracy.
        """
        myClassAcc = np.asarray(myClassAcc, dtype=np.float32)
        if support is None:
            support = np.ones(self.nClasses, dtype=np.int32)
        support = np.asarray(support, dtype=np.int32)
        improved = (
            (support > 0)
            & np.isfinite(myClassAcc)
            & (myClassAcc > self.class_best_acc)
        )
        self.class_best_acc[improved] = myClassAcc[improved]
        self.class_best_source[improved] = nodeId
        self.class_best_round[improved] = int(roundId)
        self.class_best_support[improved] = support[improved]

    def merge(self, other: "ClassRegistry"):
        """Merge another registry in-place: take max per class (no verification).

        Call verifyAndMerge instead when the other registry arrives from the
        ring and may originate from a Byzantine node.
        """
        improved = other.class_best_acc > self.class_best_acc
        self.class_best_acc[improved] = other.class_best_acc[improved]
        self.class_best_source[improved] = other.class_best_source[improved]
        self.class_best_round[improved] = other.class_best_round[improved]
        self.class_best_support[improved] = other.class_best_support[improved]

    def verifyAndMerge(
        self,
        other: "ClassRegistry",
        receivedModelClassAcc: np.ndarray,
        verifyThreshold: float = 0.05,
        receivedSupport: np.ndarray | None = None,
    ):
        """Merge, but skip classes where the registry claim is implausible.

        A Byzantine node could inflate registry entries to trick honest nodes
        into incorrect distillation targets.  We reject a class-c claim when
        the locally evaluated accuracy for that class on the received model is
        more than verifyThreshold below the claimed best accuracy — i.e. the
        model that supposedly achieved it clearly cannot reproduce the result.

        Parameters
        ----------
        other : ClassRegistry
            Registry received from the ring (possibly Byzantine).
        receivedModelClassAcc : np.float32[nClasses]
            Per-class accuracy of the received model evaluated locally.
        verifyThreshold : float
            Maximum allowed discrepancy between claim and local verification.
        """
        receivedModelClassAcc = np.asarray(receivedModelClassAcc, dtype=np.float32)
        if receivedSupport is None:
            receivedSupport = np.ones(self.nClasses, dtype=np.int32)
        receivedSupport = np.asarray(receivedSupport, dtype=np.int32)
        accepted = 0
        rejected = 0
        for c in range(self.nClasses):
            claim = other.class_best_acc[c]
            if claim <= self.class_best_acc[c]:
                continue  # not an improvement, skip
            if other.class_best_support[c] <= 0 or receivedSupport[c] <= 0:
                continue  # unknown locally; carry no numerical training signal
            # Verify: the model that made this claim should be somewhat consistent
            if receivedModelClassAcc[c] >= claim - verifyThreshold:
                self.class_best_acc[c] = claim
                self.class_best_source[c] = other.class_best_source[c]
                self.class_best_round[c] = other.class_best_round[c]
                self.class_best_support[c] = other.class_best_support[c]
                accepted += 1
            # else: claim rejected — Byzantine inflation likely
            else:
                rejected += 1
        return accepted, rejected

    def trustWeights(
        self,
        myClassAcc: np.ndarray,
        mySupport: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute distillation trust weights for each class.

        trust_weight(c) = max(0, registry_best_acc[c] - my_acc[c])

        High value → I'm behind on class c → preserve the selected reference
        model more strongly through the proximal term.
        Zero value → I'm at or above the registry → train freely.
        """
        myClassAcc = np.asarray(myClassAcc, dtype=np.float32)
        if mySupport is None:
            mySupport = np.ones(self.nClasses, dtype=np.int32)
        mySupport = np.asarray(mySupport, dtype=np.int32)
        reliable = (
            (self.class_best_support > 0)
            & (mySupport > 0)
            & np.isfinite(myClassAcc)
        )
        weights = np.zeros(self.nClasses, dtype=np.float32)
        weights[reliable] = np.maximum(
            0.0,
            self.class_best_acc[reliable] - myClassAcc[reliable],
        )
        return weights

    def reliableMask(self, mySupport: np.ndarray) -> np.ndarray:
        mySupport = np.asarray(mySupport, dtype=np.int32)
        return (self.class_best_support > 0) & (mySupport > 0)

    # ------------------------------------------------------------------
    # Serialisation helpers (for passing around the ring)
    # ------------------------------------------------------------------

    def toDict(self) -> dict:
        return {
            "class_best_acc": self.class_best_acc.tolist(),
            "class_best_source": self.class_best_source.tolist(),
            "class_best_round": self.class_best_round.tolist(),
            "class_best_support": self.class_best_support.tolist(),
            "nClasses": self.nClasses,
        }

    @classmethod
    def fromDict(cls, d: dict) -> "ClassRegistry":
        reg = cls(nClasses=d["nClasses"])
        reg.class_best_acc = np.array(d["class_best_acc"], dtype=np.float32)
        reg.class_best_source = np.array(d["class_best_source"], dtype=np.int32)
        reg.class_best_round = np.array(
            d.get("class_best_round", [-1] * reg.nClasses), dtype=np.int32
        )
        reg.class_best_support = np.array(
            d.get(
                "class_best_support",
                [1 if source >= 0 else 0 for source in reg.class_best_source],
            ),
            dtype=np.int32,
        )
        return reg

    def clone(self) -> "ClassRegistry":
        reg = ClassRegistry(nClasses=self.nClasses)
        reg.class_best_acc = self.class_best_acc.copy()
        reg.class_best_source = self.class_best_source.copy()
        reg.class_best_round = self.class_best_round.copy()
        reg.class_best_support = self.class_best_support.copy()
        return reg

    def __repr__(self):
        lines = [f"ClassRegistry(nClasses={self.nClasses})"]
        for c in range(self.nClasses):
            src = self.class_best_source[c]
            acc = self.class_best_acc[c]
            src_str = f"node {src}" if src >= 0 else "unknown"
            lines.append(f"  class {c}: {acc:.3f} ({src_str})")
        return "\n".join(lines)
