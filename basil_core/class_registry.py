# basil_core/class_registry.py
"""ClassRegistry: decentralized per-class knowledge tracker for CART.

The registry travels around the ring with the model. Each node:
  1. Evaluates its per-class accuracy after local training.
  2. Merges with the received registry (take max per class).
  3. Verifies Byzantine registry claims locally before accepting.
  4. Uses the registry to weight its distillation loss.
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

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def update(self, nodeId: int, myClassAcc: np.ndarray):
        """Update registry with this node's per-class accuracy.

        Only improves entries — never decreases best known accuracy.
        """
        myClassAcc = np.asarray(myClassAcc, dtype=np.float32)
        improved = myClassAcc > self.class_best_acc
        self.class_best_acc[improved] = myClassAcc[improved]
        self.class_best_source[improved] = nodeId

    def merge(self, other: "ClassRegistry"):
        """Merge another registry in-place: take max per class (no verification).

        Call verifyAndMerge instead when the other registry arrives from the
        ring and may originate from a Byzantine node.
        """
        improved = other.class_best_acc > self.class_best_acc
        self.class_best_acc[improved] = other.class_best_acc[improved]
        self.class_best_source[improved] = other.class_best_source[improved]

    def verifyAndMerge(
        self,
        other: "ClassRegistry",
        receivedModelClassAcc: np.ndarray,
        verifyThreshold: float = 0.05,
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
        for c in range(self.nClasses):
            claim = other.class_best_acc[c]
            if claim <= self.class_best_acc[c]:
                continue  # not an improvement, skip
            # Verify: the model that made this claim should be somewhat consistent
            if receivedModelClassAcc[c] >= claim - verifyThreshold:
                self.class_best_acc[c] = claim
                self.class_best_source[c] = other.class_best_source[c]
            # else: claim rejected — Byzantine inflation likely

    def trustWeights(self, myClassAcc: np.ndarray) -> np.ndarray:
        """Compute distillation trust weights for each class.

        trust_weight(c) = max(0, registry_best_acc[c] - my_acc[c])

        High value → I'm behind on class c → penalise deviation from registry.
        Zero value → I'm at or above the registry → train freely.
        """
        myClassAcc = np.asarray(myClassAcc, dtype=np.float32)
        return np.maximum(0.0, self.class_best_acc - myClassAcc).astype(np.float32)

    # ------------------------------------------------------------------
    # Serialisation helpers (for passing around the ring)
    # ------------------------------------------------------------------

    def toDict(self) -> dict:
        return {
            "class_best_acc": self.class_best_acc.tolist(),
            "class_best_source": self.class_best_source.tolist(),
            "nClasses": self.nClasses,
        }

    @classmethod
    def fromDict(cls, d: dict) -> "ClassRegistry":
        reg = cls(nClasses=d["nClasses"])
        reg.class_best_acc = np.array(d["class_best_acc"], dtype=np.float32)
        reg.class_best_source = np.array(d["class_best_source"], dtype=np.int32)
        return reg

    def clone(self) -> "ClassRegistry":
        reg = ClassRegistry(nClasses=self.nClasses)
        reg.class_best_acc = self.class_best_acc.copy()
        reg.class_best_source = self.class_best_source.copy()
        return reg

    def __repr__(self):
        lines = [f"ClassRegistry(nClasses={self.nClasses})"]
        for c in range(self.nClasses):
            src = self.class_best_source[c]
            acc = self.class_best_acc[c]
            src_str = f"node {src}" if src >= 0 else "unknown"
            lines.append(f"  class {c}: {acc:.3f} ({src_str})")
        return "\n".join(lines)
