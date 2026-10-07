"""Write boundaries for new research outputs."""
from pathlib import Path
from basil_core.protocol_compatibility import LEGACY_RESULT_ROOTS, LEGACY_PLOT_ROOTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_ROOTS = (PROJECT_ROOT / "experiments/results4", PROJECT_ROOT / "plots4",
                   *(PROJECT_ROOT / path for path in (*LEGACY_RESULT_ROOTS, *LEGACY_PLOT_ROOTS)))

def writable_output(path: Path) -> Path:
    path = Path(path).resolve()
    if any(path == root or path.is_relative_to(root) for root in PROTECTED_ROOTS):
        raise ValueError(f"Historical research artifacts are read-only: {path}")
    return path
