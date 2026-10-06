#!/usr/bin/env python3
"""Write the versioned current GUI config library from its source contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.config_library import (  # noqa: E402
    APPROACHES,
    CURRENT_CONFIG_ROOT,
    CURRENT_LIBRARY_VERSION,
    SPLITS,
    build_current_configs,
    config_filename,
)
from gui.baseline_study import write_json_atomic  # noqa: E402


def sync_library(root: Path = CURRENT_CONFIG_ROOT) -> dict:
    counts = {}
    for split in SPLITS:
        for approach in APPROACHES:
            folder = root / split / approach
            folder.mkdir(parents=True, exist_ok=True)
            expected = build_current_configs(split, approach)
            expected_names = {config_filename(config) for config in expected}
            for path in folder.glob("*.json"):
                if path.name not in expected_names:
                    path.unlink()
            for config in expected:
                write_json_atomic(folder / config_filename(config), config)
            counts[f"{split}/{approach}"] = len(expected)

    manifest = {
        "libraryVersion": CURRENT_LIBRARY_VERSION,
        "counts": counts,
    }
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    result = sync_library()
    print(json.dumps(result, indent=2, sort_keys=True))
