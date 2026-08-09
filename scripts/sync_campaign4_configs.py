#!/usr/bin/env python3
"""Generate the reproducible Campaign 4 JSON configuration library."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gui.campaign4 import (  # noqa: E402
    CAMPAIGN_ID,
    CONFIG_ROOT,
    PROTOCOL_REVISION,
    build_diagnostic,
    build_main_confirmation,
    build_performance_benchmark,
    build_static_controls,
    config_hash,
    write_json_atomic,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report whether the generated library is current without writing it.",
    )
    return parser.parse_args()


def _all_configs():
    configs = (
        build_main_confirmation()
        + build_static_controls()
        + build_diagnostic("merged")
        + build_diagnostic("cart")
        + build_performance_benchmark()
    )
    unique = {}
    for config in configs:
        unique[config["runId"]] = config
    return sorted(
        unique.values(),
        key=lambda config: (
            config["split"],
            config["approach"],
            config["phase"],
            config["conditionId"],
            int(config["seed"]),
        ),
    )


def _relative_path(config):
    filename = (
        f"{config['phase']}__{config['conditionId']}__seed_{config['seed']}__"
        f"{config['runId']}.json"
    )
    return Path(config["split"]) / config["approach"] / filename


def _expected_payloads():
    payloads = {_relative_path(config): config for config in _all_configs()}
    manifest = {
        "schemaVersion": 4,
        "campaignId": CAMPAIGN_ID,
        "protocolRevision": PROTOCOL_REVISION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "configCount": len(payloads),
        "configs": [
            {
                "path": str(path),
                "runId": config["runId"],
                "configHash": config_hash(config),
                "phase": config["phase"],
                "split": config["split"],
                "approach": config["approach"],
                "conditionId": config["conditionId"],
                "seed": config["seed"],
            }
            for path, config in payloads.items()
        ],
    }
    return payloads, manifest


def _is_current(root, payloads):
    import json

    existing = {
        path.relative_to(root)
        for path in root.glob("**/*.json")
        if path.name != "manifest.json"
    }
    if existing != set(payloads):
        return False
    for relative, expected in payloads.items():
        try:
            actual = json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if config_hash(actual) != config_hash(expected) or actual != expected:
            return False
    return True


def main():
    args = _parse_args()
    root = PROJECT_ROOT / CONFIG_ROOT
    payloads, manifest = _expected_payloads()
    current = _is_current(root, payloads)
    if args.check:
        print(
            f"Campaign 4 config library: {'current' if current else 'out of date'} "
            f"({len(payloads)} configs)"
        )
        return 0 if current else 1

    root.mkdir(parents=True, exist_ok=True)
    expected_paths = {root / relative for relative in payloads}
    for existing in root.glob("**/*.json"):
        if existing.name != "manifest.json" and existing not in expected_paths:
            existing.unlink()
    for relative, config in payloads.items():
        write_json_atomic(root / relative, config)
    write_json_atomic(root / "manifest.json", manifest)
    print(f"Wrote {len(payloads)} Campaign 4 configs under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
