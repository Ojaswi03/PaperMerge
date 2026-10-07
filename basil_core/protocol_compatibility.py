"""Deprecated configuration aliases and read-only historical locations."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

PROTOCOL = "sequential_basil_one_class_v1"
IID_PROTOCOL = "sequential_basil_iid_v1"

# Deprecated compatibility aliases: read old configurations, write canonical IDs.
LEGACY_PROTOCOL_ALIASES = {
    "professor_sequential_one_class_v1": PROTOCOL,
    "professor_protocol": PROTOCOL,
}
LEGACY_PURPOSE_ALIASES = {"professor_protocol_production": "research_protocol_production"}
LEGACY_RESULT_ROOTS = (
    Path("experiments/professor_protocol_results"),
    Path("experiments/professor_validation_results"),
)
LEGACY_PLOT_ROOTS = (Path("plots/professor_protocol"), Path("plots/professor_validation"))


def is_research_protocol(identifier: object) -> bool:
    return isinstance(identifier, str) and LEGACY_PROTOCOL_ALIASES.get(identifier, identifier) in {PROTOCOL, IID_PROTOCOL}


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate only known naming fields; preserve scientific and unknown metadata."""
    data = dict(config)
    for field in ("experimentProtocol", "protocol"):
        if is_research_protocol(data.get(field)):
            data[field] = LEGACY_PROTOCOL_ALIASES.get(data[field], data[field])
    if "experimentProtocol" not in data and is_research_protocol(data.get("protocol")):
        data["experimentProtocol"] = data["protocol"]
    if is_research_protocol(data.get("experimentProtocol", data.get("protocol"))):
        purpose = data.get("purpose")
        if isinstance(purpose, str):
            data["purpose"] = LEGACY_PURPOSE_ALIASES.get(purpose, purpose)
        name = data.get("experimentName")
        if isinstance(name, str):
            # Deprecated display-name alias, not a rewrite of arbitrary user notes.
            data["experimentName"] = re.sub(r"\bprofessor\b", "Sequential BASIL", name, flags=re.IGNORECASE)
    return data
