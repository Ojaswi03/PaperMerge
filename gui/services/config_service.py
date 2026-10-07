from __future__ import annotations
import json
from pathlib import Path
from gui.state.experiment_state import ExperimentConfig, from_persisted, to_persisted

class ConfigService:
    def load_dict(self, data: dict) -> ExperimentConfig:
        return from_persisted(data)
    def load(self, path: Path | str) -> ExperimentConfig:
        try: data=json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as error: raise ValueError(f"Could not load configuration: {error}") from error
        return from_persisted(data)
    def save(self, path: Path | str, config: ExperimentConfig) -> None:
        from gui.state.experiment_state import validate_experiment
        issues=validate_experiment(config)
        if issues:raise ValueError("; ".join(issue.message for issue in issues))
        from basil_core.artifact_paths import writable_output
        target=writable_output(Path(path)); target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+".tmp")
        temporary.write_text(json.dumps(to_persisted(config),indent=2,sort_keys=True)+"\n",encoding="utf-8")
        temporary.replace(target)
