from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable

LOG=logging.getLogger(__name__)
class PlotService:
    def __init__(self,protected_root: Path|str=Path("plots4")): self.protected_root=Path(protected_root).resolve()
    def generate(self,generator: Callable[...,object],*,output_dir: Path|str,**kwargs)->object:
        from basil_core.artifact_paths import writable_output
        output=writable_output(Path(output_dir))
        if output==self.protected_root or self.protected_root in output.parents: raise ValueError("Protected historical plots cannot be overwritten.")
        output.mkdir(parents=True,exist_ok=True)
        try: return generator(output_dir=output,**kwargs)
        except Exception: LOG.exception("Plot generation failed for %s",output); raise
    def generate_adaptive(self, output_dir: Path | str) -> dict:
        from reporting.adaptive_study_plots import generate_campaign4_plots
        from basil_core.artifact_paths import writable_output
        output=writable_output(Path(output_dir))
        if output==self.protected_root or self.protected_root in output.parents: raise ValueError("Select a new output directory; protected historical plots cannot be overwritten.")
        output.mkdir(parents=True,exist_ok=True)
        try:return generate_campaign4_plots(result_root=Path("experiments")/"results4"/"campaign4"/"gui",plot_root=output,only_changed=False,formats=("png",))
        except Exception:LOG.exception("Adaptive-study plot generation failed for %s",output); raise
    def generate_research(self,run_dir,output_dir):
        import json
        from basil_core.protocol_compatibility import IID_PROTOCOL
        from basil_core.artifact_paths import writable_output
        metadata=json.loads((Path(run_dir)/'run.json').read_text())
        if metadata.get('experimentProtocol')==IID_PROTOCOL:
            from reporting.iid_study_plots import generate_run_plots
            output=writable_output(Path(output_dir))
            try:
                generate_run_plots(run_dir,output)
                return dict(generated=[str(p) for p in output.glob('*.png')],errors=[])
            except Exception:LOG.exception('IID plot generation failed for %s',run_dir);raise
        from reporting.sequential_basil_plots import generate_run_plots
        try:return generate_run_plots(run_dir,writable_output(Path(output_dir)))
        except Exception:LOG.exception("Research plot generation failed for %s",run_dir);raise
