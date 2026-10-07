#!/usr/bin/env python3
"""Display-required GUI checks, separate from ordinary unit tests."""
from pathlib import Path
import sys
import tkinter as tk

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gui.app import PaperMergeApp
from gui.state import Workspace
from gui.state.experiment_state import from_persisted
from gui.research_protocol import smoke_configs, RESULT_ROOT
from basil_core.protocol_compatibility import LEGACY_RESULT_ROOTS

def main():
    root=tk.Tk();app=PaperMergeApp(root);root.geometry("1100x720")
    builder=app.shell.views[Workspace.BUILDER]
    config=next(c for c in smoke_configs() if c["conditionId"]=="smoke_noise_abs_0.01_ebm")
    app.state.experiment=from_persisted(config);builder._load_model()
    assert builder.vars["channel_noise_semantics"].get()=="Paper absolute Gaussian"
    assert builder.vars["ebm_mode"].get()=="Full gradient-norm objective"
    assert builder.widgets["channel_noise_sigma_absolute"].grid_info()
    assert not builder.widgets["channel_noise_sigma_rel"].grid_info()
    for workspace in Workspace:app.shell.navigate(workspace);root.update()
    metadata=next((Path(RESULT_ROOT)/"smoke").glob(f"{config['runId']}/run.json"),None)
    if metadata is None:
        import json
        metadata=next((p for location in LEGACY_RESULT_ROOTS for p in location.rglob("run.json")
                       if json.loads(p.read_text()).get("config",{}).get("conditionId")==config["conditionId"]
                       and json.loads(p.read_text()).get("status")=="completed"),None)
    if metadata:app.shell.network_view.load_replay(metadata);root.update()
    app.shell.navigate(Workspace.BUILDER);root.update()
    stop=app.shell.command_bar.stop
    assert stop.winfo_rootx()+stop.winfo_width()<=root.winfo_rootx()+root.winfo_width()
    title=app.shell.command_bar.title
    assert title.winfo_width()>=title.winfo_reqwidth()
    assert app.state.can_run
    try:
        from PIL import ImageGrab
        ImageGrab.grab().save("/tmp/papermerge_research_gui.png")
    except OSError:pass
    root.destroy()
    print("GUI smoke passed: five workspaces, explicit sigma labels, progressive disclosure, replay.")

if __name__=="__main__":main()
