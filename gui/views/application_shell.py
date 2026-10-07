from __future__ import annotations
from tkinter import ttk
from gui.components import CommandBar, NavigationRail
from gui.state import Workspace
from .dashboard_view import DashboardView
from .experiment_builder_view import ExperimentBuilderView
from .queue_view import QueueView
from .results_view import ResultsView
from gui.network_view import NetworkView

class ApplicationShell(ttk.Frame):
    NAMES={Workspace.DASHBOARD:"Dashboard",Workspace.BUILDER:"Experiment Builder",Workspace.QUEUE:"Queue",Workspace.RESULTS:"Results",Workspace.NETWORK:"Network"}
    def __init__(self,parent,state,queue_state,result_service,commands,runtime_estimator=None):
        super().__init__(parent); self.state=state; self.grid(row=0,column=0,sticky="nsew"); self.rowconfigure(1,weight=1); self.columnconfigure(1,weight=1)
        self.command_bar=CommandBar(self,commands); self.command_bar.grid(row=0,column=1,sticky="ew")
        NavigationRail(self,self.navigate).grid(row=0,column=0,rowspan=2,sticky="nsew")
        self.stage=ttk.Frame(self); self.stage.grid(row=1,column=1,sticky="nsew"); self.stage.rowconfigure(0,weight=1); self.stage.columnconfigure(0,weight=1)
        network_frame=ttk.Frame(self.stage)
        self.network_view=NetworkView(network_frame)
        self._network_event_index=0
        self.views={
            Workspace.DASHBOARD:DashboardView(self.stage,state,queue_state,result_service),
            Workspace.BUILDER:ExperimentBuilderView(self.stage,state,self.refresh),
            Workspace.QUEUE:QueueView(self.stage,queue_state,commands["queue_changed"],commands.get("library"),runtime_estimator,commands.get('load_queue')),
            Workspace.RESULTS:ResultsView(self.stage,result_service,self.replay,commands.get("plot")),
            Workspace.NETWORK:network_frame,
        }
        for view in self.views.values(): view.grid(row=0,column=0,sticky="nsew")
        self.navigate(state.workspace)
    def navigate(self,workspace): self.state.navigate(workspace); self.views[workspace].tkraise(); self.refresh()
    def replay(self,record):
        self.navigate(Workspace.NETWORK)
        try:self.network_view.load_replay(record.path)
        except Exception as error: self.state.status=f"Telemetry replay failed: {error}"; self.refresh()
    def refresh(self):
        self.command_bar.refresh(self.state,self.NAMES[self.state.workspace]); dashboard=self.views[Workspace.DASHBOARD]; dashboard.refresh()
        self.views[Workspace.QUEUE].refresh()
        while self._network_event_index < len(self.state.network_events):
            kind,payload=self.state.network_events[self._network_event_index]; self._network_event_index+=1
            if kind=="network_started": self.network_view.start_run(payload["lane"],payload["config"])
            elif kind=="network_updated": self.network_view.apply_node_update(payload["lane"],payload["config"],payload["payload"])
            elif kind=="network_round_updated": self.network_view.apply_round(payload["lane"],payload["config"],payload["payload"])
            elif kind=="network_finished": self.network_view.finish_run(payload["lane"],payload["status"])
