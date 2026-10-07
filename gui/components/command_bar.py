from tkinter import ttk

class CommandBar(ttk.Frame):
    def __init__(self,parent,commands):
        super().__init__(parent,style="Surface.TFrame",padding=(16,10)); self.columnconfigure(0,weight=1)
        self.title=ttk.Label(self,text="Dashboard",font=("TkDefaultFont",15,"bold"),style="Surface.TLabel"); self.title.grid(row=0,column=0,sticky="w")
        self.identity=ttk.Label(self,text="Untitled experiment",style="SurfaceMuted.TLabel",wraplength=220); self.identity.grid(row=1,column=0,sticky="w",pady=(4,0))
        actions=ttk.Frame(self,style="Surface.TFrame");actions.grid(row=0,column=1,rowspan=2,sticky="e")
        self.save=ttk.Button(actions,text="Save",width=6,command=commands["save"]); self.load=ttk.Button(actions,text="Load",width=6,command=commands["load"])
        self.queue=ttk.Button(actions,text="Add to Queue",width=12,command=commands["queue"]); self.run=ttk.Button(actions,text="Run",width=6,style="Accent.TButton",command=commands["run"])
        self.stop=ttk.Button(actions,text="Stop",width=6,style="Danger.TButton",command=commands["stop"])
        for button in (self.save,self.load,self.queue,self.run,self.stop):button.pack(side="left",padx=4)
        self.status=ttk.Label(self,text="Ready",style="SurfaceMuted.TLabel"); self.status.grid(row=2,column=0,columnspan=2,sticky="ew",pady=(5,0))
    def refresh(self,state,workspace_name):
        identity=(state.active_experiment_name or state.experiment.experiment_name) if state.can_stop or state.execution.value=='stopping' else state.experiment.experiment_name
        device=(f'{state.execution_device}: {state.device_name}' if state.execution_device else
            f"{state.experiment.extra.get('executionDevice','CPU')} requested; not started")
        self.title.configure(text=workspace_name); self.identity.configure(text=identity+("  •  Unsaved" if state.dirty else "")); self.status.configure(text=f"{state.execution.value.title()}  •  {state.status}  •  Workers: {state.worker_count}  •  Device: {device}")
        self.run.configure(state="normal" if state.can_run else "disabled"); self.queue.configure(state="normal" if not __import__('gui.state',fromlist=['validate_experiment']).validate_experiment(state.experiment) else "disabled"); self.stop.configure(state="normal" if state.can_stop else "disabled")
        self.run.configure(text="Run Queue" if workspace_name=="Queue" else "Run",width=12 if workspace_name=="Queue" else 6)
