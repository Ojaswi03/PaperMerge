import tkinter as tk
from tkinter import ttk, messagebox
from gui.state.experiment_state import to_persisted
from gui.state import QueueStatus

class QueueView(ttk.Frame):
    def __init__(self,parent,state,on_changed,add_from_library=None,runtime_estimator=None,load_queue=None):
        super().__init__(parent,padding=24); self.state=state; self.on_changed=on_changed; self.runtime_estimator=runtime_estimator; self.search=tk.StringVar(); self.status=tk.StringVar()
        ttk.Label(self,text="Queue",style="Title.TLabel").pack(anchor="w"); filters=ttk.Frame(self); filters.pack(fill="x",pady=(12,4)); ttk.Entry(filters,textvariable=self.search,width=30).pack(side="left"); ttk.Combobox(filters,textvariable=self.status,values=("","pending","running","completed","stopped","failed"),state="readonly",width=12).pack(side="left",padx=6); ttk.Button(filters,text="Filter",command=self.refresh).pack(side="left")
        tools=ttk.Frame(self); tools.pack(fill="x",pady=(4,12))
        actions=(("Load Queue",load_queue),("Add from library",add_from_library),("Move to top",lambda:self._move(-9999)),("Move up",lambda:self._move(-1)),("Move down",lambda:self._move(1)),("Retry",self._retry),("Remove selected",self._remove),("Clear pending",self._clear))
        for label,command in actions:
            if command is None: continue
            ttk.Button(tools,text=label,command=command).pack(side="left",padx=(0,6))
        self.tree=ttk.Treeview(self,columns=("status","name","dataset","rounds","done","device","recovery"),show="headings",selectmode="extended")
        for key,label,width in (("status","Status",90),("name","Experiment",280),("dataset","Dataset",90),("rounds","Rounds",65),("done","Done / total",95),("device","Device",60),("recovery","Start / recovery",260)): self.tree.heading(key,text=label); self.tree.column(key,width=width,stretch=key in {"name","recovery"})
        self.tree.pack(fill="both",expand=True); self.summary=ttk.Label(self,text="",style="Muted.TLabel"); self.summary.pack(fill="x",pady=(8,0)); self.empty=ttk.Label(self,text="Queue is empty. Add the current configuration or load entries from the configuration library.",style="Muted.TLabel"); self.refresh()
        self.detail=ttk.Label(self,text="Select a queued experiment to inspect its protocol, attackers, channel and objective.",wraplength=950,justify="left",style="Muted.TLabel");self.detail.pack(fill="x",pady=(8,0));self.tree.bind('<<TreeviewSelect>>',self._describe)
    def _describe(self,_event=None):
        entry=next((e for e in self.state.entries if e.entry_id in self.selected()),None)
        if entry:self.detail.configure(text=entry.config.dirty_summary+'\n'+entry.execution_hint)
    def selected(self): return set(self.tree.selection())
    def _move(self,offset):
        ids=list(self.selected());
        if ids: self.state.move(ids[0],offset); self.on_changed()
    def _retry(self):
        for entry_id in self.selected():
            try:self.state.retry(entry_id)
            except ValueError:pass
        self.on_changed()
    def _remove(self):
        ids=self.selected()
        if ids and messagebox.askyesno("Remove queue entries",f"Remove {len(ids)} selected queue item(s)?",parent=self): self.state.remove(ids); self.on_changed()
    def _clear(self):
        if messagebox.askyesno("Clear pending queue","Remove every pending queue item?",parent=self): self.state.clear_pending(); self.on_changed()
    def refresh(self):
        needle=self.search.get().casefold(); status=self.status.get(); visible=[entry for entry in self.state.entries if (not needle or needle in entry.config.experiment_name.casefold()) and (not status or entry.status.value==status)]
        wanted={e.entry_id for e in visible}
        for item in self.tree.get_children():
            if item not in wanted:self.tree.delete(item)
        for index,entry in enumerate(visible):
            name=entry.config.extra.get('displayName',entry.config.experiment_name).split(' — Fresh')[0]
            values=tuple(map(str,(entry.status.value.title(),name,f'{entry.config.dataset}/{entry.config.split}',entry.config.rounds,f'{entry.completed_rounds}/{entry.config.rounds}',entry.config.extra.get('executionDevice','CPU'),entry.execution_hint)))
            if not self.tree.exists(entry.entry_id):self.tree.insert('',index,iid=entry.entry_id,values=values)
            elif tuple(self.tree.item(entry.entry_id,'values'))!=values:self.tree.item(entry.entry_id,values=values)
            if self.tree.index(entry.entry_id)!=index:self.tree.move(entry.entry_id,'',index)
        pending=[to_persisted(entry.config) for entry in self.state.entries if entry.status.value=="pending"]
        if pending and all(c.get('iidCampaignId') for c in pending):
            entries=[e for e in self.state.entries if e.status is QueueStatus.PENDING]
            seconds=sum(e.estimated_remaining_seconds if e.estimated_remaining_seconds is not None else
                float(e.config.extra.get('runtimeEstimateModel',{}).get('sourceEbmSeconds' if e.config.extra.get('localObjective')=='source_ebm' else 'ceSeconds',0)) for e in entries)
            estimate=('GPU ETA measured after execution starts' if any(c.get('executionDevice')=='GPU' for c in pending) else
                f'queue CPU estimate {seconds/3600:.1f} h (includes required reconstruction)')
            summary=f'{len(self.state.entries)} entries • {len(pending)} pending • {estimate} • starts only via Run Queue'
        elif self.runtime_estimator and pending:
            estimate=self.runtime_estimator.estimate_queue(pending); summary=f"{len(self.state.entries)} entries • {len(pending)} pending • estimated {estimate.seconds/3600:.1f} h ({estimate.low_seconds/3600:.1f}–{estimate.high_seconds/3600:.1f} h)"
        else: summary=f"{len(self.state.entries)} entries • {len(pending)} pending • runtime estimate unavailable" if pending else f"{len(self.state.entries)} entries • no pending work"
        if self.summary.cget('text')!=summary:self.summary.configure(text=summary)
        if visible:self.empty.pack_forget()
        else:self.empty.pack(pady=30)
