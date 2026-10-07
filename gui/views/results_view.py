from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from gui.theme import COLORS

class ResultsView(ttk.Frame):
    def __init__(self,parent,service,on_replay,on_plot=None):
        super().__init__(parent,padding=24); self.service=service; self.on_replay=on_replay; self.on_plot=on_plot; self.records={}
        ttk.Label(self,text="Results",style="Title.TLabel").pack(anchor="w"); filters=ttk.Frame(self); filters.pack(fill="x",pady=12)
        self.search=tk.StringVar(); self.dataset=tk.StringVar(); self.approach=tk.StringVar(); self.status=tk.StringVar(); self.split=tk.StringVar(); self.mitigation=tk.StringVar(); self.noise=tk.StringVar()
        ttk.Entry(filters,textvariable=self.search,width=30).pack(side="left",padx=(0,6))
        for variable,values in ((self.dataset,("","cifar10","mnist","nmnist")),(self.approach,("","basil","noisy","merged","cart","strict_sequential_handoff")),(self.status,("","completed","failed","stopped"))): ttk.Combobox(filters,textvariable=variable,values=values,state="readonly",width=12).pack(side="left",padx=3)
        for variable,values in ((self.split,("","IID","nonIID","iid","dirichlet","one_class_per_node")),(self.mitigation,("","none","ss","ebm","ss_ebm")),(self.noise,("","off","0.005","0.01","0.02","0.2","0.4","0.5","0.6"))): ttk.Combobox(filters,textvariable=variable,values=values,state="readonly",width=9).pack(side="left",padx=3)
        ttk.Button(filters,text="Search",command=self.refresh).pack(side="left",padx=6); ttk.Button(filters,text="Refresh",command=lambda:self.refresh(True)).pack(side="left")
        panes=ttk.Panedwindow(self,orient="horizontal"); panes.pack(fill="both",expand=True)
        left=ttk.Frame(panes); right=ttk.Frame(panes,padding=(18,0)); panes.add(left,weight=3); panes.add(right,weight=2)
        self.tree=ttk.Treeview(left,columns=("status","dataset","approach","split","average"),show="headings",selectmode="extended")
        for key,label,width in (("status","Status",90),("dataset","Dataset",90),("approach","Approach",90),("split","Split",90),("average","Final average",105)): self.tree.heading(key,text=label); self.tree.column(key,width=width)
        self.tree.pack(fill="both",expand=True); self.tree.bind("<<TreeviewSelect>>",lambda _e:self._selected())
        self.details=ttk.Label(right,text="Select a run to load details. Metrics remain unloaded until selection.",justify="left",wraplength=380,style="Muted.TLabel"); self.details.pack(fill="x",anchor="nw")
        self.chart=tk.Canvas(right,height=190,bg=COLORS["surface_alt"],highlightthickness=1,highlightbackground=COLORS["border"]); self.chart.pack(fill="x",pady=(12,4))
        ttk.Button(right,text="Replay in Network",command=self._replay).pack(anchor="w",pady=(14,4)); ttk.Button(right,text="Compare selected",command=self._compare).pack(anchor="w",pady=4); ttk.Button(right,text="Open output folder",command=self._open_output).pack(anchor="w",pady=4); ttk.Button(right,text="Export metadata",command=self._export).pack(anchor="w",pady=4); ttk.Button(right,text="Generate plots…",command=self._plot).pack(anchor="w",pady=4)
        self.refresh()
    def refresh(self,force=False):
        records=self.service.filter(self.service.discover(force),search=self.search.get(),dataset=self.dataset.get(),approach=self.approach.get(),status=self.status.get(),split=self.split.get(),mitigation=self.mitigation.get(),noise=self.noise.get()); self.records={str(i):r for i,r in enumerate(records)}; self.tree.delete(*self.tree.get_children())
        for key,r in self.records.items(): self.tree.insert("","end",iid=key,text=r.name,values=(r.status.title(),r.dataset,r.approach,r.split,"—" if r.final_average is None else f"{r.final_average:.2%}"))
        self.details.configure(text="No results match the current filters." if not records else "Select a run to load details. Metrics remain unloaded until selection.")
    def _selected(self):
        ids=self.tree.selection()
        if not ids:return
        r=self.records[ids[0]]; metrics=self.service.load_metrics(r) if not r.error else {}
        self.details.configure(text=f"{r.name}\n\nStatus: {r.status}\nDataset: {r.dataset} • {r.split}\nApproach: {r.approach}\nMitigation: {r.mitigation}\nNoise: {'off' if r.noise is None else r.noise}\nFinal average: {'—' if r.final_average is None else f'{r.final_average:.2%}'}\nFinal worst node: {'—' if r.final_worst is None else f'{r.final_worst:.2%}'}\nMetric arrays loaded: {', '.join(metrics) or 'none'}\n{r.error}")
        self.details.configure(text=self.details.cget("text")+f"\nPurpose: {r.purpose or 'historical research'}\nResearch-valid: {r.research_valid}\nNoise semantics: {r.noise_semantics or 'historical'}")
        self._draw_series([metrics.get("averageAccuracy",metrics.get("avg_history",[]))],(COLORS["accent"],))
    def _replay(self):
        ids=self.tree.selection()
        if not ids:return
        r=self.records[ids[0]]
        if not self.service.telemetry_path(r): messagebox.showinfo("Network telemetry","Network telemetry is not available for this run.",parent=self); return
        self.on_replay(r)
    def _compare(self):
        selected=[self.records[key] for key in self.tree.selection()]
        if len(selected)<2: messagebox.showinfo("Compare runs","Select at least two compatible runs.",parent=self); return
        if len({(r.dataset,r.split) for r in selected})>1: messagebox.showwarning("Compare runs","Selected runs use incompatible datasets or splits.",parent=self); return
        if len({(r.protocol,r.research_valid,r.purpose,r.noise_semantics) for r in selected})>1:messagebox.showwarning("Compare runs","Do not mix smoke, preflight, production, protocol revisions or different noise semantics.",parent=self); return
        loaded=[self.service.load_metrics(r) for r in selected];series=[m.get("averageAccuracy",m.get("avg_history",[])) for m in loaded]
        if any(len(values)==0 for values in series):messagebox.showwarning("Compare runs","One or more selected runs do not contain convergence metrics.",parent=self); return
        self._draw_series(series,(COLORS["accent"],COLORS["warning"],COLORS["success"],COLORS["info"],COLORS["danger"]))
        messagebox.showinfo("Run comparison","\n".join(f"{r.name}: average {'—' if r.final_average is None else f'{r.final_average:.2%}'}, worst {'—' if r.final_worst is None else f'{r.final_worst:.2%}'}" for r in selected),parent=self)
    def _draw_series(self,series,colors):
        self.chart.delete("all"); width=max(380,self.chart.winfo_width()); height=max(190,self.chart.winfo_height()); margin=16
        if not series or not len(series[0]):self.chart.create_text(width/2,height/2,text="Convergence data unavailable",fill=COLORS["muted"]); return
        longest=max(len(values) for values in series)
        for index,values in enumerate(series):
            points=[]
            for position,value in enumerate(values):points.extend((margin+(width-2*margin)*position/max(1,longest-1),height-margin-(height-2*margin)*max(0,min(1,float(value)))))
            if len(points)>=4:self.chart.create_line(*points,fill=colors[index%len(colors)],width=2)
    def _open_output(self):
        ids=self.tree.selection()
        if not ids:return
        import subprocess
        try:subprocess.Popen(["xdg-open",str(self.records[ids[0]].path.parent)])
        except OSError as error:messagebox.showerror("Open output",str(error),parent=self)
    def _export(self):
        ids=self.tree.selection()
        if not ids:return
        import json
        from tkinter import filedialog
        path=filedialog.asksaveasfilename(parent=self,defaultextension=".json",filetypes=(("JSON","*.json"),))
        if path:
            records=[self.records[key] for key in ids]; payload=[{"name":r.name,"status":r.status,"dataset":r.dataset,"approach":r.approach,"split":r.split,"finalAverageAccuracy":r.final_average,"finalWorstAccuracy":r.final_worst,"runPath":str(r.path)} for r in records]
            try:
                from pathlib import Path
                Path(path).write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
            except OSError as error:messagebox.showerror("Export results",str(error),parent=self)
    def _plot(self):
        selected=self.tree.selection()
        record=self.records[selected[0]] if selected else None
        if self.on_plot:self.on_plot(record)
