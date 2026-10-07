from __future__ import annotations
import logging
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from gui.services import ConfigService, ExecutionService, IsolatedWorkerHandle, PlotService, QueueService, ResultService
from gui.state import ApplicationState, QueueState, Workspace
from gui.state.experiment_state import to_persisted
from gui.state.experiment_state import validate_experiment
from gui.theme import apply_theme
from gui.views import ApplicationShell
from gui.runtime_estimator import RuntimeEstimator
from gui.worker_pool import find_orphan_workers, terminate_orphan_workers
from basil_core.protocol_compatibility import LEGACY_RESULT_ROOTS

LOG=logging.getLogger(__name__); PROJECT_ROOT=Path(__file__).resolve().parents[1]

class PaperMergeApp:
    def __init__(self,root: tk.Tk):
        self.root=root; root.title("PaperMerge Experiment Workspace"); root.geometry("1440x900"); root.minsize(1100,720); root.rowconfigure(0,weight=1); root.columnconfigure(0,weight=1); apply_theme(root)
        self.state=ApplicationState(); self.config_service=ConfigService(); self.queue_service=QueueService(PROJECT_ROOT/"gui"/"queue_state.json")
        try:self.queue_state=self.queue_service.load()
        except ValueError as error:self.queue_state=QueueState(); self.state.status=str(error)
        self.queue_service.describe_recovery(self.queue_state,PROJECT_ROOT)
        self.results=ResultService(PROJECT_ROOT/"experiments"/"results4",(PROJECT_ROOT/"newResults/IID",PROJECT_ROOT/"experiments/research_protocol_results", *(PROJECT_ROOT/path for path in LEGACY_RESULT_ROOTS))); self.plot_service=PlotService(PROJECT_ROOT/"plots4"); self.execution=ExecutionService(self.state,launcher=self._launch); self._worker_handle=None; self._active_queue_entry=None; self._queue_running=False
        if self.queue_state.entries and self.queue_state.entries[0].config.extra.get('iidCampaignId'):
            self.state.experiment=self.queue_state.entries[0].config
        commands={"save":self.save_config,"load":self.load_config,"queue":self.add_to_queue,"run":self.run,"stop":self.stop,"queue_changed":self.queue_changed,"library":self.add_from_library,"plot":self.generate_plots,'load_queue':self.load_queue}
        self.runtime_estimator=RuntimeEstimator((PROJECT_ROOT/"experiments"/"results4",))
        self.shell=ApplicationShell(root,self.state,self.queue_state,self.results,commands,self.runtime_estimator); self.state.subscribe(self.shell.refresh)
        root.bind("<Control-s>",lambda _e:self.save_config()); root.bind("<Control-l>",lambda _e:self.load_config()); root.bind("<Control-r>",lambda _e:self.run()); root.bind("<Escape>",lambda _e:self.stop()); root.protocol("WM_DELETE_WINDOW",self.close); self._poll()
    def save_config(self):
        path=filedialog.asksaveasfilename(parent=self.root,defaultextension=".json",filetypes=(("JSON","*.json"),))
        if not path:return
        try:self.config_service.save(path,self.state.experiment); self.state.dirty=False; self.state.status=f"Saved {Path(path).name}"
        except Exception as error:LOG.exception("Configuration save failed"); messagebox.showerror("Persistence error",f"Configuration could not be saved.\n\n{error}",parent=self.root)
        self.state.notify()
    def load_config(self):
        path=filedialog.askopenfilename(parent=self.root,filetypes=(("JSON","*.json"),))
        if not path:return
        try:self.state.experiment=self.config_service.load(path); self.state.dirty=False; self.state.status=f"Loaded {Path(path).name}"; self.shell.views[Workspace.BUILDER].destroy(); from gui.views.experiment_builder_view import ExperimentBuilderView; view=ExperimentBuilderView(self.shell.stage,self.state,self.shell.refresh); view.grid(row=0,column=0,sticky="nsew"); self.shell.views[Workspace.BUILDER]=view; self.shell.navigate(Workspace.BUILDER)
        except Exception as error:LOG.exception("Configuration load failed"); messagebox.showerror("Persistence error",str(error),parent=self.root)
    def add_to_queue(self):
        if validate_experiment(self.state.experiment): return
        self.queue_state.add(self.config_service.load_dict(to_persisted(self.state.experiment))); self.queue_changed(); self.state.status="Configuration added to queue"
    def queue_changed(self):
        self.queue_service.describe_recovery(self.queue_state,PROJECT_ROOT)
        try:self.queue_service.save(self.queue_state)
        except Exception as error:LOG.exception("Queue persistence failed"); self.state.status=f"Queue could not be saved: {error}"
        self.state.notify()
    def add_from_library(self):
        from basil_core.iid_campaign import CONFIG_ROOT
        paths=filedialog.askopenfilenames(parent=self.root,title="Add configurations",initialdir=CONFIG_ROOT,filetypes=(("JSON","*.json"),))
        added=0
        for path in paths:
            try:self.queue_state.add(self.config_service.load(path)); added+=1
            except ValueError as error:messagebox.showwarning("Invalid configuration",f"{Path(path).name}: {error}",parent=self.root)
        if added:
            self.queue_changed(); self.state.status=f"Added {added} configuration(s) from the library"
    def load_queue(self):
        if self.state.can_stop or self._queue_running:return
        path=filedialog.askopenfilename(parent=self.root,title='Load idle queue preset',initialdir=PROJECT_ROOT/'gui/queues',filetypes=(("JSON","*.json"),))
        if not path:return
        try:
            prepared=QueueService(path).load()
            if any(validate_experiment(e.config) for e in prepared.entries):raise ValueError('Queue contains an invalid configuration.')
            if self.queue_state.entries and not messagebox.askyesno('Replace queue','Replace the current idle queue with this preset?',parent=self.root):return
            self.queue_state.entries[:]=prepared.entries
            if prepared.entries:self.state.experiment=prepared.entries[0].config
            self.queue_changed();self.shell.navigate(Workspace.QUEUE);self.state.status=f'{len(prepared.entries)} queued experiments — idle; click Run Queue to start'
            self.state.notify()
        except Exception as error:messagebox.showerror('Queue error',str(error),parent=self.root)
    def generate_plots(self,record=None):
        output=filedialog.askdirectory(parent=self.root,title="Select a new plot output directory",mustexist=False)
        if not output:return
        import threading
        self.state.status="Generating plots…"; self.state.notify()
        def work():
            try:
                result=self.plot_service.generate_research(record.path.parent,output) if record and record.protocol else self.plot_service.generate_adaptive(output); errors=result.get("errors",[])
                message=f"Generated {len(result.get('generated',[]))} plots in {output}." if not errors else f"Plot generation completed with {len(errors)} error(s). See the application log."
                self.root.after(0,self._plot_finished,message,bool(errors))
            except Exception as error:self.root.after(0,self._plot_finished,f"Plot generation failed: {error}",True)
        threading.Thread(target=work,name="plot-generation",daemon=True).start()
    def _plot_finished(self,message,failed):
        self.state.status=message; self.state.notify()
        (messagebox.showerror if failed else messagebox.showinfo)("Plot generation",message,parent=self.root)
    def run(self):
        if not self.state.can_run or self._queue_running:return
        pending=next((entry for entry in self.queue_state.entries if entry.status.value=="pending"),None) if self.state.workspace is Workspace.QUEUE else None
        if self.state.workspace is Workspace.QUEUE and pending is None:return
        if pending is not None:
            self._queue_running=True; self.state.begin_queue()
        self._start_entry(pending)
    def _start_entry(self,pending=None):
        if pending is not None:self.state.experiment=pending.config
        config=to_persisted(self.state.experiment)
        from basil_core.iid_campaign import gui_approved
        if int(config.get("campaignVersion",0)) not in {3,4} and config.get("experimentProtocol")!="sequential_basil_one_class_v1" and not gui_approved(config):
            self._queue_running=False; self.state.end_queue(); messagebox.showinfo("Run configuration","Add or load a versioned baseline/adaptive-study configuration before starting isolated execution. This prevents silently assigning research protocol fields.",parent=self.root); return
        orphans=[]
        for root in (PROJECT_ROOT/'gui/worker_state/iid',PROJECT_ROOT/"experiments"/"results3"/"r2"/"workers",PROJECT_ROOT/"experiments"/"results4"/"campaign4"/"workers",PROJECT_ROOT/"experiments/worker_state/adaptive",PROJECT_ROOT/"experiments/worker_state/baseline",PROJECT_ROOT/"experiments/research_protocol_results/workers", *(PROJECT_ROOT/path/"workers" for path in LEGACY_RESULT_ROOTS)): orphans.extend(find_orphan_workers(root))
        if orphans:
            names=", ".join(f"{item.get('runId','?')} (pid {item.get('pid','?')})" for item in orphans)
            if not messagebox.askyesno("Orphaned workers",f"Active workers from an earlier session were found:\n\n{names}\n\nRequest graceful termination before starting?",parent=self.root): self._queue_running=False; self.state.end_queue(); return
            terminate_orphan_workers(orphans)
        try:
            self.execution.start(config); self.state.status=f"Running {self.state.experiment.experiment_name}"
            if pending is not None:
                from gui.state import QueueStatus
                self._active_queue_entry=pending; self.queue_state.transition(pending.entry_id,QueueStatus.RUNNING); self.queue_changed()
        except Exception as error:
            self._queue_running=False;(PROJECT_ROOT/'gui/.queue_active.json').unlink(missing_ok=True)
            self.state.end_queue()
            LOG.exception("Execution start failed"); messagebox.showerror("Execution error",str(error),parent=self.root)
    def stop(self): self._queue_running=False; self.state.end_queue(); self.execution.request_stop(); self.state.status="Graceful stop requested"; self.state.notify()
    def _launch(self,config):
        if config.get('iidCampaignId'):
            from gui.services.iid_execution_service import IidExecutionHandle
            self._worker_handle=IidExecutionHandle(config,self.execution.post)
        else:self._worker_handle=IsolatedWorkerHandle(config,self.execution.post)
        (PROJECT_ROOT/'gui/.queue_active.json').write_text(__import__('json').dumps({'runId':config.get('runId'),'pid':__import__('os').getpid()}))
        return self._worker_handle
    def _poll(self):
        self.state.tick()
        if self._worker_handle and self._worker_handle.active:self._worker_handle.poll()
        self.execution.dispatch_pending()
        if self._active_queue_entry:
            done=max(self._active_queue_entry.completed_rounds,self.state.completed_rounds)
            if done!=self._active_queue_entry.completed_rounds:
                self._active_queue_entry.completed_rounds=done
                if hasattr(self,'shell'):self.shell.views[Workspace.QUEUE].refresh()
        if hasattr(self,'shell') and time.monotonic()-getattr(self,'_last_clock_refresh',0)>=1:
            self.shell.views[Workspace.DASHBOARD].refresh(); self._last_clock_refresh=time.monotonic()
        if self._active_queue_entry and self.state.execution.value in {"completed","failed","stopped"}:
            from gui.state import QueueStatus
            target={"completed":QueueStatus.COMPLETED,"failed":QueueStatus.FAILED,"stopped":QueueStatus.STOPPED}[self.state.execution.value]
            if self._active_queue_entry.status is QueueStatus.RUNNING:self.queue_state.transition(self._active_queue_entry.entry_id,target); self.queue_changed()
            self._active_queue_entry=None
            next_entry=next((entry for entry in self.queue_state.entries if entry.status is QueueStatus.PENDING),None)
            if self._queue_running and target is QueueStatus.COMPLETED and next_entry is not None:self.root.after(100,lambda:self._start_entry(next_entry) if self._queue_running and next_entry in self.queue_state.entries and next_entry.status is QueueStatus.PENDING else None)
            else:
                self._queue_running=False; self.state.end_queue()
        if not self._queue_running and self.state.execution.value in {'completed','failed','stopped'} and not (self._worker_handle and self._worker_handle.active):
            (PROJECT_ROOT/'gui/.queue_active.json').unlink(missing_ok=True)
        self.root.after(150,self._poll)
    def close(self):
        if self.state.execution.value in {"preparing","running","stopping"} or (self._worker_handle and self._worker_handle.active):
            if self.state.execution.value!="stopping":
                if not messagebox.askyesno("Active execution","Request a graceful stop and close after workers exit?",parent=self.root):return
                self._queue_running=False; self.state.end_queue()
                self.execution.request_stop()
            self.root.after(500,self.close); return
        self._queue_running=False; self.state.end_queue(); self.queue_changed(); (PROJECT_ROOT/'gui/.queue_active.json').unlink(missing_ok=True); self.root.destroy()

def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root=tk.Tk(); PaperMergeApp(root); root.mainloop()

if __name__=="__main__": main()
