import tkinter as tk
from tkinter import ttk
from gui.components import EmptyState
from gui.theme import COLORS

ACCURACY_SERIES = (
    ('#0072B2', (2, 3), 2, 'Node just trained'),
    ('#00875A', (), 3, 'Round mean'),
    ('#D55E00', (7, 3), 2, 'Round worst'),
)


def duration(seconds):
    hours, remainder=divmod(int(max(0,seconds)),3600)
    minutes, seconds=divmod(remainder,60)
    return f'{hours:d}h {minutes:02d}m {seconds:02d}s'

class DashboardView(ttk.Frame):
    def __init__(self,parent,state,queue_state,result_service=None):
        super().__init__(parent,padding=24)
        self.state=state; self.queue_state=queue_state; self.result_service=result_service
        self._mode=None; self._chart_signature=None
        ttk.Label(self,text="Research operations",style="Title.TLabel").pack(anchor="w")
        self.identity=ttk.Label(self,font=("TkDefaultFont",13,"bold"),wraplength=950)
        self.identity.pack(anchor="w",pady=(8,0))
        self.context=ttk.Label(self,style="Muted.TLabel",wraplength=950)
        self.context.pack(anchor="w",pady=(4,0))
        self.content=ttk.Frame(self); self.content.pack(fill="both",expand=True,pady=(18,0))
        self.empty=EmptyState(self.content,"No active experiment","Configure an experiment in Experiment Builder or select a queued run.")
        self.live=ttk.Frame(self.content)
        cards=ttk.Frame(self.live); cards.pack(fill="x")
        self.values={}
        for i,label in enumerate(("Run status","Progress","Round","Experiment elapsed","Experiment ETA remaining","Workers","Average accuracy","Worst-node accuracy","Queue","Queue elapsed (this session)","Queue ETA remaining")):
            card=ttk.Frame(cards,style="Card.TFrame",padding=12)
            card.grid(row=i//3,column=i%3,sticky="nsew",padx=4,pady=4); cards.columnconfigure(i%3,weight=1)
            ttk.Label(card,text=label,style="Card.TLabel").pack(anchor="w")
            self.values[label]=ttk.Label(card,font=("TkDefaultFont",18,"bold"),style="Card.TLabel")
            self.values[label].pack(anchor="w",pady=(8,0))
        self.latest=ttk.Label(self.live,style="Muted.TLabel"); self.latest.pack(anchor="w",pady=(12,4))
        ttk.Label(self.live,text="Live accuracy",font=("TkDefaultFont",14,"bold")).pack(anchor="w",pady=(8,4))
        self.chart=tk.Canvas(self.live,height=190,bg=COLORS["surface_alt"],highlightthickness=1,highlightbackground=COLORS["border"])
        self.chart.pack(fill="both",expand=True,pady=(8,0)); self.chart.bind("<Configure>",lambda _e:self._draw_chart())
        self.recent_frame=ttk.Frame(self.content); self.recent_frame.pack(side="bottom",fill="x",pady=(12,0))
        ttk.Label(self.recent_frame,text="Recent runs",font=("TkDefaultFont",14,"bold")).pack(anchor="w")
        self.recent=ttk.Label(self.recent_frame,style="Muted.TLabel",justify="left"); self.recent.pack(anchor="w")
        self.refresh()

    @staticmethod
    def _text(widget,text):
        if widget.cget('text')!=text:widget.configure(text=text)

    def refresh(self):
        state=self.state
        mode='empty' if state.execution.value in {'idle','stopped','completed','failed'} and state.current_round==0 else 'live'
        if mode!=self._mode:
            self.empty.pack_forget(); self.live.pack_forget()
            (self.empty if mode=='empty' else self.live).pack(fill="both",expand=mode=='live',before=self.recent_frame)
            self._mode=mode
        name=state.active_experiment_name or state.experiment.experiment_name
        running=state.execution.value in {'preparing','running','stopping'}
        self._text(self.identity,f'{"Running experiment" if running else "Experiment"}: {name}')
        recovering=state.recovery_completed<state.recovery_total
        context=(f'Rebuilding saved state: {state.recovery_completed}/{state.recovery_total} activations verified. '
            'This replays recorded training; it is not a new scientific run.' if recovering else
            'Experiment elapsed includes prior attempts. Queue elapsed is this launch session; ETAs are estimates.')
        if state.execution_device:context+=f' Device: {state.execution_device} — {state.device_name}.'
        self._text(self.context,context)
        elapsed=state.elapsed_seconds; eta=state.experiment_eta_seconds
        pending=[e for e in self.queue_state.entries if e.status.value=='pending']
        estimates=[e.estimated_remaining_seconds for e in pending]
        queue_eta=eta+sum(estimates) if state.queue_active and eta is not None and all(v is not None for v in estimates) else None
        rounds=state.active_experiment_rounds or state.experiment.rounds
        values=(state.execution.value.title(),f"{state.progress:.1%}",f"{state.current_round} / {rounds}",
            duration(elapsed),"Estimating…" if eta is None and running else "—" if eta is None else duration(eta),str(state.worker_count),
            "—" if state.average_accuracy is None else f"{state.average_accuracy:.2%}",
            "—" if state.worst_accuracy is None else f"{state.worst_accuracy:.2%}",f'{len(pending)} pending',
            duration(state.queue_elapsed_seconds) if state.queue_started_monotonic is not None else 'Not started',
            duration(queue_eta) if queue_eta is not None else 'Estimating…' if state.queue_active else 'Paused / not started')
        for label,value in zip(self.values,values):self._text(self.values[label],value)
        latest=("Waiting for the first completed node activation."
            if state.latest_node_accuracy is None else f"Latest completed node {state.latest_node}: {state.latest_node_accuracy:.2%}. Mean/worst update after a full round.")
        self._text(self.latest,latest)
        signature=(tuple(state.accuracy_history),tuple(state.activation_accuracy_history))
        if signature!=self._chart_signature:self._chart_signature=signature; self._draw_chart()
        recent=self.result_service.discover() if self.result_service else []
        protocol=state.experiment.extra.get('experimentProtocol')
        if protocol:recent=[r for r in recent if r.protocol==protocol]
        recent=recent[:5]
        self._text(self.recent,"\n".join(f"{item.status.title()}  •  {item.name}" for item in recent) or "No completed or failed runs discovered.")

    def _draw_chart(self):
        canvas=self.chart; canvas.delete("all")
        width=canvas.winfo_width(); height=canvas.winfo_height()
        if width<100 or height<60:return
        state=self.state; margin=38; bottom=height-28; top=24
        series=(state.activation_accuracy_history,[(r,a) for r,a,_ in state.accuracy_history],[(r,w) for r,_,w in state.accuracy_history])
        maximum=max([1.,*(r for data in series for r,_ in data)])
        canvas.create_text(5,top,text='100%',anchor='w',fill=COLORS['muted'])
        canvas.create_text(5,bottom,text='0%',anchor='w',fill=COLORS['muted'])
        canvas.create_text(width/2,height-10,text='Completed ring rounds (fractional = completed nodes)',fill=COLORS['muted'])
        if not any(series):
            canvas.create_text(width/2,height/2,text='Accuracy appears after each completed node activation.',fill=COLORS['muted']);return
        for i,(data,(color,dash,line_width,label)) in enumerate(zip(series,ACCURACY_SERIES)):
            canvas.create_text(margin+i*(width-2*margin)/3,10,text=label,fill=color,anchor='w')
            if not data:continue
            points=[coordinate for r,a in data for coordinate in (margin+(width-2*margin)*r/maximum,bottom-(bottom-top)*a)]
            if len(data)>1:canvas.create_line(*points,fill=color,width=line_width,dash=dash)
            x,y=points[-2:]; canvas.create_oval(x-3,y-3,x+3,y+3,fill=color,outline=color)
