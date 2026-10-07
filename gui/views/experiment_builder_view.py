from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from gui.components import ScrollFrame
from gui.state.experiment_state import ATTACK_NAMES, ExperimentConfig, validate_experiment
from gui.theme import COLORS

CHOICE_LABELS={
    "channel_noise_semantics":{"paper_absolute_gaussian":"Paper absolute Gaussian","relative_l2_gaussian":"Relative-L2 Gaussian","absolute_coordinate_gaussian":"Paper absolute Gaussian"},
    "ebm_mode":{"none":"Disabled","gradient_norm_objective":"Full gradient-norm objective","legacy_gradient_scale":"Legacy gradient scaling"},
}

class ExperimentBuilderView(ttk.Frame):
    def __init__(self,parent,state,on_change):
        super().__init__(parent); self.state=state; self.on_change=on_change; self.vars={}; self.widgets={}; self.labels={}
        scroll=ScrollFrame(self); scroll.pack(fill="both",expand=True); body=scroll.body; body.columnconfigure((0,1),weight=1)
        header=ttk.Frame(body);header.grid(row=0,column=0,columnspan=2,sticky="ew",padx=24,pady=(24,8))
        from basil_core.iid_campaign import CONDITIONS
        self.iid_presets={f'{letter} — {item[5]}':item[0] for letter,item in CONDITIONS.items()}
        self.iid_presets.update({f'{letter} — {item[5]} — GPU':f'gpu/{item[0]}_gpu' for letter,item in CONDITIONS.items()})
        for letter in 'ABCD':
            item = CONDITIONS[letter]
            self.iid_presets[f'{letter} — {item[5]} — GPU'] = f'gpu/fresh/{item[0]}_gpu'
        ttk.Label(header,text="Experiment Builder",style="Title.TLabel").pack(anchor="w"); actions=ttk.Frame(header); actions.pack(anchor="w",pady=(12,0)); self.preset=tk.StringVar(value="Preset…"); ttk.Combobox(actions,textvariable=self.preset,values=("Clean baseline","Noisy + EBM","BASIL + Hidden","Sequential BASIL five-epoch protocol",*self.iid_presets),state="readonly",width=40).pack(side="left",padx=4); ttk.Button(actions,text="Apply",width=6,command=self._apply_preset).pack(side="left",padx=4); ttk.Button(actions,text="Reset defaults",width=14,command=self._reset).pack(side="left")
        self._section(body,"Identity",1,(("Experiment name","experiment_name","entry"),))
        self._section(body,"Dataset and distribution",2,(("Dataset","dataset",("cifar10","mnist","nmnist")),("Partition","partition_strategy",("iid","dirichlet","one_class_per_node")),("Dirichlet alpha","dirichlet_alpha","entry")))
        self._section(body,"Approach and topology",3,(("Approach","approach",("basil","noisy","merged","cart")),("Node count","node_count","entry"),("Model architecture","model_architecture",("basil_paper_cnn","legacy_vgg_cnn")),("Aggregation","aggregation_mode",("strict_sequential_handoff","handoff","consensus"))))
        self._section(body,"Training parameters",4,(("Rounds","rounds","entry"),("Local epochs","local_epochs","entry"),("Batch size","batch_size","entry"),("Learning rate","learning_rate","entry"),("Epoch semantics","epoch_semantics",("full_local_dataset","fixed_steps"))))
        self.noise_frame=self._section(body,"Channel noise",5,(("Enabled","use_channel_noise","check"),("Noise semantics","channel_noise_semantics",("relative_l2_gaussian","paper_absolute_gaussian")),("Sigma_rel (0.2 ≈ 20% L2)","channel_noise_sigma_rel","entry"),("Sigma_e (coordinate std)","channel_noise_sigma_absolute","entry"),("Historical channel sigma","channel_noise_sigma","entry"),("Start round","channel_noise_start","entry")))
        self.mitigation_frame=self._section(body,"Noise mitigation",6,(("Mitigation","noise_mitigation",("none","ebm")),("EBM implementation","ebm_mode",("none","gradient_norm_objective","legacy_gradient_scale")),("EBM lambda","ebm_lambda","entry"),("Snapshot Selection","snapshot_selection","check"),("Snapshot memory","basil_memory_size","entry")))
        self.cart_frame=self._section(body,"CART-specific parameters",7,(("CART algorithm","cart_algorithm",("cart","basil","noisy","merged")),("Distillation strength","distill_strength","entry"),("Verification threshold","verify_threshold","entry")))
        self.attack_frame=ttk.Frame(body,style="Card.TFrame",padding=16); self.attack_frame.grid(row=8,column=0,columnspan=2,sticky="ew",padx=24,pady=8); ttk.Label(self.attack_frame,text="Byzantine attacks",style="Section.TLabel").grid(row=0,column=0,columnspan=4,sticky="w",pady=(0,10))
        self._field(self.attack_frame,"Attacker IDs","attacker_ids","entry",1,0)
        for index,name in enumerate(ATTACK_NAMES):
            key=f"attack_{name}"; self.vars[key]=tk.BooleanVar(value=self.state.experiment.attacks.get(name,False)); ttk.Checkbutton(self.attack_frame,text=name,variable=self.vars[key],command=self._changed).grid(row=2+index//4,column=(index%4)*2,sticky="w",padx=5,pady=5)
            start=f"start_{name}"; self.vars[start]=tk.StringVar(value=str(self.state.experiment.attack_starts.get(name,0))); entry=ttk.Entry(self.attack_frame,textvariable=self.vars[start],width=7); entry.grid(row=2+index//4,column=(index%4)*2+1,padx=(0,12)); entry.bind("<KeyRelease>",lambda _e:self._changed())
        self.advanced_frame=self._section(body,"Advanced execution settings",9,(("Momentum","momentum","entry"),)); self.advanced_visible=False; self.advanced_frame.grid_remove(); ttk.Button(body,text="Show / hide advanced settings",command=self._toggle_advanced).grid(row=10,column=0,sticky="w",padx=24,pady=8)
        self.summary=ttk.Label(body,text="",wraplength=850,justify="left",style="Muted.TLabel"); self.summary.grid(row=11,column=0,columnspan=2,sticky="ew",padx=24,pady=18)
        self.errors=ttk.Label(body,text="",foreground=COLORS["danger"],wraplength=850,justify="left"); self.errors.grid(row=12,column=0,columnspan=2,sticky="ew",padx=24,pady=(0,24)); self.refresh()
    def _section(self,parent,title,row,specs):
        frame=ttk.Frame(parent,style="Card.TFrame",padding=16); frame.grid(row=row,column=0,columnspan=2,sticky="ew",padx=24,pady=8); frame.columnconfigure((1,3),weight=1); ttk.Label(frame,text=title,style="Section.TLabel").grid(row=0,column=0,columnspan=4,sticky="w",pady=(0,10))
        for index,(label,key,kind) in enumerate(specs): self._field(frame,label,key,kind,1+index//2,(index%2)*2)
        return frame
    def _field(self,parent,label,key,kind,row,column):
        value=getattr(self.state.experiment,key); variable=tk.BooleanVar(value=value) if kind=="check" else tk.StringVar(value=str(value)); self.vars[key]=variable
        if key in CHOICE_LABELS:
            variable.set(CHOICE_LABELS[key].get(value,value));kind=tuple(CHOICE_LABELS[key][code] for code in kind)
        if kind=="check": widget=ttk.Checkbutton(parent,text=label,variable=variable,command=self._changed); widget.grid(row=row,column=column,columnspan=2,sticky="w",padx=5,pady=5)
        else:
            label_widget=ttk.Label(parent,text=label,style="Card.TLabel"); label_widget.grid(row=row,column=column,sticky="w",padx=5,pady=5); self.labels[key]=label_widget; widget=ttk.Combobox(parent,textvariable=variable,values=kind,state="readonly") if isinstance(kind,tuple) else ttk.Entry(parent,textvariable=variable)
            widget.grid(row=row,column=column+1,sticky="ew",padx=(4,18),pady=5); widget.bind("<<ComboboxSelected>>",lambda _e:self._changed()); widget.bind("<KeyRelease>",lambda _e:self._changed())
        self.widgets[key]=widget
    def _changed(self):
        cfg=self.state.experiment
        cfg.form_errors={}
        string_fields=("experiment_name","dataset","approach","partition_strategy","noise_mitigation","attacker_ids","cart_algorithm","channel_noise_semantics","ebm_mode","model_architecture","aggregation_mode","epoch_semantics")
        int_fields=("node_count","rounds","local_epochs","batch_size","channel_noise_start","basil_memory_size")
        float_fields=("dirichlet_alpha","learning_rate","channel_noise_sigma","channel_noise_sigma_rel","channel_noise_sigma_absolute","ebm_lambda","distill_strength","verify_threshold","momentum")
        for key in string_fields: setattr(cfg,key,self._value(key))
        for key in ("use_channel_noise","snapshot_selection"): setattr(cfg,key,bool(self.vars[key].get()))
        cfg.non_iid=cfg.partition_strategy!="iid"
        for key in int_fields:
            try: setattr(cfg,key,int(self.vars[key].get()))
            except ValueError: cfg.form_errors[key]=f"{key.replace('_',' ').capitalize()} must be an integer."
        for key in float_fields:
            try: setattr(cfg,key,float(self.vars[key].get()))
            except ValueError: cfg.form_errors[key]=f"{key.replace('_',' ').capitalize()} must be a number."
        for name in ATTACK_NAMES:
            cfg.attacks[name]=bool(self.vars[f"attack_{name}"].get())
            try: cfg.attack_starts[name]=int(self.vars[f"start_{name}"].get())
            except ValueError: cfg.attack_starts[name]=-1
        self.state.dirty=True; self._conditional(); self._validation(); self.on_change()
    def _toggle_advanced(self):
        self.advanced_visible=not self.advanced_visible
        self.advanced_frame.grid() if self.advanced_visible else self.advanced_frame.grid_remove()
    def _reset(self):
        self.state.experiment=ExperimentConfig(); self._load_model(); self.state.dirty=True; self.on_change()
    def _apply_preset(self):
        config=ExperimentConfig()
        if self.preset.get() in self.iid_presets:
            from basil_core.iid_campaign import CONFIG_ROOT
            from gui.services.config_service import ConfigService
            config=ConfigService().load(CONFIG_ROOT/f'{self.iid_presets[self.preset.get()]}.json')
        if self.preset.get()=="Noisy + EBM": config.use_channel_noise=True; config.noise_mitigation="ebm"
        elif self.preset.get()=="BASIL + Hidden": config.snapshot_selection=True; config.attacks["Hidden"]=True; config.attacker_ids="1,4,6,8"
        elif self.preset.get()=="Sequential BASIL five-epoch protocol":
            from gui.research_protocol import base_config
            from gui.state.experiment_state import from_persisted
            config=from_persisted(base_config())
        self.state.experiment=config; self._load_model(); self.state.dirty=True; self.on_change()
    def _load_model(self):
        config=self.state.experiment
        for key,variable in self.vars.items():
            if key.startswith("attack_"): variable.set(config.attacks.get(key[7:],False))
            elif key.startswith("start_"): variable.set(str(config.attack_starts.get(key[6:],0)))
            elif hasattr(config,key):
                value=getattr(config,key);variable.set(CHOICE_LABELS.get(key,{}).get(value,value))
        self._conditional(); self._validation()
    def _conditional(self):
        # Widgets stay grouped, while irrelevant details are disabled and visually unavailable.
        noise=bool(self.vars["use_channel_noise"].get()); ebm=self._value("ebm_mode")!="none"; snapshot=bool(self.vars["snapshot_selection"].get()); relative=self._value("channel_noise_semantics")=="relative_l2_gaussian";research=bool(self.state.experiment.extra.get("experimentProtocol"))
        for key,enabled in (("channel_noise_sigma_rel",noise and relative),("channel_noise_sigma_absolute",noise and not relative),("channel_noise_sigma",noise and not research),("channel_noise_start",noise),("ebm_lambda",ebm and self._value("ebm_mode")=="legacy_gradient_scale"),("basil_memory_size",snapshot),("dirichlet_alpha",self._value("partition_strategy")=="dirichlet")):
            if enabled:
                self.widgets[key].grid(); self.labels[key].grid()
            else:
                self.widgets[key].grid_remove(); self.labels[key].grid_remove()
        # Re-grid attack start entries only while their attack is active.
        for name in ATTACK_NAMES:
            entry=self.attack_frame.grid_slaves(row=2+ATTACK_NAMES.index(name)//4,column=(ATTACK_NAMES.index(name)%4)*2+1)
            if entry: entry[0].grid() if self.vars[f"attack_{name}"].get() else entry[0].grid_remove()
        if self.vars["approach"].get()=="cart": self.cart_frame.grid()
        else:self.cart_frame.grid_remove()
    def _validation(self):
        issues=validate_experiment(self.state.experiment); self.summary.configure(text=self.state.experiment.dirty_summary); self.errors.configure(text="Configuration ready" if not issues else "\n".join(f"• {issue.message}" for issue in issues[:6]),foreground=COLORS["success"] if not issues else COLORS["danger"])
        if not issues and self.state.experiment.extra.get('iidCampaignId'):
            self.errors.configure(text='100-round IID configuration ready. Research runs start only when you click Run Queue.',foreground=COLORS['success'])
        elif not issues and self.state.experiment.extra.get("experimentProtocol") and self.state.experiment.extra.get("researchValid",True):
            self.errors.configure(text="Production gated: run smoke and full-data preflight first. Launch approved production explicitly from the CLI.",foreground=COLORS["warning"])
    def _value(self,key):
        value=self.vars[key].get()
        return next((code for code,label in CHOICE_LABELS.get(key,{}).items() if label==value),value)
    def refresh(self): self._conditional(); self._validation()
