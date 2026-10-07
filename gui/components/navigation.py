from tkinter import ttk
from gui.state import Workspace

class NavigationRail(ttk.Frame):
    def __init__(self,parent,on_navigate):
        super().__init__(parent,style="Surface.TFrame",padding=(8,18)); self.columnconfigure(0,weight=1)
        ttk.Label(self,text="PaperMerge",font=("TkDefaultFont",17,"bold"),style="Surface.TLabel").grid(row=0,column=0,sticky="w",padx=10,pady=(0,22))
        for row,(workspace,label) in enumerate(((Workspace.DASHBOARD,"Dashboard"),(Workspace.BUILDER,"Experiment Builder"),(Workspace.QUEUE,"Queue"),(Workspace.RESULTS,"Results"),(Workspace.NETWORK,"Network")),1):
            ttk.Button(self,text=label,style="Nav.TButton",command=lambda w=workspace:on_navigate(w)).grid(row=row,column=0,sticky="ew",pady=2)
