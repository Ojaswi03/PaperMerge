import tkinter as tk
from tkinter import ttk
from gui.theme import COLORS

class EmptyState(ttk.Frame):
    def __init__(self,parent,title,message):
        super().__init__(parent,padding=32); ttk.Label(self,text=title,font=("TkDefaultFont",16,"bold")).pack(pady=(0,8)); ttk.Label(self,text=message,style="Muted.TLabel",wraplength=520,justify="center").pack()

class ScrollFrame(ttk.Frame):
    def __init__(self,parent):
        super().__init__(parent); canvas=tk.Canvas(self,bg=COLORS["bg"],highlightthickness=0); bar=ttk.Scrollbar(self,orient="vertical",command=canvas.yview)
        self.body=ttk.Frame(canvas); window=canvas.create_window((0,0),window=self.body,anchor="nw")
        self.body.bind("<Configure>",lambda _e:canvas.configure(scrollregion=canvas.bbox("all"))); canvas.bind("<Configure>",lambda e:canvas.itemconfigure(window,width=e.width)); canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left",fill="both",expand=True); bar.pack(side="right",fill="y")
