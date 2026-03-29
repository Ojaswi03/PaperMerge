"""
Graphical User Interface for BASIL + Noisy Channel Experiments

This GUI allows you to:
- Select datasets (MNIST, CIFAR-10, N-MNIST)
- Choose training approach (BASIL, Noisy Channel, Merged)
- Configure attacks and when they start
- Configure channel noise and when it starts
- Set all training parameters
- Run experiments and view results with a live accuracy chart
"""
# Set matplotlib backend BEFORE any other imports to avoid tkinter conflicts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os
import sys
import time
import re
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog, filedialog
import threading
import json
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots'))

from basil_core.data.mnist import loadMnist, makeLoaders as makeMnistLoaders
from basil_core.data.cifar import loadCifar10, makeLoaders as makeCifarLoaders
from basil_core.data.nMnist import loadNMnist, makeLoaders as makeNMnistLoaders
from basil_core.models import MNISTModel, CIFARModel, NMNISTModel
from basil_core.basil import BasilNode, basilRingTrainingWithAttack, fedAvgTrainingWithNoise
from basil_core.trainer import evaluateAll
from scripts.common import setupGpu, sendNotification
from plotGui import discoverDatasets, discoverExperiments, getColors, getMarkers

# ─── Colour palette ──────────────────────────────────────────────────────────
BG        = '#f7f8fc'
PANEL_BG  = '#ffffff'
ACCENT    = '#2563eb'
ACCENT_DK = '#1d4ed8'
DANGER    = '#dc2626'
DANGER_DK = '#b91c1c'
SUCCESS   = '#16a34a'
INFO_CLR  = '#0369a1'
WARN_CLR  = '#d97706'
HEADER    = '#1e293b'
MUTED     = '#64748b'
BORDER    = '#e2e8f0'
CHART_BG  = '#f8fafc'

# ─── Quick presets ───────────────────────────────────────────────────────────
PRESETS = {
    "Clean Baseline": {
        'dataset': 'cifar10', 'approach': 'basil', 'useBasil': True,
        'useChannelNoise': False, 'noiseMitigation': 'none',
        'attackGaussian': False, 'attackSignFlip': False, 'attackHidden': False,
        'attackerIds': '', 'nNodes': 10, 'nRounds': 100, 'localEpochs': 5,
        'learningRate': 0.05, 'batchSize': 512, 'momentum': 0.9, 'useLrDecay': True,
    },
    "Noisy + EBM": {
        'dataset': 'cifar10', 'approach': 'noisy', 'useBasil': False,
        'useChannelNoise': True, 'channelNoiseSigma': 0.2, 'noiseMitigation': 'ebm',
        'ebmLambda': 25.0, 'attackerIds': '', 'nNodes': 10, 'nRounds': 100,
        'localEpochs': 5, 'learningRate': 0.05, 'batchSize': 512,
        'momentum': 0.9, 'useLrDecay': False,
    },
    "BASIL + Gaussian": {
        'dataset': 'cifar10', 'approach': 'basil', 'useBasil': True,
        'useChannelNoise': False, 'noiseMitigation': 'none',
        'attackGaussian': True, 'attackGaussianStart': 0,
        'attackerIds': '0,3,5,7', 'nNodes': 10, 'nRounds': 100,
        'localEpochs': 5, 'learningRate': 0.05, 'batchSize': 512,
        'momentum': 0.9, 'useLrDecay': True,
    },
    "Merged (Best)": {
        'dataset': 'cifar10', 'approach': 'merged', 'useBasil': True,
        'useChannelNoise': True, 'channelNoiseSigma': 0.2, 'noiseMitigation': 'ebm',
        'ebmLambda': 25.0, 'attackGaussian': False, 'attackerIds': '',
        'nNodes': 10, 'nRounds': 100, 'localEpochs': 5, 'learningRate': 0.05,
        'batchSize': 512, 'momentum': 0.9, 'useLrDecay': False,
    },
}


class ExperimentGUI:
    # ── construction ─────────────────────────────────────────────────────────
    def __init__(self, root):
        self.root = root
        self.root.title("BASIL + Noisy Channel Experiment GUI")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 720)
        self.root.configure(bg=BG)

        # Running state
        self.isRunning = False
        self.currentThread = None

        # Live-chart / progress tracking
        self._liveAccData    = []
        self._liveWorstData  = []
        self._trainStartTime = None
        self._lastRoundEndTime = None
        self._totalRounds    = 0
        self._roundTimes     = []   # per-round durations for EMA
        self._emaRoundTime   = None # exponential moving average of round time
        self._smoothedEta    = None # smoothed ETA to avoid jarring jumps

        self._setupStyle()
        self.setupVariables()
        self.createUI()
        self._setupKeyboardShortcuts()

        self.root.protocol("WM_DELETE_WINDOW", self.onClosing)
        self._setStatus("Ready  ·  Ctrl+R = Run   Ctrl+S = Save   Ctrl+L = Load   Esc = Stop   Ctrl+Shift+R = Reload")

    # ── style ─────────────────────────────────────────────────────────────────
    def _setupStyle(self):
        s = ttk.Style()
        s.theme_use('clam')

        s.configure('.',               background=BG,     font=('Segoe UI', 9))
        s.configure('TFrame',          background=BG)
        s.configure('TLabelframe',     background=BG,     bordercolor=BORDER)
        s.configure('TLabelframe.Label', font=('Segoe UI', 9, 'bold'), foreground=HEADER, background=BG)
        s.configure('TLabel',          background=BG,     foreground=HEADER)
        s.configure('TEntry',          fieldbackground=PANEL_BG, foreground=HEADER)
        s.configure('TRadiobutton',    background=BG,     foreground=HEADER)
        s.configure('TCheckbutton',    background=BG,     foreground=HEADER)
        s.configure('TSeparator',      background=BORDER)
        s.configure('TNotebook',       background=BG,     tabmargins=[2, 2, 2, 0])
        s.configure('TNotebook.Tab',   background='#e2e8f0', foreground=MUTED,
                    padding=[12, 5], font=('Segoe UI', 9))
        s.map('TNotebook.Tab',
              background=[('selected', PANEL_BG)],
              foreground=[('selected', ACCENT)])

        # Buttons
        s.configure('TButton',         padding=(8, 4),    font=('Segoe UI', 9))
        s.configure('Run.TButton',     background=ACCENT, foreground='white',
                    font=('Segoe UI', 10, 'bold'), padding=(14, 6))
        s.map('Run.TButton',           background=[('active', ACCENT_DK), ('disabled', '#93c5fd')])
        s.configure('RunAll.TButton',  background='#0891b2', foreground='white',
                    font=('Segoe UI', 10, 'bold'), padding=(14, 6))
        s.map('RunAll.TButton',        background=[('active', '#0e7490'), ('disabled', '#67e8f9')])
        s.configure('Stop.TButton',    background=DANGER, foreground='white',
                    font=('Segoe UI', 10, 'bold'), padding=(14, 6))
        s.map('Stop.TButton',          background=[('active', DANGER_DK), ('disabled', '#fca5a5')])
        s.configure('Preset.TButton',  background='#f1f5f9', foreground=HEADER,
                    font=('Segoe UI', 8), padding=(6, 3))
        s.map('Preset.TButton',        background=[('active', '#e2e8f0')])
        s.configure('Link.TButton',    background=BG, foreground=ACCENT,
                    font=('Segoe UI', 9), relief='flat', padding=(4, 2))
        s.map('Link.TButton',          foreground=[('active', ACCENT_DK)])

        # Progress bar
        s.configure('Blue.Horizontal.TProgressbar',
                    background=ACCENT, troughcolor=BORDER, thickness=8)

        # Status bar
        s.configure('Status.TLabel', background='#e8ecf4', foreground=MUTED,
                    font=('Segoe UI', 8), padding=(6, 3), relief='flat')

    # ── variables ─────────────────────────────────────────────────────────────
    def setupVariables(self):
        self.experimentNameVar    = tk.StringVar(value="")
        self.datasetVar           = tk.StringVar(value="cifar10")
        self.approachVar          = tk.StringVar(value="basil")
        self.useBasilVar          = tk.BooleanVar(value=True)
        self.basilMemorySizeVar   = tk.IntVar(value=4)
        self.useChannelNoiseVar   = tk.BooleanVar(value=False)
        self.channelNoiseStartVar = tk.IntVar(value=0)
        self.channelNoiseSigmaVar = tk.DoubleVar(value=0.2)
        self.noiseMitigationVar   = tk.StringVar(value="none")
        self.ebmLambdaVar         = tk.DoubleVar(value=25.0)
        self.momentumVar          = tk.DoubleVar(value=0.9)
        self.useLrDecayVar        = tk.BooleanVar(value=True)
        self.usePlateauLrVar        = tk.BooleanVar(value=False)
        self.plateauPatienceVar     = tk.IntVar(value=10)
        self.plateauFactorVar       = tk.DoubleVar(value=0.5)
        self.plateauMinLrVar        = tk.DoubleVar(value=1e-4)
        self.plateauThresholdVar    = tk.DoubleVar(value=0.002)
        self.attackGaussianVar         = tk.BooleanVar(value=False)
        self.attackGaussianStartVar    = tk.IntVar(value=0)
        self.attackSignFlipVar         = tk.BooleanVar(value=False)
        self.attackSignFlipStartVar    = tk.IntVar(value=0)
        self.attackHiddenVar           = tk.BooleanVar(value=False)
        self.attackHiddenStartVar      = tk.IntVar(value=0)
        self.attackModelPoisonVar      = tk.BooleanVar(value=False)
        self.attackModelPoisonStartVar = tk.IntVar(value=0)
        self.attackScalingVar          = tk.BooleanVar(value=False)
        self.attackScalingStartVar     = tk.IntVar(value=0)
        self.attackAlieVar             = tk.BooleanVar(value=False)
        self.attackAlieStartVar        = tk.IntVar(value=0)
        self.attackIpmVar              = tk.BooleanVar(value=False)
        self.attackIpmStartVar         = tk.IntVar(value=0)
        self.attackNoiseAmpVar         = tk.BooleanVar(value=False)
        self.attackNoiseAmpStartVar    = tk.IntVar(value=0)
        self.attackerIdsVar   = tk.StringVar(value="0,3,5,7")
        self.nNodesVar        = tk.IntVar(value=10)
        self.nRoundsVar       = tk.IntVar(value=100)
        self.localEpochsVar   = tk.IntVar(value=5)
        self.learningRateVar  = tk.DoubleVar(value=0.05)
        self.batchSizeVar     = tk.IntVar(value=512)
        # progress / status (not user-facing inputs)
        self._progressVar     = tk.DoubleVar(value=0.0)

    # ── UI construction ───────────────────────────────────────────────────────
    def createUI(self):
        # ---- notebook (tabs) -----------------------------------------------
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        basicTab    = ttk.Frame(self.notebook)
        advancedTab = ttk.Frame(self.notebook)
        attackTab   = ttk.Frame(self.notebook)
        outputTab   = ttk.Frame(self.notebook)

        self.notebook.add(basicTab,    text="  Basic  ")
        self.notebook.add(advancedTab, text="  Advanced  ")
        self.notebook.add(attackTab,   text="  Attacks  ")
        self.notebook.add(outputTab,   text="  Output  ")

        self.createBasicTab(basicTab)
        self.createAdvancedTab(advancedTab)
        self.createAttackTab(attackTab)
        self.createOutputTab(outputTab)

        # ---- progress bar --------------------------------------------------
        self._createProgressFrame()

        # ---- button bar ----------------------------------------------------
        self.createButtons()

        # ---- status bar ----------------------------------------------------
        self._statusVar = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self._statusVar,
                  style='Status.TLabel', anchor=tk.W
                  ).pack(fill=tk.X, side=tk.BOTTOM)

    def _createProgressFrame(self):
        pf = ttk.Frame(self.root)
        pf.pack(fill=tk.X, padx=8, pady=(4, 0))

        self._progressBar = ttk.Progressbar(pf, variable=self._progressVar,
                                            maximum=100, length=400,
                                            style='Blue.Horizontal.TProgressbar')
        self._progressBar.pack(side=tk.LEFT, padx=(0, 10), pady=3)

        self._roundLabel = ttk.Label(pf, text="Round –/–",
                                     font=('Segoe UI', 9, 'bold'), foreground=ACCENT)
        self._roundLabel.pack(side=tk.LEFT, padx=(0, 16))

        self._etaLabel = ttk.Label(pf, text="", foreground=MUTED, font=('Segoe UI', 8))
        self._etaLabel.pack(side=tk.LEFT)

        self._accLabel = ttk.Label(pf, text="", foreground=SUCCESS,
                                   font=('Segoe UI', 9, 'bold'))
        self._accLabel.pack(side=tk.RIGHT, padx=6)

    # ── Basic tab ─────────────────────────────────────────────────────────────
    def createBasicTab(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- Quick presets -------------------------------------------------
        presetFrame = ttk.LabelFrame(outer, text="Quick Presets", padding=8)
        presetFrame.pack(fill=tk.X, pady=(0, 8))

        desc = ttk.Label(presetFrame,
                         text="Click a preset to auto-fill all settings:",
                         foreground=MUTED, font=('Segoe UI', 8))
        desc.pack(anchor=tk.W, pady=(0, 4))

        btnRow = ttk.Frame(presetFrame)
        btnRow.pack(fill=tk.X)
        for name in PRESETS:
            ttk.Button(btnRow, text=name, style='Preset.TButton',
                       command=lambda n=name: self._applyPreset(n)
                       ).pack(side=tk.LEFT, padx=3)

        # ---- Main settings -------------------------------------------------
        frame = ttk.LabelFrame(outer, text="Experiment Settings", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0

        # Experiment name
        ttk.Label(frame, text="Name:", font=('Segoe UI', 9, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.experimentNameVar, width=55,
                  font=('Segoe UI', 9)).grid(row=row, column=1, columnspan=3,
                                              sticky=tk.W+tk.E, padx=8)
        row += 1
        ttk.Label(frame, text="e.g.  Ring Topology – Noisy + EBM – No Attacks",
                  foreground=MUTED, font=('Segoe UI', 8)).grid(
            row=row, column=1, columnspan=3, sticky=tk.W, padx=8)
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(
            row=row, column=0, columnspan=4, sticky='ew', pady=8)
        row += 1

        # Dataset
        ttk.Label(frame, text="Dataset:", font=('Segoe UI', 9, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        for i, (label, value) in enumerate([("MNIST", "mnist"),
                                             ("CIFAR-10", "cifar10"),
                                             ("Neuromorphic MNIST", "nmnist")]):
            ttk.Radiobutton(frame, text=label, variable=self.datasetVar,
                            value=value).grid(row=row, column=i+1, sticky=tk.W, padx=8)
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(
            row=row, column=0, columnspan=4, sticky='ew', pady=8)
        row += 1

        # Approach
        ttk.Label(frame, text="Approach:", font=('Segoe UI', 9, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(frame, text="BASIL Only",
                        variable=self.approachVar, value="basil",
                        command=self.onApproachChange).grid(row=row, column=1, sticky=tk.W, padx=8)
        ttk.Radiobutton(frame, text="Noisy Channel Only",
                        variable=self.approachVar, value="noisy",
                        command=self.onApproachChange).grid(row=row, column=2, sticky=tk.W, padx=8)
        ttk.Radiobutton(frame, text="Merged (BASIL + Noisy)",
                        variable=self.approachVar, value="merged",
                        command=self.onApproachChange).grid(row=row, column=3, sticky=tk.W, padx=8)
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(
            row=row, column=0, columnspan=4, sticky='ew', pady=8)
        row += 1

        # Training parameters (2-column grid)
        ttk.Label(frame, text="Training Parameters:",
                  font=('Segoe UI', 9, 'bold')).grid(
            row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 4))
        row += 1

        params = [
            ("Nodes:", self.nNodesVar,       "Rounds:", self.nRoundsVar),
            ("Local Epochs:", self.localEpochsVar, "Learning Rate:", self.learningRateVar),
            ("Batch Size:", self.batchSizeVar, None, None),
        ]
        for lbl1, var1, lbl2, var2 in params:
            ttk.Label(frame, text=lbl1).grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(frame, textvariable=var1, width=10).grid(
                row=row, column=1, sticky=tk.W, padx=8)
            if lbl2:
                ttk.Label(frame, text=lbl2).grid(row=row, column=2, sticky=tk.W, pady=2)
                ttk.Entry(frame, textvariable=var2, width=10).grid(
                    row=row, column=3, sticky=tk.W, padx=8)
            row += 1

        # Auto-name button
        ttk.Button(frame, text="Auto-generate Name", style='Link.TButton',
                   command=self._autoGenerateName).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        row += 1

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

    # ── Advanced tab ──────────────────────────────────────────────────────────
    def createAdvancedTab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # BASIL
        basilFrame = ttk.LabelFrame(frame, text="BASIL Configuration", padding=10)
        basilFrame.pack(fill=tk.X, pady=(0, 8))

        ttk.Checkbutton(basilFrame, text="Use BASIL Snapshot Selection",
                        variable=self.useBasilVar).grid(row=0, column=0, columnspan=3,
                                                         sticky=tk.W, pady=5)
        ttk.Label(basilFrame, text="Memory Size (S):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(basilFrame, textvariable=self.basilMemorySizeVar, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=8)
        ttk.Label(basilFrame, text="Number of neighbour models to store",
                  foreground=MUTED, font=('Segoe UI', 8)).grid(
            row=1, column=2, sticky=tk.W)

        # Channel Noise
        noiseFrame = ttk.LabelFrame(frame, text="Channel Noise Configuration", padding=10)
        noiseFrame.pack(fill=tk.X, pady=(0, 8))

        ttk.Checkbutton(noiseFrame, text="Enable Channel Noise",
                        variable=self.useChannelNoiseVar).grid(row=0, column=0,
                                                                columnspan=3, sticky=tk.W, pady=5)

        rows = [
            ("Start Noise at Round:", self.channelNoiseStartVar, "(0 = from beginning)"),
            ("Noise Sigma (σ):",      self.channelNoiseSigmaVar, "Standard deviation of Gaussian noise"),
            ("EBM Lambda:",           self.ebmLambdaVar,         "scale = 1 + λσ²  (default λ=25, σ=0.2 → scale=2.0)"),
            ("Momentum:",             self.momentumVar,          "0.9 recommended for SGD on CIFAR-10"),
        ]
        for i, (lbl, var, hint) in enumerate(rows, start=1):
            ttk.Label(noiseFrame, text=lbl).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Entry(noiseFrame, textvariable=var, width=10).grid(
                row=i, column=1, sticky=tk.W, padx=8)
            ttk.Label(noiseFrame, text=hint, foreground=MUTED,
                      font=('Segoe UI', 8)).grid(row=i, column=2, sticky=tk.W)

        # Mitigation radio buttons
        ttk.Label(noiseFrame, text="Mitigation:").grid(row=5, column=0, sticky=tk.W, pady=5)
        mf = ttk.Frame(noiseFrame)
        mf.grid(row=5, column=1, columnspan=2, sticky=tk.W, padx=8)
        ttk.Radiobutton(mf, text="None", variable=self.noiseMitigationVar,
                        value="none").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(mf, text="EBM (Expectation-Based Model)",
                        variable=self.noiseMitigationVar, value="ebm").pack(side=tk.LEFT)

        # LR decay
        lrFrame = ttk.LabelFrame(frame, text="Learning Rate Schedule", padding=10)
        lrFrame.pack(fill=tk.X)
        ttk.Checkbutton(lrFrame,
                        text="Use LR Decay  (disable for EBM with high noise + momentum)",
                        variable=self.useLrDecayVar).pack(anchor=tk.W)
        ttk.Label(lrFrame, text="lr(t) = lr₀ / (1 + lr₀ · t)   when enabled",
                  foreground=MUTED, font=('Segoe UI', 8)).pack(anchor=tk.W, pady=(2, 0))

        # Plateau LR
        ttk.Separator(lrFrame, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Checkbutton(lrFrame,
                        text="Reduce LR on Plateau  (overrides decay schedule)",
                        variable=self.usePlateauLrVar).pack(anchor=tk.W)
        ttk.Label(lrFrame,
                  text="Halves LR when accuracy does not improve for N rounds. Prevents overshoot at convergence.",
                  foreground=MUTED, font=('Segoe UI', 8), wraplength=480, justify=tk.LEFT
                  ).pack(anchor=tk.W, pady=(2, 6))
        pRow = ttk.Frame(lrFrame)
        pRow.pack(fill=tk.X)
        ttk.Label(pRow, text="Patience (rounds):").pack(side=tk.LEFT)
        ttk.Entry(pRow, textvariable=self.plateauPatienceVar, width=6).pack(side=tk.LEFT, padx=(4, 16))
        ttk.Label(pRow, text="Factor:").pack(side=tk.LEFT)
        ttk.Entry(pRow, textvariable=self.plateauFactorVar, width=6).pack(side=tk.LEFT, padx=(4, 16))
        ttk.Label(pRow, text="Min LR:").pack(side=tk.LEFT)
        ttk.Entry(pRow, textvariable=self.plateauMinLrVar, width=8).pack(side=tk.LEFT, padx=(4, 16))
        ttk.Label(pRow, text="Threshold:").pack(side=tk.LEFT)
        ttk.Entry(pRow, textvariable=self.plateauThresholdVar, width=8).pack(side=tk.LEFT, padx=(4, 0))

    # ── Attack tab ────────────────────────────────────────────────────────────
    def createAttackTab(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _resize(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(inner_id, width=event.width)

        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', _resize)

        frame = ttk.LabelFrame(inner, text="Byzantine Attack Configuration", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Attacker Node IDs:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.attackerIdsVar, width=25).grid(
            row=0, column=1, sticky=tk.W, padx=8)
        ttk.Label(frame, text="comma-separated, e.g. 0,3,5,7",
                  foreground=MUTED, font=('Segoe UI', 8)).grid(row=0, column=2, sticky=tk.W)

        ttk.Separator(frame, orient='horizontal').grid(
            row=1, column=0, columnspan=3, sticky='ew', pady=8)

        attacks = [
            ("Gaussian Noise",             self.attackGaussianVar,    self.attackGaussianStartVar,
             "Attackers send random Gaussian noise instead of gradients"),
            ("Sign-Flip",                  self.attackSignFlipVar,    self.attackSignFlipStartVar,
             "Attackers flip the sign of all gradient values"),
            ("Hidden / Backdoor",          self.attackHiddenVar,      self.attackHiddenStartVar,
             "Behaves normally initially, then injects malicious updates at start round"),
            ("Model Poisoning",            self.attackModelPoisonVar,  self.attackModelPoisonStartVar,
             "Gradient ascent to corrupt the model - hardest to detect"),
            ("Scaling (Model Replacement)", self.attackScalingVar,    self.attackScalingStartVar,
             "Multiplies weights by a large negative factor to dominate aggregation"),
            ("ALIE (A Little Is Enough)",  self.attackAlieVar,        self.attackAlieStartVar,
             "Stealthy update within z_max std of honest distribution"),
            ("IPM (Inner Product Manip.)", self.attackIpmVar,         self.attackIpmStartVar,
             "Negates and scales weights to maximise negative inner product (Fall of Empires)"),
            ("Noise Amplification",        self.attackNoiseAmpVar,    self.attackNoiseAmpStartVar,
             "Amplified Gaussian noise proportional to layer std to evade norm-based defences"),
        ]

        r = 2
        for name, enableVar, startVar, desc in attacks:
            ttk.Checkbutton(frame, text=name, variable=enableVar).grid(
                row=r, column=0, sticky=tk.W, pady=4)
            ttk.Label(frame, text="Start round:").grid(row=r, column=1, sticky=tk.W, padx=8)
            ttk.Entry(frame, textvariable=startVar, width=8).grid(
                row=r, column=2, sticky=tk.W)
            r += 1
            ttk.Label(frame, text=f"  {desc}", foreground=MUTED,
                      font=('Segoe UI', 8)).grid(row=r, column=0, columnspan=3,
                                                  sticky=tk.W, padx=20)
            r += 1
            ttk.Separator(frame, orient='horizontal').grid(
                row=r, column=0, columnspan=3, sticky='ew', pady=6)
            r += 1

    # ── Output tab ────────────────────────────────────────────────────────────
    def createOutputTab(self, parent):
        # Split: log on left, live chart on right
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ---- Left: log text ------------------------------------------------
        leftFrame = ttk.Frame(paned)
        paned.add(leftFrame, weight=3)

        logHeader = ttk.Frame(leftFrame)
        logHeader.pack(fill=tk.X)
        ttk.Label(logHeader, text="Experiment Log",
                  font=('Segoe UI', 9, 'bold'), foreground=HEADER).pack(side=tk.LEFT)
        ttk.Button(logHeader, text="Save Errors", style='Link.TButton',
                   command=self._saveErrorsFromLog).pack(side=tk.RIGHT)
        ttk.Button(logHeader, text="Copy All", style='Link.TButton',
                   command=self._copyAllLog).pack(side=tk.RIGHT)

        self.outputText = scrolledtext.ScrolledText(
            leftFrame, wrap=tk.WORD, width=60, height=30,
            font=('Consolas', 9), bg='#0f172a', fg='#e2e8f0',
            insertbackground='white', selectbackground='#4a5568',
            selectforeground='white', state='normal')
        self.outputText.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # Right-click context menu for copying
        self._logMenu = tk.Menu(self.outputText, tearoff=0)
        self._logMenu.add_command(label="Copy",        command=self._copySelection)
        self._logMenu.add_command(label="Select All",  command=self._selectAllLog)
        self._logMenu.add_separator()
        self._logMenu.add_command(label="Copy All",     command=self._copyAllLog)
        self._logMenu.add_command(label="Save Errors", command=self._saveErrorsFromLog)
        self._logMenu.add_command(label="Clear",       command=self.clearOutput)
        self.outputText.bind('<Button-3>', self._showLogMenu)
        self.outputText.bind('<Control-a>', lambda e: self._selectAllLog())
        self.outputText.bind('<Control-A>', lambda e: self._selectAllLog())
        self.outputText.bind('<Control-c>', lambda e: self._copySelection())
        self.outputText.bind('<Control-C>', lambda e: self._copySelection())

        # Configure log colour tags
        self.outputText.tag_configure('header',  foreground='#93c5fd', font=('Consolas', 9, 'bold'))
        self.outputText.tag_configure('success', foreground='#4ade80')
        self.outputText.tag_configure('error',   foreground='#f87171')
        self.outputText.tag_configure('warn',    foreground='#fbbf24')
        self.outputText.tag_configure('info',    foreground='#67e8f9')
        self.outputText.tag_configure('muted',   foreground='#94a3b8')
        self.outputText.tag_configure('normal',  foreground='#e2e8f0')

        # ---- Right: live chart ---------------------------------------------
        rightFrame = ttk.Frame(paned)
        paned.add(rightFrame, weight=2)

        chartHeader = ttk.Frame(rightFrame)
        chartHeader.pack(fill=tk.X)
        ttk.Label(chartHeader, text="Live Accuracy",
                  font=('Segoe UI', 9, 'bold'), foreground=HEADER).pack(side=tk.LEFT)
        ttk.Button(chartHeader, text="Clear Chart", style='Link.TButton',
                   command=self._clearLiveChart).pack(side=tk.RIGHT)

        self.liveFig = Figure(figsize=(4, 3), dpi=96, facecolor=CHART_BG)
        self.liveAx  = self.liveFig.add_subplot(111)
        self._styleChartAxes()

        self.liveCanvas = FigureCanvasTkAgg(self.liveFig, master=rightFrame)
        self.liveCanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def _styleChartAxes(self):
        ax = self.liveAx
        ax.set_facecolor(CHART_BG)
        ax.set_xlabel('Round', fontsize=7, color=MUTED)
        ax.set_ylabel('Accuracy', fontsize=7, color=MUTED)
        ax.set_title('Waiting for training data…', fontsize=8, color=MUTED)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, color=BORDER)
        ax.tick_params(labelsize=6, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        self.liveFig.tight_layout(pad=1.2)

    # ── Buttons ───────────────────────────────────────────────────────────────
    def createButtons(self):
        bf = ttk.Frame(self.root)
        bf.pack(fill=tk.X, padx=8, pady=6)

        self.runButton = ttk.Button(
            bf, text="▶  Run Experiment", style='Run.TButton',
            command=self.runExperiment)
        self.runButton.pack(side=tk.LEFT, padx=(0, 4))

        self.runAllButton = ttk.Button(
            bf, text="▶▶  Run All Configs", style='RunAll.TButton',
            command=self.runAll)
        self.runAllButton.pack(side=tk.LEFT, padx=4)

        self.stopButton = ttk.Button(
            bf, text="■  Stop", style='Stop.TButton',
            command=self.stopExperiment, state=tk.DISABLED)
        self.stopButton.pack(side=tk.LEFT, padx=4)

        ttk.Separator(bf, orient='vertical').pack(side=tk.LEFT, fill=tk.Y,
                                                   padx=6, pady=2)

        ttk.Button(bf, text="Clear Log",       command=self.clearOutput).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Plot Results",    command=self.plotResults).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Save Config",     command=self.saveConfig).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Load Config",     command=self.loadConfig).pack(side=tk.LEFT, padx=2)

        ttk.Button(bf, text="Reload",          command=self.reloadGui).pack(side=tk.RIGHT, padx=2)
        ttk.Button(bf, text="Exit",            command=self.onClosing).pack(side=tk.RIGHT, padx=2)

        # Keyboard hint labels
        hints = [("Ctrl+R", "Run"), ("Ctrl+S", "Save"), ("Ctrl+L", "Load"), ("Esc", "Stop")]
        for key, label in reversed(hints):
            ttk.Label(bf, text=f"{key}={label}",
                      foreground=MUTED, font=('Segoe UI', 7)).pack(side=tk.RIGHT, padx=2)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    def _setupKeyboardShortcuts(self):
        self.root.bind('<Control-r>', lambda e: self.runExperiment())
        self.root.bind('<Control-s>', lambda e: self.saveConfig())
        self.root.bind('<Control-l>', lambda e: self.loadConfig())
        self.root.bind('<Escape>',    lambda e: self.stopExperiment() if self.isRunning else None)
        self.root.bind('<Control-Shift-R>', lambda e: self.reloadGui())

    # ── Status / progress helpers ─────────────────────────────────────────────
    def _setStatus(self, msg):
        self._statusVar.set(f"  {msg}")

    def _onRoundComplete(self, roundNum, avgAcc, worstAcc, totalRounds):
        """Called from training thread - schedules UI update on main thread."""
        now = time.time()
        if self._lastRoundEndTime is not None:
            dt = now - self._lastRoundEndTime
            if dt > 0:
                self._roundTimes.append(dt)
                # EMA with α=0.25 - weights last ~4 rounds heavily, ignores slow warmup
                if self._emaRoundTime is None:
                    self._emaRoundTime = dt
                else:
                    self._emaRoundTime = 0.25 * dt + 0.75 * self._emaRoundTime
        self._lastRoundEndTime = now
        self.root.after(0, self._updateUIAfterRound, roundNum, avgAcc, worstAcc, totalRounds)

    def _updateUIAfterRound(self, roundNum, avgAcc, worstAcc, totalRounds):
        """UI updates after each round - runs on main thread via root.after()."""
        self._liveAccData.append(avgAcc)
        self._liveWorstData.append(worstAcc)

        # Progress bar (exact at round boundaries)
        if totalRounds > 0:
            self._progressVar.set(roundNum / totalRounds * 100)
            self._roundLabel.config(text=f"Round {roundNum}/{totalRounds}")

        # Latest accuracy
        self._accLabel.config(text=f"Acc  {avgAcc:.1%}")

    def _etaTicker(self):
        """Fires every second to update ETA and progress bar in real-time."""
        if not self.isRunning:
            return

        now = time.time()
        elapsed = now - self._trainStartTime if self._trainStartTime else 0
        roundsDone = len(self._liveAccData)
        totalRounds = self._totalRounds or 1

        if roundsDone > 0 and self._emaRoundTime and elapsed > 0:
            emaTime = self._emaRoundTime

            # How far through the current round we are
            timeSinceLastRound = (now - self._lastRoundEndTime) if self._lastRoundEndTime else 0
            fraction = min(timeSinceLastRound / emaTime, 0.99) if emaTime > 0 else 0
            fractionalDone = roundsDone + fraction

            # Smooth progress bar
            self._progressVar.set(min(fractionalDone / totalRounds * 100, 99.9))
            self._roundLabel.config(text=f"Round {roundsDone}/{totalRounds}")

            # Raw ETA based on EMA round time
            rawEta = max(0.0, (totalRounds - fractionalDone) * emaTime)

            # Smooth ETA: only allow it to decrease by at most 1s per tick (no upward jumps)
            if self._smoothedEta is None:
                self._smoothedEta = rawEta
            else:
                # Allow instant drop if raw is much lower (model sped up), else ease down
                gap = self._smoothedEta - rawEta
                if gap > 5:
                    # Raw dropped significantly - catch up quickly
                    self._smoothedEta = self._smoothedEta - min(gap * 0.3, gap - 1)
                else:
                    # Normal countdown: tick down by 1 per second
                    self._smoothedEta = max(rawEta, self._smoothedEta - 1)

            remaining = self._smoothedEta
            h = int(remaining // 3600)
            m = int((remaining % 3600) // 60)
            s = int(remaining % 60)
            if h > 0:
                self._etaLabel.config(text=f"ETA  {h}h {m:02d}m {s:02d}s")
            else:
                self._etaLabel.config(text=f"ETA  {m}m {s:02d}s")
        elif elapsed > 0:
            # Waiting for first round - show elapsed
            m, s = int(elapsed // 60), int(elapsed % 60)
            self._etaLabel.config(text=f"Elapsed  {m}m {s:02d}s")

        self.root.after(1000, self._etaTicker)

        # Live chart
        self._refreshLiveChart()

        # Status bar
        accStr = f"  avg acc = {self._liveAccData[-1]:.4f}" if self._liveAccData else ""
        self._setStatus(f"Training…  round {roundsDone}/{totalRounds}{accStr}")

    def _refreshLiveChart(self):
        ax = self.liveAx
        ax.clear()
        ax.set_facecolor(CHART_BG)
        ax.grid(True, alpha=0.3, color=BORDER)
        ax.tick_params(labelsize=6, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.set_xlabel('Round', fontsize=7, color=MUTED)
        ax.set_ylabel('Accuracy', fontsize=7, color=MUTED)
        ax.set_ylim(0, 1)

        if self._liveAccData:
            rounds = list(range(len(self._liveAccData)))
            ax.plot(rounds, self._liveAccData, color=ACCENT, linewidth=1.5,
                    marker='o', markersize=2, label='Avg')
            if any(w != a for w, a in zip(self._liveWorstData, self._liveAccData)):
                ax.plot(rounds, self._liveWorstData, color=DANGER, linewidth=1,
                        linestyle='--', markersize=2, label='Worst')
                ax.legend(fontsize=7, framealpha=0.5)
            ax.set_title(f"Live Accuracy  (latest {self._liveAccData[-1]:.1%})",
                         fontsize=8, color=HEADER)
        else:
            ax.set_title("Waiting for training data…", fontsize=8, color=MUTED)

        self.liveFig.tight_layout(pad=1.2)
        self.liveCanvas.draw_idle()

    def _clearLiveChart(self):
        self._liveAccData.clear()
        self._liveWorstData.clear()
        self._refreshLiveChart()

    # ── Preset & auto-name helpers ─────────────────────────────────────────────
    def _applyPreset(self, name):
        cfg = PRESETS[name]
        mapping = {
            'dataset':           self.datasetVar,
            'approach':          self.approachVar,
            'useBasil':          self.useBasilVar,
            'useChannelNoise':   self.useChannelNoiseVar,
            'channelNoiseSigma': self.channelNoiseSigmaVar,
            'noiseMitigation':   self.noiseMitigationVar,
            'ebmLambda':         self.ebmLambdaVar,
            'attackGaussian':    self.attackGaussianVar,
            'attackSignFlip':    self.attackSignFlipVar,
            'attackHidden':      self.attackHiddenVar,
            'attackerIds':       self.attackerIdsVar,
            'nNodes':            self.nNodesVar,
            'nRounds':           self.nRoundsVar,
            'localEpochs':       self.localEpochsVar,
            'learningRate':      self.learningRateVar,
            'batchSize':         self.batchSizeVar,
            'momentum':          self.momentumVar,
            'useLrDecay':        self.useLrDecayVar,
        }
        for key, var in mapping.items():
            if key in cfg:
                var.set(cfg[key])
        self.experimentNameVar.set(name)
        self._setStatus(f"Preset applied: {name}")

    def _autoGenerateName(self):
        dataset  = self.datasetVar.get().upper()
        approach = {'basil': 'BASIL', 'noisy': 'FedAvg', 'merged': 'Merged'}.get(
            self.approachVar.get(), self.approachVar.get())
        noisy  = " + Noise" if self.useChannelNoiseVar.get() else ""
        mitig  = f" + {self.noiseMitigationVar.get().upper()}" if self.useChannelNoiseVar.get() and self.noiseMitigationVar.get() != 'none' else ""
        atks   = []
        if self.attackGaussianVar.get():  atks.append("Gaussian")
        if self.attackSignFlipVar.get():  atks.append("SignFlip")
        if self.attackHiddenVar.get():    atks.append("Hidden")
        if self.attackModelPoisonVar.get(): atks.append("Poison")
        atkStr = f" [{'+'.join(atks)}]" if atks else " [Clean]"
        name = f"{dataset} {approach}{noisy}{mitig}{atkStr}"
        self.experimentNameVar.set(name)

    # ── Log helpers ───────────────────────────────────────────────────────────
    _ERROR_FILE = "error.txt"

    def logMessage(self, message):
        tag = self._pickLogTag(message)
        self.outputText.insert(tk.END, message + "\n", tag)
        self.outputText.see(tk.END)
        self.root.update_idletasks()
        if tag == 'error':
            self._appendToErrorFile(message)

    def _appendToErrorFile(self, message):
        try:
            with open(self._ERROR_FILE, 'a') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except Exception:
            pass

    def _pickLogTag(self, msg):
        m = msg.strip()
        if m.startswith('=') or m.startswith('-'):  return 'header'
        if m.startswith('ERROR') or 'Traceback' in m or 'Exception' in m: return 'error'
        if 'FINAL RESULTS' in m or 'completed successfully' in m: return 'success'
        if m.startswith('[STOP') or m.startswith('[SKIP') or 'STOPPED' in m: return 'warn'
        if m.startswith('[pre-training]') or m.startswith('[round'): return 'info'
        if m.startswith('  ') or m.startswith('Loading') or m.startswith('Creating'): return 'muted'
        return 'normal'

    def clearOutput(self):
        # Text log
        self.outputText.delete(1.0, tk.END)

        # Live chart and its backing data
        self._liveAccData.clear()
        self._liveWorstData.clear()
        self._refreshLiveChart()

        # Progress bar, round/ETA/accuracy labels
        self._progressVar.set(0)
        self._roundLabel.config(text="Round -/-")
        self._etaLabel.config(text="")
        self._accLabel.config(text="")

        # ETA state so stale timing does not bleed into the next run
        self._roundTimes.clear()
        self._emaRoundTime  = None
        self._smoothedEta   = None
        self._lastRoundEndTime = None

        # Terminal - print a form-feed so past output scrolls out of view
        print("\033[2J\033[H", end="", flush=True)

    def _showLogMenu(self, event):
        # Save selection range before popup steals focus/selection
        try:
            self._savedSel = (self.outputText.index(tk.SEL_FIRST),
                              self.outputText.index(tk.SEL_LAST))
        except tk.TclError:
            self._savedSel = None
        try:
            self._logMenu.tk_popup(event.x_root, event.y_root)
        finally:
            self._logMenu.grab_release()

    def _copySelection(self):
        # Try live selection first, fall back to saved selection from right-click
        try:
            text = self.outputText.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            if getattr(self, '_savedSel', None):
                try:
                    text = self.outputText.get(self._savedSel[0], self._savedSel[1])
                except tk.TclError:
                    return
            else:
                return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._setStatus("Copied to clipboard.")

    def _selectAllLog(self):
        self.outputText.tag_add(tk.SEL, '1.0', tk.END)
        self.outputText.mark_set(tk.INSERT, '1.0')
        self.outputText.see(tk.INSERT)

    def _copyAllLog(self):
        text = self.outputText.get('1.0', tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._setStatus("Log copied to clipboard.")

    def _saveErrorsFromLog(self):
        """Extract every error/traceback line visible in the log and write to error.txt."""
        all_lines = self.outputText.get('1.0', tk.END).splitlines()
        error_lines = [l for l in all_lines if self._pickLogTag(l) == 'error']
        if not error_lines:
            self._setStatus("No errors found in current log.")
            return
        try:
            with open(self._ERROR_FILE, 'a') as f:
                f.write(f"\n=== Saved from log at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write('\n'.join(error_lines) + '\n')
            self._setStatus(f"Errors saved to {self._ERROR_FILE}  ({len(error_lines)} lines)")
        except Exception as e:
            self._setStatus(f"Could not save errors: {e}")

    # ── Approach change ───────────────────────────────────────────────────────
    def onApproachChange(self):
        approach = self.approachVar.get()
        if approach == "basil":
            self.useBasilVar.set(True);  self.useChannelNoiseVar.set(False)
        elif approach == "noisy":
            self.useBasilVar.set(False); self.useChannelNoiseVar.set(True)
        elif approach == "merged":
            self.useBasilVar.set(True);  self.useChannelNoiseVar.set(True)

    # ── Run / Stop ────────────────────────────────────────────────────────────
    def runExperiment(self):
        if self.isRunning:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return
        if not self.validateConfig():
            return

        self.runButton.config(state=tk.DISABLED)
        self.runAllButton.config(state=tk.DISABLED)
        self.stopButton.config(state=tk.NORMAL)
        self.isRunning = True
        self._progressVar.set(0)
        self._roundLabel.config(text="Round 0/–")
        self._etaLabel.config(text="")
        self._accLabel.config(text="")
        self.clearOutput()
        self._liveAccData.clear()
        self._liveWorstData.clear()
        self._roundTimes.clear()
        self._emaRoundTime   = None
        self._smoothedEta    = None
        self._refreshLiveChart()

        self.notebook.select(3)  # switch to Output tab
        self._setStatus("Preparing experiment…")

        self.currentThread = threading.Thread(target=self.runExperimentThread)
        self.currentThread.start()

    def stopExperiment(self):
        self.isRunning = False
        self.logMessage("\n[STOP REQUESTED] Stopping experiment…")
        self._setStatus("Stop requested - waiting for current round to finish…")

    def onClosing(self):
        if self.isRunning:
            if not messagebox.askyesno("Experiment Running",
                                       "An experiment is running. Stop it and exit?"):
                return
            self.isRunning = False
            if self.currentThread and self.currentThread.is_alive():
                self.currentThread.join(timeout=5.0)
        self.root.destroy()

    def reloadGui(self):
        """Exit with code 42 so the launcher restarts the GUI with latest code."""
        if self.isRunning:
            if not messagebox.askyesno("Experiment Running",
                                       "An experiment is running. Stop it and reload?"):
                return
            self.isRunning = False
            if self.currentThread and self.currentThread.is_alive():
                self.currentThread.join(timeout=5.0)
        self.root.destroy()
        import sys
        sys.exit(42)

    def runExperimentThread(self):
        try:
            config = self.getConfig()
            name = config.get("experimentName", "Experiment")
            self._trainStartTime = time.time()
            self._lastRoundEndTime = self._trainStartTime
            self._totalRounds = config.get('nRounds', 100)
            self.root.after(1000, self._etaTicker)
            sendNotification("Run Started", f"{name} has started.", priority="default")
            self._executeExperiment(config)
            name = config.get("experimentName", "Experiment")
            sendNotification("Run Complete", f"{name} finished successfully.", priority="high")
        except Exception as e:
            self.logMessage(f"\nERROR: {str(e)}")
            self.logMessage(traceback.format_exc())
            sendNotification("Run FAILED", str(e), priority="urgent")
            self.root.after(0, self._setStatus, f"ERROR: {str(e)[:80]}")
        finally:
            self.root.after(0, self._onRunFinished)

    def _onRunFinished(self):
        self.runButton.config(state=tk.NORMAL)
        self.runAllButton.config(state=tk.NORMAL)
        self.stopButton.config(state=tk.DISABLED)
        self.isRunning = False
        self._progressVar.set(100 if self._liveAccData else 0)
        self._setStatus("Idle  ·  Experiment finished.")

    # ── _getResultPaths / _isAlreadyRun ───────────────────────────────────────
    def _getResultPaths(self, config):
        dataset = config.get('dataset', '')
        attackParts = []
        if config.get('attackGaussian'):    attackParts.append('gaussian')
        if config.get('attackSignFlip'):    attackParts.append('signflip')
        if config.get('attackHidden'):      attackParts.append('hidden')
        if config.get('attackModelPoison'): attackParts.append('model_poison')
        if config.get('attackScaling'):     attackParts.append('scaling')
        if config.get('attackAlie'):        attackParts.append('alie')
        if config.get('attackIpm'):         attackParts.append('ipm')
        if config.get('attackNoiseAmp'):    attackParts.append('noise_amp')
        attackKey = "_".join(attackParts) if attackParts else "none"

        expName = config.get('experimentName', '').strip()
        if not expName:
            return None, None
        safeName = "".join(c if c.isalnum() or c in " _-" else "_" for c in expName).strip().replace(" ", "_")
        approach  = config.get('approach', 'basil')
        resultDir = f"experiments/results/gui/{dataset}/{attackKey}/{approach}"
        return f"{resultDir}/acc_{safeName}.npy", f"{resultDir}/config_{safeName}.json"

    def _isAlreadyRun(self, config):
        accPath, configPath = self._getResultPaths(config)
        if accPath is None or not os.path.exists(accPath):
            return False
        if not os.path.exists(configPath):
            return False
        try:
            with open(configPath, 'r') as f:
                saved = json.load(f)
            for key, val in config.items():
                if saved.get(key) != val:
                    return False
            return True
        except Exception:
            return False

    # ── Run All ───────────────────────────────────────────────────────────────
    def runAll(self):
        if self.isRunning:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return

        configDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        configFiles = sorted([
            os.path.join(configDir, f)
            for f in os.listdir(configDir) if f.endswith('.json')
        ])
        if not configFiles:
            messagebox.showwarning("No Configs", "No config files found in gui/configs/")
            return

        pending, skipped = [], []
        for filepath in configFiles:
            try:
                with open(filepath, 'r') as f:
                    cfg = json.load(f)
                (skipped if self._isAlreadyRun(cfg) else pending).append(filepath)
            except Exception:
                pending.append(filepath)

        nTotal, nSkip, nRun = len(configFiles), len(skipped), len(pending)

        if nRun == 0:
            messagebox.showinfo("All Done",
                                f"All {nTotal} experiments have already been completed.\nNothing to run.")
            return

        if not messagebox.askyesno("Run All Configs",
                                   f"Found {nTotal} configs\n\n"
                                   f"  • To run:       {nRun}\n"
                                   f"  • Already done: {nSkip}\n\n"
                                   f"Proceed?"):
            return

        self.runButton.config(state=tk.DISABLED)
        self.runAllButton.config(state=tk.DISABLED)
        self.stopButton.config(state=tk.NORMAL)
        self.isRunning = True
        self.clearOutput()
        self._liveAccData.clear()
        self._liveWorstData.clear()
        self._refreshLiveChart()
        self.notebook.select(3)

        self.currentThread = threading.Thread(target=self.runAllThread, args=(pending, nSkip))
        self.currentThread.start()

    def runAllThread(self, configFiles, nSkipped=0):
        total, completed = len(configFiles), 0
        try:
            self.logMessage("=" * 80)
            self.logMessage(f"RUN ALL: {total} experiments queued  ({nSkipped} already done, skipped)")
            self.logMessage("=" * 80 + "\n")

            for idx, filepath in enumerate(configFiles, 1):
                if not self.isRunning:
                    self.logMessage("\n[STOPPED] Run All cancelled by user.")
                    break

                with open(filepath, 'r') as f:
                    config = json.load(f)

                if self._isAlreadyRun(config):
                    self.logMessage(f"[{idx}/{total}] SKIP (already done): {os.path.basename(filepath)}")
                    completed += 1
                    continue

                self.logMessage("=" * 80)
                self.logMessage(f"[{idx}/{total}] {os.path.basename(filepath)}")
                self.logMessage("=" * 80)

                self._trainStartTime = time.time()
                self._lastRoundEndTime = self._trainStartTime
                self._totalRounds = config.get('nRounds', 100)
                self._roundTimes.clear()
                self._emaRoundTime   = None
                self._smoothedEta    = None
                self.root.after(1000, self._etaTicker)
                self._liveAccData.clear()
                self._liveWorstData.clear()
                self.root.after(0, self._progressVar.set, 0)
                self.root.after(0, self._roundLabel.config, {'text': f"Round 0/{self._totalRounds}"})

                expName = config.get("experimentName", os.path.basename(filepath))
                sendNotification("Run Started", f"[{idx}/{total}] {expName} has started.", priority="default")
                self._executeExperiment(config)
                completed += 1

                if not self.isRunning:
                    break

                sendNotification("Run Complete", f"[{idx}/{total}] {expName} finished.", priority="high")
                self.logMessage(f"\n[{idx}/{total}] Done.\n")

            self.logMessage("")
            self.logMessage("=" * 80)
            self.logMessage(f"RUN ALL FINISHED: {completed}/{total} experiments completed.")
            self.logMessage("=" * 80)
            sendNotification("Run All Complete", f"{completed}/{total} experiments finished.", priority="high")

        except Exception as e:
            self.logMessage(f"\nRUN ALL ERROR: {str(e)}")
            self.logMessage(traceback.format_exc())
            sendNotification("Run All FAILED", str(e), priority="urgent")
        finally:
            self.root.after(0, self._onRunFinished)

    # ── _executeExperiment ────────────────────────────────────────────────────
    def _executeExperiment(self, config):
        self.logMessage("=" * 80)
        self.logMessage("STARTING EXPERIMENT")
        self.logMessage("=" * 80)
        self.logMessage(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logMessage("")

        approachLabels = {
            'basil':  'BASIL Only (Ring Topology - Paper 001)',
            'noisy':  'Noisy Channel Only (FedAvg - Paper 002)',
            'merged': 'Merged (Ring + EBM)',
        }
        self.logMessage("Configuration:")
        self.logMessage(f"  Experiment: {config.get('experimentName', '(unnamed)')}")
        self.logMessage(f"  Dataset:    {config['dataset']}")
        self.logMessage(f"  Approach:   {approachLabels.get(config['approach'], config['approach'])}")
        self.logMessage(f"  Nodes: {config['nNodes']},  Rounds: {config['nRounds']}")
        if config.get('usePlateauLr', False):
            lrDecayStr = f"plateau (patience={config.get('plateauPatience',10)}, factor={config.get('plateauFactor',0.5)}, min={config.get('plateauMinLr',1e-4)}, threshold={config.get('plateauThreshold',0.002)})"
        elif config.get('useLrDecay', True):
            lrDecayStr = "decay"
        else:
            lrDecayStr = "fixed"
        self.logMessage(f"  LR: {config['learningRate']} ({lrDecayStr}),  Momentum: {config.get('momentum', 0.0)}")
        self.logMessage(f"  Use BASIL: {config['useBasil']},  Use Noise: {config['useChannelNoise']}")
        if config['useChannelNoise']:
            self.logMessage(f"  Noise σ: {config['channelNoiseSigma']},  Mitigation: {config['noiseMitigation']}")
            if config['noiseMitigation'] == 'ebm':
                scale = 1.0 + config['ebmLambda'] * config['channelNoiseSigma'] ** 2
                self.logMessage(f"  EBM λ: {config['ebmLambda']} (scale={scale:.2f})")
        self.logMessage("")

        deviceInfo = setupGpu()
        self.logMessage("=" * 40)
        if deviceInfo["device"] == "CUDA":
            self.logMessage(f"  DEVICE: CUDA (GPU)")
            for name in deviceInfo["gpus"]:
                self.logMessage(f"  GPU: {name}")
        else:
            self.logMessage("  DEVICE: CPU  (no GPU available)")
        self.logMessage("=" * 40 + "\n")

        self.logMessage(f"Loading {config['dataset'].upper()} dataset…")
        train, test = self.loadDataset(config['dataset'])
        trainLoaders, testLoader = self.makeLoaders(
            config['dataset'], train, test, config['batchSize'], config['nNodes'])
        self.logMessage(f"  Training samples: {len(train)}")
        self.logMessage(f"  Test samples:     {len(test)}\n")

        self.logMessage(f"Creating {config['nNodes']} nodes…")
        nodes = self.createNodes(config, trainLoaders)
        self.logMessage("")

        attackTypes, attackerIds = self.prepareAttacks(config)
        self.logMessage(f"Attack configuration:")
        self.logMessage(f"  Attackers:    {attackerIds if attackerIds else 'None'}")
        self.logMessage(f"  Attack types: {attackTypes}\n")

        if config['approach'] == 'noisy':
            self.logMessage(f"Starting FedAvg training for {config['nRounds']} rounds…")
            self.logMessage("Mode: FedAvg (Parallel + Averaging) - Paper 002")
            self.logMessage("-" * 80)
            avgAccHist, worstAccHist = fedAvgTrainingWithNoise(
                nodes=nodes,
                rounds=config['nRounds'],
                testLoader=testLoader,
                attackTypes=attackTypes,
                attackerIds=attackerIds,
                hiddenStartRound=config['attackHiddenStart'] if config['attackHidden'] else 999,
                sigma=config['channelNoiseSigma'] if config['useChannelNoise'] else 0.0,
                noiseModel=self.getNoiseModel(config),
                channelNoiseStart=config['channelNoiseStart'] if config['useChannelNoise'] else 0,
                lr0=config['learningRate'],
                localEpochs=config['localEpochs'],
                stepsPerEpoch=config.get('stepsPerEpoch', 5),
                stopCallback=lambda: not self.isRunning,
                useLrDecay=config.get('useLrDecay', True),
                usePlateauLr=config.get('usePlateauLr', False),
                plateauPatience=config.get('plateauPatience', 10),
                plateauFactor=config.get('plateauFactor', 0.5),
                plateauMinLr=config.get('plateauMinLr', 1e-4),
                plateauThreshold=config.get('plateauThreshold', 0.002),
                roundCallback=self._onRoundComplete,
            )
        else:
            self.logMessage(f"Starting Ring training for {config['nRounds']} rounds…")
            self.logMessage("Mode: Ring Topology (Sequential) - Paper 001")
            self.logMessage("-" * 80)
            avgAccHist, worstAccHist = basilRingTrainingWithAttack(
                nodes=nodes,
                rounds=config['nRounds'],
                testLoader=testLoader,
                attackTypes=attackTypes,
                attackerIds=attackerIds,
                hiddenStartRound=config['attackHiddenStart'] if config['attackHidden'] else 999,
                sigma=config['channelNoiseSigma'] if config['useChannelNoise'] else 0.0,
                noiseModel=self.getNoiseModel(config),
                channelNoiseStart=config['channelNoiseStart'] if config['useChannelNoise'] else 0,
                lr0=config['learningRate'],
                stepsPerEpoch=config.get('stepsPerEpoch', 5),
                useSnapshots=config['useBasil'],
                useSequential=True,
                stopCallback=lambda: not self.isRunning,
                useLrDecay=config.get('useLrDecay', True),
                usePlateauLr=config.get('usePlateauLr', False),
                plateauPatience=config.get('plateauPatience', 10),
                plateauFactor=config.get('plateauFactor', 0.5),
                plateauMinLr=config.get('plateauMinLr', 1e-4),
                plateauThreshold=config.get('plateauThreshold', 0.002),
                roundCallback=self._onRoundComplete,
            )

        if not self.isRunning:
            self.logMessage("\n" + "=" * 80)
            self.logMessage("EXPERIMENT STOPPED BY USER - no results saved")
            self.logMessage("=" * 80)
            return

        self.logMessage("")
        self.logMessage("=" * 80)
        finalAvg, finalWorst, allAccs = evaluateAll(nodes, testLoader)
        self.logMessage("FINAL RESULTS:")
        self.logMessage(f"  Average Accuracy:      {finalAvg:.4f}  ({finalAvg:.1%})")
        self.logMessage(f"  Worst Node Accuracy:   {finalWorst:.4f}  ({finalWorst:.1%})")
        self.logMessage(f"  Per-node: {[f'{a:.3f}' for a in allAccs]}")
        self.logMessage("=" * 80)

        self.saveResults(config, avgAccHist, worstAccHist, finalAvg, finalWorst)
        self.logMessage("\nExperiment completed successfully!")
        self.root.after(0, self._setStatus,
                        f"Done  ·  Avg acc {finalAvg:.1%}  ·  Worst {finalWorst:.1%}")

    # ── Validation / Config ───────────────────────────────────────────────────
    def validateConfig(self):
        try:
            if self.nRoundsVar.get() <= 0:
                messagebox.showerror("Error", "Number of rounds must be positive"); return False
            if self.nNodesVar.get() <= 0:
                messagebox.showerror("Error", "Number of nodes must be positive"); return False
            cns = self.channelNoiseStartVar.get()
            if cns < 0 or cns >= self.nRoundsVar.get():
                messagebox.showerror("Error", f"Channel noise start must be 0 … {self.nRoundsVar.get()-1}"); return False
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Invalid configuration: {e}"); return False

    def getConfig(self):
        return {
            'experimentName':        self.experimentNameVar.get(),
            'dataset':               self.datasetVar.get(),
            'approach':              self.approachVar.get(),
            'useBasil':              self.useBasilVar.get(),
            'basilMemorySize':       self.basilMemorySizeVar.get(),
            'useChannelNoise':       self.useChannelNoiseVar.get(),
            'channelNoiseStart':     self.channelNoiseStartVar.get(),
            'channelNoiseSigma':     self.channelNoiseSigmaVar.get(),
            'noiseMitigation':       self.noiseMitigationVar.get(),
            'ebmLambda':             self.ebmLambdaVar.get(),
            'momentum':              self.momentumVar.get(),
            'attackGaussian':        self.attackGaussianVar.get(),
            'attackGaussianStart':   self.attackGaussianStartVar.get(),
            'attackSignFlip':        self.attackSignFlipVar.get(),
            'attackSignFlipStart':   self.attackSignFlipStartVar.get(),
            'attackHidden':          self.attackHiddenVar.get(),
            'attackHiddenStart':     self.attackHiddenStartVar.get(),
            'attackModelPoison':     self.attackModelPoisonVar.get(),
            'attackModelPoisonStart': self.attackModelPoisonStartVar.get(),
            'attackScaling':         self.attackScalingVar.get(),
            'attackScalingStart':    self.attackScalingStartVar.get(),
            'attackAlie':            self.attackAlieVar.get(),
            'attackAlieStart':       self.attackAlieStartVar.get(),
            'attackIpm':             self.attackIpmVar.get(),
            'attackIpmStart':        self.attackIpmStartVar.get(),
            'attackNoiseAmp':        self.attackNoiseAmpVar.get(),
            'attackNoiseAmpStart':   self.attackNoiseAmpStartVar.get(),
            'attackerIds':           self.attackerIdsVar.get(),
            'nNodes':                self.nNodesVar.get(),
            'nRounds':               self.nRoundsVar.get(),
            'localEpochs':           self.localEpochsVar.get(),
            'learningRate':          self.learningRateVar.get(),
            'batchSize':             self.batchSizeVar.get(),
            'useLrDecay':            self.useLrDecayVar.get(),
            'usePlateauLr':          self.usePlateauLrVar.get(),
            'plateauPatience':       self.plateauPatienceVar.get(),
            'plateauFactor':         self.plateauFactorVar.get(),
            'plateauMinLr':          self.plateauMinLrVar.get(),
            'plateauThreshold':      self.plateauThresholdVar.get(),
        }

    # ── Data / Node helpers ───────────────────────────────────────────────────
    def loadDataset(self, dataset):
        if dataset == "mnist":    return loadMnist()
        if dataset == "cifar10":  return loadCifar10()
        if dataset == "nmnist":   return loadNMnist()
        raise ValueError(f"Unknown dataset: {dataset}")

    def makeLoaders(self, dataset, train, test, batchSize, nClients):
        if dataset == "mnist":   return makeMnistLoaders(train, test, batchSize=batchSize, nClients=nClients)
        if dataset == "cifar10": return makeCifarLoaders(train, test, batchSize=batchSize, nClients=nClients)
        if dataset == "nmnist":  return makeNMnistLoaders(train, test, batchSize=batchSize, nClients=nClients)
        raise ValueError(f"Unknown dataset: {dataset}")

    def createNodes(self, config, trainLoaders):
        modelClass = self.getModelClass(config['dataset'])
        nodes = []
        for i in range(config['nNodes']):
            nodeCfg = {
                "nodeId": i, "model": modelClass(),
                "dataLoader": trainLoaders[i],
                "S": config['basilMemorySize'],
                "noiseModel": self.getNoiseModel(config),
                "sigma": config['channelNoiseSigma'] if config['useChannelNoise'] else 0.0,
                "lr0": config['learningRate'],
                "localEpochs": config['localEpochs'],
                "momentum": config.get('momentum', 0.0),
            }
            if config['noiseMitigation'] == 'ebm':
                nodeCfg['ebmLambda'] = config['ebmLambda']
            nodes.append(BasilNode(**nodeCfg))
        return nodes

    def getModelClass(self, dataset):
        if dataset == "mnist":   return MNISTModel
        if dataset == "cifar10": return CIFARModel
        if dataset == "nmnist":  return NMNISTModel
        raise ValueError(f"Unknown dataset: {dataset}")

    def getNoiseModel(self, config):
        if not config['useChannelNoise']:         return "none"
        if config['noiseMitigation'] == 'ebm':    return "ebm"
        return "noisy"

    def prepareAttacks(self, config):
        try:
            attackerIds = [int(x.strip()) for x in config['attackerIds'].split(',') if x.strip()]
        except Exception:
            attackerIds = []
        if not attackerIds:
            return ["none"], []

        attacks = []
        if config['attackGaussian']:   attacks.append(('gaussian',    config['attackGaussianStart']))
        if config['attackSignFlip']:   attacks.append(('signFlip',    config['attackSignFlipStart']))
        if config['attackHidden']:     attacks.append(('hidden',      config['attackHiddenStart']))
        if config.get('attackModelPoison'): attacks.append(('model_poison', config['attackModelPoisonStart']))
        if config.get('attackScaling'):     attacks.append(('scaling',      config['attackScalingStart']))
        if config.get('attackAlie'):        attacks.append(('alie',         config['attackAlieStart']))
        if config.get('attackIpm'):         attacks.append(('ipm',          config['attackIpmStart']))
        if config.get('attackNoiseAmp'):    attacks.append(('noise_amp',    config['attackNoiseAmpStart']))
        if not attacks:
            return ["none"], []
        attacks.sort(key=lambda x: x[1])
        return [a[0] for a in attacks], attackerIds

    # ── Save / Load results & config ──────────────────────────────────────────
    def saveResults(self, config, avgAccHist, worstAccHist, finalAvg, finalWorst):
        dataset = config['dataset']
        attackParts = []
        for k, s in [('attackGaussian','gaussian'),('attackSignFlip','signflip'),
                     ('attackHidden','hidden'),('attackModelPoison','model_poison'),
                     ('attackScaling','scaling'),('attackAlie','alie'),
                     ('attackIpm','ipm'),('attackNoiseAmp','noise_amp')]:
            if config.get(k):
                attackParts.append(s)
        attackKey = "_".join(attackParts) if attackParts else "none"

        expName = config.get('experimentName', '').strip()
        if expName:
            safeName = "".join(c if c.isalnum() or c in " _-" else "_" for c in expName).strip().replace(" ", "_")
        else:
            safeName = f"{config['approach']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        approach  = config.get('approach', 'basil')
        resultDir = f"experiments/results/gui/{dataset}/{attackKey}/{approach}"
        os.makedirs(resultDir, exist_ok=True)

        avgPath    = f"{resultDir}/acc_{safeName}.npy"
        configPath = f"{resultDir}/config_{safeName}.json"
        np.save(avgPath, np.array(avgAccHist))
        with open(configPath, 'w') as f:
            json.dump(config, f, indent=2)

        self.logMessage(f"\nResults saved:")
        self.logMessage(f"  {avgPath}")
        self.logMessage(f"  {configPath}")

    def saveConfig(self):
        config  = self.getConfig()
        expName = config.get('experimentName', '').strip()
        default = expName or f"config_{config['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        userInput = simpledialog.askstring(
            "Save Configuration", "Enter a name for this configuration:",
            initialvalue=default, parent=self.root)
        if not userInput:
            return
        if not userInput.endswith('.json'):
            userInput += '.json'
        filepath = os.path.join("gui", "configs", userInput)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        messagebox.showinfo("Saved", f"Configuration saved to:\n{filepath}")
        self._setStatus(f"Config saved: {filepath}")

    def loadConfig(self):
        filepath = filedialog.askopenfilename(
            title="Load Configuration", initialdir="gui/configs",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not filepath:
            return
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            self.experimentNameVar.set(config.get('experimentName', ''))
            self.datasetVar.set(config.get('dataset', 'mnist'))
            self.approachVar.set(config.get('approach', 'basil'))
            self.useBasilVar.set(config.get('useBasil', True))
            self.basilMemorySizeVar.set(config.get('basilMemorySize', 10))
            self.useChannelNoiseVar.set(config.get('useChannelNoise', False))
            self.channelNoiseStartVar.set(config.get('channelNoiseStart', 0))
            self.channelNoiseSigmaVar.set(config.get('channelNoiseSigma', 0.1))
            self.noiseMitigationVar.set(config.get('noiseMitigation', 'none'))
            self.ebmLambdaVar.set(config.get('ebmLambda', 25.0))
            self.momentumVar.set(config.get('momentum', 0.9))
            self.attackGaussianVar.set(config.get('attackGaussian', False))
            self.attackGaussianStartVar.set(config.get('attackGaussianStart', 0))
            self.attackSignFlipVar.set(config.get('attackSignFlip', False))
            self.attackSignFlipStartVar.set(config.get('attackSignFlipStart', 0))
            self.attackHiddenVar.set(config.get('attackHidden', False))
            self.attackHiddenStartVar.set(config.get('attackHiddenStart', 0))
            self.attackModelPoisonVar.set(config.get('attackModelPoison', False))
            self.attackModelPoisonStartVar.set(config.get('attackModelPoisonStart', 0))
            self.attackScalingVar.set(config.get('attackScaling', False))
            self.attackScalingStartVar.set(config.get('attackScalingStart', 0))
            self.attackAlieVar.set(config.get('attackAlie', False))
            self.attackAlieStartVar.set(config.get('attackAlieStart', 0))
            self.attackIpmVar.set(config.get('attackIpm', False))
            self.attackIpmStartVar.set(config.get('attackIpmStart', 0))
            self.attackNoiseAmpVar.set(config.get('attackNoiseAmp', False))
            self.attackNoiseAmpStartVar.set(config.get('attackNoiseAmpStart', 0))
            self.attackerIdsVar.set(config.get('attackerIds', '0,5'))
            self.nNodesVar.set(config.get('nNodes', 10))
            self.nRoundsVar.set(config.get('nRounds', 100))
            self.localEpochsVar.set(config.get('localEpochs', 1))
            self.learningRateVar.set(config.get('learningRate', 0.05))
            self.batchSizeVar.set(config.get('batchSize', 32))
            self.useLrDecayVar.set(config.get('useLrDecay', True))
            self.usePlateauLrVar.set(config.get('usePlateauLr', False))
            self.plateauPatienceVar.set(config.get('plateauPatience', 10))
            self.plateauFactorVar.set(config.get('plateauFactor', 0.5))
            self.plateauMinLrVar.set(config.get('plateauMinLr', 1e-4))
            self.plateauThresholdVar.set(config.get('plateauThreshold', 0.002))
            messagebox.showinfo("Loaded", "Configuration loaded successfully!")
            self._setStatus(f"Config loaded: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")

    # ── Plot results ──────────────────────────────────────────────────────────
    def plotResults(self):
        try:
            from plotGui import (discoverDatasets, discoverAttackTypes, discoverApproaches,
                                 discoverExperiments, plotDatasetExperiments,
                                 plotDatasetGrid, plotFinalAccuracyBar)

            self.logMessage("\n" + "=" * 60)
            self.logMessage("GENERATING PLOTS")
            self.logMessage("=" * 60)

            datasets = discoverDatasets()
            if not datasets:
                self.logMessage("No experiment results found. Run some experiments first!")
                messagebox.showinfo("No Results", "No experiment results found.\nRun some experiments first!")
                return

            self.logMessage(f"Found datasets: {datasets}")
            for dataset in datasets:
                attackTypes = discoverAttackTypes(dataset)
                if not attackTypes:
                    continue
                self.logMessage(f"\nDataset: {dataset.upper()} - attacks: {attackTypes}")
                for attackKey in attackTypes:
                    approaches = discoverApproaches(dataset, attackKey)
                    if not approaches:
                        continue
                    for approach in approaches:
                        experiments = discoverExperiments(dataset, attackKey, approach)
                        if not experiments:
                            continue
                        self.logMessage(f"  {attackKey} | {approach}  ({len(experiments)} experiments)")
                        plotDatasetExperiments(dataset, attackKey, approach, experiments)
                        if len(experiments) > 1:
                            plotDatasetGrid(dataset, attackKey, approach, experiments)
                        plotFinalAccuracyBar(dataset, attackKey, approach, experiments)

            self.logMessage("\nAll plots saved to: plots/images/gui/")
            self.logMessage("=" * 60)
            messagebox.showinfo("Success", "Plots saved to plots/images/gui/")
            self._setStatus("Plots saved to plots/images/gui/")
        except Exception as e:
            self.logMessage(f"\nERROR generating plots: {e}")
            self.logMessage(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to generate plots:\n{e}")


def main():
    root = tk.Tk()
    app = ExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
