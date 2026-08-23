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
import signal
import sys
import time
import re
import subprocess
import traceback
import gc
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog, filedialog
import threading
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots'))

from basil_core.data.mnist import loadMnist, makeLoaders as makeMnistLoaders
from basil_core.data.cifar import loadCifar10, makeLoaders as makeCifarLoaders
from basil_core.data.nMnist import loadNMnist, makeLoaders as makeNMnistLoaders
from basil_core.models import MNISTModel, CIFARModel, NMNISTModel
from basil_core.basil import BasilNode, basilRingTrainingWithAttack, fedAvgTrainingWithNoise
from basil_core.cart import CARTNode, cartRingTraining
from basil_core.campaign_engine import run_campaign_three
from basil_core.trainer import evaluateAll, getParams, setParams
from gui.campaign3 import (
    PLOT_ROOT as CAMPAIGN3_PLOT_ROOT,
    PRESET_LABELS as CAMPAIGN3_PRESET_LABELS,
    RESULT_ROOT as CAMPAIGN3_RESULT_ROOT,
    build_preset as buildCampaign3Preset,
    config_hash as campaign3ConfigHash,
    freeze_calibration_if_ready,
    is_completed as isCampaign3Completed,
    result_paths as campaign3ResultPaths,
    write_json_atomic,
    write_npz_atomic,
)
from gui.campaign4 import (
    CAMPAIGN_STATE as CAMPAIGN4_STATE,
    CONFIG_ROOT as CAMPAIGN4_CONFIG_ROOT,
    PERFORMANCE_PROFILE as CAMPAIGN4_PERFORMANCE_PROFILE,
    PLOT_ROOT as CAMPAIGN4_PLOT_ROOT,
    PRESET_LABELS as CAMPAIGN4_PRESET_LABELS,
    RESULT_BASE as CAMPAIGN4_RESULT_BASE,
    RESULT_ROOT as CAMPAIGN4_RESULT_ROOT,
    WORKER_ROOT as CAMPAIGN4_WORKER_ROOT,
    apply_performance_profile as applyCampaign4PerformanceProfile,
    build_preset as buildCampaign4Preset,
    diagnostic_completion as campaign4DiagnosticCompletion,
    freeze_campaign_state as freezeCampaign4State,
    is_completed as isCampaign4Completed,
    is_confirmation_frozen as isCampaign4ConfirmationFrozen,
    load_campaign_state as loadCampaign4State,
    result_paths as campaign4ResultPaths,
)
from gui.campaign_workers import (
    CampaignWorkerPool,
    WORK_DIR as CAMPAIGN3_WORKER_ROOT,
    find_orphan_workers,
    terminate_orphan_workers,
)
from gui.campaign4_execution import (
    memory_limit_for_config as campaign4MemoryLimit,
    select_next_config as selectCampaign4Config,
    settings_from_profile as campaign4ExecutionSettings,
)
from gui.network_view import CampaignNetworkView
from gui.runtime_estimator import (
    RuntimeEstimator,
    format_duration,
    load_worker_profile,
)
from scripts.common import (
    cleanupTensorflowMemory,
    sendNotification,
    setExperimentSeed,
    setupGpu,
)
from plotGui import discoverDatasets, discoverExperiments, getColors, getMarkers

# ─── Dark colour palette ─────────────────────────────────────────────────────
BG             = '#080d14'
PANEL_BG       = '#101823'
SURFACE        = '#172231'
SURFACE_ACTIVE = '#223044'
INPUT_BG       = '#0c141f'
BORDER         = '#2b3a4d'
CHART_BG       = '#0d1520'
HEADER         = '#e6edf7'
MUTED          = '#91a0b5'
ACCENT         = '#4f8cff'
ACCENT_DK      = '#3975df'
DANGER         = '#ef5b67'
DANGER_DK      = '#c94753'
SUCCESS        = '#3ecf8e'
INFO_CLR       = '#54b8e8'
WARN_CLR       = '#f0ad4e'
QUEUE_CLR      = '#16a3a5'
QUEUE_DK       = '#118486'
RUN_ALL_CLR    = '#d88932'
RUN_ALL_DK     = '#b56d23'
DISABLED_BG    = '#293546'
DISABLED_FG    = '#718096'
SELECTION_BG   = '#315f9f'
LOG_BG         = '#070c12'

# ─── Quick presets ───────────────────────────────────────────────────────────
PRESETS = {
    "Clean Baseline": {
        'dataset': 'cifar10', 'approach': 'merged', 'useBasil': False,
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
        'attackerIds': '1,4,6,8', 'nNodes': 10, 'nRounds': 100,
        'localEpochs': 5, 'learningRate': 0.05, 'batchSize': 512,
        'momentum': 0.9, 'useLrDecay': True,
    },
    "Merged Noisy + EBM": {
        'dataset': 'cifar10', 'approach': 'merged', 'useBasil': False,
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
        self.root.geometry("1280x900")
        self.root.minsize(1000, 720)
        self.root.configure(bg=BG)

        # Running state
        self.isRunning = False
        self.currentThread = None
        self._activeExperimentObjects = {}

        # Config queue (list of config dicts; same config may appear multiple times)
        self.configQueue = []
        self._queueLock = threading.Lock()
        self._queueStatePath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "queue_state.json"
        )
        # Sentinel the launcher's file-watcher checks so a code change never
        # restarts the GUI (and orphans GPU workers) while a queue is running.
        self._queueActivePath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".queue_active.json"
        )
        self._queuePreset1Path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "presets", "preset1_queue.json"
        )
        self.queueButton = None   # set when toolbar is built
        self._runtimeEstimator = RuntimeEstimator(
            (
                Path("experiments") / "results3" / "r2",
                CAMPAIGN4_RESULT_BASE,
            )
        )
        self._workerProfile = load_worker_profile()
        self._workerProfile4 = load_worker_profile(CAMPAIGN4_PERFORMANCE_PROFILE)
        if self._workerProfile4.get("status") == "not_benchmarked":
            self._workerProfile4["gpuMemoryLimitMb"] = 7600
        self._runtimeEstimator.set_execution_profile(self._workerProfile4)
        self._campaignWorkerPool = None
        self._campaignWorkerFailures = {}
        self._benchmarkProcess = None
        self._queueLastEstimateRefresh = 0.0
        self._queueBatchNeedsPlot = False
        self._campaignPlotCondition = threading.Condition()
        self._campaignPlotPendingSplits = set()
        self._campaignPlotThread = None

        # Live-chart / progress tracking
        self._liveAccData    = []
        self._liveWorstData  = []
        self._campaignLiveData = {}
        self._trainStartTime = None
        self._lastRoundEndTime = None
        self._totalRounds    = 0
        self._roundTimes     = []   # per-round durations for EMA
        self._emaRoundTime   = None # exponential moving average of round time
        self._smoothedEta    = None # smoothed ETA to avoid jarring jumps

        self._setupStyle()
        self.setupVariables()
        self.createUI()
        self._loadQueueState()
        self._setupKeyboardShortcuts()

        # A fresh GUI has no running queue; drop any sentinel a killed
        # session left behind, then surface workers it may have leaked.
        self._clearQueueActiveSentinel()
        startupOrphans = self._findOrphanWorkers()
        if startupOrphans:
            names = ", ".join(
                f"{entry.get('runId', '?')} (pid {entry.get('pid', '?')})"
                for entry in startupOrphans
            )
            self.logMessage(
                f"WARNING: {len(startupOrphans)} campaign worker(s) from a "
                f"previous session are still running: {names}. They hold GPU "
                "memory; starting the queue will offer to terminate them."
            )

        self.root.protocol("WM_DELETE_WINDOW", self.onClosing)
        self._setStatus("Ready  ·  Ctrl+R = Run   Ctrl+S = Save   Ctrl+L = Load   Esc = Stop   Ctrl+Shift+R = Reload")

    # ── style ─────────────────────────────────────────────────────────────────
    def _setupStyle(self):
        s = ttk.Style()
        s.theme_use('clam')

        self.root.option_add('*Menu.background', PANEL_BG)
        self.root.option_add('*Menu.foreground', HEADER)
        self.root.option_add('*Menu.activeBackground', SURFACE_ACTIVE)
        self.root.option_add('*Menu.activeForeground', HEADER)
        self.root.option_add('*Menu.selectColor', ACCENT)
        self.root.option_add('*Menu.borderWidth', 1)

        s.configure('.',               background=BG, foreground=HEADER,
                    font=('Segoe UI', 9))
        s.configure('TFrame',          background=BG)
        s.configure('TLabelframe',     background=BG, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER)
        s.configure('TLabelframe.Label', font=('Segoe UI', 9, 'bold'), foreground=HEADER, background=BG)
        s.configure('TLabel',          background=BG,     foreground=HEADER)
        s.configure('TEntry',          fieldbackground=INPUT_BG, foreground=HEADER,
                    insertcolor=HEADER, bordercolor=BORDER, lightcolor=BORDER,
                    darkcolor=BORDER, padding=(4, 3))
        s.map('TEntry',
              fieldbackground=[('disabled', DISABLED_BG), ('readonly', INPUT_BG)],
              foreground=[('disabled', DISABLED_FG)],
              bordercolor=[('focus', ACCENT)])
        s.configure('TSpinbox',        fieldbackground=INPUT_BG, foreground=HEADER,
                    arrowcolor=MUTED, insertcolor=HEADER, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, padding=(4, 3))
        s.map('TSpinbox',
              fieldbackground=[('disabled', DISABLED_BG), ('readonly', INPUT_BG)],
              foreground=[('disabled', DISABLED_FG)],
              bordercolor=[('focus', ACCENT)],
              arrowcolor=[('active', HEADER)])
        s.configure('TCombobox',       fieldbackground=INPUT_BG, foreground=HEADER,
                    background=SURFACE, arrowcolor=MUTED, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, padding=(4, 3))
        s.map('TCombobox',
              fieldbackground=[('readonly', INPUT_BG), ('disabled', DISABLED_BG)],
              foreground=[('readonly', HEADER), ('disabled', DISABLED_FG)],
              selectbackground=[('readonly', INPUT_BG)],
              selectforeground=[('readonly', HEADER)],
              arrowcolor=[('active', HEADER)])
        s.configure('TRadiobutton',    background=BG,     foreground=HEADER)
        s.map('TRadiobutton',
              background=[('active', BG)],
              foreground=[('disabled', DISABLED_FG), ('active', HEADER)],
              indicatorcolor=[('selected', ACCENT), ('!selected', INPUT_BG)])
        s.configure('TCheckbutton',    background=BG,     foreground=HEADER)
        s.map('TCheckbutton',
              background=[('active', BG)],
              foreground=[('disabled', DISABLED_FG), ('active', HEADER)],
              indicatorcolor=[('selected', ACCENT), ('!selected', INPUT_BG)])
        s.configure('TSeparator',      background=BORDER)
        s.configure('TNotebook',       background=BG,     tabmargins=[2, 2, 2, 0])
        s.configure('TNotebook.Tab',   background=SURFACE, foreground=MUTED,
                    padding=[16, 8], font=('Segoe UI', 9, 'bold'))
        s.map('TNotebook.Tab',
              background=[('selected', PANEL_BG)],
              foreground=[('selected', ACCENT), ('active', HEADER)])
        s.configure('TPanedwindow',    background=BORDER, sashwidth=4)

        # Buttons
        s.configure('TButton',         background=SURFACE, foreground=HEADER,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    padding=(10, 6), font=('Segoe UI', 9))
        s.map('TButton',
              background=[('disabled', DISABLED_BG), ('active', SURFACE_ACTIVE)],
              foreground=[('disabled', DISABLED_FG), ('active', HEADER)],
              bordercolor=[('focus', ACCENT)])
        s.configure('Run.TButton',     background=ACCENT, foreground='white',
                    font=('Segoe UI', 10, 'bold'), padding=(16, 8), borderwidth=0)
        s.map('Run.TButton',           background=[('active', ACCENT_DK), ('disabled', DISABLED_BG)],
                    foreground=[('disabled', DISABLED_FG)])
        s.configure('RunAll.TButton',  background=RUN_ALL_CLR, foreground='#101820',
                    font=('Segoe UI', 10, 'bold'), padding=(16, 8), borderwidth=0)
        s.map('RunAll.TButton',        background=[('active', RUN_ALL_DK), ('disabled', DISABLED_BG)],
                    foreground=[('disabled', DISABLED_FG)])
        s.configure('Stop.TButton',    background=DANGER, foreground='white',
                    font=('Segoe UI', 10, 'bold'), padding=(16, 8), borderwidth=0)
        s.map('Stop.TButton',          background=[('active', DANGER_DK), ('disabled', DISABLED_BG)],
                    foreground=[('disabled', DISABLED_FG)])
        s.configure('Queue.TButton',   background=QUEUE_CLR, foreground='#071416',
                    font=('Segoe UI', 10, 'bold'), padding=(16, 8), borderwidth=0)
        s.map('Queue.TButton',         background=[('active', QUEUE_DK), ('disabled', DISABLED_BG)],
                    foreground=[('disabled', DISABLED_FG)])
        s.configure('Preset.TButton',  background=SURFACE, foreground=HEADER,
                    font=('Segoe UI', 8), padding=(8, 5))
        s.map('Preset.TButton',        background=[('active', SURFACE_ACTIVE)])
        s.configure('Link.TButton',    background=BG, foreground=ACCENT,
                    font=('Segoe UI', 9), relief='flat', padding=(4, 2))
        s.map('Link.TButton',
              background=[('active', BG)],
              foreground=[('active', INFO_CLR)])
        s.configure('More.TMenubutton', background=SURFACE, foreground=HEADER,
                    font=('Segoe UI', 9, 'bold'), padding=(12, 7))
        s.map('More.TMenubutton',       background=[('active', SURFACE_ACTIVE)])

        # Progress bar
        s.configure('Blue.Horizontal.TProgressbar',
                    background=ACCENT, troughcolor=INPUT_BG,
                    bordercolor=BORDER, lightcolor=ACCENT,
                    darkcolor=ACCENT_DK, thickness=10)

        # Queue tables and scrollbars
        s.configure('Treeview', background=INPUT_BG, fieldbackground=INPUT_BG,
                    foreground=HEADER, bordercolor=BORDER, lightcolor=BORDER,
                    darkcolor=BORDER, rowheight=28)
        s.map('Treeview',
              background=[('selected', SELECTION_BG)],
              foreground=[('selected', '#ffffff')])
        s.configure('Treeview.Heading', background=SURFACE, foreground=HEADER,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    font=('Segoe UI', 9, 'bold'), padding=(6, 7))
        s.map('Treeview.Heading',
              background=[('active', SURFACE_ACTIVE)])
        s.configure('Vertical.TScrollbar', background=SURFACE,
                    troughcolor=INPUT_BG, bordercolor=BORDER,
                    arrowcolor=MUTED, lightcolor=BORDER, darkcolor=BORDER)
        s.map('Vertical.TScrollbar',
              background=[('active', SURFACE_ACTIVE)],
              arrowcolor=[('active', HEADER)])

        # Status bar
        s.configure('Status.TLabel', background=PANEL_BG, foreground=MUTED,
                    font=('Segoe UI', 8), padding=(8, 4), relief='flat')

    # ── variables ─────────────────────────────────────────────────────────────
    def setupVariables(self):
        self.experimentNameVar    = tk.StringVar(value="")
        self.datasetVar           = tk.StringVar(value="cifar10")
        self.approachVar          = tk.StringVar(value="basil")
        self.useBasilVar          = tk.BooleanVar(value=False)
        self.basilMemorySizeVar   = tk.IntVar(value=5)
        self.useChannelNoiseVar   = tk.BooleanVar(value=False)
        self.channelNoiseStartVar = tk.IntVar(value=0)
        self.channelNoiseSigmaVar = tk.DoubleVar(value=0.2)
        self.noiseMitigationVar   = tk.StringVar(value="none")
        self.ebmLambdaVar         = tk.DoubleVar(value=25.0)
        self.momentumVar          = tk.DoubleVar(value=0.9)
        self.distillStrengthVar   = tk.DoubleVar(value=0.5)
        self.verifyThresholdVar   = tk.DoubleVar(value=0.05)
        self.cartAlgorithmVar     = tk.StringVar(value="cart")   # cart | basil | noisy | merged
        self.useLrDecayVar        = tk.BooleanVar(value=True)
        self.usePlateauLrVar        = tk.BooleanVar(value=True)
        self.plateauPatienceVar     = tk.IntVar(value=8)
        self.plateauFactorVar       = tk.DoubleVar(value=0.7)
        self.plateauMinLrVar        = tk.DoubleVar(value=0.001)
        self.plateauThresholdVar    = tk.DoubleVar(value=0.01)
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
        self.nonIIDVar        = tk.BooleanVar(value=True)
        self.dirichletAlphaVar = tk.DoubleVar(value=0.2)
        self.attackerIdsVar   = tk.StringVar(value="1,4,6,8")
        self.nNodesVar        = tk.IntVar(value=10)
        self.nRoundsVar       = tk.IntVar(value=100)
        self.localEpochsVar   = tk.IntVar(value=5)
        self.learningRateVar  = tk.DoubleVar(value=0.05)
        self.batchSizeVar     = tk.IntVar(value=512)
        self._queueEtaVar = tk.StringVar(value="Queue estimate: empty")
        self._workerProfileVar = tk.StringVar(value="")
        # progress / status (not user-facing inputs)
        self._progressVar     = tk.DoubleVar(value=0.0)

    # ── UI construction ───────────────────────────────────────────────────────
    def createUI(self):
        # ---- status bar (very bottom) --------------------------------------
        self._statusVar = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self._statusVar,
                  style='Status.TLabel', anchor=tk.W
                  ).pack(side=tk.BOTTOM, fill=tk.X)

        # ---- progress bar (above status) -----------------------------------
        self._createProgressFrame()

        # ---- command header (top) ------------------------------------------
        self.createButtons()

        # ---- notebook (fills all remaining space) --------------------------
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        basicTab    = ttk.Frame(self.notebook)
        advancedTab = ttk.Frame(self.notebook)
        attackTab   = ttk.Frame(self.notebook)
        outputTab   = ttk.Frame(self.notebook)
        networkTab  = ttk.Frame(self.notebook)

        self.notebook.add(basicTab,    text="  Basic  ")
        self.notebook.add(advancedTab, text="  Advanced  ")
        self.notebook.add(attackTab,   text="  Attacks  ")
        self.notebook.add(outputTab,   text="  Output  ")
        self.notebook.add(networkTab,  text="  Network  ")

        self.createBasicTab(basicTab)
        self.createAdvancedTab(advancedTab)
        self.createAttackTab(attackTab)
        self.createOutputTab(outputTab)
        self.networkView = CampaignNetworkView(networkTab)

    def _createProgressFrame(self):
        pf = tk.Frame(self.root, bg=PANEL_BG, highlightthickness=1,
                      highlightbackground=BORDER, highlightcolor=BORDER)
        pf.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(6, 6))

        self._progressBar = ttk.Progressbar(pf, variable=self._progressVar,
                                            maximum=100, length=400,
                                            style='Blue.Horizontal.TProgressbar')
        self._progressBar.pack(side=tk.LEFT, padx=(12, 12), pady=10)

        self._roundLabel = ttk.Label(pf, text="Round –/–",
                                     font=('Segoe UI', 9, 'bold'), foreground=ACCENT,
                                     background=PANEL_BG)
        self._roundLabel.pack(side=tk.LEFT, padx=(0, 16))

        self._etaLabel = ttk.Label(pf, text="", foreground=MUTED, font=('Segoe UI', 8),
                                   background=PANEL_BG)
        self._etaLabel.pack(side=tk.LEFT)

        self._accLabel = ttk.Label(pf, text="", foreground=SUCCESS,
                                   font=('Segoe UI', 9, 'bold'), background=PANEL_BG)
        self._accLabel.pack(side=tk.RIGHT, padx=12)

    # ── Basic tab ─────────────────────────────────────────────────────────────
    def createBasicTab(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- Quick presets -------------------------------------------------
        presetFrame = ttk.LabelFrame(outer, text="Quick Presets", padding=8)
        presetFrame.pack(fill=tk.X, pady=(0, 8))

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

        # Approach — 2 rows × 2 cols so all 4 fit without crowding
        ttk.Label(frame, text="Approach:", font=('Segoe UI', 9, 'bold')).grid(
            row=row, column=0, rowspan=2, sticky=tk.W, pady=5)
        ttk.Radiobutton(frame, text="BASIL Only",
                        variable=self.approachVar, value="basil",
                        command=self.onApproachChange).grid(row=row, column=1, sticky=tk.W, padx=8)
        ttk.Radiobutton(frame, text="Noisy Channel Only",
                        variable=self.approachVar, value="noisy",
                        command=self.onApproachChange).grid(row=row, column=2, sticky=tk.W, padx=8)
        ttk.Radiobutton(frame, text="Merged (BASIL + Noisy)",
                        variable=self.approachVar, value="merged",
                        command=self.onApproachChange).grid(row=row, column=3, sticky=tk.W, padx=8)
        ttk.Radiobutton(frame, text="CART (Class-Aware Ring - Paper 003)",
                        variable=self.approachVar, value="cart",
                        command=self.onApproachChange).grid(row=row+1, column=1, columnspan=3, sticky=tk.W, padx=8)
        row += 2

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

        # Data Distribution
        distFrame = ttk.LabelFrame(frame, text="Data Distribution", padding=10)
        distFrame.pack(fill=tk.X, pady=(8, 0))

        ttk.Checkbutton(distFrame, text="Non-IID Data (Dirichlet partitioning)",
                        variable=self.nonIIDVar,
                        command=self._onNonIIDChange).grid(row=0, column=0, columnspan=3,
                                                            sticky=tk.W, pady=5)
        ttk.Label(distFrame, text="Dirichlet Alpha (α):").grid(row=1, column=0, sticky=tk.W)
        self._alphaEntry = ttk.Entry(distFrame, textvariable=self.dirichletAlphaVar, width=10)
        self._alphaEntry.grid(row=1, column=1, sticky=tk.W, padx=8)
        ttk.Label(distFrame,
                  text="Lower = more skewed  (0.1 = extreme, 0.2 = severe, 0.5 = moderate, 1.0 = mild)",
                  foreground=MUTED, font=('Segoe UI', 8)).grid(row=1, column=2, sticky=tk.W)
        self._onNonIIDChange()

        # CART
        cartFrame = ttk.LabelFrame(frame, text="CART Configuration (Paper 003)", padding=10)
        cartFrame.pack(fill=tk.X, pady=(8, 0))

        # Algorithm selector — lets you run baselines under the same non-IID CART setting
        ttk.Label(cartFrame, text="Algorithm:", font=('Segoe UI', 9, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 4))
        algFrame = ttk.Frame(cartFrame)
        algFrame.grid(row=0, column=1, columnspan=2, sticky=tk.W)
        for _alg, _lbl in (("cart",   "CART (Paper 003)"),
                            ("basil",  "BASIL Ring (baseline)"),
                            ("noisy",  "FedAvg / Noisy (baseline)"),
                            ("merged", "Merged Ring+EBM (baseline)")):
            ttk.Radiobutton(algFrame, text=_lbl,
                            variable=self.cartAlgorithmVar, value=_alg).pack(anchor=tk.W)

        ttk.Separator(cartFrame, orient='horizontal').grid(
            row=1, column=0, columnspan=3, sticky='ew', pady=6)

        cartRows = [
            ("Distillation Strength (γ):", self.distillStrengthVar,
             "Proximal coefficient scale — higher = stronger class knowledge preservation"),
            ("Verify Threshold:",           self.verifyThresholdVar,
             "Max allowed registry claim gap before rejecting Byzantine inflation"),
        ]
        for i, (lbl, var, hint) in enumerate(cartRows, start=2):
            ttk.Label(cartFrame, text=lbl).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Entry(cartFrame, textvariable=var, width=10).grid(
                row=i, column=1, sticky=tk.W, padx=8)
            ttk.Label(cartFrame, text=hint, foreground=MUTED,
                      font=('Segoe UI', 8)).grid(row=i, column=2, sticky=tk.W)

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
        ttk.Label(frame, text="comma-separated, e.g. 1,4,6,8",
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
            font=('Consolas', 9), bg=LOG_BG, fg=HEADER,
            insertbackground=HEADER, selectbackground=SELECTION_BG,
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
        self.outputText.tag_configure('header',  foreground='#8eb7ff', font=('Consolas', 9, 'bold'))
        self.outputText.tag_configure('success', foreground=SUCCESS)
        self.outputText.tag_configure('error',   foreground='#ff7b86')
        self.outputText.tag_configure('warn',    foreground=WARN_CLR)
        self.outputText.tag_configure('info',    foreground=INFO_CLR)
        self.outputText.tag_configure('muted',   foreground=MUTED)
        self.outputText.tag_configure('normal',  foreground=HEADER)

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
        shell = tk.Frame(self.root, bg=PANEL_BG, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=BORDER)
        shell.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(12, 8))

        headerRow = tk.Frame(shell, bg=PANEL_BG)
        headerRow.pack(fill=tk.X)

        titleBlock = tk.Frame(headerRow, bg=PANEL_BG)
        titleBlock.pack(side=tk.LEFT, fill=tk.Y, padx=(14, 10), pady=10)
        tk.Label(titleBlock, text="BASIL + Noisy Channel Lab",
                 bg=PANEL_BG, fg=HEADER, font=('Segoe UI', 14, 'bold')
                 ).pack(anchor=tk.W)

        actions = tk.Frame(headerRow, bg=PANEL_BG)
        actions.pack(side=tk.RIGHT, padx=12, pady=10)

        self.runButton = ttk.Button(
            actions, text="Run", style='Run.TButton',
            command=self.runExperiment)
        self.runButton.pack(side=tk.LEFT, padx=(0, 6))

        self.runAllButton = ttk.Button(
            actions, text="Run All", style='RunAll.TButton',
            command=self.runAll)
        self.runAllButton.pack(side=tk.LEFT, padx=6)

        self.queueButton = ttk.Button(
            actions, text="Queue (0)", style='Queue.TButton',
            command=self.openQueueManager)
        self.queueButton.pack(side=tk.LEFT, padx=6)

        self.stopButton = ttk.Button(
            actions, text="Stop", style='Stop.TButton',
            command=self.stopExperiment, state=tk.DISABLED)
        self.stopButton.pack(side=tk.LEFT, padx=6)

        moreMenu = tk.Menu(shell, tearoff=0)
        moreMenu.add_command(label="Save Config", accelerator="Ctrl+S",
                             command=self.saveConfig)
        moreMenu.add_command(label="Load Config", accelerator="Ctrl+L",
                             command=self.loadConfig)
        moreMenu.add_separator()
        moreMenu.add_command(label="Plot Results", command=self.plotResults)
        moreMenu.add_command(label="Clear Log", command=self.clearOutput)
        moreMenu.add_separator()
        moreMenu.add_command(label="Reload GUI", accelerator="Ctrl+Shift+R",
                             command=self.reloadGui)
        moreMenu.add_command(label="Exit", command=self.onClosing)

        self.moreButton = ttk.Menubutton(
            actions, text="More", style='More.TMenubutton')
        self.moreButton["menu"] = moreMenu
        self.moreButton.pack(side=tk.LEFT, padx=(10, 0))

        tk.Label(
            shell,
            textvariable=self._queueEtaVar,
            bg=PANEL_BG,
            fg=INFO_CLR,
            font=('Segoe UI', 8),
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=14, pady=(0, 8))

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

        campaignSeries = [
            (lane, series)
            for lane, series in sorted(self._campaignLiveData.items())
            if series.get("avg")
        ]
        if campaignSeries:
            laneColors = (ACCENT, RUN_ALL_CLR, SUCCESS, INFO_CLR)
            latest = None
            multipleLanes = len(campaignSeries) > 1
            for lane, series in campaignSeries:
                color = laneColors[lane % len(laneColors)]
                prefix = f"L{lane + 1} " if multipleLanes else ""
                rounds = series["rounds"]
                avgValues = series["avg"]
                worstValues = series["worst"]
                ax.plot(
                    rounds,
                    avgValues,
                    color=color,
                    linewidth=1.6,
                    marker='o',
                    markersize=2,
                    label=f"{prefix}Avg",
                )
                if any(
                    abs(worst - avg) > 1e-12
                    for worst, avg in zip(worstValues, avgValues)
                ):
                    ax.plot(
                        rounds,
                        worstValues,
                        color=color,
                        linewidth=1.0,
                        linestyle='--',
                        alpha=0.78,
                        label=f"{prefix}Worst",
                    )
                if latest is None or series["updated"] > latest["updated"]:
                    latest = series
            ax.legend(fontsize=7, framealpha=0.5)
            ax.set_title(
                f"Live Accuracy  (latest {latest['avg'][-1]:.1%})",
                fontsize=8,
                color=HEADER,
            )
        elif self._campaignLiveData:
            ax.set_title(
                "Worker started; waiting for first completed round...",
                fontsize=8,
                color=MUTED,
            )
        elif self._liveAccData:
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
        self._campaignLiveData.clear()
        self._refreshLiveChart()

    # ── Preset & auto-name helpers ─────────────────────────────────────────────
    def _onNonIIDChange(self):
        state = tk.NORMAL if self.nonIIDVar.get() else tk.DISABLED
        self._alphaEntry.configure(state=state)

    def _applyPreset(self, name):
        cfg = PRESETS[name]
        for variable in (
            self.attackGaussianVar,
            self.attackSignFlipVar,
            self.attackHiddenVar,
            self.attackModelPoisonVar,
            self.attackScalingVar,
            self.attackAlieVar,
            self.attackIpmVar,
            self.attackNoiseAmpVar,
        ):
            variable.set(False)
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
            'attackModelPoison': self.attackModelPoisonVar,
            'attackScaling':     self.attackScalingVar,
            'attackAlie':        self.attackAlieVar,
            'attackIpm':         self.attackIpmVar,
            'attackNoiseAmp':    self.attackNoiseAmpVar,
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
        approach = {'basil': 'BASIL', 'noisy': 'FedAvg', 'merged': 'Merged', 'cart': 'CART'}.get(
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
        # Tk widgets are only safe to touch from the main thread; queue and
        # plot threads log through the event loop instead of directly.
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, self.logMessage, message)
            return
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
        self._campaignLiveData.clear()
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
            self.useBasilVar.set(False); self.useChannelNoiseVar.set(False)
        elif approach == "noisy":
            self.useBasilVar.set(False); self.useChannelNoiseVar.set(True)
        elif approach == "merged":
            self.useBasilVar.set(False); self.useChannelNoiseVar.set(False)
        elif approach == "cart":
            self.useBasilVar.set(False); self.useChannelNoiseVar.set(False)

    # ── Run / Stop ────────────────────────────────────────────────────────────
    def runExperiment(self):
        if self.isRunning:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return
        if not self.validateConfig():
            return

        self.runButton.config(state=tk.DISABLED)
        self.runAllButton.config(state=tk.DISABLED)
        self.queueButton.config(state=tk.NORMAL)
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

    def _cleanupTensorflow(self, context="experiment boundary"):
        cleanupTensorflowMemory(
            logger=self.logMessage,
            context=context,
            collectCycles=3,
        )

    def _releaseExperimentObjects(self, nodes=None, trainLoaders=None, testLoader=None,
                                  train=None, test=None):
        if nodes:
            for nd in nodes:
                try:
                    nd.neighborMemory.clear()
                except Exception:
                    pass
                try:
                    nd.classRegistry = None
                except Exception:
                    pass
                try:
                    nd._cartStep = None
                    nd._compiledStep = None
                    nd._opt = None
                    nd._refParams = None
                except Exception:
                    pass
                try:
                    nd.model = None
                except Exception:
                    pass
                try:
                    nd.dataLoader = None
                except Exception:
                    pass
        if trainLoaders:
            try:
                trainLoaders.clear()
            except Exception:
                pass
        testLoader = None
        train = None
        test = None
        gc.collect()

    def stopExperiment(self):
        self.isRunning = False
        if self._campaignWorkerPool is not None:
            self._campaignWorkerPool.request_stop()
        if self._benchmarkProcess is not None and self._benchmarkProcess.poll() is None:
            try:
                os.killpg(os.getpgid(self._benchmarkProcess.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                self._benchmarkProcess.terminate()
        self.logMessage("\n[STOP REQUESTED] Stopping experiment…")
        self._setStatus("Stop requested - waiting for current round to finish…")

    def onClosing(self):
        if self.isRunning:
            if not messagebox.askyesno("Experiment Running",
                                       "An experiment is running. Stop it and exit?"):
                return
            self.isRunning = False
            if self._campaignWorkerPool is not None:
                self._campaignWorkerPool.request_stop()
            if self.currentThread and self.currentThread.is_alive():
                self.currentThread.join(timeout=5.0)
            if self._campaignWorkerPool is not None:
                self._campaignWorkerPool.kill_remaining()
            if self._benchmarkProcess is not None and self._benchmarkProcess.poll() is None:
                try:
                    os.killpg(os.getpgid(self._benchmarkProcess.pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    self._benchmarkProcess.terminate()
        with self._queueLock:
            self._saveQueueState()
        self.root.destroy()

    def reloadGui(self):
        """Exit with code 42 so the launcher restarts the GUI with latest code."""
        if self.isRunning:
            if not messagebox.askyesno("Experiment Running",
                                       "An experiment is running. Stop it and reload?"):
                return
            self.isRunning = False
            if self._campaignWorkerPool is not None:
                self._campaignWorkerPool.request_stop()
            if self.currentThread and self.currentThread.is_alive():
                self.currentThread.join(timeout=5.0)
            if self._campaignWorkerPool is not None:
                self._campaignWorkerPool.kill_remaining()
            if self._benchmarkProcess is not None and self._benchmarkProcess.poll() is None:
                try:
                    os.killpg(os.getpgid(self._benchmarkProcess.pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    self._benchmarkProcess.terminate()
        with self._queueLock:
            self._saveQueueState()
        self.root.destroy()
        import sys
        sys.exit(42)

    def runExperimentThread(self):
        try:
            config = self.getConfig()
            config['aggregationMode'] = self.defaultAggregationMode(config)
            name = config.get("experimentName", "Experiment")
            self._trainStartTime = time.time()
            self._lastRoundEndTime = self._trainStartTime
            self._totalRounds = config.get('nRounds', 100)
            self.root.after(1000, self._etaTicker)
            sendNotification("Run Started", f"{name} has started.", priority="default")
            self._executeExperiment(config)
            if (
                int(config.get("campaignVersion", 0)) == 3
                and isCampaign3Completed(config)
                and config.get("autoPlotCampaign3", True)
            ):
                self.generatePlots(
                    showDialog=False,
                    config=config,
                    onlyMissing=True,
                )
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
        self.queueButton.config(state=tk.NORMAL)
        self.stopButton.config(state=tk.DISABLED)
        self.isRunning = False
        hasLiveData = bool(
            self._liveAccData
            or any(
                series.get("avg")
                for series in self._campaignLiveData.values()
            )
        )
        self._progressVar.set(100 if hasLiveData else 0)
        self._runtimeEstimator.reload()
        self._refreshQueueEstimate()
        self._setStatus("Idle  ·  Experiment finished.")

    # ── _getResultPaths / _isAlreadyRun ───────────────────────────────────────
    def _getResultPaths(self, config):
        if int(config.get('campaignVersion', 0)) == 3:
            metricsPath, runPath = campaign3ResultPaths(config)
            return str(metricsPath), str(runPath)
        if int(config.get('campaignVersion', 0)) == 4:
            metricsPath, runPath, _ = campaign4ResultPaths(config)
            return str(metricsPath), str(runPath)

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
        split = 'nonIID' if config.get('nonIID', True) else 'IID'
        resultDir = self._resultDir(config, split, dataset, attackKey, approach)
        return f"{resultDir}/acc_{safeName}.npy", f"{resultDir}/config_{safeName}.json"

    def _resultDir(self, config, split, dataset, attackKey, approach):
        resultRoot = "results2" if approach in ("merged", "cart") else "results"
        parts = ["experiments", resultRoot, "gui", split, dataset, attackKey, approach]
        if config.get('useChannelNoise', False):
            sigma = float(config.get('channelNoiseSigma', 0.0))
            sigmaLabel = f"sigma_{sigma:.1f}".replace('.', '_')
            parts.append(sigmaLabel)
        return os.path.join(*parts)

    def _isAlreadyRun(self, config):
        if int(config.get('campaignVersion', 0)) == 3:
            return isCampaign3Completed(config)
        if int(config.get('campaignVersion', 0)) == 4:
            return isCampaign4Completed(config)

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

    def defaultAggregationMode(self, config):
        if config.get('aggregationMode'):
            return config['aggregationMode']
        if config.get('snapshotSelection', config.get('useBasil', False)):
            return 'handoff'
        if config.get('approach') in ('merged', 'cart'):
            return 'consensus'
        return 'handoff'

    # ── Run All ───────────────────────────────────────────────────────────────
    def runAll(self):
        if self.isRunning:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return

        configDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        configFiles = sorted([
            os.path.join(root, fname)
            for root, _, files in os.walk(configDir)
            for fname in files if fname.endswith('.json')
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
        self.queueButton.config(state=tk.NORMAL)
        self.stopButton.config(state=tk.NORMAL)
        self.isRunning = True
        self._queueBatchNeedsPlot = False
        self._queueLastEstimateRefresh = 0.0
        self.clearOutput()
        self._liveAccData.clear()
        self._liveWorstData.clear()
        self._refreshLiveChart()
        self.notebook.select(3)

        self.currentThread = threading.Thread(target=self.runAllThread, args=(pending, nSkip))
        self.currentThread.start()

    def runAllThread(self, configFiles, nSkipped=0):
        total, completed = len(configFiles), 0
        campaignPlotConfig = None
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
                if (
                    int(config.get("campaignVersion", 0)) == 3
                    and isCampaign3Completed(config)
                ):
                    campaignPlotConfig = config
                    if config.get("autoPlotCampaign3", True):
                        self._scheduleCampaign3PlotRefresh(config)

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
            if campaignPlotConfig is not None:
                self._waitForCampaign3PlotRefresh()
                self.generatePlots(
                    showDialog=False,
                    config=campaignPlotConfig,
                    onlyMissing=True,
                )
            self.root.after(0, self._onRunFinished)

    # ── _executeExperiment ────────────────────────────────────────────────────
    def _executeExperiment(self, config):
        cleanAdjusted = self._applyCleanReferenceSemantics(config)
        self._validateMitigationSemantics(config)
        self._cleanupTensorflow("before experiment")
        self._activeExperimentObjects = {}
        try:
            if cleanAdjusted and config.get('_cleanReferenceNote'):
                self.logMessage(config['_cleanReferenceNote'])
            return self._executeExperimentImpl(config)
        finally:
            activeObjects = self._activeExperimentObjects
            self._releaseExperimentObjects(
                nodes=activeObjects.get("nodes"),
                trainLoaders=activeObjects.get("trainLoaders"),
                testLoader=activeObjects.get("testLoader"),
                train=activeObjects.get("train"),
                test=activeObjects.get("test"),
            )
            self._activeExperimentObjects = {}
            self._cleanupTensorflow("after experiment")

    def _executeExperimentImpl(self, config):
        setExperimentSeed(config.get("seed"))
        self.logMessage("=" * 80)
        self.logMessage("STARTING EXPERIMENT")
        self.logMessage("=" * 80)
        self.logMessage(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logMessage("")

        approachLabels = {
            'basil':  'BASIL Only (Ring Topology - Paper 001)',
            'noisy':  'Noisy Channel Only (FedAvg - Paper 002)',
            'merged': 'Merged (Ring + EBM)',
            'cart':   f"CART — {config.get('cartAlgorithm','cart').upper()} algorithm",
        }
        self.logMessage("Configuration:")
        self.logMessage(f"  Experiment: {config.get('experimentName', '(unnamed)')}")
        self.logMessage(f"  Dataset:    {config['dataset']}")
        self.logMessage(f"  Approach:   {approachLabels.get(config['approach'], config['approach'])}")
        self.logMessage(f"  Nodes: {config['nNodes']},  Rounds: {config['nRounds']}")
        if config.get('usePlateauLr', True):
            lrDecayStr = f"plateau (patience={config.get('plateauPatience',10)}, factor={config.get('plateauFactor',0.5)}, min={config.get('plateauMinLr',1e-4)}, threshold={config.get('plateauThreshold',0.01)})"
        elif config.get('useLrDecay', True):
            lrDecayStr = "decay"
        else:
            lrDecayStr = "fixed"
        self.logMessage(f"  LR: {config['learningRate']} ({lrDecayStr}),  Momentum: {config.get('momentum', 0.0)}")
        self.logMessage(f"  Use BASIL: {config['useBasil']},  Use Noise: {config['useChannelNoise']}")
        if config['useChannelNoise']:
            self.logMessage(f"  Noise σ: {config['channelNoiseSigma']},  Mitigation: {config['noiseMitigation']}")
            if config['noiseMitigation'] == 'ebm':
                coefficient = (
                    config['ebmLambda'] * config['channelNoiseSigma'] ** 2
                )
                if int(config.get('campaignVersion', 0)) == 3:
                    self.logMessage(
                        f"  EBM λ: {config['ebmLambda']} "
                        f"(λσ² objective coefficient={coefficient:.6g})"
                    )
                else:
                    self.logMessage(
                        f"  EBM λ: {config['ebmLambda']} "
                        f"(legacy scale={1.0 + coefficient:.2f})"
                    )
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
        self._activeExperimentObjects["train"] = train
        self._activeExperimentObjects["test"] = test
        iid = not config.get('nonIID', True)
        alpha = config.get('dirichletAlpha', 0.2)
        isCampaign3 = int(config.get('campaignVersion', 0)) == 3
        loaderResult = self.makeLoaders(
            config['dataset'], train, test, config['batchSize'], config['nNodes'],
            iid=iid,
            dirichletAlpha=alpha,
            seed=config.get('seed'),
            returnMetadata=isCampaign3,
        )
        if isCampaign3:
            trainLoaders, testLoader, dataMetadata = loaderResult
        else:
            trainLoaders, testLoader = loaderResult
            dataMetadata = None
        self._activeExperimentObjects["trainLoaders"] = trainLoaders
        self._activeExperimentObjects["testLoader"] = testLoader
        self.logMessage(f"  Training samples: {len(train)}")
        self.logMessage(f"  Test samples:     {len(test)}")
        if config.get('nonIID', True):
            self.logMessage(f"  Data split:       Non-IID (Dirichlet α={alpha})")
        else:
            self.logMessage(f"  Data split:       IID (uniform random)")
        if config.get("seed") is not None:
            self.logMessage(f"  Experiment seed:  {config['seed']}")
        self.logMessage("")

        if isCampaign3:
            self._executeCampaignThree(
                config,
                trainLoaders=trainLoaders,
                testLoader=testLoader,
                dataMetadata=dataMetadata,
            )
            return

        self.logMessage(f"Creating {config['nNodes']} nodes…")
        nodes = self.createNodes(config, trainLoaders)
        self._activeExperimentObjects["nodes"] = nodes
        self.logMessage("")

        attackTypes, attackerIds = self.prepareAttacks(config)
        self.logMessage(f"Attack configuration:")
        self.logMessage(f"  Attackers:    {attackerIds if attackerIds else 'None'}")
        self.logMessage(f"  Attack types: {attackTypes}\n")

        cartAlg = config.get('cartAlgorithm', 'cart')
        aggregationMode = self.defaultAggregationMode(config)
        config['aggregationMode'] = aggregationMode

        if config['approach'] == 'cart' and cartAlg == 'cart':
            self.logMessage(f"Starting CART ring training for {config['nRounds']} rounds…")
            self.logMessage(f"Mode: CART (Class-Aware Ring Training, {aggregationMode} aggregation) - Paper 003")
            self.logMessage("-" * 80)
            avgAccHist, worstAccHist = cartRingTraining(
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
                useSnapshots=config.get('useBasil', True),
                aggregationMode=aggregationMode,
                stopCallback=lambda: not self.isRunning,
                useLrDecay=config.get('useLrDecay', True),
                usePlateauLr=config.get('usePlateauLr', True),
                plateauPatience=config.get('plateauPatience', 8),
                plateauFactor=config.get('plateauFactor', 0.7),
                plateauMinLr=config.get('plateauMinLr', 0.001),
                plateauThreshold=config.get('plateauThreshold', 0.01),
                nClasses=10,
                verifyThreshold=config.get('verifyThreshold', 0.05),
                roundCallback=self._onRoundComplete,
            )
        elif config['approach'] == 'cart' and cartAlg == 'noisy':
            self.logMessage(f"Starting CART-setting FedAvg (baseline) for {config['nRounds']} rounds…")
            self.logMessage("Mode: FedAvg baseline in CART non-IID setting")
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
                useSnapshots=False,
                S=config.get('basilMemorySize', 4),
                stopCallback=lambda: not self.isRunning,
                useLrDecay=config.get('useLrDecay', True),
                usePlateauLr=config.get('usePlateauLr', True),
                plateauPatience=config.get('plateauPatience', 8),
                plateauFactor=config.get('plateauFactor', 0.7),
                plateauMinLr=config.get('plateauMinLr', 0.001),
                plateauThreshold=config.get('plateauThreshold', 0.01),
                roundCallback=self._onRoundComplete,
            )
        elif config['approach'] == 'cart' and cartAlg in ('basil', 'merged'):
            _ssOn  = True   # both basil and merged use SS
            _ebm   = (cartAlg == 'merged')  # merged also uses EBM
            _label = 'BASIL Ring baseline' if cartAlg == 'basil' else 'Merged Ring+EBM baseline'
            self.logMessage(f"Starting CART-setting {_label} for {config['nRounds']} rounds…")
            self.logMessage(f"Mode: {_label} in CART non-IID setting")
            self.logMessage("-" * 80)
            # Override noise model for merged: enable EBM
            _noiseModel = self.getNoiseModel(config)
            if _ebm and config.get('channelNoiseSigma', 0) > 0:
                _noiseModel = 'ebm'
            avgAccHist, worstAccHist = basilRingTrainingWithAttack(
                nodes=nodes,
                rounds=config['nRounds'],
                testLoader=testLoader,
                attackTypes=attackTypes,
                attackerIds=attackerIds,
                hiddenStartRound=config['attackHiddenStart'] if config['attackHidden'] else 999,
                sigma=config['channelNoiseSigma'] if config['useChannelNoise'] else 0.0,
                noiseModel=_noiseModel,
                channelNoiseStart=config['channelNoiseStart'] if config['useChannelNoise'] else 0,
                lr0=config['learningRate'],
                stepsPerEpoch=config.get('stepsPerEpoch', 5),
                useSnapshots=_ssOn,
                useSequential=True,
                aggregationMode=aggregationMode,
                stopCallback=lambda: not self.isRunning,
                useLrDecay=config.get('useLrDecay', True),
                usePlateauLr=config.get('usePlateauLr', True),
                plateauPatience=config.get('plateauPatience', 8),
                plateauFactor=config.get('plateauFactor', 0.7),
                plateauMinLr=config.get('plateauMinLr', 0.001),
                plateauThreshold=config.get('plateauThreshold', 0.01),
                roundCallback=self._onRoundComplete,
            )
        elif config['approach'] == 'noisy':
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
                useSnapshots=config.get('useBasil', False),
                S=config.get('basilMemorySize', 4),
                stopCallback=lambda: not self.isRunning,
                useLrDecay=config.get('useLrDecay', True),
                usePlateauLr=config.get('usePlateauLr', True),
                plateauPatience=config.get('plateauPatience', 8),
                plateauFactor=config.get('plateauFactor', 0.7),
                plateauMinLr=config.get('plateauMinLr', 0.001),
                plateauThreshold=config.get('plateauThreshold', 0.01),
                roundCallback=self._onRoundComplete,
            )
        else:
            self.logMessage(f"Starting Ring training for {config['nRounds']} rounds…")
            self.logMessage(f"Mode: Ring Topology (Sequential, {aggregationMode} aggregation)")
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
                aggregationMode=aggregationMode,
                stopCallback=lambda: not self.isRunning,
                useLrDecay=config.get('useLrDecay', True),
                usePlateauLr=config.get('usePlateauLr', True),
                plateauPatience=config.get('plateauPatience', 8),
                plateauFactor=config.get('plateauFactor', 0.7),
                plateauMinLr=config.get('plateauMinLr', 0.001),
                plateauThreshold=config.get('plateauThreshold', 0.01),
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

    def _partitionHash(self, dataMetadata):
        digest = hashlib.sha256()
        for indices in dataMetadata.get("clientIndices", []):
            array = np.asarray(indices, dtype=np.int64)
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def _codeRevision(self):
        gitDir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".git",
        )
        try:
            with open(os.path.join(gitDir, "HEAD"), "r", encoding="utf-8") as handle:
                head = handle.read().strip()
            if head.startswith("ref: "):
                refPath = os.path.join(gitDir, *head[5:].split("/"))
                with open(refPath, "r", encoding="utf-8") as handle:
                    return handle.read().strip()
            return head
        except OSError:
            return "unknown"

    def _executeCampaignThree(self, config, trainLoaders, testLoader, dataMetadata):
        metricsPath, runPath = campaign3ResultPaths(config)
        startedAt = datetime.now().astimezone().isoformat()
        partitionHash = self._partitionHash(dataMetadata)
        runMetadata = {
            "schemaVersion": 3,
            "campaignId": config.get("campaignId"),
            "runId": config.get("runId"),
            "status": "running",
            "configHash": campaign3ConfigHash(config),
            "partitionHash": partitionHash,
            "codeRevision": self._codeRevision(),
            "startedAt": startedAt,
            "config": dict(config),
        }
        write_json_atomic(runPath, runMetadata)

        self.logMessage("Starting Campaign 3 shared-worker ring training…")
        self.logMessage(
            "Mode: one compiled GPU worker; deterministic logical-node state"
        )
        self.logMessage("-" * 80)
        try:
            result = run_campaign_three(
                config=config,
                model_class=self.getModelClass(config["dataset"]),
                train_loaders=trainLoaders,
                test_loader=testLoader,
                data_metadata=dataMetadata,
                stop_callback=lambda: not self.isRunning,
                round_callback=self._onRoundComplete,
            )
        except Exception as error:
            runMetadata["status"] = "failed"
            runMetadata["failedAt"] = datetime.now().astimezone().isoformat()
            runMetadata["error"] = str(error)
            write_json_atomic(runPath, runMetadata)
            raise

        if result.get("stopped") or not self.isRunning:
            runMetadata["status"] = "stopped"
            runMetadata["completedRounds"] = int(len(result.get("avg_history", [])))
            runMetadata["stoppedAt"] = datetime.now().astimezone().isoformat()
            write_json_atomic(runPath, runMetadata)
            self.logMessage("\nCampaign 3 run stopped; queue item and metadata retained.")
            return

        initializationHash = result.pop("initialization_hash")
        result.pop("stopped", None)
        write_npz_atomic(metricsPath, **result)

        runMetadata.update({
            "status": "completed",
            "completedAt": datetime.now().astimezone().isoformat(),
            "initializationHash": initializationHash,
            "finalAverageAccuracy": float(result["final_avg"]),
            "finalWorstAccuracy": float(result["final_worst"]),
            "runtimeSeconds": float(result["runtime_seconds"]),
            "peakGpuBytes": int(result["peak_gpu_bytes"]),
        })
        write_json_atomic(runPath, runMetadata)

        finalAvg = float(result["final_avg"])
        finalWorst = float(result["final_worst"])
        finalNodes = np.asarray(result["final_node_accuracy"])
        self.logMessage("")
        self.logMessage("=" * 80)
        self.logMessage("FINAL CAMPAIGN 3 RESULTS:")
        self.logMessage(f"  Average Accuracy:      {finalAvg:.4f}  ({finalAvg:.1%})")
        self.logMessage(f"  Worst Node Accuracy:   {finalWorst:.4f}  ({finalWorst:.1%})")
        self.logMessage(f"  Per-node: {[f'{value:.3f}' for value in finalNodes]}")
        self.logMessage(f"  Runtime: {float(result['runtime_seconds']):.1f} seconds")
        self.logMessage("=" * 80)
        self.logMessage("\nResult files:")
        self.logMessage(f"  saved: {metricsPath}")
        self.logMessage(f"  saved: {runPath}")
        calibrationState = freeze_calibration_if_ready()
        if calibrationState is not None:
            if calibrationState.get("status") == "frozen":
                gammaSchedule = calibrationState.get("cartGammaBySigma", {})
                self.logMessage(
                    "  calibration frozen: CART gamma schedule="
                    f"{gammaSchedule}"
                )
            else:
                self.logMessage(
                    "  calibration completed but no non-zero CART gamma passed "
                    "the predeclared selection rule."
                )
        self.root.after(
            0,
            self._setStatus,
            f"Done  ·  Avg acc {finalAvg:.1%}  ·  Worst {finalWorst:.1%}",
        )

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
            self._validateMitigationSemantics(self.getConfig())
            return True
        except ValueError as e:
            messagebox.showerror("Invalid Mitigation", str(e)); return False
        except Exception as e:
            messagebox.showerror("Error", f"Invalid configuration: {e}"); return False

    def _hasByzantineAttack(self, config):
        if not str(config.get('attackerIds', '')).strip():
            return False
        attackKeys = (
            'attackGaussian', 'attackSignFlip', 'attackHidden',
            'attackModelPoison', 'attackScaling', 'attackAlie',
            'attackIpm', 'attackNoiseAmp',
        )
        return any(bool(config.get(key)) for key in attackKeys)

    def _validateMitigationSemantics(self, config):
        """Reject defense combinations that do not match an active failure mode."""
        hasAttack = self._hasByzantineAttack(config)
        hasNoise = bool(config.get("useChannelNoise", False))
        usesSnapshots = bool(
            config.get("snapshotSelection", False) or config.get("useBasil", False)
        )
        usesEbm = config.get("noiseMitigation", "none") == "ebm"

        if usesSnapshots and not hasAttack:
            raise ValueError(
                "Snapshot Selection requires an active Byzantine attack and "
                "at least one attacker ID."
            )
        if usesEbm and not hasNoise:
            raise ValueError("EBM requires active channel noise.")
        if hasNoise and float(config.get("channelNoiseSigma", 0.0)) <= 0.0:
            raise ValueError("Active channel noise requires sigma > 0.")

    def _isCartTwoAttackTwoMitigation(self, config):
        return (
            config.get('approach') == 'cart'
            and config.get('cartAlgorithm', 'cart') == 'cart'
            and bool(config.get('useBasil', False))
            and bool(config.get('useChannelNoise', False))
            and config.get('noiseMitigation', 'none') == 'ebm'
            and self._hasByzantineAttack(config)
        )

    def _isNamedCleanReference(self, config):
        return (
            config.get('experimentName', '').strip()
            == '0 - Byzantine Nodes + No Channel Noise + No Mitigation'
        )

    def _applyCleanReferenceSemantics(self, config):
        """The named clean reference must stay attack-free, noiseless, and unmitigated."""
        if int(config.get("campaignVersion", 0)) == 3:
            return False
        if not self._isNamedCleanReference(config):
            return False

        changed = False
        cleanValues = {
            'useBasil': False,
            'useChannelNoise': False,
            'channelNoiseSigma': 0.0,
            'noiseMitigation': 'none',
            'attackerIds': '',
            'attackGaussian': False,
            'attackSignFlip': False,
            'attackHidden': False,
            'attackModelPoison': False,
            'attackScaling': False,
            'attackAlie': False,
            'attackIpm': False,
            'attackNoiseAmp': False,
        }
        for key, value in cleanValues.items():
            if config.get(key) != value:
                config[key] = value
                changed = True

        if config.get('approach') in ('basil', 'merged', 'cart'):
            if config.get('aggregationMode') != 'consensus':
                config['aggregationMode'] = 'consensus'
                changed = True

        if changed:
            config['_cleanReferenceNote'] = (
                "Applied clean reference semantics: no Byzantine attack, no channel "
                "noise, no SS/EBM, consensus over correct updates."
            )
        return changed

    def _applyCartTargetTuning(self, config):
        """Legacy compatibility hook; campaign runs never receive hidden tuning."""
        return False

    def getConfig(self):
        config = {
            'experimentName':        self.experimentNameVar.get(),
            'dataset':               self.datasetVar.get(),
            'approach':              self.approachVar.get(),
            'useBasil':              self.useBasilVar.get(),
            'snapshotSelection':     self.useBasilVar.get(),
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
            'nonIID':                self.nonIIDVar.get(),
            'dirichletAlpha':        self.dirichletAlphaVar.get(),
            'distillStrength':       self.distillStrengthVar.get(),
            'verifyThreshold':       self.verifyThresholdVar.get(),
            'cartAlgorithm':         self.cartAlgorithmVar.get(),
            'attackerIds':           self.attackerIdsVar.get(),
            'nNodes':                self.nNodesVar.get(),
            'nRounds':               self.nRoundsVar.get(),
            'localEpochs':           self.localEpochsVar.get(),
            'learningRate':          self.learningRateVar.get(),
            'batchSize':             self.batchSizeVar.get(),
            'stepsPerEpoch':         5,
            'useLrDecay':            self.useLrDecayVar.get(),
            'usePlateauLr':          self.usePlateauLrVar.get(),
            'plateauPatience':       self.plateauPatienceVar.get(),
            'plateauFactor':         self.plateauFactorVar.get(),
            'plateauMinLr':          self.plateauMinLrVar.get(),
            'plateauThreshold':      self.plateauThresholdVar.get(),
        }
        if self._applyCleanReferenceSemantics(config):
            self.useBasilVar.set(config['useBasil'])
            self.useChannelNoiseVar.set(config['useChannelNoise'])
            self.channelNoiseSigmaVar.set(config['channelNoiseSigma'])
            self.noiseMitigationVar.set(config['noiseMitigation'])
            self.attackerIdsVar.set(config['attackerIds'])
            self.attackGaussianVar.set(config['attackGaussian'])
            self.attackSignFlipVar.set(config['attackSignFlip'])
            self.attackHiddenVar.set(config['attackHidden'])
            self.attackModelPoisonVar.set(config['attackModelPoison'])
            self.attackScalingVar.set(config['attackScaling'])
            self.attackAlieVar.set(config['attackAlie'])
            self.attackIpmVar.set(config['attackIpm'])
            self.attackNoiseAmpVar.set(config['attackNoiseAmp'])
        return config

    # ── Data / Node helpers ───────────────────────────────────────────────────
    def loadDataset(self, dataset):
        if dataset == "mnist":    return loadMnist()
        if dataset == "cifar10":  return loadCifar10()
        if dataset == "nmnist":   return loadNMnist()
        raise ValueError(f"Unknown dataset: {dataset}")

    def makeLoaders(
        self,
        dataset,
        train,
        test,
        batchSize,
        nClients,
        iid=True,
        dirichletAlpha=0.2,
        seed=None,
        returnMetadata=False,
    ):
        kwargs = dict(
            batchSize=batchSize,
            nClients=nClients,
            iid=iid,
            dirichletAlpha=dirichletAlpha,
        )
        if dataset == "cifar10":
            kwargs.update(seed=seed, returnMetadata=returnMetadata)
        if dataset == "mnist":   return makeMnistLoaders(train, test, **kwargs)
        if dataset == "cifar10": return makeCifarLoaders(train, test, **kwargs)
        if dataset == "nmnist":  return makeNMnistLoaders(train, test, **kwargs)
        raise ValueError(f"Unknown dataset: {dataset}")

    def createNodes(self, config, trainLoaders):
        modelClass = self.getModelClass(config['dataset'])
        isCart = (config.get('approach') == 'cart' and
                  config.get('cartAlgorithm', 'cart') == 'cart')
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
            if isCart:
                nodeCfg['distillStrength'] = config.get('distillStrength', 0.5)
                nodeCfg['verifyThreshold'] = config.get('verifyThreshold', 0.05)
                nodes.append(CARTNode(**nodeCfg))
            else:
                nodes.append(BasilNode(**nodeCfg))
        if nodes:
            initialParams = getParams(nodes[0].model)
            for nd in nodes[1:]:
                setParams(nd.model, initialParams)
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
        config['aggregationMode'] = self.defaultAggregationMode(config)
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
        split = 'nonIID' if config.get('nonIID', True) else 'IID'
        resultDir = self._resultDir(config, split, dataset, attackKey, approach)
        os.makedirs(resultDir, exist_ok=True)

        avgPath    = f"{resultDir}/acc_{safeName}.npy"
        configPath = f"{resultDir}/config_{safeName}.json"

        self.logMessage(f"\nResult files:")
        if os.path.exists(avgPath):
            self.logMessage(f"  exists, skipped: {avgPath}")
        else:
            np.save(avgPath, np.array(avgAccHist))
            self.logMessage(f"  saved: {avgPath}")

        if os.path.exists(configPath):
            self.logMessage(f"  exists, skipped: {configPath}")
        else:
            with open(configPath, 'w') as f:
                json.dump(config, f, indent=2)
            self.logMessage(f"  saved: {configPath}")

        self.generatePlots(showDialog=False, config=config, onlyMissing=True)

    # ── Config Queue ──────────────────────────────────────────────────────────
    def _workerSettings(self, campaignVersion=None):
        if campaignVersion is None:
            with self._queueLock:
                campaignVersion = (
                    int(self.configQueue[0].get("campaignVersion", 0))
                    if self.configQueue
                    else 3
                )
        profile = (
            self._workerProfile4
            if int(campaignVersion) == 4
            else self._workerProfile
        ) or {}
        if int(campaignVersion) == 4:
            return campaign4ExecutionSettings(profile)
        validated = (
            profile.get("status") == "validated"
            and int(profile.get("recommendedLanes", 1)) == 2
        )
        lanes = 2 if validated else 1
        return {
            "lanes": lanes,
            "gpuMemoryLimitMb": int(
                profile.get(
                    "gpuMemoryLimitMb",
                    7600 if int(campaignVersion) == 4 else 4200,
                )
            ),
            "concurrencySlowdown": (
                float(profile.get("concurrencySlowdown", 1.0))
                if lanes > 1
                else 1.0
            ),
            "status": str(profile.get("status", "not_benchmarked")),
        }

    def _queueEstimateText(self):
        with self._queueLock:
            queueSnapshot = list(self.configQueue)
        settings = self._workerSettings()
        activeElapsed = (
            self._campaignWorkerPool.active_elapsed_by_run_id()
            if self._campaignWorkerPool is not None
            else {}
        )
        estimate = self._runtimeEstimator.estimate_queue(
            queueSnapshot,
            lanes=settings["lanes"],
            active_elapsed=activeElapsed,
            concurrency_slowdown=settings["concurrencySlowdown"],
            max_concurrent_ebm=settings.get("maxConcurrentEbm"),
            allow_mixed_ebm_standard=settings.get(
                "allowMixedEbmStandard",
                True,
            ),
            allow_dual_standard=settings.get("allowDualStandard", True),
            prioritize_ebm=settings.get("prioritizeEbm", False),
            runtime_slowdowns=settings.get("runtimeSlowdowns"),
        )
        if not queueSnapshot:
            return "Queue estimate: empty"
        finish = datetime.now().astimezone() + timedelta(seconds=estimate.seconds)
        wallText = (
            f"{format_duration(estimate.seconds)} "
            f"({format_duration(estimate.low_seconds)}–"
            f"{format_duration(estimate.high_seconds)})"
        )
        if estimate.lanes == 1:
            durationText = f"Total remaining: {wallText}"
        else:
            durationText = (
                f"Wall-clock remaining: {wallText} · "
                f"total config work: {format_duration(estimate.work_seconds)} "
                f"({format_duration(estimate.low_work_seconds)}–"
                f"{format_duration(estimate.high_work_seconds)})"
            )
        return (
            f"{durationText} · {estimate.item_count} config"
            f"{'s' if estimate.item_count != 1 else ''} · "
            f"{estimate.lanes} GPU lane{'s' if estimate.lanes != 1 else ''} · "
            f"{estimate.historical_samples} recorded runtimes · "
            f"finish about {finish.strftime('%a %b %d, %I:%M %p')}"
        )

    def _refreshQueueEstimate(self):
        self._queueEtaVar.set(self._queueEstimateText())
        settings = self._workerSettings()
        with self._queueLock:
            campaignVersion = (
                int(self.configQueue[0].get("campaignVersion", 0))
                if self.configQueue
                else 3
            )
        profile = (
            self._workerProfile4
            if campaignVersion == 4
            else self._workerProfile
        ) or {}
        selectedProfile = profile.get("selectedProfile") or {}
        precisionValidation = profile.get("precisionValidation") or {}
        backendSpeedup = float(precisionValidation.get("medianSpeedup", 0.0))
        executionName = (
            f"{selectedProfile.get('precisionProfile', 'float32')} / "
            f"{selectedProfile.get('gpuAllocator', 'bfc')}"
            if campaignVersion == 4
            else "isolated workers"
        )
        speedText = (
            f"{backendSpeedup:.2f}x backend / "
            f"{settings.get('measuredSpeedup', 0.0):.2f}x lane throughput"
            if campaignVersion == 4
            else f"{float(profile.get('measuredSpeedup', 0.0)):.2f}x benchmark speedup"
        )
        if settings["lanes"] == 2:
            limits = settings.get("memoryLimitsMb", {})
            profileText = (
                f"Execution: validated resource-aware 2-lane mode · "
                f"{executionName} · "
                f"{speedText} · "
                f"max {settings.get('maxConcurrentEbm', 1)} EBM lane · "
                f"standard {limits.get('standard', '?')} MB / "
                f"EBM {limits.get('ebm', '?')} MB"
            )
        elif settings.get("status") == "single_lane_required":
            profileText = (
                f"Execution: 1 isolated GPU worker · {executionName} · "
                "two-lane benchmark did not "
                "meet the safety/throughput gate"
            )
        else:
            profileText = (
                f"Execution: 1 isolated GPU worker · {executionName} · "
                "benchmark 2 lanes before "
                "enabling concurrency"
            )
        prefix = "Campaign 4" if campaignVersion == 4 else "Campaign 3"
        self._workerProfileVar.set(f"{prefix} · {profileText}")

    def _scheduleQueueEstimateRefresh(self):
        try:
            text = self._queueEstimateText()
        except Exception as error:
            text = f"Queue estimate unavailable: {error}"
        self.root.after(0, self._queueEtaVar.set, text)

    def _configsByEstimatedDuration(self, configs):
        """Return a stable shortest-estimated-runtime-first config list."""
        return sorted(
            list(configs),
            key=lambda config: self._runtimeEstimator.estimate(config).seconds,
        )

    def _saveQueueState(self):
        try:
            write_json_atomic(self._queueStatePath, self.configQueue)
        except Exception as e:
            self.logMessage(f"WARNING: could not save queue state: {e}")

    def _setQueueActiveSentinel(self):
        try:
            write_json_atomic(
                self._queueActivePath,
                {"pid": os.getpid(), "startedAt": time.time()},
            )
        except Exception as e:
            self.logMessage(f"WARNING: could not mark queue active: {e}")

    def _clearQueueActiveSentinel(self):
        try:
            os.unlink(self._queueActivePath)
        except OSError:
            pass

    def _workerRoots(self):
        return (CAMPAIGN3_WORKER_ROOT, CAMPAIGN4_WORKER_ROOT)

    def _findOrphanWorkers(self):
        orphans = []
        for root in self._workerRoots():
            orphans.extend(find_orphan_workers(root))
        return orphans

    def _handleOrphanWorkers(self, parent=None):
        """Detect leaked GPU workers; offer to terminate before a queue run.

        Returns True when it is safe to start the queue.
        """
        orphans = self._findOrphanWorkers()
        if not orphans:
            return True
        names = ", ".join(
            f"{entry.get('runId', '?')} (pid {entry.get('pid', '?')})"
            for entry in orphans
        )
        if not messagebox.askyesno(
            "Orphaned Workers Detected",
            f"{len(orphans)} campaign worker(s) from a previous session are "
            f"still training on the GPU:\n\n{names}\n\n"
            "Starting the queue now would compete with them for GPU memory. "
            "Terminate them and start the queue?",
            parent=parent,
        ):
            self.logMessage(
                "Queue start cancelled: orphaned workers are still running "
                f"({names})."
            )
            return False
        terminate_orphan_workers(orphans)
        self.logMessage(
            f"Terminated {len(orphans)} orphaned worker(s): {names}. "
            "Their stopped configs remain queued."
        )
        return True

    def _loadQueueState(self):
        try:
            if os.path.exists(self._queueStatePath):
                with open(self._queueStatePath, 'r') as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    self.configQueue = [
                        self._applyCampaign4PerformanceProfile(dict(cfg))
                        if int(cfg.get("campaignVersion", 0)) == 4
                        else dict(cfg)
                        for cfg in loaded
                        if isinstance(cfg, dict)
                    ]
        except Exception as e:
            self.logMessage(f"WARNING: could not load queue state: {e}")
        self._updateQueueButton()

    def _refreshQueuedCampaign4Profiles(self):
        """Apply the current validated backend to persisted Campaign 4 items."""
        changed = 0
        with self._queueLock:
            for index, config in enumerate(self.configQueue):
                if int(config.get("campaignVersion", 0)) != 4:
                    continue
                profiled = applyCampaign4PerformanceProfile(
                    config,
                    self._workerProfile4,
                )
                if profiled != config:
                    self.configQueue[index] = profiled
                    changed += 1
            if changed:
                self._saveQueueState()
        return changed

    def _updateQueueButton(self):
        n = len(self.configQueue)
        if self.queueButton is not None:
            self.queueButton.config(text=f"Queue ({n})")
        self._refreshQueueEstimate()

    def _appendCampaign3Preset(self, presetKey, parent=None, refreshCallback=None):
        try:
            configs = buildCampaign3Preset(presetKey)
        except Exception as error:
            messagebox.showerror(
                "Campaign 3",
                str(error),
                parent=parent,
            )
            return 0

        with self._queueLock:
            queuedRunIds = {
                config.get("runId")
                for config in self.configQueue
                if config.get("runId")
            }
            added = 0
            skippedQueued = 0
            skippedCompleted = 0
            for config in configs:
                runId = config["runId"]
                if runId in queuedRunIds:
                    skippedQueued += 1
                    continue
                if isCampaign3Completed(config):
                    skippedCompleted += 1
                    continue
                self.configQueue.append(dict(config))
                queuedRunIds.add(runId)
                added += 1
            self._saveQueueState()

        if refreshCallback is not None:
            refreshCallback()
        self._updateQueueButton()
        label = CAMPAIGN3_PRESET_LABELS.get(presetKey, presetKey)
        self._setStatus(
            f"{label}: added {added}; skipped {skippedQueued} queued and "
            f"{skippedCompleted} completed."
        )
        if parent is not None:
            messagebox.showinfo(
                "Campaign 3",
                f"{label}\n\nAdded: {added}\n"
                f"Already queued: {skippedQueued}\n"
                f"Already completed: {skippedCompleted}",
                parent=parent,
            )
        return added

    def _replaceWithCampaign3Preset(
        self,
        presetKey,
        parent=None,
        refreshCallback=None,
    ):
        if self.isRunning:
            messagebox.showwarning(
                "Queue Running",
                "Stop the active queue before replacing it.",
                parent=parent,
            )
            return 0
        try:
            configs = buildCampaign3Preset(presetKey)
        except Exception as error:
            messagebox.showerror("Campaign 3", str(error), parent=parent)
            return 0

        pending = [config for config in configs if not isCampaign3Completed(config)]
        completed = len(configs) - len(pending)
        with self._queueLock:
            existingCount = len(self.configQueue)
        prompt = (
            f"Replace the current {existingCount}-item queue with the missing "
            f"runs from {CAMPAIGN3_PRESET_LABELS.get(presetKey, presetKey)}?\n\n"
            f"Preset total: {len(configs)}\n"
            f"Already completed: {completed}\n"
            f"Will be queued: {len(pending)}\n\n"
            "Existing result files are not changed."
        )
        if existingCount and not messagebox.askyesno(
            "Replace Queue",
            prompt,
            parent=parent,
        ):
            return 0

        with self._queueLock:
            self.configQueue = [dict(config) for config in pending]
            self._saveQueueState()
        if refreshCallback is not None:
            refreshCallback()
        self._updateQueueButton()
        self._setStatus(
            f"Core confirmation queue loaded: {len(pending)} pending, "
            f"{completed} completed."
        )
        messagebox.showinfo(
            "Campaign 3",
            f"Loaded {len(pending)} pending runs.\n"
            f"Reused {completed} completed results.\n\n"
            f"{self._queueEtaVar.get()}",
            parent=parent,
        )
        return len(pending)

    def _requireCampaign4Freeze(self, parent=None):
        if isCampaign4ConfirmationFrozen(execution_profile=self._workerProfile4):
            return True
        completion = campaign4DiagnosticCompletion(
            execution_profile=self._workerProfile4
        )
        state = loadCampaign4State()
        if completion["complete"]:
            reason = (
                "The diagnostic sweep is complete, but the freeze record is "
                "missing or no longer matches the exact Campaign 4 method/code."
            )
        else:
            reason = (
                f"Only {completion['completed']}/{completion['expected']} "
                "Merged+CART diagnostic runs are complete."
            )
            if completion.get("sourceMismatchRunIds"):
                reason += (
                    f" {len(completion['sourceMismatchRunIds'])} artifact(s) "
                    "were produced by stale Campaign 4 source files."
                )
        if state.get("status") == "frozen":
            reason += " The existing freeze record is stale and must be reviewed."
        messagebox.showwarning(
            "Campaign 4 Confirmation Locked",
            f"{reason}\n\nComplete all diagnostics, review their plots, then "
            "use 'Freeze Campaign 4 method' before queuing confirmation runs.",
            parent=parent,
        )
        return False

    def _freezeCampaign4Method(self, parent=None, refreshCallback=None):
        if self.isRunning:
            messagebox.showwarning(
                "Queue Running",
                "Stop the active queue before freezing the Campaign 4 method.",
                parent=parent,
            )
            return False
        completion = campaign4DiagnosticCompletion(
            execution_profile=self._workerProfile4
        )
        if not completion["complete"]:
            mismatchText = (
                f"\nStale-source artifacts: "
                f"{len(completion.get('sourceMismatchRunIds', []))}."
                if completion.get("sourceMismatchRunIds")
                else ""
            )
            messagebox.showwarning(
                "Diagnostics Incomplete",
                "Campaign 4 requires all Merged and CART diagnostics before "
                f"freezing. Completed: {completion['completed']}/"
                f"{completion['expected']}.{mismatchText}",
                parent=parent,
            )
            return False
        if isCampaign4ConfirmationFrozen(execution_profile=self._workerProfile4):
            state = loadCampaign4State()
            messagebox.showinfo(
                "Campaign 4 Already Frozen",
                f"The current method contract is frozen at "
                f"{state.get('frozenAt', 'an unknown time')}.\n\n"
                f"Freeze record: {CAMPAIGN4_STATE}",
                parent=parent,
            )
            return True
        rationale = simpledialog.askstring(
            "Freeze Campaign 4 Method",
            "Record why this adaptive controller is accepted for confirmation. "
            "This must be based only on diagnostic seed 2025:",
            parent=parent,
        )
        if rationale is None:
            return False
        rationale = rationale.strip()
        if not rationale:
            messagebox.showerror(
                "Rationale Required",
                "Enter a non-empty diagnostic decision rationale.",
                parent=parent,
            )
            return False
        margin = simpledialog.askfloat(
            "Predeclare Non-Inferiority Margin",
            "Enter the advisor-approved accuracy margin before viewing "
            "confirmation results (0.05 means five percentage points):",
            initialvalue=0.05,
            minvalue=0.0,
            maxvalue=1.0,
            parent=parent,
        )
        if margin is None:
            return False
        if not messagebox.askyesno(
            "Confirm Method Freeze",
            "Freeze the exact Campaign 4 adaptive method, engine source, "
            "diagnostic run identities, confirmation matrix, and declared "
            f"margin ({margin:.4f})?\n\nSubsequent code or protocol changes "
            "invalidate this freeze record.",
            parent=parent,
        ):
            return False
        try:
            state = freezeCampaign4State(
                rationale=rationale,
                noninferiority_margin=margin,
                execution_profile=self._workerProfile4,
            )
        except Exception as error:
            messagebox.showerror(
                "Campaign 4 Freeze Failed",
                str(error),
                parent=parent,
            )
            return False
        if refreshCallback is not None:
            refreshCallback()
        self.logMessage(
            f"Campaign 4 method frozen at {state['frozenAt']}; "
            f"non-inferiority margin={margin:.4f}; state={CAMPAIGN4_STATE}"
        )
        self._setStatus("Campaign 4 method frozen; confirmation is unlocked.")
        messagebox.showinfo(
            "Campaign 4 Frozen",
            f"Confirmation presets are now unlocked.\n\n"
            f"Freeze record: {CAMPAIGN4_STATE}",
            parent=parent,
        )
        return True

    def _appendCampaign4Preset(self, presetKey, parent=None, refreshCallback=None):
        if presetKey in (
            "main_confirmation",
            "non_iid_confirmation",
            "iid_confirmation",
        ) and not self._requireCampaign4Freeze(parent=parent):
            return 0
        try:
            configs = [
                self._applyCampaign4PerformanceProfile(dict(config))
                for config in buildCampaign4Preset(presetKey)
            ]
        except Exception as error:
            messagebox.showerror("Campaign 4", str(error), parent=parent)
            return 0

        with self._queueLock:
            queuedRunIds = {
                config.get("runId")
                for config in self.configQueue
                if config.get("runId")
            }
            added = 0
            skippedQueued = 0
            skippedCompleted = 0
            for config in configs:
                runId = config["runId"]
                if runId in queuedRunIds:
                    skippedQueued += 1
                    continue
                if isCampaign4Completed(config):
                    skippedCompleted += 1
                    continue
                self.configQueue.append(dict(config))
                queuedRunIds.add(runId)
                added += 1
            if not self.isRunning:
                self.configQueue = self._configsByEstimatedDuration(self.configQueue)
            self._saveQueueState()

        if refreshCallback is not None:
            refreshCallback()
        self._updateQueueButton()
        label = CAMPAIGN4_PRESET_LABELS.get(presetKey, presetKey)
        self._setStatus(
            f"{label}: added {added}; skipped {skippedQueued} queued and "
            f"{skippedCompleted} completed."
        )
        if parent is not None:
            messagebox.showinfo(
                "Campaign 4",
                f"{label}\n\nAdded: {added}\n"
                f"Already queued: {skippedQueued}\n"
                f"Already completed: {skippedCompleted}",
                parent=parent,
            )
        return added

    def _replaceWithCampaign4Preset(
        self,
        presetKey,
        parent=None,
        refreshCallback=None,
    ):
        if presetKey in (
            "main_confirmation",
            "non_iid_confirmation",
            "iid_confirmation",
        ) and not self._requireCampaign4Freeze(parent=parent):
            return 0
        if self.isRunning:
            messagebox.showwarning(
                "Queue Running",
                "Stop the active queue before replacing it.",
                parent=parent,
            )
            return 0
        try:
            configs = [
                self._applyCampaign4PerformanceProfile(dict(config))
                for config in buildCampaign4Preset(presetKey)
            ]
        except Exception as error:
            messagebox.showerror("Campaign 4", str(error), parent=parent)
            return 0
        pending = [config for config in configs if not isCampaign4Completed(config)]
        completed = len(configs) - len(pending)
        with self._queueLock:
            existingCount = len(self.configQueue)
        prompt = (
            f"Replace the current {existingCount}-item queue with pending runs "
            f"from {CAMPAIGN4_PRESET_LABELS.get(presetKey, presetKey)}?\n\n"
            f"Preset total: {len(configs)}\nAlready completed: {completed}\n"
            f"Will be queued: {len(pending)}\n\nExisting results are not changed."
        )
        if existingCount and not messagebox.askyesno(
            "Replace Queue", prompt, parent=parent
        ):
            return 0
        with self._queueLock:
            self.configQueue = self._configsByEstimatedDuration(
                [dict(config) for config in pending]
            )
            self._saveQueueState()
        if refreshCallback is not None:
            refreshCallback()
        self._updateQueueButton()
        messagebox.showinfo(
            "Campaign 4",
            f"Loaded {len(pending)} pending runs.\n"
            f"Reused {completed} completed results.\n\n{self._queueEtaVar.get()}",
            parent=parent,
        )
        return len(pending)

    def _applyCampaign4PerformanceProfile(self, config):
        profiled = applyCampaign4PerformanceProfile(config, self._workerProfile4)
        config.clear()
        config.update(profiled)
        return config

    def _benchmarkCampaign4Profiles(self, parent=None):
        if self.isRunning:
            messagebox.showwarning(
                "Busy",
                "Stop the current experiment or queue before profiling.",
                parent=parent,
            )
            return
        if not messagebox.askyesno(
            "Benchmark Campaign 4 GPU",
            "Benchmark exact full-batch float32/BF16 execution, allocator choices, "
            "and memory-safe two-lane combinations across standard, SS, EBM, "
            "and CART+SS+EBM paths?\n\nA faster BF16 candidate must also pass two "
            "100-round no-save canaries, so a complete first-time profile can take "
            "roughly 2–3 hours. Two lanes are enabled only when fingerprints, "
            "finite values, VRAM headroom, and measured throughput all pass.",
            parent=parent,
        ):
            return
        self.isRunning = True
        self.runButton.config(state=tk.DISABLED)
        self.runAllButton.config(state=tk.DISABLED)
        self.stopButton.config(state=tk.NORMAL)
        self._setStatus("Benchmarking Campaign 4 GPU profiles…")
        self.currentThread = threading.Thread(
            target=self._runCampaign4BenchmarkThread,
            daemon=True,
        )
        self.currentThread.start()

    def _runCampaign4BenchmarkThread(self):
        try:
            projectRoot = Path(__file__).resolve().parents[1]
            def runCommand(command):
                self._benchmarkProcess = subprocess.Popen(
                    command,
                    cwd=projectRoot,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                if self._benchmarkProcess.stdout is not None:
                    for line in self._benchmarkProcess.stdout:
                        self.root.after(0, self.logMessage, line.rstrip())
                return self._benchmarkProcess.wait()

            returnCode = runCommand(
                [
                    sys.executable,
                    str(projectRoot / "scripts" / "benchmark_campaign4.py"),
                    "--gpu-memory-mb",
                    "7600",
                    "--include-cuda-malloc-async",
                    "--include-mixed-bfloat16",
                ]
            )
            if returnCode == 0:
                pendingProfile = load_worker_profile(
                    CAMPAIGN4_PERFORMANCE_PROFILE
                )
                if pendingProfile.get("status") == "requires_full_round_validation":
                    returnCode = runCommand(
                        [
                            sys.executable,
                            str(
                                projectRoot
                                / "scripts"
                                / "validate_campaign4_precision.py"
                            ),
                            "--gpu-memory-mb",
                            "7600",
                        ]
                    )
            if returnCode == 0:
                returnCode = runCommand(
                    [
                        sys.executable,
                        str(
                            projectRoot
                            / "scripts"
                            / "benchmark_campaign4_lanes.py"
                        ),
                        "--calibration-memory-mb",
                        "7600",
                    ]
                )
            self._workerProfile4 = load_worker_profile(
                CAMPAIGN4_PERFORMANCE_PROFILE
            )
            updatedQueueItems = self._refreshQueuedCampaign4Profiles()
            self._runtimeEstimator.reload()
            self._runtimeEstimator.set_execution_profile(self._workerProfile4)
            self.root.after(0, self._refreshQueueEstimate)
            if returnCode != 0:
                raise RuntimeError(
                    f"Campaign 4 GPU benchmark exited with code {returnCode}."
                )
            selected = self._workerProfile4.get("selectedProfile") or {}
            execution = self._workerProfile4.get("concurrencyProfile") or {}
            self.root.after(
                0,
                messagebox.showinfo,
                "Campaign 4 GPU Benchmark",
                "Validated profile: "
                f"{selected.get('profileId', 'none')}. "
                f"Execution lanes: {execution.get('recommendedLanes', 1)}. "
                f"Updated pending Campaign 4 queue items: {updatedQueueItems}.",
            )
        except Exception as error:
            self.root.after(0, self.logMessage, f"CAMPAIGN 4 BENCHMARK ERROR: {error}")
            self.root.after(
                0,
                messagebox.showerror,
                "Campaign 4 GPU Benchmark",
                str(error),
            )
        finally:
            self._benchmarkProcess = None
            self.root.after(0, self._onRunFinished)

    def _benchmarkCampaignWorkers(self, parent=None):
        if self.isRunning:
            messagebox.showwarning(
                "Busy",
                "Stop the current experiment or queue before benchmarking.",
                parent=parent,
            )
            return
        if not messagebox.askyesno(
            "Benchmark GPU Lanes",
            "Run one warm-up, two sequential pilots, and two concurrent pilots?\n\n"
            "The pilots use the existing EBM equation, batch size 512, and two "
            "rounds. They are not saved as experiment results. The GUI enables "
            "two lanes only if outputs match and throughput improves by at least 1.4x.",
            parent=parent,
        ):
            return

        self.isRunning = True
        self.runButton.config(state=tk.DISABLED)
        self.runAllButton.config(state=tk.DISABLED)
        self.stopButton.config(state=tk.NORMAL)
        self._setStatus("Benchmarking isolated GPU workers…")
        self.currentThread = threading.Thread(
            target=self._runWorkerBenchmarkThread,
            daemon=True,
        )
        self.currentThread.start()

    def _runWorkerBenchmarkThread(self):
        try:
            command = [
                sys.executable,
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scripts",
                    "benchmark_campaign_workers.py",
                ),
            ]
            self._benchmarkProcess = subprocess.Popen(
                command,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if self._benchmarkProcess.stdout is not None:
                for line in self._benchmarkProcess.stdout:
                    self.root.after(0, self.logMessage, line.rstrip())
            returnCode = self._benchmarkProcess.wait()
            self._workerProfile = load_worker_profile()
            self._runtimeEstimator.reload()
            self.root.after(0, self._refreshQueueEstimate)
            if returnCode != 0:
                raise RuntimeError(
                    f"GPU worker benchmark exited with code {returnCode}."
                )
            settings = self._workerSettings()
            message = (
                f"Benchmark complete. The queue will use "
                f"{settings['lanes']} isolated GPU lane"
                f"{'s' if settings['lanes'] != 1 else ''}."
            )
            self.root.after(
                0,
                messagebox.showinfo,
                "GPU Worker Benchmark",
                message,
            )
        except Exception as error:
            self.root.after(0, self.logMessage, f"WORKER BENCHMARK ERROR: {error}")
            self.root.after(
                0,
                messagebox.showerror,
                "GPU Worker Benchmark",
                str(error),
            )
        finally:
            self._benchmarkProcess = None
            self.root.after(0, self._onRunFinished)

    def openQueueManager(self):
        configDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        currentConfigDir = os.path.join(configDir, "current")
        campaign4ConfigDir = str(Path(__file__).resolve().parents[1] / CAMPAIGN4_CONFIG_ROOT)

        dlg = tk.Toplevel(self.root)
        dlg.title("Config Queue")
        dlg.configure(bg=BG)
        dlg.resizable(True, True)
        dlg.minsize(920, 620)
        dlg.transient(self.root)

        # ── Top controls ──
        topFrame = tk.Frame(dlg, bg=BG)
        topFrame.pack(fill=tk.X, padx=12, pady=(12, 4))

        ttk.Button(topFrame, text="+ Add Current Config",
                   command=lambda: addCurrent()).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(topFrame, text="+ Add from File",
                   command=lambda: addFromFile()).pack(side=tk.LEFT, padx=4)
        ttk.Button(topFrame, text="+ Add All…",
                   command=lambda: addAllDialog()).pack(side=tk.LEFT, padx=4)
        ttk.Button(topFrame, text="Preset1",
                   command=lambda: loadPreset1()).pack(side=tk.LEFT, padx=(12, 4))

        tk.Label(topFrame, text="Repeat:", bg=BG, fg=HEADER,
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(16, 4))
        repeatVar = tk.IntVar(value=1)
        ttk.Spinbox(topFrame, from_=1, to=50, width=4,
                    textvariable=repeatVar).pack(side=tk.LEFT)
        tk.Label(topFrame, text="times", bg=BG, fg=MUTED,
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(4, 0))

        # ── Campaign-four presets ──
        campaign4Frame = ttk.LabelFrame(
            dlg,
            text="Campaign 4 · Adaptive high-noise study (new isolated results)",
            padding=8,
        )
        campaign4Frame.pack(fill=tk.X, padx=12, pady=(2, 6))
        campaign4Buttons = (
            ("Diagnose non-IID Merged · 33", "diagnostic_non_iid_merged"),
            ("Diagnose non-IID CART · 33", "diagnostic_non_iid_cart"),
            ("Static EBM controls · 60", "static_controls"),
            ("non-IID confirmation · 198", "non_iid_confirmation"),
            ("IID confirmation · 198", "iid_confirmation"),
            ("GPU benchmark · 4", "performance_benchmark"),
        )
        for index, (text, presetKey) in enumerate(campaign4Buttons):
            ttk.Button(
                campaign4Frame,
                text=text,
                style="Preset.TButton",
                command=lambda key=presetKey: self._appendCampaign4Preset(
                    key,
                    parent=dlg,
                    refreshCallback=refreshTree,
                ),
            ).grid(
                row=index // 3,
                column=index % 3,
                sticky=tk.EW,
                padx=3,
                pady=3,
            )
        ttk.Button(
            campaign4Frame,
            text="Main confirmation · Add / Restore Missing · 396",
            style="Queue.TButton",
            command=lambda: self._appendCampaign4Preset(
                "main_confirmation",
                parent=dlg,
                refreshCallback=refreshTree,
            ),
        ).grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=3, pady=3)
        ttk.Button(
            campaign4Frame,
            text="Replace with Main",
            command=lambda: self._replaceWithCampaign4Preset(
                "main_confirmation",
                parent=dlg,
                refreshCallback=refreshTree,
            ),
        ).grid(row=2, column=2, sticky=tk.EW, padx=3, pady=3)
        ttk.Button(
            campaign4Frame,
            text="Freeze Campaign 4 method",
            style="Queue.TButton",
            command=lambda: self._freezeCampaign4Method(
                parent=dlg,
                refreshCallback=refreshTree,
            ),
        ).grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=3, pady=(4, 0))
        ttk.Button(
            campaign4Frame,
            text="Profile GPU + lanes",
            command=lambda: self._benchmarkCampaign4Profiles(parent=dlg),
        ).grid(row=3, column=2, sticky=tk.E, padx=3, pady=(4, 0))
        for column in range(3):
            campaign4Frame.columnconfigure(column, weight=1)

        # ── Campaign-three presets ──
        campaignFrame = ttk.LabelFrame(
            dlg,
            text="Campaign 3 R2 · Hidden attack study",
            padding=8,
        )
        campaignFrame.pack(fill=tk.X, padx=12, pady=(2, 6))
        ttk.Button(
            campaignFrame,
            text="Paper Core 101 · Add / Restore Missing",
            style="Queue.TButton",
            command=lambda: self._appendCampaign3Preset(
                "core_confirmation",
                parent=dlg,
                refreshCallback=refreshTree,
            ),
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky=tk.EW,
            padx=3,
            pady=3,
        )
        ttk.Button(
            campaignFrame,
            text="Replace with Paper Core",
            command=lambda: self._replaceWithCampaign3Preset(
                "core_confirmation",
                parent=dlg,
                refreshCallback=refreshTree,
            ),
        ).grid(
            row=0,
            column=2,
            sticky=tk.EW,
            padx=3,
            pady=3,
        )
        campaignButtons = (
            ("CART σ0.2 Refinement R2 · 6", "cart_low_noise_refinement"),
            ("Low-Noise Repair R2 · 6", "repair"),
            ("Calibration R2 · 49", "calibration"),
            ("Merged Core R2 · 99", "merged_core"),
            ("CART Add-on R2 · 99", "cart_addon"),
            ("CART Non-IID Controls R2 · 9", "cart_controls"),
            ("IID Controls R2 · 48", "iid_controls"),
            ("Full Campaign R2 · 246", "full_campaign"),
        )
        for index, (text, presetKey) in enumerate(campaignButtons):
            ttk.Button(
                campaignFrame,
                text=text,
                style="Preset.TButton",
                command=lambda key=presetKey: self._appendCampaign3Preset(
                    key,
                    parent=dlg,
                    refreshCallback=refreshTree,
                ),
            ).grid(
                row=1 + index // 3,
                column=index % 3,
                sticky=tk.EW,
                padx=3,
                pady=3,
            )
        for column in range(3):
            campaignFrame.columnconfigure(column, weight=1)
        buttonRows = 1 + (len(campaignButtons) + 2) // 3

        ttk.Label(
            campaignFrame,
            textvariable=self._workerProfileVar,
            foreground=MUTED,
            font=("Segoe UI", 8),
        ).grid(
            row=buttonRows,
            column=0,
            columnspan=2,
            sticky=tk.W,
            padx=3,
            pady=(4, 0),
        )
        ttk.Button(
            campaignFrame,
            text="Benchmark 1 vs 2 GPU lanes",
            command=lambda: self._benchmarkCampaignWorkers(parent=dlg),
        ).grid(
            row=buttonRows,
            column=2,
            sticky=tk.E,
            padx=3,
            pady=(4, 0),
        )
        ttk.Label(
            campaignFrame,
            textvariable=self._queueEtaVar,
            foreground=INFO_CLR,
            font=("Segoe UI", 8, "bold"),
            wraplength=930,
        ).grid(
            row=buttonRows + 1,
            column=0,
            columnspan=3,
            sticky=tk.W,
            padx=3,
            pady=(5, 0),
        )

        # ── Queue treeview ──
        treeFrame = tk.Frame(dlg, bg=BG)

        cols = (
            '#',
            'Experiment Name',
            'Approach',
            'Attack',
            'Noise',
            'Estimated Time',
        )
        tree = ttk.Treeview(treeFrame, columns=cols, show='headings',
                            selectmode='extended', height=10)
        tree.heading('#',               text='#',            anchor=tk.CENTER)
        tree.heading('Experiment Name', text='Experiment Name')
        tree.heading('Approach',        text='Approach',     anchor=tk.CENTER)
        tree.heading('Attack',          text='Attack',       anchor=tk.CENTER)
        tree.heading('Noise',           text='Noise',        anchor=tk.CENTER)
        tree.heading('Estimated Time',  text='Estimated Time', anchor=tk.CENTER)
        tree.column('#',               width=35,  stretch=False, anchor=tk.CENTER)
        tree.column('Experiment Name', width=380, stretch=True)
        tree.column('Approach',        width=85,  stretch=False, anchor=tk.CENTER)
        tree.column('Attack',          width=80,  stretch=False, anchor=tk.CENTER)
        tree.column('Noise',           width=60,  stretch=False, anchor=tk.CENTER)
        tree.column('Estimated Time',  width=190, stretch=False, anchor=tk.CENTER)

        sb = ttk.Scrollbar(treeFrame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Bottom bar ──
        botFrame = tk.Frame(dlg, bg=BG)
        botFrame.pack(fill=tk.X, padx=12, pady=(0, 12))

        ttk.Button(botFrame, text="↑ Move Up",   command=lambda: moveUp()).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(botFrame, text="↓ Move Down", command=lambda: moveDown()).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            botFrame,
            text="Move Selected to Top",
            command=lambda: moveSelectedToTop(),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            botFrame,
            text="Shortest First",
            command=lambda: sortShortestFirst(),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(botFrame, text="✕ Remove Selected", command=lambda: removeSelected()).pack(side=tk.LEFT, padx=2)
        ttk.Button(botFrame, text="Clear All",   command=lambda: clearAll()).pack(side=tk.LEFT, padx=(12, 2))

        countLabel = tk.Label(botFrame, text="0 items", bg=BG, fg=MUTED,
                              font=('Segoe UI', 9))
        countLabel.pack(side=tk.LEFT, padx=12)

        ttk.Button(botFrame, text="Run Queue", style='Queue.TButton',
                   command=lambda: runQueueAndClose()).pack(side=tk.RIGHT)

        # Reserve the action bar before the resizable table receives space.
        botFrame.pack_forget()
        botFrame.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(0, 12))
        treeFrame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # ── Helpers ──
        def _attackSummary(cfg):
            parts = []
            for k, s in [('attackGaussian','Gaussian'),('attackSignFlip','SignFlip'),
                         ('attackHidden','Hidden'),('attackModelPoison','ModelPoison'),
                         ('attackScaling','Scaling'),('attackAlie','ALIE'),
                         ('attackIpm','IPM'),('attackNoiseAmp','NoiseAmp')]:
                if cfg.get(k):
                    parts.append(s)
            return '+'.join(parts) if parts else 'None'

        def refreshTree():
            tree.delete(*tree.get_children())
            with self._queueLock:
                queueSnapshot = list(self.configQueue)
            activeElapsed = (
                self._campaignWorkerPool.active_elapsed_by_run_id()
                if self._campaignWorkerPool is not None
                else {}
            )
            for i, cfg in enumerate(queueSnapshot, 1):
                name    = cfg.get('experimentName', f'Config #{i}')
                approach = cfg.get('approach', '?')
                attack   = _attackSummary(cfg)
                noise    = 'Yes' if cfg.get('useChannelNoise') else 'No'
                estimate = self._runtimeEstimator.estimate(cfg)
                elapsed = activeElapsed.get(str(cfg.get('runId', '')))
                if elapsed is not None:
                    # Actively running: show remaining time, not the full
                    # from-scratch estimate, so this matches the live
                    # "Total remaining" banner instead of double-counting
                    # progress already made.
                    elapsed = max(0.0, float(elapsed))
                    estimatedTime = (
                        f"{format_duration(max(0.0, estimate.seconds - elapsed))} left "
                        f"({format_duration(max(0.0, estimate.low_seconds - elapsed))}–"
                        f"{format_duration(max(0.0, estimate.high_seconds - elapsed))}) · "
                        f"running, {format_duration(elapsed)} elapsed"
                    )
                else:
                    estimatedTime = (
                        f"{format_duration(estimate.seconds)} "
                        f"({format_duration(estimate.low_seconds)}–"
                        f"{format_duration(estimate.high_seconds)})"
                    )
                tree.insert(
                    '',
                    tk.END,
                    iid=str(i-1),
                    values=(
                        i,
                        name,
                        approach,
                        attack,
                        noise,
                        estimatedTime,
                    ),
                )
            countLabel.config(
                text=(
                    f"{len(queueSnapshot)} item"
                    f"{'s' if len(queueSnapshot) != 1 else ''}"
                )
            )
            self._refreshQueueEstimate()
            self._updateQueueButton()

        def addCurrent():
            cfg = self.getConfig()
            with self._queueLock:
                for _ in range(repeatVar.get()):
                    self.configQueue.append(dict(cfg))
                if not self.isRunning:
                    self.configQueue = self._configsByEstimatedDuration(
                        self.configQueue
                    )
                self._saveQueueState()
            refreshTree()

        def loadPreset1():
            try:
                with open(self._queuePreset1Path, 'r') as f:
                    loaded = json.load(f)
                if not isinstance(loaded, list):
                    raise ValueError("Preset1 must contain a list of configs.")
                preset = [dict(cfg) for cfg in loaded if isinstance(cfg, dict)]
            except Exception as e:
                messagebox.showerror("Preset1", f"Could not load Preset1:\n{e}", parent=dlg)
                return

            with self._queueLock:
                existingCount = len(self.configQueue)
            if existingCount:
                ok = messagebox.askyesno(
                    "Load Preset1",
                    f"Replace the current {existingCount}-item queue with {len(preset)} Preset1 configs?",
                    parent=dlg,
                )
                if not ok:
                    return

            with self._queueLock:
                self.configQueue = preset
                self._saveQueueState()
            refreshTree()
            self._setStatus(f"Preset1 loaded: {len(preset)} configs")

        def addFromFile():
            currentApproach = self.approachVar.get() if hasattr(self, 'approachVar') else 'basil'
            pickerDlg = tk.Toplevel(dlg)
            pickerDlg.title("Add Config from File")
            pickerDlg.configure(bg=BG)
            pickerDlg.resizable(True, True)
            pickerDlg.transient(dlg)
            pickerDlg.grab_set()

            splitVar2    = tk.StringVar(value='nonIID')
            approachVar2 = tk.StringVar(value=currentApproach)
            sourceVar2   = tk.StringVar(value='campaign4')

            sourceFrame2 = tk.Frame(pickerDlg, bg=BG)
            sourceFrame2.pack(fill=tk.X, padx=12, pady=(12, 0))
            tk.Label(
                sourceFrame2,
                text="Library:",
                bg=BG,
                fg=HEADER,
                font=('Segoe UI', 9, 'bold'),
            ).pack(side=tk.LEFT, padx=(0, 8))
            for label, value in (
                ("Campaign 4", "campaign4"),
                ("Campaign 3 R2", "current"),
                ("Legacy / Custom", "legacy"),
            ):
                ttk.Radiobutton(
                    sourceFrame2,
                    text=label,
                    variable=sourceVar2,
                    value=value,
                    command=lambda: refreshPicker(),
                ).pack(side=tk.LEFT, padx=6)
            sourceNotice2 = tk.Label(
                pickerDlg,
                text="Current reproducible hidden/noise study configs.",
                bg=BG,
                fg=SUCCESS,
                font=('Segoe UI', 8, 'bold'),
            )
            sourceNotice2.pack(fill=tk.X, padx=12, pady=(4, 0))

            # ── Data split selector ──
            splitFrame2 = tk.Frame(pickerDlg, bg=BG)
            splitFrame2.pack(fill=tk.X, padx=12, pady=(8, 0))
            tk.Label(splitFrame2, text="Data Split:", bg=BG, fg=HEADER,
                     font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 8))
            for sp in ('nonIID', 'IID'):
                ttk.Radiobutton(splitFrame2, text=sp, variable=splitVar2, value=sp,
                                command=lambda: refreshPicker()).pack(side=tk.LEFT, padx=6)

            # ── Approach selector ──
            tabFrame2 = tk.Frame(pickerDlg, bg=BG)
            tabFrame2.pack(fill=tk.X, padx=12, pady=(6, 0))
            tk.Label(tabFrame2, text="Approach:", bg=BG, fg=HEADER,
                     font=('Segoe UI', 9, 'bold')).grid(row=0, column=0, rowspan=2, sticky=tk.W, padx=(0, 8))

            apCountLabels2 = {}
            _apList2 = ('basil', 'noisy', 'merged', 'cart')
            for _idx2, ap in enumerate(_apList2):
                _r2, _c2 = divmod(_idx2, 2)
                lbl_var = tk.StringVar(value=f"{ap.capitalize()}  (?)")
                apCountLabels2[ap] = lbl_var
                ttk.Radiobutton(tabFrame2, textvariable=lbl_var,
                                variable=approachVar2, value=ap,
                                command=lambda: refreshPicker()).grid(row=_r2, column=_c2 + 1, sticky=tk.W, padx=6, pady=1)

            listFrame2 = tk.Frame(pickerDlg, bg=BG)
            listFrame2.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
            sb2 = ttk.Scrollbar(listFrame2, orient=tk.VERTICAL)
            sb2.pack(side=tk.RIGHT, fill=tk.Y)
            listbox2 = tk.Listbox(listFrame2, yscrollcommand=sb2.set, selectmode=tk.EXTENDED,
                                  activestyle='dotbox', font=('Segoe UI', 9),
                                  bg=PANEL_BG, fg=HEADER,
                                  selectbackground=ACCENT, selectforeground='white',
                                  relief=tk.FLAT, highlightthickness=1, highlightcolor=BORDER)
            listbox2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb2.config(command=listbox2.yview)

            botFrame2 = tk.Frame(pickerDlg, bg=BG)
            botFrame2.pack(fill=tk.X, padx=12, pady=(0, 12))
            countLabel2 = tk.Label(botFrame2, text="0 files", bg=BG, fg=MUTED,
                                   font=('Segoe UI', 9))
            countLabel2.pack(side=tk.LEFT)
            ttk.Button(botFrame2, text="Cancel", command=pickerDlg.destroy).pack(side=tk.RIGHT, padx=(4, 0))
            ttk.Button(botFrame2, text="Add Selected to Queue", style='Run.TButton',
                       command=lambda: onPickAdd()).pack(side=tk.RIGHT, padx=(0, 4))

            def refreshPicker():
                sp = splitVar2.get()
                ap = approachVar2.get()
                rootDir = {
                    'campaign4': campaign4ConfigDir,
                    'current': currentConfigDir,
                    'legacy': configDir,
                }[sourceVar2.get()]
                if sourceVar2.get() == 'campaign4':
                    sourceNotice2.config(
                        text=(
                            "Campaign 4 adaptive/static study configs; outputs stay in results4."
                        ),
                        fg=SUCCESS,
                    )
                elif sourceVar2.get() == 'current':
                    sourceNotice2.config(
                        text=(
                            "Current reproducible hidden/noise study configs "
                            "with explicit seeds."
                        ),
                        fg=SUCCESS,
                    )
                else:
                    sourceNotice2.config(
                        text=(
                            "Legacy/custom archive; values may not match the "
                            "current R2 protocol."
                        ),
                        fg=DANGER,
                    )
                # Update counts on all approach labels
                for _ap, lv in apCountLabels2.items():
                    _f = os.path.join(rootDir, sp, _ap)
                    _n = len([x for x in os.listdir(_f) if x.endswith('.json')]) if os.path.isdir(_f) else 0
                    lv.set(f"{_ap.capitalize()}  ({_n})")
                folder = os.path.join(rootDir, sp, ap)
                files = sorted([f for f in os.listdir(folder) if f.endswith('.json')]) \
                        if os.path.isdir(folder) else []
                listbox2.delete(0, tk.END)
                for f in files:
                    listbox2.insert(tk.END, f[:-5])
                countLabel2.config(text=f"{len(files)} file{'s' if len(files) != 1 else ''}")

            def onPickAdd():
                sel = listbox2.curselection()
                if not sel:
                    return
                selectedConfigs = []
                for idx in sel:
                    name = listbox2.get(idx) + '.json'
                    rootDir = {
                        'campaign4': campaign4ConfigDir,
                        'current': currentConfigDir,
                        'legacy': configDir,
                    }[sourceVar2.get()]
                    path = os.path.join(
                        rootDir,
                        splitVar2.get(),
                        approachVar2.get(),
                        name,
                    )
                    try:
                        with open(path) as f:
                            cfg = json.load(f)
                        cfg = self._applyCampaign4PerformanceProfile(cfg)
                        for _ in range(repeatVar.get()):
                            selectedConfigs.append(dict(cfg))
                    except Exception as e:
                        messagebox.showerror("Error", str(e), parent=pickerDlg)
                        return
                selectedConfigs = self._configsByEstimatedDuration(
                    selectedConfigs
                )
                with self._queueLock:
                    self.configQueue.extend(selectedConfigs)
                    if not self.isRunning:
                        self.configQueue = self._configsByEstimatedDuration(
                            self.configQueue
                        )
                    self._saveQueueState()
                refreshTree()
                pickerDlg.destroy()

            listbox2.bind('<Double-Button-1>', lambda e: onPickAdd())
            pickerDlg.bind('<Escape>', lambda e: pickerDlg.destroy())

            pickerDlg.update_idletasks()
            w2, h2 = 580, 460
            x2 = dlg.winfo_x() + (dlg.winfo_width()  - w2) // 2
            y2 = dlg.winfo_y() + (dlg.winfo_height() - h2) // 2
            pickerDlg.geometry(f"{w2}x{h2}+{x2}+{y2}")
            refreshPicker()
            pickerDlg.wait_window()

        def addAllDialog():
            """Small dialog: pick split + approach → bulk-add all configs."""
            adlg = tk.Toplevel(dlg)
            adlg.title("Add All Configs")
            adlg.configure(bg=BG)
            adlg.resizable(False, False)
            adlg.transient(dlg)
            adlg.grab_set()

            splitVarA = tk.StringVar(value='nonIID')
            apVarA    = tk.StringVar(value='cart')
            sourceVarA = tk.StringVar(value='campaign4')

            tk.Label(adlg, text="Library:", bg=BG, fg=HEADER,
                     font=('Segoe UI', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=12, pady=(14, 4))
            sourceFA = tk.Frame(adlg, bg=BG)
            sourceFA.grid(row=0, column=1, sticky=tk.W, padx=4, pady=(14, 4))
            for label, value in (
                ("Campaign 4", "campaign4"),
                ("Campaign 3 R2", "current"),
                ("Legacy / Custom", "legacy"),
            ):
                ttk.Radiobutton(
                    sourceFA,
                    text=label,
                    variable=sourceVarA,
                    value=value,
                ).pack(side=tk.LEFT, padx=6)

            tk.Label(adlg, text="Data Split:", bg=BG, fg=HEADER,
                     font=('Segoe UI', 9, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=12, pady=4)
            splitF = tk.Frame(adlg, bg=BG)
            splitF.grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
            for sp in ('nonIID', 'IID'):
                ttk.Radiobutton(splitF, text=sp, variable=splitVarA, value=sp).pack(side=tk.LEFT, padx=6)

            tk.Label(adlg, text="Approach:", bg=BG, fg=HEADER,
                     font=('Segoe UI', 9, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=12, pady=4)
            apF = tk.Frame(adlg, bg=BG)
            apF.grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)
            for ap in ('basil', 'noisy', 'merged', 'cart'):
                ttk.Radiobutton(apF, text=ap.capitalize(), variable=apVarA, value=ap).pack(side=tk.LEFT, padx=6)

            btnF = tk.Frame(adlg, bg=BG)
            btnF.grid(row=3, column=0, columnspan=2, pady=(8, 14), padx=12)
            ttk.Button(btnF, text="Cancel", command=adlg.destroy).pack(side=tk.RIGHT, padx=(4, 0))
            ttk.Button(btnF, text="Add All to Queue", style='Run.TButton',
                       command=lambda: doAddAll()).pack(side=tk.RIGHT)

            def doAddAll():
                sp = splitVarA.get()
                ap = apVarA.get()
                rootDir = {
                    'campaign4': campaign4ConfigDir,
                    'current': currentConfigDir,
                    'legacy': configDir,
                }[sourceVarA.get()]
                folder = os.path.join(rootDir, sp, ap)
                if not os.path.isdir(folder):
                    messagebox.showerror("Error", f"Folder not found: {sp}/{ap}", parent=adlg)
                    return
                files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])
                if not files:
                    messagebox.showwarning("No Configs", f"No JSON configs in {sp}/{ap}/", parent=adlg)
                    return
                added = 0
                selectedConfigs = []
                errors = []
                for fname in files:
                    try:
                        with open(os.path.join(folder, fname)) as f:
                            cfg = json.load(f)
                        cfg = self._applyCampaign4PerformanceProfile(cfg)
                        for _ in range(repeatVar.get()):
                            selectedConfigs.append(dict(cfg))
                        added += 1
                    except Exception as error:
                        errors.append(f"{fname}: {error}")
                selectedConfigs = self._configsByEstimatedDuration(
                    selectedConfigs
                )
                with self._queueLock:
                    self.configQueue.extend(selectedConfigs)
                    if not self.isRunning:
                        self.configQueue = self._configsByEstimatedDuration(
                            self.configQueue
                        )
                    self._saveQueueState()
                adlg.destroy()
                refreshTree()
                message = (
                    f"Added {added} {sp}/{ap.upper()} config files "
                    "shortest-first."
                )
                if errors:
                    message += f"\n\nSkipped {len(errors)} unreadable files."
                messagebox.showinfo("Added", message, parent=dlg)

            adlg.update_idletasks()
            x = dlg.winfo_x() + (dlg.winfo_width()  - adlg.winfo_width())  // 2
            y = dlg.winfo_y() + (dlg.winfo_height() - adlg.winfo_height()) // 2
            adlg.geometry(f"+{x}+{y}")
            adlg.wait_window()

        def _selectedIdx():
            idxs = _selectedIdxs()
            return idxs[0] if idxs else None

        def _selectedIdxs():
            return sorted(int(iid) for iid in tree.selection())

        def moveUp():
            idx = _selectedIdx()
            if idx is None or idx == 0:
                return
            with self._queueLock:
                self.configQueue[idx], self.configQueue[idx-1] = \
                    self.configQueue[idx-1], self.configQueue[idx]
                self._saveQueueState()
            refreshTree()
            tree.selection_set(str(idx-1))

        def moveDown():
            idx = _selectedIdx()
            if idx is None or idx >= len(self.configQueue) - 1:
                return
            with self._queueLock:
                self.configQueue[idx], self.configQueue[idx+1] = \
                    self.configQueue[idx+1], self.configQueue[idx]
                self._saveQueueState()
            refreshTree()
            tree.selection_set(str(idx+1))

        def moveSelectedToTop():
            idxs = _selectedIdxs()
            if not idxs:
                return
            selectedSet = set(idxs)
            with self._queueLock:
                selected = [
                    config
                    for index, config in enumerate(self.configQueue)
                    if index in selectedSet
                ]
                remaining = [
                    config
                    for index, config in enumerate(self.configQueue)
                    if index not in selectedSet
                ]
                self.configQueue = selected + remaining
                self._saveQueueState()
            refreshTree()
            tree.selection_set(*(str(index) for index in range(len(selected))))

        def sortShortestFirst():
            if self.isRunning:
                messagebox.showwarning(
                    "Queue Running",
                    "Stop the active queue before sorting all remaining items.",
                    parent=dlg,
                )
                return
            with self._queueLock:
                self.configQueue = self._configsByEstimatedDuration(
                    self.configQueue
                )
                self._saveQueueState()
            refreshTree()
            self._setStatus("Queue sorted by estimated duration, shortest first")

        def removeSelected():
            idxs = _selectedIdxs()
            if not idxs:
                return
            with self._queueLock:
                for idx in reversed(idxs):
                    if 0 <= idx < len(self.configQueue):
                        del self.configQueue[idx]
                self._saveQueueState()
            refreshTree()

        def clearAll():
            if self.configQueue and not messagebox.askyesno(
                    "Clear Queue", f"Remove all {len(self.configQueue)} items?", parent=dlg):
                return
            with self._queueLock:
                self.configQueue.clear()
                self._saveQueueState()
            refreshTree()

        def runQueueAndClose():
            if not self.configQueue:
                messagebox.showwarning("Empty Queue", "Queue is empty.", parent=dlg)
                return
            dlg.destroy()
            self.runQueue()

        # Center and populate
        dlg.update_idletasks()
        w, h = 1120, 760
        x = self.root.winfo_x() + (self.root.winfo_width()  - w) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - h) // 2
        dlg.geometry(f"{w}x{h}+{x}+{y}")
        refreshTree()
        tree.bind('<Delete>', lambda e: removeSelected())

        def _liveRefreshTick():
            # Keeps the per-row "Estimated Time" column in sync with the
            # "Total remaining" banner while a queue is running - both are
            # recomputed on the same 1s cadence. RuntimeEstimator.estimate()
            # is cached by runId (see runtime_estimator.py), so redrawing
            # this table every second stays cheap even at a few hundred
            # queued rows -- only entries whose underlying data actually
            # changed cost a real recompute.
            if not dlg.winfo_exists():
                return
            if self.isRunning:
                refreshTree()
            dlg.after(1000, _liveRefreshTick)

        dlg.after(1000, _liveRefreshTick)

        dragState = {"iid": None, "moved": False}

        def beginDrag(event):
            if self.isRunning or tree.identify_region(event.x, event.y) != "cell":
                dragState["iid"] = None
                return
            iid = tree.identify_row(event.y)
            dragState["iid"] = iid or None
            dragState["moved"] = False

        def dragRow(event):
            source = dragState["iid"]
            if source is None or self.isRunning:
                return
            target = tree.identify_row(event.y)
            if not target or target == source:
                return
            tree.move(source, "", tree.index(target))
            tree.configure(cursor="fleur")
            dragState["moved"] = True

        def endDrag(_event):
            source = dragState["iid"]
            moved = dragState["moved"]
            dragState["iid"] = None
            dragState["moved"] = False
            tree.configure(cursor="")
            if source is None or not moved or self.isRunning:
                return
            order = [int(iid) for iid in tree.get_children("")]
            with self._queueLock:
                previous = list(self.configQueue)
                if len(previous) != len(order):
                    refreshTree()
                    return
                self.configQueue = [previous[index] for index in order]
                self._saveQueueState()
            newIndex = order.index(int(source))
            refreshTree()
            tree.selection_set(str(newIndex))
            tree.see(str(newIndex))
            self._setStatus("Queue order changed by drag and drop")

        tree.bind('<ButtonPress-1>', beginDrag, add='+')
        tree.bind('<B1-Motion>', dragRow, add='+')
        tree.bind('<ButtonRelease-1>', endDrag, add='+')

    def _removeQueuedConfig(self, target):
        removed = False
        with self._queueLock:
            for index, config in enumerate(self.configQueue):
                if config is target:
                    del self.configQueue[index]
                    removed = True
                    self._saveQueueState()
                    break
        self.root.after(0, self._updateQueueButton)
        return removed

    def _nextCampaignConfig(
        self,
        activeObjectIds,
        activeRunIds,
        campaignVersion,
        *,
        activeConfigs=(),
        workerSettings=None,
    ):
        with self._queueLock:
            if int(campaignVersion) == 4:
                return selectCampaign4Config(
                    self.configQueue,
                    active_configs=activeConfigs,
                    active_object_ids=activeObjectIds,
                    active_run_ids=activeRunIds,
                    campaign_version=campaignVersion,
                    settings=workerSettings or self._workerSettings(4),
                )
            for config in self.configQueue:
                if id(config) in activeObjectIds:
                    continue
                if (
                    config.get("runId")
                    and config.get("runId") in activeRunIds
                ):
                    continue
                if int(config.get("campaignVersion", 0)) != int(campaignVersion):
                    return None
                return config
        return None

    def _startCampaignLiveRun(self, lane, config):
        self._campaignLiveData[lane] = {
            "runId": config.get("runId"),
            "name": config.get("experimentName", "Campaign run"),
            "rounds": [],
            "avg": [],
            "worst": [],
            "updated": time.monotonic(),
        }
        self._refreshLiveChart()

    def _recordCampaignRound(
        self,
        lane,
        config,
        roundNum,
        avgAcc,
        worstAcc,
    ):
        series = self._campaignLiveData.get(lane)
        if series is None or series.get("runId") != config.get("runId"):
            self._startCampaignLiveRun(lane, config)
            series = self._campaignLiveData[lane]
        series["rounds"].append(int(roundNum))
        series["avg"].append(float(avgAcc))
        series["worst"].append(float(worstAcc))
        series["updated"] = time.monotonic()
        self._refreshLiveChart()

    def _logCampaignConfiguration(self, lane, config, workerSettings):
        split = "Non-IID" if config.get("nonIID", True) else "IID"
        if config.get("nonIID", True):
            split += f" (Dirichlet alpha={config.get('dirichletAlpha', 0.2)})"
        hasHidden = bool(config.get("attackHidden"))
        hasNoise = bool(config.get("useChannelNoise"))
        useSs = bool(
            config.get("snapshotSelection", config.get("useBasil", False))
        )
        useEbm = bool(
            hasNoise
            and config.get("noiseMitigation") == "ebm"
            and float(config.get("ebmLambda", 0.0)) > 0.0
        )
        mitigation = (
            "SS + EBM"
            if useSs and useEbm
            else "SS"
            if useSs
            else "EBM"
            if useEbm
            else "No Mitigation"
        )
        environment = {
            "clean": "Clean Environment",
            "clean_ceiling": "Ideal Full-Consensus Clean Ceiling",
            "hidden": "Hidden Byzantine Attack",
            "noise": "Channel Noise",
            "hidden_noise": "Hidden Byzantine Attack + Channel Noise",
        }.get(config.get("environment"), str(config.get("environment", "unknown")))

        self.logMessage("=" * 80)
        campaignVersion = int(config.get("campaignVersion", 3))
        self.logMessage(
            f"[lane {lane + 1}] STARTING CAMPAIGN {campaignVersion} EXPERIMENT"
        )
        self.logMessage("=" * 80)
        self.logMessage("Configuration:")
        self.logMessage(
            f"  Experiment: {config.get('experimentName', '(unnamed)')}"
        )
        self.logMessage(f"  Run ID:     {config.get('runId', '(none)')}")
        self.logMessage(
            f"  Dataset:    {config.get('dataset', 'cifar10')}  |  Split: {split}"
        )
        self.logMessage(
            f"  Approach:   {str(config.get('approach', '?')).upper()}"
            f"  |  Environment: {environment}"
        )
        self.logMessage(
            f"  Seed: {config.get('seed')}  |  Nodes: {config.get('nNodes')}"
            f"  |  Rounds: {config.get('nRounds')}"
        )
        self.logMessage(
            f"  Local work: epochs={config.get('localEpochs')}, "
            f"steps/epoch={config.get('stepsPerEpoch')}, "
            f"batch={config.get('batchSize')}"
        )
        self.logMessage(
            f"  LR: {config.get('learningRate')}  |  "
            f"Momentum: {config.get('momentum')}  |  "
            f"Decay: {bool(config.get('useLrDecay'))}"
        )
        self.logMessage(
            f"  Aggregation: {config.get('aggregationMode')}  |  "
            f"Mitigation: {mitigation}"
        )
        self.logMessage(
            f"  Hidden attack: {'active' if hasHidden else 'inactive'}"
            + (
                f", nodes={config.get('attackerIds')}, "
                f"starts round {config.get('attackHiddenStart')}"
                if hasHidden
                else ""
            )
        )
        self.logMessage(
            f"  Channel noise: {'active' if hasNoise else 'inactive'}"
            + (
                f", sigma={float(config.get('channelNoiseSigma', 0.0)):.1f}, "
                f"starts round {config.get('channelNoiseStart', 0)}"
                if hasNoise
                else ""
            )
        )
        self.logMessage(
            f"  Snapshot Selection: {'active' if useSs else 'inactive'}"
            + (
                f", memory={config.get('basilMemorySize')}"
                if useSs
                else ""
            )
        )
        if useEbm:
            if campaignVersion == 4:
                self.logMessage(
                    f"  EBM: mode={config.get('ebmMode')}, initial coefficient="
                    f"{float(config.get('ebmInitialCoefficient', 0.0)):.8g}, "
                    f"bounded=[{float(config.get('adaptiveEbmCoefficientMin', 0.0)):.1g}, "
                    f"{float(config.get('adaptiveEbmCoefficientMax', 0.0)):.3g}]"
                )
            else:
                self.logMessage(
                    f"  EBM: lambda={float(config.get('ebmLambda', 0.0)):.8g}, "
                    f"lambda*sigma^2="
                    f"{float(config.get('ebmTargetCoefficient', 0.0)):.8g}"
                )
        if config.get("approach") == "cart":
            self.logMessage(
                f"  CART: gamma={float(config.get('distillStrength', 0.0)):.8g}, "
                f"verify threshold={config.get('verifyThreshold')}"
            )
        self.logMessage(
            f"  Worker: isolated GPU lane {lane + 1}, "
            f"memory cap={workerSettings.get('activeMemoryLimitMb', workerSettings['gpuMemoryLimitMb'])} MB"
        )
        if campaignVersion == 4:
            self.logMessage(
                f"  GPU profile: microbatch={config.get('internalMicroBatchSize')}, "
                f"precision={config.get('precisionProfile')}, "
                f"XLA={bool(config.get('jitCompile'))}, "
                f"allocator={config.get('gpuAllocator', 'bfc')}, "
                f"optimizer state={config.get('optimizerStateMode')}"
            )
        self.logMessage("-" * 80)

    def _updateCampaignPoolProgress(self, laneProgress, activeCount, latestLane):
        if not laneProgress:
            return
        summaries = []
        fractions = []
        for lane in sorted(laneProgress):
            roundNum, totalRounds, avgAcc = laneProgress[lane]
            summaries.append(f"L{lane + 1} {roundNum}/{totalRounds}")
            if totalRounds > 0:
                fractions.append(roundNum / totalRounds)
        if fractions:
            self._progressVar.set(100.0 * float(np.mean(fractions)))
        self._roundLabel.config(
            text=f"{activeCount} active · " + " · ".join(summaries)
        )
        if latestLane not in laneProgress:
            latestLane = max(laneProgress)
        self._accLabel.config(text=f"Acc  {laneProgress[latestLane][2]:.1%}")
        self._etaLabel.config(text=self._queueEtaVar.get())
        self._setStatus(
            f"Campaign queue running · {activeCount} isolated worker"
            f"{'s' if activeCount != 1 else ''}"
        )

    def _drainCampaignWorkerEvents(self, pool, laneProgress, campaignVersion):
        for kind, lane, config, payload in pool.drain_events():
            name = config.get("experimentName", config.get("runId", "Campaign run"))
            if kind == "line":
                if payload.startswith(("[campaign3 round ", "[campaign4 round ")):
                    continue
                self.root.after(0, self.logMessage, f"[lane {lane + 1}] {payload}")
                continue

            event = payload.get("event")
            if event == "round":
                roundNum = int(payload["round"])
                totalRounds = int(payload["totalRounds"])
                avgAcc = float(payload["averageAccuracy"])
                worstAcc = float(payload["worstAccuracy"])
                laneProgress[lane] = (
                    roundNum,
                    totalRounds,
                    avgAcc,
                )
                self.root.after(
                    0,
                    self._recordCampaignRound,
                    lane,
                    config,
                    roundNum,
                    avgAcc,
                    worstAcc,
                )
                if int(campaignVersion) == 4:
                    self.root.after(
                        0,
                        self.networkView.apply_round,
                        lane,
                        config,
                        dict(payload),
                    )
                self.root.after(
                    0,
                    self._updateCampaignPoolProgress,
                    dict(laneProgress),
                    pool.active_count,
                    lane,
                )
            elif event == "node_update" and int(campaignVersion) == 4:
                self.root.after(
                    0,
                    self.networkView.apply_node_update,
                    lane,
                    config,
                    dict(payload),
                )
            elif event == "preparing":
                self.root.after(
                    0,
                    self.logMessage,
                    f"[lane {lane + 1}] Preparing {name}",
                )
            elif event == "started":
                estimate = self._runtimeEstimator.estimate(config)
                self.root.after(
                    0,
                    self.logMessage,
                    f"[lane {lane + 1}] Started {name} · estimated "
                    f"{format_duration(estimate.seconds)} "
                    f"({format_duration(estimate.low_seconds)}–"
                    f"{format_duration(estimate.high_seconds)})",
                )
            elif event == "completed":
                wallSeconds = float(payload["wallRuntimeSeconds"])
                finalWorst = float(payload.get("finalWorstAccuracy", 0.0))
                self.root.after(
                    0,
                    self.logMessage,
                    f"[lane {lane + 1}] Completed {name}: "
                    f"avg={float(payload['finalAverageAccuracy']):.4f}, "
                    f"worst={finalWorst:.4f}, "
                    f"time={format_duration(wallSeconds)} "
                    f"({wallSeconds:.1f}s)",
                )
                if int(campaignVersion) == 4:
                    metricsPath, runPath, telemetryPath = campaign4ResultPaths(config)
                else:
                    metricsPath, runPath = campaign3ResultPaths(config)
                    telemetryPath = None
                self.root.after(
                    0,
                    self.logMessage,
                    f"  saved metrics: {metricsPath}",
                )
                self.root.after(
                    0,
                    self.logMessage,
                    f"  saved config:  {runPath}",
                )
                if telemetryPath is not None:
                    self.root.after(
                        0,
                        self.logMessage,
                        f"  saved telemetry: {telemetryPath}",
                    )
                    self.root.after(0, self.networkView.finish_run, lane, "completed")
            elif event == "stopped":
                self.root.after(
                    0,
                    self.logMessage,
                    f"[lane {lane + 1}] Stopped {name}; queue item retained.",
                )
                if int(campaignVersion) == 4:
                    self.root.after(0, self.networkView.finish_run, lane, "stopped")
            elif event == "failed":
                self._campaignWorkerFailures[(int(campaignVersion), lane)] = dict(payload)
                self.root.after(
                    0,
                    self.logMessage,
                    f"[lane {lane + 1}] ERROR {name}: {payload.get('error', 'unknown')}",
                )
                if int(campaignVersion) == 4:
                    self.root.after(0, self.networkView.finish_run, lane, "failed")

    def _runIsolatedCampaignBlock(self, campaignVersion):
        campaignVersion = int(campaignVersion)
        if campaignVersion == 4:
            isCompleted = isCampaign4Completed
            workerScript = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "run_campaign4_worker.py"
            )
            workDir = CAMPAIGN4_WORKER_ROOT
        else:
            isCompleted = isCampaign3Completed
            workerScript = None
            workDir = None

        settings = self._workerSettings(campaignVersion)
        poolArguments = {
            "lanes": settings["lanes"],
            "gpu_memory_limit_mb": settings["gpuMemoryLimitMb"],
        }
        if workerScript is not None:
            poolArguments.update(
                {"worker_script": workerScript, "work_dir": workDir}
            )
        pool = CampaignWorkerPool(**poolArguments)
        self._campaignWorkerPool = pool
        completed = 0
        plotConfig = None
        failedMessage = None
        laneProgress = {}
        stopSent = False
        confirmationFrozen = (
            isCampaign4ConfirmationFrozen(execution_profile=self._workerProfile4)
            if campaignVersion == 4
            else True
        )
        if campaignVersion == 4 and settings["lanes"] > 1:
            limits = settings.get("memoryLimitsMb", {})
            memoryText = (
                f"standard cap {limits.get('standard')} MB, "
                f"full-batch EBM cap {limits.get('ebm')} MB, "
                f"max concurrent EBM {settings.get('maxConcurrentEbm', 1)}"
            )
        else:
            memoryText = f"{settings['gpuMemoryLimitMb']} MB cap per worker"
        self.logMessage(
            f"Campaign {campaignVersion} execution: {settings['lanes']} isolated GPU lane"
            f"{'s' if settings['lanes'] != 1 else ''}, {memoryText}."
        )

        try:
            while True:
                self._drainCampaignWorkerEvents(
                    pool, laneProgress, campaignVersion
                )
                completions = pool.poll_finished()
                if completions:
                    self._drainCampaignWorkerEvents(
                        pool, laneProgress, campaignVersion
                    )
                for completion in completions:
                    laneProgress.pop(completion.lane, None)
                    config = completion.config
                    name = config.get("experimentName", config.get("runId"))
                    if completion.return_code == 0 and isCompleted(config):
                        self._removeQueuedConfig(config)
                        completed += 1
                        plotConfig = config
                        self._queueBatchNeedsPlot = True
                        self._runtimeEstimator.reload()
                        if campaignVersion == 3:
                            freeze_calibration_if_ready()
                        # Plot refresh is deliberately deferred to the queue
                        # boundary (see runQueueThread's finally block) rather
                        # than scheduled here per completion: with hundreds of
                        # results on disk, a per-completion live refresh
                        # reloads/regroups the entire results tree on every
                        # single queue item, which dominated wall-clock time
                        # on large batches.
                        sendNotification(
                            "Queue Step Done",
                            f"{name} finished.",
                            priority="high",
                        )
                    elif completion.return_code == 2 or not self.isRunning:
                        self.logMessage(
                            f"[lane {completion.lane + 1}] {name} remains queued."
                        )
                    else:
                        failure = self._campaignWorkerFailures.pop(
                            (campaignVersion, completion.lane), {}
                        )
                        errorText = str(failure.get("error", ""))
                        isOom = campaignVersion == 4 and (
                            bool(failure.get("resourceExhausted"))
                            or failure.get("errorType") == "ResourceExhaustedError"
                            or "ResourceExhausted" in errorText
                            or "OOM" in errorText.upper()
                            or "RESOURCE_EXHAUSTED" in errorText.upper()
                        )
                        if isOom:
                            failedMessage = (
                                f"{name} exceeded its validated full-batch GPU cap in "
                                f"lane {completion.lane + 1}. The unchanged batch-512 "
                                "config remains queued; concurrency has stopped so the "
                                "machine profile can be re-benchmarked."
                            )
                        else:
                            failedMessage = (
                                f"{name} failed in isolated worker lane "
                                f"{completion.lane + 1} (exit {completion.return_code})."
                            )
                        self.logMessage(f"QUEUE ERROR: {failedMessage}")
                        self.isRunning = False

                if not self.isRunning and not stopSent:
                    pool.request_stop()
                    stopSent = True

                launched = False
                while self.isRunning and pool.available_lanes:
                    activeConfigs = pool.active_configs()
                    activeObjectIds = {id(config) for config in activeConfigs}
                    activeRunIds = {
                        config.get("runId")
                        for config in activeConfigs
                        if config.get("runId")
                    }
                    config = self._nextCampaignConfig(
                        activeObjectIds,
                        activeRunIds,
                        campaignVersion,
                        activeConfigs=activeConfigs,
                        workerSettings=settings,
                    )
                    if config is None:
                        break
                    if campaignVersion == 4:
                        if (
                            config.get("phase") == "confirmation"
                            and not confirmationFrozen
                        ):
                            failedMessage = (
                                "Campaign 4 confirmation is locked. Complete all "
                                "66 diagnostics and freeze the method contract first. "
                                "The current queue item was retained."
                            )
                            self.logMessage(f"QUEUE BLOCKED: {failedMessage}")
                            self.isRunning = False
                            break
                        oldRunId = config.get("runId")
                        self._applyCampaign4PerformanceProfile(config)
                        if config.get("runId") != oldRunId:
                            self._saveQueueState()
                    if isCompleted(config):
                        self.logMessage(
                            f"[SKIP] Already completed: "
                            f"{config.get('experimentName', config.get('runId'))}"
                        )
                        self._removeQueuedConfig(config)
                        continue
                    memoryLimit = (
                        campaign4MemoryLimit(config, settings)
                        if campaignVersion == 4
                        else settings["gpuMemoryLimitMb"]
                    )
                    active = pool.launch(
                        config,
                        gpu_memory_limit_mb=memoryLimit,
                    )
                    laneProgress[active.lane] = (
                        0,
                        int(config.get("nRounds", 100)),
                        0.0,
                    )
                    self.root.after(
                        0,
                        self._startCampaignLiveRun,
                        active.lane,
                        config,
                    )
                    self.root.after(
                        0,
                        self._logCampaignConfiguration,
                        active.lane,
                        config,
                        {**settings, "activeMemoryLimitMb": memoryLimit},
                    )
                    if campaignVersion == 4:
                        self.root.after(
                            0,
                            self.networkView.start_run,
                            active.lane,
                            config,
                        )
                    launched = True
                    sendNotification(
                        "Queue Started",
                        f"{config.get('experimentName', config.get('runId'))} has started.",
                        priority="default",
                    )

                now = time.monotonic()
                if now - self._queueLastEstimateRefresh >= 1.0:
                    self._queueLastEstimateRefresh = now
                    self._scheduleQueueEstimateRefresh()

                if not pool.has_active and not launched:
                    break
                time.sleep(0.2)
        finally:
            if pool.has_active:
                pool.request_stop()
                deadline = time.monotonic() + 120.0
                while pool.has_active and time.monotonic() < deadline:
                    self._drainCampaignWorkerEvents(
                        pool, laneProgress, campaignVersion
                    )
                    pool.poll_finished()
                    time.sleep(0.2)
                if pool.has_active:
                    pool.kill_remaining()
                    pool.wait(timeout=10.0)
            self._drainCampaignWorkerEvents(pool, laneProgress, campaignVersion)
            self._campaignWorkerPool = None
            self.root.after(0, self._updateQueueButton)
        return completed, plotConfig, failedMessage

    def runQueue(self):
        if self.isRunning:
            self.openQueueManager()
            return
        if not self.configQueue:
            messagebox.showwarning("Empty Queue", "The config queue is empty.")
            return

        if not self._handleOrphanWorkers():
            return

        with self._queueLock:
            self._saveQueueState()
        self._setQueueActiveSentinel()
        self.runButton.config(state=tk.DISABLED)
        self.runAllButton.config(state=tk.DISABLED)
        self.queueButton.config(state=tk.NORMAL)
        self.stopButton.config(state=tk.NORMAL)
        self.isRunning = True
        self._queueBatchNeedsPlot = False
        self._queueLastEstimateRefresh = 0.0
        self.clearOutput()
        self._liveAccData.clear()
        self._liveWorstData.clear()
        self._refreshLiveChart()
        self.notebook.select(3)
        self._refreshQueueEstimate()

        self.currentThread = threading.Thread(target=self.runQueueThread)
        self.currentThread.start()

    def runQueueThread(self):
        completed = 0
        plottedConfig = None
        try:
            self.logMessage("=" * 80)
            with self._queueLock:
                initialQueued = len(self.configQueue)
            self.logMessage(f"RUN QUEUE: {initialQueued} experiment(s) queued")
            self.logMessage(self._queueEstimateText())
            self.logMessage("=" * 80 + "\n")

            while self.isRunning:
                with self._queueLock:
                    if not self.configQueue:
                        break
                    config = self.configQueue[0]
                    remaining = max(0, len(self.configQueue) - 1)
                self.root.after(0, self._updateQueueButton)

                if not self.isRunning:
                    self.logMessage("\n[STOPPED] Queue run cancelled by user.")
                    break

                campaignVersion = int(config.get("campaignVersion", 0))
                if campaignVersion in (3, 4):
                    blockCompleted, blockPlotConfig, failure = (
                        self._runIsolatedCampaignBlock(campaignVersion)
                    )
                    completed += blockCompleted
                    plottedConfig = blockPlotConfig or plottedConfig
                    if failure or not self.isRunning:
                        break
                    continue

                self.logMessage("=" * 80)
                expName = config.get('experimentName', f'Queue item #{completed + 1}')
                self.logMessage(f"[{completed + 1}] {expName}  ({remaining} still queued)")
                self.logMessage("=" * 80)

                self._trainStartTime  = time.time()
                self._lastRoundEndTime = self._trainStartTime
                self._totalRounds     = config.get('nRounds', 100)
                self._roundTimes.clear()
                self._emaRoundTime  = None
                self._smoothedEta   = None
                self.root.after(1000, self._etaTicker)
                self._liveAccData.clear()
                self._liveWorstData.clear()
                self.root.after(0, self._progressVar.set, 0)
                self.root.after(0, self._roundLabel.config, {'text': f"Round 0/{self._totalRounds}"})

                sendNotification("Queue Started", f"{expName} has started.", priority="default")
                config['aggregationMode'] = self.defaultAggregationMode(config)
                self._executeExperiment(config)

                if not self.isRunning:
                    self.logMessage("\n[STOPPED] Current queue item was left in the queue.")
                    break

                removedCompleted = False
                with self._queueLock:
                    for idx, queuedConfig in enumerate(self.configQueue):
                        if queuedConfig is config:
                            del self.configQueue[idx]
                            removedCompleted = True
                            self._saveQueueState()
                            break
                self.root.after(0, self._updateQueueButton)
                completed += 1
                plottedConfig = config

                sendNotification("Queue Step Done", f"{expName} finished.", priority="high")
                if removedCompleted:
                    self.logMessage(f"\n[{completed}] Done. Removed completed item from queue.\n")
                else:
                    self.logMessage(f"\n[{completed}] Done. Queue item was already removed manually.\n")

            self.logMessage("")
            self.logMessage("=" * 80)
            self.logMessage(f"QUEUE FINISHED: {completed} experiment(s) completed.")
            self.logMessage("=" * 80)
            with self._queueLock:
                remaining = len(self.configQueue)
            sendNotification("Queue Complete", f"{completed} queue item(s) finished, {remaining} remaining.", priority="high")

        except Exception as e:
            self.logMessage(f"\nQUEUE ERROR: {str(e)}")
            self.logMessage("Current queue item was left in the queue.")
            self.logMessage(traceback.format_exc())
            sendNotification("Queue FAILED", str(e), priority="urgent")
        finally:
            with self._queueLock:
                self._saveQueueState()
            if self._queueBatchNeedsPlot and plottedConfig is not None:
                campaignVersion = int(plottedConfig.get("campaignVersion", 0))
                if campaignVersion != 4:
                    self._waitForCampaign3PlotRefresh()
                self.logMessage(
                    f"\nGenerating changed Campaign {campaignVersion} plots "
                    "at the queue boundary…"
                )
                self.generatePlots(
                    showDialog=False,
                    config=plottedConfig,
                    onlyMissing=True,
                )
                self._queueBatchNeedsPlot = False
            self._clearQueueActiveSentinel()
            self.root.after(0, self._onRunFinished)

    def saveConfig(self):
        config   = self.getConfig()
        approach = config.get('approach', 'basil')
        split    = 'nonIID' if config.get('nonIID', True) else 'IID'
        expName  = config.get('experimentName', '').strip()
        default  = expName or f"config_{config['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not default.endswith('.json'):
            default += '.json'
        configDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        initialdir = os.path.join(configDir, split, approach)
        os.makedirs(initialdir, exist_ok=True)
        filepath = filedialog.asksaveasfilename(
            title=f"Save Configuration — {approach.capitalize()}",
            initialdir=initialdir,
            initialfile=default,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.root)
        if not filepath:
            return
        if not filepath.endswith('.json'):
            filepath += '.json'
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        messagebox.showinfo("Saved", f"Configuration saved to:\n{filepath}")
        self._setStatus(f"Config saved: {os.path.basename(filepath)}")

    def loadConfig(self):
        configDir       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        currentConfigDir = os.path.join(configDir, "current")
        currentApproach = self.approachVar.get() if hasattr(self, 'approachVar') else 'basil'

        # ── Custom picker dialog ──────────────────────────────────────────────
        dlg = tk.Toplevel(self.root)
        dlg.title("Load Configuration")
        dlg.configure(bg=BG)
        dlg.resizable(True, True)
        dlg.transient(self.root)
        dlg.grab_set()

        selectedPath = [None]
        splitVar     = tk.StringVar(value='nonIID')
        approachVar  = tk.StringVar(value=currentApproach)
        sourceVar    = tk.StringVar(value='current')

        sourceFrame = tk.Frame(dlg, bg=BG)
        sourceFrame.pack(fill=tk.X, padx=12, pady=(12, 0))
        tk.Label(
            sourceFrame,
            text="Library:",
            bg=BG,
            fg=HEADER,
            font=('Segoe UI', 9, 'bold'),
        ).pack(side=tk.LEFT, padx=(0, 8))
        for label, value in (
            ("Current", "current"),
            ("Legacy / Custom", "legacy"),
        ):
            ttk.Radiobutton(
                sourceFrame,
                text=label,
                variable=sourceVar,
                value=value,
                command=lambda: refreshList(),
            ).pack(side=tk.LEFT, padx=6)

        # ── Data split selector ──
        splitFrame = tk.Frame(dlg, bg=BG)
        splitFrame.pack(fill=tk.X, padx=12, pady=(8, 0))
        tk.Label(splitFrame, text="Data Split:", bg=BG, fg=HEADER,
                 font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 8))
        for sp in ('nonIID', 'IID'):
            ttk.Radiobutton(splitFrame, text=sp, variable=splitVar, value=sp,
                            command=lambda: refreshList()).pack(side=tk.LEFT, padx=6)

        # ── Approach selector with live counts ──
        tabFrame = tk.Frame(dlg, bg=BG)
        tabFrame.pack(fill=tk.X, padx=12, pady=(6, 0))
        tk.Label(tabFrame, text="Approach:", bg=BG, fg=HEADER,
                 font=('Segoe UI', 9, 'bold')).grid(row=0, column=0, rowspan=2, sticky=tk.W, padx=(0, 8))

        apCountLabels = {}
        _apList = ('basil', 'noisy', 'merged', 'cart')
        for _idx, ap in enumerate(_apList):
            _r, _c = divmod(_idx, 2)
            lbl_var = tk.StringVar(value=f"{ap.capitalize()}  (?)")
            apCountLabels[ap] = lbl_var
            ttk.Radiobutton(tabFrame, textvariable=lbl_var, variable=approachVar, value=ap,
                            command=lambda: refreshList()).grid(row=_r, column=_c + 1, sticky=tk.W, padx=6, pady=1)

        # ── File listbox ──
        listFrame = tk.Frame(dlg, bg=BG)
        listFrame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        sb = ttk.Scrollbar(listFrame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(listFrame, yscrollcommand=sb.set, selectmode=tk.SINGLE,
                             activestyle='dotbox', font=('Segoe UI', 9),
                             bg=PANEL_BG, fg=HEADER,
                             selectbackground=ACCENT, selectforeground='white',
                             relief=tk.FLAT, highlightthickness=1, highlightcolor=BORDER)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=listbox.yview)

        # ── Bottom bar ──
        bottomFrame = tk.Frame(dlg, bg=BG)
        bottomFrame.pack(fill=tk.X, padx=12, pady=(0, 12))
        countLabel = tk.Label(bottomFrame, text="0 files", bg=BG, fg=MUTED,
                              font=('Segoe UI', 9))
        countLabel.pack(side=tk.LEFT)
        ttk.Button(bottomFrame, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(bottomFrame, text="Load", style='Run.TButton',
                   command=lambda: onLoad()).pack(side=tk.RIGHT, padx=(0, 4))

        def refreshList():
            sp = splitVar.get()
            ap = approachVar.get()
            rootDir = (
                currentConfigDir
                if sourceVar.get() == 'current'
                else configDir
            )
            # Update counts on all approach labels
            for _ap, lv in apCountLabels.items():
                _f = os.path.join(rootDir, sp, _ap)
                _n = len([x for x in os.listdir(_f) if x.endswith('.json')]) if os.path.isdir(_f) else 0
                lv.set(f"{_ap.capitalize()}  ({_n})")
            folder = os.path.join(rootDir, sp, ap)
            files  = sorted([f for f in os.listdir(folder) if f.endswith('.json')]) \
                     if os.path.isdir(folder) else []
            listbox.delete(0, tk.END)
            for f in files:
                listbox.insert(tk.END, f[:-5])
            countLabel.config(text=f"{len(files)} file{'s' if len(files) != 1 else ''}")

        def onLoad():
            sel = listbox.curselection()
            if not sel:
                return
            name = listbox.get(sel[0]) + '.json'
            rootDir = (
                currentConfigDir
                if sourceVar.get() == 'current'
                else configDir
            )
            selectedPath[0] = os.path.join(
                rootDir,
                splitVar.get(),
                approachVar.get(),
                name,
            )
            dlg.destroy()

        listbox.bind('<Double-Button-1>', lambda e: onLoad())
        dlg.bind('<Return>', lambda e: onLoad())
        dlg.bind('<Escape>', lambda e: dlg.destroy())

        dlg.update_idletasks()
        w, h = 640, 520
        x = self.root.winfo_x() + (self.root.winfo_width()  - w) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - h) // 2
        dlg.geometry(f"{w}x{h}+{x}+{y}")

        refreshList()
        dlg.wait_window()

        filepath = selectedPath[0]
        if not filepath:
            return
        # ─────────────────────────────────────────────────────────────────────
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            self.experimentNameVar.set(config.get('experimentName', ''))
            self.datasetVar.set(config.get('dataset', 'mnist'))
            self.approachVar.set(config.get('approach', 'basil'))
            self.useBasilVar.set(config.get('useBasil', False))
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
            self.nonIIDVar.set(config.get('nonIID', True))
            self.dirichletAlphaVar.set(config.get('dirichletAlpha', 0.2))
            self.distillStrengthVar.set(config.get('distillStrength', 0.5))
            self.verifyThresholdVar.set(config.get('verifyThreshold', 0.05))
            self.cartAlgorithmVar.set(config.get('cartAlgorithm', 'cart'))
            self._onNonIIDChange()
            self.attackerIdsVar.set(config.get('attackerIds', '1,4,6,8'))
            self.nNodesVar.set(config.get('nNodes', 10))
            self.nRoundsVar.set(config.get('nRounds', 100))
            self.localEpochsVar.set(config.get('localEpochs', 1))
            self.learningRateVar.set(config.get('learningRate', 0.05))
            self.batchSizeVar.set(config.get('batchSize', 32))
            self.useLrDecayVar.set(config.get('useLrDecay', True))
            self.usePlateauLrVar.set(config.get('usePlateauLr', True))
            self.plateauPatienceVar.set(config.get('plateauPatience', 8))
            self.plateauFactorVar.set(config.get('plateauFactor', 0.7))
            self.plateauMinLrVar.set(config.get('plateauMinLr', 0.001))
            self.plateauThresholdVar.set(config.get('plateauThreshold', 0.01))
            messagebox.showinfo("Loaded", "Configuration loaded successfully!")
            self._setStatus(f"Config loaded: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")

    # ── Plot results ──────────────────────────────────────────────────────────
    def _scheduleCampaign3PlotRefresh(self, config):
        """Queue one nonblocking PNG refresh after a completed Campaign 3 run."""
        if (
            int(config.get("campaignVersion", 0)) != 3
            or not config.get("autoPlotCampaign3", True)
        ):
            return
        split = "nonIID" if config.get("nonIID", True) else "IID"
        with self._campaignPlotCondition:
            self._campaignPlotPendingSplits.add(split)
            if (
                self._campaignPlotThread is not None
                and self._campaignPlotThread.is_alive()
            ):
                self._campaignPlotCondition.notify_all()
                return
            self._campaignPlotThread = threading.Thread(
                target=self._campaign3PlotRefreshLoop,
                name="campaign3-plot-refresh",
                daemon=True,
            )
            self._campaignPlotThread.start()

    def _campaign3PlotRefreshLoop(self):
        from plotCampaign3 import generate_campaign3_live_plots

        while True:
            with self._campaignPlotCondition:
                if not self._campaignPlotPendingSplits:
                    self._campaignPlotThread = None
                    self._campaignPlotCondition.notify_all()
                    return
                split = sorted(self._campaignPlotPendingSplits)[0]
                self._campaignPlotPendingSplits.remove(split)

            self.root.after(
                0,
                self.logMessage,
                f"\n[plots3] Refreshing changed {split} PNG previews…",
            )
            try:
                result = generate_campaign3_live_plots(split=split)
                message = (
                    f"[plots3] Live refresh complete: "
                    f"{len(result['generated'])} updated, "
                    f"{len(result['skipped'])} unchanged."
                )
                self.root.after(0, self.logMessage, message)
                for error in result["errors"]:
                    self.root.after(
                        0,
                        self.logMessage,
                        f"[plots3] WARNING: {error}",
                    )
            except Exception as error:
                self.root.after(
                    0,
                    self.logMessage,
                    f"[plots3] Live refresh failed: {error}",
                )
                self.root.after(
                    0,
                    self.logMessage,
                    traceback.format_exc(),
                )

    def _waitForCampaign3PlotRefresh(self):
        """Wait for pending previews before writing final PNG/PDF/EPS figures."""
        while True:
            with self._campaignPlotCondition:
                thread = self._campaignPlotThread
                pending = bool(self._campaignPlotPendingSplits)
            if thread is None and not pending:
                return
            if thread is not None:
                thread.join(timeout=0.25)
            else:
                time.sleep(0.05)

    def plotResults(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Generate Plots")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(
            dlg,
            text="Choose plot output",
            bg=BG,
            fg=HEADER,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W, padx=18, pady=(16, 2))
        tk.Label(
            dlg,
            text=(
                f"Campaign 4 writes {CAMPAIGN4_PLOT_ROOT}; Campaign 3 R2 remains "
                f"under {CAMPAIGN3_PLOT_ROOT}."
            ),
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, padx=18, pady=(0, 12))

        campaign4Frame = ttk.LabelFrame(dlg, text="Campaign 4", padding=10)
        campaign4Frame.pack(fill=tk.X, padx=18, pady=(0, 8))

        def runCampaign4(mode):
            dlg.destroy()
            self.generateCampaign4Plots(
                mode=mode,
                showDialog=True,
                onlyChanged=False,
            )

        for text, mode in (
            ("Paper Figures", "paper"),
            ("Diagnostics", "diagnostics"),
            ("Paper + Diagnostics", "both"),
        ):
            ttk.Button(
                campaign4Frame,
                text=text,
                style="Run.TButton" if mode == "both" else "TButton",
                command=lambda selected=mode: runCampaign4(selected),
            ).pack(side=tk.LEFT, padx=6)

        campaignFrame = ttk.LabelFrame(dlg, text="Campaign 3 R2", padding=10)
        campaignFrame.pack(fill=tk.X, padx=18, pady=(0, 8))

        def runCampaign(mode):
            dlg.destroy()
            self.generateCampaign3Plots(
                mode=mode,
                showDialog=True,
                onlyChanged=False,
            )

        ttk.Button(
            campaignFrame,
            text="Paper Figures",
            command=lambda: runCampaign("paper"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            campaignFrame,
            text="Diagnostics",
            command=lambda: runCampaign("diagnostics"),
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            campaignFrame,
            text="Paper + Diagnostics",
            style="Run.TButton",
            command=lambda: runCampaign("both"),
        ).pack(side=tk.LEFT, padx=(6, 0))

        legacyFrame = ttk.LabelFrame(dlg, text="Legacy results", padding=10)
        legacyFrame.pack(fill=tk.X, padx=18, pady=(0, 12))
        ttk.Button(
            legacyFrame,
            text="Generate plots / plots2",
            command=lambda: (dlg.destroy(), self.generatePlots(showDialog=True)),
        ).pack(side=tk.LEFT)
        ttk.Button(
            dlg,
            text="Cancel",
            command=dlg.destroy,
        ).pack(anchor=tk.E, padx=18, pady=(0, 16))

        dlg.update_idletasks()
        width, height = 560, 335
        x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
        dlg.geometry(f"{width}x{height}+{x}+{y}")

    def _attackKeyFromConfig(self, config):
        attackParts = []
        for k, s in [('attackGaussian','gaussian'),('attackSignFlip','signflip'),
                     ('attackHidden','hidden'),('attackModelPoison','model_poison'),
                     ('attackScaling','scaling'),('attackAlie','alie'),
                     ('attackIpm','ipm'),('attackNoiseAmp','noise_amp')]:
            if config.get(k):
                attackParts.append(s)
        return "_".join(attackParts) if attackParts else "none"

    def generateCampaign4Plots(
        self,
        *,
        mode="both",
        showDialog=True,
        split=None,
        onlyChanged=True,
    ):
        try:
            from plotCampaign4 import generate_campaign4_plots

            self.logMessage("\n" + "=" * 60)
            self.logMessage(f"GENERATING CAMPAIGN 4 {mode.upper()} PLOTS")
            self.logMessage("=" * 60)
            result = generate_campaign4_plots(
                mode=mode,
                split=split,
                only_changed=onlyChanged,
            )
            self.logMessage(f"Completed records found: {result['records']}")
            self.logMessage(f"Figures generated:       {len(result['generated'])}")
            self.logMessage(f"Unchanged figures:       {len(result['skipped'])}")
            for warning in result.get("warnings", []):
                self.logMessage(f"NOTICE: {warning}")
            for error in result["errors"]:
                self.logMessage(f"WARNING: {error}")
            self.logMessage(f"Output: {result['plotRoot']}")
            self.logMessage("=" * 60)
            if result["records"] == 0:
                message = (
                    "No completed Campaign 4 results were found in "
                    f"{CAMPAIGN4_RESULT_ROOT}."
                )
                self._setStatus("No completed Campaign 4 results found.")
                if showDialog:
                    messagebox.showinfo("No Campaign 4 Results", message)
                return result
            self._setStatus(
                f"Campaign 4 plots: {len(result['generated'])} generated, "
                f"{len(result['skipped'])} unchanged."
            )
            if showDialog:
                messagebox.showinfo(
                    "Campaign 4 Plots",
                    f"Generated: {len(result['generated'])}\n"
                    f"Unchanged: {len(result['skipped'])}\n"
                    f"Notices: {len(result.get('warnings', []))}\n"
                    f"Plot errors: {len(result['errors'])}\n\n"
                    f"Saved under {CAMPAIGN4_PLOT_ROOT}/.",
                )
            return result
        except Exception as error:
            self.logMessage(f"\nERROR generating Campaign 4 plots: {error}")
            self.logMessage(traceback.format_exc())
            if showDialog:
                messagebox.showerror(
                    "Campaign 4 Plot Error",
                    f"Failed to generate Campaign 4 plots:\n{error}",
                )
            return None

    def generateCampaign3Plots(
        self,
        *,
        mode="both",
        showDialog=True,
        split=None,
        onlyChanged=True,
    ):
        try:
            from plotCampaign3 import generate_campaign3_plots

            self.logMessage("\n" + "=" * 60)
            self.logMessage(f"GENERATING CAMPAIGN 3 {mode.upper()} PLOTS")
            self.logMessage("=" * 60)
            result = generate_campaign3_plots(
                mode=mode,
                split=split,
                only_changed=onlyChanged,
            )
            self.logMessage(f"Completed records found: {result['records']}")
            self.logMessage(f"Figures generated:       {len(result['generated'])}")
            self.logMessage(f"Unchanged figures:       {len(result['skipped'])}")
            for error in result["errors"]:
                self.logMessage(f"WARNING: {error}")
            self.logMessage(f"Output: {result['plotRoot']}")
            self.logMessage("=" * 60)
            if result["records"] == 0:
                message = (
                    "No completed Campaign 3 R2 results were found in "
                    f"{CAMPAIGN3_RESULT_ROOT}."
                )
                self._setStatus("No completed Campaign 3 results found.")
                if showDialog:
                    messagebox.showinfo("No Campaign 3 Results", message)
                return result
            self._setStatus(
                f"Campaign 3 plots: {len(result['generated'])} generated, "
                f"{len(result['skipped'])} unchanged."
            )
            if showDialog:
                detail = (
                    f"Generated: {len(result['generated'])}\n"
                    f"Unchanged: {len(result['skipped'])}\n"
                    f"Warnings: {len(result['errors'])}\n\n"
                    f"Saved under {CAMPAIGN3_PLOT_ROOT}/."
                )
                messagebox.showinfo("Campaign 3 Plots", detail)
            return result
        except Exception as error:
            self.logMessage(f"\nERROR generating Campaign 3 plots: {error}")
            self.logMessage(traceback.format_exc())
            if showDialog:
                messagebox.showerror(
                    "Campaign 3 Plot Error",
                    f"Failed to generate Campaign 3 plots:\n{error}",
                )
            return None

    def generatePlots(self, showDialog=True, config=None, onlyMissing=False):
        if config is not None and int(config.get("campaignVersion", 0)) == 4:
            split = str(
                config.get(
                    "split",
                    "nonIID" if config.get("nonIID", True) else "IID",
                )
            )
            return self.generateCampaign4Plots(
                mode="both",
                showDialog=showDialog,
                split=split,
                onlyChanged=onlyMissing,
            )
        if config is not None and int(config.get("campaignVersion", 0)) == 3:
            split = "nonIID" if config.get("nonIID", True) else "IID"
            return self.generateCampaign3Plots(
                mode="both",
                showDialog=showDialog,
                split=split,
                onlyChanged=onlyMissing,
            )
        try:
            from plotGui import (discoverDataSplits, discoverDatasets, discoverAttackTypes, discoverApproaches,
                                 discoverExperiments, groupExperimentsByNoiseBucket, noiseBucket,
                                 plotExperimentSet, plotMitigationSweepComparison)

            self.logMessage("\n" + "=" * 60)
            self.logMessage("GENERATING PLOTS")
            self.logMessage("=" * 60)

            if config is not None:
                split = 'nonIID' if config.get('nonIID', True) else 'IID'
                dataset = config.get('dataset')
                attackKey = self._attackKeyFromConfig(config)
                approach = config.get('approach', 'basil')
                experiments = discoverExperiments(dataset, attackKey, approach, split=split)
                if not experiments:
                    self.logMessage(f"No saved experiments found for {split}/{dataset}/{attackKey}/{approach}")
                    return

                self.logMessage(
                    f"Target: {split} | {dataset} | {attackKey} | {approach} "
                    f"({len(experiments)} experiments)"
                )
                plotExperimentSet(
                    dataset, attackKey, approach, experiments,
                    split=split, skipExisting=onlyMissing
                )

                bucket = noiseBucket(config)
                bucketExperiments = groupExperimentsByNoiseBucket(experiments).get(bucket, [])
                if bucketExperiments:
                    self.logMessage(f"  bucket: {bucket} ({len(bucketExperiments)} experiments)")
                    plotExperimentSet(
                        dataset, attackKey, approach, bucketExperiments,
                        split=split, plotSubdir=bucket, skipExisting=onlyMissing
                    )
                if approach in ('merged', 'cart'):
                    plotMitigationSweepComparison(
                        dataset, attackKey, split=split, skipExisting=onlyMissing
                    )

                self.logMessage("\nPlots checked for completed experiment.")
                self.logMessage("=" * 60)
                self._setStatus("Plots checked for completed experiment.")
                return

            splits = discoverDataSplits()
            if not splits:
                self.logMessage("No experiment results found. Run some experiments first!")
                if showDialog:
                    messagebox.showinfo("No Results", "No experiment results found.\nRun some experiments first!")
                return

            self.logMessage(f"Found data splits: {splits}")
            for split in splits:
                datasets = discoverDatasets(split=split)
                if not datasets:
                    continue
                self.logMessage(f"\nSplit: {split} - datasets: {datasets}")
                for dataset in datasets:
                    attackTypes = discoverAttackTypes(dataset, split=split)
                    if not attackTypes:
                        continue
                    self.logMessage(f"\nDataset: {dataset.upper()} [{split}] - attacks: {attackTypes}")
                    for attackKey in attackTypes:
                        approaches = discoverApproaches(dataset, attackKey, split=split)
                        if not approaches:
                            continue
                        for approach in approaches:
                            experiments = discoverExperiments(dataset, attackKey, approach, split=split)
                            if not experiments:
                                continue
                            self.logMessage(f"  {split} | {attackKey} | {approach}  ({len(experiments)} experiments)")
                            plotExperimentSet(dataset, attackKey, approach, experiments, split=split)
                            for bucket, bucketExperiments in groupExperimentsByNoiseBucket(experiments).items():
                                self.logMessage(f"    bucket: {bucket} ({len(bucketExperiments)} experiments)")
                                plotExperimentSet(dataset, attackKey, approach, bucketExperiments,
                                                  split=split, plotSubdir=bucket)
                        plotMitigationSweepComparison(dataset, attackKey, split=split)

            self.logMessage("\nAll plots saved to: plots/images/gui/{split}/")
            self.logMessage("Merged/CART plots saved to: plots2/images/gui/{split}/")
            self.logMessage("=" * 60)
            if showDialog:
                messagebox.showinfo(
                    "Success",
                    "Plots saved.\nMerged/CART plots are in plots2/images/gui/{IID|nonIID}/"
                )
            self._setStatus("Plots saved. Merged/CART output: plots2/images/gui/{IID|nonIID}/")
        except Exception as e:
            self.logMessage(f"\nERROR generating plots: {e}")
            self.logMessage(traceback.format_exc())
            if showDialog:
                messagebox.showerror("Error", f"Failed to generate plots:\n{e}")



def main():
    root = tk.Tk()
    app = ExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
