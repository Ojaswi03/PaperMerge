"""
Graphical User Interface for BASIL + Noisy Channel Experiments

This GUI allows you to:
- Select datasets (MNIST, CIFAR-10, N-MNIST)
- Choose training approach (BASIL, Noisy Channel, Merged)
- Configure attacks and when they start
- Configure channel noise and when it starts
- Set all training parameters
- Run experiments and view results
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basil_core.data.mnist import loadMnist, makeLoaders as makeMnistLoaders
from basil_core.data.cifar import loadCifar10, makeLoaders as makeCifarLoaders
from basil_core.data.nMnist import loadNMnist, makeLoaders as makeNMnistLoaders
from basil_core.models import MNISTModel, CIFARModel, NMNISTModel
from basil_core.basil import BasilNode, basilRingTrainingWithAttack
from basil_core.trainer import evaluateAll
from scripts.common import setupGpu
import numpy as np


class ExperimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BASIL + Noisy Channel Experiment GUI")
        self.root.geometry("900x800")

        # Variables
        self.setupVariables()

        # Create UI
        self.createUI()

        # Running state
        self.isRunning = False
        self.currentThread = None

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.onClosing)

    def setupVariables(self):
        """Initialize all tkinter variables"""
        # Dataset selection
        self.datasetVar = tk.StringVar(value="mnist")

        # Approach selection
        self.approachVar = tk.StringVar(value="basil")

        # BASIL parameters
        self.useBasilVar = tk.BooleanVar(value=True)
        self.basilMemorySizeVar = tk.IntVar(value=10)

        # Noisy Channel parameters
        self.useChannelNoiseVar = tk.BooleanVar(value=False)
        self.channelNoiseStartVar = tk.IntVar(value=0)
        self.channelNoiseSigmaVar = tk.DoubleVar(value=0.1)
        self.noiseMitigationVar = tk.StringVar(value="none")
        self.ebmLambdaVar = tk.DoubleVar(value=0.01)
        self.wcmLambdaVar = tk.DoubleVar(value=0.1)
        self.wcmSamplesVar = tk.IntVar(value=5)
        self.wcmRhoVar = tk.DoubleVar(value=0.5)

        # Attack parameters
        self.attackGaussianVar = tk.BooleanVar(value=False)
        self.attackGaussianStartVar = tk.IntVar(value=0)
        self.attackSignFlipVar = tk.BooleanVar(value=False)
        self.attackSignFlipStartVar = tk.IntVar(value=0)
        self.attackHiddenVar = tk.BooleanVar(value=False)
        self.attackHiddenStartVar = tk.IntVar(value=10)
        self.attackerIdsVar = tk.StringVar(value="0,5")

        # Training parameters
        self.nNodesVar = tk.IntVar(value=10)
        self.nRoundsVar = tk.IntVar(value=30)
        self.localEpochsVar = tk.IntVar(value=1)
        self.learningRateVar = tk.DoubleVar(value=0.05)
        self.batchSizeVar = tk.IntVar(value=32)

    def createUI(self):
        """Create the user interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Basic Configuration
        basicTab = ttk.Frame(notebook)
        notebook.add(basicTab, text="Basic Configuration")
        self.createBasicTab(basicTab)

        # Tab 2: Advanced Configuration
        advancedTab = ttk.Frame(notebook)
        notebook.add(advancedTab, text="Advanced Configuration")
        self.createAdvancedTab(advancedTab)

        # Tab 3: Attack Configuration
        attackTab = ttk.Frame(notebook)
        notebook.add(attackTab, text="Attack Configuration")
        self.createAttackTab(attackTab)

        # Tab 4: Output
        outputTab = ttk.Frame(notebook)
        notebook.add(outputTab, text="Output")
        self.createOutputTab(outputTab)

        # Bottom buttons
        self.createButtons()

    def createBasicTab(self, parent):
        """Create basic configuration tab"""
        frame = ttk.LabelFrame(parent, text="Basic Settings", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        row = 0

        # Dataset selection
        ttk.Label(frame, text="Dataset:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=5)
        datasets = [("MNIST", "mnist"), ("CIFAR-10", "cifar10"), ("Neuromorphic MNIST", "nmnist")]
        for i, (label, value) in enumerate(datasets):
            ttk.Radiobutton(frame, text=label, variable=self.datasetVar, value=value).grid(row=row, column=i+1, sticky=tk.W, padx=10)
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        row += 1

        # Approach selection
        ttk.Label(frame, text="Approach:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(frame, text="BASIL Only", variable=self.approachVar, value="basil",
                       command=self.onApproachChange).grid(row=row, column=1, sticky=tk.W, padx=10)
        ttk.Radiobutton(frame, text="Noisy Channel Only", variable=self.approachVar, value="noisy",
                       command=self.onApproachChange).grid(row=row, column=2, sticky=tk.W, padx=10)
        ttk.Radiobutton(frame, text="Merged (BASIL + Noisy)", variable=self.approachVar, value="merged",
                       command=self.onApproachChange).grid(row=row, column=3, sticky=tk.W, padx=10)
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        row += 1

        # Training parameters
        ttk.Label(frame, text="Training Parameters:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(frame, text="Number of Nodes:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.nNodesVar, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
        ttk.Label(frame, text="Training Rounds:").grid(row=row, column=2, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.nRoundsVar, width=10).grid(row=row, column=3, sticky=tk.W, padx=10)
        row += 1

        ttk.Label(frame, text="Local Epochs:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.localEpochsVar, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
        ttk.Label(frame, text="Learning Rate:").grid(row=row, column=2, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.learningRateVar, width=10).grid(row=row, column=3, sticky=tk.W, padx=10)
        row += 1

        ttk.Label(frame, text="Batch Size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.batchSizeVar, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1

    def createAdvancedTab(self, parent):
        """Create advanced configuration tab"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # BASIL Configuration
        basilFrame = ttk.LabelFrame(frame, text="BASIL Configuration", padding=10)
        basilFrame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(basilFrame, text="Use BASIL Snapshot Selection", variable=self.useBasilVar).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(basilFrame, text="Memory Size (S):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(basilFrame, textvariable=self.basilMemorySizeVar, width=10).grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Label(basilFrame, text="Number of past models to store").grid(row=1, column=2, sticky=tk.W, padx=10)

        # Channel Noise Configuration
        noiseFrame = ttk.LabelFrame(frame, text="Channel Noise Configuration", padding=10)
        noiseFrame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(noiseFrame, text="Enable Channel Noise", variable=self.useChannelNoiseVar).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(noiseFrame, text="Start Noise at Round:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(noiseFrame, textvariable=self.channelNoiseStartVar, width=10).grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Label(noiseFrame, text="(0 = from beginning)").grid(row=1, column=2, sticky=tk.W, padx=10)

        ttk.Label(noiseFrame, text="Noise Sigma (σ):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(noiseFrame, textvariable=self.channelNoiseSigmaVar, width=10).grid(row=2, column=1, sticky=tk.W, padx=10)
        ttk.Label(noiseFrame, text="Standard deviation of Gaussian noise").grid(row=2, column=2, sticky=tk.W, padx=10)

        ttk.Label(noiseFrame, text="Mitigation:").grid(row=3, column=0, sticky=tk.W, pady=5)
        mitigationFrame = ttk.Frame(noiseFrame)
        mitigationFrame.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=10)
        ttk.Radiobutton(mitigationFrame, text="None", variable=self.noiseMitigationVar, value="none").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mitigationFrame, text="EBM", variable=self.noiseMitigationVar, value="ebm").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mitigationFrame, text="WCM", variable=self.noiseMitigationVar, value="wcm").pack(side=tk.LEFT, padx=5)

        # EBM parameters
        ttk.Label(noiseFrame, text="EBM Lambda:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(noiseFrame, textvariable=self.ebmLambdaVar, width=10).grid(row=4, column=1, sticky=tk.W, padx=10)
        ttk.Label(noiseFrame, text="Regularization strength for EBM").grid(row=4, column=2, sticky=tk.W, padx=10)

        # WCM parameters
        ttk.Label(noiseFrame, text="WCM Lambda:").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(noiseFrame, textvariable=self.wcmLambdaVar, width=10).grid(row=5, column=1, sticky=tk.W, padx=10)
        ttk.Label(noiseFrame, text="Regularization strength for WCM").grid(row=5, column=2, sticky=tk.W, padx=10)

        ttk.Label(noiseFrame, text="WCM Samples:").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Entry(noiseFrame, textvariable=self.wcmSamplesVar, width=10).grid(row=6, column=1, sticky=tk.W, padx=10)
        ttk.Label(noiseFrame, text="Number of boundary samples").grid(row=6, column=2, sticky=tk.W, padx=10)

        ttk.Label(noiseFrame, text="WCM Rho:").grid(row=7, column=0, sticky=tk.W, pady=2)
        ttk.Entry(noiseFrame, textvariable=self.wcmRhoVar, width=10).grid(row=7, column=1, sticky=tk.W, padx=10)
        ttk.Label(noiseFrame, text="SCA convex combination parameter").grid(row=7, column=2, sticky=tk.W, padx=10)

    def createAttackTab(self, parent):
        """Create attack configuration tab"""
        frame = ttk.LabelFrame(parent, text="Byzantine Attack Configuration", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Select attacks and when they should start:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=10)

        row = 1

        # Attacker node IDs
        ttk.Label(frame, text="Attacker Node IDs:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.attackerIdsVar, width=20).grid(row=row, column=1, sticky=tk.W, padx=10)
        ttk.Label(frame, text="(comma-separated, e.g., 0,5)").grid(row=row, column=2, sticky=tk.W, padx=10)
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        # Gaussian Attack
        ttk.Checkbutton(frame, text="Gaussian Noise Attack", variable=self.attackGaussianVar).grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Label(frame, text="Start at round:").grid(row=row, column=1, sticky=tk.W, padx=10)
        ttk.Entry(frame, textvariable=self.attackGaussianStartVar, width=10).grid(row=row, column=2, sticky=tk.W, padx=10)
        row += 1
        ttk.Label(frame, text="  Attackers send random Gaussian noise instead of gradients").grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=20)
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        # Sign Flip Attack
        ttk.Checkbutton(frame, text="Sign-Flip Attack", variable=self.attackSignFlipVar).grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Label(frame, text="Start at round:").grid(row=row, column=1, sticky=tk.W, padx=10)
        ttk.Entry(frame, textvariable=self.attackSignFlipStartVar, width=10).grid(row=row, column=2, sticky=tk.W, padx=10)
        row += 1
        ttk.Label(frame, text="  Attackers flip the sign of their gradients").grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=20)
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        # Hidden/Backdoor Attack
        ttk.Checkbutton(frame, text="Hidden/Backdoor Attack", variable=self.attackHiddenVar).grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Label(frame, text="Start at round:").grid(row=row, column=1, sticky=tk.W, padx=10)
        ttk.Entry(frame, textvariable=self.attackHiddenStartVar, width=10).grid(row=row, column=2, sticky=tk.W, padx=10)
        row += 1
        ttk.Label(frame, text="  Attackers behave normally initially, then inject malicious updates").grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=20)
        row += 1

    def createOutputTab(self, parent):
        """Create output tab with log"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Experiment Output:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)

        self.outputText = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=30, font=('Courier', 9))
        self.outputText.pack(fill=tk.BOTH, expand=True)

    def createButtons(self):
        """Create control buttons"""
        buttonFrame = ttk.Frame(self.root)
        buttonFrame.pack(fill=tk.X, padx=5, pady=5)

        self.runButton = ttk.Button(buttonFrame, text="Run Experiment", command=self.runExperiment, style='Accent.TButton')
        self.runButton.pack(side=tk.LEFT, padx=5)

        self.stopButton = ttk.Button(buttonFrame, text="Stop", command=self.stopExperiment, state=tk.DISABLED)
        self.stopButton.pack(side=tk.LEFT, padx=5)

        ttk.Button(buttonFrame, text="Clear Output", command=self.clearOutput).pack(side=tk.LEFT, padx=5)

        ttk.Button(buttonFrame, text="Save Configuration", command=self.saveConfig).pack(side=tk.LEFT, padx=5)

        ttk.Button(buttonFrame, text="Load Configuration", command=self.loadConfig).pack(side=tk.LEFT, padx=5)

        ttk.Button(buttonFrame, text="Exit", command=self.onClosing).pack(side=tk.RIGHT, padx=5)

    def onApproachChange(self):
        """Handle approach selection change"""
        approach = self.approachVar.get()
        if approach == "basil":
            self.useBasilVar.set(True)
            self.useChannelNoiseVar.set(False)
        elif approach == "noisy":
            self.useBasilVar.set(False)
            self.useChannelNoiseVar.set(True)
        elif approach == "merged":
            self.useBasilVar.set(True)
            self.useChannelNoiseVar.set(True)

    def logMessage(self, message):
        """Add message to output log"""
        self.outputText.insert(tk.END, message + "\n")
        self.outputText.see(tk.END)
        self.root.update_idletasks()

    def clearOutput(self):
        """Clear output log"""
        self.outputText.delete(1.0, tk.END)

    def runExperiment(self):
        """Run the experiment in a separate thread"""
        if self.isRunning:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return

        # Validate configuration
        if not self.validateConfig():
            return

        # Disable run button, enable stop button
        self.runButton.config(state=tk.DISABLED)
        self.stopButton.config(state=tk.NORMAL)
        self.isRunning = True

        # Clear output
        self.clearOutput()

        # Run in separate thread
        self.currentThread = threading.Thread(target=self.runExperimentThread)
        self.currentThread.start()

    def stopExperiment(self):
        """Stop the running experiment"""
        self.isRunning = False
        self.logMessage("\n[STOP REQUESTED] Stopping experiment...")

    def onClosing(self):
        """Handle window close event"""
        if self.isRunning:
            # Ask for confirmation if experiment is running
            response = messagebox.askyesno(
                "Experiment Running",
                "An experiment is currently running. Do you want to stop it and exit?"
            )
            if not response:
                return  # User cancelled, don't close

            # Stop the experiment
            self.isRunning = False
            self.logMessage("\n[WINDOW CLOSING] Stopping experiment...")

            # Wait for thread to finish (with timeout)
            if self.currentThread is not None and self.currentThread.is_alive():
                self.logMessage("Waiting for experiment to terminate...")
                self.currentThread.join(timeout=5.0)  # Wait up to 5 seconds

                if self.currentThread.is_alive():
                    self.logMessage("Warning: Thread did not terminate cleanly")

        # Destroy the window
        self.root.destroy()

    def runExperimentThread(self):
        """The actual experiment execution (runs in separate thread)"""
        try:
            self.logMessage("="*80)
            self.logMessage("STARTING EXPERIMENT")
            self.logMessage("="*80)
            self.logMessage(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logMessage("")

            # Get configuration
            config = self.getConfig()
            self.logMessage("Configuration:")
            self.logMessage(f"  Dataset: {config['dataset']}")
            self.logMessage(f"  Approach: {config['approach']}")
            self.logMessage(f"  Nodes: {config['nNodes']}, Rounds: {config['nRounds']}")
            self.logMessage(f"  Use BASIL: {config['useBasil']}")
            self.logMessage(f"  Use Channel Noise: {config['useChannelNoise']}")
            self.logMessage("")

            # Setup GPU
            self.logMessage("Setting up GPU/CPU...")
            setupGpu()
            self.logMessage("")

            # Load data
            self.logMessage(f"Loading {config['dataset'].upper()} dataset...")
            train, test = self.loadDataset(config['dataset'])
            trainLoaders, testLoader = self.makeLoaders(config['dataset'], train, test, config['batchSize'], config['nNodes'])
            self.logMessage(f"  Training samples: {len(train)}")
            self.logMessage(f"  Test samples: {len(test)}")
            self.logMessage("")

            # Create nodes
            self.logMessage(f"Creating {config['nNodes']} nodes...")
            nodes = self.createNodes(config, trainLoaders)
            self.logMessage("")

            # Prepare attacks
            attackTypes, attackerIds = self.prepareAttacks(config)
            self.logMessage(f"Attack configuration:")
            self.logMessage(f"  Attackers: {attackerIds if attackerIds else 'None'}")
            self.logMessage(f"  Attack types: {attackTypes}")
            self.logMessage("")

            # Run training
            self.logMessage(f"Starting training for {config['nRounds']} rounds...")
            self.logMessage("-"*80)

            avgAccHist, worstAccHist = basilRingTrainingWithAttack(
                nodes=nodes,
                rounds=config['nRounds'],
                testLoader=testLoader,
                attackTypes=attackTypes,
                attackerIds=attackerIds,
                hiddenStartRound=config['attackHiddenStart'] if config['attackHidden'] else 999,
                sigma=config['channelNoiseSigma'] if config['useChannelNoise'] else 0.0,
                noiseModel=self.getNoiseModel(config),
                lr0=config['learningRate'],
                lrAlpha=0.6,
                stepsPerEpoch=100,
                useSnapshots=config['useBasil'],
                stopCallback=lambda: not self.isRunning,  # Check if stop was requested
            )

            # Check if stopped early
            if not self.isRunning:
                self.logMessage("")
                self.logMessage("="*80)
                self.logMessage("EXPERIMENT STOPPED BY USER")
                self.logMessage("="*80)

            # Final evaluation
            self.logMessage("")
            self.logMessage("="*80)
            finalAvg, finalWorst, allAccs = evaluateAll(nodes, testLoader)
            self.logMessage(f"FINAL RESULTS:")
            self.logMessage(f"  Average Accuracy: {finalAvg:.4f}")
            self.logMessage(f"  Worst Node Accuracy: {finalWorst:.4f}")
            self.logMessage(f"  Per-node accuracies: {[f'{acc:.4f}' for acc in allAccs]}")
            self.logMessage("="*80)

            # Save results (even if stopped early)
            self.saveResults(config, avgAccHist, worstAccHist, finalAvg, finalWorst)

            if not self.isRunning:
                self.logMessage("\nExperiment stopped early but partial results saved.")
            else:
                self.logMessage("\nExperiment completed successfully!")

        except Exception as e:
            self.logMessage(f"\nERROR: {str(e)}")
            import traceback
            self.logMessage(traceback.format_exc())

        finally:
            # Re-enable run button
            self.runButton.config(state=tk.NORMAL)
            self.stopButton.config(state=tk.DISABLED)
            self.isRunning = False

    def validateConfig(self):
        """Validate configuration before running"""
        try:
            if self.nRoundsVar.get() <= 0:
                messagebox.showerror("Error", "Number of rounds must be positive")
                return False
            if self.nNodesVar.get() <= 0:
                messagebox.showerror("Error", "Number of nodes must be positive")
                return False
            if self.channelNoiseStartVar.get() < 0 or self.channelNoiseStartVar.get() >= self.nRoundsVar.get():
                messagebox.showerror("Error", f"Channel noise start round must be between 0 and {self.nRoundsVar.get()-1}")
                return False
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Invalid configuration: {str(e)}")
            return False

    def getConfig(self):
        """Get current configuration as dictionary"""
        return {
            'dataset': self.datasetVar.get(),
            'approach': self.approachVar.get(),
            'useBasil': self.useBasilVar.get(),
            'basilMemorySize': self.basilMemorySizeVar.get(),
            'useChannelNoise': self.useChannelNoiseVar.get(),
            'channelNoiseStart': self.channelNoiseStartVar.get(),
            'channelNoiseSigma': self.channelNoiseSigmaVar.get(),
            'noiseMitigation': self.noiseMitigationVar.get(),
            'ebmLambda': self.ebmLambdaVar.get(),
            'wcmLambda': self.wcmLambdaVar.get(),
            'wcmSamples': self.wcmSamplesVar.get(),
            'wcmRho': self.wcmRhoVar.get(),
            'attackGaussian': self.attackGaussianVar.get(),
            'attackGaussianStart': self.attackGaussianStartVar.get(),
            'attackSignFlip': self.attackSignFlipVar.get(),
            'attackSignFlipStart': self.attackSignFlipStartVar.get(),
            'attackHidden': self.attackHiddenVar.get(),
            'attackHiddenStart': self.attackHiddenStartVar.get(),
            'attackerIds': self.attackerIdsVar.get(),
            'nNodes': self.nNodesVar.get(),
            'nRounds': self.nRoundsVar.get(),
            'localEpochs': self.localEpochsVar.get(),
            'learningRate': self.learningRateVar.get(),
            'batchSize': self.batchSizeVar.get(),
        }

    def loadDataset(self, dataset):
        """Load specified dataset"""
        if dataset == "mnist":
            return loadMnist()
        elif dataset == "cifar10":
            return loadCifar10()
        elif dataset == "nmnist":
            return loadNMnist()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def makeLoaders(self, dataset, train, test, batchSize, nClients):
        """Make data loaders for specified dataset"""
        if dataset == "mnist":
            return makeMnistLoaders(train, test, batchSize=batchSize, nClients=nClients)
        elif dataset == "cifar10":
            return makeCifarLoaders(train, test, batchSize=batchSize, nClients=nClients)
        elif dataset == "nmnist":
            return makeNMnistLoaders(train, test, batchSize=batchSize, nClients=nClients)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def createNodes(self, config, trainLoaders):
        """Create nodes for training"""
        modelClass = self.getModelClass(config['dataset'])
        nodes = []

        for i in range(config['nNodes']):
            model = modelClass()
            nodeConfig = {
                "nodeId": i,
                "model": model,
                "dataLoader": trainLoaders[i],
                "S": config['basilMemorySize'],
                "noiseModel": self.getNoiseModel(config),
                "sigma": config['channelNoiseSigma'] if config['useChannelNoise'] else 0.0,
                "lr0": config['learningRate'],
                "localEpochs": config['localEpochs'],
            }

            # Add mitigation-specific parameters
            if config['noiseMitigation'] == 'ebm':
                nodeConfig['ebmLambda'] = config['ebmLambda']
            elif config['noiseMitigation'] == 'wcm':
                nodeConfig['wcmLambda'] = config['wcmLambda']
                nodeConfig['wcmSamples'] = config['wcmSamples']
                nodeConfig['wcmRho'] = config['wcmRho']

            from basil_core.basil import BasilNode
            nodes.append(BasilNode(**nodeConfig))

        return nodes

    def getModelClass(self, dataset):
        """Get model class for dataset"""
        if dataset == "mnist":
            return MNISTModel
        elif dataset == "cifar10":
            return CIFARModel
        elif dataset == "nmnist":
            return NMNISTModel
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def getNoiseModel(self, config):
        """Determine noise model based on configuration"""
        if not config['useChannelNoise']:
            return "none"
        elif config['noiseMitigation'] == 'ebm':
            return "ebm"
        elif config['noiseMitigation'] == 'wcm':
            return "wcm"
        else:
            return "noisy"

    def prepareAttacks(self, config):
        """Prepare attack configuration"""
        attackTypes = []

        # Parse attacker IDs
        try:
            attackerIds = [int(x.strip()) for x in config['attackerIds'].split(',') if x.strip()]
        except:
            attackerIds = []

        if not attackerIds:
            return ["none"], []

        # Collect active attacks with their start rounds
        attacks = []
        if config['attackGaussian']:
            attacks.append(('gaussian', config['attackGaussianStart']))
        if config['attackSignFlip']:
            attacks.append(('signFlip', config['attackSignFlipStart']))
        if config['attackHidden']:
            attacks.append(('hidden', config['attackHiddenStart']))

        if not attacks:
            return ["none"], []

        # Sort by start round
        attacks.sort(key=lambda x: x[1])
        attackTypes = [a[0] for a in attacks]

        return attackTypes, attackerIds

    def saveResults(self, config, avgAccHist, worstAccHist, finalAvg, finalWorst):
        """Save experiment results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        approach = config['approach']
        dataset = config['dataset']

        resultDir = f"experiments/results/gui/{dataset}"
        os.makedirs(resultDir, exist_ok=True)

        # Save accuracy curves
        avgPath = f"{resultDir}/acc_{approach}_{timestamp}_avg.npy"
        worstPath = f"{resultDir}/acc_{approach}_{timestamp}_worst.npy"

        np.save(avgPath, np.array(avgAccHist))
        np.save(worstPath, np.array(worstAccHist))

        # Save configuration
        configPath = f"{resultDir}/config_{approach}_{timestamp}.json"
        with open(configPath, 'w') as f:
            json.dump(config, f, indent=2)

        self.logMessage(f"\nResults saved:")
        self.logMessage(f"  {avgPath}")
        self.logMessage(f"  {worstPath}")
        self.logMessage(f"  {configPath}")

    def saveConfig(self):
        """Save configuration to file with editable name"""
        config = self.getConfig()
        defaultName = f"config_{config['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Ask user for filename
        from tkinter import simpledialog
        userInput = simpledialog.askstring(
            "Save Configuration",
            "Enter a name for this configuration:",
            initialvalue=defaultName,
            parent=self.root
        )

        if not userInput:
            return  # User cancelled

        # Ensure .json extension
        if not userInput.endswith('.json'):
            userInput += '.json'

        filepath = os.path.join("gui", "configs", userInput)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        messagebox.showinfo("Success", f"Configuration saved to:\n{filepath}")

    def loadConfig(self):
        """Load configuration from file"""
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Load Configuration",
            initialdir="gui/configs",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                config = json.load(f)

            # Set all variables
            self.datasetVar.set(config.get('dataset', 'mnist'))
            self.approachVar.set(config.get('approach', 'basil'))
            self.useBasilVar.set(config.get('useBasil', True))
            self.basilMemorySizeVar.set(config.get('basilMemorySize', 10))
            self.useChannelNoiseVar.set(config.get('useChannelNoise', False))
            self.channelNoiseStartVar.set(config.get('channelNoiseStart', 0))
            self.channelNoiseSigmaVar.set(config.get('channelNoiseSigma', 0.1))
            self.noiseMitigationVar.set(config.get('noiseMitigation', 'none'))
            self.ebmLambdaVar.set(config.get('ebmLambda', 0.01))
            self.wcmLambdaVar.set(config.get('wcmLambda', 0.1))
            self.wcmSamplesVar.set(config.get('wcmSamples', 5))
            self.wcmRhoVar.set(config.get('wcmRho', 0.5))
            self.attackGaussianVar.set(config.get('attackGaussian', False))
            self.attackGaussianStartVar.set(config.get('attackGaussianStart', 0))
            self.attackSignFlipVar.set(config.get('attackSignFlip', False))
            self.attackSignFlipStartVar.set(config.get('attackSignFlipStart', 0))
            self.attackHiddenVar.set(config.get('attackHidden', False))
            self.attackHiddenStartVar.set(config.get('attackHiddenStart', 10))
            self.attackerIdsVar.set(config.get('attackerIds', '0,5'))
            self.nNodesVar.set(config.get('nNodes', 10))
            self.nRoundsVar.set(config.get('nRounds', 30))
            self.localEpochsVar.set(config.get('localEpochs', 1))
            self.learningRateVar.set(config.get('learningRate', 0.05))
            self.batchSizeVar.set(config.get('batchSize', 32))

            messagebox.showinfo("Success", "Configuration loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{str(e)}")


def main():
    root = tk.Tk()
    app = ExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
