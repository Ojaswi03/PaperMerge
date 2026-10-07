"""Live research ring visualization and bounded telemetry inspector."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from gui.theme import COLORS


BG = COLORS["surface_alt"]
PANEL = COLORS["surface"]
SURFACE = COLORS["input"]
BORDER = COLORS["border"]
TEXT = COLORS["text"]
MUTED = COLORS["muted"]
ACCENT = COLORS["accent"]
SUCCESS = COLORS["success"]
WARNING = COLORS["warning"]
DANGER = COLORS["danger"]


class NetworkView:
    """Reduce run events into a live ring and selected-node inspector."""

    def __init__(self, parent):
        self.parent = parent
        self.root = parent.winfo_toplevel()
        self.runs = {}
        self.active_lane = tk.IntVar(value=0)
        self.selected_node = tk.IntVar(value=0)
        self.round_var = tk.IntVar(value=0)
        self._redraw_pending = False
        self._node_positions = {}
        self._replay = None
        self._inspector_selection = None
        self._inspector_payload = object()
        self._build()

    def _build(self):
        toolbar = ttk.Frame(self.parent)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 6))
        ttk.Label(toolbar, text="Worker lane").pack(side=tk.LEFT)
        self.lane_combo = ttk.Combobox(
            toolbar,
            width=18,
            state="readonly",
            values=("Lane 1",),
        )
        self.lane_combo.current(0)
        self.lane_combo.pack(side=tk.LEFT, padx=(6, 14))
        self.lane_combo.bind("<<ComboboxSelected>>", self._on_lane)
        self.status_label = ttk.Label(toolbar, text="No research run active", foreground=MUTED)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(toolbar, text="Replay telemetry", command=self._choose_replay).pack(side=tk.RIGHT)

        pane = ttk.Panedwindow(self.parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        ring_frame = ttk.Frame(pane)
        inspector = ttk.Frame(pane, width=380)
        pane.add(ring_frame, weight=3)
        pane.add(inspector, weight=2)

        self.canvas = tk.Canvas(
            ring_frame,
            bg=BG,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.request_redraw())
        self.canvas.bind("<Button-1>", self._canvas_click)

        summary = ttk.LabelFrame(inspector, text="Run", padding=8)
        summary.pack(fill=tk.X, padx=(8, 0), pady=(0, 6))
        self.run_text = tk.StringVar(value="Waiting for a research run event.")
        ttk.Label(summary, textvariable=self.run_text, wraplength=340, justify=tk.LEFT).pack(fill=tk.X)

        node_header = ttk.Frame(inspector)
        node_header.pack(fill=tk.X, padx=(8, 0), pady=(0, 6))
        ttk.Label(node_header, text="Inspect node").pack(side=tk.LEFT)
        self.node_combo = ttk.Combobox(node_header, width=12, state="readonly", values=("Node 0",))
        self.node_combo.current(0)
        self.node_combo.pack(side=tk.LEFT, padx=6)
        self.node_combo.bind("<<ComboboxSelected>>", self._on_node)

        replay = ttk.LabelFrame(inspector, text="Round replay", padding=8)
        replay.pack(fill=tk.X, padx=(8, 0), pady=(0, 6))
        self.round_scale = ttk.Scale(
            replay,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self._on_replay_round,
        )
        self.round_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.round_label = ttk.Label(replay, text="Live")
        self.round_label.pack(side=tk.RIGHT, padx=(8, 0))

        notebook = ttk.Notebook(inspector)
        notebook.pack(fill=tk.BOTH, expand=True, padx=(8, 0))
        details_tab = ttk.Frame(notebook)
        candidates_tab = ttk.Frame(notebook)
        links_tab = ttk.Frame(notebook)
        notebook.add(details_tab, text="Details")
        notebook.add(candidates_tab, text="SS candidates")
        notebook.add(links_tab, text="Outgoing links")

        self.details = tk.Text(
            details_tab,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief=tk.FLAT,
            wrap=tk.WORD,
            padx=10,
            pady=8,
            font=("Consolas", 9),
            state=tk.DISABLED,
        )
        self.details.pack(fill=tk.BOTH, expand=True)

        self.candidate_tree = ttk.Treeview(
            candidates_tab,
            columns=("source", "distance", "loss", "valid", "selected"),
            show="headings",
            height=8,
        )
        for key, title, width in (
            ("source", "Source", 60),
            ("distance", "Distance", 76),
            ("loss", "Loss", 76),
            ("valid", "Eligible", 72),
            ("selected", "Selected", 66),
        ):
            self.candidate_tree.heading(key, text=title)
            self.candidate_tree.column(key, width=width, anchor=tk.CENTER)
        self.candidate_tree.pack(fill=tk.BOTH, expand=True)

        self.link_tree = ttk.Treeview(
            links_tab,
            columns=("receiver", "noise", "relative"),
            show="headings",
            height=8,
        )
        for key, title, width in (
            ("receiver", "Receiver", 70),
            ("noise", "Noise norm", 95),
            ("relative", "Relative", 80),
        ):
            self.link_tree.heading(key, text=title)
            self.link_tree.column(key, width=width, anchor=tk.CENTER)
        self.link_tree.pack(fill=tk.BOTH, expand=True)

    def start_run(self, lane, config):
        lane = int(lane)
        count = int(config.get("nNodes", 10))
        self.runs[lane] = {
            "config": dict(config),
            "round": 0,
            "average": None,
            "worst": None,
            "perNode": [None] * count,
            "nodes": {node: {} for node in range(count)},
            "status": "running",
            "activeNode": None,
            "lastRoundEvent": -1,
        }
        self._replay = None
        self.active_lane.set(lane)
        self._refresh_lane_values()
        self.round_scale.configure(from_=0, to=max(0, int(config.get("nRounds", 1)) - 1))
        self.round_label.configure(text="Live")
        self._refresh_node_values(count)
        self.request_redraw()

    def apply_node_update(self, lane, config, payload):
        lane = int(lane)
        if lane not in self.runs or self.runs[lane]["config"].get("runId") != config.get("runId"):
            self.start_run(lane, config)
        state = self.runs[lane]
        node_id = int(payload["nodeId"])
        previous = state["nodes"].get(node_id, {})
        if int(payload.get("round", -1)) < int(previous.get("round", -1)):
            return
        state["nodes"][node_id] = dict(payload)
        if payload.get('globalTestAccuracy') is not None:
            state['perNode'][node_id]=float(payload['globalTestAccuracy'])
        state["activeNode"] = node_id
        state["round"] = max(state["round"], int(payload.get("round", 0)))
        if lane == self.active_lane.get():
            self.request_redraw()

    def apply_round(self, lane, config, payload):
        lane = int(lane)
        if lane not in self.runs or self.runs[lane]["config"].get("runId") != config.get("runId"):
            self.start_run(lane, config)
        state = self.runs[lane]
        incoming_round = int(payload.get("round", state["lastRoundEvent"]))
        if incoming_round < int(state["lastRoundEvent"]):
            return
        state["lastRoundEvent"] = incoming_round
        state["round"] = int(payload.get("round", state["round"]))
        state["average"] = float(payload.get("averageAccuracy", 0.0))
        state["worst"] = float(payload.get("worstNodeAccuracy",payload.get("worstAccuracy", 0.0)))
        state["perNode"] = list(payload.get("perNodeAccuracy", state["perNode"]))
        if lane == self.active_lane.get():
            self.request_redraw()

    def finish_run(self, lane, status):
        if int(lane) in self.runs:
            self.runs[int(lane)]["status"] = str(status)
            self.request_redraw()

    def request_redraw(self):
        if self._redraw_pending:
            return
        self._redraw_pending = True
        self.root.after(125, self._redraw)

    def _state(self):
        return self.runs.get(self.active_lane.get())

    def _redraw(self):
        self._redraw_pending = False
        state = self._state()
        self.canvas.delete("all")
        self._node_positions = {}
        if not state:
            self.canvas.create_text(
                max(20, self.canvas.winfo_width() / 2),
                max(20, self.canvas.winfo_height() / 2),
                text="Start a queued experiment to inspect the live ring, or replay a completed run.",
                fill=MUTED,
                font=("Segoe UI", 11),
            )
            self._refresh_inspector(None)
            return

        config = state["config"]
        count = int(config.get("nNodes", 10))
        width = max(420, self.canvas.winfo_width())
        height = max(360, self.canvas.winfo_height())
        center_x, center_y = width / 2, height / 2
        radius = min(width, height) * 0.34
        positions = []
        for node in range(count):
            angle = -math.pi / 2 + 2 * math.pi * node / count
            positions.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
        self._node_positions = {node: position for node, position in enumerate(positions)}

        for node in range(count):
            x1, y1 = positions[node]
            x2, y2 = positions[(node + 1) % count]
            dx, dy = x2 - x1, y2 - y1
            length = max(math.hypot(dx, dy), 1)
            margin = 30
            payload = state["nodes"].get(node, {})
            links = payload.get("outgoing", [])
            ring_link = next(
                (
                    link
                    for link in links
                    if int(link.get("receiverId", -1)) == (node + 1) % count
                ),
                {},
            )
            relative_noise = max(0.0, float(ring_link.get("relativeNoise", 0.0)))
            edge_color = WARNING if relative_noise > 0 else BORDER
            self.canvas.create_line(
                x1 + margin * dx / length,
                y1 + margin * dy / length,
                x2 - margin * dx / length,
                y2 - margin * dy / length,
                fill=edge_color,
                width=2 + min(3, 5 * relative_noise),
                arrow=tk.LAST,
                arrowshape=(8, 10, 4),
            )

        active_node = state.get("activeNode")
        active_payload = state["nodes"].get(active_node, {}) if active_node is not None else {}
        if active_payload:
            target_x, target_y = positions[active_node]
            incoming = active_payload.get("incoming", {})
            sender = int(incoming.get("senderId", -1))
            if 0 <= sender < count and sender != active_node:
                source_x, source_y = positions[sender]
                self.canvas.create_line(
                    source_x,
                    source_y,
                    target_x,
                    target_y,
                    fill=ACCENT,
                    width=3,
                    dash=(7, 4),
                    arrow=tk.LAST,
                )
            for candidate in active_payload.get("selection", {}).get("candidates", []):
                candidate_sender = int(candidate.get("senderId", -1))
                if (
                    not candidate.get("plausible", True)
                    and 0 <= candidate_sender < count
                    and candidate_sender != active_node
                ):
                    source_x, source_y = positions[candidate_sender]
                    self.canvas.create_line(
                        source_x,
                        source_y,
                        target_x,
                        target_y,
                        fill=DANGER,
                        width=1,
                        dash=(2, 5),
                    )

        selected = self.selected_node.get()
        attackers = set(config.get("resolvedAttackerIds",[])) or {int(value) for value in str(config.get("attackerIds", "")).split(",") if value}
        for node, (x, y) in enumerate(positions):
            payload = state["nodes"].get(node, {})
            attack_active = bool(payload.get("attack", {}).get("active", False))
            configured_attacker = node in attackers
            fill = DANGER if attack_active else WARNING if configured_attacker else SUCCESS
            outline = ACCENT if node == selected else WARNING if node == active_node else TEXT
            width_value = 4 if node in (selected, active_node) else 2
            self.canvas.create_oval(x - 25, y - 25, x + 25, y + 25, fill=fill, outline=outline, width=width_value)
            self.canvas.create_text(x, y, text=str(node), fill="#ffffff", font=("Segoe UI", 10, "bold"))
            accuracy = state["perNode"][node] if node < len(state["perNode"]) else None
            if accuracy is not None:
                self.canvas.create_text(x, y + 37, text=f"{float(accuracy):.1%}", fill=TEXT, font=("Segoe UI", 8))

        channel_label=config.get('channelNoiseSemantics','noisy channel') if config.get('useChannelNoise') else 'clean channel'
        summary = (
            f"{config.get('approach', 'BASIL').upper()} | {config.get('partitionStrategy',config.get('split', '?'))} | "
            f"{channel_label} | round {state['round']}/{config.get('nRounds', '?')}"
        )
        if state["average"] is not None:
            summary += f" | avg {state['average']:.1%} | worst {state['worst']:.1%}"
        self.status_label.configure(text=summary)
        self.run_text.set(
            f"{config.get('experimentName', 'Research run')}\n"
            f"Status: {state['status']} | EBM: {config.get('ebmMode', 'none')} | "
            f"channel={config.get('channelNoiseSemantics','historical')} "
            f"sigma={float(config.get('channelNoiseSigmaAbsolute',0.0) if config.get('channelNoiseSemantics')=='paper_absolute_gaussian' else config.get('channelNoiseSigmaRelative',config.get('channelNoiseSigma',0.0))):g}"
        )
        self._refresh_inspector(state["nodes"].get(selected))

    def _refresh_inspector(self, payload):
        if payload is self._inspector_payload and self.selected_node.get()==self._inspector_selection:return
        self._inspector_payload=payload; self._inspector_selection=self.selected_node.get()
        self.details.configure(state=tk.NORMAL)
        self.details.delete("1.0", tk.END)
        for tree in (self.candidate_tree, self.link_tree):
            tree.delete(*tree.get_children())
        if payload:
            attack = payload.get("attack", {})
            incoming = payload.get("incoming", {})
            training = payload.get("training", {})
            ebm = payload.get("ebm", {})
            cart = payload.get("cart", {})
            lines = [
                f"ROUND {payload.get('round')}  NODE {payload.get('nodeId')}",
                "Global test accuracy: "+('—' if payload.get('globalTestAccuracy') is None else f"{payload['globalTestAccuracy']:.2%}"),
                "",
                f"Attack configured : {attack.get('configured', False)}",
                f"Attack active     : {attack.get('active', False)}",
                f"Attack rel. norm  : {float(attack.get('relativeNorm', 0.0)):.5f}",
                "",
                f"Incoming source   : node {incoming.get('senderId', '-')}",
                f"Incoming round    : {incoming.get('sourceRound', '-')}",
                f"Incoming rel.noise: {float(incoming.get('relativeNoise', 0.0)):.5f}",
                f"Incoming attacked : {incoming.get('attacked', False)}",
                f"Incoming signature: {incoming.get('stateSignature', '-')}",
                "",
                f"State signature   : {training.get('stateSignature', '-')}",
                f"Momentum mode     : {training.get('momentumMode', '-')}",
                f"Momentum norm     : {float(training.get('momentumNorm', 0.0)):.5f}",
                f"Base grad norm    : {float(training.get('baseGradientNorm', 0.0)):.5f}",
                f"Preclip grad norm : {float(training.get('preclipGradientNorm', 0.0)):.5f}",
                f"Clip fraction     : {float(training.get('clipFraction', 0.0)):.3f}",
                "",
                f"EBM mode          : {ebm.get('mode', 'none')}",
                f"EBM coefficient   : {float(ebm.get('coefficient', 0.0)):.8g}",
            ]
            for key,label in (('stress','Stress EMA'),('requestedCoefficient','EBM requested'),('activeRatio','EBM active ratio'),('effectiveCoordinateSigma','Coordinate sigma')):
                if key in ebm:lines.append(f'{label}: {float(ebm[key]):.8g}')
            if cart:
                lines += ['',f"CART gap: {cart.get('gap','—')}",f"CART mu: {cart.get('mu','—')}",
                    f"Registry coverage: {cart.get('coverage','—')}",f"Claims accepted: {cart.get('accepted','—')}",f"Claims rejected: {cart.get('rejected','—')}"]
            if payload.get('telemetryComplete') is False:
                lines=['This older worker event includes accuracy only. Candidate/link details are unavailable.']
            self.details.insert("1.0", "\n".join(lines))
            for candidate in payload.get("selection", {}).get("candidates", []):
                self.candidate_tree.insert(
                    "",
                    tk.END,
                    values=(
                        candidate.get("senderId"),
                        f"{float(candidate.get('distance', 0.0)):.5f}",
                        "-" if candidate.get("loss") is None else f"{float(candidate['loss']):.5f}",
                        "yes" if candidate.get("plausible") else "no",
                        "yes" if candidate.get("selected") else "",
                    ),
                )
            for outgoing in payload.get("outgoing", []):
                self.link_tree.insert(
                    "",
                    tk.END,
                    values=(
                        outgoing.get("receiverId"),
                        f"{float(outgoing.get('noiseNorm', 0.0)):.5f}",
                        f"{float(outgoing.get('relativeNoise', 0.0)):.5f}",
                    ),
                )
        self.details.configure(state=tk.DISABLED)

    def _refresh_lane_values(self):
        lanes = sorted(self.runs)
        self.lane_combo.configure(values=tuple(f"Lane {lane + 1}" for lane in lanes))
        if lanes:
            current = self.active_lane.get()
            self.lane_combo.current(lanes.index(current) if current in lanes else 0)

    def _refresh_node_values(self, count):
        self.node_combo.configure(values=tuple(f"Node {node}" for node in range(count)))
        self.node_combo.current(min(self.selected_node.get(), count - 1))

    def _on_lane(self, _event=None):
        lanes = sorted(self.runs)
        index = self.lane_combo.current()
        if 0 <= index < len(lanes):
            self.active_lane.set(lanes[index])
            count = int(self.runs[lanes[index]]["config"].get("nNodes", 10))
            self._refresh_node_values(count)
            self.request_redraw()

    def _on_node(self, _event=None):
        index = self.node_combo.current()
        if index >= 0:
            self.selected_node.set(index)
            self.request_redraw()

    def _canvas_click(self, event):
        if not self._node_positions:
            return
        node = min(
            self._node_positions,
            key=lambda key: (self._node_positions[key][0] - event.x) ** 2 + (self._node_positions[key][1] - event.y) ** 2,
        )
        x, y = self._node_positions[node]
        if (x - event.x) ** 2 + (y - event.y) ** 2 <= 35 ** 2:
            self.selected_node.set(node)
            self.node_combo.current(node)
            self.request_redraw()

    def _choose_replay(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select research run metadata",
            filetypes=(("Run metadata", "run.json"), ("JSON", "*.json")),
        )
        if path:
            try:
                self.load_replay(path)
            except Exception as error:
                messagebox.showerror("Replay telemetry", str(error), parent=self.root)

    def load_replay(self, run_path):
        run_path = Path(run_path)
        metadata = json.loads(run_path.read_text(encoding="utf-8"))
        from basil_core.protocol_compatibility import is_research_protocol
        if is_research_protocol(metadata.get("experimentProtocol", metadata.get("config", {}).get("experimentProtocol"))):
            from gui.services.research_telemetry import load_replay
            replay=load_replay(run_path)
            self.start_run(0,replay["config"]);self._replay=replay
            rounds=len(replay["metrics"]["avg_history"])
            self.round_scale.configure(from_=0,to=max(0,rounds-1));self.round_scale.set(max(0,rounds-1))
            self._show_replay_round(max(0,rounds-1))
            return
        if int(metadata.get("config", {}).get("campaignVersion", 0)) != 4:
            raise ValueError("Network replay is unavailable for this result format.")
        telemetry_path = run_path.with_name("telemetry.npz")
        metrics_path = run_path.with_name("metrics.npz")
        if not telemetry_path.exists() or not metrics_path.exists():
            raise ValueError("Replay requires metrics.npz and telemetry.npz beside run.json.")
        with np.load(telemetry_path, allow_pickle=False) as archive:
            telemetry = {key: np.array(archive[key], copy=True) for key in archive.files}
        with np.load(metrics_path, allow_pickle=False) as archive:
            metrics = {key: np.array(archive[key], copy=True) for key in archive.files}
        self._replay = {"config": metadata["config"], "telemetry": telemetry, "metrics": metrics}
        self.start_run(0, metadata["config"])
        self._replay = {"config": metadata["config"], "telemetry": telemetry, "metrics": metrics}
        rounds = len(metrics.get("avg_history", []))
        self.round_scale.configure(from_=0, to=max(0, rounds - 1))
        self.round_scale.set(max(0, rounds - 1))
        self._show_replay_round(max(0, rounds - 1))

    def _on_replay_round(self, value):
        if self._replay is not None:
            self._show_replay_round(int(round(float(value))))

    def _show_replay_round(self, round_index):
        if self._replay is None:
            return
        config = self._replay["config"]
        telemetry = self._replay["telemetry"]
        metrics = self._replay["metrics"]
        rounds = len(metrics.get("avg_history", []))
        if rounds <= 0:
            return
        round_index = max(0, min(round_index, rounds - 1))
        state = self.runs[0]
        state["round"] = round_index + 1
        state["average"] = float(metrics["avg_history"][round_index])
        state["worst"] = float(metrics["worst_history"][round_index])
        state["perNode"] = metrics.get("per_node_history", np.zeros((rounds, int(config["nNodes"]))))[round_index].tolist()
        if "researchRecords" in self._replay:
            state["nodes"]={node:self._replay["researchRecords"][(round_index,node)] for node in range(10)}
            state["activeNode"]=9;self.round_label.configure(text=f"Round {round_index+1}/{rounds}")
            self.request_redraw()
            return
        for node in range(int(config["nNodes"])):
            senders = telemetry.get("candidate_senders", np.empty((0,)))[round_index, node] if "candidate_senders" in telemetry else []
            distances = telemetry.get("candidate_distances", np.empty((0,)))[round_index, node] if "candidate_distances" in telemetry else []
            losses = telemetry.get("candidate_losses", np.empty((0,)))[round_index, node] if "candidate_losses" in telemetry else []
            plausible = telemetry.get("candidate_plausible", np.empty((0,)))[round_index, node] if "candidate_plausible" in telemetry else []
            selected = int(telemetry.get("selected_sources", np.full((rounds, int(config["nNodes"])), -1))[round_index, node])
            candidates = []
            for sender, distance, loss, valid in zip(senders, distances, losses, plausible):
                if int(sender) < 0:
                    continue
                candidates.append(
                    {
                        "senderId": int(sender),
                        "distance": float(distance),
                        "loss": None if not np.isfinite(loss) else float(loss),
                        "plausible": bool(valid),
                        "selected": int(sender) == selected,
                    }
                )
            signature_value = telemetry.get(
                "state_signature",
                np.full((rounds, int(config["nNodes"])), b""),
            )[round_index, node]
            if isinstance(signature_value, bytes):
                signature_value = signature_value.decode("ascii")
            else:
                signature_value = str(signature_value)
            state["nodes"][node] = {
                "round": round_index + 1,
                "nodeId": node,
                "attack": {
                    "configured": node in {int(v) for v in str(config.get("attackerIds", "")).split(",") if v},
                    "active": bool(telemetry.get("attack_active", np.zeros((rounds, int(config["nNodes"])), bool))[round_index, node]),
                    "relativeNorm": float(telemetry.get("attack_relative_norm", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                },
                "incoming": {
                    "senderId": selected,
                    "sourceRound": round_index,
                    "relativeNoise": float(telemetry.get("selected_relative_noise", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                    "attacked": bool(telemetry.get("selected_attacked", np.zeros((rounds, int(config["nNodes"])), bool))[round_index, node]),
                },
                "selection": {"candidates": candidates},
                "training": {
                    "stateSignature": signature_value,
                    "momentumMode": config.get("optimizerStateMode", "visit_reset"),
                    "momentumNorm": float(telemetry.get("momentum_norm", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                    "baseGradientNorm": float(telemetry.get("base_gradient_norm", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                    "preclipGradientNorm": float(telemetry.get("preclip_gradient_norm", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                    "clipFraction": float(telemetry.get("clip_fraction", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                },
                "ebm": {
                    "mode": config.get("ebmMode", "none"),
                    "stress": float(telemetry.get("stress_ema", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                    "coefficient": float(telemetry.get("ebm_coefficient", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                    "activeRatio": float(telemetry.get("active_ebm_ratio", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                },
                "cart": {
                    "gap": float(telemetry.get("cart_gap", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                    "mu": float(telemetry.get("cart_mu", np.zeros((rounds, int(config["nNodes"]))))[round_index, node]),
                    "coverage": float(metrics.get("registry_coverage", np.zeros(rounds))[round_index]),
                    "accepted": int(telemetry.get("registry_accepted", np.zeros((rounds, int(config["nNodes"])), int))[round_index, node]),
                    "rejected": int(telemetry.get("registry_rejected", np.zeros((rounds, int(config["nNodes"])), int))[round_index, node]),
                },
                "outgoing": [],
            }
        self.round_label.configure(text=f"Round {round_index + 1}/{rounds}")
        self.request_redraw()


# Deprecated import alias for existing integrations, not a user-facing label.
CampaignNetworkView = NetworkView
__all__ = ["NetworkView", "CampaignNetworkView"]
