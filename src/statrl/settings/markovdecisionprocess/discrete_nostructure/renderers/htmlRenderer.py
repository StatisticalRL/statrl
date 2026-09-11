"""
HTML renderer for STATRL discrete, unstructured MDPs.

The renderer follows the same interface as the STATRL renderers:

    renderer.start(env)
    renderer.render(env, env.last)
    renderer.stop(env)

It produces a single self-contained HTML file containing:
  - an interactive SVG representation of the MDP,
  - action-colored transition edges,
  - transition probabilities and expected rewards,
  - initial-state probabilities,
  - trajectory playback,
  - play/pause, previous/next/first/last controls,
  - a timeline slider,
  - frame information and transition inspection.

No JavaScript, CSS, or Python packages are required by the generated HTML.

Python dependencies:
    numpy

The renderer itself does not require NetworkX, matplotlib, scipy, or a
browser-side library.
"""

from __future__ import annotations

import html
import json
import math
import os
import string
from pathlib import Path
from typing import Any


class HTMLRenderer:
    """Render a discrete MDP and its trajectory as a standalone HTML file.

    Parameters
    ----------
    output_dir:
        Directory in which the HTML file is written.
    filename:
        Optional filename. If None, ``<env.name>.html`` is used.
    width, height:
        Nominal dimensions of the SVG graph.
    autoplay:
        Whether the generated visualization starts playing immediately.
    fps:
        Default playback speed.
    show_probabilities:
        Display transition probabilities on edge labels.
    show_rewards:
        Display expected rewards on edge labels.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike = "renderings",
        filename: str | None = None,
        width: int = 1100,
        height: int = 700,
        autoplay: bool = False,
        fps: float = 2.0,
        show_probabilities: bool = True,
        show_rewards: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.width = int(width)
        self.height = int(height)
        self.autoplay = bool(autoplay)
        self.fps = float(fps)
        self.show_probabilities = bool(show_probabilities)
        self.show_rewards = bool(show_rewards)

        self.started = False
        self.frames: list[dict[str, Any]] = []
        self.model: dict[str, Any] | None = None
        self.output_path: Path | None = None

    # ------------------------------------------------------------------
    # Public renderer interface
    # ------------------------------------------------------------------

    def start(self, env) -> None:
        """Initialize the MDP representation and reset the trajectory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        name = getattr(env, "name", None)
        if name is None:
            name = getattr(env, "displayname", "DiscreteMDP")

        if self.filename is None:
            safe_name = "".join(
                c if c.isalnum() or c in "-_." else "_"
                for c in str(name)
            )
            self.output_path = self.output_dir / f"{safe_name}.html"
        else:
            self.output_path = self.output_dir / self.filename
            if self.output_path.suffix.lower() != ".html":
                self.output_path = self.output_path.with_suffix(".html")

        self.model = self._build_model(env)
        self.frames = []
        self.started = True

    def render(self, env, last) -> None:
        """Record one trajectory frame.

        ``last`` is the STATRL convention ``(state, action, reward)``.
        """
        if not self.started:
            self.start(env)

        state, action, reward = last

        frame = {
            "state": None if state is None else int(state),
            "action": None if action is None else int(action),
            "reward": float(reward) if reward is not None else 0.0,
            "step": len(self.frames),
            "transitions": []
        }

        # Add the currently selected transition when it can be inferred.
        if state is not None and action is not None:
            frame["transitions"] = self._transitions_from_state_action(
                env, int(state), int(action)
            )
        else:
            frame["transitions"] = []

        self.frames.append(frame)

    def stop(self, env) -> None:
        """Write the complete standalone HTML visualization."""
        if not self.started:
            return

        if self.model is None:
            self.model = self._build_model(env)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        document = self._build_html()
        assert self.output_path is not None
        self.output_path.write_text(document, encoding="utf-8")

        self.started = False

    # ------------------------------------------------------------------
    # MDP extraction
    # ------------------------------------------------------------------

    def _build_model(self, env) -> dict[str, Any]:
        nS = int(env.nS)
        nA = int(env.nA)

        action_names = self._action_names(env, nA)
        positions = self._layout(nS)

        nodes = []
        isd = getattr(env, "isd", [0.0] * nS)

        for s in range(nS):
            try:
                initial_probability = float(isd[s])
            except Exception:
                initial_probability = 0.0

            nodes.append(
                {
                    "id": s,
                    "label": str(s),
                    "x": positions[s][0],
                    "y": positions[s][1],
                    "initial_probability": initial_probability,
                }
            )

        edges = []
        edge_id = 0

        for s in range(nS):
            for a in range(nA):
                transitions = env.P[s][a]

                # Aggregate entries with the same destination. This is
                # important because a valid P[s][a] may contain repeated
                # destinations.
                aggregated: dict[int, dict[str, Any]] = {}

                for transition in transitions:
                    p, next_state, done = transition[:3]
                    next_state = int(next_state)
                    p = float(p)

                    if p <= 0:
                        continue

                    if next_state not in aggregated:
                        aggregated[next_state] = {
                            "probability": 0.0,
                            "done": False,
                        }

                    aggregated[next_state]["probability"] += p
                    aggregated[next_state]["done"] |= bool(done)

                try:
                    mean_reward = float(env.R[s][a].mean())
                except Exception:
                    try:
                        mean_reward = float(env.getMeanReward(s, a))
                    except Exception:
                        mean_reward = 0.0

                destinations = list(aggregated.items())
                for rank, (next_state, data) in enumerate(destinations):
                    edges.append(
                        {
                            "id": edge_id,
                            "source": s,
                            "target": next_state,
                            "action": a,
                            "action_name": action_names[a],
                            "probability": data["probability"],
                            "reward_mean": mean_reward,
                            "done": bool(data["done"]),
                            "curve_index": rank,
                            "curve_count": len(destinations),
                        }
                    )
                    edge_id += 1

        return {
            "name": str(
                getattr(
                    env,
                    "name",
                    getattr(env, "displayname", "DiscreteMDP"),
                )
            ),
            "nS": nS,
            "nA": nA,
            "nodes": nodes,
            "edges": edges,
            "actions": [
                {"id": a, "name": action_names[a]}
                for a in range(nA)
            ],
            "width": self.width,
            "height": self.height,
        }

    def _action_names(self, env, nA: int) -> list[str]:
        names = getattr(env, "nameActions", None)

        if names:
            result = [str(x) for x in names]
            if len(result) >= nA:
                return result[:nA]

        # STATRL's current TextRenderer uses A, B, ..., Z.
        # Continue with A0, A1, ... if there are more than 26 actions.
        result = []
        for a in range(nA):
            if a < 26:
                result.append(string.ascii_uppercase[a])
            else:
                result.append(f"A{a}")
        return result

    def _transitions_from_state_action(
        self, env, state: int, action: int
    ) -> list[dict[str, Any]]:
        result = []
        for p, next_state, done in env.P[state][action]:
            result.append(
                {
                    "next_state": int(next_state),
                    "probability": float(p),
                    "done": bool(done),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _layout(self, nS: int) -> dict[int, tuple[float, float]]:
        """Deterministic, dependency-free layout.

        The layout is deliberately stable across runs. Small MDPs are placed
        on a circle; larger MDPs use a rectangular grid. This is intentionally
        conservative: the browser visualization can remain completely
        dependency-free and the layout never jumps during playback.
        """
        margin_x = 90
        margin_y = 90

        if nS <= 1:
            return {0: (self.width / 2, self.height / 2)}

        if nS <= 12:
            cx = self.width / 2
            cy = self.height / 2
            radius = min(self.width, self.height) * 0.34

            positions = {}
            for s in range(nS):
                angle = -math.pi / 2 + 2 * math.pi * s / nS
                positions[s] = (
                    cx + radius * math.cos(angle),
                    cy + radius * math.sin(angle),
                )
            return positions

        columns = max(2, math.ceil(math.sqrt(nS * self.width / self.height)))
        rows = math.ceil(nS / columns)

        usable_w = self.width - 2 * margin_x
        usable_h = self.height - 2 * margin_y

        positions = {}
        for s in range(nS):
            row = s // columns
            col = s % columns

            x = (
                margin_x
                if columns == 1
                else margin_x + usable_w * col / (columns - 1)
            )
            y = (
                margin_y
                if rows == 1
                else margin_y + usable_h * row / (rows - 1)
            )
            positions[s] = (x, y)

        return positions

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def _build_html(self) -> str:
        assert self.model is not None

        model_json = json.dumps(
            self.model,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        frames_json = json.dumps(
            self.frames,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        title = html.escape(str(self.model["name"]))

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — STATRL MDP</title>

<style>
:root {{
    --bg: #f5f7fa;
    --panel: #ffffff;
    --text: #17202a;
    --muted: #697586;
    --border: #d9dee7;
    --accent: #2563eb;
    --current: #f59e0b;
    --possible: #dbeafe;
    --shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
}}

.app {{
    max-width: 1450px;
    margin: 0 auto;
    padding: 22px;
}}

.header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 20px;
    margin-bottom: 16px;
}}

.title {{
    font-size: 25px;
    font-weight: 700;
    letter-spacing: -0.02em;
}}

.subtitle {{
    color: var(--muted);
    margin-top: 4px;
    font-size: 14px;
}}

.layout {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 300px;
    gap: 16px;
}}

.panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow);
}}

.graph-panel {{
    min-height: 700px;
    overflow: hidden;
}}

#graph {{
    width: 100%;
    height: 700px;
    display: block;
    background:
        radial-gradient(circle at 50% 50%, #ffffff 0%, #fafbfd 75%);
}}

.side {{
    padding: 18px;
}}

.section {{
    padding-bottom: 18px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--border);
}}

.section:last-child {{
    border-bottom: none;
    margin-bottom: 0;
}}

.section-title {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: var(--muted);
    margin-bottom: 10px;
}}

.value {{
    font-size: 15px;
    font-weight: 600;
}}

.small {{
    font-size: 13px;
    color: var(--muted);
}}

.stat {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 5px 0;
    font-size: 14px;
}}

.controls {{
    margin-top: 16px;
    padding: 15px 18px;
}}

.buttons {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-bottom: 12px;
}}

button {{
    border: 1px solid var(--border);
    background: white;
    color: var(--text);
    border-radius: 8px;
    padding: 7px 11px;
    cursor: pointer;
    font-size: 13px;
}}

button:hover {{
    background: #f1f5f9;
}}

button.primary {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
}}

.timeline {{
    width: 100%;
}}

.timeline-row {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

#slider {{
    flex: 1;
}}

.frame-counter {{
    min-width: 90px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--muted);
    font-size: 13px;
}}

.speed {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
    font-size: 13px;
    color: var(--muted);
}}

select {{
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 5px 7px;
    background: white;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    font-size: 13px;
}}

.legend-line {{
    width: 28px;
    height: 4px;
    border-radius: 4px;
}}

.legend-node {{
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 2px solid #64748b;
}}

.edge {{
    fill: none;
    stroke-linecap: round;
    cursor: pointer;
    transition:
        stroke-width 120ms ease,
        opacity 120ms ease,
        filter 120ms ease;
}}

.edge:hover {{
    filter: drop-shadow(0 2px 4px rgba(0,0,0,.22));
}}

.edge.selected {{
    stroke-width: 5 !important;
    filter: drop-shadow(0 2px 5px rgba(15,23,42,.25));
}}

.edge-label {{
    font-size: 11px;
    pointer-events: none;
    font-weight: 600;
}}

.edge-label-bg {{
    fill: white;
    opacity: .92;
}}

.node {{
    cursor: pointer;
}}

.node-circle {{
    fill: white;
    stroke: #64748b;
    stroke-width: 2;
    transition: all 120ms ease;
}}

.node.current .node-circle {{
    fill: var(--current);
    stroke: #b45309;
    stroke-width: 3;
}}

.node.possible .node-circle {{
    fill: var(--possible);
    stroke: #3b82f6;
}}

.node.selected .node-circle {{
    stroke: #111827;
    stroke-width: 4;
}}

.node-label {{
    font-size: 14px;
    font-weight: 700;
    pointer-events: none;
    text-anchor: middle;
    dominant-baseline: central;
}}

.node.initial-marker {{
    fill: #64748b;
}}

.node.initial-text {{
    font-size: 10px;
    fill: #64748b;
    text-anchor: middle;
}}

.token {{
    fill: #111827;
    stroke: white;
    stroke-width: 2;
    transition: cx 180ms ease, cy 180ms ease;
}}

.terminal-marker {{
    fill: white;
    stroke: #64748b;
    stroke-width: 2;
}}

.tooltip {{
    position: fixed;
    z-index: 20;
    pointer-events: none;
    display: none;
    background: rgba(15,23,42,.96);
    color: white;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 12px;
    line-height: 1.45;
    box-shadow: 0 8px 24px rgba(0,0,0,.2);
    max-width: 260px;
}}

@media (max-width: 1000px) {{
    .layout {{
        grid-template-columns: 1fr;
    }}

    #graph {{
        height: 600px;
    }}

    .graph-panel {{
        min-height: 600px;
    }}
}}
</style>
</head>

<body>
<div class="app">

    <div class="header">
        <div>
            <div class="title">{title}</div>
            <div class="subtitle">
                Interactive discrete Markov decision process · STATRL
            </div>
        </div>
        <div class="small" id="model-summary"></div>
    </div>

    <div class="layout">

        <div>
            <div class="panel graph-panel">
                <svg id="graph"
     viewBox="0 0 {self.width} {self.height}"
     preserveAspectRatio="xMidYMid meet"
     aria-label="Interactive MDP graph">

    <defs id="defs"></defs>

    <g id="viewport">
        <g id="edges"></g>
        <g id="edge-labels"></g>
        <g id="nodes"></g>
        <g id="token-layer"></g>
    </g>
</svg>
            </div>

            <div class="panel controls">
                <div class="buttons">
                    <button onclick="firstFrame()">|&lt;</button>
                    <button onclick="previousFrame()">&lt;</button>
                    <button class="primary" id="play-button"
                            onclick="togglePlay()">▶ Play</button>
                    <button onclick="nextFrame()">&gt;</button>
                    <button onclick="lastFrame()">&gt;|</button>
    <button onclick="zoomGraph(1.2)">+</button>
    <button onclick="zoomGraph(1/1.2)">−</button>
    <button onclick="resetGraphView()">Reset view</button>
                </div>
                
                

                <div class="timeline-row">
                    <input id="slider"
                           class="timeline"
                           type="range"
                           min="0"
                           max="0"
                           value="0"
                           step="1"
                           oninput="goToFrame(Number(this.value))">
                    <div class="frame-counter" id="frame-counter">
                        0 / 0
                    </div>
                </div>

                <div class="speed">
                    Speed
                    <select id="speed" onchange="setSpeed(Number(this.value))">
                        <option value="0.5">0.5×</option>
                        <option value="1" selected>1×</option>
                        <option value="2">2×</option>
                        <option value="4">4×</option>
                        <option value="8">8×</option>
                    </select>
                </div>
            </div>
        </div>

        <div class="panel side">

            <div class="section">
                <div class="section-title">Current frame</div>

                <div class="stat">
                    <span>Step</span>
                    <span class="value" id="info-step">—</span>
                </div>

                <div class="stat">
                    <span>State</span>
                    <span class="value" id="info-state">—</span>
                </div>

                <div class="stat">
                    <span>Action</span>
                    <span class="value" id="info-action">—</span>
                </div>

                <div class="stat">
                    <span>Reward</span>
                    <span class="value" id="info-reward">—</span>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Selection</div>
                <div id="selection">
                    <div class="small">
                        Click a state or transition to inspect it.
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Actions</div>
                <div id="legend"></div>
            </div>

            <div class="section">
                <div class="section-title">Visualization</div>

                <label class="stat">
                    <span>Probabilities</span>
                    <input id="toggle-prob"
                           type="checkbox"
                           checked
                           onchange="redrawEdges()">
                </label>

                <label class="stat">
                    <span>Expected rewards</span>
                    <input id="toggle-reward"
                           type="checkbox"
                           checked
                           onchange="redrawEdges()">
                </label>

                <div class="small" style="margin-top:8px">
                    Edge color identifies the action. Edge opacity and width
                    represent transition probability.
                </div>
            </div>

        </div>
    </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const MODEL = {model_json};
const FRAMES = {frames_json};

const AUTOPLAY = {str(self.autoplay).lower()};
const BASE_FPS = {self.fps};

const SVG_NS = "http://www.w3.org/2000/svg";

let currentFrame = 0;
let playing = false;
let timer = null;
let speed = 1.0;
let selectedNode = null;
let selectedEdge = null;

// ------------------------------------------------------------
// Interactive graph state
// ------------------------------------------------------------

let zoom = 1.0;
let panX = 0.0;
let panY = 0.0;

let draggedNode = null;
let dragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragNodeX = 0;
let dragNodeY = 0;

const COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#ea580c",
    "#9333ea",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#7c3aed",
    "#475569"
];

const graph = document.getElementById("graph");
const edgesGroup = document.getElementById("edges");
const labelsGroup = document.getElementById("edge-labels");
const nodesGroup = document.getElementById("nodes");
const tokenLayer = document.getElementById("token-layer");
const defs = document.getElementById("defs");
const tooltip = document.getElementById("tooltip");

function updateViewport() {{
    const viewport = document.getElementById("viewport");

    viewport.setAttribute(
        "transform",
        `translate(${{panX}} ${{panY}}) scale(${{zoom}})`
    );
}}


function mousePosition(event) {{
    const rect = graph.getBoundingClientRect();

    const x =
        (event.clientX - rect.left) *
        MODEL.width / rect.width;

    const y =
        (event.clientY - rect.top) *
        MODEL.height / rect.height;

    return {{
        x: (x - panX) / zoom,
        y: (y - panY) / zoom
    }};
}}


document.getElementById("model-summary").textContent =
    MODEL.nS + " states · " + MODEL.nA + " actions";

document.getElementById("slider").max =
    Math.max(0, FRAMES.length - 1);

function esc(value) {{
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
}}

function svgElement(name, attrs = {{}}) {{
    const el = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) {{
        el.setAttribute(key, value);
    }}
    return el;
}}

function colorForAction(action) {{
    return COLORS[action % COLORS.length];
}}

function formatNumber(x) {{
    if (x === null || x === undefined || Number.isNaN(x)) return "—";
    if (Math.abs(x) >= 100) return x.toFixed(1);
    if (Math.abs(x) >= 1) return x.toFixed(2);
    return x.toFixed(3);
}}

function nodePosition(id) {{
    return MODEL.nodes.find(n => n.id === id);
}}

function edgeCurve(edge) {{
    const s = nodePosition(edge.source);
    const t = nodePosition(edge.target);

    if (edge.source === edge.target) {{
        const direction = (edge.action % 2 === 0) ? -1 : 1;
        const dx = 50;
        const dy = 70 * direction;

        return {{
            path:
                "M " + s.x + " " + (s.y - 22) +
                " C " + (s.x + dx) + " " + (s.y + dy) +
                " " + (s.x - dx) + " " + (s.y + dy) +
                " " + s.x + " " + (s.y + 22),
            labelX: s.x + dx * 0.65,
            labelY: s.y + dy * 0.55
        }};
    }}

    const dx = t.x - s.x;
    const dy = t.y - s.y;
    const length = Math.max(1, Math.sqrt(dx * dx + dy * dy));

    // Perpendicular unit vector.
    const px = -dy / length;
    const py = dx / length;

    // Separate parallel edges by action and destination rank.
    const centered =
        edge.curve_index - (edge.curve_count - 1) / 2;

    const curvature = 42 * centered;

    const mx = (s.x + t.x) / 2 + px * curvature;
    const my = (s.y + t.y) / 2 + py * curvature;

    return {{
        path:
            "M " + s.x + " " + s.y +
            " Q " + mx + " " + my +
            " " + t.x + " " + t.y,
        labelX: 0.25 * s.x + 0.5 * mx + 0.25 * t.x,
        labelY: 0.25 * s.y + 0.5 * my + 0.25 * t.y
    }};
}}

function addArrowMarker(action) {{
    const id = "arrow-" + action;
    if (document.getElementById(id)) return id;

    const marker = svgElement("marker", {{
        id: id,
        viewBox: "0 0 10 10",
        refX: "8",
        refY: "5",
        markerWidth: "7",
        markerHeight: "7",
        orient: "auto-start-reverse",
        markerUnits: "strokeWidth"
    }});

    const path = svgElement("path", {{
        d: "M 0 0 L 10 5 L 0 10 z",
        fill: colorForAction(action)
    }});

    marker.appendChild(path);
    defs.appendChild(marker);
    return id;
}}

function updateNodeSVG(node) {{
    const g = document.getElementById("node-" + node.id);

    if (!g) {{
        return;
    }}

    const circle = g.querySelector(".node-circle");
    const label = g.querySelector(".node-label");
    const initialMarker = g.querySelector(".initial-marker");
    const initialText = g.querySelector(".initial-text");

    if (circle) {{
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
    }}

    if (label) {{
        label.setAttribute("x", node.x);
        label.setAttribute("y", node.y);
    }}

    if (initialMarker) {{
        initialMarker.setAttribute("cx", node.x - 20);
        initialMarker.setAttribute("cy", node.y - 20);
    }}

    if (initialText) {{
        initialText.setAttribute("x", node.x);
        initialText.setAttribute("y", node.y + 37);
    }}
}}



function drawGraph() {{
    edgesGroup.innerHTML = "";
    labelsGroup.innerHTML = "";
    nodesGroup.innerHTML = "";
    tokenLayer.innerHTML = "";

    MODEL.edges.forEach(edge => {{
        const curve = edgeCurve(edge);
        const path = svgElement("path", {{
            d: curve.path,
            class: "edge",
            id: "edge-" + edge.id,
            stroke: colorForAction(edge.action),
            "stroke-width":
                String(1.5 + 4.5 * Math.sqrt(Math.max(0, edge.probability))),
            opacity:
                String(0.18 + 0.82 * Math.min(1, edge.probability)),
            "marker-end": "url(#" + addArrowMarker(edge.action) + ")"
        }});

        path.addEventListener("click", event => {{
            event.stopPropagation();
            selectEdge(edge.id);
        }});
        
        path.addEventListener("pointerdown", event => {{
            event.preventDefault();
            event.stopPropagation();
        
            draggedNode = node.id;
            dragging = true;
        
            const p = mousePosition(event);
        
            dragStartX = p.x;
            dragStartY = p.y;
        
            dragNodeX = node.x;
            dragNodeY = node.y;
        
            path.setPointerCapture(event.pointerId);
        }});

    path.addEventListener("pointermove", event => {{
        if (!dragging || draggedNode !== node.id) {{
            return;
        }}
    
        event.preventDefault();
    
        const p = mousePosition(event);
    
        node.x =
            dragNodeX + (p.x - dragStartX);
    
        node.y =
            dragNodeY + (p.y - dragStartY);
    
        updateNodeSVG(node);
        redrawGeometry();
    }});

    path.addEventListener("pointerup", event => {{
        if (draggedNode === node.id) {{
            dragging = false;
            draggedNode = null;
    
            try {{
                path.releasePointerCapture(event.pointerId);
            }} catch (_) {{}}
        }}
    }});

        path.addEventListener("mousemove", event => {{
            showTooltip(
                event,
                "<b>" + esc(edge.action_name) + "</b><br>" +
                "Probability: " + formatNumber(edge.probability) +
                "<br>Expected reward: " + formatNumber(edge.reward_mean) +
                (edge.done ? "<br><b>Terminal transition</b>" : "")
            );
        }});

        path.addEventListener("mouseleave", hideTooltip);

        edgesGroup.appendChild(path);

        const label = svgElement("g", {{
            id: "label-" + edge.id,
            class: "edge-label"
        }});

        const bg = svgElement("rect", {{
            class: "edge-label-bg",
            rx: "4",
            ry: "4"
        }});

        const text = svgElement("text", {{
            class: "edge-label",
            "text-anchor": "middle",
            fill: colorForAction(edge.action)
        }});

        const pieces = [];
        if (document.getElementById("toggle-prob").checked) {{
            pieces.push("p=" + formatNumber(edge.probability));
        }}
        if (
            document.getElementById("toggle-reward").checked &&
            Math.abs(edge.reward_mean) > 1e-12
        ) {{
            pieces.push("r̄=" + formatNumber(edge.reward_mean));
        }}

        text.textContent = pieces.join("  ");

        // Position the background after the text is rendered.
        const x = curve.labelX;
        const y = curve.labelY;
        text.setAttribute("x", x);
        text.setAttribute("y", y);

        label.appendChild(bg);
        label.appendChild(text);
        labelsGroup.appendChild(label);

        requestAnimationFrame(() => {{
            try {{
                const box = text.getBBox();
                bg.setAttribute("x", box.x - 4);
                bg.setAttribute("y", box.y - 2);
                bg.setAttribute("width", box.width + 8);
                bg.setAttribute("height", box.height + 4);
            }} catch (_) {{}}
        }});

        if (edge.done) {{
            const terminal = svgElement("circle", {{
                class: "terminal-marker",
                cx: curve.labelX,
                cy: curve.labelY + 14,
                r: "5"
            }});
            labelsGroup.appendChild(terminal);
        }}
    }});

    MODEL.nodes.forEach(node => {{
        const g = svgElement("g", {{
            class: "node",
            id: "node-" + node.id
        }});

        const circle = svgElement("circle", {{
            class: "node-circle",
            cx: node.x,
            cy: node.y,
            r: "22"
        }});

        const label = svgElement("text", {{
            class: "node-label",
            x: node.x,
            y: node.y
        }});
        label.textContent = node.label;

        g.appendChild(circle);
        g.appendChild(label);

        if (node.initial_probability > 0) {{
            const marker = svgElement("circle", {{
                class: "initial-marker",
                cx: node.x - 20,
                cy: node.y - 20,
                r: "5"
            }});
            g.appendChild(marker);

            const initialText = svgElement("text", {{
                class: "initial-text",
                x: node.x,
                y: node.y + 37
            }});
            initialText.textContent =
                "P₀=" + formatNumber(node.initial_probability);
            g.appendChild(initialText);
        }}

        g.addEventListener("click", event => {{
            event.stopPropagation();
            selectNode(node.id);
        }});

        g.addEventListener("mousemove", event => {{
            showTooltip(
                event,
                "<b>State " + esc(node.label) + "</b><br>" +
                "Initial probability: " +
                formatNumber(node.initial_probability)
            );
        }});

        g.addEventListener("mouseleave", hideTooltip);

        nodesGroup.appendChild(g);
    }});

    const token = svgElement("circle", {{
        class: "token",
        r: "9",
        id: "token"
    }});
    tokenLayer.appendChild(token);

    graph.addEventListener("click", () => {{
        selectedNode = null;
        selectedEdge = null;
        updateSelection();
        updateHighlighting();
    }});

    updateFrame();
}}


function updateFrame() {{
    if (FRAMES.length === 0) {{
        document.getElementById("frame-counter").textContent = "0 / 0";
        return;
    }}

    currentFrame = Math.max(
        0,
        Math.min(currentFrame, FRAMES.length - 1)
    );

    const frame = FRAMES[currentFrame];
    const slider = document.getElementById("slider");
    slider.value = currentFrame;

    document.getElementById("frame-counter").textContent =
        (currentFrame + 1) + " / " + FRAMES.length;

    document.getElementById("info-step").textContent =
        frame.step;

    document.getElementById("info-state").textContent =
        frame.state === null ? "—" : frame.state;

    document.getElementById("info-action").textContent =
        frame.action === null
            ? "—"
            : MODEL.actions[frame.action].name;

    document.getElementById("info-reward").textContent =
        formatNumber(frame.reward);

    updateHighlighting();
    updateToken();
    updateSelection();
}}


function updateHighlighting() {{
    document.querySelectorAll(".node").forEach(el => {{
        el.classList.remove("current");
        el.classList.remove("possible");
        el.classList.remove("selected");
    }});

    if (selectedNode !== null) {{
        const el = document.getElementById("node-" + selectedNode);
        if (el) el.classList.add("selected");
    }}

    if (FRAMES.length === 0) return;

    const frame = FRAMES[currentFrame];

    if (frame.state !== null) {{
        const current = document.getElementById("node-" + frame.state);
        if (current) current.classList.add("current");

        frame.transitions.forEach(t => {{
            if (t.probability <= 0) return;
            const possible =
                document.getElementById("node-" + t.next_state);
            if (possible && t.next_state !== frame.state) {{
                possible.classList.add("possible");
            }}
        }});
    }}

    document.querySelectorAll(".edge").forEach(el => {{
        el.classList.remove("selected");
        el.style.filter = "";
    }});

    if (selectedEdge !== null) {{
        const edge = document.getElementById("edge-" + selectedEdge);
        if (edge) edge.classList.add("selected");
    }}

    // Make the action actually played visually prominent.
    if (frame.action !== null && frame.state !== null) {{
        MODEL.edges.forEach(edge => {{
            if (
                edge.source === frame.state &&
                edge.action === frame.action
            ) {{
                const el = document.getElementById("edge-" + edge.id);
                if (el) {{
                    el.style.filter =
                        "drop-shadow(0 2px 4px rgba(15,23,42,.22))";
                    el.style.strokeWidth =
                        String(
                            2.5 +
                            5.0 * Math.sqrt(Math.max(0, edge.probability))
                        );
                }}
            }}
        }});
    }}
}}



function updateToken() {{
    const token = document.getElementById("token");
    if (!token || FRAMES.length === 0) {{
        if (token) token.style.display = "none";
        return;
    }}

    const frame = FRAMES[currentFrame];

    if (frame.state === null) {{
        token.style.display = "none";
        return;
    }}

    const node = nodePosition(frame.state);
    token.style.display = "block";
    token.setAttribute("cx", node.x);
    token.setAttribute("cy", node.y - 34);
}}

function redrawEdges() {{
    drawGraph();
}}

function goToFrame(index) {{
    currentFrame = Number(index);
    updateFrame();
}}

function nextFrame() {{
    if (FRAMES.length === 0) return;

    if (currentFrame >= FRAMES.length - 1) {{
        stopPlaying();
        return;
    }}

    currentFrame++;
    updateFrame();
}}

function previousFrame() {{
    if (FRAMES.length === 0) return;
    currentFrame = Math.max(0, currentFrame - 1);
    updateFrame();
}}

function firstFrame() {{
    currentFrame = 0;
    updateFrame();
}}

function lastFrame() {{
    if (FRAMES.length === 0) return;
    currentFrame = FRAMES.length - 1;
    updateFrame();
}}

function togglePlay() {{
    if (playing) {{
        stopPlaying();
    }} else {{
        startPlaying();
    }}
}}

function startPlaying() {{
    if (FRAMES.length <= 1) return;

    playing = true;
    document.getElementById("play-button").textContent = "⏸ Pause";

    clearInterval(timer);

    const interval = 1000 / (BASE_FPS * speed);
    timer = setInterval(() => {{
        if (currentFrame >= FRAMES.length - 1) {{
            stopPlaying();
        }} else {{
            currentFrame++;
            updateFrame();
        }}
    }}, interval);
}}

function stopPlaying() {{
    playing = false;
    document.getElementById("play-button").textContent = "▶ Play";
    clearInterval(timer);
    timer = null;
}}

function setSpeed(value) {{
    speed = Number(value);
    if (playing) {{
        stopPlaying();
        startPlaying();
    }}
}}

function selectNode(id) {{
    selectedNode = id;
    selectedEdge = null;

    const node = nodePosition(id);
    let actions = "";

    MODEL.actions.forEach(action => {{
        const edges = MODEL.edges.filter(
            e => e.source === id && e.action === action.id
        );

        if (edges.length === 0) return;

        actions +=
            "<div style='margin-top:8px'>" +
            "<b>" + esc(action.name) + "</b>";

        edges.forEach(edge => {{
            actions +=
                "<div class='small' style='margin-left:8px'>" +
                "→ " + edge.target +
                " &nbsp; p=" + formatNumber(edge.probability) +
                (edge.done ? " · terminal" : "") +
                "</div>";
        }});

        const reward = edges.length > 0 ? edges[0].reward_mean : 0;
        actions +=
            "<div class='small' style='margin-left:8px'>" +
            "r̄=" + formatNumber(reward) +
            "</div></div>";
    }});

    document.getElementById("selection").innerHTML =
        "<div class='value'>State " + node.label + "</div>" +
        "<div class='small'>Initial probability: " +
        formatNumber(node.initial_probability) +
        "</div>" +
        actions;

    updateHighlighting();
}}

function selectEdge(id) {{
    selectedEdge = id;
    selectedNode = null;

    const edge = MODEL.edges.find(e => e.id === id);
    if (!edge) return;

    document.getElementById("selection").innerHTML =
        "<div class='value'>" +
        esc(edge.action_name) +
        "</div>" +
        "<div class='stat'><span>From</span><span>" +
        edge.source + "</span></div>" +
        "<div class='stat'><span>To</span><span>" +
        edge.target + "</span></div>" +
        "<div class='stat'><span>Probability</span><span>" +
        formatNumber(edge.probability) +
        "</span></div>" +
        "<div class='stat'><span>Expected reward</span><span>" +
        formatNumber(edge.reward_mean) +
        "</span></div>" +
        (edge.done
            ? "<div class='small' style='margin-top:8px'>" +
              "Terminal transition</div>"
            : "");

    updateHighlighting();
}}

function showTooltip(event, content) {{
    tooltip.innerHTML = content;
    tooltip.style.display = "block";
    tooltip.style.left = (event.clientX + 12) + "px";
    tooltip.style.top = (event.clientY + 12) + "px";
}}

function hideTooltip() {{
    tooltip.style.display = "none";
}}

function buildLegend() {{
    const legend = document.getElementById("legend");

    MODEL.actions.forEach(action => {{
        const row = document.createElement("div");
        row.className = "legend-item";

        const line = document.createElement("span");
        line.className = "legend-line";
        line.style.background = colorForAction(action.id);

        const text = document.createElement("span");
        text.textContent = action.name;

        row.appendChild(line);
        row.appendChild(text);
        legend.appendChild(row);
    }});
}}


function zoomGraph(factor) {{
    zoom = Math.max(
        0.25,
        Math.min(5.0, zoom * factor)
    );

    updateViewport();
}}

function resetGraphView() {{
    zoom = 1.0;
    panX = 0.0;
    panY = 0.0;

    updateViewport();
}}

buildLegend();
drawGraph();

graph.addEventListener("wheel", event => {{
    event.preventDefault();

    const rect = graph.getBoundingClientRect();

    const mouseX =
        (event.clientX - rect.left) *
        MODEL.width / rect.width;

    const mouseY =
        (event.clientY - rect.top) *
        MODEL.height / rect.height;

    // Position in graph coordinates before zoom.
    const graphX =
        (mouseX - panX) / zoom;

    const graphY =
        (mouseY - panY) / zoom;

    const factor =
        event.deltaY < 0 ? 1.15 : 1 / 1.15;

    const newZoom =
        Math.max(0.25, Math.min(5.0, zoom * factor));

    if (newZoom === zoom) {{
        return;
    }}

    // Keep the point under the mouse fixed.
    panX =
        mouseX - graphX * newZoom;

    panY =
        mouseY - graphY * newZoom;

    zoom = newZoom;

    updateViewport();
}}, {{ passive: false }});

if (AUTOPLAY && FRAMES.length > 1) {{
    startPlaying();
}}
</script>
</body>
</html>
"""


# Backward-compatible alias if one prefers the shorter name.
GraphRenderer = HTMLRenderer


if __name__ == "__main__":
    # Minimal standalone demonstration with no STATRL installation required.
    class _Reward:
        def __init__(self, mean):
            self._mean = mean

        def mean(self):
            return self._mean

    class _DemoEnv:
        name = "DemoMDP"
        nS = 5
        nA = 2
        states = range(nS)
        actions = range(nA)
        isd = [1.0, 0.0, 0.0, 0.0, 0.0]
        P = {
            0: {
                0: [(1.0, 0, False)],
                1: [(0.7, 1, False), (0.3, 2, False)],
            },
            1: {
                0: [(1.0, 0, False)],
                1: [(0.8, 2, False), (0.2, 3, False)],
            },
            2: {
                0: [(0.5, 1, False), (0.5, 2, False)],
                1: [(0.9, 3, False), (0.1, 4, True)],
            },
            3: {
                0: [(1.0, 2, False)],
                1: [(1.0, 4, True)],
            },
            4: {
                0: [(1.0, 4, True)],
                1: [(1.0, 4, True)],
            },
        }
        R = {}
        for s in states:
            R[s] = {a: _Reward(0.0) for a in actions}
        nameActions = ["Left", "Right"]

    env = _DemoEnv()
    renderer = HTMLRenderer(output_dir="renderings", filename="demo_mdp.html")
    renderer.start(env)

    demo_frames = [
        (0, None, 0.0),
        (1, 1, 0.2),
        (2, 1, 0.4),
        (3, 1, 0.8),
        (4, 1, 1.0),
    ]

    for frame in demo_frames:
        renderer.render(env, frame)

    renderer.stop(env)
    print(f"Generated: {renderer.output_path}")
