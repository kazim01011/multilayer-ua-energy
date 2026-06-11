#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "overleaf" / "figures"


BLUE = "#0f5bc6"
LIGHT_BLUE = "#d8ebff"
GRAY = "#8b8b8b"
LIGHT_GRAY = "#eeeeee"
ORANGE = "#f26b21"
YELLOW = "#ffbf00"
RED = "#e31a1c"
GREEN = "#20964b"
PURPLE = "#6f4ab5"
INK = "#222222"


def rounded(ax, xy, w, h, fc, ec=INK, lw=1.4, radius=0.035, z=1):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color="#555555", lw=1.7, ms=13, rad=0.0, z=8, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=9, weight="normal", color=INK, ha="center", va="center", rotation=0):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        rotation=rotation,
        family="DejaVu Sans",
    )


def bs_marker(ax, x, y, idx, active=True, load=0.5):
    color = BLUE if active else GRAY
    fill = "#ffffff" if active else "#f7f7f7"
    ax.add_patch(Rectangle((x - 0.026, y - 0.026), 0.052, 0.052, facecolor=fill, edgecolor=color, lw=2.0, zorder=9))
    label(ax, x, y + 0.004, "BS", size=8.3, weight="bold", color=color)
    label(ax, x, y - 0.048, f"{idx}: {'ON' if active else 'SLEEP'}", size=7.5, color=INK)
    ax.add_patch(Rectangle((x - 0.032, y + 0.039), 0.064, 0.010, facecolor="#ffffff", edgecolor="#555555", lw=0.6, zorder=10))
    ax.add_patch(Rectangle((x - 0.031, y + 0.040), 0.062 * load, 0.008, facecolor=GREEN if load < 0.7 else ORANGE, edgecolor="none", zorder=11))


def draw_hotspot(ax, center=(0.365, 0.47), rx=0.135, ry=0.19):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 80):
        rj = 1 + 0.08 * math.sin(3 * t) + 0.06 * math.cos(5 * t)
        pts.append((center[0] + rx * rj * math.cos(t), center[1] + ry * rj * math.sin(t)))
    poly = Polygon(pts, closed=True, facecolor="#fff2e8", edgecolor=ORANGE, lw=1.8, linestyle="--", zorder=2)
    ax.add_patch(poly)
    label(ax, center[0] + 0.135, center[1] + 0.125, "traffic\nhotspot", size=8.5, color=ORANGE, ha="left")


def demand_color(v):
    if v < 0.35:
        return YELLOW
    if v < 0.70:
        return "#ff7f1a"
    return RED


def draw_snapshot(ax):
    rounded(ax, (0.035, 0.08), 0.555, 0.82, "#ffffff", BLUE, lw=1.5, radius=0.018)
    label(ax, 0.312, 0.864, "Dense HetNet Control Snapshot", size=14, weight="bold")
    label(ax, 0.065, 0.815, "Macro overlay, controllable small cells, UE traffic demand", size=8.6, color="#444444", ha="left")

    c = (0.315, 0.470)
    macro_r = 0.288
    ax.add_patch(Circle(c, macro_r, facecolor="#f8fbff", edgecolor="#1b4fa3", lw=1.8, zorder=1))
    label(ax, c[0] - 0.255, c[1] - 0.255, "macro service region", size=7.8, color="#1b4fa3", ha="left")
    draw_hotspot(ax)

    bs = [
        (0.300, 0.470, 0, True, 0.63),
        (0.405, 0.475, 1, True, 0.84),
        (0.370, 0.655, 2, False, 0.18),
        (0.205, 0.655, 3, True, 0.52),
        (0.165, 0.435, 4, False, 0.12),
        (0.245, 0.270, 5, False, 0.15),
        (0.390, 0.300, 6, True, 0.58),
    ]

    for x, y, _, active, _ in bs:
        ax.add_patch(
            Circle(
                (x, y),
                0.106,
                facecolor=LIGHT_BLUE if active else LIGHT_GRAY,
                edgecolor=BLUE if active else GRAY,
                lw=1.1,
                alpha=0.48 if active else 0.35,
                linestyle="-" if active else "--",
                zorder=2,
            )
        )

    rng = np.random.default_rng(7)
    ue_points = []
    # Background UEs.
    while len(ue_points) < 46:
        x = rng.uniform(c[0] - macro_r * 0.93, c[0] + macro_r * 0.93)
        y = rng.uniform(c[1] - macro_r * 0.93, c[1] + macro_r * 0.93)
        if (x - c[0]) ** 2 + (y - c[1]) ** 2 <= (macro_r * 0.93) ** 2:
            demand = float(rng.beta(1.7, 2.1))
            ue_points.append((x, y, demand))
    # Hotspot UEs.
    for _ in range(28):
        x = rng.normal(0.395, 0.054)
        y = rng.normal(0.470, 0.078)
        if (x - c[0]) ** 2 + (y - c[1]) ** 2 <= (macro_r * 0.93) ** 2:
            ue_points.append((x, y, float(rng.uniform(0.65, 1.0))))

    active_bs = [(x, y, idx) for x, y, idx, active, _ in bs if active]
    for i, (x, y, demand) in enumerate(ue_points):
        nearest = min(active_bs, key=lambda b: (b[0] - x) ** 2 + (b[1] - y) ** 2)
        if i % 5 == 0 or demand > 0.82:
            ax.plot([x, nearest[0]], [y, nearest[1]], color="#86bdfa", lw=0.65, alpha=0.55, zorder=3)
        ax.add_patch(Circle((x, y), 0.0065, facecolor=demand_color(demand), edgecolor="#703000", lw=0.45, zorder=8))

    # Explicit reassociation examples from sleeping cells.
    for start, end in [((0.158, 0.405), (0.300, 0.470)), ((0.360, 0.645), (0.405, 0.475)), ((0.255, 0.285), (0.390, 0.300))]:
        arrow(ax, start, end, color=ORANGE, lw=1.8, ms=11, rad=0.12, z=7)
    label(ax, 0.105, 0.300, "UEs near sleep cells\nare reassociated", size=8.1, color=ORANGE, ha="left")

    for x, y, idx, active, load in bs:
        bs_marker(ax, x, y, idx, active, load)

    # Macro/core anchor.
    ax.plot([0.300, 0.300], [0.728, 0.780], color="#333333", lw=1.1, zorder=9)
    ax.add_patch(Polygon([(0.282, 0.728), (0.318, 0.728), (0.300, 0.785)], closed=True, facecolor="#ffffff", edgecolor="#333333", lw=1.2, zorder=10))
    label(ax, 0.300, 0.802, "macro anchor", size=7.8)

    # Scale bar.
    ax.plot([0.070, 0.195], [0.135, 0.135], color=INK, lw=2.1)
    ax.plot([0.070, 0.070], [0.128, 0.142], color=INK, lw=1.5)
    ax.plot([0.195, 0.195], [0.128, 0.142], color=INK, lw=1.5)
    label(ax, 0.132, 0.113, "400 m", size=8)

    # Legend.
    rounded(ax, (0.055, 0.020), 0.500, 0.050, "#ffffff", "#777777", lw=0.9, radius=0.010)
    ax.add_patch(Rectangle((0.070, 0.035), 0.022, 0.022, facecolor="#ffffff", edgecolor=BLUE, lw=1.6))
    label(ax, 0.104, 0.046, "ON BS", size=7.2, ha="left")
    ax.add_patch(Rectangle((0.155, 0.035), 0.022, 0.022, facecolor="#f7f7f7", edgecolor=GRAY, lw=1.6))
    label(ax, 0.188, 0.046, "SLEEP BS", size=7.2, ha="left")
    for k, col in enumerate([YELLOW, "#ff7f1a", RED]):
        ax.add_patch(Circle((0.272 + 0.016 * k, 0.046), 0.006, facecolor=col, edgecolor="#703000", lw=0.3))
    label(ax, 0.331, 0.046, "UE demand", size=7.2, ha="left")
    ax.plot([0.414, 0.462], [0.046, 0.046], color="#86bdfa", lw=1.3)
    label(ax, 0.474, 0.046, "association", size=7.2, ha="left")


def controller_block(ax, x, y, title, body, color, fill="#ffffff", h=0.095):
    rounded(ax, (x, y), 0.310, h, fill, color, lw=1.35, radius=0.015)
    label(ax, x + 0.017, y + h - 0.027, title, size=10.4, weight="bold", color=color, ha="left")
    lines = body.split("\n")
    yy = y + h - 0.060
    for line in lines:
        label(ax, x + 0.017, yy, line, size=8.1, color=INK, ha="left")
        yy -= 0.028


def draw_controller(ax):
    rounded(ax, (0.630, 0.080), 0.335, 0.820, "#fbfbfb", "#777777", lw=1.1, radius=0.018)
    label(ax, 0.797, 0.864, "Network-Side Energy-Saving Control", size=12.8, weight="bold")

    x = 0.645
    controller_block(ax, x, 0.750, "1) Measurement reports", "UE positions/demands, RSRP/SINR,\nchannel gains, BS loads and states", "#444444", "#ffffff", h=0.105)
    controller_block(ax, x, 0.615, "2) SON / RIC / edge controller", "Runs policy at network side;\nnot a UE-side battery algorithm", PURPLE, "#f5effd", h=0.105)
    controller_block(ax, x, 0.480, "3) Multilayer UE graph", "association competition, interference,\nload pressure, temporal similarity", BLUE, "#eef5ff", h=0.105)
    controller_block(ax, x, 0.345, "4) ML-GNN decision + repair", "predict association scores and active cells;\nrepair infeasible QoS/load violations", GREEN, "#eef9f0", h=0.105)
    controller_block(ax, x, 0.195, "5) Control actions", "UE-to-BS association + BS ON/SLEEP vector;\nevaluate RAN energy and service feasibility", ORANGE, "#fff3ea", h=0.120)

    for y0, y1 in [(0.750, 0.720), (0.615, 0.585), (0.480, 0.450), (0.345, 0.315)]:
        arrow(ax, (0.797, y0), (0.797, y1), color="#666666", lw=1.5, ms=12)

    # Data/action arrows between snapshot and controller.
    arrow(ax, (0.592, 0.755), (0.645, 0.805), color="#555555", lw=1.4, ms=12, rad=0.02)
    label(ax, 0.616, 0.793, "reports", size=7.7, color="#555555", rotation=28)
    arrow(ax, (0.645, 0.255), (0.592, 0.365), color=ORANGE, lw=1.6, ms=12, rad=-0.06)
    label(ax, 0.616, 0.303, "commands", size=7.7, color=ORANGE, rotation=-61)

    rounded(ax, (0.650, 0.105), 0.300, 0.055, "#ffffff", "#999999", lw=0.9, radius=0.010)
    label(ax, 0.800, 0.133, r"RAN energy: active fixed power + load power + sleep-mode power", size=8.1, color="#333333")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.0, 8.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(ax, 0.5, 0.955, "Illustrative Dense Wireless Scenario for Switching-Aware User Association", size=17, weight="bold")
    draw_snapshot(ax)
    draw_controller(ax)

    out_png = OUT_DIR / "fig_hetnet_switching_control_scenario.png"
    out_pdf = OUT_DIR / "fig_hetnet_switching_control_scenario.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
