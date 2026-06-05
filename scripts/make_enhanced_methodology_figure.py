#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "overleaf" / "figures"


COLORS = {
    "assoc": "#e86f2d",
    "interf": "#2f6fda",
    "load": "#d79b18",
    "temp": "#2f9a4a",
    "purple": "#6f4ab5",
    "light_purple": "#f2ecfb",
    "gray": "#555555",
    "light_gray": "#f6f6f6",
    "blue_gray": "#eef4fb",
}


def rounded(ax, xy, w, h, fc, ec="#333333", lw=1.4, radius=0.03, z=1):
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


def arrow(ax, start, end, color="#555555", lw=2.0, ms=14, style="-|>", z=5, rad=0.0):
    arr = FancyArrowPatch(
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
    ax.add_patch(arr)
    return arr


def label(ax, x, y, text, size=10, weight="normal", color="#222222", ha="center", va="center", rotation=0):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        family="DejaVu Sans",
        rotation=rotation,
    )


def draw_feature_panel(ax):
    rounded(ax, (0.035, 0.20), 0.115, 0.60, "#ffffff", "#333333", lw=1.2, radius=0.018)
    label(ax, 0.0925, 0.745, "UE Feature\nEncoding", size=12, weight="bold")

    # Small feature matrix with green entries.
    x0, y0, w, h = 0.057, 0.47, 0.071, 0.145
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#fbfbfb", edgecolor="#777777", linewidth=1.0))
    rows, cols = 5, 4
    for r in range(rows + 1):
        ax.plot([x0, x0 + w], [y0 + h * r / rows] * 2, color="#9a9a9a", lw=0.55)
    for c in range(cols + 1):
        ax.plot([x0 + w * c / cols] * 2, [y0, y0 + h], color="#9a9a9a", lw=0.55)
    vals = [
        ["#35783c", "#3b8e45", "#44a64f", "#58ba5d"],
        ["#42a04a", "#37b84b", "#27783a", "#4fae55"],
        ["#2d7340", "#51bd56", "#35a64b", "#38824a"],
        ["#46aa50", "#2f8d42", "#58c45d", "#3a8f45"],
        ["#2f7a3e", "#3da348", "#4ab451", "#62c667"],
    ]
    for r in range(rows):
        for c in range(cols):
            cx = x0 + w * (c + 0.5) / cols
            cy = y0 + h * (rows - r - 0.5) / rows
            ax.add_patch(Circle((cx, cy), 0.0053, facecolor=vals[r][c], edgecolor="white", lw=0.4))

    bullets = [
        "position",
        "traffic demand",
        "RSRP / SINR",
        "time index",
        "BS radio state",
    ]
    for i, b in enumerate(bullets):
        y = 0.405 - 0.043 * i
        ax.add_patch(Circle((0.058, y), 0.0036, facecolor="#2f7d3b", edgecolor="none"))
        label(ax, 0.068, y, b, size=8.7, ha="left")


def draw_graph(ax, cx, cy, color):
    pts = [
        (cx - 0.023, cy + 0.020),
        (cx + 0.004, cy + 0.036),
        (cx + 0.028, cy + 0.014),
        (cx + 0.018, cy - 0.026),
        (cx - 0.020, cy - 0.022),
    ]
    edges = [(0, 1), (0, 4), (1, 3), (1, 4), (2, 3), (3, 4), (0, 3)]
    for a, b in edges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=color, lw=1.25, zorder=3)
    for p in pts:
        ax.add_patch(Circle(p, 0.007, facecolor="#f7f7f7", edgecolor="#111111", lw=0.8, zorder=4))


def draw_multilayer_planes(ax):
    label(ax, 0.305, 0.842, "Multilayer Wireless\nGraph Construction", size=10.8, weight="bold")
    layers = [
        ("Association", "strongest-BS competition", COLORS["assoc"], "#fff0e8", 0.704),
        ("Interference", "channel-profile similarity", COLORS["interf"], "#edf5ff", 0.562),
        ("Load", "demand and load pressure", COLORS["load"], "#fff7df", 0.420),
        ("Temporal", "spatial-traffic similarity", COLORS["temp"], "#edf9ee", 0.278),
    ]
    for name, desc, color, fill, y in layers:
        rounded(ax, (0.195, y - 0.055), 0.226, 0.104, fill, color, lw=1.35, radius=0.018)
        label(ax, 0.210, y + 0.020, f"{name} Layer", size=10.6, weight="bold", color=color, ha="left")
        label(ax, 0.210, y - 0.014, desc, size=8.3, ha="left")
        draw_graph(ax, 0.356, y, color)
        label(ax, 0.392, y - 0.040, r"$N \times N$", size=7.2)


def draw_gcn_streams(ax):
    label(ax, 0.548, 0.842, "Layer-Specific ML-GCN\nPropagation", size=10.8, weight="bold")
    specs = [
        ("Assoc.\nGCN", COLORS["assoc"], 0.704),
        ("Interf.\nGCN", COLORS["interf"], 0.562),
        ("Load\nGCN", COLORS["load"], 0.420),
        ("Temp.\nGCN", COLORS["temp"], 0.278),
    ]
    for txt, color, y in specs:
        arrow(ax, (0.422, y), (0.458, y), color="#666666", lw=1.7, ms=11)
        rounded(ax, (0.468, y - 0.034), 0.065, 0.068, "#ffffff", color, lw=1.2, radius=0.012)
        label(ax, 0.5005, y, txt, size=8.9)
        arrow(ax, (0.534, y), (0.570, y), color="#666666", lw=1.7, ms=11)
        # Embedding vector.
        rounded(ax, (0.578, y - 0.043), 0.024, 0.086, "#ffffff", "#222222", lw=1.0, radius=0.007)
        for k in range(3):
            ax.add_patch(Circle((0.590, y + 0.023 - 0.023 * k), 0.0067, facecolor=color, edgecolor="#111111", lw=0.35))
        label(ax, 0.616, y + 0.036, rf"$\tilde{{\mathbf{{h}}}}^{{({txt.split()[0].lower()[0]})}}$", size=8.0, ha="left")


def draw_fusion_and_decision(ax):
    # Curved colored streams into fusion.
    ys = [0.704, 0.562, 0.420, 0.278]
    colors = [COLORS["assoc"], COLORS["interf"], COLORS["load"], COLORS["temp"]]
    for y, color, rad in zip(ys, colors, [-0.10, -0.04, 0.04, 0.10]):
        arrow(ax, (0.602, y), (0.670, 0.500), color=color, lw=1.7, ms=11, rad=rad)

    rounded(ax, (0.670, 0.387), 0.106, 0.226, COLORS["light_purple"], COLORS["purple"], lw=1.5, radius=0.018)
    label(ax, 0.723, 0.552, "Cross-Layer\nFusion", size=11.2, weight="bold", color="#31245c")
    label(ax, 0.723, 0.500, "fixed fusion\nor attention", size=8.7, color="#31245c")
    label(ax, 0.723, 0.442, r"$\alpha_a,\alpha_i,\alpha_l,\alpha_t$", size=8.4, color="#31245c")

    arrow(ax, (0.776, 0.500), (0.812, 0.500), color="#555555", lw=1.8, ms=12)
    rounded(ax, (0.818, 0.443), 0.035, 0.114, "#ffffff", "#222222", lw=1.0, radius=0.010)
    for k in range(4):
        ax.add_patch(Circle((0.8355, 0.529 - 0.020 * k), 0.0064, facecolor=COLORS["purple"], edgecolor="#111111", lw=0.35))
    label(ax, 0.8355, 0.415, r"$\mathbf{E}$", size=10.5, weight="bold")
    arrow(ax, (0.853, 0.500), (0.888, 0.500), color="#555555", lw=1.8, ms=12)

    x = 0.895
    rounded(ax, (x, 0.600), 0.083, 0.060, "#ffffff", "#333333", lw=1.1, radius=0.012)
    label(ax, x + 0.0415, 0.630, "UE-BS\nsoftmax", size=8.7)
    arrow(ax, (x + 0.0415, 0.600), (x + 0.0415, 0.565), color="#555555", lw=1.6, ms=11)
    rounded(ax, (x, 0.505), 0.083, 0.060, "#ffffff", "#333333", lw=1.1, radius=0.012)
    label(ax, x + 0.0415, 0.535, "Active-BS\nselector", size=8.7)
    arrow(ax, (x + 0.0415, 0.505), (x + 0.0415, 0.470), color="#555555", lw=1.6, ms=11)
    rounded(ax, (x, 0.410), 0.083, 0.060, "#f0fbf1", COLORS["temp"], lw=1.3, radius=0.012)
    label(ax, x + 0.0415, 0.440, "QoS/load\nrepair", size=8.7)
    arrow(ax, (x + 0.0415, 0.410), (x + 0.0415, 0.375), color="#555555", lw=1.6, ms=11)
    rounded(ax, (x - 0.006, 0.307), 0.095, 0.068, "#edf5ff", COLORS["interf"], lw=1.3, radius=0.012)
    label(ax, x + 0.0415, 0.341, "association +\nactive BS set", size=8.5)
    arrow(ax, (x + 0.0415, 0.307), (x + 0.0415, 0.270), color="#555555", lw=1.6, ms=11)
    rounded(ax, (x - 0.006, 0.205), 0.095, 0.064, "#fff8e8", COLORS["load"], lw=1.3, radius=0.012)
    label(ax, x + 0.0415, 0.237, "energy +\nfeasibility metrics", size=8.3)


def draw_stage_labels(ax):
    stages = [
        (0.0925, "1. Features"),
        (0.310, "2. Relation layers"),
        (0.532, "3. Layer filters"),
        (0.723, "4. Fusion"),
        (0.936, "5. Feasible control"),
    ]
    for x, t in stages:
        label(ax, x, 0.105, t, size=10.2, weight="bold", color="#333333")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 8.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    label(ax, 0.5, 0.955, "Proposed Multilayer Graph Learning Framework for Energy-Aware User Association", size=17, weight="bold")
    rounded(ax, (0.178, 0.156), 0.805, 0.732, "#ffffff", "#444444", lw=1.1, radius=0.020, z=0)

    draw_feature_panel(ax)
    arrow(ax, (0.150, 0.500), (0.185, 0.500), color=COLORS["interf"], lw=2.0, ms=13)
    draw_multilayer_planes(ax)
    draw_gcn_streams(ax)
    draw_fusion_and_decision(ax)
    draw_stage_labels(ax)

    # Short notation strip.
    rounded(ax, (0.426, 0.164), 0.308, 0.058, "#fbf9ff", COLORS["purple"], lw=1.0, radius=0.014)
    label(
        ax,
        0.440,
        0.197,
        r"$\mathcal{G}=(\mathcal{V},\mathbf{X},\{\mathbf{A}^{(\ell)}\}_{\ell\in\mathcal{L}})$",
        size=8.8,
        ha="left",
        color="#2d2355",
    )
    label(ax, 0.440, 0.176, r"$\mathcal{L}=\{\mathrm{assoc.},\mathrm{interf.},\mathrm{load},\mathrm{temp.}\}$", size=8.0, ha="left", color="#2d2355")

    png = OUT_DIR / "fig_methodology_framework_enhanced.png"
    pdf = OUT_DIR / "fig_methodology_framework_enhanced.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
