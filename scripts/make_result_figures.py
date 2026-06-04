#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from mlua.active_selector import fit_active_selector
from mlua.config import ExperimentConfig, SimConfig
from mlua.experiment import (
    active_selector_assignments,
    mean_oracle_active_count,
    standardize_graphs,
)
from mlua.models import ModelConfig, build_model
from mlua.baselines import rsrp_association
from mlua.simulator import (
    generate_dataset,
    repair_assignment,
    evaluate_assignment,
)


ROOT = Path(__file__).resolve().parents[1]
OVERLEAF_FIG = ROOT / "overleaf" / "figures" / "results"
MATLAB_DIR = ROOT / "matlab_figures" / "results"
DATA_DIR = MATLAB_DIR / "data"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_HEAD = font(25, True)
F_TEXT = font(21)
F_SMALL = font(17)
F_TINY = font(14)

COLORS = {
    "oracle": (35, 105, 190),
    "ml_gcn": (45, 150, 70),
    "attn_ml_gcn": (130, 85, 200),
    "agg_gcn": (220, 135, 35),
    "flat_mlp": (130, 130, 130),
    "load_aware": (205, 85, 70),
    "rsrp": (100, 160, 210),
    "sinr": (100, 160, 210),
}

POLICY_LABELS = {
    "oracle": "Reference",
    "ml_gcn": "ML-GCN",
    "attn_ml_gcn": "Attn-ML-GCN",
    "agg_gcn": "Agg. GCN",
    "flat_mlp": "Flat MLP",
    "load_aware": "Load-aware",
    "rsrp": "RSRP",
    "sinr": "SINR",
}


def ensure_dirs() -> None:
    OVERLEAF_FIG.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def rounded(draw: ImageDraw.ImageDraw, xy, fill, outline=(80, 80, 80), width=2, radius=14):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_title(draw: ImageDraw.ImageDraw, width: int, text: str) -> None:
    box = draw.textbbox((0, 0), text, font=F_TITLE)
    draw.text(((width - (box[2] - box[0])) / 2, 24), text, font=F_TITLE, fill=(20, 20, 20))


def save_matlab_script(name: str, body: str) -> None:
    path = MATLAB_DIR / f"{name}.m"
    path.write_text(body.strip() + "\n", encoding="utf-8")


def save_png(img: Image.Image, name: str) -> None:
    img.save(OVERLEAF_FIG / f"{name}.png", dpi=(300, 300))


def policy_order() -> list[str]:
    return ["oracle", "ml_gcn", "attn_ml_gcn", "agg_gcn", "flat_mlp", "load_aware", "rsrp"]


def axis(draw, box, y_label: str, x_label: str = "", y_min=0.0, y_max=1.0, ticks=5):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=(40, 40, 40), width=2)
    draw.line((x0, y0, x0, y1), fill=(40, 40, 40), width=2)
    for t in range(ticks + 1):
        frac = t / ticks
        y = y1 - frac * (y1 - y0)
        val = y_min + frac * (y_max - y_min)
        draw.line((x0 - 6, y, x0, y), fill=(40, 40, 40), width=2)
        draw.text((x0 - 62, y - 9), f"{val:.2f}", font=F_TINY, fill=(35, 35, 35))
        if t > 0:
            draw.line((x0, y, x1, y), fill=(225, 225, 225), width=1)
    draw.text((x0, y0 - 34), y_label, font=F_SMALL, fill=(35, 35, 35))
    if x_label:
        draw.text(((x0 + x1) / 2 - 35, y1 + 52), x_label, font=F_SMALL, fill=(35, 35, 35))


def bar_panel(draw, box, labels, values, ylabel, colors, y_min=0.0, y_max=None, value_fmt="{:.2f}"):
    if y_max is None:
        y_max = max(values) * 1.15 if values else 1.0
    x0, y0, x1, y1 = box
    axis(draw, box, ylabel, y_min=y_min, y_max=y_max)
    n = len(labels)
    gap = 14
    bar_w = (x1 - x0 - gap * (n + 1)) / n
    for i, (lab, val, color) in enumerate(zip(labels, values, colors)):
        x = x0 + gap + i * (bar_w + gap)
        h = (val - y_min) / max(y_max - y_min, 1e-12) * (y1 - y0)
        y = y1 - h
        draw.rectangle((x, y, x + bar_w, y1), fill=color, outline=(60, 60, 60))
        draw.text((x + bar_w / 2 - 18, y - 23), value_fmt.format(val), font=F_TINY, fill=(25, 25, 25))
        draw.text((x + bar_w / 2 - 36, y1 + 12), lab, font=F_TINY, fill=(25, 25, 25))


def line_panel(draw, box, data: dict[str, tuple[list[float], list[float]]], ylabel, xlabel, y_min=0.0, y_max=1.0):
    x0, y0, x1, y1 = box
    axis(draw, box, ylabel, xlabel, y_min=y_min, y_max=y_max)
    all_x = [x for xs, _ in data.values() for x in xs]
    xmin, xmax = min(all_x), max(all_x)
    def px(x): return x0 + (x - xmin) / max(xmax - xmin, 1e-12) * (x1 - x0)
    def py(y):
        raw = y1 - (y - y_min) / max(y_max - y_min, 1e-12) * (y1 - y0)
        return max(y0, min(y1, raw))
    for policy, (xs, ys) in data.items():
        color = COLORS.get(policy, (80, 80, 80))
        pts = [(px(x), py(y)) for x, y in zip(xs, ys)]
        for a, b in zip(pts, pts[1:]):
            draw.line((a, b), fill=color, width=4)
        for x, y in pts:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(40, 40, 40))
    for x in sorted(set(all_x)):
        draw.text((px(x) - 14, y1 + 12), f"{x:g}", font=F_TINY, fill=(30, 30, 30))


def legend(draw, x, y, policies):
    for idx, p in enumerate(policies):
        yy = y + idx * 28
        draw.rectangle((x, yy, x + 20, yy + 14), fill=COLORS.get(p, (80, 80, 80)))
        draw.text((x + 30, yy - 3), POLICY_LABELS.get(p, p), font=F_SMALL, fill=(25, 25, 25))


def main_comparison():
    mean = pd.read_csv(ROOT / "results" / "multiseed_initial" / "summary_mean.csv")
    std = pd.read_csv(ROOT / "results" / "multiseed_initial" / "summary_std.csv")
    policies = policy_order()
    data = mean[mean["policy"].isin(policies)].set_index("policy").loc[policies].reset_index()
    data.to_csv(DATA_DIR / "main_comparison.csv", index=False)
    std[std["policy"].isin(policies)].set_index("policy").loc[policies].reset_index().to_csv(DATA_DIR / "main_comparison_std.csv", index=False)

    img = Image.new("RGB", (1900, 1080), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, 1900, "Main Comparative Performance Across Three Random Seeds")
    labels = [POLICY_LABELS[p] for p in policies]
    colors = [COLORS[p] for p in policies]
    bar_panel(draw, (120, 150, 890, 455), labels, data["energy_saving_vs_all_on"].tolist(), "Energy saving", colors, y_max=0.72)
    bar_panel(draw, (1030, 150, 1800, 455), labels, data["energy_gap_vs_oracle"].tolist(), "Reference energy gap", colors, y_max=1.45)
    bar_panel(draw, (120, 650, 890, 955), labels, data["active_bs"].tolist(), "Active BS count", colors, y_max=7.5, value_fmt="{:.1f}")
    bar_panel(draw, (1030, 650, 1800, 955), labels, data["served_ratio"].tolist(), "Served ratio", colors, y_min=0.94, y_max=1.01)
    save_png(img, "fig_results_main_comparison")

    save_matlab_script("fig_results_main_comparison", """
T = readtable('data/main_comparison.csv');
labels = categorical(T.policy);
labels = reordercats(labels, T.policy);
figure('Color','w','Position',[100 100 1200 700]);
tiledlayout(2,2,'TileSpacing','compact');
nexttile; bar(labels,T.energy_saving_vs_all_on); ylabel('Energy saving'); grid on;
nexttile; bar(labels,T.energy_gap_vs_oracle); ylabel('Reference energy gap'); grid on;
nexttile; bar(labels,T.active_bs); ylabel('Active BS count'); grid on;
nexttile; bar(labels,T.served_ratio); ylabel('Served ratio'); ylim([0.94 1.01]); grid on;
savefig('fig_results_main_comparison.fig');
""")


def sweep_figures():
    sweep = pd.read_csv(ROOT / "results" / "paper_sweeps" / "sweep_summary.csv")
    policies = ["oracle", "ml_gcn", "attn_ml_gcn", "agg_gcn", "load_aware", "rsrp"]
    sweep[sweep["policy"].isin(policies)].to_csv(DATA_DIR / "sweep_summary.csv", index=False)

    def line_data(kind, metric):
        out = {}
        sub = sweep[(sweep["sweep"] == kind) & (sweep["policy"].isin(policies))]
        for p in policies:
            s = sub[sub["policy"] == p].sort_values("value")
            out[p] = (s["value"].tolist(), s[metric].tolist())
        return out

    img = Image.new("RGB", (1900, 980), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, 1900, "Bandwidth and QoS Sensitivity")
    line_panel(draw, (130, 150, 820, 455), line_data("bandwidth", "energy_saving_vs_all_on"), "Energy saving", "Bandwidth (MHz)", y_min=-0.05, y_max=0.85)
    line_panel(draw, (1030, 150, 1720, 455), line_data("bandwidth", "served_ratio"), "Served ratio", "Bandwidth (MHz)", y_min=0.0, y_max=1.05)
    line_panel(draw, (130, 620, 820, 925), line_data("qos", "energy_saving_vs_all_on"), "Energy saving", "QoS target (Mbps)", y_max=0.8)
    line_panel(draw, (1030, 620, 1720, 925), line_data("qos", "active_bs"), "Active BS count", "QoS target (Mbps)", y_max=7.5)
    legend(draw, 1740, 190, policies)
    save_png(img, "fig_results_bandwidth_qos")

    img2 = Image.new("RGB", (1900, 980), "white")
    draw = ImageDraw.Draw(img2)
    draw_title(draw, 1900, "Scalability and Channel-Robustness Sweeps")
    line_panel(draw, (130, 150, 820, 455), line_data("density", "energy_saving_vs_all_on"), "Energy saving", "Number of UEs", y_max=0.8)
    line_panel(draw, (1030, 150, 1720, 455), line_data("density", "active_bs"), "Active BS count", "Number of UEs", y_max=7.5)
    line_panel(draw, (130, 620, 820, 925), line_data("shadowing", "energy_saving_vs_all_on"), "Energy saving", "Shadowing std. (dB)", y_max=0.8)
    line_panel(draw, (1030, 620, 1720, 925), line_data("shadowing", "served_ratio"), "Served ratio", "Shadowing std. (dB)", y_min=0.90, y_max=1.02)
    legend(draw, 1740, 190, policies)
    save_png(img2, "fig_results_scalability_robustness")

    save_matlab_script("fig_results_sweeps", """
T = readtable('data/sweep_summary.csv');
policies = {'oracle','ml_gcn','attn_ml_gcn','agg_gcn','load_aware','rsrp'};
figure('Color','w','Position',[100 100 1200 760]);
tiledlayout(2,2,'TileSpacing','compact');
settings = {'bandwidth','qos','density','shadowing'};
ylabels = {'Energy saving','Energy saving','Energy saving','Served ratio'};
metrics = {'energy_saving_vs_all_on','energy_saving_vs_all_on','energy_saving_vs_all_on','served_ratio'};
xlabels = {'Bandwidth (MHz)','QoS target (Mbps)','Number of UEs','Shadowing std. (dB)'};
for k=1:4
    nexttile; hold on;
    for p=1:numel(policies)
        S = T(strcmp(T.sweep,settings{k}) & strcmp(T.policy,policies{p}),:);
        S = sortrows(S,'value');
        plot(S.value,S.(metrics{k}),'-o','LineWidth',1.8);
    end
    xlabel(xlabels{k}); ylabel(ylabels{k}); grid on;
end
legend(policies,'Location','bestoutside');
savefig('fig_results_sweeps.fig');
""")


def ablation_attention_figures():
    ab = pd.read_csv(ROOT / "results" / "ablation_initial" / "summary.csv")
    diag = pd.read_csv(ROOT / "results" / "ablation_initial" / "diagnostics.csv")
    ab.to_csv(DATA_DIR / "layer_ablation.csv", index=False)
    diag.to_csv(DATA_DIR / "attention.csv", index=False)

    full_es = float(ab.loc[ab["policy"] == "ml_gcn", "energy_saving_vs_all_on"].iloc[0])
    flat_es = float(pd.read_csv(ROOT / "results" / "multiseed_initial" / "summary_mean.csv").loc[lambda d: d["policy"] == "flat_mlp", "energy_saving_vs_all_on"].iloc[0])
    layers = ["association", "interference", "load", "temporal"]
    leave_vals = []
    only_vals = []
    for layer in layers:
        leave_es = float(ab.loc[ab["policy"] == f"ml_gcn_no_{layer}", "energy_saving_vs_all_on"].iloc[0])
        only_es = float(ab.loc[ab["policy"] == f"ml_gcn_{layer}", "energy_saving_vs_all_on"].iloc[0])
        leave_vals.append(full_es - leave_es)
        only_vals.append(only_es - flat_es)

    img = Image.new("RGB", (1900, 780), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, 1900, "Layer Ablation and Single-Layer Contribution")
    labels = ["Assoc.", "Interf.", "Load", "Temp."]
    colors = [(220, 105, 35), (55, 110, 205), (218, 160, 40), (55, 150, 65)]
    bar_panel(draw, (150, 160, 850, 610), labels, leave_vals, "Full ES - ES without layer", colors, y_min=0.0, y_max=max(0.03, max(leave_vals) * 1.4), value_fmt="{:.3f}")
    bar_panel(draw, (1030, 160, 1730, 610), labels, only_vals, "Single-layer ES - Flat ES", colors, y_min=0.0, y_max=max(0.025, max(only_vals) * 1.5), value_fmt="{:.3f}")
    save_png(img, "fig_results_layer_ablation")

    attn = diag[diag["policy"] == "attn_ml_gcn"].iloc[0]
    attn_vals = [float(attn[f"attention_{l}"]) for l in layers]
    # Dissimilarity from a representative snapshot.
    cfg = SimConfig(num_bs=7, num_ues=50, area_radius_m=650.0, bandwidth_hz=20e6, qos_mbps=1.0, seed=17)
    graph = generate_dataset(cfg, 1, seed_offset=10000)[0]
    diss = np.zeros((4, 4), dtype=float)
    for i, a in enumerate(layers):
        for j, b in enumerate(layers):
            denom = np.linalg.norm(graph.layers[a]) + 1e-12
            diss[i, j] = np.linalg.norm(graph.layers[a] - graph.layers[b]) / denom
    pd.DataFrame(diss, index=layers, columns=layers).to_csv(DATA_DIR / "layer_dissimilarity.csv")

    img2 = Image.new("RGB", (1900, 820), "white")
    draw = ImageDraw.Draw(img2)
    draw_title(draw, 1900, "Layer Attention and Relation Dissimilarity")
    bar_panel(draw, (160, 170, 760, 620), labels, attn_vals, "Attention weight", colors, y_max=0.35, value_fmt="{:.3f}")
    hm = (1030, 165, 1570, 705)
    cell = (hm[2] - hm[0]) / 4
    max_d = float(diss.max())
    for i in range(4):
        for j in range(4):
            val = diss[i, j]
            shade = int(255 - 165 * val / max(max_d, 1e-12))
            color = (shade, shade + 20 if shade < 235 else 255, 255)
            x0 = hm[0] + j * cell
            y0 = hm[1] + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(80, 80, 80))
            draw.text((x0 + cell / 2 - 22, y0 + cell / 2 - 10), f"{val:.2f}", font=F_SMALL, fill=(20, 20, 20))
    for i, lab in enumerate(labels):
        draw.text((hm[0] - 80, hm[1] + i * cell + cell / 2 - 10), lab, font=F_SMALL, fill=(20, 20, 20))
        draw.text((hm[0] + i * cell + 12, hm[3] + 18), lab, font=F_SMALL, fill=(20, 20, 20))
    draw.text((1030, 120), "Pairwise support dissimilarity", font=F_HEAD, fill=(35, 35, 35))
    save_png(img2, "fig_results_attention_redundancy")

    pd.DataFrame({"layer": layers, "leave_one_out_gain": leave_vals, "single_layer_gain": only_vals, "attention": attn_vals}).to_csv(DATA_DIR / "ablation_attention_summary.csv", index=False)
    save_matlab_script("fig_results_layer_analysis", """
A = readtable('data/ablation_attention_summary.csv');
D = readmatrix('data/layer_dissimilarity.csv','NumHeaderLines',1);
figure('Color','w','Position',[100 100 1200 720]);
tiledlayout(2,2,'TileSpacing','compact');
nexttile; bar(categorical(A.layer),A.leave_one_out_gain); ylabel('Full ES - ES without layer'); grid on;
nexttile; bar(categorical(A.layer),A.single_layer_gain); ylabel('Single-layer ES - Flat ES'); grid on;
nexttile; bar(categorical(A.layer),A.attention); ylabel('Attention weight'); grid on;
nexttile; imagesc(D(:,2:end)); colorbar; title('Layer dissimilarity'); axis square;
savefig('fig_results_layer_analysis.fig');
""")


def surface_energy_figure():
    path = ROOT / "results" / "surface_sweep" / "surface_summary.csv"
    if not path.exists():
        print("Skipping surface figure; run scripts/run_surface_sweep.py first.")
        return
    df = pd.read_csv(path)
    df = df[df["policy"] == "ml_gcn"].copy()
    df.to_csv(DATA_DIR / "surface_energy.csv", index=False)

    qos_vals = sorted(df["qos_mbps"].unique())
    ue_vals = sorted(df["num_ues"].unique())
    shadow_vals = sorted(df["shadowing_db"].unique())
    z_min = float(df["energy_saving_vs_all_on"].min())
    z_max = float(df["energy_saving_vs_all_on"].max())

    img = Image.new("RGB", (1700, 1050), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, 1700, "Joint QoS-Density Surface for ML-GCN Energy Saving")

    origin = (850, 660)
    x_vec = (95, 42)
    y_vec = (-105, 42)
    z_vec = (0, -390)

    def norm(value, values):
        return (value - min(values)) / max(max(values) - min(values), 1e-12)

    def project(qos, ue, z):
        xn = norm(qos, qos_vals)
        yn = norm(ue, ue_vals)
        zn = (z - z_min) / max(z_max - z_min, 1e-12)
        return (
            origin[0] + x_vec[0] * (xn * 3.0) + y_vec[0] * (yn * 3.0) + z_vec[0] * zn,
            origin[1] + x_vec[1] * (xn * 3.0) + y_vec[1] * (yn * 3.0) + z_vec[1] * zn,
        )

    # Axes.
    p000 = project(min(qos_vals), min(ue_vals), z_min)
    px = project(max(qos_vals), min(ue_vals), z_min)
    py = project(min(qos_vals), max(ue_vals), z_min)
    pz = project(min(qos_vals), min(ue_vals), z_max)
    for a, b in [(p000, px), (p000, py), (p000, pz)]:
        draw.line((a, b), fill=(45, 45, 45), width=3)
    draw.text((px[0] - 80, px[1] + 50), "QoS threshold (Mbps)", font=F_TEXT, fill=(25, 25, 25))
    draw.text((py[0] - 175, py[1] + 34), "Number of UEs", font=F_TEXT, fill=(25, 25, 25))
    draw.text((pz[0] - 85, pz[1] - 38), "Energy saving", font=F_TEXT, fill=(25, 25, 25))

    for q in qos_vals:
        p = project(q, min(ue_vals), z_min)
        draw.line((p[0], p[1] - 5, p[0], p[1] + 5), fill=(60, 60, 60), width=2)
        draw.text((p[0] - 12, p[1] + 10), f"{q:g}", font=F_TINY, fill=(30, 30, 30))
    for n in ue_vals:
        p = project(min(qos_vals), n, z_min)
        draw.line((p[0] - 5, p[1], p[0] + 5, p[1]), fill=(60, 60, 60), width=2)
        draw.text((p[0] - 44, p[1] + 2), f"{int(n)}", font=F_TINY, fill=(30, 30, 30))
    for z in np.linspace(z_min, z_max, 5):
        p = project(min(qos_vals), min(ue_vals), float(z))
        draw.line((p[0] - 5, p[1], p[0] + 5, p[1]), fill=(60, 60, 60), width=2)
        draw.text((p[0] - 52, p[1] - 8), f"{z:.2f}", font=F_TINY, fill=(30, 30, 30))

    base_colors = {
        shadow_vals[0]: (255, 70, 25),
        shadow_vals[1]: (20, 210, 205),
        shadow_vals[2]: (25, 65, 235),
    }
    cells = []
    for shadow in shadow_vals:
        pivot = df[df["shadowing_db"] == shadow].pivot(index="num_ues", columns="qos_mbps", values="energy_saving_vs_all_on")
        for yi in range(len(ue_vals) - 1):
            for xi in range(len(qos_vals) - 1):
                qs = [qos_vals[xi], qos_vals[xi + 1], qos_vals[xi + 1], qos_vals[xi]]
                ns = [ue_vals[yi], ue_vals[yi], ue_vals[yi + 1], ue_vals[yi + 1]]
                zs = [float(pivot.loc[n, q]) for q, n in zip(qs, ns)]
                pts = [project(q, n, z) for q, n, z in zip(qs, ns, zs)]
                depth = sum(ns) + 0.2 * sum(qs) - 100 * sum(zs)
                color = base_colors[shadow]
                shade = 0.82 + 0.18 * (sum(zs) / len(zs) - z_min) / max(z_max - z_min, 1e-12)
                fill = tuple(min(255, int(c * shade + 255 * (1 - shade))) for c in color)
                cells.append((depth, pts, fill, color))
    for _, pts, fill, edge in sorted(cells, key=lambda x: x[0], reverse=True):
        draw.polygon(pts, fill=fill, outline=(170, 170, 170))
        draw.line(pts + [pts[0]], fill=edge, width=2)

    legend_x, legend_y = 1180, 260
    rounded(draw, (legend_x - 25, legend_y - 35, legend_x + 360, legend_y + 100), (255, 255, 255), (120, 120, 120))
    draw.text((legend_x, legend_y - 18), "Shadowing level", font=F_HEAD, fill=(30, 30, 30))
    for idx, shadow in enumerate(shadow_vals):
        y = legend_y + 24 + idx * 26
        draw.rectangle((legend_x, y, legend_x + 36, y + 16), fill=base_colors[shadow], outline=(60, 60, 60))
        draw.text((legend_x + 50, y - 3), f"{shadow:g} dB", font=F_SMALL, fill=(30, 30, 30))

    draw.text(
        (180, 900),
        "Surface metric: mean energy saving of ML-GCN after QoS/load repair.",
        font=F_TEXT,
        fill=(35, 35, 35),
    )
    save_png(img, "fig_results_surface_energy")

    save_matlab_script("fig_results_surface_energy", """
T = readtable('data/surface_energy.csv');
qos = unique(T.qos_mbps)';
ues = unique(T.num_ues)';
shadows = unique(T.shadowing_db)';
[Q,N] = meshgrid(qos, ues);
figure('Color','w','Position',[100 100 1000 720]); hold on;
colors = [1 0.15 0.05; 0 0.80 0.80; 0.05 0.20 0.95];
for s = 1:numel(shadows)
    S = T(T.shadowing_db == shadows(s),:);
    Z = nan(numel(ues), numel(qos));
    for i = 1:numel(ues)
        for j = 1:numel(qos)
            row = S(S.num_ues == ues(i) & S.qos_mbps == qos(j),:);
            Z(i,j) = row.energy_saving_vs_all_on(1);
        end
    end
    surf(Q,N,Z,'FaceColor',colors(s,:),'FaceAlpha',0.78,'EdgeColor',[0.55 0.55 0.55]);
end
xlabel('QoS threshold (Mbps)');
ylabel('Number of UEs');
zlabel('Energy saving');
legend(strcat(string(shadows),' dB shadowing'),'Location','northeast');
grid on; view(42,28);
savefig('fig_results_surface_energy.fig');
""")


def color_for_label(label: int):
    palette = [(35, 105, 190), (220, 105, 35), (45, 150, 70), (130, 85, 200), (205, 85, 70), (80, 160, 175), (175, 145, 55)]
    return palette[label % len(palette)]


def representative_snapshot():
    cfg = ExperimentConfig(
        run_name="representative",
        train_graphs=60,
        test_graphs=1,
        epochs=80,
        patience=20,
        hidden_dim=24,
        learning_rate=0.018,
        seed=17,
        sim=SimConfig(num_bs=7, num_ues=50, area_radius_m=650.0, bandwidth_hz=20e6, qos_mbps=1.0, seed=17),
    )
    train = generate_dataset(cfg.sim, cfg.train_graphs, seed_offset=0)
    test = generate_dataset(cfg.sim, cfg.test_graphs, seed_offset=10000)
    standardize_graphs(train, test)
    graph = test[0]
    model_cfg = ModelConfig(graph.features.shape[1], cfg.hidden_dim, cfg.sim.num_bs, cfg.learning_rate, cfg.seed)
    model = build_model("ml_gcn", model_cfg)
    model.fit(train, [], cfg.epochs, cfg.patience)
    train_probs = model.predict_proba_graphs(train)
    probs = model.predict_proba_graphs(test)
    selector = fit_active_selector(train, train_probs, cfg.sim, seed=17)
    scores = selector.predict_scores(test, probs, cfg.sim)
    active_count = mean_oracle_active_count(train)
    preds, active_sets = active_selector_assignments(probs, scores, active_count)
    ml_labels = repair_assignment(cfg.sim, preds[0], graph.context["gain"], graph.context["demand_mbps"], active_sets[0])
    oracle_labels = graph.labels
    rsrp_labels = rsrp_association(graph.context["gain"])

    np.savetxt(DATA_DIR / "snapshot_sinr.csv", graph.context["sinr_all_on"], delimiter=",")
    pd.DataFrame(graph.context["ue_pos"], columns=["x", "y"]).assign(
        oracle=oracle_labels, ml_gcn=ml_labels, rsrp=rsrp_labels, demand=graph.context["demand_mbps"]
    ).to_csv(DATA_DIR / "snapshot_ues.csv", index=False)
    pd.DataFrame(graph.context["bs_pos"], columns=["x", "y"]).to_csv(DATA_DIR / "snapshot_bs.csv", index=False)

    img = Image.new("RGB", (1900, 920), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, 1900, "Representative Test Snapshot")

    # Heatmap
    heat = graph.context["sinr_all_on"]
    db = 10 * np.log10(np.maximum(heat, 1e-12))
    order = np.argsort(oracle_labels)
    db = db[order]
    hm = (80, 145, 640, 745)
    rows, cols = db.shape
    cw, ch = (hm[2] - hm[0]) / cols, (hm[3] - hm[1]) / rows
    vmin, vmax = np.percentile(db, 5), np.percentile(db, 95)
    for i in range(rows):
        for j in range(cols):
            frac = float(np.clip((db[i, j] - vmin) / max(vmax - vmin, 1e-12), 0, 1))
            color = (int(245 - 210 * frac), int(245 - 105 * frac), int(255 - 35 * (1 - frac)))
            x0, y0 = hm[0] + j * cw, hm[1] + i * ch
            draw.rectangle((x0, y0, x0 + cw + 1, y0 + ch + 1), fill=color)
    draw.rectangle(hm, outline=(45, 45, 45), width=2)
    draw.text((hm[0], 105), "All-on SINR heatmap (UE x BS)", font=F_HEAD, fill=(35, 35, 35))
    for j in range(cols):
        draw.text((hm[0] + j * cw + cw / 2 - 5, hm[3] + 10), str(j), font=F_TINY, fill=(20, 20, 20))

    def draw_map(box, labels, title):
        x0, y0, x1, y1 = box
        rounded(draw, box, (252, 252, 252), (90, 90, 90))
        draw.text((x0 + 25, y0 + 22), title, font=F_HEAD, fill=(35, 35, 35))
        cx, cy, rr = (x0 + x1) / 2, (y0 + y1) / 2 + 30, min(x1 - x0, y1 - y0) * 0.36
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=(80, 105, 140), width=3, fill=(243, 248, 255))
        def px(pos):
            return cx + pos[0] / cfg.sim.area_radius_m * rr, cy - pos[1] / cfg.sim.area_radius_m * rr
        active = set(np.unique(labels).astype(int))
        for bi, bpos in enumerate(graph.context["bs_pos"]):
            bx, by = px(bpos)
            fill = (35, 105, 190) if bi in active else (180, 180, 180)
            draw.rounded_rectangle((bx - 13, by - 13, bx + 13, by + 13), radius=5, fill=fill, outline=(35, 35, 35))
            draw.text((bx - 4, by + 17), str(bi), font=F_TINY, fill=(20, 20, 20))
        for pos, lab, demand in zip(graph.context["ue_pos"], labels, graph.context["demand_mbps"]):
            ux, uy = px(pos)
            size = 4 + demand / cfg.sim.demand_high_mbps * 5
            draw.ellipse((ux - size, uy - size, ux + size, uy + size), fill=color_for_label(int(lab)), outline=(40, 40, 40))
    draw_map((710, 145, 1260, 745), oracle_labels, "Reference association")
    draw_map((1320, 145, 1870, 745), ml_labels, "ML-GCN association")

    ref = evaluate_assignment(cfg.sim, oracle_labels, graph.context["gain"], graph.context["demand_mbps"])
    ml = evaluate_assignment(cfg.sim, ml_labels, graph.context["gain"], graph.context["demand_mbps"])
    draw.text((720, 790), f"Reference: energy={ref['energy_w']:.1f} W, active BSs={ref['active_bs']:.0f}, served={ref['served_ratio']:.2f}", font=F_TEXT, fill=(35, 35, 35))
    draw.text((1325, 790), f"ML-GCN: energy={ml['energy_w']:.1f} W, active BSs={ml['active_bs']:.0f}, served={ml['served_ratio']:.2f}", font=F_TEXT, fill=(35, 35, 35))
    save_png(img, "fig_results_representative_snapshot")

    save_matlab_script("fig_results_representative_snapshot", """
S = readmatrix('data/snapshot_sinr.csv');
U = readtable('data/snapshot_ues.csv');
B = readtable('data/snapshot_bs.csv');
figure('Color','w','Position',[100 100 1300 520]);
tiledlayout(1,3,'TileSpacing','compact');
nexttile; imagesc(10*log10(max(S,1e-12))); colorbar; xlabel('BS'); ylabel('UE'); title('All-on SINR (dB)');
nexttile; gscatter(U.x,U.y,U.oracle); hold on; scatter(B.x,B.y,120,'ks','filled'); axis equal; title('Reference association'); grid on;
nexttile; gscatter(U.x,U.y,U.ml_gcn); hold on; scatter(B.x,B.y,120,'ks','filled'); axis equal; title('ML-GCN association'); grid on;
savefig('fig_results_representative_snapshot.fig');
""")


def main() -> None:
    ensure_dirs()
    main_comparison()
    representative_snapshot()
    sweep_figures()
    surface_energy_figure()
    ablation_attention_figures()
    print(f"Saved paper figures to {OVERLEAF_FIG}")
    print(f"Saved MATLAB scripts and data to {MATLAB_DIR}")


if __name__ == "__main__":
    main()
