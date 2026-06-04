#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create initial SVG experiment figures.")
    parser.add_argument("--results", type=Path, required=True, help="Result folder containing summary.csv.")
    return parser.parse_args()


def svg_bar(summary: pd.DataFrame, metric: str, ylabel: str, path: Path, high_is_good: bool) -> None:
    order = summary.sort_values(metric, ascending=not high_is_good).reset_index(drop=True)
    labels = order["policy"].tolist()
    values = order[metric].astype(float).tolist()
    width, height = 920, 520
    left, right, top, bottom = 90, 30, 40, 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_v = min(0.0, min(values))
    max_v = max(values) if max(values) > min_v else min_v + 1.0
    scale = plot_h / max(max_v - min_v, 1e-12)
    bar_gap = 12
    bar_w = (plot_w - bar_gap * (len(labels) - 1)) / len(labels)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="Arial" font-size="18" font-weight="bold">{ylabel}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * (bar_w + bar_gap)
        h = (value - min_v) * scale
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#4267b2"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 18}" text-anchor="end" '
            f'font-family="Arial" font-size="12" transform="rotate(-35 {x + bar_w / 2:.1f},{top + plot_h + 18})">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.results / "summary.csv")
    fig_dir = args.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    svg_bar(summary, "energy_saving_vs_all_on", "Mean energy saving vs all-on", fig_dir / "energy_saving.svg", True)
    svg_bar(summary, "served_ratio", "QoS served ratio", fig_dir / "served_ratio.svg", True)
    svg_bar(summary, "assignment_accuracy", "Oracle-assignment agreement", fig_dir / "assignment_accuracy.svg", True)
    svg_bar(summary, "active_bs", "Mean active base stations", fig_dir / "active_bs.svg", False)

    diag_path = args.results / "diagnostics.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        attn = diag[diag["policy"] == "attn_ml_gcn"]
        if not attn.empty:
            cols = [c for c in attn.columns if c.startswith("attention_")]
            values = attn.iloc[0][cols].astype(float)
            attn_summary = pd.DataFrame(
                {"policy": [c.replace("attention_", "") for c in cols], "attention": values.values}
            )
            svg_bar(attn_summary, "attention", "Learned layer attention", fig_dir / "layer_attention.svg", True)
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()

