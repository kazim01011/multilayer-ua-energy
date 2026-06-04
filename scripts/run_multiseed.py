#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mlua.config import ExperimentConfig, SimConfig
from mlua.experiment import run_experiment
from mlua.utils import deep_update, load_json_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-seed benchmark.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 31], help="Seeds to evaluate.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output directory.")
    return parser.parse_args()


def build_config(path: Path, seed: int) -> ExperimentConfig:
    raw = load_json_config(path)
    base = asdict(ExperimentConfig())
    merged = deep_update(base, raw)
    merged["seed"] = seed
    merged["sim"]["seed"] = seed
    merged["sim"] = SimConfig(**merged["sim"])
    return ExperimentConfig(**merged)


def main() -> None:
    args = parse_args()
    base_cfg = load_json_config(args.config)
    run_name = base_cfg.get("run_name", args.config.stem)
    output_root = args.output or Path("results") / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(output_root / "multiseed_config.json", {"config": base_cfg, "seeds": args.seeds})

    summaries = []
    for seed in args.seeds:
        cfg = build_config(args.config, seed)
        seed_dir = output_root / f"seed_{seed}"
        summary = run_experiment(cfg, seed_dir)
        summary = summary.copy()
        summary["seed"] = seed
        summaries.append(summary)
        print(f"\nSeed {seed}")
        print(summary.to_string(index=False))

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_root / "summary_by_seed.csv", index=False)
    metric_cols = [
        "assignment_accuracy",
        "energy_w",
        "energy_saving_vs_all_on",
        "energy_gap_vs_oracle",
        "served_ratio",
        "active_bs",
        "max_load",
    ]
    mean = combined.groupby("policy")[metric_cols].mean().reset_index()
    std = combined.groupby("policy")[metric_cols].std(ddof=1).fillna(0.0).reset_index()
    mean.to_csv(output_root / "summary_mean.csv", index=False)
    std.to_csv(output_root / "summary_std.csv", index=False)
    print("\nMean summary")
    print(mean.sort_values(["served_ratio", "energy_w"], ascending=[False, True]).to_string(index=False))
    print(f"\nSaved multi-seed outputs to {output_root}")


if __name__ == "__main__":
    main()

