#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd

from mlua.config import ExperimentConfig, SimConfig
from mlua.experiment import run_experiment
from mlua.utils import deep_update, save_json


def build_cfg(run_name: str, sim_updates: dict[str, float | int]) -> ExperimentConfig:
    base = asdict(ExperimentConfig())
    base.update(
        {
            "run_name": run_name,
            "train_graphs": 45,
            "test_graphs": 15,
            "epochs": 60,
            "patience": 15,
            "hidden_dim": 24,
            "learning_rate": 0.018,
            "seed": 17,
            "models": ("flat_mlp", "agg_gcn", "ml_gcn", "attn_ml_gcn"),
        }
    )
    base["sim"].update(
        {
            "num_bs": 7,
            "num_ues": 50,
            "area_radius_m": 650.0,
            "bandwidth_hz": 20e6,
            "qos_mbps": 1.0,
            "shadowing_db": 6.0,
            "seed": 17,
        }
    )
    merged = deep_update(base, {"sim": sim_updates})
    merged["sim"] = SimConfig(**merged["sim"])
    return ExperimentConfig(**merged)


def main() -> None:
    output_root = Path("results") / "paper_sweeps"
    output_root.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, str, float, dict[str, float | int]]] = []
    for bw_mhz in (10, 20, 40, 80):
        specs.append(("bandwidth", "bandwidth_mhz", float(bw_mhz), {"bandwidth_hz": bw_mhz * 1e6}))
    for qos in (0.5, 1.0, 1.5, 2.0):
        specs.append(("qos", "qos_mbps", float(qos), {"qos_mbps": qos}))
    for num_ues in (30, 50, 70, 90):
        specs.append(("density", "num_ues", float(num_ues), {"num_ues": int(num_ues)}))
    for shadow in (2, 6, 10, 14):
        specs.append(("shadowing", "shadowing_db", float(shadow), {"shadowing_db": shadow}))

    rows = []
    save_json(output_root / "sweep_specs.json", {"specs": specs})
    for sweep, parameter, value, sim_updates in specs:
        run_name = f"{sweep}_{str(value).replace('.', 'p')}"
        cfg = build_cfg(run_name, sim_updates)
        out_dir = output_root / sweep / run_name
        start = perf_counter()
        summary = run_experiment(cfg, out_dir)
        elapsed = perf_counter() - start
        summary = summary.copy()
        summary["sweep"] = sweep
        summary["parameter"] = parameter
        summary["value"] = value
        summary["runtime_s"] = elapsed
        rows.append(summary)
        print(f"Completed {run_name} in {elapsed:.1f}s")

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_root / "sweep_summary.csv", index=False)
    print(f"Saved sweep summary to {output_root / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
