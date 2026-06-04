#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mlua.config import ExperimentConfig, SimConfig
from mlua.experiment import run_experiment
from mlua.utils import deep_update, save_json


def build_cfg(qos_mbps: float, num_ues: int, shadowing_db: float) -> ExperimentConfig:
    base = asdict(ExperimentConfig())
    base.update(
        {
            "run_name": f"surface_q{qos_mbps}_n{num_ues}_s{shadowing_db}",
            "train_graphs": 36,
            "test_graphs": 12,
            "epochs": 50,
            "patience": 12,
            "hidden_dim": 24,
            "learning_rate": 0.018,
            "seed": 17,
            "models": ("ml_gcn",),
        }
    )
    base["sim"].update(
        {
            "num_bs": 7,
            "num_ues": num_ues,
            "area_radius_m": 650.0,
            "bandwidth_hz": 20e6,
            "qos_mbps": qos_mbps,
            "shadowing_db": shadowing_db,
            "seed": 17,
        }
    )
    base["sim"] = SimConfig(**base["sim"])
    return ExperimentConfig(**base)


def main() -> None:
    out_root = Path("results") / "surface_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    qos_values = [0.5, 2.0, 4.0, 6.0]
    ue_values = [30, 50, 70, 90]
    shadowing_values = [2.0, 6.0, 10.0]
    save_json(
        out_root / "surface_config.json",
        {
            "qos_mbps": qos_values,
            "num_ues": ue_values,
            "shadowing_db": shadowing_values,
            "metric": "energy_saving_vs_all_on",
        },
    )

    rows = []
    for shadow in shadowing_values:
        for num_ues in ue_values:
            for qos in qos_values:
                cfg = build_cfg(qos, num_ues, shadow)
                run_dir = out_root / f"shadow_{shadow:g}" / f"n_{num_ues}_q_{str(qos).replace('.', 'p')}"
                summary = run_experiment(cfg, run_dir)
                summary = summary.copy()
                summary["qos_mbps"] = qos
                summary["num_ues"] = num_ues
                summary["shadowing_db"] = shadow
                rows.append(summary)
                ml = summary[summary["policy"] == "ml_gcn"].iloc[0]
                print(
                    f"shadow={shadow:g} dB, N={num_ues}, QoS={qos:g} Mbps: "
                    f"ES={ml.energy_saving_vs_all_on:.3f}, SR={ml.served_ratio:.3f}"
                )

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out_root / "surface_summary.csv", index=False)
    print(f"Saved {out_root / 'surface_summary.csv'}")


if __name__ == "__main__":
    main()
