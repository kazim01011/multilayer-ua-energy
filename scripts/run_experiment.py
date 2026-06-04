#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from mlua.config import ExperimentConfig, SimConfig
from mlua.experiment import run_experiment
from mlua.utils import deep_update, load_json_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multilayer user-association experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output directory.")
    return parser.parse_args()


def build_config(path: Path) -> ExperimentConfig:
    raw = load_json_config(path)
    base = asdict(ExperimentConfig())
    merged = deep_update(base, raw)
    merged["sim"] = SimConfig(**merged["sim"])
    return ExperimentConfig(**merged)


def main() -> None:
    args = parse_args()
    cfg = build_config(args.config)
    output = args.output or Path("results") / cfg.run_name
    summary = run_experiment(cfg, output)
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {output}")


if __name__ == "__main__":
    main()

