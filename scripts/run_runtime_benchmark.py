#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from mlua.active_selector import fit_active_selector
from mlua.baselines import baseline_assignment
from mlua.config import ExperimentConfig
from mlua.experiment import active_selector_assignments, mean_oracle_active_count, standardize_graphs
from mlua.models import ModelConfig, build_model
from mlua.simulator import generate_dataset, oracle_assignment, repair_assignment
from mlua.utils import save_json


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "runtime_benchmark"


def learned_decision(cfg: ExperimentConfig, model, selector, active_count: int, graph) -> np.ndarray:
    probs = model.predict_proba_graphs([graph])
    scores = selector.predict_scores([graph], probs, cfg.sim)
    predictions, active_sets = active_selector_assignments(probs, scores, active_count)
    return repair_assignment(
        cfg.sim,
        predictions[0],
        graph.context["gain"],
        graph.context["demand_mbps"],
        allowed_active=active_sets[0],
    )


def time_policy(fn, graphs, repeats: int = 3) -> tuple[float, float, float]:
    for graph in graphs:
        fn(graph)
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        for graph in graphs:
            fn(graph)
        elapsed = perf_counter() - start
        samples.append(1000.0 * elapsed / max(len(graphs), 1))
    std = 0.0 if len(samples) < 2 else float(np.std(samples, ddof=1))
    return float(np.median(samples)), float(np.mean(samples)), std


def main() -> None:
    cfg = ExperimentConfig(
        run_name="runtime_benchmark",
        train_graphs=12,
        test_graphs=5,
        epochs=30,
        patience=8,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUT_DIR / "runtime_config.json", asdict(cfg))

    print("Generating benchmark snapshots...", flush=True)
    train_graphs = generate_dataset(cfg.sim, cfg.train_graphs, seed_offset=0)
    test_graphs = generate_dataset(cfg.sim, cfg.test_graphs, seed_offset=10_000)
    standardize_graphs(train_graphs, test_graphs)

    rows: list[dict[str, float | str]] = []

    complexity = {
        "rsrp": r"$\mathcal{O}(NB)$",
        "sinr": r"$\mathcal{O}(NB)$",
        "load_aware": r"$\mathcal{O}(NB+N\log N)$",
        "oracle": r"$\mathcal{O}(2^{B}NB)$",
        "flat_mlp": r"$\mathcal{O}(Ndh+Nh^2+NhB)$",
        "agg_gcn": r"$\mathcal{O}(|E|d+Ndh+NhB)$",
        "ml_gcn": r"$\mathcal{O}(\sum_{\ell=1}^{L}|E_{\ell}|d+LNdh+LNhB)$",
        "attn_ml_gcn": r"$\mathcal{O}(\sum_{\ell=1}^{L}|E_{\ell}|d+LNdh+LNhB)$",
    }

    labels = {
        "rsrp": "RSRP",
        "sinr": "SINR",
        "load_aware": "Load-aware",
        "oracle": "Reference search",
        "flat_mlp": "Flat MLP",
        "agg_gcn": "Aggregated GCN",
        "ml_gcn": "ML-GCN",
        "attn_ml_gcn": "Attn-ML-GCN",
    }

    for baseline in cfg.baselines:
        print(f"Timing {baseline}...", flush=True)
        if baseline == "oracle":
            median_ms, mean_ms, std_ms = time_policy(
                lambda graph: oracle_assignment(cfg.sim, graph.context["gain"], graph.context["demand_mbps"])[0],
                test_graphs,
                repeats=1,
            )
            train_s = 0.0
        else:
            def decide(graph, name=baseline):
                raw = baseline_assignment(name, cfg.sim, graph)
                return repair_assignment(cfg.sim, raw, graph.context["gain"], graph.context["demand_mbps"])

            median_ms, mean_ms, std_ms = time_policy(decide, test_graphs)
            train_s = 0.0
        rows.append(
            {
                "policy": baseline,
                "method": labels[baseline],
                "offline_training_s": train_s,
                "median_decision_ms_per_snapshot": median_ms,
                "mean_decision_ms_per_snapshot": mean_ms,
                "decision_std_ms": std_ms,
                "asymptotic_cost": complexity[baseline],
            }
        )

    input_dim = train_graphs[0].features.shape[1]
    model_cfg = ModelConfig(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.sim.num_bs,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
    )
    active_count = mean_oracle_active_count(train_graphs)

    for idx, model_name in enumerate(cfg.models):
        print(f"Training and timing {model_name}...", flush=True)
        model = build_model(model_name, model_cfg)
        start = perf_counter()
        model.fit(train_graphs, [], cfg.epochs, cfg.patience)
        train_probs = model.predict_proba_graphs(train_graphs)
        selector = fit_active_selector(train_graphs, train_probs, cfg.sim, seed=cfg.seed + idx)
        train_s = perf_counter() - start

        median_ms, mean_ms, std_ms = time_policy(
            lambda graph, m=model, s=selector: learned_decision(cfg, m, s, active_count, graph),
            test_graphs,
        )
        rows.append(
            {
                "policy": model_name,
                "method": labels[model_name],
                "offline_training_s": train_s,
                "median_decision_ms_per_snapshot": median_ms,
                "mean_decision_ms_per_snapshot": mean_ms,
                "decision_std_ms": std_ms,
                "asymptotic_cost": complexity[model_name],
            }
        )

    df = pd.DataFrame(rows)
    order = ["rsrp", "sinr", "load_aware", "flat_mlp", "agg_gcn", "ml_gcn", "attn_ml_gcn", "oracle"]
    df["order"] = df["policy"].map({name: idx for idx, name in enumerate(order)})
    df = df.sort_values("order").drop(columns=["order"])
    df.to_csv(OUT_DIR / "runtime_summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
