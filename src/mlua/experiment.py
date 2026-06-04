from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .active_selector import fit_active_selector
from .baselines import baseline_assignment
from .config import ExperimentConfig
from .models import ModelConfig, build_model
from .simulator import UAGraph, evaluate_assignment, generate_dataset, repair_assignment
from .utils import save_json


def standardize_graphs(train_graphs: list[UAGraph], test_graphs: list[UAGraph]) -> None:
    x = np.vstack([g.features for g in train_graphs])
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    for graph in train_graphs + test_graphs:
        graph.features = (graph.features - mean) / std


def split_model_predictions(graphs: list[UAGraph], labels: np.ndarray) -> list[np.ndarray]:
    out = []
    cursor = 0
    for graph in graphs:
        n = graph.labels.shape[0]
        out.append(labels[cursor : cursor + n])
        cursor += n
    return out


def mean_oracle_active_count(graphs: list[UAGraph]) -> int:
    active_counts = [len(np.unique(g.labels)) for g in graphs]
    return max(1, int(round(float(np.mean(active_counts)))))


def active_set_assignments(probabilities: list[np.ndarray], active_count: int) -> list[np.ndarray]:
    assignments = []
    for probs in probabilities:
        scores = probs.sum(axis=0)
        active = np.argsort(-scores)[:active_count]
        local_choice = np.argmax(probs[:, active], axis=1)
        assignments.append(active[local_choice].astype(int))
    return assignments


def active_selector_assignments(
    probabilities: list[np.ndarray],
    active_scores: list[np.ndarray],
    active_count: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    assignments = []
    active_sets = []
    for probs, scores in zip(probabilities, active_scores):
        active = np.argsort(-scores)[:active_count]
        active = np.asarray(sorted(set(int(x) for x in active)), dtype=int)
        if active.size == 0:
            active = np.asarray([int(np.argmax(scores))], dtype=int)
        local_choice = np.argmax(probs[:, active], axis=1)
        assignments.append(active[local_choice].astype(int))
        active_sets.append(active)
    return assignments, active_sets


def evaluate_policy(
    cfg: ExperimentConfig,
    graphs: list[UAGraph],
    policy_name: str,
    assignments: list[np.ndarray],
    active_sets: list[np.ndarray] | None = None,
    repair: bool = True,
) -> list[dict[str, float | str]]:
    rows = []
    if active_sets is None:
        active_sets = [None] * len(graphs)
    for graph, labels, active_set in zip(graphs, assignments, active_sets):
        raw_accuracy = float(np.mean(labels == graph.labels))
        if repair and policy_name != "oracle":
            labels = repair_assignment(
                cfg.sim,
                labels,
                graph.context["gain"],
                graph.context["demand_mbps"],
                allowed_active=active_set,
            )
        metrics = evaluate_assignment(cfg.sim, labels, graph.context["gain"], graph.context["demand_mbps"])
        accuracy = float(np.mean(labels == graph.labels))
        all_on_energy = cfg.sim.num_bs * (cfg.sim.bs_fixed_power_w + cfg.sim.bs_dynamic_power_w)
        oracle_energy = float(graph.context["oracle_energy_w"])
        rows.append(
            {
                "policy": policy_name,
                "graph_id": int(graph.context["graph_id"]),
                "hour": int(graph.context["hour"]),
                "raw_assignment_accuracy": raw_accuracy,
                "assignment_accuracy": accuracy,
                "energy_w": metrics["energy_w"],
                "energy_saving_vs_all_on": (all_on_energy - metrics["energy_w"]) / all_on_energy,
                "energy_gap_vs_oracle": (metrics["energy_w"] - oracle_energy) / max(oracle_energy, 1e-12),
                "served_ratio": metrics["served_ratio"],
                "active_bs": metrics["active_bs"],
                "max_load": metrics["max_load"],
                "mean_active_load": metrics["mean_active_load"],
            }
        )
    return rows


def run_experiment(cfg: ExperimentConfig, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", asdict(cfg))
    train_graphs = generate_dataset(cfg.sim, cfg.train_graphs, seed_offset=0)
    test_graphs = generate_dataset(cfg.sim, cfg.test_graphs, seed_offset=10_000)
    standardize_graphs(train_graphs, test_graphs)

    rows: list[dict[str, float | str]] = []
    for baseline in cfg.baselines:
        assignments = [baseline_assignment(baseline, cfg.sim, graph) for graph in test_graphs]
        rows.extend(evaluate_policy(cfg, test_graphs, baseline, assignments))

    input_dim = train_graphs[0].features.shape[1]
    model_cfg = ModelConfig(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.sim.num_bs,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
    )
    active_count = mean_oracle_active_count(train_graphs)
    diagnostics = []
    for model_name in cfg.models:
        model = build_model(model_name, model_cfg)
        model.fit(train_graphs, [], cfg.epochs, cfg.patience)
        train_probs = model.predict_proba_graphs(train_graphs)
        test_probs = model.predict_proba_graphs(test_graphs)
        selector = fit_active_selector(train_graphs, train_probs, cfg.sim, seed=cfg.seed + len(diagnostics))
        active_scores = selector.predict_scores(test_graphs, test_probs, cfg.sim)
        predictions, active_sets = active_selector_assignments(test_probs, active_scores, active_count)
        rows.extend(evaluate_policy(cfg, test_graphs, model_name, predictions, active_sets=active_sets))
        diag = {
            "policy": model_name,
            "active_count_hint": active_count,
            "mean_predicted_active_score": float(np.mean(np.concatenate(active_scores))),
            **model.diagnostics(),
        }
        diagnostics.append(diag)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary = (
        metrics.groupby("policy")
        .agg(
            assignment_accuracy=("assignment_accuracy", "mean"),
            energy_w=("energy_w", "mean"),
            energy_saving_vs_all_on=("energy_saving_vs_all_on", "mean"),
            energy_gap_vs_oracle=("energy_gap_vs_oracle", "mean"),
            served_ratio=("served_ratio", "mean"),
            active_bs=("active_bs", "mean"),
            max_load=("max_load", "mean"),
        )
        .reset_index()
        .sort_values(["served_ratio", "energy_w"], ascending=[False, True])
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "diagnostics.csv", index=False)
    return summary
