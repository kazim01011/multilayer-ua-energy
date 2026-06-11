from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .active_selector import fit_active_selector
from .baselines import baseline_decision
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


def score_guided_pruned_assignments(
    cfg: ExperimentConfig,
    graphs: list[UAGraph],
    probabilities: list[np.ndarray],
    active_scores: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    assignments = []
    active_sets = []
    for graph, probs, scores in zip(graphs, probabilities, active_scores):
        gain = graph.context["gain"]
        demand = graph.context["demand_mbps"]
        active = np.ones(cfg.sim.num_bs, dtype=bool)
        preferred = np.argmax(probs, axis=1).astype(int)
        labels = repair_assignment(cfg.sim, preferred, gain, demand, allowed_active=np.where(active)[0])
        best_metrics = evaluate_assignment(cfg.sim, labels, gain, demand)

        for bs in np.argsort(scores):
            if int(active.sum()) <= 1:
                break
            candidate_active = active.copy()
            candidate_active[int(bs)] = False
            candidate_indices = np.where(candidate_active)[0]
            if candidate_indices.size == 0:
                continue
            candidate_preferred = candidate_indices[np.argmax(probs[:, candidate_indices], axis=1)].astype(int)
            candidate_labels = repair_assignment(
                cfg.sim,
                candidate_preferred,
                gain,
                demand,
                allowed_active=candidate_indices,
            )
            candidate_metrics = evaluate_assignment(cfg.sim, candidate_labels, gain, demand)
            feasible = (
                candidate_metrics["served_ratio"] >= 0.999
                and candidate_metrics["max_load"] <= cfg.sim.load_limit + 1e-9
            )
            if feasible and candidate_metrics["energy_w"] <= best_metrics["energy_w"] + 1e-9:
                active = candidate_active
                labels = candidate_labels
                best_metrics = candidate_metrics

        assignments.append(labels.astype(int))
        active_sets.append(np.where(active)[0].astype(int))
    return assignments, active_sets


def score_guided_stable_assignments(
    cfg: ExperimentConfig,
    graphs: list[UAGraph],
    probabilities: list[np.ndarray],
    active_scores: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    assignments = []
    active_sets = []
    previous_active: np.ndarray | None = None

    for graph, probs, scores in zip(graphs, probabilities, active_scores):
        gain = graph.context["gain"]
        demand = graph.context["demand_mbps"]

        if previous_active is None:
            first_assignments, first_sets = score_guided_pruned_assignments(cfg, [graph], [probs], [scores])
            labels = first_assignments[0]
            active = np.zeros(cfg.sim.num_bs, dtype=bool)
            active[first_sets[0]] = True
            assignments.append(labels)
            active_sets.append(first_sets[0])
            previous_active = active
            continue

        def decision_for(active_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
            candidates = np.where(active_mask)[0]
            preferred = candidates[np.argmax(probs[:, candidates], axis=1)].astype(int)
            candidate_labels = repair_assignment(
                cfg.sim,
                preferred,
                gain,
                demand,
                allowed_active=candidates,
            )
            return candidate_labels, evaluate_assignment(cfg.sim, candidate_labels, gain, demand)

        def feasible(metrics: dict[str, float]) -> bool:
            return (
                metrics["served_ratio"] >= 0.999
                and metrics["max_load"] <= cfg.sim.load_limit + 1e-9
            )

        def objective(metrics: dict[str, float], active_mask: np.ndarray) -> float:
            switches = int(np.sum(np.logical_xor(active_mask, previous_active)))
            return float(metrics["energy_w"] + cfg.sim.bs_switching_cost_w * switches)

        active = previous_active.copy()
        if not np.any(active):
            active[int(np.argmax(scores))] = True
        labels, best_metrics = decision_for(active)

        while not feasible(best_metrics) and not active.all():
            inactive = np.where(~active)[0]
            active[int(inactive[np.argmax(scores[inactive])])] = True
            labels, best_metrics = decision_for(active)

        best_objective = objective(best_metrics, active)
        for bs in np.argsort(scores):
            if not active[int(bs)] or int(active.sum()) <= 1:
                continue
            candidate_active = active.copy()
            candidate_active[int(bs)] = False
            candidate_labels, candidate_metrics = decision_for(candidate_active)
            if not feasible(candidate_metrics):
                continue
            candidate_objective = objective(candidate_metrics, candidate_active)
            if candidate_objective <= best_objective + 1e-9:
                active = candidate_active
                labels = candidate_labels
                best_metrics = candidate_metrics
                best_objective = candidate_objective

        pruned_assignments, pruned_sets = score_guided_pruned_assignments(cfg, [graph], [probs], [scores])
        pruned_active = np.zeros(cfg.sim.num_bs, dtype=bool)
        pruned_active[pruned_sets[0]] = True
        pruned_metrics = evaluate_assignment(cfg.sim, pruned_assignments[0], gain, demand)
        pruned_objective = objective(pruned_metrics, pruned_active)
        if feasible(pruned_metrics) and pruned_objective <= best_objective + 1e-9:
            active = pruned_active
            labels = pruned_assignments[0]

        assignments.append(labels.astype(int))
        active_indices = np.where(active)[0].astype(int)
        active_sets.append(active_indices)
        previous_active = active

    return assignments, active_sets


def active_mask_from_labels(labels: np.ndarray, num_bs: int) -> np.ndarray:
    active = np.zeros(num_bs, dtype=bool)
    active[np.unique(np.asarray(labels, dtype=int))] = True
    return active


def evaluate_policy(
    cfg: ExperimentConfig,
    graphs: list[UAGraph],
    policy_name: str,
    assignments: list[np.ndarray],
    active_sets: list[np.ndarray] | None = None,
    repair: bool = True,
) -> list[dict[str, float | str]]:
    rows = []
    previous_active: np.ndarray | None = None
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
        active = active_mask_from_labels(labels, cfg.sim.num_bs)
        if previous_active is None:
            switch_events = 0
            activation_events = 0
            deactivation_events = 0
        else:
            activated = np.logical_and(active, np.logical_not(previous_active))
            deactivated = np.logical_and(np.logical_not(active), previous_active)
            activation_events = int(np.sum(activated))
            deactivation_events = int(np.sum(deactivated))
            switch_events = activation_events + deactivation_events
        previous_active = active
        switching_penalty = cfg.sim.bs_switching_cost_w * switch_events
        energy_with_switching = metrics["energy_w"] + switching_penalty
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
                "switching_penalty_w": switching_penalty,
                "energy_w_with_switching": energy_with_switching,
                "energy_saving_vs_all_on": (all_on_energy - metrics["energy_w"]) / all_on_energy,
                "energy_saving_with_switching_vs_all_on": (all_on_energy - energy_with_switching) / all_on_energy,
                "energy_gap_vs_oracle": (metrics["energy_w"] - oracle_energy) / max(oracle_energy, 1e-12),
                "energy_gap_with_switching_vs_oracle_static": (energy_with_switching - oracle_energy)
                / max(oracle_energy, 1e-12),
                "served_ratio": metrics["served_ratio"],
                "active_bs": metrics["active_bs"],
                "switch_events": switch_events,
                "activation_events": activation_events,
                "deactivation_events": deactivation_events,
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
        decisions = [baseline_decision(baseline, cfg.sim, graph) for graph in test_graphs]
        assignments = [labels for labels, _ in decisions]
        active_sets = [active_set for _, active_set in decisions]
        rows.extend(evaluate_policy(cfg, test_graphs, baseline, assignments, active_sets=active_sets))

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
        pruned_predictions, pruned_active_sets = score_guided_pruned_assignments(
            cfg,
            test_graphs,
            test_probs,
            active_scores,
        )
        rows.extend(
            evaluate_policy(
                cfg,
                test_graphs,
                f"{model_name}_pruned",
                pruned_predictions,
                active_sets=pruned_active_sets,
                repair=False,
            )
        )
        stable_predictions, stable_active_sets = score_guided_stable_assignments(
            cfg,
            test_graphs,
            test_probs,
            active_scores,
        )
        rows.extend(
            evaluate_policy(
                cfg,
                test_graphs,
                f"{model_name}_stable",
                stable_predictions,
                active_sets=stable_active_sets,
                repair=False,
            )
        )
        diag = {
            "policy": model_name,
            "active_count_hint": active_count,
            "mean_predicted_active_score": float(np.mean(np.concatenate(active_scores))),
            **model.diagnostics(),
        }
        diagnostics.append(diag)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    oracle_switching_energy = float(
        metrics.loc[metrics["policy"] == "oracle", "energy_w_with_switching"].mean()
    )
    summary = (
        metrics.groupby("policy")
        .agg(
            assignment_accuracy=("assignment_accuracy", "mean"),
            energy_w=("energy_w", "mean"),
            switching_penalty_w=("switching_penalty_w", "mean"),
            energy_w_with_switching=("energy_w_with_switching", "mean"),
            energy_saving_vs_all_on=("energy_saving_vs_all_on", "mean"),
            energy_saving_with_switching_vs_all_on=("energy_saving_with_switching_vs_all_on", "mean"),
            energy_gap_vs_oracle=("energy_gap_vs_oracle", "mean"),
            energy_gap_with_switching_vs_oracle_static=("energy_gap_with_switching_vs_oracle_static", "mean"),
            served_ratio=("served_ratio", "mean"),
            active_bs=("active_bs", "mean"),
            switch_events=("switch_events", "mean"),
            activation_events=("activation_events", "mean"),
            deactivation_events=("deactivation_events", "mean"),
            max_load=("max_load", "mean"),
        )
        .reset_index()
        .sort_values(["served_ratio", "energy_w"], ascending=[False, True])
    )
    summary["energy_gap_with_switching_vs_oracle"] = (
        summary["energy_w_with_switching"] - oracle_switching_energy
    ) / max(oracle_switching_energy, 1e-12)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "diagnostics.csv", index=False)
    return summary
