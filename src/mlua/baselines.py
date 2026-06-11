from __future__ import annotations

import numpy as np

from .config import SimConfig
from .simulator import downlink_sinr, evaluate_assignment, greedy_energy_assignment, spectral_efficiency, w_to_dbm


def rsrp_association(gain: np.ndarray) -> np.ndarray:
    return np.argmax(gain, axis=1)


def sinr_association(cfg: SimConfig, gain: np.ndarray) -> np.ndarray:
    sinr = downlink_sinr(cfg, gain, np.ones(cfg.num_bs, dtype=bool))
    return np.argmax(sinr, axis=1)


def active_rsrp_association(gain: np.ndarray, active: np.ndarray) -> np.ndarray:
    candidates = np.where(active)[0]
    local = np.argmax(gain[:, candidates], axis=1)
    return candidates[local].astype(int)


def active_sinr_association(cfg: SimConfig, gain: np.ndarray, active: np.ndarray) -> np.ndarray:
    candidates = np.where(active)[0]
    sinr = downlink_sinr(cfg, gain, active)
    local = np.argmax(sinr[:, candidates], axis=1)
    return candidates[local].astype(int)


def load_aware_association(cfg: SimConfig, gain: np.ndarray, demand: np.ndarray) -> np.ndarray:
    return active_load_aware_association(cfg, gain, demand, np.ones(cfg.num_bs, dtype=bool))


def active_load_aware_association(cfg: SimConfig, gain: np.ndarray, demand: np.ndarray, active: np.ndarray) -> np.ndarray:
    sinr = downlink_sinr(cfg, gain, active)
    eff = spectral_efficiency(sinr)
    labels = np.full(demand.shape[0], -1, dtype=int)
    loads = np.zeros(cfg.num_bs, dtype=float)
    candidates = np.where(active)[0]
    for ue in np.argsort(-demand):
        choices = []
        for bs in candidates:
            if w_to_dbm(cfg.bs_tx_power_w * gain[ue, bs]) < cfg.min_rsrp_dbm:
                continue
            capacity = cfg.bandwidth_hz * eff[ue, bs] / 1e6
            load_inc = demand[ue] / max(capacity, 1e-6)
            choices.append((loads[bs] + load_inc, -eff[ue, bs], bs, load_inc))
        if not choices:
            labels[ue] = int(candidates[np.argmax(gain[ue, candidates])])
            continue
        _, _, bs, load_inc = min(choices)
        labels[ue] = int(bs)
        loads[bs] += load_inc
    return labels


def _association_on_active(base: str, cfg: SimConfig, gain: np.ndarray, demand: np.ndarray, active: np.ndarray) -> np.ndarray:
    if base == "rsrp":
        return active_rsrp_association(gain, active)
    if base == "sinr":
        return active_sinr_association(cfg, gain, active)
    if base == "load_aware":
        return active_load_aware_association(cfg, gain, demand, active)
    raise ValueError(f"Unsupported pruning base policy: {base}")


def _is_feasible(cfg: SimConfig, metrics: dict[str, float]) -> bool:
    return bool(metrics["served_ratio"] >= 0.999 and metrics["max_load"] <= cfg.load_limit + 1e-9)


def greedy_pruned_association(
    base: str,
    cfg: SimConfig,
    gain: np.ndarray,
    demand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily deactivate BSs while preserving QoS/load feasibility."""
    active = np.ones(cfg.num_bs, dtype=bool)
    labels = _association_on_active(base, cfg, gain, demand, active)
    current = evaluate_assignment(cfg, labels, gain, demand)

    while int(active.sum()) > 1:
        best: tuple[float, int, np.ndarray, dict[str, float]] | None = None
        for bs in np.where(active)[0]:
            candidate_active = active.copy()
            candidate_active[int(bs)] = False
            candidate_labels = _association_on_active(base, cfg, gain, demand, candidate_active)
            metrics = evaluate_assignment(cfg, candidate_labels, gain, demand)
            if not _is_feasible(cfg, metrics):
                continue
            if metrics["energy_w"] >= current["energy_w"] - 1e-9:
                continue
            item = (metrics["energy_w"], int(bs), candidate_labels, metrics)
            if best is None or item[0] < best[0]:
                best = item
        if best is None:
            break
        _, removed_bs, labels, current = best
        active[int(removed_bs)] = False
    return labels, np.where(active)[0].astype(int)


def greedy_sleep_control(
    cfg: SimConfig,
    gain: np.ndarray,
    demand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward greedy BS sleep control using the energy-aware assignment rule."""
    active = np.ones(cfg.num_bs, dtype=bool)
    labels, current = greedy_energy_assignment(cfg, gain, demand, active)

    while int(active.sum()) > 1:
        best: tuple[float, int, np.ndarray, dict[str, float]] | None = None
        for bs in np.where(active)[0]:
            candidate_active = active.copy()
            candidate_active[int(bs)] = False
            candidate_labels, metrics = greedy_energy_assignment(cfg, gain, demand, candidate_active)
            if not _is_feasible(cfg, metrics):
                continue
            if metrics["energy_w"] >= current["energy_w"] - 1e-9:
                continue
            item = (metrics["energy_w"], int(bs), candidate_labels, metrics)
            if best is None or item[0] < best[0]:
                best = item
        if best is None:
            break
        _, removed_bs, labels, current = best
        active[int(removed_bs)] = False
    return labels, np.where(active)[0].astype(int)


def baseline_assignment(name: str, cfg: SimConfig, graph) -> np.ndarray:
    gain = graph.context["gain"]
    demand = graph.context["demand_mbps"]
    if name == "rsrp":
        return rsrp_association(gain)
    if name == "sinr":
        return sinr_association(cfg, gain)
    if name == "load_aware":
        return load_aware_association(cfg, gain, demand)
    if name == "oracle":
        return graph.labels.copy()
    labels, _ = baseline_decision(name, cfg, graph)
    return labels


def baseline_decision(name: str, cfg: SimConfig, graph) -> tuple[np.ndarray, np.ndarray | None]:
    gain = graph.context["gain"]
    demand = graph.context["demand_mbps"]
    if name == "rsrp":
        return rsrp_association(gain), None
    if name == "sinr":
        return sinr_association(cfg, gain), None
    if name == "load_aware":
        return load_aware_association(cfg, gain, demand), None
    if name == "oracle":
        return graph.labels.copy(), np.unique(graph.labels).astype(int)
    if name == "rsrp_sleep":
        return greedy_pruned_association("rsrp", cfg, gain, demand)
    if name == "sinr_sleep":
        return greedy_pruned_association("sinr", cfg, gain, demand)
    if name == "load_sleep":
        return greedy_pruned_association("load_aware", cfg, gain, demand)
    if name == "greedy_sleep":
        return greedy_sleep_control(cfg, gain, demand)
    raise ValueError(f"Unknown baseline: {name}")
