from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .config import LAYER_NAMES, SimConfig
from .utils import dbm_to_w, normalize_rows, w_to_dbm


@dataclass
class UAGraph:
    features: np.ndarray
    labels: np.ndarray
    layers: dict[str, np.ndarray]
    context: dict[str, np.ndarray | float | int]


def generate_dataset(cfg: SimConfig, num_graphs: int, seed_offset: int = 0) -> list[UAGraph]:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    bs_pos = base_station_positions(cfg)
    graphs = []
    for graph_id in range(num_graphs):
        hour = graph_id % cfg.temporal_periods
        graphs.append(generate_snapshot(cfg, rng, bs_pos, hour, graph_id))
    return graphs


def base_station_positions(cfg: SimConfig) -> np.ndarray:
    if cfg.num_bs == 1:
        return np.zeros((1, 2))
    positions = [np.array([0.0, 0.0])]
    ring = cfg.area_radius_m * 0.55
    for idx in range(cfg.num_bs - 1):
        theta = 2.0 * np.pi * idx / (cfg.num_bs - 1)
        positions.append(np.array([ring * np.cos(theta), ring * np.sin(theta)]))
    return np.vstack(positions)


def generate_snapshot(
    cfg: SimConfig,
    rng: np.random.Generator,
    bs_pos: np.ndarray,
    hour: int,
    graph_id: int,
) -> UAGraph:
    ue_pos = sample_points_in_disc(rng, cfg.num_ues, cfg.area_radius_m)
    demand = sample_traffic_demand(cfg, rng, ue_pos, hour)
    gain = channel_gain(cfg, rng, ue_pos, bs_pos)
    rsrp_w = cfg.bs_tx_power_w * gain
    rsrp_dbm = w_to_dbm(rsrp_w)
    sinr = downlink_sinr(cfg, gain, np.ones(cfg.num_bs, dtype=bool))
    labels, oracle = oracle_assignment(cfg, gain, demand)
    features = node_features(cfg, ue_pos, demand, gain, rsrp_dbm, sinr, hour)
    layers = build_layers(cfg, ue_pos, demand, gain, labels, hour)
    context = {
        "graph_id": graph_id,
        "hour": hour,
        "ue_pos": ue_pos,
        "bs_pos": bs_pos,
        "demand_mbps": demand,
        "gain": gain,
        "rsrp_dbm": rsrp_dbm,
        "sinr_all_on": sinr,
        "oracle_energy_w": oracle["energy_w"],
        "oracle_served_ratio": oracle["served_ratio"],
        "oracle_active_bs": oracle["active_bs"],
    }
    return UAGraph(features=features, labels=labels, layers=layers, context=context)


def sample_points_in_disc(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def sample_traffic_demand(
    cfg: SimConfig,
    rng: np.random.Generator,
    ue_pos: np.ndarray,
    hour: int,
) -> np.ndarray:
    evening_peak = 0.55 + 0.45 * np.sin((hour - 8) / cfg.temporal_periods * 2.0 * np.pi) ** 2
    spatial_hotspot = np.exp(-np.linalg.norm(ue_pos - np.array([120.0, -80.0]), axis=1) / 450.0)
    raw = rng.lognormal(mean=0.0, sigma=0.55, size=ue_pos.shape[0])
    scaled = raw * (0.65 + evening_peak) * (0.7 + 0.6 * spatial_hotspot)
    lo, hi = cfg.demand_low_mbps, cfg.demand_high_mbps
    return lo + (hi - lo) * (scaled - scaled.min()) / max(float(scaled.max() - scaled.min()), 1e-12)


def channel_gain(
    cfg: SimConfig,
    rng: np.random.Generator,
    ue_pos: np.ndarray,
    bs_pos: np.ndarray,
) -> np.ndarray:
    dist = np.linalg.norm(ue_pos[:, None, :] - bs_pos[None, :, :], axis=2)
    dist = np.maximum(dist, 25.0)
    reference_loss_db = 32.4 + 20.0 * np.log10(cfg.carrier_hz / 1e9)
    path_loss_db = reference_loss_db + 10.0 * cfg.path_loss_exp * np.log10(dist)
    shadow = rng.normal(0.0, cfg.shadowing_db, size=dist.shape)
    fading = rng.rayleigh(scale=1.0, size=dist.shape) ** 2
    large_scale = 10.0 ** (-(path_loss_db + shadow) / 10.0)
    return large_scale * np.maximum(fading, 1e-5)


def downlink_sinr(cfg: SimConfig, gain: np.ndarray, active: np.ndarray) -> np.ndarray:
    signal = cfg.bs_tx_power_w * gain
    active_signal = signal * active[None, :]
    interference = active_signal.sum(axis=1, keepdims=True) - active_signal
    noise = dbm_to_w(cfg.noise_dbm)
    return active_signal / np.maximum(interference + noise, 1e-18)


def spectral_efficiency(sinr: np.ndarray) -> np.ndarray:
    return np.log2(1.0 + np.maximum(sinr, 0.0))


def oracle_assignment(cfg: SimConfig, gain: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    best_labels = np.argmax(gain, axis=1)
    best_score = float("inf")
    best_result: dict[str, float] | None = None
    bs_indices = range(cfg.num_bs)
    min_active = 1
    for active_count in range(min_active, cfg.num_bs + 1):
        for active_tuple in combinations(bs_indices, active_count):
            active = np.zeros(cfg.num_bs, dtype=bool)
            active[list(active_tuple)] = True
            labels, score_parts = greedy_energy_assignment(cfg, gain, demand, active)
            score = score_parts["energy_w"] + 1800.0 * (1.0 - score_parts["served_ratio"])
            score += 120.0 * max(0.0, score_parts["max_load"] - cfg.load_limit)
            if score < best_score:
                best_score = score
                best_labels = labels
                best_result = score_parts
    assert best_result is not None
    return best_labels, best_result


def greedy_energy_assignment(
    cfg: SimConfig,
    gain: np.ndarray,
    demand: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    sinr = downlink_sinr(cfg, gain, active)
    eff = spectral_efficiency(sinr)
    labels = np.full(demand.shape[0], -1, dtype=int)
    loads = np.zeros(cfg.num_bs, dtype=float)
    candidates = np.where(active)[0]
    order = np.argsort(-demand)
    for ue in order:
        feasible = []
        for bs in candidates:
            if w_to_dbm(cfg.bs_tx_power_w * gain[ue, bs]) < cfg.min_rsrp_dbm:
                continue
            capacity_mbps = cfg.bandwidth_hz * eff[ue, bs] / 1e6
            if capacity_mbps < cfg.qos_mbps:
                continue
            load_inc = demand[ue] / max(capacity_mbps, 1e-6)
            projected = loads[bs] + load_inc
            feasible.append((projected, -eff[ue, bs], bs, load_inc))
        if not feasible:
            labels[ue] = int(candidates[np.argmax(gain[ue, candidates])])
            continue
        projected, _, bs, load_inc = min(feasible)
        labels[ue] = int(bs)
        loads[bs] = projected
    served = labels >= 0
    for ue in np.where(~served)[0]:
        labels[ue] = int(candidates[np.argmax(gain[ue, candidates])])
    energy = network_energy(cfg, labels, gain, demand)
    eval_parts = evaluate_assignment(cfg, labels, gain, demand)
    eval_parts["energy_w"] = energy
    return labels, eval_parts


def network_energy(cfg: SimConfig, labels: np.ndarray, gain: np.ndarray, demand: np.ndarray) -> float:
    stats = assignment_stats(cfg, labels, gain, demand)
    active = stats["active"]
    loads = stats["loads"]
    energy = np.where(
        active,
        cfg.bs_fixed_power_w + cfg.bs_dynamic_power_w * np.minimum(loads, 1.25),
        cfg.bs_sleep_power_w,
    )
    return float(energy.sum())


def assignment_stats(
    cfg: SimConfig,
    labels: np.ndarray,
    gain: np.ndarray,
    demand: np.ndarray,
) -> dict[str, np.ndarray | float]:
    labels = np.asarray(labels, dtype=int)
    active = np.zeros(cfg.num_bs, dtype=bool)
    active[np.unique(labels)] = True
    sinr = downlink_sinr(cfg, gain, active)
    eff = spectral_efficiency(sinr)
    loads = np.zeros(cfg.num_bs, dtype=float)
    served = np.ones(labels.shape[0], dtype=bool)
    for ue, bs in enumerate(labels):
        capacity = cfg.bandwidth_hz * eff[ue, bs] / 1e6
        rsrp_ok = w_to_dbm(cfg.bs_tx_power_w * gain[ue, bs]) >= cfg.min_rsrp_dbm
        qos_ok = capacity >= cfg.qos_mbps
        load_inc = demand[ue] / max(capacity, 1e-6)
        loads[bs] += load_inc
        served[ue] = bool(rsrp_ok and qos_ok)
    served &= loads[labels] <= cfg.load_limit
    return {"active": active, "loads": loads, "served": served, "sinr": sinr}


def evaluate_assignment(
    cfg: SimConfig,
    labels: np.ndarray,
    gain: np.ndarray,
    demand: np.ndarray,
) -> dict[str, float]:
    stats = assignment_stats(cfg, labels, gain, demand)
    loads = stats["loads"]
    active = stats["active"]
    served = stats["served"]
    energy = network_energy(cfg, labels, gain, demand)
    return {
        "energy_w": energy,
        "served_ratio": float(np.mean(served)),
        "active_bs": float(np.sum(active)),
        "max_load": float(np.max(loads)),
        "mean_active_load": float(np.mean(loads[active])) if np.any(active) else 0.0,
    }


def repair_assignment(
    cfg: SimConfig,
    preferred_labels: np.ndarray,
    gain: np.ndarray,
    demand: np.ndarray,
    allowed_active: np.ndarray | None = None,
) -> np.ndarray:
    """Convert association preferences into a QoS/load-guarded decision."""
    preferred_labels = np.asarray(preferred_labels, dtype=int)
    active = np.zeros(cfg.num_bs, dtype=bool)
    if allowed_active is None:
        active[np.unique(preferred_labels)] = True
    else:
        active[np.asarray(allowed_active, dtype=int)] = True
    if not np.any(active):
        active[np.argmax(gain.mean(axis=0))] = True

    for _ in range(cfg.num_bs):
        labels = _assign_with_active_set(cfg, preferred_labels, gain, demand, active)
        stats = assignment_stats(cfg, labels, gain, demand)
        if float(np.mean(stats["served"])) >= 0.999 and float(np.max(stats["loads"])) <= cfg.load_limit + 1e-9:
            return labels
        if active.all():
            return labels
        overload = stats["loads"][labels] > cfg.load_limit
        unserved = np.logical_not(stats["served"])
        stressed = np.where(np.logical_or(overload, unserved))[0]
        if stressed.size == 0:
            stressed = np.argsort(-demand)[: max(1, min(5, demand.shape[0]))]
        inactive = np.where(~active)[0]
        scores = (gain[stressed][:, inactive] * demand[stressed, None]).sum(axis=0)
        active[int(inactive[int(np.argmax(scores))])] = True
    return _assign_with_active_set(cfg, preferred_labels, gain, demand, active)


def _assign_with_active_set(
    cfg: SimConfig,
    preferred_labels: np.ndarray,
    gain: np.ndarray,
    demand: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    sinr = downlink_sinr(cfg, gain, active)
    eff = spectral_efficiency(sinr)
    labels = np.full(preferred_labels.shape[0], -1, dtype=int)
    loads = np.zeros(cfg.num_bs, dtype=float)
    active_indices = np.where(active)[0]
    for ue in np.argsort(-demand):
        preferred = int(preferred_labels[ue])
        ranked = []
        if active[preferred]:
            ranked.append(preferred)
        ranked.extend([int(bs) for bs in active_indices[np.argsort(-gain[ue, active_indices])] if int(bs) != preferred])
        choices = []
        soft_choices = []
        for rank, bs in enumerate(ranked):
            if w_to_dbm(cfg.bs_tx_power_w * gain[ue, bs]) < cfg.min_rsrp_dbm:
                continue
            bs_eff = eff[ue, bs]
            capacity = cfg.bandwidth_hz * bs_eff / 1e6
            if capacity < cfg.qos_mbps:
                continue
            load_inc = demand[ue] / max(capacity, 1e-6)
            projected = loads[bs] + load_inc
            preference_penalty = 0.03 * rank
            soft_choices.append((projected, -bs_eff, bs, load_inc))
            if projected <= cfg.load_limit:
                choices.append((projected + preference_penalty, -bs_eff, bs, load_inc))
        if choices or soft_choices:
            _, _, bs, load_inc = min(choices or soft_choices)
            labels[ue] = int(bs)
            loads[bs] += load_inc
        else:
            labels[ue] = int(active_indices[np.argmax(gain[ue, active_indices])])
    return labels


def node_features(
    cfg: SimConfig,
    ue_pos: np.ndarray,
    demand: np.ndarray,
    gain: np.ndarray,
    rsrp_dbm: np.ndarray,
    sinr: np.ndarray,
    hour: int,
) -> np.ndarray:
    pos = ue_pos / cfg.area_radius_m
    demand_norm = demand[:, None] / cfg.demand_high_mbps
    hour_angle = 2.0 * np.pi * hour / cfg.temporal_periods
    hour_features = np.tile([np.sin(hour_angle), np.cos(hour_angle)], (ue_pos.shape[0], 1))
    rsrp_norm = np.clip((rsrp_dbm + 130.0) / 70.0, 0.0, 1.5)
    sinr_norm = np.clip(np.log1p(sinr) / 6.0, 0.0, 1.5)
    best_rsrp = np.max(rsrp_norm, axis=1, keepdims=True)
    second_rsrp = np.partition(rsrp_norm, -2, axis=1)[:, -2:-1]
    return np.hstack([pos, demand_norm, hour_features, best_rsrp, best_rsrp - second_rsrp, rsrp_norm, sinr_norm])


def build_layers(
    cfg: SimConfig,
    ue_pos: np.ndarray,
    demand: np.ndarray,
    gain: np.ndarray,
    labels: np.ndarray,
    hour: int,
) -> dict[str, np.ndarray]:
    top_bs = np.argmax(gain, axis=1)
    n = ue_pos.shape[0]
    association = (top_bs[:, None] == top_bs[None, :]).astype(float)
    np.fill_diagonal(association, 0.0)

    channel_profile = gain / np.maximum(gain.sum(axis=1, keepdims=True), 1e-18)
    interference = channel_profile @ channel_profile.T
    np.fill_diagonal(interference, 0.0)
    interference = threshold_topk(interference, k=min(10, n - 1))

    demand_distance = np.abs(demand[:, None] - demand[None, :]) / max(cfg.demand_high_mbps, 1e-12)
    load = np.exp(-4.0 * demand_distance) * association
    load = threshold_topk(load, k=min(8, n - 1))

    pos_dist = np.linalg.norm(ue_pos[:, None, :] - ue_pos[None, :, :], axis=2) / cfg.area_radius_m
    hour_factor = 0.7 + 0.3 * np.sin(2.0 * np.pi * hour / cfg.temporal_periods) ** 2
    temporal = np.exp(-2.0 * pos_dist) * np.exp(-2.0 * demand_distance) * hour_factor
    np.fill_diagonal(temporal, 0.0)
    temporal = threshold_topk(temporal, k=min(8, n - 1))

    layers = {
        "association": normalize_rows(association),
        "interference": normalize_rows(interference),
        "load": normalize_rows(load),
        "temporal": normalize_rows(temporal),
    }
    assert set(layers) == set(LAYER_NAMES)
    return layers


def threshold_topk(weights: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(weights)
    out = np.zeros_like(weights)
    idx = np.argpartition(weights, -k, axis=1)[:, -k:]
    rows = np.arange(weights.shape[0])[:, None]
    out[rows, idx] = weights[rows, idx]
    out = np.maximum(out, out.T)
    return out
