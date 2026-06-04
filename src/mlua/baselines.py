from __future__ import annotations

import numpy as np

from .config import SimConfig
from .simulator import downlink_sinr, spectral_efficiency, w_to_dbm


def rsrp_association(gain: np.ndarray) -> np.ndarray:
    return np.argmax(gain, axis=1)


def sinr_association(cfg: SimConfig, gain: np.ndarray) -> np.ndarray:
    sinr = downlink_sinr(cfg, gain, np.ones(cfg.num_bs, dtype=bool))
    return np.argmax(sinr, axis=1)


def load_aware_association(cfg: SimConfig, gain: np.ndarray, demand: np.ndarray) -> np.ndarray:
    sinr = downlink_sinr(cfg, gain, np.ones(cfg.num_bs, dtype=bool))
    eff = spectral_efficiency(sinr)
    labels = np.full(demand.shape[0], -1, dtype=int)
    loads = np.zeros(cfg.num_bs, dtype=float)
    for ue in np.argsort(-demand):
        choices = []
        for bs in range(cfg.num_bs):
            if w_to_dbm(cfg.bs_tx_power_w * gain[ue, bs]) < cfg.min_rsrp_dbm:
                continue
            capacity = cfg.bandwidth_hz * eff[ue, bs] / 1e6
            load_inc = demand[ue] / max(capacity, 1e-6)
            choices.append((loads[bs] + load_inc, -eff[ue, bs], bs, load_inc))
        if not choices:
            labels[ue] = int(np.argmax(gain[ue]))
            continue
        _, _, bs, load_inc = min(choices)
        labels[ue] = int(bs)
        loads[bs] += load_inc
    return labels


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
    raise ValueError(f"Unknown baseline: {name}")

