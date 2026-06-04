from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .simulator import UAGraph, downlink_sinr, spectral_efficiency, w_to_dbm


@dataclass
class ActiveSelector:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray

    def predict_scores(self, graphs: list[UAGraph], probabilities: list[np.ndarray], cfg: SimConfig) -> list[np.ndarray]:
        scores = []
        for graph, probs in zip(graphs, probabilities):
            x = active_features(graph, probs, cfg)
            x = (x - self.mean) / self.std
            logits = x @ self.weights + self.bias
            scores.append(sigmoid(logits))
        return scores


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def active_features(graph: UAGraph, probs: np.ndarray, cfg: SimConfig) -> np.ndarray:
    gain = graph.context["gain"]
    demand = graph.context["demand_mbps"]
    rsrp_dbm = graph.context["rsrp_dbm"]
    ue_pos = graph.context["ue_pos"]
    bs_pos = graph.context["bs_pos"]
    sinr = downlink_sinr(cfg, gain, np.ones(cfg.num_bs, dtype=bool))
    capacity = cfg.bandwidth_hz * spectral_efficiency(sinr) / 1e6
    top_prob = np.argmax(probs, axis=1)
    top_gain = np.argmax(gain, axis=1)
    total_demand = max(float(demand.sum()), 1e-12)
    features = []
    for bs in range(cfg.num_bs):
        prob_bs = probs[:, bs]
        prob_load = float(np.sum(prob_bs * demand / np.maximum(capacity[:, bs], 1e-6)))
        best_load = float(np.sum(demand[top_gain == bs] / np.maximum(capacity[top_gain == bs, bs], 1e-6)))
        dist = np.linalg.norm(ue_pos - bs_pos[bs], axis=1) / cfg.area_radius_m
        features.append(
            [
                float(prob_bs.mean()),
                float(prob_bs.max()),
                float(np.sum(top_prob == bs) / probs.shape[0]),
                float(np.sum(prob_bs * demand) / total_demand),
                prob_load,
                float(np.sum(top_gain == bs) / probs.shape[0]),
                best_load,
                float(np.mean((rsrp_dbm[:, bs] + 130.0) / 70.0)),
                float(np.max((rsrp_dbm[:, bs] + 130.0) / 70.0)),
                float(np.mean(dist)),
            ]
        )
    return np.asarray(features, dtype=float)


def active_targets(graphs: list[UAGraph], cfg: SimConfig) -> np.ndarray:
    targets = []
    for graph in graphs:
        active = np.zeros(cfg.num_bs, dtype=float)
        active[np.unique(graph.labels)] = 1.0
        targets.append(active)
    return np.concatenate(targets)


def fit_active_selector(
    graphs: list[UAGraph],
    probabilities: list[np.ndarray],
    cfg: SimConfig,
    seed: int,
    epochs: int = 450,
    learning_rate: float = 0.08,
) -> ActiveSelector:
    x = np.vstack([active_features(g, p, cfg) for g, p in zip(graphs, probabilities)])
    y = active_targets(graphs, cfg)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    x_norm = (x - mean) / std
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, size=x_norm.shape[1])
    bias = 0.0
    pos_weight = float((1.0 - y).sum() / max(y.sum(), 1.0))
    sample_weight = np.where(y > 0.5, pos_weight, 1.0)
    normalizer = max(float(sample_weight.sum()), 1.0)
    for _ in range(epochs):
        logits = x_norm @ weights + bias
        pred = sigmoid(logits)
        err = (pred - y) * sample_weight / normalizer
        grad_w = x_norm.T @ err
        grad_b = float(err.sum())
        weights -= learning_rate * np.clip(grad_w, -3.0, 3.0)
        bias -= learning_rate * np.clip(grad_b, -3.0, 3.0)
    return ActiveSelector(weights=weights, bias=bias, mean=mean, std=std)

