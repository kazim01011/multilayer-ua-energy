from __future__ import annotations

from dataclasses import dataclass, field


LAYER_NAMES = ("association", "interference", "load", "temporal")


@dataclass
class SimConfig:
    num_bs: int = 7
    num_ues: int = 80
    area_radius_m: float = 650.0
    carrier_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    bs_tx_power_w: float = 20.0
    noise_dbm: float = -94.0
    path_loss_exp: float = 3.2
    shadowing_db: float = 6.0
    min_rsrp_dbm: float = -115.0
    qos_mbps: float = 1.0
    demand_low_mbps: float = 0.5
    demand_high_mbps: float = 6.0
    bs_fixed_power_w: float = 90.0
    bs_dynamic_power_w: float = 55.0
    bs_sleep_power_w: float = 8.0
    bs_switching_cost_w: float = 35.0
    load_limit: float = 1.0
    temporal_periods: int = 24
    seed: int = 7


@dataclass
class ExperimentConfig:
    run_name: str = "default"
    train_graphs: int = 180
    test_graphs: int = 60
    epochs: int = 160
    patience: int = 30
    hidden_dim: int = 32
    learning_rate: float = 0.015
    seed: int = 7
    models: tuple[str, ...] = (
        "flat_mlp",
        "agg_gcn",
        "ml_gcn",
        "attn_ml_gcn",
    )
    baselines: tuple[str, ...] = (
        "rsrp",
        "sinr",
        "load_aware",
        "rsrp_sleep",
        "sinr_sleep",
        "load_sleep",
        "greedy_sleep",
        "oracle",
    )
    sim: SimConfig = field(default_factory=SimConfig)
