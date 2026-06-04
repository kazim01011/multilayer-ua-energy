from __future__ import annotations

from mlua.config import ExperimentConfig, SimConfig
from mlua.experiment import run_experiment


def test_smoke_run(tmp_path):
    cfg = ExperimentConfig(
        run_name="pytest_smoke",
        train_graphs=4,
        test_graphs=2,
        epochs=3,
        patience=2,
        hidden_dim=8,
        learning_rate=0.02,
        sim=SimConfig(num_bs=4, num_ues=12, seed=3),
    )
    summary = run_experiment(cfg, tmp_path)
    assert not summary.empty
    assert {"oracle", "attn_ml_gcn"}.issubset(set(summary["policy"]))

