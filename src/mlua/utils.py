from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def dbm_to_w(dbm: float | np.ndarray) -> float | np.ndarray:
    return 10 ** ((dbm - 30.0) / 10.0)


def w_to_dbm(watts: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(watts, 1e-18)) + 30.0


def normalize_rows(a: np.ndarray) -> np.ndarray:
    out = a.astype(float).copy()
    out += np.eye(out.shape[0])
    row_sum = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sum, 1e-12)


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged

