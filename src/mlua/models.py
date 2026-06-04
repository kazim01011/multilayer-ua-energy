from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LAYER_NAMES
from .simulator import UAGraph


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=float)
    out[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return out


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int
    num_classes: int
    learning_rate: float
    seed: int
    layers: tuple[str, ...] = LAYER_NAMES


class BaseModel:
    name = "base"

    def fit(self, train_graphs: list[UAGraph], val_graphs: list[UAGraph], epochs: int, patience: int) -> None:
        raise NotImplementedError

    def predict_graphs(self, graphs: list[UAGraph]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba_graphs(self, graphs: list[UAGraph]) -> list[np.ndarray]:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, float]:
        return {}


class FlatMLP(BaseModel):
    name = "flat_mlp"

    def __init__(self, cfg: ModelConfig):
        self.rng = np.random.default_rng(cfg.seed)
        self.lr = cfg.learning_rate
        self.num_classes = cfg.num_classes
        h = cfg.hidden_dim
        self.weights = {
            "w1": self.rng.normal(0.0, np.sqrt(2.0 / cfg.input_dim), size=(cfg.input_dim, h)),
            "b1": np.zeros(h),
            "w2": self.rng.normal(0.0, np.sqrt(2.0 / h), size=(h, h)),
            "b2": np.zeros(h),
            "w3": self.rng.normal(0.0, np.sqrt(2.0 / h), size=(h, cfg.num_classes)),
            "b3": np.zeros(cfg.num_classes),
        }

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.weights["w1"] + self.weights["b1"]
        h1 = relu(z1)
        z2 = h1 @ self.weights["w2"] + self.weights["b2"]
        h2 = relu(z2)
        logits = h2 @ self.weights["w3"] + self.weights["b3"]
        return logits, {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}

    def _step(self, x: np.ndarray, y: np.ndarray) -> float:
        logits, cache = self._forward(x)
        probs = softmax(logits)
        target = one_hot(y, self.num_classes)
        loss = -float(np.mean(np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1)))
        dlogits = (probs - target) / max(y.shape[0], 1)
        dw3 = cache["h2"].T @ dlogits
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ self.weights["w3"].T
        dz2 = dh2 * relu_grad(cache["z2"])
        dw2 = cache["h1"].T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.weights["w2"].T
        dz1 = dh1 * relu_grad(cache["z1"])
        dw1 = cache["x"].T @ dz1
        db1 = dz1.sum(axis=0)
        for key, grad in {
            "w1": dw1,
            "b1": db1,
            "w2": dw2,
            "b2": db2,
            "w3": dw3,
            "b3": db3,
        }.items():
            self.weights[key] -= self.lr * np.clip(grad, -5.0, 5.0)
        return loss

    def fit(self, train_graphs: list[UAGraph], val_graphs: list[UAGraph], epochs: int, patience: int) -> None:
        del val_graphs, patience
        x = np.vstack([g.features for g in train_graphs])
        y = np.concatenate([g.labels for g in train_graphs])
        for _ in range(epochs):
            order = self.rng.permutation(x.shape[0])
            self._step(x[order], y[order])

    def predict_graphs(self, graphs: list[UAGraph]) -> np.ndarray:
        labels = []
        for probs in self.predict_proba_graphs(graphs):
            labels.append(np.argmax(probs, axis=1))
        return np.concatenate(labels)

    def predict_proba_graphs(self, graphs: list[UAGraph]) -> list[np.ndarray]:
        probs = []
        for graph in graphs:
            logits, _ = self._forward(graph.features)
            probs.append(softmax(logits))
        return probs


class AggregatedGCN(BaseModel):
    name = "agg_gcn"

    def __init__(self, cfg: ModelConfig):
        self.rng = np.random.default_rng(cfg.seed)
        self.lr = cfg.learning_rate
        self.layers = tuple(cfg.layers)
        self.num_classes = cfg.num_classes
        h = cfg.hidden_dim
        self.w = self.rng.normal(0.0, np.sqrt(2.0 / cfg.input_dim), size=(cfg.input_dim, h))
        self.out_w = self.rng.normal(0.0, np.sqrt(2.0 / h), size=(h, cfg.num_classes))
        self.out_b = np.zeros(cfg.num_classes)

    def _support(self, graph: UAGraph) -> np.ndarray:
        support = np.zeros_like(next(iter(graph.layers.values())))
        for layer in self.layers:
            support += graph.layers[layer]
        return support / max(len(self.layers), 1)

    def _forward_graph(self, graph: UAGraph) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        support = self._support(graph)
        ax = support @ graph.features
        z = ax @ self.w
        hidden = relu(z)
        logits = hidden @ self.out_w + self.out_b
        return logits, {"ax": ax, "z": z, "hidden": hidden}

    def _step(self, graphs: list[UAGraph]) -> float:
        grad_w = np.zeros_like(self.w)
        grad_out_w = np.zeros_like(self.out_w)
        grad_out_b = np.zeros_like(self.out_b)
        total_nodes = sum(g.labels.shape[0] for g in graphs)
        losses = []
        for graph in graphs:
            logits, cache = self._forward_graph(graph)
            probs = softmax(logits)
            target = one_hot(graph.labels, self.num_classes)
            losses.append(-float(np.mean(np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1))))
            dlogits = (probs - target) / max(total_nodes, 1)
            grad_out_w += cache["hidden"].T @ dlogits
            grad_out_b += dlogits.sum(axis=0)
            dh = dlogits @ self.out_w.T
            dz = dh * relu_grad(cache["z"])
            grad_w += cache["ax"].T @ dz
        self.out_w -= self.lr * np.clip(grad_out_w, -5.0, 5.0)
        self.out_b -= self.lr * np.clip(grad_out_b, -5.0, 5.0)
        self.w -= self.lr * np.clip(grad_w, -5.0, 5.0)
        return float(np.mean(losses))

    def fit(self, train_graphs: list[UAGraph], val_graphs: list[UAGraph], epochs: int, patience: int) -> None:
        best = float("inf")
        stale = 0
        for _ in range(epochs):
            loss = self._step(train_graphs)
            if loss < best - 1e-5:
                best = loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    def predict_graphs(self, graphs: list[UAGraph]) -> np.ndarray:
        labels = []
        for probs in self.predict_proba_graphs(graphs):
            labels.append(np.argmax(probs, axis=1))
        return np.concatenate(labels)

    def predict_proba_graphs(self, graphs: list[UAGraph]) -> list[np.ndarray]:
        probs = []
        for graph in graphs:
            logits, _ = self._forward_graph(graph)
            probs.append(softmax(logits))
        return probs


class MultilayerGCN(BaseModel):
    name = "ml_gcn"

    def __init__(self, cfg: ModelConfig):
        self.rng = np.random.default_rng(cfg.seed)
        self.lr = cfg.learning_rate
        self.layers = tuple(cfg.layers)
        self.num_classes = cfg.num_classes
        h = cfg.hidden_dim
        self.relation_weights = {
            layer: self.rng.normal(0.0, np.sqrt(2.0 / cfg.input_dim), size=(cfg.input_dim, h))
            for layer in self.layers
        }
        self.out_w = self.rng.normal(0.0, np.sqrt(2.0 / (h * len(self.layers))), size=(h * len(self.layers), cfg.num_classes))
        self.out_b = np.zeros(cfg.num_classes)

    def _forward_graph(self, graph: UAGraph) -> tuple[np.ndarray, dict[str, np.ndarray | dict[str, np.ndarray]]]:
        rel_inputs = {}
        rel_pre = {}
        rel_hidden = {}
        for layer in self.layers:
            ax = graph.layers[layer] @ graph.features
            z = ax @ self.relation_weights[layer]
            rel_inputs[layer] = ax
            rel_pre[layer] = z
            rel_hidden[layer] = relu(z)
        hidden = np.concatenate([rel_hidden[layer] for layer in self.layers], axis=1)
        logits = hidden @ self.out_w + self.out_b
        return logits, {"hidden": hidden, "rel_inputs": rel_inputs, "rel_pre": rel_pre}

    def _step(self, graphs: list[UAGraph]) -> float:
        grad_rel = {layer: np.zeros_like(w) for layer, w in self.relation_weights.items()}
        grad_out_w = np.zeros_like(self.out_w)
        grad_out_b = np.zeros_like(self.out_b)
        total_nodes = sum(g.labels.shape[0] for g in graphs)
        losses = []
        for graph in graphs:
            logits, cache = self._forward_graph(graph)
            probs = softmax(logits)
            target = one_hot(graph.labels, self.num_classes)
            losses.append(-float(np.mean(np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1))))
            dlogits = (probs - target) / max(total_nodes, 1)
            hidden = cache["hidden"]
            grad_out_w += hidden.T @ dlogits
            grad_out_b += dlogits.sum(axis=0)
            dhidden = dlogits @ self.out_w.T
            chunks = np.split(dhidden, len(self.layers), axis=1)
            for layer, dh in zip(self.layers, chunks):
                dz = dh * relu_grad(cache["rel_pre"][layer])
                grad_rel[layer] += cache["rel_inputs"][layer].T @ dz
        self.out_w -= self.lr * np.clip(grad_out_w, -5.0, 5.0)
        self.out_b -= self.lr * np.clip(grad_out_b, -5.0, 5.0)
        for layer in self.layers:
            self.relation_weights[layer] -= self.lr * np.clip(grad_rel[layer], -5.0, 5.0)
        return float(np.mean(losses))

    def fit(self, train_graphs: list[UAGraph], val_graphs: list[UAGraph], epochs: int, patience: int) -> None:
        best = float("inf")
        stale = 0
        for _ in range(epochs):
            loss = self._step(train_graphs)
            if loss < best - 1e-5:
                best = loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    def predict_graphs(self, graphs: list[UAGraph]) -> np.ndarray:
        labels = []
        for probs in self.predict_proba_graphs(graphs):
            labels.append(np.argmax(probs, axis=1))
        return np.concatenate(labels)

    def predict_proba_graphs(self, graphs: list[UAGraph]) -> list[np.ndarray]:
        probs = []
        for graph in graphs:
            logits, _ = self._forward_graph(graph)
            probs.append(softmax(logits))
        return probs


class AttentionMultilayerGCN(MultilayerGCN):
    name = "attn_ml_gcn"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.attention_logits = np.zeros(len(self.layers), dtype=float)

    def _attention(self) -> np.ndarray:
        shifted = self.attention_logits - self.attention_logits.max()
        weights = np.exp(shifted)
        return weights / np.maximum(weights.sum(), 1e-12)

    def _forward_graph(self, graph: UAGraph) -> tuple[np.ndarray, dict[str, np.ndarray | dict[str, np.ndarray]]]:
        alpha = self._attention()
        gate = len(self.layers) * alpha
        rel_inputs = {}
        rel_pre = {}
        rel_hidden = {}
        for idx, layer in enumerate(self.layers):
            ax = graph.layers[layer] @ graph.features
            z = ax @ self.relation_weights[layer]
            rel_inputs[layer] = ax
            rel_pre[layer] = z
            rel_hidden[layer] = relu(z)
        hidden = np.concatenate([gate[idx] * rel_hidden[layer] for idx, layer in enumerate(self.layers)], axis=1)
        logits = hidden @ self.out_w + self.out_b
        return logits, {
            "alpha": alpha,
            "gate": gate,
            "hidden": hidden,
            "rel_inputs": rel_inputs,
            "rel_pre": rel_pre,
            "rel_hidden": rel_hidden,
        }

    def _step(self, graphs: list[UAGraph]) -> float:
        grad_rel = {layer: np.zeros_like(w) for layer, w in self.relation_weights.items()}
        grad_attention = np.zeros_like(self.attention_logits)
        grad_out_w = np.zeros_like(self.out_w)
        grad_out_b = np.zeros_like(self.out_b)
        total_nodes = sum(g.labels.shape[0] for g in graphs)
        losses = []
        for graph in graphs:
            logits, cache = self._forward_graph(graph)
            probs = softmax(logits)
            target = one_hot(graph.labels, self.num_classes)
            losses.append(-float(np.mean(np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1))))
            dlogits = (probs - target) / max(total_nodes, 1)
            alpha = cache["alpha"]
            gate = cache["gate"]
            hidden = cache["hidden"]
            grad_out_w += hidden.T @ dlogits
            grad_out_b += dlogits.sum(axis=0)
            dhidden = dlogits @ self.out_w.T
            chunks = np.split(dhidden, len(self.layers), axis=1)
            grad_alpha = np.zeros_like(alpha)
            for idx, (layer, dh) in enumerate(zip(self.layers, chunks)):
                h = cache["rel_hidden"][layer]
                grad_alpha[idx] = len(self.layers) * np.sum(dh * h)
                dz = (gate[idx] * dh) * relu_grad(cache["rel_pre"][layer])
                grad_rel[layer] += cache["rel_inputs"][layer].T @ dz
            grad_attention += alpha * (grad_alpha - np.sum(alpha * grad_alpha))
        self.out_w -= self.lr * np.clip(grad_out_w, -5.0, 5.0)
        self.out_b -= self.lr * np.clip(grad_out_b, -5.0, 5.0)
        self.attention_logits -= self.lr * np.clip(grad_attention, -5.0, 5.0)
        for layer in self.layers:
            self.relation_weights[layer] -= self.lr * np.clip(grad_rel[layer], -5.0, 5.0)
        return float(np.mean(losses))

    def diagnostics(self) -> dict[str, float]:
        alpha = self._attention()
        return {f"attention_{layer}": float(alpha[idx]) for idx, layer in enumerate(self.layers)}


def build_model(name: str, cfg: ModelConfig) -> BaseModel:
    if name == "flat_mlp":
        return FlatMLP(cfg)
    if name.startswith("agg_gcn"):
        cfg = config_with_layers(cfg, name.replace("agg_gcn", "ml_gcn", 1))
        return AggregatedGCN(cfg)
    if name.startswith("ml_gcn"):
        cfg = config_with_layers(cfg, name)
        return MultilayerGCN(cfg)
    if name.startswith("attn_ml_gcn"):
        cfg = config_with_layers(cfg, name.replace("attn_", "", 1))
        return AttentionMultilayerGCN(cfg)
    raise ValueError(f"Unknown model: {name}")


def config_with_layers(cfg: ModelConfig, name: str) -> ModelConfig:
    layer_map = {
        "ml_gcn": LAYER_NAMES,
        "ml_gcn_association": ("association",),
        "ml_gcn_interference": ("interference",),
        "ml_gcn_load": ("load",),
        "ml_gcn_temporal": ("temporal",),
        "ml_gcn_no_association": tuple(layer for layer in LAYER_NAMES if layer != "association"),
        "ml_gcn_no_interference": tuple(layer for layer in LAYER_NAMES if layer != "interference"),
        "ml_gcn_no_load": tuple(layer for layer in LAYER_NAMES if layer != "load"),
        "ml_gcn_no_temporal": tuple(layer for layer in LAYER_NAMES if layer != "temporal"),
    }
    if name not in layer_map:
        return cfg
    return ModelConfig(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        layers=layer_map[name],
    )
