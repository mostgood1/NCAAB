from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

# Half targets we will train if data available
HALF_TARGETS = [
    "target_total_1h",
    "target_total_2h",
    "target_margin_1h",
    "target_margin_2h",
]


def _feature_matrix(df: pd.DataFrame, exclude_extra: set[str] | None = None) -> Tuple[np.ndarray, list[str]]:
    # Exclude identifiers, targets, and any leakage columns (actual half scores)
    exclude = {"game_id", "date", "home_team", "away_team"} | set(HALF_TARGETS) | {
        "home_score_1h", "away_score_1h", "home_score_2h", "away_score_2h"
    }
    if exclude_extra:
        exclude |= set(exclude_extra)
    cols = [
        c
        for c in df.columns
        if c not in exclude
        and not str(c).startswith("target_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[cols].fillna(0.0).to_numpy(dtype=np.float32)
    return X, cols


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xn = (X - mu) / sigma
    n, d = Xn.shape
    Xa = np.concatenate([Xn, np.ones((n, 1), dtype=Xn.dtype)], axis=1)
    I = np.eye(d + 1, dtype=Xn.dtype)
    I[-1, -1] = 0.0
    A = Xa.T @ Xa + alpha * I
    w_ext = np.linalg.solve(A, Xa.T @ y)
    W = w_ext[:-1]
    b = float(w_ext[-1])
    return W, b, mu, sigma


def _export_linear(out_path: Path, in_dim: int, W: np.ndarray, b: float, mu: np.ndarray, sigma: np.ndarray) -> None:
    X = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, in_dim])
    Y = oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None, 1])
    mu_init = onh.from_array(mu.astype(np.float32), name="mu")
    sigma_init = onh.from_array(sigma.astype(np.float32), name="sigma")
    W_init = onh.from_array(W.reshape(1, -1).astype(np.float32), name="W")
    b_init = onh.from_array(np.array([b], dtype=np.float32), name="B")
    sub = oh.make_node("Sub", inputs=["input", "mu"], outputs=["X_centered"])
    div = oh.make_node("Div", inputs=["X_centered", "sigma"], outputs=["X_norm"])
    gemm = oh.make_node(
        "Gemm",
        inputs=["X_norm", "W", "B"],
        outputs=["output"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )
    graph = oh.make_graph(
        nodes=[sub, div, gemm],
        name="ridge_linear_half",
        inputs=[X],
        outputs=[Y],
        initializer=[mu_init, sigma_init, W_init, b_init],
    )
    model = oh.make_model(graph, producer_name="ncaab_model", opset_imports=[oh.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path.as_posix())


def build_half_targets(features_csv: Path, games_csv: Path) -> pd.DataFrame:
    feats = pd.read_csv(features_csv)
    games = pd.read_csv(games_csv)
    if "game_id" not in feats.columns or "game_id" not in games.columns:
        raise ValueError("Both features and games files must contain game_id column for join.")
    games["game_id"] = games["game_id"].astype(str)
    feats["game_id"] = feats["game_id"].astype(str)
    merged = feats.merge(
        games[
            [
                "game_id",
                "home_score_1h",
                "away_score_1h",
                "home_score_2h",
                "away_score_2h",
            ]
        ],
        on="game_id",
        how="left",
    )
    # Compute targets; only keep rows where both sides present for each half
    def _safe_sum(a, b):
        return a + b if (pd.notna(a) and pd.notna(b)) else np.nan
    def _safe_diff(a, b):
        return a - b if (pd.notna(a) and pd.notna(b)) else np.nan
    merged["target_total_1h"] = [
        _safe_sum(h, a) for h, a in zip(merged.get("home_score_1h"), merged.get("away_score_1h"))
    ]
    merged["target_total_2h"] = [
        _safe_sum(h, a) for h, a in zip(merged.get("home_score_2h"), merged.get("away_score_2h"))
    ]
    merged["target_margin_1h"] = [
        _safe_diff(h, a) for h, a in zip(merged.get("home_score_1h"), merged.get("away_score_1h"))
    ]
    merged["target_margin_2h"] = [
        _safe_diff(h, a) for h, a in zip(merged.get("home_score_2h"), merged.get("away_score_2h"))
    ]
    return merged


def train_half_models(features_csv: Path, games_csv: Path, out_dir: Path, alpha: float = 1.0) -> dict:
    df = build_half_targets(features_csv, games_csv)
    # Drop rows without any half targets
    avail_targets = [t for t in HALF_TARGETS if t in df.columns and df[t].notna().any()]
    if not avail_targets:
        raise ValueError("No half targets available (home_score_1h/2h & away_score_1h/2h missing or all NaN).")
    # We exclude full-game target columns if present to keep feature matrix consistent
    exclude_extra = {"target_total", "target_margin"}
    X, cols = _feature_matrix(df, exclude_extra=exclude_extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"feature_columns": cols, "models": {}}
    for target in avail_targets:
        y = df[target].to_numpy(dtype=np.float32)
        # Remove rows with NaN target
        mask = np.isfinite(y)
        X_t = X[mask]
        y_t = y[mask]
        if len(y_t) < 25:
            # Skip training very small datasets to avoid overfit
            continue
        W, b, mu, sigma = _ridge_fit(X_t, y_t, alpha=alpha)
        pred = ((X_t - mu) / sigma) @ W + b
        mae = float(np.mean(np.abs(y_t - pred)))
        onnx_path = out_dir / f"baseline_{target}.onnx"
        _export_linear(onnx_path, in_dim=X_t.shape[1], W=W, b=b, mu=mu, sigma=sigma)
        results["models"][target] = {"mae_in_sample": mae, "onnx_path": str(onnx_path), "n": int(len(y_t))}
    # Save columns file (distinct so half models can be loaded separately)
    with open(out_dir / "feature_columns_halves.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cols))
    return results
