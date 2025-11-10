from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh


TARGETS = ["target_total", "target_margin"]


def _feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
    exclude = {"game_id", "date", "home_team", "away_team"} | set(TARGETS)
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
    """Fit ridge regression with standardized features and an explicit bias term.

    Returns (W, b, mu, sigma) where prediction is:
        y_hat = ((X - mu) / sigma) @ W + b
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xn = (X - mu) / sigma
    n, d = Xn.shape
    # Augment with ones for bias (no regularization on bias)
    Xa = np.concatenate([Xn, np.ones((n, 1), dtype=Xn.dtype)], axis=1)
    I = np.eye(d + 1, dtype=Xn.dtype)
    I[-1, -1] = 0.0  # do not regularize bias
    A = Xa.T @ Xa + alpha * I
    w_ext = np.linalg.solve(A, Xa.T @ y)
    W = w_ext[:-1]
    b = float(w_ext[-1])
    return W, b, mu, sigma


def _export_onnx_linear(out_path: Path, in_dim: int, W: np.ndarray, b: float, mu: np.ndarray, sigma: np.ndarray) -> None:
    # ONNX graph: y = Gemm((input - mu) / sigma, W^T, b)
    X = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, in_dim])
    Y = oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None, 1])

    mu_init = onh.from_array(mu.astype(np.float32), name="mu")
    sigma_init = onh.from_array(sigma.astype(np.float32), name="sigma")
    W_init = onh.from_array(W.reshape(1, -1).astype(np.float32), name="W")  # shape (1, d)
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
        name="ridge_linear",
        inputs=[X],
        outputs=[Y],
        initializer=[mu_init, sigma_init, W_init, b_init],
    )
    model = oh.make_model(graph, producer_name="ncaab_model", opset_imports=[oh.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path.as_posix())


def _score_fold(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, alpha: float, loss: str, huber_delta: float) -> float:
    if loss == "huber":
        W, b, mu, sigma = _huber_fit(X_tr, y_tr, alpha=alpha, delta=huber_delta)
    else:
        W, b, mu, sigma = _ridge_fit(X_tr, y_tr, alpha=alpha)
    y_hat = ((X_va - mu) / sigma) @ W + b
    return float(np.mean(np.abs(y_va - y_hat)))


def _huber_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    delta: float = 8.0,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit linear model with Huber loss using IRLS and ridge regularization on weights (not bias).

    Returns (W, b, mu, sigma) with the same standardization as _ridge_fit.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xn = (X - mu) / sigma
    n, d = Xn.shape
    Xa = np.concatenate([Xn, np.ones((n, 1), dtype=Xn.dtype)], axis=1)
    I = np.eye(d + 1, dtype=Xn.dtype)
    I[-1, -1] = 0.0  # do not regularize bias

    # Initialize with ridge solution
    A0 = Xa.T @ Xa + alpha * I
    w_ext = np.linalg.solve(A0, Xa.T @ y)
    prev = w_ext.copy()

    for _ in range(max_iter):
        r = y - Xa @ w_ext
        # Huber weights: w_i = 1 if |r|<=delta else delta/|r|
        abs_r = np.abs(r) + 1e-8
        w_i = np.where(abs_r <= delta, 1.0, delta / abs_r)
        Wm = np.diag(w_i.astype(Xn.dtype))
        A = Xa.T @ Wm @ Xa + alpha * I
        b_vec = Xa.T @ (Wm @ y)
        w_ext = np.linalg.solve(A, b_vec)
        if np.linalg.norm(w_ext - prev) <= tol * (np.linalg.norm(prev) + 1e-8):
            break
        prev = w_ext.copy()

    W = w_ext[:-1]
    b = float(w_ext[-1])
    return W, b, mu, sigma


def train_baseline(
    features_csv: Path,
    out_dir: Path,
    alpha: float = 1.0,
    alpha_grid: list[float] | None = None,
    k_folds: int = 5,
    loss_totals: str = "ridge",
    huber_delta: float = 8.0,
) -> dict:
    df = pd.read_csv(features_csv)
    df = df.dropna(subset=TARGETS)
    X, cols = _feature_matrix(df)
    results: dict = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        y = df[target].to_numpy(dtype=np.float32)
        # Cross-validate alpha if grid provided or by default
        grid = alpha_grid if alpha_grid else [0.1, 0.3, 1.0, 3.0, 10.0]
        best_alpha = float(alpha)
        best_cv_mae = float("inf")
        if k_folds and k_folds > 1 and len(y) >= max(20, k_folds):
            # Manual K-fold split (deterministic)
            n = len(y)
            idx = np.arange(n)
            # Stratification not applicable; use simple round-robin folds
            folds = [idx[i::k_folds] for i in range(k_folds)]
            cv_scores: dict[float, list[float]] = {a: [] for a in grid}
            for i in range(k_folds):
                va_idx = folds[i]
                tr_idx = np.setdiff1d(idx, va_idx, assume_unique=False)
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_va, y_va = X[va_idx], y[va_idx]
                for a in grid:
                    loss_mode = (loss_totals if target == "target_total" else "ridge")
                    cv_scores[a].append(_score_fold(X_tr, y_tr, X_va, y_va, alpha=float(a), loss=loss_mode, huber_delta=huber_delta))
            for a, scores in cv_scores.items():
                m = float(np.mean(scores)) if scores else float("inf")
                if m < best_cv_mae:
                    best_cv_mae = m
                    best_alpha = float(a)
        else:
            # Fallback to provided alpha
            best_alpha = float(alpha)
        # Fit final with best alpha
        if target == "target_total" and loss_totals == "huber":
            W, b, mu, sigma = _huber_fit(X, y, alpha=best_alpha, delta=huber_delta)
        else:
            W, b, mu, sigma = _ridge_fit(X, y, alpha=best_alpha)
        pred = ((X - mu) / sigma) @ W + b
        mae = float(np.mean(np.abs(y - pred)))
        onnx_path = out_dir / f"baseline_{target}.onnx"
        _export_onnx_linear(onnx_path, in_dim=X.shape[1], W=W, b=b, mu=mu, sigma=sigma)
        results[target] = {
            "mae_in_sample": mae,
            "onnx_path": str(onnx_path),
            "alpha": best_alpha,
            **({"cv_mae": best_cv_mae} if best_cv_mae != float("inf") else {}),
        }

    with open(out_dir / "feature_columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cols))
    results["feature_columns"] = cols
    return results
