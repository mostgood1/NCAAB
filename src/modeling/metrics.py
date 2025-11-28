from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

try:
    from sklearn import metrics as skm  # type: ignore
except Exception:  # pragma: no cover
    skm = None  # type: ignore


def _safe(item: Any, default: float = float("nan")) -> float:
    try:
        if item is None:
            return default
        v = float(item)
        return v
    except Exception:
        return default


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute a simple ECE with equal-width bins in [0, 1]."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1]) if i < n_bins - 1 else (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if not np.any(mask):
            continue
        bin_acc = float(np.mean((y_true[mask] > 0.5).astype(float)))
        bin_conf = float(np.mean(y_prob[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(bin_acc - bin_conf)
    return float(ece)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Return standard classification metrics: accuracy, log_loss, brier, auc, ece."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, float] = {}
    if skm is not None:
        try:
            out["accuracy"] = _safe(skm.accuracy_score(y_true, y_pred))
        except Exception:
            out["accuracy"] = float("nan")
        try:
            out["log_loss"] = _safe(skm.log_loss(y_true, y_prob))
        except Exception:
            out["log_loss"] = float("nan")
        try:
            out["brier"] = _safe(skm.brier_score_loss(y_true, y_prob))
        except Exception:
            out["brier"] = float("nan")
        try:
            out["roc_auc"] = _safe(skm.roc_auc_score(y_true, y_prob))
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        # Fallbacks when sklearn not present
        out["accuracy"] = _safe(np.mean((y_true > 0.5) == (y_prob >= threshold)))
        # crude approximations for others
        out["log_loss"] = float("nan")
        out["brier"] = _safe(np.mean((y_prob - (y_true > 0.5)) ** 2))
        out["roc_auc"] = float("nan")

    try:
        out["ece"] = _safe(expected_calibration_error(y_true, y_prob))
    except Exception:
        out["ece"] = float("nan")
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    out: Dict[str, float] = {}
    if skm is not None:
        try:
            out["mae"] = _safe(skm.mean_absolute_error(y_true, y_pred))
        except Exception:
            out["mae"] = float("nan")
        try:
            out["rmse"] = _safe(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
        except Exception:
            out["rmse"] = float("nan")
        try:
            out["r2"] = _safe(skm.r2_score(y_true, y_pred))
        except Exception:
            out["r2"] = float("nan")
    else:
        out["mae"] = _safe(np.mean(np.abs(y_true - y_pred)))
        out["rmse"] = _safe(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        out["r2"] = float("nan")
    return out


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Return per-bin accuracy vs confidence and counts for plotting / analysis."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "acc": np.nan, "conf": (lo + hi) / 2})
            continue
        acc = float(np.mean((y_true[mask] > 0.5).astype(float)))
        conf = float(np.mean(y_prob[mask]))
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(np.sum(mask)), "acc": acc, "conf": conf})
    return {"bins": rows}


def sharpness(y_prob: np.ndarray) -> float:
    """Sharpness: variance of predicted probabilities as a simple dispersion proxy."""
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.var(y_prob))


def dispersion(y_prob: np.ndarray) -> float:
    """Mean absolute deviation from 0.5 (higher => more confident)."""
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean(np.abs(y_prob - 0.5)))
