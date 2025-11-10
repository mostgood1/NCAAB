from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class AccuracySummary:
    n: int
    mae_total: float | None
    rmse_total: float | None
    bias_total: float | None
    r2_total: float | None
    mae_margin: float | None
    rmse_margin: float | None
    bias_margin: float | None
    r2_margin: float | None


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float | None, float | None, float | None, float | None]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return None, None, None, None
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    # r2 with simple formula; guard for zero variance
    var = np.var(yt)
    if var <= 1e-12:
        r2 = None
    else:
        r2 = float(1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2))
    return mae, rmse, bias, r2


def compute_accuracy(games: pd.DataFrame, preds: pd.DataFrame) -> Tuple[pd.DataFrame, AccuracySummary]:
    g = games.copy()
    p = preds.copy()
    # realized totals and margins
    g["total_actual"] = g[["home_score", "away_score"]].sum(axis=1, min_count=2)
    g["margin_actual"] = g["home_score"] - g["away_score"]
    # normalize ids dtype
    if "game_id" in g.columns:
        g["game_id"] = g["game_id"].astype(str)
    if "game_id" in p.columns:
        p["game_id"] = p["game_id"].astype(str)
    m = g.merge(p[["game_id", "pred_total", "pred_margin"]], on="game_id", how="inner")

    # per-row errors
    m["err_total"] = m["pred_total"] - m["total_actual"]
    m["abs_err_total"] = m["err_total"].abs()
    m["err_margin"] = m["pred_margin"] - m["margin_actual"]
    m["abs_err_margin"] = m["err_margin"].abs()

    # summary
    mae_t, rmse_t, bias_t, r2_t = _safe_metrics(m["total_actual"].to_numpy(float), m["pred_total"].to_numpy(float))
    mae_m, rmse_m, bias_m, r2_m = _safe_metrics(m["margin_actual"].to_numpy(float), m["pred_margin"].to_numpy(float))
    summ = AccuracySummary(
        n=int(len(m)),
        mae_total=mae_t,
        rmse_total=rmse_t,
        bias_total=bias_t,
        r2_total=r2_t,
        mae_margin=mae_m,
        rmse_margin=rmse_m,
        bias_margin=bias_m,
        r2_margin=r2_m,
    )
    return m, summ


def compare_vs_closing(games: pd.DataFrame, closing: pd.DataFrame, preds: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Compare model accuracy vs closing totals.

    Returns per-game table and a summary dict with:
      - n
      - mae_model, mae_closing, mae_delta (closing - model)
      - beat_rate (fraction where |model error| < |closing error|)
      - edge_corr: corr((pred_total - closing_total), (total_actual - closing_total))
    """
    g = games.copy()
    c = closing.copy()
    p = preds.copy()
    # filter to totals full_game
    if {"market", "period"}.issubset(c.columns):
        c = c[(c["market"] == "totals") & (c["period"] == "full_game")]
    # realized
    g["total_actual"] = g[["home_score", "away_score"]].sum(axis=1, min_count=2)
    if "game_id" in g.columns:
        g["game_id"] = g["game_id"].astype(str)
    if "game_id" in p.columns:
        p["game_id"] = p["game_id"].astype(str)
    # aggregate closing per game_id (average across books if multiple)
    if "game_id" in c.columns:
        c["game_id"] = c["game_id"].astype(str)
        cg = c.groupby("game_id", as_index=False)["total"].mean().rename(columns={"total": "closing_total"})
    else:
        raise ValueError("Closing lines must be joined with game_id. Use join-closing first.")

    m = g.merge(cg, on="game_id", how="inner").merge(p[["game_id", "pred_total"]], on="game_id", how="inner")
    m["err_model"] = m["pred_total"] - m["total_actual"]
    m["err_closing"] = m["closing_total"] - m["total_actual"]
    m["abs_err_model"] = m["err_model"].abs()
    m["abs_err_closing"] = m["err_closing"].abs()
    # who beats whom
    m["model_beats_closing"] = m["abs_err_model"] < m["abs_err_closing"]
    # edge correlation
    m["edge"] = m["pred_total"] - m["closing_total"]
    m["realized_over_closing"] = m["total_actual"] - m["closing_total"]
    mask = np.isfinite(m["edge"]) & np.isfinite(m["realized_over_closing"])  # noqa
    edge_corr = float(np.corrcoef(m.loc[mask, "edge"], m.loc[mask, "realized_over_closing"])[0, 1]) if mask.sum() > 1 else None

    mae_model = float(m["abs_err_model"].mean()) if len(m) else None
    mae_closing = float(m["abs_err_closing"].mean()) if len(m) else None
    beat_rate = float(m["model_beats_closing"].mean()) if len(m) else None
    summary = {
        "n": int(len(m)),
        "mae_model": mae_model,
        "mae_closing": mae_closing,
        "mae_delta": (mae_closing - mae_model) if (mae_model is not None and mae_closing is not None) else None,
        "beat_rate": beat_rate,
        "edge_corr": edge_corr,
    }
    return m, summary
