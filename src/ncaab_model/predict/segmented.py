from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import math
import pandas as pd
import numpy as np

# This module scores per-segment linear models trained by train.segmented.train_segmented.
# Model JSONL entries contain:
#  - segment_key
#  - n_rows
#  - feature_columns
#  - weights_total, bias_total, mu_total, sigma_total
#  - weights_margin, bias_margin, mu_margin, sigma_margin


def _load_models(models_path: Path) -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {}
    with models_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = str(obj.get("segment_key"))
            if not key:
                continue
            models[key] = obj
    return models


def _feature_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    # Safe numeric extraction with NaN->0 fallback
    X = []
    for c in cols:
        if c in df.columns:
            X.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy())
        else:
            X.append(np.zeros(len(df), dtype=float))
    Xm = np.vstack(X).T.astype(np.float32)
    return Xm


def _apply_linear(X: np.ndarray, w: np.ndarray, b: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # Standardize with guard (sigma==0 -> 1)
    sig = np.where(np.asarray(sigma, dtype=float) == 0.0, 1.0, np.asarray(sigma, dtype=float))
    Xz = (X - np.asarray(mu, dtype=float)) / sig
    return (Xz @ np.asarray(w, dtype=float)) + float(b)


def _predict_for_segment(df: pd.DataFrame, model: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    cols = list(model.get("feature_columns", []))
    X = _feature_matrix(df, cols)
    yt = _apply_linear(X,
                       np.asarray(model.get("weights_total", []), dtype=float),
                       float(model.get("bias_total", 0.0)),
                       np.asarray(model.get("mu_total", [0.0]*len(cols)), dtype=float),
                       np.asarray(model.get("sigma_total", [1.0]*len(cols)), dtype=float))
    ym = _apply_linear(X,
                       np.asarray(model.get("weights_margin", []), dtype=float),
                       float(model.get("bias_margin", 0.0)),
                       np.asarray(model.get("mu_margin", [0.0]*len(cols)), dtype=float),
                       np.asarray(model.get("sigma_margin", [1.0]*len(cols)), dtype=float))
    return yt, ym


def score_segmented(features_csv: Path, models_path: Path, segment: str = "team") -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    if segment not in {"team", "conference"}:
        raise ValueError("segment must be 'team' or 'conference'")
    models = _load_models(models_path)
    if not models:
        raise ValueError(f"No models found in {models_path}")
    key_home = f"home_{'team' if segment=='team' else 'conference'}"
    key_away = f"away_{'team' if segment=='team' else 'conference'}"
    if key_home not in df.columns or key_away not in df.columns:
        raise ValueError(f"Missing {key_home}/{key_away} columns in features")
    # Prepare result columns
    df_out = df.copy()
    df_out["pred_total_seg_home"] = np.nan
    df_out["pred_total_seg_away"] = np.nan
    df_out["pred_margin_seg_home"] = np.nan
    df_out["pred_margin_seg_away"] = np.nan
    df_out["seg_n_rows_home"] = np.nan
    df_out["seg_n_rows_away"] = np.nan

    # Vectorized grouping per segment to reduce overhead
    for side, key_col, pref in [("home", key_home, "home"), ("away", key_away, "away")]:
        # Group by segment key and apply that key's model if present
        for seg_key, g in df.groupby(key_col):
            seg_key = str(seg_key)
            model = models.get(seg_key)
            if model is None or g.empty:
                continue
            yt, ym = _predict_for_segment(g, model)
            idx = g.index
            df_out.loc[idx, f"pred_total_seg_{pref}"] = yt
            df_out.loc[idx, f"pred_margin_seg_{pref}"] = ym
            df_out.loc[idx, f"seg_n_rows_{pref}"] = float(model.get("n_rows") or 0)

    # Aggregate to per-row segment predictions (average home/away when both exist)
    df_out["pred_total_seg"] = df_out[["pred_total_seg_home", "pred_total_seg_away"]].mean(axis=1, skipna=True)
    df_out["pred_margin_seg"] = df_out[["pred_margin_seg_home", "pred_margin_seg_away"]].mean(axis=1, skipna=True)
    return df_out


def blend_predictions(
    base_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    min_rows: int = 25,
    max_weight: float = 0.6,
) -> pd.DataFrame:
    # Merge on game_id when present; else fall back to (date, teams) keys if available
    df = base_df.copy()
    on_cols = [c for c in ["game_id"] if c in base_df.columns and c in seg_df.columns]
    if on_cols:
        m = df.merge(seg_df[[*on_cols, "pred_total_seg", "pred_margin_seg", "seg_n_rows_home", "seg_n_rows_away"]], on=on_cols, how="left")
    else:
        # Best-effort fallback
        m = df
    # Effective rows = min(home, away); if one side missing, use the available side
    eff_n = m[["seg_n_rows_home", "seg_n_rows_away"]].min(axis=1, skipna=True)
    eff_n = eff_n.fillna(m[["seg_n_rows_home", "seg_n_rows_away"]].max(axis=1, skipna=True))
    # Weight grows with sample size; cap at max_weight
    def _weight(n: float) -> float:
        try:
            return float(max(0.0, min(max_weight, (float(n) - float(min_rows)) / (float(min_rows) * 3.0))))
        except Exception:
            return 0.0
    w = eff_n.map(_weight)
    # Preserve baseline and segmented columns explicitly for downstream diagnostics
    if "pred_total" in m.columns:
        m["pred_total_base"] = pd.to_numeric(m["pred_total"], errors="coerce")
    if "pred_margin" in m.columns:
        m["pred_margin_base"] = pd.to_numeric(m["pred_margin"], errors="coerce")
    if "pred_total_seg" in m.columns:
        m["pred_total_seg"] = pd.to_numeric(m["pred_total_seg"], errors="coerce")
    if "pred_margin_seg" in m.columns:
        m["pred_margin_seg"] = pd.to_numeric(m["pred_margin_seg"], errors="coerce")

    # Blend
    blend_total = []
    blend_margin = []
    for bt, st, bm, sm, wt in zip(
        m.get("pred_total_base", pd.Series(dtype=float)),
        m.get("pred_total_seg", pd.Series(dtype=float)),
        m.get("pred_margin_base", pd.Series(dtype=float)),
        m.get("pred_margin_seg", pd.Series(dtype=float)),
        w,
    ):
        # If segmented missing (NaN), force weight 0
        if pd.isna(st):
            blend_total.append(bt)
        else:
            blend_total.append((1.0 - wt) * bt + wt * st)
        if pd.isna(sm):
            blend_margin.append(bm)
        else:
            blend_margin.append((1.0 - wt) * bm + wt * sm)
    m["pred_total_blend"] = blend_total
    m["pred_margin_blend"] = blend_margin
    m["blend_weight"] = w.astype(float)
    m["seg_eff_n_rows"] = eff_n.astype(float)
    return m
