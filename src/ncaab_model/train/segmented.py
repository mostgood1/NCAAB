from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from .baseline import _feature_matrix, _ridge_fit

# Segmented training across team or conference dimensions.
# Produces per-segment linear model weights saved to CSV instead of ONNX for simplicity (small models).


def train_segmented(features_csv: Path, out_dir: Path, segment: str = "team", min_rows: int = 25, alpha: float = 1.0) -> Dict[str, Dict]:
    df = pd.read_csv(features_csv)
    if segment not in {"team", "conference"}:
        raise ValueError("segment must be 'team' or 'conference'")
    key_col_home = f"home_{'team' if segment=='team' else 'conference'}"
    key_col_away = f"away_{'team' if segment=='team' else 'conference'}"
    if key_col_home not in df.columns or key_col_away not in df.columns:
        raise ValueError(f"Missing {key_col_home}/{key_col_away} columns in features for segmentation")
    # Build per-segment row sets by stacking home and away occurrences with side indicator
    parts = []
    for side, col in [("home", key_col_home), ("away", key_col_away)]:
        sub = df.copy()
        sub["segment_key"] = sub[col].astype(str)
        parts.append(sub)
    seg_df = pd.concat(parts, ignore_index=True)
    # Feature matrix global to ensure consistent column ordering
    X_all, cols = _feature_matrix(seg_df)
    # Map segment_key to indices
    models: Dict[str, Dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for seg_key, group in seg_df.groupby("segment_key"):
        if len(group) < min_rows:
            continue
        Xg, _ = _feature_matrix(group)
        # Targets
        if {"target_total", "target_margin"}.issubset(group.columns):
            y_tot = group["target_total"].to_numpy(dtype=np.float32)
            y_mar = group["target_margin"].to_numpy(dtype=np.float32)
            Wt, bt, mut, sigt = _ridge_fit(Xg, y_tot, alpha=alpha)
            Wm, bm, mum, sigm = _ridge_fit(Xg, y_mar, alpha=alpha)
            models[seg_key] = {
                "n_rows": len(group),
                "weights_total": Wt.tolist(),
                "bias_total": float(bt),
                "mu_total": mut.tolist(),
                "sigma_total": sigt.tolist(),
                "weights_margin": Wm.tolist(),
                "bias_margin": float(bm),
                "mu_margin": mum.tolist(),
                "sigma_margin": sigm.tolist(),
                "feature_columns": cols,
            }
    # Persist models JSONL
    out_path = out_dir / f"segmented_{segment}_models.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for k, v in models.items():
            import json
            v["segment_key"] = k
            f.write(json.dumps(v) + "\n")
    return {"segment": segment, "n_models": len(models), "model_path": str(out_path)}
