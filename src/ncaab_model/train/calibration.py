from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ZRecenterArtifact:
    method: str
    created: str
    n_samples: int
    z_center: float
    z_scale: float
    metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _actual_total_from_row(r: pd.Series) -> float:
    # Prefer explicit target_total if present
    if "target_total" in r.index and pd.notna(r["target_total"]):
        return float(r["target_total"])
    # Else sum of scores if available
    for a, b in (("home_score", "away_score"), ("score_home", "score_away")):
        if a in r.index and b in r.index and pd.notna(r[a]) and pd.notna(r[b]):
            try:
                return float(r[a]) + float(r[b])
            except Exception:
                pass
    return np.nan


def build_z_recenter_artifact(df: pd.DataFrame, min_rows: int = 200) -> ZRecenterArtifact:
    """Compute global mean/scale adjustments for model z-scores.

    Expects columns pred_total_mu, pred_total_sigma, and either target_total or score columns.
    """
    req = {"pred_total_mu", "pred_total_sigma"}
    if not req.issubset(df.columns):
        raise ValueError(f"DataFrame missing required columns: {req}")
    # Actual totals
    if "target_total" not in df.columns and not ({"home_score", "away_score"}.issubset(df.columns) or {"score_home", "score_away"}.issubset(df.columns)):
        raise ValueError("No actual totals present (need target_total or score_* columns)")

    work = df.copy()
    work["actual_total"] = work.apply(_actual_total_from_row, axis=1)
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["pred_total_mu", "pred_total_sigma", "actual_total"])
    if work.empty or len(work) < min_rows:
        raise ValueError(f"Insufficient rows for calibration: {len(work)} < {min_rows}")
    # Guard against non-positive sigma
    work = work[work["pred_total_sigma"].astype(float) > 0.0]
    if work.empty or len(work) < min_rows:
        raise ValueError(f"Insufficient rows after sigma>0 filter: {len(work)} < {min_rows}")

    z_raw = (work["actual_total"].astype(float) - work["pred_total_mu"].astype(float)) / work["pred_total_sigma"].astype(float)
    # Robust center/scale: use mean/std with clipping of extreme outliers
    z_clipped = z_raw.clip(lower=-10.0, upper=10.0)
    m = float(np.nanmean(z_clipped))
    s = float(np.nanstd(z_clipped))
    if not np.isfinite(s) or s <= 1e-6:
        s = 1.0
    art = ZRecenterArtifact(
        method="z_recenter",
        created=_now_iso(),
        n_samples=int(len(work)),
        z_center=m,
        z_scale=s,
        metrics=None,
    )
    return art


def save_artifact(artifact: ZRecenterArtifact, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact.to_dict(), f, indent=2)


def load_artifact(path: Path) -> Optional[ZRecenterArtifact]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return ZRecenterArtifact(**data)
    except Exception:
        return None
