"""Inference harness: generates model-only predictions for an upcoming slate.

Usage:
  python -m src.modeling.infer --date 2025-11-19 --models-dir outputs/models

Writes: outputs/predictions_model_<date>.csv with columns:
  game_id, pred_total_model, pred_margin_model

If models are missing, exits gracefully.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import datetime as dt
import pandas as pd
import numpy as np
from joblib import load
from . import train_total as _train_total_mod  # noqa: F401  # Ensure custom wrapper classes are registered
from . import train_margin as _train_margin_mod  # noqa: F401

from .data import load_features
from .utils import canon_slug

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

# Wrapper classes reproduced to satisfy pickle resolution when training occurred under __main__
class _BoosterWrapper:
    def __init__(self, booster):
        self._b = booster
    def predict(self, X_any):  # noqa: ANN001
        return self._b.predict(X_any)

class _XGBWrapper:
    def __init__(self, booster):
        self._b = booster
    def predict(self, X_any):  # noqa: ANN001
        import xgboost as _xgb  # type: ignore
        return self._b.predict(_xgb.DMatrix(X_any))

class _MeanBaseline:
    def __init__(self, value: float):
        self.value = value
    def predict(self, X_any):  # noqa: ANN001
        import numpy as _np
        return _np.full(len(X_any), self.value)

class _ZeroBaseline:
    def predict(self, X_any):  # noqa: ANN001
        import numpy as _np
        return _np.zeros(len(X_any))


def _latest_model_path(name: str, models_root: pathlib.Path) -> pathlib.Path | None:
    candidates = sorted(models_root.glob("*/" + name), reverse=True)
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().strftime("%Y-%m-%d"))
    ap.add_argument("--models-dir", default=str(OUT / "models"))
    ap.add_argument("--features-file", default="features_curr.csv", help="Feature file for upcoming slate")
    args = ap.parse_args()
    date_str = args.date
    models_root = pathlib.Path(args.models_dir)
    feat_file = OUT / args.features_file if not pathlib.Path(args.features_file).is_absolute() else pathlib.Path(args.features_file)

    if not models_root.exists():
        print("Models directory missing", file=sys.stderr)
        sys.exit(0)
    total_p = _latest_model_path("total_model.pkl", models_root)
    margin_p = _latest_model_path("margin_model.pkl", models_root)
    if not (total_p and total_p.exists() and margin_p and margin_p.exists()):
        print("Model artifacts not found", file=sys.stderr)
        sys.exit(0)
    try:
        total_model = load(total_p)
    except Exception as e:
        print(f"Failed loading total model: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        margin_model = load(margin_p)
    except Exception as e:
        print(f"Failed loading margin model: {e}", file=sys.stderr)
        sys.exit(1)

    feat_df = load_features()
    if feat_df.empty:
        print("Features empty", file=sys.stderr)
        sys.exit(0)
    # Filter to requested date if a date column exists
    if "date" in feat_df.columns:
        feat_df["date"] = pd.to_datetime(feat_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        slate_df = feat_df[feat_df["date"] == date_str].copy()
    else:
        slate_df = feat_df.copy()
    if slate_df.empty:
        # Attempt fallback: explicitly load features_curr.csv if initial source lacked date rows
        try:
            fallback_path = OUT / "features_curr.csv"
            if fallback_path.exists():
                f2 = pd.read_csv(fallback_path)
                if "date" in f2.columns:
                    f2["date"] = pd.to_datetime(f2["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                slate_df = f2[f2.get("date").astype(str) == date_str] if "date" in f2.columns else f2
        except Exception:
            slate_df = pd.DataFrame()
        if slate_df.empty:
            print("No feature rows for date", file=sys.stderr)
            sys.exit(0)
    slate_df["game_id"] = slate_df.get("game_id").astype(str)
    # Attach latest prior team-level features (team_features.csv) if available so model meta feature list is satisfied
    # Attach team-level aggregates using exact-date merge then merge_asof fallback (vectorized)
    try:
        tf_path = OUT / "team_features.csv"
        if tf_path.exists() and {"home_team","away_team"}.issubset(slate_df.columns):
            tf = pd.read_csv(tf_path)
            if not tf.empty and {"team_slug","date"}.issubset(tf.columns):
                # Prepare types
                tf["date"] = pd.to_datetime(tf["date"], errors="coerce")
                slate_df["date"] = pd.to_datetime(slate_df.get("date"), errors="coerce")
                # Slugs
                slate_df["_home_slug"] = slate_df["home_team"].astype(str).map(canon_slug)
                slate_df["_away_slug"] = slate_df["away_team"].astype(str).map(canon_slug)
                # Exact date merge first
                home_tfeat = tf.copy(); away_tfeat = tf.copy()
                home_tfeat = home_tfeat.rename(columns={c: f"home_team_{c}" for c in home_tfeat.columns if c not in {"team_slug","date"}})
                away_tfeat = away_tfeat.rename(columns={c: f"away_team_{c}" for c in away_tfeat.columns if c not in {"team_slug","date"}})
                merged = slate_df.merge(home_tfeat, left_on=["_home_slug","date"], right_on=["team_slug","date"], how="left")
                merged = merged.merge(away_tfeat, left_on=["_away_slug","date"], right_on=["team_slug","date"], how="left", suffixes=("","_awaydup"))
                for drop_col in ["team_slug","team_slug_awaydup"]:
                    if drop_col in merged.columns:
                        merged.drop(columns=[drop_col], inplace=True)
                # Fallback merge_asof for missing team rows
                if not any(c.startswith("home_team_season_off_ppg") for c in merged.columns):
                    tf_sorted = tf.sort_values(["team_slug","date"]).copy()
                    merged_sorted = merged.sort_values(["_home_slug","date"]).copy()
                    import pandas as _pd
                    merged_sorted["date_dt"] = merged_sorted["date"]
                    tf_sorted["date_dt"] = tf_sorted["date"]
                    home_asof = _pd.merge_asof(
                        merged_sorted,
                        tf_sorted.sort_values(["team_slug","date_dt"]),
                        left_on="date_dt", right_on="date_dt", left_by="_home_slug", right_by="team_slug",
                        direction="backward", allow_exact_matches=True
                    )
                    away_asof = _pd.merge_asof(
                        merged_sorted,
                        tf_sorted.sort_values(["team_slug","date_dt"]),
                        left_on="date_dt", right_on="date_dt", left_by="_away_slug", right_by="team_slug",
                        direction="backward", allow_exact_matches=True
                    )
                    team_cols = [c for c in tf_sorted.columns if c not in {"team_slug","date","date_dt"}]
                    for c in team_cols:
                        hc = f"home_team_{c}"; ac = f"away_team_{c}"
                        if hc not in merged.columns:
                            merged[hc] = home_asof[c]
                        if ac not in merged.columns:
                            merged[ac] = away_asof[c]
                    if "date_dt" in merged.columns:
                        merged.drop(columns=["date_dt"], inplace=True)
                slate_df = merged
                # Differential features
                def _mk_diff(pair_name: str):
                    hc = f"home_team_{pair_name}"; ac = f"away_team_{pair_name}"; dc = f"diff_{pair_name}"
                    if hc in slate_df.columns and ac in slate_df.columns and dc not in slate_df.columns:
                        try:
                            # Boolean cast safeguard
                            if slate_df[hc].dtype == bool or slate_df[ac].dtype == bool:
                                lhs = slate_df[hc].astype(int)
                                rhs = slate_df[ac].astype(int)
                            else:
                                lhs = pd.to_numeric(slate_df[hc], errors="coerce")
                                rhs = pd.to_numeric(slate_df[ac], errors="coerce")
                            slate_df[dc] = lhs - rhs
                        except Exception:
                            pass
                for base in [
                    "season_off_ppg","season_def_ppg","last5_off_ppg","last5_def_ppg","last10_off_ppg","last10_def_ppg",
                    "rolling15_off_ppg","rolling15_def_ppg","rest_days","season_margin_std","season_total_std","ewm_off_ppg",
                    "ewm_def_ppg","ewm_margin_avg","back_to_back"
                ]:
                    _mk_diff(base)
    except Exception as e:
        print(f"WARNING: team feature merge failed: {e}", file=sys.stderr)
    # Dynamically reconstruct feature matrix based on model meta (ensures uniformity with training)
    meta_path_total = pathlib.Path(str(total_p).replace("total_model.pkl","total_meta.json")) if total_p else None
    meta_path_margin = pathlib.Path(str(margin_p).replace("margin_model.pkl","margin_meta.json")) if margin_p else None
    feature_list = []
    for mp in [meta_path_total, meta_path_margin]:
        if mp and mp.exists():
            try:
                import json as _json
                meta_obj = _json.loads(mp.read_text(encoding="utf-8"))
                fl = meta_obj.get("features") or []
                feature_list.extend([f for f in fl if f not in feature_list])
            except Exception:
                pass
    if not feature_list:
        # Fallback to baseline list
        feature_list = [
            "home_off_rating","away_off_rating","home_def_rating","away_def_rating",
            "off_diff","def_diff","tempo_avg","tempo_rating_sum"
        ]
    # Guarantee base columns present for diff computations
    base_ensure = ["home_off_rating","away_off_rating","home_def_rating","away_def_rating","home_tempo_rating","away_tempo_rating","tempo_rating_sum"]
    for c in base_ensure:
        if c not in slate_df.columns:
            slate_df[c] = np.nan
    # Reconstruct derived columns if referenced
    if "off_diff" in feature_list and "off_diff" not in slate_df.columns:
        slate_df["off_diff"] = slate_df["home_off_rating"] - slate_df["away_off_rating"]
    if "def_diff" in feature_list and "def_diff" not in slate_df.columns:
        slate_df["def_diff"] = slate_df["home_def_rating"] - slate_df["away_def_rating"]
    if "tempo_avg" in feature_list and "tempo_avg" not in slate_df.columns:
        slate_df["tempo_avg"] = (pd.to_numeric(slate_df["home_tempo_rating"], errors="coerce") + pd.to_numeric(slate_df["away_tempo_rating"], errors="coerce")) / 2.0
    # Build X by selecting available columns (including newly merged team aggregates), imputing with medians
    X = pd.DataFrame({f: slate_df.get(f) for f in feature_list})
    X = X.apply(pd.to_numeric, errors="coerce")
    for c in X.columns:
        if X[c].isna().any():
            # Fill with median or 0 if all NaN
            med = X[c].median()
            if pd.isna(med):
                med = 0.0
            X[c] = X[c].fillna(med)
    # Simple variance check to warn if uniform feature rows
    try:
        unique_rows = X.nunique(axis=0).sum()
        if X.nunique().sum() == len(X.columns):  # each column single value
            print("WARNING: Inference feature columns all constant; predictions likely uniform", file=sys.stderr)
    except Exception:
        pass
    try:
        pred_total = total_model.predict(X)
    except Exception:
        pred_total = np.full(len(X), np.nan)
    try:
        pred_margin = margin_model.predict(X)
    except Exception:
        pred_margin = np.full(len(X), np.nan)
    out_df = pd.DataFrame({
        "game_id": slate_df["game_id"],
        "pred_total_model": pred_total,
        "pred_margin_model": pred_margin,
        "pred_total_model_basis": "model_v1"
    })
    # Deduplicate game rows (keep first prediction per game_id)
    out_df = out_df.groupby("game_id", as_index=False).first()
    out_path = OUT / f"predictions_model_{date_str}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")

if __name__ == "__main__":
    main()
