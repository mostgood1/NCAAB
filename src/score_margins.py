from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .model_totals import TotalsModel, _safe_read_csv, OUT


def _parse_date_safe(s: object) -> Optional[pd.Timestamp]:
    try:
        if s is None:
            return None
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts).normalize()
    except Exception:
        return None


def _augment_with_rolling_score_features(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Best-effort add rolling PF/PA/TOT means for home/away teams.

    Produces the same feature names as `train_margins_roll.py` so scoring and
    training stay aligned even if per-date `features_<date>.csv` is sparse.
    """
    if df is None or df.empty:
        return df
    if not {"home_team", "away_team"}.issubset(df.columns):
        return df

    target_date = _parse_date_safe(date_str)
    if target_date is None:
        return df

    daily_dir = OUT / "daily_results"
    if not daily_dir.exists():
        return df

    team_rows: list[dict] = []
    for p in sorted(daily_dir.glob("results_*.csv")):
        try:
            token = p.stem.replace("results_", "")
            d = _parse_date_safe(token)
            if d is None or d >= target_date:
                continue
            res = pd.read_csv(p)
        except Exception:
            continue
        if res is None or res.empty:
            continue

        try:
            if "completed" in res.columns:
                res = res[pd.to_numeric(res["completed"], errors="coerce").fillna(0).astype(int) == 1]
        except Exception:
            pass

        if not {"home_team", "away_team", "home_score", "away_score"}.issubset(res.columns):
            continue
        res = res.copy()
        res["home_team"] = res["home_team"].astype(str)
        res["away_team"] = res["away_team"].astype(str)
        hs = pd.to_numeric(res["home_score"], errors="coerce")
        aw = pd.to_numeric(res["away_score"], errors="coerce")
        tot = hs + aw

        for ht, at, hsp, asp, tp in zip(res["home_team"].tolist(), res["away_team"].tolist(), hs.tolist(), aw.tolist(), tot.tolist()):
            try:
                if pd.isna(hsp) or pd.isna(asp) or pd.isna(tp):
                    continue
                team_rows.append({"team": str(ht), "date": d, "pf": float(hsp), "pa": float(asp), "tot": float(tp)})
                team_rows.append({"team": str(at), "date": d, "pf": float(asp), "pa": float(hsp), "tot": float(tp)})
            except Exception:
                continue

    td = pd.DataFrame(team_rows)
    if td.empty:
        return df

    td = td.dropna(subset=["team", "date"]).copy()
    td["team"] = td["team"].astype(str)
    td["date"] = pd.to_datetime(td["date"], errors="coerce")
    td = td.dropna(subset=["date"]).sort_values(["team", "date"], kind="mergesort")

    try:
        td["gp"] = td.groupby("team").cumcount() + 1
    except Exception:
        td["gp"] = np.nan

    for w in (5, 15):
        for col in ("pf", "pa", "tot"):
            out_col = f"{col}{w}"
            try:
                td[out_col] = td.groupby("team")[col].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
            except Exception:
                td[out_col] = np.nan

    try:
        asof = td[td["date"] < target_date].groupby("team", as_index=False).tail(1)
    except Exception:
        asof = td
    if asof.empty:
        return df

    asof = asof.set_index("team")
    out = df.copy()
    out["home_team"] = out["home_team"].astype(str)
    out["away_team"] = out["away_team"].astype(str)

    def _map(team_series: pd.Series, key: str) -> pd.Series:
        try:
            m = asof[key].to_dict()
            return team_series.map(m)
        except Exception:
            return pd.Series([pd.NA] * len(team_series))

    out["home_gp"] = _map(out["home_team"], "gp")
    out["away_gp"] = _map(out["away_team"], "gp")

    for w in (5, 15):
        out[f"home_pf{w}"] = _map(out["home_team"], f"pf{w}")
        out[f"home_pa{w}"] = _map(out["home_team"], f"pa{w}")
        out[f"home_tot{w}"] = _map(out["home_team"], f"tot{w}")
        out[f"away_pf{w}"] = _map(out["away_team"], f"pf{w}")
        out[f"away_pa{w}"] = _map(out["away_team"], f"pa{w}")
        out[f"away_tot{w}"] = _map(out["away_team"], f"tot{w}")

    # Derived features matching training
    out["home_net5"] = pd.to_numeric(out["home_pf5"], errors="coerce") - pd.to_numeric(out["home_pa5"], errors="coerce")
    out["away_net5"] = pd.to_numeric(out["away_pf5"], errors="coerce") - pd.to_numeric(out["away_pa5"], errors="coerce")
    out["diff_net5"] = out["home_net5"] - out["away_net5"]

    out["home_net15"] = pd.to_numeric(out["home_pf15"], errors="coerce") - pd.to_numeric(out["home_pa15"], errors="coerce")
    out["away_net15"] = pd.to_numeric(out["away_pf15"], errors="coerce") - pd.to_numeric(out["away_pa15"], errors="coerce")
    out["diff_net15"] = out["home_net15"] - out["away_net15"]

    return out


def score_date(date_str: str, model_path: Path) -> dict:
    model = TotalsModel.load(model_path)

    candidates = [
        OUT / f"predictions_unified_enriched_{date_str}.csv",
        OUT / f"predictions_unified_{date_str}.csv",
        OUT / f"predictions_display_{date_str}.csv",
        OUT / f"features_{date_str}.csv",
        OUT / f"features_{date_str}_augmented.csv",
        OUT / "features_curr.csv",
        OUT / "features_curr_augmented.csv",
    ]
    src = None
    df = pd.DataFrame()
    for p in candidates:
        if p.exists():
            df = _safe_read_csv(p)
            if not df.empty:
                src = str(p)
                break
    if df.empty:
        return {"error": "No features found for date", "date": date_str}

    if "game_id" not in df.columns:
        return {"error": "Missing game_id in source frame", "date": date_str, "source": src}

    try:
        df = _augment_with_rolling_score_features(df, date_str)
    except Exception:
        pass

    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)

    # Predict (numeric feature frame only; model aligns/handles missing values)
    preds = model.predict(df.select_dtypes(include=[np.number]))
    out_df = pd.DataFrame({
        "date": date_str,
        "game_id": df["game_id"].astype(str) if "game_id" in df.columns else pd.Series([""] * len(df)),
        "pred_margin_model": preds.get("mean"),
        "pred_margin_q10": preds.get("q0.1"),
        "pred_margin_q50": preds.get("q0.5"),
        "pred_margin_q90": preds.get("q0.9"),
    })

    out_path = OUT / f"predictions_model_margins_{date_str}.csv"
    out_df.to_csv(out_path, index=False)

    payload = {
        "date": date_str,
        "rows": int(len(out_df)),
        "source_features": src,
        "predictions_path": str(out_path),
        "model": str(model_path),
    }
    (OUT / f"score_margins_{date_str}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Score margins for a date using trained quantile model")
    ap.add_argument("--date", type=str, required=True, help="Target date YYYY-MM-DD")

    default_model = os.getenv("NCAAB_MARGINS_QUANTILE_MODEL")
    if not default_model or default_model.strip() == "":
        default_model = str(OUT / "models" / "margins_roll_v1.joblib")

    ap.add_argument("--model", type=str, default=default_model, help="Model path")
    args = ap.parse_args()
    payload = score_date(args.date, Path(args.model))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
