from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_totals import MODELS, OUT, TotalsModel


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


def _read_results_rows(daily_dir: Path, end_date: str | None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for p in sorted(daily_dir.glob("results_*.csv")):
        date_str = p.stem.replace("results_", "")

        if end_date:
            try:
                d = pd.to_datetime(date_str, errors="coerce")
                cutoff = pd.to_datetime(end_date, errors="coerce")
                if pd.notna(d) and pd.notna(cutoff) and d > cutoff:
                    continue
            except Exception:
                pass

        df = _safe_read_csv(p)
        if df.empty:
            continue

        try:
            if "completed" in df.columns:
                comp = pd.to_numeric(df["completed"], errors="coerce").fillna(0).astype(int)
                df = df[comp == 1]
        except Exception:
            pass

        need = {"game_id", "home_team", "away_team", "home_score", "away_score"}
        if not need.issubset(df.columns):
            continue

        out = df[["game_id", "home_team", "away_team", "home_score", "away_score"]].copy()
        out["date"] = date_str
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    all_games = pd.concat(rows, ignore_index=True)
    all_games["game_id"] = all_games["game_id"].astype(str)
    all_games["home_team"] = all_games["home_team"].astype(str)
    all_games["away_team"] = all_games["away_team"].astype(str)
    all_games["home_score"] = pd.to_numeric(all_games["home_score"], errors="coerce")
    all_games["away_score"] = pd.to_numeric(all_games["away_score"], errors="coerce")
    all_games["actual_margin"] = all_games["home_score"] - all_games["away_score"]
    all_games = all_games[pd.to_numeric(all_games["actual_margin"], errors="coerce").notna()].copy()

    try:
        all_games["date"] = pd.to_datetime(all_games["date"], errors="coerce")
    except Exception:
        all_games["date"] = pd.NaT
    all_games = all_games.dropna(subset=["date"])
    all_games["date"] = all_games["date"].dt.strftime("%Y-%m-%d")

    all_games = all_games.sort_values(["date", "game_id"], kind="mergesort")
    all_games = all_games.drop_duplicates(subset=["game_id", "date"], keep="last").reset_index(drop=True)
    return all_games


def _build_rolling_features(games: pd.DataFrame) -> pd.DataFrame:
    tl_rows: list[dict[str, Any]] = []
    for r in games.itertuples(index=False):
        try:
            gid = str(getattr(r, "game_id"))
            date_str = str(getattr(r, "date"))
            ht = str(getattr(r, "home_team"))
            at = str(getattr(r, "away_team"))
            hs = float(getattr(r, "home_score"))
            aw = float(getattr(r, "away_score"))
            tot = float(hs + aw)
        except Exception:
            continue
        tl_rows.append({"game_id": gid, "date": date_str, "team": ht, "pf": hs, "pa": aw, "tot": tot})
        tl_rows.append({"game_id": gid, "date": date_str, "team": at, "pf": aw, "pa": hs, "tot": tot})

    tl = pd.DataFrame(tl_rows)
    if tl.empty:
        return pd.DataFrame()

    tl["date"] = pd.to_datetime(tl["date"], errors="coerce")
    tl = tl.dropna(subset=["date"])
    tl = tl.sort_values(["team", "date", "game_id"], kind="mergesort").reset_index(drop=True)

    tl["gp"] = tl.groupby("team").cumcount()

    for w in (5, 15):
        for col in ("pf", "pa", "tot"):
            out_col = f"{col}{w}"
            tl[out_col] = tl.groupby("team")[col].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))

    tl["date"] = tl["date"].dt.strftime("%Y-%m-%d")

    base = games[["game_id", "date", "home_team", "away_team", "actual_margin"]].copy()
    base["game_id"] = base["game_id"].astype(str)
    base["home_team"] = base["home_team"].astype(str)
    base["away_team"] = base["away_team"].astype(str)

    feat_cols = ["gp", "pf5", "pa5", "tot5", "pf15", "pa15", "tot15"]

    home = base[["game_id", "home_team"]].rename(columns={"home_team": "team"}).merge(
        tl[["game_id", "team", *feat_cols]], on=["game_id", "team"], how="left"
    )
    home = home.rename(
        columns={
            "gp": "home_gp",
            "pf5": "home_pf5",
            "pa5": "home_pa5",
            "tot5": "home_tot5",
            "pf15": "home_pf15",
            "pa15": "home_pa15",
            "tot15": "home_tot15",
        }
    ).drop(columns=["team"])

    away = base[["game_id", "away_team"]].rename(columns={"away_team": "team"}).merge(
        tl[["game_id", "team", *feat_cols]], on=["game_id", "team"], how="left"
    )
    away = away.rename(
        columns={
            "gp": "away_gp",
            "pf5": "away_pf5",
            "pa5": "away_pa5",
            "tot5": "away_tot5",
            "pf15": "away_pf15",
            "pa15": "away_pa15",
            "tot15": "away_tot15",
        }
    ).drop(columns=["team"])

    out = base.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")

    # Derived margin-relevant features
    out["home_net5"] = pd.to_numeric(out["home_pf5"], errors="coerce") - pd.to_numeric(out["home_pa5"], errors="coerce")
    out["away_net5"] = pd.to_numeric(out["away_pf5"], errors="coerce") - pd.to_numeric(out["away_pa5"], errors="coerce")
    out["diff_net5"] = out["home_net5"] - out["away_net5"]

    out["home_net15"] = pd.to_numeric(out["home_pf15"], errors="coerce") - pd.to_numeric(out["home_pa15"], errors="coerce")
    out["away_net15"] = pd.to_numeric(out["away_pf15"], errors="coerce") - pd.to_numeric(out["away_pa15"], errors="coerce")
    out["diff_net15"] = out["home_net15"] - out["away_net15"]

    feature_names = [
        "home_gp",
        "away_gp",
        "home_pf5",
        "home_pa5",
        "home_tot5",
        "away_pf5",
        "away_pa5",
        "away_tot5",
        "home_pf15",
        "home_pa15",
        "home_tot15",
        "away_pf15",
        "away_pa15",
        "away_tot15",
        "home_net5",
        "away_net5",
        "diff_net5",
        "home_net15",
        "away_net15",
        "diff_net15",
    ]

    for c in feature_names:
        out[c] = pd.to_numeric(out.get(c), errors="coerce")
        if out[c].isna().any():
            mu = float(out[c].mean()) if out[c].notna().any() else 0.0
            out[c] = out[c].fillna(mu)

    return out


def train(model_name: str = "margins_roll_v1", end_date: str | None = None) -> dict[str, Any]:
    daily_dir = OUT / "daily_results"
    games = _read_results_rows(daily_dir, end_date=end_date)
    if games.empty:
        return {"error": "No finalized results found", "daily_dir": str(daily_dir), "end_date": end_date}

    feats = _build_rolling_features(games)
    if feats.empty:
        return {"error": "Failed to build rolling features", "games": int(len(games))}

    feature_cols = [c for c in feats.columns if c not in ("actual_margin", "game_id", "date", "home_team", "away_team")]
    feature_cols = [c for c in feature_cols if c in feats.columns]
    X = feats[feature_cols].copy()
    y = pd.to_numeric(feats["actual_margin"], errors="coerce")

    model = TotalsModel(model_name)
    metrics = model.fit(X, y)

    path = MODELS / f"{model_name}.joblib"
    model.save(path)

    out = {
        "model": model_name,
        "train_rows": int(len(feats)),
        "feature_cols": model.feature_cols,
        "metrics": metrics,
        "model_path": str(path),
    }
    (OUT / f"train_margins_{model_name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a margin quantile model from rolling PF/PA/TOT features derived from daily_results")
    ap.add_argument("--name", type=str, default="margins_roll_v1", help="Model name")
    ap.add_argument("--end-date", type=str, default=None, help="Optional cutoff date YYYY-MM-DD (inclusive) for training rows")
    args = ap.parse_args()
    payload = train(model_name=args.name, end_date=args.end_date)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
