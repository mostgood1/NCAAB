"""Quick diagnostic script to inspect the current training frame feature columns.

Run:
  python -m src.modeling.diagnose_training_frame
"""
from __future__ import annotations
from .data import build_training_frame
import pandas as pd
import json


def main() -> None:  # pragma: no cover
    X, y_total, y_margin = build_training_frame()
    summary = {
        "total_cols": len(X.columns),
        "team_cols_count": sum(c.startswith("home_team_") or c.startswith("away_team_") for c in X.columns),
        "diff_cols_count": sum(c.startswith("diff_") for c in X.columns),
        "first_30": X.columns[:30].tolist(),
        "team_samples": [c for c in X.columns if c.startswith("home_team_") or c.startswith("away_team_")][:15],
        "diff_samples": [c for c in X.columns if c.startswith("diff_")][:15],
        "probes": {},
        "rows": len(X),
        "y_total_non_null": int(y_total.notna().sum()),
        "y_margin_non_null": int(y_margin.notna().sum()),
        "unique_dates": [],
    }
    # Attempt to derive raw merged frame again for date listing (non-performant but fine for diagnostics)
    try:
        from .data import load_features, load_historical_games  # type: ignore
        feat = load_features(); games = load_historical_games();
        if not feat.empty and not games.empty:
            feat["game_id"] = feat.get("game_id").astype(str)
            games["game_id"] = games.get("game_id").astype(str)
            m = feat.merge(games, on="game_id", how="inner")
            if "date" in m.columns:
                dcol = m["date"]
            elif "date_x" in m.columns:
                dcol = m["date_x"]
            elif "date_y" in m.columns:
                dcol = m["date_y"]
            else:
                dcol = []
            summary["unique_dates"] = sorted(pd.to_datetime(dcol, errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    except Exception:
        pass
    for probe in [
        "home_team_season_off_ppg",
        "away_team_season_off_ppg",
        "diff_season_off_ppg",
        "home_team_rest_days",
        "diff_rest_days",
        "home_team_ewm_off_ppg",
        "diff_ewm_off_ppg",
    ]:
        if probe in X.columns:
            summary["probes"][probe] = int(X[probe].notna().sum())
        else:
            summary["probes"][probe] = None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
