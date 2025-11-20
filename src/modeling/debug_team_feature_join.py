"""Debug script to inspect overlap between training frame slugs/dates and team_features rows.

Run:
  python -m src.modeling.debug_team_feature_join
"""
from __future__ import annotations
import pandas as pd
from .data import load_features, load_historical_games, _load_team_features  # type: ignore
from .utils import canon_slug
import json


def main():  # pragma: no cover
    feat = load_features()
    games = load_historical_games()
    tfeat = _load_team_features()
    if feat.empty or games.empty:
        print(json.dumps({"status":"no_base_data"}, indent=2))
        return
    feat["game_id"] = feat.get("game_id").astype(str)
    games["game_id"] = games.get("game_id").astype(str)
    df = feat.merge(games, on="game_id", how="inner")
    # Handle potential duplicate date columns from merge
    if "date" not in df.columns:
        if "date_x" in df.columns:
            df["date"] = df["date_x"]
        elif "date_y" in df.columns:
            df["date"] = df["date_y"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # Handle potential duplicate team columns from merge (home_team_x/home_team_y etc.)
    home_team_col = next((c for c in df.columns if c.startswith("home_team")), None)
    away_team_col = next((c for c in df.columns if c.startswith("away_team")), None)
    if home_team_col is None or away_team_col is None:
        print(json.dumps({"status":"missing_team_columns","cols":df.columns.tolist()[:50]}, indent=2))
        return
    df["_home_slug"] = df[home_team_col].astype(str).map(canon_slug)
    df["_away_slug"] = df[away_team_col].astype(str).map(canon_slug)
    tfeat["date"] = tfeat["date"].astype(str)
    home_slug_set = set(df["_home_slug"].unique())
    away_slug_set = set(df["_away_slug"].unique())
    tfeat_slug_set = set(tfeat["team_slug"].unique())
    slug_intersection_home = len(home_slug_set & tfeat_slug_set)
    slug_intersection_away = len(away_slug_set & tfeat_slug_set)
    # Sample missing slugs
    missing_home = list(home_slug_set - tfeat_slug_set)[:15]
    missing_away = list(away_slug_set - tfeat_slug_set)[:15]
    # Date coverage: for a sample slug, count matching dates
    sample_slug = next(iter(home_slug_set & tfeat_slug_set), None)
    date_match_count = None
    if sample_slug:
        game_dates = set(df[df["_home_slug"] == sample_slug]["date"].unique()) | set(df[df["_away_slug"] == sample_slug]["date"].unique())
        feat_dates = set(tfeat[tfeat["team_slug"] == sample_slug]["date"].unique())
        date_match_count = len(game_dates & feat_dates)
    result = {
        "status": "ok",
        "total_games": int(len(df)),
        "unique_home_slugs": int(len(home_slug_set)),
        "unique_away_slugs": int(len(away_slug_set)),
        "unique_team_feature_slugs": int(len(tfeat_slug_set)),
        "home_slug_intersection": slug_intersection_home,
        "away_slug_intersection": slug_intersection_away,
        "sample_slug": sample_slug,
        "sample_slug_date_match_count": date_match_count,
        "missing_home_slug_samples": missing_home,
        "missing_away_slug_samples": missing_away,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
