from __future__ import annotations

import pandas as pd
from typing import Optional


def build_four_factor_rolling_features(
    games: pd.DataFrame,
    boxscores: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Compute team-level rolling four-factor features from ESPN boxscores.

    Inputs:
    - games: DataFrame with columns [game_id, date, home_team, away_team]
    - boxscores: DataFrame produced by fetch-boxscores with columns including
      [game_id, home_possessions, away_possessions, home_efg, home_tov_rate, home_orb_rate, home_ftr,
       away_efg, away_tov_rate, away_orb_rate, away_ftr]

    Returns a DataFrame keyed by game_id with columns:
      home_efg{window}, home_tov_rate{window}, home_orb_rate{window}, home_ftr{window}, home_poss{window}
      away_efg{window}, away_tov_rate{window}, away_orb_rate{window}, away_ftr{window}, away_poss{window}
    """
    if boxscores is None or boxscores.empty:
        return pd.DataFrame({"game_id": []})

    g = games[["game_id", "date", "home_team", "away_team"]].copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")

    bs = boxscores.copy()
    # Drop potentially conflicting columns from boxscores to avoid suffixing
    bs = bs.drop(columns=[c for c in ["date", "home_team", "away_team"] if c in bs.columns], errors="ignore")
    # Join to ensure team names are aligned from games
    merged = g.merge(bs, on="game_id", how="inner")

    # Build long team-game rows with per-team metrics
    home_long = merged[[
        "game_id", "date", "home_team",
        "home_efg", "home_tov_rate", "home_orb_rate", "home_ftr", "home_possessions"
    ]].rename(columns={
        "home_team": "team",
        "home_efg": "efg",
        "home_tov_rate": "tov_rate",
        "home_orb_rate": "orb_rate",
        "home_ftr": "ftr",
        "home_possessions": "poss"
    })

    away_long = merged[[
        "game_id", "date", "away_team",
        "away_efg", "away_tov_rate", "away_orb_rate", "away_ftr", "away_possessions"
    ]].rename(columns={
        "away_team": "team",
        "away_efg": "efg",
        "away_tov_rate": "tov_rate",
        "away_orb_rate": "orb_rate",
        "away_ftr": "ftr",
        "away_possessions": "poss"
    })

    long = pd.concat([home_long, away_long], ignore_index=True)
    long = long.sort_values(["team", "date"])  # chronological per team

    # Compute rolling means excluding current game (shift by 1)
    for col in ["efg", "tov_rate", "orb_rate", "ftr", "poss"]:
        long[f"{col}_roll"] = long.groupby("team")[col].transform(lambda s: s.shift(1).rolling(window).mean())

    # Map back to game-level: collect home and away features
    home_feat = long.merge(g[["game_id", "home_team"]], left_on=["game_id", "team"], right_on=["game_id", "home_team"], how="inner")
    away_feat = long.merge(g[["game_id", "away_team"]], left_on=["game_id", "team"], right_on=["game_id", "away_team"], how="inner")

    out = g[["game_id"]].copy()
    out = out.merge(
        home_feat[["game_id", "efg_roll", "tov_rate_roll", "orb_rate_roll", "ftr_roll", "poss_roll"]]
        .rename(columns={
            "efg_roll": f"home_efg{window}",
            "tov_rate_roll": f"home_tov_rate{window}",
            "orb_rate_roll": f"home_orb_rate{window}",
            "ftr_roll": f"home_ftr{window}",
            "poss_roll": f"home_poss{window}",
        }),
        on="game_id",
        how="left",
    )
    out = out.merge(
        away_feat[["game_id", "efg_roll", "tov_rate_roll", "orb_rate_roll", "ftr_roll", "poss_roll"]]
        .rename(columns={
            "efg_roll": f"away_efg{window}",
            "tov_rate_roll": f"away_tov_rate{window}",
            "orb_rate_roll": f"away_orb_rate{window}",
            "ftr_roll": f"away_ftr{window}",
            "poss_roll": f"away_poss{window}",
        }),
        on="game_id",
        how="left",
    )

    return out
