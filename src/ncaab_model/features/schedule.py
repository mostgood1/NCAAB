from __future__ import annotations

import pandas as pd
import numpy as np


def compute_rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days for home and away teams based on prior games.

    Expects columns: game_id, date, home_team, away_team
    Returns a DataFrame with: game_id, rest_home, rest_away, b2b_home, b2b_away, neutral_site (if present)
    """
    df = games.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # build long format (team, date)
    home = df[["game_id", "date", "home_team"]].rename(columns={"home_team": "team"})
    away = df[["game_id", "date", "away_team"]].rename(columns={"away_team": "team"})
    long = pd.concat([home, away], ignore_index=True)

    # sort and compute previous date per team
    long = long.sort_values(["team", "date"]) \
             .assign(prev_date=lambda x: x.groupby("team")["date"].shift(1))

    # rest in days per appearance
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days

    # pivot back to game-level
    rest_home = long.merge(home[["game_id", "team"]], on=["game_id", "team"], how="inner")[
        ["game_id", "rest_days"]
    ].rename(columns={"rest_days": "rest_home"})

    rest_away = long.merge(away[["game_id", "team"]], on=["game_id", "team"], how="inner")[
        ["game_id", "rest_days"]
    ].rename(columns={"rest_days": "rest_away"})

    out = df[["game_id"]].merge(rest_home, on="game_id", how="left").merge(rest_away, on="game_id", how="left")

    # back-to-back flags (<= 1 day rest)
    out["b2b_home"] = out["rest_home"].apply(lambda x: bool(x is not None and x <= 1))
    out["b2b_away"] = out["rest_away"].apply(lambda x: bool(x is not None and x <= 1))

    # carry neutral_site if present
    if "neutral_site" in df.columns:
        out = out.merge(df[["game_id", "neutral_site"]], on="game_id", how="left")

    return out
