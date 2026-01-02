from __future__ import annotations

import pandas as pd


def build_shooting_rolling_features(
    games: pd.DataFrame,
    boxscores: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Compute rolling shooting and paint metrics per team from boxscores.

    Inputs:
      - games: DataFrame with [game_id, date, home_team, away_team]
      - boxscores: DataFrame with per-game team metrics, including:
        home_3pt_rate, home_3pt_pct, home_2pt_rate, home_2pt_pct, home_pip, home_fbp, home_scp
        and away_* counterparts.

    Returns per-game features keyed by game_id:
      home_3pt_rate{w}, home_3pt_pct{w}, home_2pt_rate{w}, home_2pt_pct{w}, home_pip{w}, home_fbp{w}, home_scp{w}
      away_3pt_rate{w}, away_3pt_pct{w}, away_2pt_rate{w}, away_2pt_pct{w}, away_pip{w}, away_fbp{w}, away_scp{w}
    """
    if boxscores is None or boxscores.empty:
        return pd.DataFrame({"game_id": []})

    g = games[["game_id", "date", "home_team", "away_team"]].copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")

    bs = boxscores.copy()
    bs = bs.drop(columns=[c for c in ["date", "home_team", "away_team"] if c in bs.columns], errors="ignore")
    m = g.merge(bs, on="game_id", how="inner")

    # Long format per team with shooting metrics
    home_cols = [
        "home_3pt_rate", "home_3pt_pct", "home_2pt_rate", "home_2pt_pct", "home_pip", "home_fbp", "home_scp"
    ]
    away_cols = [
        "away_3pt_rate", "away_3pt_pct", "away_2pt_rate", "away_2pt_pct", "away_pip", "away_fbp", "away_scp"
    ]

    home_long = m[["game_id", "date", "home_team"] + home_cols].rename(columns={
        "home_team": "team",
        "home_3pt_rate": "_3pt_rate",
        "home_3pt_pct": "_3pt_pct",
        "home_2pt_rate": "_2pt_rate",
        "home_2pt_pct": "_2pt_pct",
        "home_pip": "_pip",
        "home_fbp": "_fbp",
        "home_scp": "_scp",
    })
    away_long = m[["game_id", "date", "away_team"] + away_cols].rename(columns={
        "away_team": "team",
        "away_3pt_rate": "_3pt_rate",
        "away_3pt_pct": "_3pt_pct",
        "away_2pt_rate": "_2pt_rate",
        "away_2pt_pct": "_2pt_pct",
        "away_pip": "_pip",
        "away_fbp": "_fbp",
        "away_scp": "_scp",
    })

    long = pd.concat([home_long, away_long], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    # Rolling means excluding current game
    for col in ["_3pt_rate", "_3pt_pct", "_2pt_rate", "_2pt_pct", "_pip", "_fbp", "_scp"]:
        long[f"{col}_roll"] = long.groupby("team")[col].transform(lambda s: s.shift(1).rolling(window).mean())

    # Map back to game-level home/away features
    home_feat = long.merge(g[["game_id", "home_team"]], left_on=["game_id", "team"], right_on=["game_id", "home_team"], how="inner")
    away_feat = long.merge(g[["game_id", "away_team"]], left_on=["game_id", "team"], right_on=["game_id", "away_team"], how="inner")

    out = g[["game_id"]].copy()
    # Home
    out = out.merge(
        home_feat[["game_id", "_3pt_rate_roll", "_3pt_pct_roll", "_2pt_rate_roll", "_2pt_pct_roll", "_pip_roll", "_fbp_roll", "_scp_roll"]]
        .rename(columns={
            "_3pt_rate_roll": f"home_3pt_rate{window}",
            "_3pt_pct_roll": f"home_3pt_pct{window}",
            "_2pt_rate_roll": f"home_2pt_rate{window}",
            "_2pt_pct_roll": f"home_2pt_pct{window}",
            "_pip_roll": f"home_pip{window}",
            "_fbp_roll": f"home_fbp{window}",
            "_scp_roll": f"home_scp{window}",
        }),
        on="game_id",
        how="left",
    )
    # Away
    out = out.merge(
        away_feat[["game_id", "_3pt_rate_roll", "_3pt_pct_roll", "_2pt_rate_roll", "_2pt_pct_roll", "_pip_roll", "_fbp_roll", "_scp_roll"]]
        .rename(columns={
            "_3pt_rate_roll": f"away_3pt_rate{window}",
            "_3pt_pct_roll": f"away_3pt_pct{window}",
            "_2pt_rate_roll": f"away_2pt_rate{window}",
            "_2pt_pct_roll": f"away_2pt_pct{window}",
            "_pip_roll": f"away_pip{window}",
            "_fbp_roll": f"away_fbp{window}",
            "_scp_roll": f"away_scp{window}",
        }),
        on="game_id",
        how="left",
    )

    return out
