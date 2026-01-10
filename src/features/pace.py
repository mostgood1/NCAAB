from __future__ import annotations
import pandas as pd
import numpy as np


def estimate_possessions(fga: float | int | None,
                         fta: float | int | None,
                         orb: float | int | None,
                         tov: float | int | None) -> float | None:
    """Dean Oliver possession estimate for a single side.

    Formula: Poss ≈ FGA + 0.475*FTA - OR + TOV
    Returns None if all inputs are None/NaN.
    """
    vals = [fga, fta, orb, tov]
    if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
        return None
    fga_v = float(fga) if fga is not None and not pd.isna(fga) else 0.0
    fta_v = float(fta) if fta is not None and not pd.isna(fta) else 0.0
    orb_v = float(orb) if orb is not None and not pd.isna(orb) else 0.0
    tov_v = float(tov) if tov is not None and not pd.isna(tov) else 0.0
    return fga_v + 0.475 * fta_v - orb_v + tov_v


def estimate_team_possessions(df: pd.DataFrame,
                              home_prefix: str = "home_",
                              away_prefix: str = "away_",
                              col_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Estimate possessions for home/away and add columns to df.

    Attempts to find columns for FGA/FTA/OR/TOV using given prefixes.
    `col_map` can override default names per prefix, e.g.:
      {"fga": "field_goals_attempted", "fta": "free_throws_attempted",
       "orb": "offensive_rebounds", "tov": "turnovers"}
    """
    df = df.copy()
    # Default field names
    defaults = {"fga": "fga", "fta": "fta", "orb": "or", "tov": "to"}
    if col_map:
        defaults.update({k: v for k, v in col_map.items() if k in defaults})

    def col(pfx: str, key: str) -> str:
        base = defaults[key]
        # Try exact, then a few common alternates
        candidates = [f"{pfx}{base}", f"{pfx}{base}_p", f"{pfx}{base}_g"]
        # Also accept explicit provided map values without prefix
        if col_map and key in col_map:
            candidates.insert(0, f"{pfx}{col_map[key]}")
            candidates.insert(0, col_map[key])
        for c in candidates:
            if c in df.columns:
                return c
        return candidates[0]

    h_fga = pd.to_numeric(df.get(col(home_prefix, "fga")), errors="coerce")
    h_fta = pd.to_numeric(df.get(col(home_prefix, "fta")), errors="coerce")
    h_orb = pd.to_numeric(df.get(col(home_prefix, "orb")), errors="coerce")
    h_tov = pd.to_numeric(df.get(col(home_prefix, "tov")), errors="coerce")

    a_fga = pd.to_numeric(df.get(col(away_prefix, "fga")), errors="coerce")
    a_fta = pd.to_numeric(df.get(col(away_prefix, "fta")), errors="coerce")
    a_orb = pd.to_numeric(df.get(col(away_prefix, "orb")), errors="coerce")
    a_tov = pd.to_numeric(df.get(col(away_prefix, "tov")), errors="coerce")

    df["possessions_home_est"] = h_fga + 0.475 * h_fta - h_orb + h_tov
    df["possessions_away_est"] = a_fga + 0.475 * a_fta - a_orb + a_tov
    return df


def estimate_game_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate game possessions by averaging team estimates.

    Poss_game ≈ 0.5 * (poss_home + poss_away)
    """
    df = df.copy()
    ph = pd.to_numeric(df.get("possessions_home_est"), errors="coerce")
    pa = pd.to_numeric(df.get("possessions_away_est"), errors="coerce")
    df["possessions_game_est"] = 0.5 * (ph + pa)
    return df


def pace_per_minutes(possessions: float | int | None,
                     minutes_played: float | int | None = 40.0) -> float | None:
    """Compute pace scaled to a minute baseline (default 40 for NCAAB).

    Pace(40) = possessions * (40 / minutes_played)
    If minutes are missing, assume regulation 40.
    """
    if possessions is None or pd.isna(possessions):
        return None
    mins = float(minutes_played) if minutes_played is not None and not pd.isna(minutes_played) else 40.0
    mins = max(mins, 1e-6)
    return float(possessions) * (40.0 / mins)


def attach_pace_features(df: pd.DataFrame,
                         minutes_col: str | None = None) -> pd.DataFrame:
    """Convenience to produce `possessions_game_est` and `pace_game_est`.

    If `minutes_col` is given and exists, uses it for pace scaling; otherwise 40.
    """
    df = df.copy()
    if "possessions_home_est" not in df.columns or "possessions_away_est" not in df.columns:
        df = estimate_team_possessions(df)
    df = estimate_game_possessions(df)
    mins_series = pd.to_numeric(df.get(minutes_col), errors="coerce") if minutes_col and minutes_col in df.columns else None
    if mins_series is not None:
        df["pace_game_est"] = df["possessions_game_est"].astype(float) * (40.0 / mins_series.replace(0, np.nan).fillna(40.0))
    else:
        df["pace_game_est"] = df["possessions_game_est"].astype(float)
    return df
