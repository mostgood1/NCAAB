from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def _build_team_index(teams: pd.Series) -> Dict[str, int]:
    uniq = sorted(t for t in teams.dropna().astype(str).unique())
    return {t: i for i, t in enumerate(uniq)}


def compute_margin_ratings(games: pd.DataFrame, lambda_reg: float = 10.0) -> pd.DataFrame:
    """Compute simple ridge ratings from game margins using a graph Laplacian formulation.

    rating_home - rating_away ≈ (home_score - away_score)

    Returns a DataFrame with columns [team, rating_margin]. The mean rating is ~0.
    Unresolved games (scores missing or 0-0) are ignored.
    """
    # Only consider rows with resolved scores
    if not {"home_team", "away_team", "home_score", "away_score"}.issubset(games.columns):
        return pd.DataFrame(columns=["team", "rating_margin"])  # missing columns

    g = games.copy()
    # Guard against premature 0-0 scores
    mask_resolved = (g["home_score"].fillna(-1) > 0) | (g["away_score"].fillna(-1) > 0)
    g = g.loc[mask_resolved & ((g["home_score"] != 0) | (g["away_score"] != 0))]
    if g.empty:
        return pd.DataFrame(columns=["team", "rating_margin"])  # nothing to fit

    teams = pd.concat([g["home_team"].astype(str), g["away_team"].astype(str)], ignore_index=True)
    index = _build_team_index(teams)
    n = len(index)
    if n == 0:
        return pd.DataFrame(columns=["team", "rating_margin"])  # safety

    L = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    for _, row in g.iterrows():
        hi = index[row["home_team"]]
        ai = index[row["away_team"]]
        margin = float(row["home_score"]) - float(row["away_score"])  # home - away
        # Update Laplacian and RHS
        L[hi, hi] += 1.0
        L[ai, ai] += 1.0
        L[hi, ai] -= 1.0
        L[ai, hi] -= 1.0
        b[hi] += margin
        b[ai] -= margin

    # Regularize to ensure invertibility and shrink towards zero-mean
    A = L + lambda_reg * np.eye(n, dtype=np.float64)
    try:
        r = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least-squares
        r, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Center ratings to have mean ~ 0
    r = r - r.mean()

    inv_index = {i: t for t, i in index.items()}
    rows = [{"team": inv_index[i], "rating_margin": float(r[i])} for i in range(n)]
    return pd.DataFrame(rows)


def build_adj_rating_features(games: pd.DataFrame, lambda_reg: float = 10.0) -> pd.DataFrame:
    """Attach basic opponent-adjusted rating features derived from margins.

    Outputs per-game features with columns:
      game_id, home_rating_margin, away_rating_margin, rating_margin_diff
    """
    if "game_id" not in games.columns:
        return pd.DataFrame(columns=["game_id", "home_rating_margin", "away_rating_margin", "rating_margin_diff"])

    ratings = compute_margin_ratings(games, lambda_reg=lambda_reg)
    if ratings.empty:
        return pd.DataFrame(columns=["game_id", "home_rating_margin", "away_rating_margin", "rating_margin_diff"])

    # Merge ratings back to games
    g = games[["game_id", "home_team", "away_team"]].copy()
    g = g.merge(ratings.rename(columns={"team": "home_team", "rating_margin": "home_rating_margin"}), on="home_team", how="left")
    g = g.merge(ratings.rename(columns={"team": "away_team", "rating_margin": "away_rating_margin"}), on="away_team", how="left")
    g["rating_margin_diff"] = g["home_rating_margin"].fillna(0.0) - g["away_rating_margin"].fillna(0.0)
    return g[["game_id", "home_rating_margin", "away_rating_margin", "rating_margin_diff"]]


def compute_off_def_tempo_ratings(
    games: pd.DataFrame,
    boxscores: pd.DataFrame,
    lambda_reg: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ridge-style opponent-adjusted offensive, defensive, and tempo ratings.

    Off/Def model:
      y = [home_off_eff, away_off_eff] where off_eff = points / possessions * 100.
      For each game we write two equations:
        off_eff_home ≈ O_home - D_away
        off_eff_away ≈ O_away - D_home
      Unknowns x = [O (n), D (n)] for n teams.

    Tempo model:
      pace ≈ T_home + T_away, where pace is average possessions per team (or provided pace).

    Returns:
      ratings_od: DataFrame [team, off_rating, def_rating, net_rating]
      ratings_tempo: DataFrame [team, tempo_rating]
    """
    required_g_cols = {"game_id", "home_team", "away_team", "home_score", "away_score"}
    required_b_cols = {"game_id", "home_possessions", "away_possessions", "pace"}
    if not required_g_cols.issubset(games.columns) or not required_b_cols.issubset(boxscores.columns):
        return (
            pd.DataFrame(columns=["team", "off_rating", "def_rating", "net_rating"]),
            pd.DataFrame(columns=["team", "tempo_rating"]),
        )

    g = games.copy()
    b = boxscores[["game_id", "home_possessions", "away_possessions", "pace"]].copy()
    m = g.merge(b, on="game_id", how="left")
    # Filter resolved and valid possessions
    m = m[(m["home_score"].fillna(-1) > 0) | (m["away_score"].fillna(-1) > 0)]
    m = m[(m["home_score"] != 0) | (m["away_score"] != 0)]
    m = m[(m["home_possessions"].fillna(0) > 0) & (m["away_possessions"].fillna(0) > 0)]
    if m.empty:
        return (
            pd.DataFrame(columns=["team", "off_rating", "def_rating", "net_rating"]),
            pd.DataFrame(columns=["team", "tempo_rating"]),
        )

    # Observed efficiencies
    m["home_off_eff"] = m["home_score"].astype(float) / m["home_possessions"].astype(float) * 100.0
    m["away_off_eff"] = m["away_score"].astype(float) / m["away_possessions"].astype(float) * 100.0
    # Pace per game (possessions per team); use provided pace or mean of team possessions
    m["pace_use"] = m["pace"].where(m["pace"].notna(), (m["home_possessions"] + m["away_possessions"]) / 2.0)

    # Team index
    teams = pd.concat([m["home_team"].astype(str), m["away_team"].astype(str)], ignore_index=True)
    index = _build_team_index(teams)
    n = len(index)

    # Build Off/Def system: A x = y, x size 2n
    rows = []
    y = []
    for _, r in m.iterrows():
        hi = index[r["home_team"]]
        ai = index[r["away_team"]]
        # Home equation: +O_home, -D_away
        rows.append((hi, ai, r["home_off_eff"]))
        # Away equation: +O_away, -D_home
        rows.append((ai, hi, r["away_off_eff"]))
    m_eq = len(rows)
    # Build matrices explicitly (dense is fine for moderate team counts)
    A = np.zeros((m_eq, 2 * n), dtype=np.float64)
    y_vec = np.zeros(m_eq, dtype=np.float64)
    for i, (o_idx, d_idx, val) in enumerate(rows):
        A[i, o_idx] = 1.0
        A[i, n + d_idx] = -1.0
        y_vec[i] = float(val)
    # Ridge solve
    AtA = A.T @ A
    reg = lambda_reg * np.eye(2 * n, dtype=np.float64)
    rhs = A.T @ y_vec
    try:
        x = np.linalg.solve(AtA + reg, rhs)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(AtA + reg, rhs, rcond=None)
    O = x[:n]
    D = x[n:]
    # Center O and D to mean zero for stability; net reflects difference
    O = O - O.mean()
    D = D - D.mean()

    inv_index = {i: t for t, i in index.items()}
    ratings_od = pd.DataFrame({
        "team": [inv_index[i] for i in range(n)],
        "off_rating": O.astype(float),
        "def_rating": D.astype(float),
    })
    ratings_od["net_rating"] = ratings_od["off_rating"] - ratings_od["def_rating"]

    # Tempo system: pace ≈ T_home + T_away
    m_t = m.dropna(subset=["pace_use"])  # ensure pace present
    if m_t.empty:
        ratings_tempo = pd.DataFrame({"team": ratings_od["team"], "tempo_rating": np.zeros(n, dtype=float)})
    else:
        mt = len(m_t)
        A_t = np.zeros((mt, n), dtype=np.float64)
        y_t = np.zeros(mt, dtype=np.float64)
        for i, row in enumerate(m_t.itertuples(index=False)):
            hi = index[getattr(row, "home_team")]
            ai = index[getattr(row, "away_team")]
            A_t[i, hi] = 1.0
            A_t[i, ai] = 1.0
            y_t[i] = float(getattr(row, "pace_use"))
        AtA_t = A_t.T @ A_t
        rhs_t = A_t.T @ y_t
        try:
            t = np.linalg.solve(AtA_t + lambda_reg * np.eye(n, dtype=np.float64), rhs_t)
        except np.linalg.LinAlgError:
            t, *_ = np.linalg.lstsq(AtA_t + lambda_reg * np.eye(n, dtype=np.float64), rhs_t, rcond=None)
        t = t - t.mean()
        ratings_tempo = pd.DataFrame({
            "team": [inv_index[i] for i in range(n)],
            "tempo_rating": t.astype(float),
        })

    return ratings_od, ratings_tempo


def build_adj_offdef_tempo_features(
    games: pd.DataFrame,
    boxscores: pd.DataFrame,
    lambda_reg: float = 10.0,
) -> pd.DataFrame:
    """Build per-game features from opponent-adjusted Off/Def/Tempo ratings.

    Output columns:
      game_id,
      home_off_rating, away_off_rating, off_rating_diff,
      home_def_rating, away_def_rating, def_rating_diff,
      home_tempo_rating, away_tempo_rating, tempo_rating_sum
    """
    if "game_id" not in games.columns:
        return pd.DataFrame(columns=[
            "game_id",
            "home_off_rating", "away_off_rating", "off_rating_diff",
            "home_def_rating", "away_def_rating", "def_rating_diff",
            "home_tempo_rating", "away_tempo_rating", "tempo_rating_sum",
        ])

    ratings_od, ratings_tempo = compute_off_def_tempo_ratings(games, boxscores, lambda_reg=lambda_reg)
    if ratings_od.empty:
        return pd.DataFrame(columns=[
            "game_id",
            "home_off_rating", "away_off_rating", "off_rating_diff",
            "home_def_rating", "away_def_rating", "def_rating_diff",
            "home_tempo_rating", "away_tempo_rating", "tempo_rating_sum",
        ])

    g = games[["game_id", "home_team", "away_team"]].copy()
    g = g.merge(ratings_od.rename(columns={"team": "home_team"}), on="home_team", how="left")
    g = g.rename(columns={
        "off_rating": "home_off_rating",
        "def_rating": "home_def_rating",
        "net_rating": "home_net_rating",
    })
    g = g.merge(ratings_od.rename(columns={"team": "away_team"}), on="away_team", how="left")
    g = g.rename(columns={
        "off_rating": "away_off_rating",
        "def_rating": "away_def_rating",
        "net_rating": "away_net_rating",
    })
    g = g.merge(ratings_tempo.rename(columns={"team": "home_team", "tempo_rating": "home_tempo_rating"}), on="home_team", how="left")
    g = g.merge(ratings_tempo.rename(columns={"team": "away_team", "tempo_rating": "away_tempo_rating"}), on="away_team", how="left")

    g["off_rating_diff"] = g["home_off_rating"].fillna(0.0) - g["away_off_rating"].fillna(0.0)
    g["def_rating_diff"] = g["home_def_rating"].fillna(0.0) - g["away_def_rating"].fillna(0.0)
    g["tempo_rating_sum"] = g["home_tempo_rating"].fillna(0.0) + g["away_tempo_rating"].fillna(0.0)

    return g[[
        "game_id",
        "home_off_rating", "away_off_rating", "off_rating_diff",
        "home_def_rating", "away_def_rating", "def_rating_diff",
        "home_tempo_rating", "away_tempo_rating", "tempo_rating_sum",
    ]]
