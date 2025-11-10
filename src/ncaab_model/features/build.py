from __future__ import annotations

import pandas as pd
from typing import Iterable


def build_team_rolling_features(games: pd.DataFrame, windows: int | Iterable[int] = 5, add_volatility: bool = True) -> pd.DataFrame:
    """Compute per-team rolling mean (and optional std) stats prior to each game.

    Rolling mean columns per window w:
      home_pf{w}, home_pa{w}, home_tot{w}, away_pf{w}, away_pa{w}, away_tot{w}
    When add_volatility=True, also:
      home_pf_std{w}, home_pa_std{w}, home_tot_std{w}, away_pf_std{w}, away_pa_std{w}, away_tot_std{w}
    Additional interaction enrichments appended:
      pf_vs_opp_pa_diff{w}, away_pf_vs_home_pa_diff{w}, home_net_scoring{w}, away_net_scoring{w}
    """
    df = games.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total"] = df[["home_score", "away_score"]].sum(axis=1, min_count=2)
    if {"home_score_1h", "away_score_1h"}.issubset(df.columns):
        df["total_1h"] = df[["home_score_1h", "away_score_1h"]].sum(axis=1, min_count=2)

    # Normalize windows list
    if isinstance(windows, int):
        win_list = [windows]
    else:
        win_list = sorted({int(w) for w in windows if int(w) > 0}) or [5]

    # Long format team rows
    base_cols = ["game_id", "date", "total"]
    home = df[base_cols + ["home_team", "home_score", "away_score", "total_1h"]].rename(
        columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against"}
    )
    away = df[base_cols + ["away_team", "away_score", "home_score", "total_1h"]].rename(
        columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against"}
    )
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)

    roll_frames: dict[int, pd.DataFrame] = {}
    for w in win_list:
        tmp = long.copy()
        grp_pf = tmp.groupby("team")["points_for"]
        grp_pa = tmp.groupby("team")["points_against"]
        grp_tot = tmp.groupby("team")["total"]
        tmp[f"pf_roll_{w}"] = grp_pf.transform(lambda s: s.shift(1).rolling(w).mean())
        tmp[f"pa_roll_{w}"] = grp_pa.transform(lambda s: s.shift(1).rolling(w).mean())
        tmp[f"tot_roll_{w}"] = grp_tot.transform(lambda s: s.shift(1).rolling(w).mean())
        if add_volatility:
            tmp[f"pf_std_roll_{w}"] = grp_pf.transform(lambda s: s.shift(1).rolling(w).std())
            tmp[f"pa_std_roll_{w}"] = grp_pa.transform(lambda s: s.shift(1).rolling(w).std())
            tmp[f"tot_std_roll_{w}"] = grp_tot.transform(lambda s: s.shift(1).rolling(w).std())
        cols = ["game_id", "team", f"pf_roll_{w}", f"pa_roll_{w}", f"tot_roll_{w}"]
        if add_volatility:
            cols += [f"pf_std_roll_{w}", f"pa_std_roll_{w}", f"tot_std_roll_{w}"]
        roll_frames[w] = tmp[cols]

    features = df[["game_id", "date", "home_team", "away_team", "home_score", "away_score", "total"]].copy()
    if "total_1h" in df.columns:
        features["total_1h"] = df["total_1h"]

    for w in win_list:
        tmp = roll_frames[w]
        home_feat = tmp.merge(df[["game_id", "home_team"]], left_on=["game_id", "team"], right_on=["game_id", "home_team"], how="inner")
        away_feat = tmp.merge(df[["game_id", "away_team"]], left_on=["game_id", "team"], right_on=["game_id", "away_team"], how="inner")
        # Means
        features = features.merge(
            home_feat[["game_id", f"pf_roll_{w}", f"pa_roll_{w}", f"tot_roll_{w}"]].rename(
                columns={f"pf_roll_{w}": f"home_pf{w}", f"pa_roll_{w}": f"home_pa{w}", f"tot_roll_{w}": f"home_tot{w}"}
            ),
            on="game_id",
            how="left",
        )
        features = features.merge(
            away_feat[["game_id", f"pf_roll_{w}", f"pa_roll_{w}", f"tot_roll_{w}"]].rename(
                columns={f"pf_roll_{w}": f"away_pf{w}", f"pa_roll_{w}": f"away_pa{w}", f"tot_roll_{w}": f"away_tot{w}"}
            ),
            on="game_id",
            how="left",
        )
        if add_volatility:
            features = features.merge(
                home_feat[["game_id", f"pf_std_roll_{w}", f"pa_std_roll_{w}", f"tot_std_roll_{w}"]].rename(
                    columns={f"pf_std_roll_{w}": f"home_pf_std{w}", f"pa_std_roll_{w}": f"home_pa_std{w}", f"tot_std_roll_{w}": f"home_tot_std{w}"}
                ),
                on="game_id",
                how="left",
            )
            features = features.merge(
                away_feat[["game_id", f"pf_std_roll_{w}", f"pa_std_roll_{w}", f"tot_std_roll_{w}"]].rename(
                    columns={f"pf_std_roll_{w}": f"away_pf_std{w}", f"pa_std_roll_{w}": f"away_pa_std{w}", f"tot_std_roll_{w}": f"away_tot_std{w}"}
                ),
                on="game_id",
                how="left",
            )

    # Targets
    features["target_total"] = features["total"]
    features["target_margin"] = features["home_score"] - features["away_score"]
    if "total_1h" in features.columns:
        features["target_1h_total"] = features["total_1h"]

    # Remove leakage columns
    features = features.drop(columns=["home_score", "away_score", "total", "total_1h"], errors="ignore")

    # Interaction enrichments
    for w in win_list:
        hp, ap = f"home_pf{w}", f"away_pf{w}"
        hd, ad = f"home_pa{w}", f"away_pa{w}"
        if hp in features.columns and ad in features.columns:
            features[f"pf_vs_opp_pa_diff{w}"] = features[hp] - features[ad]
        if ap in features.columns and hd in features.columns:
            features[f"away_pf_vs_home_pa_diff{w}"] = features[ap] - features[hd]
        if hp in features.columns and hd in features.columns:
            features[f"home_net_scoring{w}"] = features[hp] - features[hd]
        if ap in features.columns and ad in features.columns:
            features[f"away_net_scoring{w}"] = features[ap] - features[ad]

    return features
