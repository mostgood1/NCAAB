from __future__ import annotations

import pandas as pd


def enrich_game_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Add secondary derived features that combine existing base feature columns.

    This layer is intentionally lightweight and only uses already-materialized columns so it can
    be applied after merges (schedule, ratings, rolling, boxscores). All operations are pure and
    will not raise if columns are missing; absent dependencies simply skip that feature.

    Added columns (when source columns present):
      - rest_diff: rest_home - rest_away
      - home_rest_adv: max(rest_home - rest_away, 0)
      - b2b_diff: int(b2b_home) - int(b2b_away)
      - tempo_rating_diff: home_tempo_rating - away_tempo_rating
      - off_def_combined_adv: (home_off_rating - away_def_rating) - (away_off_rating - home_def_rating)
      - pace_off_synergy: tempo_rating_sum * (off_rating_diff)
    """
    out = feats.copy()
    if {"rest_home", "rest_away"}.issubset(out.columns):
        out["rest_diff"] = out["rest_home"].fillna(0) - out["rest_away"].fillna(0)
        out["home_rest_adv"] = (out["rest_diff"].clip(lower=0)).astype(float)
    if {"b2b_home", "b2b_away"}.issubset(out.columns):
        out["b2b_diff"] = out["b2b_home"].astype(int) - out["b2b_away"].astype(int)
    if {"home_tempo_rating", "away_tempo_rating"}.issubset(out.columns):
        out["tempo_rating_diff"] = out["home_tempo_rating"].fillna(0) - out["away_tempo_rating"].fillna(0)
    if {"home_off_rating", "away_off_rating", "home_def_rating", "away_def_rating"}.issubset(out.columns):
        home_adv = out["home_off_rating"].fillna(0) - out["away_def_rating"].fillna(0)
        away_adv = out["away_off_rating"].fillna(0) - out["home_def_rating"].fillna(0)
        out["off_def_combined_adv"] = home_adv - away_adv
    if {"tempo_rating_sum", "off_rating_diff"}.issubset(out.columns):
        out["pace_off_synergy"] = out["tempo_rating_sum"].fillna(0) * out["off_rating_diff"].fillna(0)
    return out
