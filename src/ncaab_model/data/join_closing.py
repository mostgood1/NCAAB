from __future__ import annotations

import pandas as pd
from .team_normalize import canonical_slug as normalize_name
from typing import Dict, Optional
import pathlib

def _load_mapping(mapping_csv: str | pathlib.Path | None) -> Dict[str, str]:
    """Load a two-column mapping CSV with columns: odds_slug,suggested_game_slug OR raw,canonical.

    Returns dict mapping source slug -> target slug (both canonical_slug already or will be canonicalized here).
    """
    if mapping_csv is None:
        return {}
    p = pathlib.Path(mapping_csv)
    if not p.exists():
        return {}
    import pandas as _pd
    try:
        df = _pd.read_csv(p)
    except Exception:
        return {}
    cols = {c.lower(): c for c in df.columns}
    # Accept either (odds_slug, suggested_game_slug) or (raw, canonical)
    src_col = None
    dst_col = None
    if "odds_slug" in cols and "suggested_game_slug" in cols:
        src_col = cols["odds_slug"]
        dst_col = cols["suggested_game_slug"]
    elif "raw" in cols and "canonical" in cols:
        src_col = cols["raw"]
        dst_col = cols["canonical"]
    if not src_col or not dst_col:
        return {}
    m: Dict[str, str] = {}
    for s, d in zip(df[src_col].astype(str), df[dst_col].astype(str)):
        s_norm = normalize_name(s)
        d_norm = normalize_name(d)
        if s_norm and d_norm and s_norm != d_norm:
            m[s_norm] = d_norm
    return m


def pair_key(home: str, away: str, mapping: Optional[Dict[str, str]] = None) -> str:
    a = normalize_name(home)
    b = normalize_name(away)
    if mapping:
        a = mapping.get(a, a)
        b = mapping.get(b, b)
    return "::".join(sorted([a, b]))


def prepare_games_keys(games: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce").dt.date
    g["pair_key"] = g.apply(lambda r: pair_key(r["home_team"], r["away_team"], mapping), axis=1)
    return g


def prepare_closing_keys(closing: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    c = closing.copy()
    # commence_time may be missing; fall back to NaT and date as None
    if "commence_time" in c.columns:
        c["commence_time"] = pd.to_datetime(c["commence_time"], errors="coerce")
        c["date"] = c["commence_time"].dt.date
    else:
        c["date"] = pd.NaT
    # Build pair key from names if present
    if {"home_team_name", "away_team_name"}.issubset(c.columns):
        c["pair_key"] = c.apply(lambda r: pair_key(str(r["home_team_name"]), str(r["away_team_name"]), mapping) if pd.notna(r["home_team_name"]) and pd.notna(r["away_team_name"]) else None, axis=1)
    else:
        c["pair_key"] = None
    return c


def join_games_with_closing(
    games: pd.DataFrame,
    closing: pd.DataFrame,
    date_tolerance_days: int = 1,
    mapping_csv: str | pathlib.Path | None = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Join games to closing lines by normalized team pair and approximate date.

    - Computes pair keys for both games and closing lines using normalized names
    - Matches primarily on pair_key
    - Filters to rows where the absolute difference between the game's local date and the line's commence date
      is <= date_tolerance_days (default 1) to accommodate UTC vs local date rollovers

    Returns a possibly multi-row per game DataFrame (per-book and per-market/period).
    """
    mapping = _load_mapping(mapping_csv)
    g = prepare_games_keys(games, mapping)
    c = prepare_closing_keys(closing, mapping)
    # First join on pair_key to avoid missing UTC date rollovers
    merged = g.merge(c, on=["pair_key"], how="inner", suffixes=("_game", "_line"))
    # Keep within tolerance window by calendar date
    if "date_game" not in merged.columns:
        merged = merged.rename(columns={"date_x": "date_game", "date_y": "date_line"})
    else:
        # Construct explicit date columns
        merged["date_game"] = merged["date_game"]
        merged["date_line"] = merged["date_line"]
    # Ensure proper types
    merged["date_game"] = pd.to_datetime(merged["date_game"], errors="coerce").dt.date
    merged["date_line"] = pd.to_datetime(merged["date_line"], errors="coerce").dt.date
    # Compute absolute day difference
    merged["_day_diff"] = (pd.to_datetime(merged["date_line"]) - pd.to_datetime(merged["date_game"])).dt.days.abs()
    tol = int(max(0, date_tolerance_days))
    merged = merged[merged["_day_diff"] <= tol].copy()
    # Prefer exact-date matches when multiple per game/book exist; otherwise keep nearest by day diff
    merged.sort_values(["game_id", "_day_diff", "book" if "book" in merged.columns else "pair_key"], inplace=True)
    # If there are multiple rows per game/book/market/period, keep the first (closest date)
    subset_cols = [c for c in ["game_id", "book", "market", "period"] if c in merged.columns]
    if subset_cols:
        merged = merged.drop_duplicates(subset=subset_cols + ["event_id"] if "event_id" in merged.columns else subset_cols, keep="first")
    if not allow_partial:
        return merged
    # Partial matching: for games with no rows, attempt to match lines where at least one team slug coincides.
    matched_game_ids = set(merged.get("game_id", pd.Series(dtype=str)).astype(str)) if "game_id" in merged.columns else set()
    remaining = g[~g["game_id"].astype(str).isin(matched_game_ids)].copy() if "game_id" in g.columns else g.iloc[0:0]
    if remaining.empty:
        return merged
    # Build team slug columns
    def _canon(x: str) -> str:
        return normalize_name(str(x))
    c_slugs = c.copy()
    if {"home_team_name", "away_team_name"}.issubset(c_slugs.columns):
        c_slugs["_home_slug"] = c_slugs["home_team_name"].map(_canon)
        c_slugs["_away_slug"] = c_slugs["away_team_name"].map(_canon)
    else:
        c_slugs["_home_slug"] = None
        c_slugs["_away_slug"] = None
    partial_rows = []
    for _, gr in remaining.iterrows():
        hs = _canon(gr["home_team"])
        as_ = _canon(gr["away_team"])
        cand = c_slugs[(c_slugs["_home_slug"] == hs) | (c_slugs["_home_slug"] == as_) | (c_slugs["_away_slug"] == hs) | (c_slugs["_away_slug"] == as_)]
        if cand.empty:
            continue
        # Tag partial
        cand2 = cand.copy()
        cand2["game_id"] = gr.get("game_id")
        cand2["home_team_game"] = gr.get("home_team")
        cand2["away_team_game"] = gr.get("away_team")
        # Set date columns and apply day tolerance filter similar to exact matches
        cand2["date_game"] = pd.to_datetime(gr.get("date"), errors="coerce").date() if pd.notna(gr.get("date")) else pd.NaT
        # date_line already computed as 'date' in prepare_closing_keys; rename/copy for consistency
        if "date" in cand2.columns:
            cand2["date_line"] = pd.to_datetime(cand2["date"], errors="coerce").dt.date
        else:
            cand2["date_line"] = pd.NaT
        try:
            cand2["_day_diff"] = (pd.to_datetime(cand2["date_line"]) - pd.to_datetime(cand2["date_game"])).dt.days.abs()
            cand2 = cand2[cand2["_day_diff"] <= tol]
        except Exception:
            pass
        cand2["partial_pair"] = True
        partial_rows.append(cand2)
    if partial_rows:
        pr_df = pd.concat(partial_rows, ignore_index=True)
        # Harmonize columns with merged
        for col in merged.columns:
            if col not in pr_df.columns:
                pr_df[col] = None
        # Ensure consistent naming
        if "partial_pair" not in merged.columns:
            merged["partial_pair"] = False
        merged = pd.concat([merged, pr_df[merged.columns]], ignore_index=True)
    return merged
