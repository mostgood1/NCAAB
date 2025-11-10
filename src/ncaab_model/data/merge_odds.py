from __future__ import annotations

import pandas as pd
from typing import Dict, Optional
try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover
    # Provide a dummy fuzz interface if rapidfuzz is not installed yet
    class _DummyFuzz:
        @staticmethod
        def token_set_ratio(a: str, b: str) -> int:
            return 0
    fuzz = _DummyFuzz()  # type: ignore
from .team_normalize import canonical_slug as normalize_name
from .team_normalize import pair_key as _pair_key


def match_key(team_a: str, team_b: str) -> str:
    return _pair_key(team_a, team_b)


def join_odds_to_games(
    games: pd.DataFrame,
    odds: pd.DataFrame,
    team_map: Optional[Dict[str, str]] = None,
    use_fuzzy: bool = True,
    fuzzy_threshold: int = 92,
    date_tolerance_days: int = 1,
) -> pd.DataFrame:
    """
    Join odds to games using normalized team name pairs and same-day dates.

    games: expects columns ['date', 'home_team', 'away_team']
    odds: expects columns ['commence_time', 'home_team_name', 'away_team_name', 'book', 'spread', 'total', ...]

    Returns a DataFrame with one row per game per bookmaker (may duplicate games across books), including odds columns.
    """
    g = games.copy()
    # Apply mapping on game team names if provided
    if team_map:
        def map_team(x: str) -> str:
            key = normalize_name(x)
            return team_map.get(key, x)
        g['home_team'] = g['home_team'].astype(str).map(map_team)
        g['away_team'] = g['away_team'].astype(str).map(map_team)
    # Ensure datetime types
    g['date'] = pd.to_datetime(g['date'])
    g['game_date'] = g['date'].dt.date
    g['game_key'] = g.apply(lambda r: match_key(r['home_team'], r['away_team']), axis=1)

    o = odds.copy()
    if 'commence_time' in o.columns:
        o['commence_time'] = pd.to_datetime(o['commence_time'], errors='coerce')
        o['odds_date'] = o['commence_time'].dt.date
    else:
        o['odds_date'] = pd.NaT
    o['home_team_name'] = o.get('home_team_name', '')
    o['away_team_name'] = o.get('away_team_name', '')

    # Apply mapping on odds team names if provided
    if team_map:
        def map_team_odds(x: str) -> str:
            key = normalize_name(x)
            return team_map.get(key, x)
        o['home_team_name'] = o['home_team_name'].astype(str).map(map_team_odds)
        o['away_team_name'] = o['away_team_name'].astype(str).map(map_team_odds)
    o['odds_key'] = o.apply(lambda r: match_key(r['home_team_name'], r['away_team_name']), axis=1)

    merged = g.merge(
        o,
        left_on=['game_key', 'game_date'],
        right_on=['odds_key', 'odds_date'],
        how='left',
        suffixes=("", "_odds"),
    )

    # Ensure boolean-like columns can hold missing values without dtype issues
    try:
        import pandas as _pd
        for col in merged.columns:
            if _pd.api.types.is_bool_dtype(merged[col].dtype):
                merged[col] = merged[col].astype('boolean')  # pandas nullable BooleanDtype
    except Exception:
        pass

    # Fallback: date tolerance join for rows without odds (pair-key only, then filter by abs day diff)
    if 'book' in merged.columns:
        no_match_mask = merged['book'].isna()
    else:
        no_match_mask = merged['odds_key'].isna()
    if no_match_mask.any():
        # Build pair-only merged frame
        g_missing = merged.loc[no_match_mask, ['game_id', 'game_key', 'game_date']].drop_duplicates()
        if not g_missing.empty:
            pair_only = g_missing.merge(
                o,
                left_on='game_key',
                right_on='odds_key',
                how='left',
                suffixes=("", "_pair"),
            )
            if 'commence_time' in pair_only.columns:
                pair_only['commence_time'] = pd.to_datetime(pair_only['commence_time'], errors='coerce')
                pair_only['_odds_cal_date'] = pair_only['commence_time'].dt.date
            else:
                pair_only['_odds_cal_date'] = pd.NaT
            # Filter by day diff tolerance
            pair_only['_day_diff'] = (pd.to_datetime(pair_only['_odds_cal_date']) - pd.to_datetime(pair_only['game_date'])).dt.days.abs()
            tol = int(max(0, date_tolerance_days))
            pair_only = pair_only[pair_only['_day_diff'] <= tol]
            # Prefer closest date then keep first per game/book/market/period
            if not pair_only.empty:
                sort_cols = [c for c in ['_day_diff', 'book', 'market', 'period'] if c in pair_only.columns]
                pair_only = pair_only.sort_values(sort_cols)
                dedupe_subset = [c for c in ['game_id', 'book', 'market', 'period'] if c in pair_only.columns]
                if dedupe_subset:
                    pair_only = pair_only.drop_duplicates(subset=dedupe_subset, keep='first')
                # Align columns to merged
                for c in merged.columns:
                    if c not in pair_only.columns:
                        pair_only[c] = pd.NA
                # Overwrite missing rows with fallback odds info
                fallback_cols = [c for c in pair_only.columns if c in merged.columns]
                # Build index by game_id + book for assignment (ensure uniqueness)
                for _, prow in pair_only.iterrows():
                    gid = prow.get('game_id')
                    book = prow.get('book')
                    if pd.isna(book):
                        continue
                    # Locate target rows still unmatched
                    target_mask = (merged['game_id'] == gid) & no_match_mask
                    if target_mask.any():
                        # Assign odds columns into first unmatched row for that game
                        idx = merged.loc[target_mask].index[0]
                        for c in fallback_cols:
                            if c in ['game_id', 'game_key', 'game_date']:
                                continue
                            val = prow.get(c)
                            # Skip missing values of any dtype
                            if val is None or pd.isna(val):
                                continue
                            # Cast bool-like columns safely
                            if c in merged.columns:
                                target_dtype = merged[c].dtype
                                try:
                                    # Handle both native bool and pandas nullable boolean
                                    if str(target_dtype) in ('bool', 'boolean'):
                                        merged.at[idx, c] = bool(val)
                                    else:
                                        merged.at[idx, c] = val
                                except Exception:
                                    merged.at[idx, c] = val
                        # Mark as matched
                        no_match_mask.loc[idx] = False
    if not use_fuzzy:
        return merged

    # Attempt fuzzy remapping for unmatched games (no book/odds after merge)
    if merged.empty or o.empty:
        return merged

    # Identify games without any matched odds rows
    no_odds_mask = ~merged.groupby('game_id')['book'].transform(lambda x: x.notna().any()) if 'book' in merged.columns else merged['odds_key'].isna()
    unmet = merged[no_odds_mask].drop_duplicates(subset=['game_id']) if 'game_id' in merged.columns else pd.DataFrame()
    if unmet.empty:
        return merged

    # Build candidate name pools for the date from odds
    odds_by_date = o.groupby('odds_date') if 'odds_date' in o.columns else {}
    new_rows = []
    for _, row in unmet.iterrows():
        date = row.get('game_date')
        if date not in odds_by_date.groups:
            continue
        pool = odds_by_date.get_group(date)
        g_home = str(row.get('home_team'))
        g_away = str(row.get('away_team'))

        # Fuzzy match home and away separately among candidate odds rows
        candidates = pool[['home_team_name', 'away_team_name']].fillna('').astype(str)
        # Compute best match pair by average of two directions
        best_idx = None
        best_score = -1
        for idx, cand in candidates.iterrows():
            h, a = cand['home_team_name'], cand['away_team_name']
            s1 = fuzz.token_set_ratio(g_home, h)
            s2 = fuzz.token_set_ratio(g_away, a)
            s_alt1 = fuzz.token_set_ratio(g_home, a)
            s_alt2 = fuzz.token_set_ratio(g_away, h)
            score = max((s1 + s2) / 2, (s_alt1 + s_alt2) / 2)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score >= fuzzy_threshold and best_idx is not None:
            # Take all odds rows for that event/book combo for this date
            match_key_val = pool.loc[best_idx, 'odds_key']
            matched = pool[pool['odds_key'] == match_key_val]
            # Rebuild game_key/date join equivalently
            matched = matched.copy()
            matched['game_key'] = row['game_key']
            matched['game_date'] = row['game_date']
            # Merge columns to match structure
            join_cols = [c for c in matched.columns if c not in {'game_key','game_date'}]
            base = row.to_dict()
            for _, mrow in matched.iterrows():
                comb = base.copy()
                for c in join_cols:
                    comb[c] = mrow.get(c)
                new_rows.append(comb)

    if new_rows:
        add_df = pd.DataFrame(new_rows)
        # Keep columns aligned with merged
        for c in merged.columns:
            if c not in add_df.columns:
                add_df[c] = pd.NA
        merged = pd.concat([merged, add_df[merged.columns]], ignore_index=True)
        # Drop duplicates again
        merged = merged.drop_duplicates(subset=[c for c in ['game_id','book','market','period','event_id'] if c in merged.columns], keep='first')
        # Recompute no_odds_mask after fuzzy
        if 'book' in merged.columns:
            no_odds_mask = ~merged.groupby('game_id')['book'].transform(lambda x: x.notna().any())
        else:
            no_odds_mask = merged['odds_key'].isna()

    # Second-pass smart matching: institution-only core comparison with day tolerance
    still_unmet = merged[no_odds_mask].drop_duplicates(subset=['game_id']) if 'game_id' in merged.columns else pd.DataFrame()
    if not still_unmet.empty:
        def _institution_core(n: str) -> str:
            s = normalize_name(str(n))
            # Strip common mascot suffixes if present
            mascots = (
                'redstorm','crimsontide','goldengophers','braves','knights','tigers','owls','rattlers',
                'gauchos','cowboys','tribe','minutemen','greatdanes','spartans','titans','zips','peacocks'
            )
            for m in mascots:
                if s.endswith(m) and len(s) > len(m) + 2:
                    return s[: -len(m)]
            return s

        g_core = still_unmet.copy()
        g_core['inst_home'] = g_core['home_team'].astype(str).map(_institution_core)
        g_core['inst_away'] = g_core['away_team'].astype(str).map(_institution_core)
        g_core['inst_key'] = g_core.apply(lambda r: _pair_key(r['inst_home'], r['inst_away']), axis=1)

        o_core = o.copy()
        o_core['inst_home'] = o_core['home_team_name'].astype(str).map(_institution_core)
        o_core['inst_away'] = o_core['away_team_name'].astype(str).map(_institution_core)
        o_core['inst_key'] = o_core.apply(lambda r: _pair_key(r['inst_home'], r['inst_away']), axis=1)
        # Attach by inst_key with date tolerance
        if 'commence_time' in o_core.columns:
            o_core['commence_time'] = pd.to_datetime(o_core['commence_time'], errors='coerce')
            o_core['_odds_cal_date'] = o_core['commence_time'].dt.date
        else:
            o_core['_odds_cal_date'] = pd.NaT
        tol = int(max(0, date_tolerance_days))
        rows2 = []
        for _, r in g_core.iterrows():
            pool = o_core[o_core['inst_key'] == r['inst_key']].copy()
            if pool.empty:
                continue
            pool['_day_diff'] = (pd.to_datetime(pool['_odds_cal_date']) - pd.to_datetime(r['game_date'])).dt.days.abs()
            pool = pool[pool['_day_diff'] <= tol]
            if pool.empty:
                continue
            # Prefer closest date; then dedupe per book/market/period
            sort_cols = [c for c in ['_day_diff','book','market','period'] if c in pool.columns]
            pool = pool.sort_values(sort_cols)
            dedupe_subset = [c for c in ['book','market','period'] if c in pool.columns]
            if dedupe_subset:
                pool = pool.drop_duplicates(subset=dedupe_subset, keep='first')
            base = merged.loc[merged['game_id'] == r['game_id']].iloc[0].to_dict()
            for _, mrow in pool.iterrows():
                comb = base.copy()
                for c in o_core.columns:
                    if c in {'inst_home','inst_away','inst_key','_odds_cal_date','_day_diff'}:
                        continue
                    if c in comb:
                        comb[c] = mrow.get(c)
                rows2.append(comb)
        if rows2:
            add2 = pd.DataFrame(rows2)
            for c in merged.columns:
                if c not in add2.columns:
                    add2[c] = pd.NA
            merged = pd.concat([merged, add2[merged.columns]], ignore_index=True)
            merged = merged.drop_duplicates(subset=[c for c in ['game_id','book','market','period','event_id'] if c in merged.columns], keep='first')
    return merged
