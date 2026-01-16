import argparse
import hashlib
import os
from pathlib import Path
import sys
import sqlite3
import datetime as dt
import pandas as pd
import numpy as np

try:
    from ncaab_model.features.schedule import compute_rest_days
except Exception:  # pragma: no cover
    compute_rest_days = None

OUT = Path(os.getenv("NCAAB_OUTPUTS_DIR", "outputs"))
DB_PATH = Path(os.getenv("NCAAB_DB_PATH", "data/ncaab.sqlite"))

# Rolling window for boxscore-derived pace/PPP stats.
BOX_LOOKBACK_DAYS = int(os.getenv("NCAAB_BOX_LOOKBACK_DAYS", "120"))
BOX_MAX_GAMES_PER_TEAM = int(os.getenv("NCAAB_BOX_MAX_GAMES_PER_TEAM", "15"))


def _as_date(s: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _norm_team_name(v: object) -> str:
    s = str(v or "").strip().lower()
    return " ".join(s.split())


def _load_team_boxscore_stats(date_str: str) -> dict[str, dict[str, float]]:
    """Load rolling team stats from the SQLite `boxscores` table.

    Produces per-team:
      - pace_mu, pace_sigma
      - ppp_mu, ppp_sigma  (points / possessions)
      - ppp_allowed_mu, ppp_allowed_sigma
      - four factors means when present

    Best-effort: returns empty dict if DB/table unavailable.
    """
    if not DB_PATH.exists():
        return {}

    # Best-effort: our `boxscores` table doesn't reliably have dates and may not
    # join to `games` by game_id. We still want to use it to estimate typical
    # pace/PPP levels and volatility per team by taking the most recent boxscores
    # (proxying recency via game_id ordering).
    sql = (
        "SELECT b.game_id, b.home_team, b.away_team, b.home_score, b.away_score, "
        "b.home_possessions, b.away_possessions, b.pace, "
        "b.home_efg, b.home_tov_rate, b.home_orb_rate, b.home_ftr, "
        "b.away_efg, b.away_tov_rate, b.away_orb_rate, b.away_ftr "
        "FROM boxscores b "
        "WHERE b.game_id IS NOT NULL"
    )
    try:
        con = sqlite3.connect(DB_PATH)
        bs = pd.read_sql_query(sql, con)
        con.close()
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        return {}

    if bs.empty:
        return {}

    # Sort by game_id as a crude recency proxy.
    try:
        bs["game_id"] = pd.to_numeric(bs["game_id"], errors="coerce")
        bs = bs.sort_values("game_id", ascending=False)
    except Exception:
        pass
    # Long-form team rows
    rows = []
    for _, r in bs.iterrows():
        try:
            pace = float(r.get("pace")) if pd.notna(r.get("pace")) else np.nan
        except Exception:
            pace = np.nan

        # Home perspective
        try:
            h_team = str(r.get("home_team") or "").strip()
            a_team = str(r.get("away_team") or "").strip()
            h_pts = float(r.get("home_score")) if pd.notna(r.get("home_score")) else np.nan
            a_pts = float(r.get("away_score")) if pd.notna(r.get("away_score")) else np.nan
            h_poss = float(r.get("home_possessions")) if pd.notna(r.get("home_possessions")) else np.nan
            a_poss = float(r.get("away_possessions")) if pd.notna(r.get("away_possessions")) else np.nan
        except Exception:
            continue

        # Treat placeholder zeros as missing.
        if np.isfinite(pace) and pace <= 0:
            pace = np.nan
        if np.isfinite(h_poss) and h_poss <= 0:
            h_poss = np.nan
        if np.isfinite(a_poss) and a_poss <= 0:
            a_poss = np.nan

        if not np.isfinite(pace) and np.isfinite(h_poss) and np.isfinite(a_poss):
            pace = float(0.5 * (h_poss + a_poss))

        if h_team:
            rows.append({
                "game_id": r.get("game_id"),
                "team": h_team,
                "opp": a_team,
                "pace": pace,
                "ppp": (h_pts / h_poss) if (np.isfinite(h_pts) and np.isfinite(h_poss) and h_poss > 0) else np.nan,
                "ppp_allowed": (a_pts / a_poss) if (np.isfinite(a_pts) and np.isfinite(a_poss) and a_poss > 0) else np.nan,
                "efg": pd.to_numeric(r.get("home_efg"), errors="coerce"),
                "tov_rate": pd.to_numeric(r.get("home_tov_rate"), errors="coerce"),
                "orb_rate": pd.to_numeric(r.get("home_orb_rate"), errors="coerce"),
                "ftr": pd.to_numeric(r.get("home_ftr"), errors="coerce"),
            })
        if a_team:
            rows.append({
                "game_id": r.get("game_id"),
                "team": a_team,
                "opp": h_team,
                "pace": pace,
                "ppp": (a_pts / a_poss) if (np.isfinite(a_pts) and np.isfinite(a_poss) and a_poss > 0) else np.nan,
                "ppp_allowed": (h_pts / h_poss) if (np.isfinite(h_pts) and np.isfinite(h_poss) and h_poss > 0) else np.nan,
                "efg": pd.to_numeric(r.get("away_efg"), errors="coerce"),
                "tov_rate": pd.to_numeric(r.get("away_tov_rate"), errors="coerce"),
                "orb_rate": pd.to_numeric(r.get("away_orb_rate"), errors="coerce"),
                "ftr": pd.to_numeric(r.get("away_ftr"), errors="coerce"),
            })

    long = pd.DataFrame(rows)
    if long.empty or "team" not in long.columns:
        return {}

    # Use most recent games per team by game_id
    stats: dict[str, dict[str, float]] = {}
    for team, g in long.groupby("team", as_index=False):
        try:
            g2 = g.sort_values("game_id", ascending=False).head(max(1, BOX_MAX_GAMES_PER_TEAM))
        except Exception:
            g2 = g
        out: dict[str, float] = {}

        def _mu_sig(col: str, prefix: str, positive_only: bool = False):
            s = pd.to_numeric(g2.get(col), errors="coerce")
            if positive_only:
                s = s.where(s > 0)
            s = s.dropna()
            # With ddof=1, std is defined for >=2 samples.
            if len(s) >= 2:
                out[prefix + "_mu"] = float(s.mean())
                out[prefix + "_sigma"] = float(s.std(ddof=1))
            elif len(s) >= 1:
                out[prefix + "_mu"] = float(s.mean())
                out[prefix + "_sigma"] = float(0.0)

        _mu_sig("pace", "pace", positive_only=True)
        _mu_sig("ppp", "ppp")
        _mu_sig("ppp_allowed", "ppp_allowed")

        for c, nm in [("efg", "efg"), ("tov_rate", "tov_rate"), ("orb_rate", "orb_rate"), ("ftr", "ftr")]:
            s = pd.to_numeric(g2.get(c), errors="coerce").dropna()
            if len(s) >= 2:
                out[nm + "_mu"] = float(s.mean())
                out[nm + "_sigma"] = float(s.std(ddof=1))
            elif len(s) >= 1:
                out[nm + "_mu"] = float(s.mean())
                out[nm + "_sigma"] = float(0.0)

        if out:
            stats[_norm_team_name(team)] = out
    return stats

# Deterministic small integer from a string
_def_hash_mod = 10

def _h(val: str, mod: int = _def_hash_mod) -> int:
    try:
        return int(hashlib.sha1(val.encode("utf-8")).hexdigest(), 16) % mod
    except Exception:
        return 0


def build_features(date_str: str | None) -> pd.DataFrame:
    games_path_today = OUT / "games_curr.csv"
    games_specific = OUT / f"games_{date_str}.csv" if date_str else None
    df_games = pd.DataFrame()
    # Prefer date-specific file if exists and matches date; else games_curr
    try:
        if games_specific and games_specific.exists():
            gtmp = pd.read_csv(games_specific)
            if not gtmp.empty:
                df_games = gtmp
    except Exception:
        df_games = pd.DataFrame()
    if df_games.empty and games_path_today.exists():
        try:
            df_games = pd.read_csv(games_path_today)
        except Exception:
            df_games = pd.DataFrame()
    if df_games.empty:
        print("No games file available; exiting." , file=sys.stderr)
        return pd.DataFrame()
    # Filter date if column present
    if date_str and "date" in df_games.columns:
        try:
            df_games = df_games[df_games["date"].astype(str) == str(date_str)]
        except Exception:
            pass
    if df_games.empty:
        print(f"No games rows for date {date_str}; exiting.", file=sys.stderr)
        return pd.DataFrame()

    def _load_priors() -> dict[str, dict[str, float]]:
        """Load team-level priors for ratings.

        Returns mapping team -> {off_rating, def_rating, tempo_rating}.
        Best effort: priors.csv (preferred) then features_last2.csv aggregation.
        """
        pri: dict[str, dict[str, float]] = {}
        # 1) priors.csv (preferred)
        p_path = OUT / "priors.csv"
        try:
            if p_path.exists():
                p = pd.read_csv(p_path)
                if not p.empty and "team" in p.columns:
                    for r in p.to_dict(orient="records"):
                        team = str(r.get("team") or "").strip()
                        if not team:
                            continue
                        out = {}
                        for k_src, k_dst in (
                            ("off_rating", "off_rating"),
                            ("def_rating", "def_rating"),
                            ("tempo_rating", "tempo_rating"),
                        ):
                            try:
                                v = float(r.get(k_src))
                                if np.isfinite(v):
                                    out[k_dst] = v
                            except Exception:
                                continue
                        if out:
                            pri[team] = out
        except Exception:
            pri = {}

        if pri:
            return pri

        # 2) features_last2.csv aggregation fallback
        f_path = OUT / "features_last2.csv"
        try:
            if f_path.exists():
                f = pd.read_csv(f_path)
                if not f.empty and {"home_team", "away_team"}.issubset(f.columns):
                    def _avg_by_team(home_col: str, away_col: str) -> dict[str, float]:
                        parts = []
                        if home_col in f.columns:
                            a = f[["home_team", home_col]].rename(columns={"home_team": "team", home_col: "v"})
                            parts.append(a)
                        if away_col in f.columns:
                            b = f[["away_team", away_col]].rename(columns={"away_team": "team", away_col: "v"})
                            parts.append(b)
                        if not parts:
                            return {}
                        both = pd.concat(parts, ignore_index=True)
                        both["team"] = both["team"].astype(str)
                        both["v"] = pd.to_numeric(both["v"], errors="coerce")
                        both = both.dropna(subset=["team", "v"])
                        if both.empty:
                            return {}
                        g = both.groupby("team", as_index=True)["v"].mean()
                        return {str(k): float(v) for k, v in g.to_dict().items() if np.isfinite(v)}

                    off_map = _avg_by_team("home_off_rating", "away_off_rating")
                    def_map = _avg_by_team("home_def_rating", "away_def_rating")
                    tmp_map = _avg_by_team("home_tempo_rating", "away_tempo_rating")

                    teams = set(off_map) | set(def_map) | set(tmp_map)
                    for t in teams:
                        out = {}
                        if t in off_map:
                            out["off_rating"] = off_map[t]
                        if t in def_map:
                            out["def_rating"] = def_map[t]
                        if t in tmp_map:
                            out["tempo_rating"] = tmp_map[t]
                        if out:
                            pri[t] = out
        except Exception:
            pri = {}

        return pri

    # Compute schedule context (rest/B2B/neutral) using historical + slate games when possible
    sched_today = pd.DataFrame()
    try:
        if compute_rest_days is not None:
            games_all = pd.DataFrame()
            try:
                p_all = OUT / "games_all.csv"
                if p_all.exists():
                    games_all = pd.read_csv(p_all)
            except Exception:
                games_all = pd.DataFrame()
            combined = pd.concat([g for g in [games_all, df_games] if isinstance(g, pd.DataFrame) and not g.empty], ignore_index=True)
            if not combined.empty and {"game_id", "date", "home_team", "away_team"}.issubset(combined.columns):
                combined["game_id"] = combined["game_id"].astype(str)
                combined = combined.drop_duplicates(subset=["game_id"], keep="last")
                sched = compute_rest_days(combined)
                if isinstance(sched, pd.DataFrame) and not sched.empty and "game_id" in sched.columns:
                    sched["game_id"] = sched["game_id"].astype(str)
                    sched_today = sched[sched["game_id"].isin(df_games["game_id"].astype(str))].copy()
    except Exception:
        sched_today = pd.DataFrame()

    priors = _load_priors()
    team_box = _load_team_boxscore_stats(date_str) if date_str else {}
    # Ensure needed columns
    home_col = next((c for c in ["home_team","home"] if c in df_games.columns), None)
    away_col = next((c for c in ["away_team","away"] if c in df_games.columns), None)
    if not home_col or not away_col:
        print("Missing home/away team columns; exiting.", file=sys.stderr)
        return pd.DataFrame()
    if "game_id" not in df_games.columns:
        # Build deterministic game_id if absent
        df_games["game_id"] = [f"g_{_h(str(r[home_col])+str(r[away_col]),100000)}" for _, r in df_games.iterrows()]
    # Build feature rows
    rows = []
    for _, r in df_games.iterrows():
        home = str(r.get(home_col))
        away = str(r.get(away_col))
        gid = str(r.get("game_id"))

        # Ratings priors (real-data preferred) with deterministic fallback
        h_seed = _h(home, 17)
        a_seed = _h(away, 17)
        pair_seed = _h(home + "::" + away, 11)

        def _prior(team: str, key: str) -> float | None:
            try:
                d = priors.get(team)
                if not d:
                    return None
                v = float(d.get(key))
                return v if np.isfinite(v) else None
            except Exception:
                return None

        # Off/Def ratings (approx 90-120-ish); tempo ~ 65-75 (possessions/40)
        home_off = _prior(home, "off_rating")
        away_off = _prior(away, "off_rating")
        home_def = _prior(home, "def_rating")
        away_def = _prior(away, "def_rating")
        home_tempo = _prior(home, "tempo_rating")
        away_tempo = _prior(away, "tempo_rating")

        if home_off is None:
            home_off = float(100 + (h_seed % 11))
        if away_off is None:
            away_off = float(100 + (a_seed % 13))
        if home_def is None:
            home_def = float(100 - (h_seed % 7))
        if away_def is None:
            away_def = float(100 - (a_seed % 9))
        if home_tempo is None or away_tempo is None:
            tempo_base = float(68 + (pair_seed % 9))
            if home_tempo is None:
                home_tempo = tempo_base + float((h_seed % 3) - 1)
            if away_tempo is None:
                away_tempo = tempo_base + float((a_seed % 3) - 1)

        tempo_sum = float(home_tempo) + float(away_tempo)

        # Boxscore-derived pace/PPP features (rolling)
        hb = team_box.get(_norm_team_name(home)) if team_box else None
        ab = team_box.get(_norm_team_name(away)) if team_box else None

        home_pace_mu = float(hb.get("pace_mu")) if hb and hb.get("pace_mu") is not None else np.nan
        away_pace_mu = float(ab.get("pace_mu")) if ab and ab.get("pace_mu") is not None else np.nan
        home_pace_sigma = float(hb.get("pace_sigma")) if hb and hb.get("pace_sigma") is not None else np.nan
        away_pace_sigma = float(ab.get("pace_sigma")) if ab and ab.get("pace_sigma") is not None else np.nan

        # Matchup pace estimate: prefer boxscore pace means; fallback to tempo ratings.
        if np.isfinite(home_pace_mu) and home_pace_mu > 0 and np.isfinite(away_pace_mu) and away_pace_mu > 0:
            pace_game_est = float(0.5 * (home_pace_mu + away_pace_mu))
        elif np.isfinite(home_pace_mu) and home_pace_mu > 0:
            pace_game_est = float(home_pace_mu)
        elif np.isfinite(away_pace_mu) and away_pace_mu > 0:
            pace_game_est = float(away_pace_mu)
        else:
            pace_game_est = float(0.5 * (float(home_tempo) + float(away_tempo)))

        # Pace sigma: combine team sigmas when available; fallback to modest default.
        sigs = [s for s in [home_pace_sigma, away_pace_sigma] if np.isfinite(s) and s > 0]
        if len(sigs) == 2:
            pace_sigma_game_est = float(np.sqrt(0.5 * (sigs[0] ** 2 + sigs[1] ** 2)))
        elif len(sigs) == 1:
            pace_sigma_game_est = float(sigs[0])
        else:
            pace_sigma_game_est = float(3.5)
        pace_sigma_game_est = float(np.clip(pace_sigma_game_est, 1.5, 7.0))

        # PPP volatility (rolling): used by pace simulator when present
        home_ppp_mu = float(hb.get("ppp_mu")) if hb and hb.get("ppp_mu") is not None else np.nan
        away_ppp_mu = float(ab.get("ppp_mu")) if ab and ab.get("ppp_mu") is not None else np.nan
        home_ppp_sigma = float(hb.get("ppp_sigma")) if hb and hb.get("ppp_sigma") is not None else np.nan
        away_ppp_sigma = float(ab.get("ppp_sigma")) if ab and ab.get("ppp_sigma") is not None else np.nan

        home_ppp_allowed_mu = float(hb.get("ppp_allowed_mu")) if hb and hb.get("ppp_allowed_mu") is not None else np.nan
        away_ppp_allowed_mu = float(ab.get("ppp_allowed_mu")) if ab and ab.get("ppp_allowed_mu") is not None else np.nan

        # Schedule features
        rest_home = None
        rest_away = None
        b2b_home = None
        b2b_away = None
        neutral_site = None
        try:
            if not sched_today.empty:
                sr = sched_today[sched_today["game_id"].astype(str) == gid]
                if not sr.empty:
                    row0 = sr.iloc[0].to_dict()
                    rest_home = row0.get("rest_home")
                    rest_away = row0.get("rest_away")
                    b2b_home = row0.get("b2b_home")
                    b2b_away = row0.get("b2b_away")
                    neutral_site = row0.get("neutral_site")
        except Exception:
            pass

        if neutral_site is None and "neutral_site" in df_games.columns:
            neutral_site = r.get("neutral_site")

        rows.append({
            "game_id": gid,
            "date": date_str or r.get("date"),
            "home_team": home,
            "away_team": away,
            "home_off_rating": float(home_off),
            "away_off_rating": float(away_off),
            "home_def_rating": float(home_def),
            "away_def_rating": float(away_def),
            "home_tempo_rating": float(home_tempo),
            "away_tempo_rating": float(away_tempo),
            "tempo_rating_sum": float(tempo_sum),
            # Pace / possessions estimates (for pace-aware simulation)
            "pace_game_est": float(pace_game_est),
            "possessions_game_est": float(pace_game_est),
            "pace_sigma_game_est": float(pace_sigma_game_est),
            "home_pace_mu": home_pace_mu,
            "away_pace_mu": away_pace_mu,
            "home_pace_sigma": home_pace_sigma,
            "away_pace_sigma": away_pace_sigma,
            # PPP means/volatility (best-effort)
            "home_ppp_mu": home_ppp_mu,
            "away_ppp_mu": away_ppp_mu,
            "home_ppp_sigma": home_ppp_sigma,
            "away_ppp_sigma": away_ppp_sigma,
            "home_ppp_allowed_mu": home_ppp_allowed_mu,
            "away_ppp_allowed_mu": away_ppp_allowed_mu,
            "rest_home": rest_home,
            "rest_away": rest_away,
            "b2b_home": b2b_home,
            "b2b_away": b2b_away,
            "neutral_site": neutral_site,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Generate simplistic feature rows for today's slate or a specified date.")
    ap.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD; defaults to today inferred from system timezone if omitted.")
    ap.add_argument("--write-dated", action="store_true", help="Also write a dated features_<date>.csv artifact.")
    args = ap.parse_args()
    date_str = args.date
    # Infer today if not provided
    if not date_str:
        try:
            import datetime as dt
            date_str = dt.datetime.now().strftime("%Y-%m-%d")
        except Exception:
            pass
    df = build_features(date_str)
    if df.empty:
        print("No features generated.", file=sys.stderr)
        sys.exit(1)
    out_path_curr = OUT / "features_curr.csv"
    try:
        df.to_csv(out_path_curr, index=False)
        print(f"Wrote {len(df)} feature rows to {out_path_curr}")
    except Exception as e:
        print(f"Failed writing features_curr.csv: {e}", file=sys.stderr)
    if args.write_dated and date_str:
        out_dated = OUT / f"features_{date_str}.csv"
        try:
            df.to_csv(out_dated, index=False)
            print(f"Wrote dated features file {out_dated}")
        except Exception as e:
            print(f"Failed writing dated features file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
