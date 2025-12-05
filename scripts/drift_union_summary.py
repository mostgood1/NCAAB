import argparse
import json
from pathlib import Path
import pandas as pd

"""
Union odds_history snapshots to compute drift vs market lines with thresholds.

Inputs:
- outputs/odds_history/odds_*.csv (today/last/closing snapshots, any available)
- outputs/predictions_unified_enriched_*.csv (model totals/margins)

Outputs:
- outputs/drift/drift_union_summary.json with threshold counts and basic stats
"""

ODDS_DIR = Path("outputs/odds_history")
DRIFT_DIR = Path("outputs/drift")
PRED_DIR = Path("outputs")


def load_odds_union(limit_days: int) -> pd.DataFrame:
    rows = []
    if not ODDS_DIR.exists():
        return pd.DataFrame()
    for p in sorted(ODDS_DIR.glob("odds_*.csv"), reverse=True):
        try:
            df = pd.read_csv(p, low_memory=False)
            df["_src"] = p.name
            # Filter to full-game totals market if columns exist
            if "market" in df.columns:
                df = df[df["market"].astype(str).str.lower() == "totals"]
            # Do not filter by period; provider may omit/label unexpectedly
            rows.append(df)
        except Exception:
            continue
        if len(rows) > 0 and limit_days > 0:
            # We will filter by date later if the column exists; just accumulate for now.
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    # Prefer closing snapshots: group by event_id+book, keep latest fetched_at
    if set(["event_id", "book", "fetched_at"]).issubset(df.columns):
        try:
            df["_fetched_ts"] = pd.to_datetime(df["fetched_at"], errors="coerce")
            df = df.sort_values(["event_id", "book", "_fetched_ts"]).groupby(["event_id", "book"], as_index=False).tail(1)
        except Exception:
            pass
    # Normalize game_id to string for joins
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    return df


def load_spreads_union(limit_days: int) -> pd.DataFrame:
    rows = []
    if not ODDS_DIR.exists():
        return pd.DataFrame()
    for p in sorted(ODDS_DIR.glob("odds_*.csv"), reverse=True):
        try:
            df = pd.read_csv(p, low_memory=False)
            df["_src"] = p.name
            # Filter to spreads market if columns exist
            if "market" in df.columns:
                m = df["market"].astype(str).str.lower()
                df = df[m == "spreads"]
            # If market label missing, keep rows that have spread columns
            if "home_spread" in df.columns or "away_spread" in df.columns:
                pass
            else:
                # no spreads info, skip
                continue
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    # Prefer closing snapshots
    if set(["event_id", "book", "fetched_at"]).issubset(df.columns):
        try:
            df["_fetched_ts"] = pd.to_datetime(df["fetched_at"], errors="coerce")
            df = df.sort_values(["event_id", "book", "_fetched_ts"]).groupby(["event_id", "book"], as_index=False).tail(1)
        except Exception:
            pass
    return df


def load_predictions_union(limit_days: int) -> pd.DataFrame:
    rows = []
    for p in sorted(PRED_DIR.glob("predictions_unified_enriched_*.csv"), reverse=True):
        try:
            df = pd.read_csv(p, low_memory=False)
            df["_date"] = p.stem.split("_")[-1]
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    # Normalize game_id to string
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    return df


def pick_cols(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive match
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def compute_summary(limit_days: int):
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    odds = load_odds_union(limit_days)
    odds_spreads = load_spreads_union(limit_days)
    preds = load_predictions_union(limit_days)
    if odds.empty or preds.empty:
        summary = {"window_days": limit_days, "counts": {}, "note": "missing odds or predictions union"}
        with open(DRIFT_DIR / "drift_union_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary))
        return

    # Pick market totals and margins columns from odds
    # From odds_history headers, available columns include: event_id, book, total, home_spread, away_spread
    # Use 'total' for market_total and 'home_spread' as market spread proxy (home side)
    market_total_col = pick_cols(odds, ["total"]) 
    market_spread_col = pick_cols(odds_spreads, ["home_spread", "away_spread"]) 

    # Pick model totals and margins from preds
    # From predictions enriched headers, prefer pred_total_model/pred_margin_model, else fall back to pred_total/pred_margin
    model_total_col = pick_cols(preds, ["pred_total_model", "pred_total"]) 
    model_margin_col = pick_cols(preds, ["pred_margin_model", "pred_margin"]) 

    joined = None
    # Join key: odds_history lacks game_id; join on team names + date if present
    team_keys_odds = [c for c in ["home_team_name", "away_team_name"] if c in odds.columns]
    team_keys_odds_spreads = [c for c in ["home_team_name", "away_team_name"] if c in odds_spreads.columns]
    team_keys_preds = [c for c in ["home_team", "away_team"] if c in preds.columns]
    date_cols_odds = [c for c in ["commence_time", "date"] if c in odds.columns]
    date_cols_preds = [c for c in ["date", "display_date"] if c in preds.columns]

    # Build simplified keys for join
    def norm_team(s):
        return s.str.lower().str.replace(" ", "", regex=False)

    ok = odds.copy()
    ok_s = odds_spreads.copy()
    pk = preds.copy()
    if team_keys_odds and team_keys_preds:
        ok["_home"] = norm_team(ok[team_keys_odds[0]])
        ok["_away"] = norm_team(ok[team_keys_odds[1]])
        pk["_home"] = norm_team(pk[team_keys_preds[0]])
        pk["_away"] = norm_team(pk[team_keys_preds[1]])
    else:
        ok["_home"] = ""
        ok["_away"] = ""
        pk["_home"] = ""
        pk["_away"] = ""

    if team_keys_odds_spreads and team_keys_preds:
        ok_s["_home"] = norm_team(ok_s[team_keys_odds_spreads[0]])
        ok_s["_away"] = norm_team(ok_s[team_keys_odds_spreads[1]])
    else:
        ok_s["_home"] = ok["_home"] if "_home" in ok.columns else ""
        ok_s["_away"] = ok["_away"] if "_away" in ok.columns else ""

    # Attempt to extract canonical date from commence_time (YYYY-MM-DD prefix)
    if "commence_time" in ok.columns:
        ok["_date"] = ok["commence_time"].astype(str).str.slice(0, 10)
    elif "date" in ok.columns:
        ok["_date"] = ok["date"].astype(str)
    else:
        ok["_date"] = ""
    if "date" in pk.columns:
        pk["_date"] = pk["date"].astype(str)
    elif "display_date" in pk.columns:
        pk["_date"] = pk["display_date"].astype(str)
    else:
        pk["_date"] = ""

    join_cols = ["_date", "_home", "_away"]

    # Optionally align by canonical slate date window using predictions' display_date/date
    # Keep only rows where dates match exactly to reduce cross-day noise
    ok = ok[ok["_date"].isin(pk["_date"].unique())]
    if "commence_time" in ok_s.columns:
        ok_s["_date"] = ok_s["commence_time"].astype(str).str.slice(0, 10)
    elif "date" in ok_s.columns:
        ok_s["_date"] = ok_s["date"].astype(str)
    else:
        ok_s["_date"] = ""
    ok_s = ok_s[ok_s["_date"].isin(pk["_date"].unique())]

    if market_total_col and model_total_col:
        jt = ok[join_cols + [market_total_col]].merge(
            pk[join_cols + [model_total_col]], on=join_cols, how="inner", suffixes=("_mkt", "_model")
        )
        jt["delta_total"] = jt[market_total_col] - jt[model_total_col]
        joined = jt if joined is None else joined.merge(jt[join_cols + ["delta_total"]], on=join_cols, how="outer")
    if market_spread_col and model_margin_col:
        js = ok_s[join_cols + [market_spread_col]].merge(
            pk[join_cols + [model_margin_col]], on=join_cols, how="inner", suffixes=("_mkt", "_model")
        )
        # If using away_spread, invert sign to align with home margin direction
        if market_spread_col == "away_spread":
            js["_mkt_spread"] = -js[market_spread_col]
        else:
            js["_mkt_spread"] = js[market_spread_col]
        js["delta_margin"] = js["_mkt_spread"] - js[model_margin_col]
        joined = js if joined is None else joined.merge(js[join_cols + ["delta_margin"]], on=join_cols, how="outer")

    if joined is None or joined.empty:
        summary = {"window_days": limit_days, "counts": {}, "note": "no joined deltas"}
        with open(DRIFT_DIR / "drift_union_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary))
        return

    # Threshold counts
    def counts_for(col):
        if col not in joined.columns:
            return {}
        s = joined[col].dropna().abs()
        return {
            ">3": int((s > 3).sum()),
            ">5": int((s > 5).sum()),
            ">7": int((s > 7).sum()),
            "n": int(s.size),
            "median": float(s.median()) if s.size else 0.0,
        }

    counts = {
        "total": counts_for("delta_total"),
        "margin": counts_for("delta_margin"),
    }

    summary = {"window_days": limit_days, "counts": counts}
    with open(DRIFT_DIR / "drift_union_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=60)
    args = ap.parse_args()
    compute_summary(args.limit_days)


if __name__ == "__main__":
    main()
