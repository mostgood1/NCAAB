import os
import re
import sys
from typing import Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_DIR = os.path.join(ROOT, "outputs", "daily_results")
CLOSING_PATH = os.path.join(ROOT, "outputs", "games_with_closing.csv")

DATE_RE = re.compile(r"^results_(\d{4}-\d{2}-\d{2})\.csv$")

# Minimal alias map; extend via data/provider_aliases.csv if present
ALIASES_PATH = os.path.join(ROOT, "data", "provider_aliases.csv")
ALIASES = {}
if os.path.exists(ALIASES_PATH):
    try:
        alias_df = pd.read_csv(ALIASES_PATH)
        for _, r in alias_df.iterrows():
            src = str(r.get("provider_name", "")).strip().lower()
            canon = str(r.get("canon_name", "")).strip().lower()
            if src and canon:
                ALIASES[src] = canon
    except Exception:
        pass

# Team branding map for canonical names
TEAM_MAP_PATH = os.path.join(ROOT, "data", "team_map.csv")
TEAM_CANON = {}
if os.path.exists(TEAM_MAP_PATH):
    try:
        tdf = pd.read_csv(TEAM_MAP_PATH)
        for _, r in tdf.iterrows():
            raw = str(r.get("raw_name", "")).strip().lower()
            canon = str(r.get("canon_name", "")).strip().lower()
            if raw and canon:
                TEAM_CANON[raw] = canon
    except Exception:
        pass


def canonize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    if s in ALIASES:
        s = ALIASES[s]
    if s in TEAM_CANON:
        s = TEAM_CANON[s]
    # normalize punctuation spacing
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_closing() -> pd.DataFrame:
    if not os.path.exists(CLOSING_PATH):
        raise FileNotFoundError(f"Missing {CLOSING_PATH}")
    df = pd.read_csv(CLOSING_PATH)
    # Expect columns: date, home_team, away_team, close_total, close_home_spread
    cols = df.columns.str.lower()
    df.columns = cols
    # Try to establish a date column if missing
    if "date" not in df.columns:
        # Common alternates: game_date, event_date, start_time
        for alt in ("game_date", "event_date"):
            if alt in df.columns:
                df["date"] = df[alt].astype(str).str.slice(0, 10)
                break
        if "date" not in df.columns and "start_time" in df.columns:
            # Attempt to parse ISO datetime
            try:
                dt = pd.to_datetime(df["start_time"], errors="coerce")
                df["date"] = dt.dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    # Standardize expected columns
    rename = {}
    if "close_total" in df.columns:
        rename["close_total"] = "market_total"
    if "close_home_spread" in df.columns:
        rename["close_home_spread"] = "spread_home"
    df = df.rename(columns=rename)
    # Canonical keys
    for c in ("home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(canonize)
    # Keep only necessary columns
    keep = [c for c in ["date", "home_team", "away_team", "market_total", "spread_home"] if c in df.columns]
    # If date could not be established, return without dropping on date
    if keep:
        if "date" in df.columns:
            return df[keep].dropna(subset=["date"])
        else:
            return df[keep]
    else:
        return pd.DataFrame(columns=["date","home_team","away_team","market_total","spread_home"]) 


def persist_for_file(path: str, closing_df: pd.DataFrame) -> Tuple[int, int, int]:
    base = os.path.basename(path)
    m = DATE_RE.match(base)
    if not m:
        return (0, 0, 0)
    date = m.group(1)
    df = pd.read_csv(path)
    cols = df.columns.str.lower()
    df.columns = cols
    # Create canonical join keys
    for c in ("home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(canonize)
    df["date"] = df.get("date", date)
    df["date"] = df["date"].astype(str)

    # Prefer merge by game_id if available
    use_game_id = "game_id" in df.columns and "game_id" in closing_df.columns
    # Merge odds from closing by date + pair when date available; otherwise by pair only
    if not use_game_id:
        if "date" in closing_df.columns:
            closing_on_date = closing_df[closing_df["date"] == date]
        else:
            closing_on_date = closing_df.copy()
    before_total = df["market_total"].notna().sum() if "market_total" in df.columns else 0
    before_spread = df["spread_home"].notna().sum() if "spread_home" in df.columns else 0

    if use_game_id:
        merged = pd.merge(df, closing_df, on=["game_id"], how="left", suffixes=("", "_closing"))
    else:
        join_keys = [k for k in ["date", "home_team", "away_team"] if k in closing_on_date.columns]
        merged = pd.merge(df, closing_on_date, on=join_keys, how="left", suffixes=("", "_closing"))
    # Fill standardized columns
    if "market_total" not in merged.columns and "market_total_closing" in merged.columns:
        merged["market_total"] = merged["market_total_closing"]
    elif "market_total" in merged.columns and "market_total_closing" in merged.columns:
        merged["market_total"] = merged["market_total"].fillna(merged["market_total_closing"])

    if "spread_home" not in merged.columns and "spread_home_closing" in merged.columns:
        merged["spread_home"] = merged["spread_home_closing"]
    elif "spread_home" in merged.columns and "spread_home_closing" in merged.columns:
        merged["spread_home"] = merged["spread_home"].fillna(merged["spread_home_closing"])

    after_total = merged["market_total"].notna().sum() if "market_total" in merged.columns else 0
    after_spread = merged["spread_home"].notna().sum() if "spread_home" in merged.columns else 0

    # Clean helper columns
    for c in ["market_total_closing", "spread_home_closing"]:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    # Write back
    merged.to_csv(path, index=False)
    improved_total = max(0, after_total - before_total)
    improved_spread = max(0, after_spread - before_spread)
    return (len(df), improved_total, improved_spread)


def main():
    closing_df = load_closing()
    if closing_df.empty:
        print("No closing odds available; nothing to persist.")
        return 0

    files = [os.path.join(DAILY_DIR, f) for f in os.listdir(DAILY_DIR) if DATE_RE.match(f)]
    files.sort()
    total_rows = 0
    total_improved_total = 0
    total_improved_spread = 0
    unmatched = 0

    for fp in files:
        rows, inc_total, inc_spread = persist_for_file(fp, closing_df)
        total_rows += rows
        total_improved_total += inc_total
        total_improved_spread += inc_spread
        # Track unmatched by checking remaining NaNs for this date
        df = pd.read_csv(fp)
        c_total = df["market_total"].notna().sum() if "market_total" in df.columns else 0
        c_spread = df["spread_home"].notna().sum() if "spread_home" in df.columns else 0
        unmatched += (rows - max(c_total, c_spread))
        print(f"Updated {os.path.basename(fp)}: +tot {inc_total}, +spr {inc_spread}")

    print(f"Summary: rows={total_rows}, improved_total={total_improved_total}, improved_spread={total_improved_spread}, unmatched_est={unmatched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
