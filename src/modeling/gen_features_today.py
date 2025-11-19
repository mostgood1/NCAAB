import argparse
import hashlib
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np

OUT = Path(os.getenv("NCAAB_OUTPUTS_DIR", "outputs"))

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
        # Base deterministic components
        h_seed = _h(home, 17)
        a_seed = _h(away, 17)
        pair_seed = _h(home + "::" + away, 11)
        # Offense ratings (rough plausible NCAA range 95-115)
        home_off = 100 + (h_seed % 11)  # 100-110
        away_off = 100 + (a_seed % 13)  # 100-112
        # Defense ratings (lower better); invert seed slightly
        home_def = 100 - (h_seed % 7)   # 93-100
        away_def = 100 - (a_seed % 9)   # 91-100
        # Tempo ratings (possessions ~ 65-75) using pair_seed anchor
        tempo_base = 68 + (pair_seed % 9)  # 68-76
        # Mild home/away tempo adjustments
        home_tempo = tempo_base + ((h_seed % 3) - 1)  # -1,0,+1
        away_tempo = tempo_base + ((a_seed % 3) - 1)
        tempo_sum = home_tempo + away_tempo
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
