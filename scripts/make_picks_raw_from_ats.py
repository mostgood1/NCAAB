import sys
import argparse
from pathlib import Path
import pandas as pd

def build_picks_raw(date: str, outputs_dir: Path) -> Path | None:
    ats_path = outputs_dir / "picks" / f"ats_picks_{date}.csv"
    if not ats_path.exists():
        return None
    df = pd.read_csv(ats_path)
    if df.empty:
        return None
    # Deduplicate any repeated headers
    try:
        df = df.loc[:, ~df.columns.duplicated()]
    except Exception:
        pass
    # Normalize expected columns
    def col(name: str):
        for c in df.columns:
            if c.lower() == name:
                return df[c]
        return None
    out = pd.DataFrame()
    out["game_id"] = col("game_id")
    out["date"] = date
    out["home_team"] = col("home_team")
    out["away_team"] = col("away_team")
    out["market"] = "spreads"
    out["period"] = "full_game"
    side = col("ats_side")
    out["bet"] = (side.astype(str).str.lower() if side is not None else "home")
    # Derive line: home closing spread (fallback to spread_home); flip for away
    csh = pd.to_numeric(col("closing_spread_home"), errors="coerce") if col("closing_spread_home") is not None else None
    sh = pd.to_numeric(col("spread_home"), errors="coerce") if col("spread_home") is not None else None
    def line_for_row(i: int):
        try:
            ln_home = None
            if csh is not None and i < len(csh):
                ln_home = float(csh.iloc[i]) if pd.notna(csh.iloc[i]) else None
            if ln_home is None and sh is not None and i < len(sh):
                ln_home = float(sh.iloc[i]) if pd.notna(sh.iloc[i]) else None
            if ln_home is None:
                return None
            sel = str(out["bet"].iloc[i]).lower()
            return ln_home if sel == "home" else (0 - ln_home)
        except Exception:
            return None
    out["line"] = [ line_for_row(i) for i in range(len(out)) ]
    out["price"] = None
    delt = pd.to_numeric(col("_delta"), errors="coerce") if col("_delta") is not None else None
    out["edge"] = (delt.abs() if delt is not None else None)
    pm = pd.to_numeric(col("_pred_margin_blend"), errors="coerce") if col("_pred_margin_blend") is not None else None
    out["pred_margin"] = pm
    out["pred_total"] = None
    out["rec_type"] = "Spread"
    out["rec_code"] = "ATS"
    try:
        if "game_id" in out.columns:
            out["game_id"] = out["game_id"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            out["_edge_abs"] = pd.to_numeric(out.get("edge"), errors="coerce").abs()
            out = (
                out.sort_values(["game_id", "_edge_abs"], ascending=[True, False], na_position="last")
                .drop_duplicates(subset=["game_id"], keep="first")
                .drop(columns=["_edge_abs"], errors="ignore")
                .reset_index(drop=True)
            )
    except Exception:
        pass
    # Write to outputs/picks_raw.csv
    out_path = outputs_dir / "picks_raw.csv"
    out.to_csv(out_path, index=False)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Create picks_raw.csv from per-date ATS picks")
    ap.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    ap.add_argument("--outputs", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    args = ap.parse_args()
    out = Path(args.outputs)
    out_path = build_picks_raw(args.date, out)
    if out_path is None:
        print(f"[skip] ATS picks not found or empty for {args.date}")
        sys.exit(0)
    print(f"[ok] Wrote picks_raw: {out_path}")

if __name__ == "__main__":
    main()
