import os
import sys
import pandas as pd

def main(date_str: str) -> int:
    base = os.path.join(os.path.dirname(__file__), "..", "outputs")
    base = os.path.abspath(base)
    edges = os.path.join(base, f"align_period_{date_str}_edges.csv")
    out = os.path.join(base, f"predictions_display_{date_str}.csv")
    if not os.path.exists(edges):
        print(f"Missing edges file: {edges}")
        return 2
    df = pd.read_csv(edges, low_memory=False)
    cols: dict[str, pd.Series] = {}
    for k in ("game_id","home_team","away_team","pred_total","pred_margin","display_date","start_time","start_time_local","start_time_iso"):
        if k in df.columns:
            cols[k] = df[k]
    if "total" in df.columns:
        cols["market_total"] = df["total"]
    order = [
        "game_id","home_team","away_team","pred_total","pred_margin",
        "market_total","display_date","start_time","start_time_local","start_time_iso"
    ]
    out_df = pd.DataFrame({k: cols.get(k) for k in order if k in cols})
    out_df.to_csv(out, index=False)
    print(out)
    return 0

if __name__ == "__main__":
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    if not date_str:
        print("Usage: generate_display_from_edges.py YYYY-MM-DD", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(main(date_str))
