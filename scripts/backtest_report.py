from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True, help="Path to backtest daily CSV")
    ap.add_argument("--picks", required=True, help="Path to backtest picks CSV")
    args = ap.parse_args()

    daily_path = Path(args.daily)
    picks_path = Path(args.picks)

    x = pd.read_csv(daily_path).fillna(
        {"wins": 0, "losses": 0, "pushes": 0, "picks": 0, "units": 0.0}
    )
    p = pd.read_csv(picks_path)
    res = p["result"].astype(str).str.upper().str.strip()
    p = p.assign(_W=res.eq("W"), _L=res.eq("L"), _P=res.eq("P"))

    tw = int(x.wins.sum())
    tl = int(x.losses.sum())
    tp = int(x.pushes.sum())
    total_picks = int(x.picks.sum())
    tu = float(x.units.sum())
    wr = (tw / (tw + tl)) if (tw + tl) > 0 else 0.0

    print("AGGREGATE")
    print(
        f"days={len(x)} picks={total_picks} W-L-P={tw}-{tl}-{tp} "
        f"win_rate={wr:.3f} units={tu:+.3f}"
    )

    print("\nMARKETS")
    rows = []
    for m, g in p.groupby("market"):
        w = int(g._W.sum())
        l = int(g._L.sum())
        pu = int(g._P.sum())
        wrm = (w / (w + l)) if (w + l) > 0 else None
        rows.append((m, len(g), w, l, pu, wrm))

    rows.sort(key=lambda t: t[1], reverse=True)
    for m, n, w, l, pu, wrm in rows:
        wrs = f"{wrm:.3f}" if wrm is not None else "n/a"
        print(f"{m}: n={n} W-L-P={w}-{l}-{pu} win_rate={wrs}")

    print("\nDAILY (date picks W-L-P win_rate units)")
    for _, r in x.iterrows():
        w = int(r.wins)
        l = int(r.losses)
        pu = int(r.pushes)
        picks = int(r.picks)
        units = float(r.units)
        wrd = (w / (w + l)) if (w + l) > 0 else 0.0
        print(f"{r.date} {picks:>2} {w}-{l}-{pu} {wrd:.3f} {units:+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
