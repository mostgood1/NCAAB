from __future__ import annotations

from pathlib import Path

import pandas as pd


def _summarize(label: str, df: pd.DataFrame) -> dict:
    df = df.copy()
    df["profit_units"] = pd.to_numeric(df.get("profit_units"), errors="coerce")
    df["result"] = pd.to_numeric(df.get("result"), errors="coerce")
    df = df[df["profit_units"].notna()].copy()

    n = int(len(df))
    if n <= 0:
        return {"label": label, "n": 0}

    wins = int((df["result"] == 1.0).sum())
    losses = int((df["result"] == 0.0).sum())
    pushes = int((df["result"] == 0.5).sum())
    denom = wins + losses
    win_rate = (wins / denom) if denom > 0 else None
    roi = float(df["profit_units"].sum() / n)
    return {
        "label": label,
        "n": n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "roi_units_per_bet": roi,
    }


def _print_row(s: dict) -> None:
    if s.get("n", 0) == 0:
        print(f"{s['label']:<28s} n=0")
        return
    wr = s.get("win_rate")
    wr_s = f"{wr:.3f}" if isinstance(wr, (int, float)) else "None"
    roi = float(s.get("roi_units_per_bet") or 0.0)
    print(
        f"{s['label']:<28s} n={int(s['n']):4d} "
        f"w={int(s['wins']):3d} l={int(s['losses']):3d} "
        f"wr={wr_s:>5s} roi={roi:+.4f}"
    )


def main() -> None:
    # Default window aligns to the recent rollup run.
    dates = [
        "2026-02-14",
        "2026-02-15",
        "2026-02-16",
        "2026-02-17",
        "2026-02-18",
        "2026-02-19",
        "2026-02-20",
    ]

    paths: list[Path] = []
    for d in dates:
        p = Path("outputs") / f"live_lens_ats_side_accuracy_{d}.csv"
        if p.exists():
            paths.append(p)

    if not paths:
        print("No inputs found.")
        return

    dfs: list[pd.DataFrame] = []
    for p in paths:
        dfx = pd.read_csv(p)
        if dfx.empty:
            continue
        if "date" not in dfx.columns:
            dfx["date"] = p.stem.split("_")[-1]
        dfs.append(dfx)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print("loaded_rows", len(df), "dates", sorted(df["date"].unique().tolist()) if not df.empty else [])
    if df.empty:
        return

    # Normalize numeric fields
    df["edge"] = pd.to_numeric(df.get("edge"), errors="coerce")
    df["elapsed"] = pd.to_numeric(df.get("elapsed"), errors="coerce")

    _print_row(_summarize("BASE all", df))

    print("\nby side (base)")
    for side, g in df.groupby("side"):
        _print_row(_summarize(f"side={side}", g))

    print("\nwhat-if: drop away 1h/2h")
    _print_row(_summarize("drop_1h2h_away", df[~((df.side == "away") & (df.lens.isin(["1h", "2h"])))]))

    print("\naway edge threshold sweep")
    for t in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
        keep = (df.side != "away") | (df.edge >= float(t))
        _print_row(_summarize(f"away_edge>={t:.2f}", df[keep]))

    print("\n1H/2H away edge threshold sweep")
    for t in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
        keep = ~((df.side == "away") & (df.lens.isin(["1h", "2h"])) & (df.edge < float(t)))
        _print_row(_summarize(f"1h2h_away_edge>={t:.2f}", df[keep]))


if __name__ == "__main__":
    main()
