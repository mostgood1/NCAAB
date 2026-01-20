from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _norm_gid(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.str.replace(r"\.0$", "", regex=True).str.strip()


def merge_boxscores(base_path: Path, new_path: Path, out_path: Path) -> dict:
    base = _read_csv(base_path) if base_path.exists() else pd.DataFrame()
    new = _read_csv(new_path) if new_path.exists() else pd.DataFrame()

    if new.empty and base.empty:
        return {"status": "empty", "out": str(out_path)}

    # Union columns
    cols = sorted(set(base.columns).union(set(new.columns)))
    if not base.empty:
        base = base.reindex(columns=cols)
    if not new.empty:
        new = new.reindex(columns=cols)

    df = pd.concat([base, new], ignore_index=True)

    if "game_id" in df.columns:
        df["game_id"] = _norm_gid(df["game_id"])
        # Prefer newer rows (keep last occurrence)
        df = df.drop_duplicates(subset=["game_id"], keep="last")
    elif {"home_team", "away_team", "date"}.issubset(df.columns):
        df["date"] = df["date"].astype(str)
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return {
        "status": "ok",
        "base_rows": int(len(base)) if not base.empty else 0,
        "new_rows": int(len(new)) if not new.empty else 0,
        "out_rows": int(len(df)),
        "out": str(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge boxscore CSVs into a single deduped artifact.")
    ap.add_argument("--base", required=True, help="Existing boxscores.csv path")
    ap.add_argument("--new", required=True, help="New boxscores CSV path to merge in")
    ap.add_argument("--out", required=True, help="Output path (often same as --base)")
    args = ap.parse_args()

    res = merge_boxscores(Path(args.base), Path(args.new), Path(args.out))
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
