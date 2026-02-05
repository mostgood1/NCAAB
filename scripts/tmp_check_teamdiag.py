from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_team_map_from_sim_segments(dates: list[str]) -> pd.DataFrame:
    out_dir = Path("outputs")
    rows: list[pd.DataFrame] = []
    for date_iso in sorted(set(str(d) for d in dates)):
        seg_path = out_dir / f"sim_segments_{date_iso}.csv"
        if not seg_path.exists():
            continue
        try:
            seg = pd.read_csv(seg_path, usecols=lambda c: c in {"date", "game_id", "home_team", "away_team"})
        except Exception:
            continue
        if seg.empty or "game_id" not in seg.columns:
            continue
        if "date" not in seg.columns:
            seg["date"] = date_iso
        seg["date"] = seg["date"].astype(str)
        seg["game_id"] = seg["game_id"].astype(str)
        seg = seg.drop_duplicates(subset=["date", "game_id"])
        rows.append(seg[["date", "game_id", "home_team", "away_team"]])
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _load_features_from_daily_files(dates: list[str]) -> pd.DataFrame:
    out_dir = Path("outputs")
    wanted = {"date", "game_id", "home_team", "away_team"}
    parts: list[pd.DataFrame] = []
    for date_iso in sorted(set(str(d) for d in dates)):
        cand = [
            out_dir / f"features_{date_iso}_augmented.csv",
            out_dir / f"features_{date_iso}.csv",
        ]
        path = next((p for p in cand if p.exists()), None)
        if path is None:
            continue
        try:
            f = pd.read_csv(path, usecols=lambda c: c in wanted)
        except Exception:
            continue
        if f.empty or "game_id" not in f.columns:
            continue
        if "date" not in f.columns:
            f["date"] = date_iso
        f["date"] = f["date"].astype(str)
        f["game_id"] = f["game_id"].astype(str)
        f = f.drop_duplicates(subset=["date", "game_id"])
        parts.append(f)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def main() -> None:
    master = Path("outputs/backtests/segments_5min_master.csv")
    feat = Path("outputs/features_all.csv")

    df = pd.read_csv(master)
    df["date"] = df["date"].astype(str)
    df["game_id"] = df["game_id"].astype(str)

    f = pd.read_csv(feat, usecols=lambda c: c in {"date", "game_id", "home_team", "away_team"})
    f["date"] = f["date"].astype(str)
    f["game_id"] = f["game_id"].astype(str)

    merged = df.merge(f, on=["date", "game_id"], how="left")
    print("master rows", len(df))
    print(
        "merged home_team non-null",
        float(merged["home_team"].notna().mean()),
        "away_team non-null",
        float(merged["away_team"].notna().mean()),
    )
    print("sample home teams", merged["home_team"].dropna().astype(str).head(5).tolist())

    dates = df["date"].astype(str).unique().tolist()
    tm = _load_team_map_from_sim_segments(dates)
    print("sim_segments team map rows", len(tm))
    if not tm.empty:
        m2 = df.merge(tm, on=["date", "game_id"], how="left")
        print(
            "sim_segments home_team non-null",
            float(m2["home_team"].notna().mean()),
            "away_team non-null",
            float(m2["away_team"].notna().mean()),
        )

    fx = _load_features_from_daily_files(dates)
    print("daily features rows", len(fx))
    if not fx.empty:
        m3 = df.merge(fx, on=["date", "game_id"], how="left")
        print(
            "daily features home_team non-null",
            float(m3["home_team"].notna().mean()),
            "away_team non-null",
            float(m3["away_team"].notna().mean()),
        )

    base_cols = [c for c in ("date", "game_id", "end_min", "err_q50", "abs_err_q50") if c in merged.columns]
    home = merged[base_cols + ["home_team"]].copy().rename(columns={"home_team": "team"})
    home["side"] = "home"
    away = merged[base_cols + ["away_team"]].copy().rename(columns={"away_team": "team"})
    away["side"] = "away"
    long_df = pd.concat([home, away], ignore_index=True)

    long_df["team"] = long_df["team"].astype("string").str.strip()
    long_df = long_df[long_df["team"].notna() & (long_df["team"] != "")]
    long_df = long_df[~long_df["team"].str.lower().isin({"nan", "none", "<na>"})]

    print("long rows", len(long_df), "unique teams", int(long_df["team"].nunique()))
    print("top team counts")
    print(long_df["team"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
