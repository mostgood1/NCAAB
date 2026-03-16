import os
import sys
from pathlib import Path

import pandas as pd


def _norm_gid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _first_nonblank(series: pd.Series):
    for value in series.tolist():
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return value
    return None


def _median_numeric(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    try:
        return float(vals.median())
    except Exception:
        return None


def _aggregate_edges(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    if df.empty or "game_id" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["game_id"] = _norm_gid(work["game_id"])
    work = work[work["game_id"].ne("")].copy()
    if work.empty:
        return work

    if "period" in work.columns:
        try:
            period = work["period"].astype(str).str.lower().str.strip()
            full_mask = period.isin(["full_game", "full", "game", "fg"])
            if full_mask.any():
                work = work[full_mask].copy()
        except Exception:
            pass

    if "date" not in work.columns:
        for cand in ("display_date", "date_game"):
            if cand in work.columns:
                work["date"] = work[cand]
                break
    if "date" in work.columns:
        try:
            work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            work = work[work["date"].astype(str) == str(date_str)].copy()
        except Exception:
            pass

    if "market_total" not in work.columns:
        for cand in ("total", "closing_total"):
            if cand in work.columns:
                work["market_total"] = work[cand]
                break

    agg: dict[str, object] = {}
    for col in ("date", "home_team", "away_team", "display_date", "start_time", "start_time_local", "start_time_iso"):
        if col in work.columns:
            agg[col] = _first_nonblank
    for col in ("pred_total", "pred_margin", "market_total"):
        if col in work.columns:
            agg[col] = _median_numeric

    grouped = work.groupby("game_id", as_index=False).agg(agg)
    if "date" not in grouped.columns:
        grouped["date"] = date_str
    if "display_date" not in grouped.columns:
        grouped["display_date"] = grouped.get("date", date_str)
    return grouped


def _load_predictions(pred_path: Path, date_str: str) -> pd.DataFrame:
    if not pred_path.exists():
        return pd.DataFrame()
    preds = pd.read_csv(pred_path, low_memory=False)
    if preds.empty or "game_id" not in preds.columns:
        return pd.DataFrame()
    preds = preds.copy()
    preds["game_id"] = _norm_gid(preds["game_id"])
    keep = [
        c
        for c in [
            "game_id",
            "date",
            "home_team",
            "away_team",
            "pred_total",
            "pred_margin",
            "display_date",
            "start_time",
            "start_time_local",
            "start_time_iso",
        ]
        if c in preds.columns
    ]
    out = preds[keep].drop_duplicates(subset=["game_id"], keep="first").copy()
    if "date" in out.columns:
        try:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            out = out[out["date"].astype(str) == str(date_str)].copy()
        except Exception:
            pass
    return out


def _load_games(games_path: Path, date_str: str) -> pd.DataFrame:
    if not games_path.exists():
        return pd.DataFrame()
    games = pd.read_csv(games_path, low_memory=False)
    if games.empty or "game_id" not in games.columns:
        return pd.DataFrame()
    games = games.copy()
    games["game_id"] = _norm_gid(games["game_id"])
    keep = [
        c
        for c in [
            "game_id",
            "date",
            "home_team",
            "away_team",
            "display_date",
            "start_time",
            "start_time_local",
            "start_time_iso",
            "tournament_label",
            "tournament_note",
        ]
        if c in games.columns
    ]
    out = games[keep].drop_duplicates(subset=["game_id"], keep="first").copy()
    if "date" in out.columns:
        try:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            out = out[out["date"].astype(str) == str(date_str)].copy()
        except Exception:
            pass
    return out


def build_display_frame(base_dir: Path, date_str: str) -> pd.DataFrame:
    edges_path = base_dir / f"align_period_{date_str}_edges.csv"
    pred_path = base_dir / f"predictions_{date_str}.csv"
    games_path = base_dir / f"games_{date_str}.csv"

    if not edges_path.exists() and not pred_path.exists() and not games_path.exists():
        return pd.DataFrame()

    edges = pd.read_csv(edges_path, low_memory=False) if edges_path.exists() else pd.DataFrame()
    edge_agg = _aggregate_edges(edges, date_str)
    preds = _load_predictions(pred_path, date_str)
    games = _load_games(games_path, date_str)

    if not preds.empty:
        out_df = preds.copy()
        if not edge_agg.empty:
            out_df = out_df.merge(edge_agg, on="game_id", how="left", suffixes=("", "_edge"))
            for col in ("date", "home_team", "away_team", "pred_total", "pred_margin", "display_date", "start_time", "start_time_local", "start_time_iso", "market_total"):
                edge_col = f"{col}_edge"
                if edge_col not in out_df.columns:
                    continue
                if col not in out_df.columns:
                    out_df[col] = out_df[edge_col]
                    continue
                try:
                    base = out_df[col]
                    if base.dtype == object:
                        miss = base.isna() | base.astype(str).str.strip().eq("") | base.astype(str).str.strip().str.lower().isin(["nan", "none", "null"])
                    else:
                        miss = base.isna()
                    if miss.any():
                        out_df.loc[miss, col] = out_df.loc[miss, edge_col]
                except Exception:
                    pass
            out_df = out_df.drop(columns=[c for c in out_df.columns if c.endswith("_edge")], errors="ignore")
    else:
        out_df = edge_agg.copy()

    if not out_df.empty and not games.empty:
        out_df = out_df.merge(games, on="game_id", how="left", suffixes=("", "_game"))
        for col in ("date", "home_team", "away_team", "display_date", "start_time", "start_time_local", "start_time_iso", "tournament_label", "tournament_note"):
            game_col = f"{col}_game"
            if game_col not in out_df.columns:
                continue
            if col not in out_df.columns:
                out_df[col] = out_df[game_col]
                continue
            try:
                base = out_df[col]
                if base.dtype == object:
                    miss = base.isna() | base.astype(str).str.strip().eq("") | base.astype(str).str.strip().str.lower().isin(["nan", "none", "null"])
                else:
                    miss = base.isna()
                if miss.any():
                    out_df.loc[miss, col] = out_df.loc[miss, game_col]
            except Exception:
                pass
        out_df = out_df.drop(columns=[c for c in out_df.columns if c.endswith("_game")], errors="ignore")

    if out_df.empty:
        return out_df

    if "date" not in out_df.columns:
        out_df["date"] = date_str
    else:
        try:
            out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            out_df.loc[out_df["date"].isna(), "date"] = date_str
        except Exception:
            pass
    if "display_date" not in out_df.columns:
        out_df["display_date"] = out_df["date"]

    try:
        out_df = out_df.drop_duplicates(subset=["game_id"], keep="first")
    except Exception:
        pass

    try:
        sort_col = next((c for c in ("start_time_iso", "start_time") if c in out_df.columns), None)
        if sort_col:
            order = pd.to_datetime(out_df[sort_col], errors="coerce", utc=True)
            out_df = out_df.assign(_sort_ts=order).sort_values(["_sort_ts", "game_id"], ascending=[True, True], na_position="last")
            out_df = out_df.drop(columns=["_sort_ts"], errors="ignore")
    except Exception:
        pass

    order = [
        "game_id",
        "date",
        "home_team",
        "away_team",
        "tournament_label",
        "tournament_note",
        "pred_total",
        "pred_margin",
        "market_total",
        "display_date",
        "start_time",
        "start_time_local",
        "start_time_iso",
    ]
    keep = [c for c in order if c in out_df.columns]
    return out_df[keep].reset_index(drop=True)


def main(date_str: str) -> int:
    base = Path(__file__).resolve().parents[1] / "outputs"
    out = base / f"predictions_display_{date_str}.csv"
    out_df = build_display_frame(base, date_str)
    if out_df.empty:
        print(f"Missing display sources for {date_str}")
        return 2
    out_df.to_csv(out, index=False)
    print(out)
    return 0

if __name__ == "__main__":
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    if not date_str:
        print("Usage: generate_display_from_edges.py YYYY-MM-DD", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(main(date_str))
