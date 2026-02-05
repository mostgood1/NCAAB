from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MASTER = Path("outputs") / "backtests" / "segments_5min_master.csv"
DEFAULT_OUT_DIR = Path("outputs") / "backtests" / "diagnostics"
FEATURES_HIST_AUG = Path("outputs") / "features_hist_augmented.csv"
FEATURES_HIST = Path("outputs") / "features_hist.csv"
FEATURES_ALL = Path("outputs") / "features_all.csv"
GAMES_ALL = Path("outputs") / "games_all.csv"
OUTPUTS_DIR = Path("outputs")


def _safe_mean(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _safe_count(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").dropna().shape[0])


def _summarize_frame(df: pd.DataFrame, *, prefix: str = "") -> dict:
    out: dict = {}

    # Prefer total columns (current schema)
    err = "err_q50" if "err_q50" in df.columns else None
    abs_err = "abs_err_q50" if "abs_err_q50" in df.columns else None

    if err is not None:
        out[prefix + "bias_q50"] = _safe_mean(df[err])
        out[prefix + "n"] = _safe_count(df[err])
    if abs_err is not None:
        out[prefix + "mae_q50"] = _safe_mean(df[abs_err])

    for col in ("pinball_q10", "pinball_q50", "pinball_q90"):
        if col in df.columns:
            out[prefix + col] = _safe_mean(df[col])

    return out


def _maybe_join_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join in home/away team and a handful of numeric features if available.

    We prefer outputs/features_all.csv (has tempo/ratings) and fall back to outputs/games_all.csv
    (teams only) if needed.
    """

    meta: dict = {
        "joined": False,
        "source": None,
        "rows": int(len(df)),
        "matched_rows": 0,
    }

    required_keys = [c for c in ("date", "game_id") if c in df.columns]
    if required_keys != ["date", "game_id"]:
        return df, meta

    if "home_team" in df.columns and "away_team" in df.columns:
        # Already has teams; no need to join.
        return df, meta

    left = df.copy()
    left["date"] = left["date"].astype(str)
    left["game_id"] = left["game_id"].astype(str)

    # Feature sources in priority order. We accept a merge if it matches at least one row.
    feature_sources: list[Path] = [FEATURES_HIST_AUG, FEATURES_HIST, FEATURES_ALL]
    for src in feature_sources:
        if not src.exists():
            continue
        try:
            f = pd.read_csv(src)
            if f.empty or "game_id" not in f.columns or "date" not in f.columns:
                continue
            f["date"] = f["date"].astype(str)
            f["game_id"] = f["game_id"].astype(str)
            merged = left.merge(f, on=["date", "game_id"], how="left", suffixes=("", "_feat"))
            matched = int(pd.notna(merged.get("home_team")).sum()) if "home_team" in merged.columns else 0
            if matched <= 0:
                continue

            # Backfill missing team strings from games_all if possible.
            if GAMES_ALL.exists() and ("home_team" in merged.columns and merged["home_team"].isna().any()):
                try:
                    g = pd.read_csv(GAMES_ALL, usecols=lambda c: c in {"game_id", "date", "home_team", "away_team", "neutral_site"})
                    if not g.empty and "game_id" in g.columns and "date" in g.columns:
                        g["date"] = g["date"].astype(str)
                        g["game_id"] = g["game_id"].astype(str)
                        merged2 = merged.merge(g, on=["date", "game_id"], how="left", suffixes=("", "_game"))
                        for col in ("home_team", "away_team", "neutral_site"):
                            game_col = f"{col}_game"
                            if col in merged2.columns and game_col in merged2.columns:
                                merged2[col] = merged2[col].where(pd.notna(merged2[col]), merged2[game_col])
                                merged2 = merged2.drop(columns=[game_col])
                        merged = merged2
                except Exception:
                    pass

            meta.update(
                {
                    "joined": True,
                    "source": str(src),
                    "matched_rows": int(pd.notna(merged.get("home_team")).sum()) if "home_team" in merged.columns else matched,
                }
            )
            return merged, meta
        except Exception:
            continue

    # Fall back to games_all for team strings
    if GAMES_ALL.exists():
        try:
            g = pd.read_csv(GAMES_ALL, usecols=lambda c: c in {"game_id", "date", "home_team", "away_team", "neutral_site"})
            if not g.empty and "game_id" in g.columns and "date" in g.columns:
                g["date"] = g["date"].astype(str)
                g["game_id"] = g["game_id"].astype(str)
                merged = left.merge(g, on=["date", "game_id"], how="left", suffixes=("", "_game"))
                matched = int(pd.notna(merged.get("home_team")).sum()) if "home_team" in merged.columns else 0
                if matched > 0:
                    meta.update(
                        {
                            "joined": True,
                            "source": str(GAMES_ALL),
                            "matched_rows": matched,
                        }
                    )
                    return merged, meta
        except Exception:
            pass

    return df, meta


def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 50:
        return None
    try:
        return float(np.corrcoef(x[m].to_numpy(dtype=float), y[m].to_numpy(dtype=float))[0, 1])
    except Exception:
        return None


def _load_team_map_from_sim_segments(dates: Iterable[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for date_iso in sorted(set(str(d) for d in dates)):
        seg_path = OUTPUTS_DIR / f"sim_segments_{date_iso}.csv"
        if not seg_path.exists():
            continue
        try:
            seg = pd.read_csv(seg_path, usecols=lambda c: c in {"date", "game_id", "home_team", "away_team"})
        except Exception:
            continue
        if seg.empty or "game_id" not in seg.columns:
            continue
        seg["date"] = seg.get("date", date_iso).astype(str)
        seg["game_id"] = seg["game_id"].astype(str)
        seg = seg.drop_duplicates(subset=["date", "game_id"])
        rows.append(seg[["date", "game_id", "home_team", "away_team"]])
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _load_features_from_daily_files(dates: Iterable[str]) -> pd.DataFrame:
    """Load and concat per-date features files for the requested dates.

    This is the most reliable way to get features for the current season without
    relying on long-window aggregate CSVs.
    """

    wanted = {
        "game_id",
        "date",
        "home_team",
        "away_team",
        "neutral_site",
        "home_off_rating",
        "away_off_rating",
        "home_def_rating",
        "away_def_rating",
        "home_tempo_rating",
        "away_tempo_rating",
        "tempo_rating_sum",

        # Current per-date schema (pace/PPP estimates)
        "pace_game_est",
        "possessions_game_est",
        "pace_sigma_game_est",
        "home_pace_mu",
        "away_pace_mu",
        "home_pace_sigma",
        "away_pace_sigma",
        "home_ppp_mu",
        "away_ppp_mu",
        "home_ppp_sigma",
        "away_ppp_sigma",
        "home_ppp_allowed_mu",
        "away_ppp_allowed_mu",

        # Rest flags
        "rest_home",
        "rest_away",
        "b2b_home",
        "b2b_away",
    }

    parts: list[pd.DataFrame] = []
    for date_iso in sorted(set(str(d) for d in dates)):
        cand = [
            OUTPUTS_DIR / f"features_{date_iso}_augmented.csv",
            OUTPUTS_DIR / f"features_{date_iso}.csv",
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
    ap = argparse.ArgumentParser(description="Analyze segments_5min master backtest CSV")
    ap.add_argument("--master", default=str(DEFAULT_MASTER), help="Path to master CSV")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for diagnostics")
    ap.add_argument("--min-n", type=int, default=200, help="Minimum rows per group to include")
    ap.add_argument("--min-n-team", type=int, default=10, help="Minimum rows per team/end_min/side to include")
    args = ap.parse_args()

    master_csv = Path(args.master)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not master_csv.exists():
        raise SystemExit(f"Master CSV not found: {master_csv}")

    df = pd.read_csv(master_csv)
    if df.empty:
        raise SystemExit(f"Master CSV is empty: {master_csv}")

    for c in ("date", "game_id"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "end_min" in df.columns:
        df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce")

    # Optionally enrich with team + feature columns (enables by-team / feature-driven diagnostics)
    df, join_meta = _maybe_join_features(df)

    # Backfill teams from sim_segments_<date>.csv if still missing.
    if ("home_team" not in df.columns or df["home_team"].isna().any()) and "date" in df.columns and "game_id" in df.columns:
        tm = _load_team_map_from_sim_segments(df["date"].astype(str).unique().tolist())
        if not tm.empty:
            df = df.merge(tm, on=["date", "game_id"], how="left", suffixes=("", "_seg"))
            for col in ("home_team", "away_team"):
                seg_col = f"{col}_seg"
                if col in df.columns and seg_col in df.columns:
                    df[col] = df[col].where(pd.notna(df[col]), df[seg_col])
                    df = df.drop(columns=[seg_col])
                elif seg_col in df.columns and col not in df.columns:
                    df = df.rename(columns={seg_col: col})

    # Backfill richer feature columns from per-date features files when coverage is poor.
    if "date" in df.columns and "game_id" in df.columns:
        def _cov(col: str) -> float:
            if col not in df.columns:
                return 0.0
            try:
                return float(pd.notna(df[col]).mean())
            except Exception:
                return 0.0

        # Even if columns exist, the aggregate file can be sparse for this season window.
        # Trigger daily-feature backfill if teams/key features are mostly missing.
        need_fx = (
            _cov("home_team") < 0.50
            or _cov("away_team") < 0.50
            or _cov("tempo_rating_sum") < 0.50
            or _cov("pace_game_est") < 0.50
            or _cov("home_pace_mu") < 0.50
            or _cov("home_ppp_mu") < 0.50
        )

        if need_fx:
            fx = _load_features_from_daily_files(df["date"].astype(str).unique().tolist())
            if not fx.empty:
                df = df.merge(fx, on=["date", "game_id"], how="left", suffixes=("", "_fx"))
                # Coalesce any overlapping columns
                for col in fx.columns:
                    if col in ("date", "game_id"):
                        continue
                    fx_col = f"{col}_fx"
                    if col in df.columns and fx_col in df.columns:
                        df[col] = df[col].where(pd.notna(df[col]), df[fx_col])
                        df = df.drop(columns=[fx_col])
                    elif fx_col in df.columns and col not in df.columns:
                        df = df.rename(columns={fx_col: col})

    # Derived features (if available)
    if "home_pace5" in df.columns and "away_pace5" in df.columns and "pace_sum5" not in df.columns:
        df["pace_sum5"] = pd.to_numeric(df["home_pace5"], errors="coerce") + pd.to_numeric(df["away_pace5"], errors="coerce")

    if "home_off_rating" in df.columns and "away_off_rating" in df.columns:
        if "off_rating_sum" not in df.columns:
            df["off_rating_sum"] = pd.to_numeric(df["home_off_rating"], errors="coerce") + pd.to_numeric(df["away_off_rating"], errors="coerce")
        if "off_rating_diff" not in df.columns:
            df["off_rating_diff"] = pd.to_numeric(df["home_off_rating"], errors="coerce") - pd.to_numeric(df["away_off_rating"], errors="coerce")

    if "home_def_rating" in df.columns and "away_def_rating" in df.columns:
        if "def_rating_sum" not in df.columns:
            df["def_rating_sum"] = pd.to_numeric(df["home_def_rating"], errors="coerce") + pd.to_numeric(df["away_def_rating"], errors="coerce")
        if "def_rating_diff" not in df.columns:
            df["def_rating_diff"] = pd.to_numeric(df["home_def_rating"], errors="coerce") - pd.to_numeric(df["away_def_rating"], errors="coerce")

    if "home_pace_mu" in df.columns and "away_pace_mu" in df.columns:
        if "pace_mu_sum" not in df.columns:
            df["pace_mu_sum"] = pd.to_numeric(df["home_pace_mu"], errors="coerce") + pd.to_numeric(df["away_pace_mu"], errors="coerce")
        if "pace_mu_diff" not in df.columns:
            df["pace_mu_diff"] = pd.to_numeric(df["home_pace_mu"], errors="coerce") - pd.to_numeric(df["away_pace_mu"], errors="coerce")

    if "home_ppp_mu" in df.columns and "away_ppp_mu" in df.columns:
        if "ppp_mu_sum" not in df.columns:
            df["ppp_mu_sum"] = pd.to_numeric(df["home_ppp_mu"], errors="coerce") + pd.to_numeric(df["away_ppp_mu"], errors="coerce")
        if "ppp_mu_diff" not in df.columns:
            df["ppp_mu_diff"] = pd.to_numeric(df["home_ppp_mu"], errors="coerce") - pd.to_numeric(df["away_ppp_mu"], errors="coerce")

    if "home_ppp_allowed_mu" in df.columns and "away_ppp_allowed_mu" in df.columns:
        if "ppp_allowed_sum" not in df.columns:
            df["ppp_allowed_sum"] = pd.to_numeric(df["home_ppp_allowed_mu"], errors="coerce") + pd.to_numeric(df["away_ppp_allowed_mu"], errors="coerce")
        if "ppp_allowed_diff" not in df.columns:
            df["ppp_allowed_diff"] = pd.to_numeric(df["home_ppp_allowed_mu"], errors="coerce") - pd.to_numeric(df["away_ppp_allowed_mu"], errors="coerce")

    def _cov(col: str) -> float:
        if col not in df.columns:
            return 0.0
        try:
            return float(pd.notna(df[col]).mean())
        except Exception:
            return 0.0

    summary: dict = {
        "master_csv": str(master_csv),
        "rows": int(len(df)),
        "date_min": str(df["date"].min()) if "date" in df.columns else None,
        "date_max": str(df["date"].max()) if "date" in df.columns else None,
        "columns": list(df.columns),
        "feature_join": join_meta,
        "coverage": {
            "home_team": _cov("home_team"),
            "away_team": _cov("away_team"),
            "tempo_rating_sum": _cov("tempo_rating_sum"),
            "pace_game_est": _cov("pace_game_est"),
            "possessions_game_est": _cov("possessions_game_est"),
            "pace_mu_sum": _cov("pace_mu_sum"),
            "ppp_mu_sum": _cov("ppp_mu_sum"),
        },
    }

    summary.update(_summarize_frame(df))

    by_end_min: list[dict] = []
    if "end_min" in df.columns:
        for end_min, g in df.groupby("end_min"):
            d = {"end_min": int(end_min)}
            d.update(_summarize_frame(g))
            if d.get("n", 0) >= int(args.min_n):
                by_end_min.append(d)
        by_end_min.sort(key=lambda x: x["end_min"])

    summary["by_end_min"] = by_end_min

    # Empirical curve: average actual cumulative fraction of final points by endpoint.
    # This helps tune the simulator's time-weighting away from uniform increments.
    curve_rows: list[dict] = []
    derived_segment_weights: dict | None = None
    if all(c in df.columns for c in ("date", "game_id", "end_min", "actual_total")):
        finals = df[df["end_min"] == 40][["date", "game_id", "actual_total"]].rename(columns={"actual_total": "actual_final"})
        if not finals.empty:
            tmp = df.merge(finals, on=["date", "game_id"], how="left")
            tmp["actual_total"] = pd.to_numeric(tmp["actual_total"], errors="coerce")
            tmp["actual_final"] = pd.to_numeric(tmp["actual_final"], errors="coerce")
            tmp["frac_of_final"] = tmp["actual_total"] / tmp["actual_final"].where(tmp["actual_final"].notna() & (tmp["actual_final"] != 0))
            for end_min, g in tmp.groupby("end_min"):
                frac = pd.to_numeric(g["frac_of_final"], errors="coerce")
                frac = frac[(frac.notna()) & (frac >= 0) & (frac <= 2)]
                curve_rows.append(
                    {
                        "end_min": int(end_min),
                        "n": int(frac.shape[0]),
                        "mean_frac_of_final": float(frac.mean()) if not frac.empty else None,
                        "median_frac_of_final": float(frac.median()) if not frac.empty else None,
                    }
                )
            curve_rows.sort(key=lambda r: r["end_min"])

            # Derive recommended per-half 5-min segment weights (4 segments per half)
            # from the mean cumulative fraction curve.
            try:
                frac_map = {int(r["end_min"]): float(r["mean_frac_of_final"]) for r in curve_rows if r.get("mean_frac_of_final") is not None}
                if all(k in frac_map for k in (5, 10, 15, 20, 25, 30, 35, 40)):
                    f5, f10, f15, f20, f25, f30, f35, f40 = (frac_map[k] for k in (5, 10, 15, 20, 25, 30, 35, 40))
                    h1_total = max(1e-9, float(f20))
                    h2_total = max(1e-9, float(f40 - f20))
                    w1 = [max(0.0, f5 - 0.0), max(0.0, f10 - f5), max(0.0, f15 - f10), max(0.0, f20 - f15)]
                    w2 = [max(0.0, f25 - f20), max(0.0, f30 - f25), max(0.0, f35 - f30), max(0.0, f40 - f35)]
                    w1 = [float(x / h1_total) for x in w1]
                    w2 = [float(x / h2_total) for x in w2]
                    derived_segment_weights = {"half1": w1, "half2": w2, "source": "segments_5min_actual_fraction_curve"}
            except Exception:
                derived_segment_weights = None

    # Per-team stats: attribute total-score error to both teams (home/away) so we can
    # see systematic total-bias patterns by team + endpoint.
    team_rows: list[dict] = []
    if "home_team" in df.columns and "away_team" in df.columns and "err_q50" in df.columns:
        parts: list[pd.DataFrame] = []
        base_cols = [c for c in ("date", "game_id", "end_min", "err_q50", "abs_err_q50") if c in df.columns]
        home = df[base_cols + ["home_team"]].copy().rename(columns={"home_team": "team"})
        home["side"] = "home"
        away = df[base_cols + ["away_team"]].copy().rename(columns={"away_team": "team"})
        away["side"] = "away"
        parts.extend([home, away])

        long_df = pd.concat(parts, ignore_index=True, sort=False)

        # Add a combined view across home/away for more stable sample sizes.
        both_df = long_df.copy()
        both_df["side"] = "both"
        long_df = pd.concat([long_df, both_df], ignore_index=True, sort=False)

        # Keep missing values as missing (avoid turning NaN into literal "nan" strings).
        long_df["team"] = long_df["team"].astype("string")
        long_df["team"] = long_df["team"].str.strip()
        long_df = long_df[long_df["team"].notna() & (long_df["team"] != "")]
        # Extra guard: drop any accidental stringified nulls.
        long_df = long_df[~long_df["team"].str.lower().isin({"nan", "none", "<na>"})]

        if not long_df.empty:
            for (team, end_min, side), g in long_df.groupby(["team", "end_min", "side"], dropna=True):
                d = {"team": str(team), "end_min": int(end_min), "side": str(side)}
                d.update(_summarize_frame(g))
                if d.get("n", 0) >= int(args.min_n_team):
                    team_rows.append(d)
            team_rows.sort(key=lambda x: (x["end_min"], x.get("side", ""), -x.get("n", 0), abs(x.get("bias_q50") or 0.0)))

    summary["by_team_end_min"] = team_rows

    # Feature diagnostics (correlations + binned bias curves)
    feature_cols = [
        "tempo_rating_sum",
        "pace_game_est",
        "possessions_game_est",
        "pace_mu_sum",
        "pace_mu_diff",
        "ppp_mu_sum",
        "ppp_mu_diff",
        "ppp_allowed_sum",
        "ppp_allowed_diff",
        "off_rating_sum",
        "off_rating_diff",
        "def_rating_sum",
        "def_rating_diff",
        "neutral_site",
        "rest_home",
        "rest_away",
        "b2b_home",
        "b2b_away",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    corr_rows: list[dict] = []
    bin_rows: list[dict] = []
    if feature_cols and "end_min" in df.columns and "err_q50" in df.columns:
        # Correlations by endpoint
        for end_min, g in df.groupby("end_min"):
            for feat in feature_cols:
                corr = _safe_corr(g["err_q50"], g[feat])
                if corr is None:
                    continue
                corr_rows.append({"end_min": int(end_min), "feature": feat, "n": int(pd.to_numeric(g[feat], errors="coerce").dropna().shape[0]), "corr_err_q50": float(corr)})
        corr_rows.sort(key=lambda r: (r["end_min"], -abs(r["corr_err_q50"])) )

        # Binned bias curves (global bins for stability)
        for feat in feature_cols:
            vals = pd.to_numeric(df[feat], errors="coerce")
            try:
                bins = pd.qcut(vals, q=10, duplicates="drop")
            except Exception:
                continue
            tmp = df.copy()
            tmp["_bin"] = bins.astype(str)
            for (end_min, b), g in tmp.groupby(["end_min", "_bin"], dropna=True):
                d = {"feature": feat, "end_min": int(end_min), "bin": str(b)}
                d.update(_summarize_frame(g))
                if d.get("n", 0) >= int(args.min_n):
                    bin_rows.append(d)
        bin_rows.sort(key=lambda r: (r["feature"], r["end_min"], r["bin"]))

    summary["feature_cols"] = feature_cols
    summary["feature_corr_by_end_min"] = corr_rows

    out_json = out_dir / "segments_5min_master_diagnostics.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Also write a compact CSV for easy pivoting.
    if by_end_min:
        pd.DataFrame(by_end_min).to_csv(out_dir / "segments_5min_by_end_min.csv", index=False)

    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(out_dir / "segments_5min_actual_fraction_curve.csv", index=False)

    if derived_segment_weights is not None:
        (out_dir / "segment_weights_from_actual_curve.json").write_text(
            json.dumps(derived_segment_weights, indent=2, sort_keys=True), encoding="utf-8"
        )

    # Always overwrite these so stale files don't linger.
    team_cols = ["team", "end_min", "side", "bias_q50", "mae_q50", "n", "pinball_q10", "pinball_q50", "pinball_q90"]
    pd.DataFrame(team_rows, columns=team_cols).to_csv(out_dir / "segments_5min_by_team_end_min.csv", index=False)

    corr_cols = ["end_min", "feature", "n", "corr_err_q50"]
    pd.DataFrame(corr_rows, columns=corr_cols).to_csv(out_dir / "segments_5min_feature_corr_by_end_min.csv", index=False)

    if bin_rows:
        pd.DataFrame(bin_rows).to_csv(out_dir / "segments_5min_feature_bins.csv", index=False)

    print(json.dumps({"status": "ok", "out_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
