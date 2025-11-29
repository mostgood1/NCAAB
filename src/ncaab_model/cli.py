from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import os
import datetime as dt
from zoneinfo import ZoneInfo
import platform
import struct
import numpy as np
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None
import pandas as pd
import typer
from rich import print
import time

from .onnx.export import create_dummy_regression_model
from .onnx.infer import OnnxPredictor, NumpyLinearPredictor
from .config import settings
from .data.adapters.ncaa_scoreboard import iter_games_by_date as iter_games_ncaa
from .data.adapters.espn_scoreboard import iter_games_by_date as iter_games_espn
from .data.adapters.odds_theoddsapi import TheOddsAPIAdapter
from .data.merge_odds import join_odds_to_games, normalize_name
from .data.join_closing import join_games_with_closing
from .data.join_closing import prepare_games_keys, prepare_closing_keys
from .data.adapters.espn_boxscore import iter_boxscores
from .features.build import build_team_rolling_features
from .features.schedule import compute_rest_days
from .train.baseline import train_baseline
from .train.distributional import train_distributional_totals, predict_distributional_totals
from .train.segmented import train_segmented
from .predict.segmented import score_segmented, blend_predictions
from .train.halves import train_half_models
from .eval.backtest import backtest_totals, backtest_totals_with_closing
from .data.odds_closing import make_closing_lines, make_last_odds
from .data.odds_closing import compute_closing_lines, compute_last_odds, compute_edges
from .data.odds_closing import read_directory_for_dates, load_snapshots
from .data.branding import fetch_espn_branding, write_branding_csv
from .eval.accuracy import compute_accuracy, compare_vs_closing
from .train.calibration import build_z_recenter_artifact, save_artifact, load_artifact
from .store.sqlite import connect as sqlite_connect, ingest_csv as sqlite_ingest

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _today_local() -> dt.date:
    """Return 'today' in the configured schedule timezone (defaults America/New_York).

    Render dynos run UTC; college basketball slates are anchored to US/Eastern. Using
    UTC early morning (00:00-04:59 UTC) can still reference the previous US date and
    undercount games. This helper mirrors the Flask app's _today_local logic.
    """
    tz_name = os.getenv("NCAAB_SCHEDULE_TZ", "America/New_York")
    try:
        tz = ZoneInfo(tz_name)
        return dt.datetime.now(tz).date()
    except Exception:
        return dt.date.today()

@app.command(name="backtest-walkforward")
def backtest_walkforward(
    start: str = typer.Option(None, help="Start date YYYY-MM-DD (defaults to 28 days ago)"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD (defaults to today)"),
    stake: float = typer.Option(1.0, help="Flat stake per bet in units"),
    price: float = typer.Option(1.91, help="Decimal price (approx -110)"),
    out_json: Path = typer.Option(settings.outputs_dir / "backtest_summary.json", help="Summary JSON output"),
    out_csv: Path = typer.Option(settings.outputs_dir / "backtest_daily.csv", help="Per-date CSV output"),
):
    """Run a simple walk-forward backtest with flat staking over a date range.

    Computes daily OU and ATS hit rates and a naive PnL using constant price.
    Requires predictions and results artifacts in outputs/.
    """
    tz_name = os.getenv("NCAAB_SCHEDULE_TZ", "America/New_York")
    tz = ZoneInfo(tz_name)
    today = dt.datetime.now(tz).date()
    if end is None:
        end = today.isoformat()
    if start is None:
        start = (today - dt.timedelta(days=28)).isoformat()
    # Load predictions and games
    pred_path = settings.outputs_dir / "predictions_all.csv"
    games_path = settings.outputs_dir / "games_all.csv"
    if not pred_path.exists() or not games_path.exists():
        print(f"[red]Missing outputs/predictions_all.csv or outputs/games_all.csv[/red]")
        raise typer.Exit(code=1)
    preds = pd.read_csv(pred_path)
    games = pd.read_csv(games_path)
    for c in ("game_id",):
        if c in preds.columns:
            preds[c] = preds[c].astype(str)
        if c in games.columns:
            games[c] = games[c].astype(str)
    df = preds.merge(games[["game_id","date"]], on="game_id", how="left")
    # Load actuals (aggregate daily_results/results_*.csv if present)
    actuals = []
    for p in os.listdir(settings.outputs_dir):
        if p.startswith("results_") and p.endswith(".csv"):
            try:
                f = pd.read_csv(settings.outputs_dir / p)
                if "game_id" in f.columns:
                    f["game_id"] = f["game_id"].astype(str)
                actuals.append(f)
            except Exception:
                pass
    if not actuals:
        print("[yellow]No results_*.csv found; finalize may be needed[/yellow]")
        raise typer.Exit(code=0)
    act = pd.concat(actuals, ignore_index=True)
    df = df.merge(act[["game_id","ats_result","actual_total","market_total","closing_total"]], on="game_id", how="left")
    # Filter date range
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        d0 = pd.to_datetime(start)
        d1 = pd.to_datetime(end)
        df = df[(df["date"]>=d0) & (df["date"]<=d1)]
    except Exception:
        pass
    if df.empty:
        print("[yellow]No rows in date range[/yellow]")
        raise typer.Exit(code=0)
    # Compute OU outcome vs market or closing
    def ou_outcome(row):
        mt = row["market_total"] if pd.notna(row.get("market_total")) else row.get("closing_total")
        if pd.isna(mt) or pd.isna(row.get("actual_total")):
            return None
        at = float(row["actual_total"]) ; mt = float(mt)
        if at>mt: return "Over"
        if at<mt: return "Under"
        return "Push"
    df["ou_actual"] = df.apply(ou_outcome, axis=1)
    # Simple MAE for totals when pred_total present
    if "pred_total" in df.columns and "actual_total" in df.columns:
        try:
            df["mae_total"] = (pd.to_numeric(df["pred_total"], errors="coerce") - pd.to_numeric(df["actual_total"], errors="coerce")).abs()
        except Exception:
            df["mae_total"] = pd.NA
    # Daily aggregation
    rows = []
    dec_price = float(price)
    win_return = stake * (dec_price - 1.0)
    lose_return = -stake
    for iso, grp in df.groupby(df["date"].dt.strftime("%Y-%m-%d")):
        n = len(grp)
        # OU hits where we have a lean
        ou_mask = grp["lean_ou_side"].notna() & grp["ou_actual"].notna()
        ou_hits = (grp.loc[ou_mask, "lean_ou_side"] == grp.loc[ou_mask, "ou_actual"]).sum()
        ou_push = (grp.loc[ou_mask, "ou_actual"] == "Push").sum()
        ou_bets = int(ou_mask.sum())
        ou_pnl = (ou_hits * win_return) + ((ou_bets - ou_hits - ou_push) * lose_return)
        # ATS hits from ats_result text
        ats_text = grp["ats_result"].fillna("")
        ats_push = (ats_text == "Push").sum()
        ats_bets = int((ats_text != "").sum())
        ats_hits = int(ats_text.str.contains("Cover", case=False).sum())
        ats_pnl = (ats_hits * win_return) + ((ats_bets - ats_hits - ats_push) * lose_return)
        rows.append({
            "date": iso,
            "rows": int(n),
            "ou_bets": ou_bets, "ou_hits": int(ou_hits), "ou_push": int(ou_push), "ou_pnl": float(ou_pnl),
            "ats_bets": ats_bets, "ats_hits": int(ats_hits), "ats_push": int(ats_push), "ats_pnl": float(ats_pnl),
            "total_pnl": float(ou_pnl + ats_pnl),
            "mae_total_mean": float(grp["mae_total"].mean()) if "mae_total" in grp.columns else None
        })
    daily = pd.DataFrame(rows).sort_values("date")
    total_pnl = float(daily["total_pnl"].sum())
    total_bets = int(daily["ou_bets"].sum() + daily["ats_bets"].sum())
    hit_rate = float((daily["ou_hits"].sum() + daily["ats_hits"].sum()) / max(1, total_bets))

    # Optional probability-based metrics if available in predictions
    brier_ats = None
    brier_ou = None
    auc_ats = None
    auc_ou = None
    crps_total = None
    try:
        # Build arrays for ATS/OU probs and outcomes if present
        ats_prob = pd.to_numeric(df.get("ats_prob"), errors="coerce") if "ats_prob" in df.columns else None
        ou_prob = pd.to_numeric(df.get("ou_prob"), errors="coerce") if "ou_prob" in df.columns else None
        ats_hit = df.get("ats_result").fillna("").str.contains("Cover", case=False) if "ats_result" in df.columns else None
        ou_actual = df.get("ou_actual") if "ou_actual" in df.columns else None
        if ats_prob is not None and ats_hit is not None:
            pa = ats_prob.dropna()
            ya = ats_hit.loc[pa.index].astype(int)
            if len(pa) > 0:
                brier_ats = float(np.mean((pa.values - ya.values) ** 2))
                if roc_auc_score is not None and len(np.unique(ya.values)) > 1:
                    try:
                        auc_ats = float(roc_auc_score(ya.values, pa.values))
                    except Exception:
                        auc_ats = None
        if ou_prob is not None and ou_actual is not None:
            mask = ou_actual.isin(["Over","Under"]) & ou_prob.notna()
            po = ou_prob[mask]
            yo = (ou_actual[mask] == "Over").astype(int)
            if len(po) > 0:
                brier_ou = float(np.mean((po.values - yo.values) ** 2))
                if roc_auc_score is not None and len(np.unique(yo.values)) > 1:
                    try:
                        auc_ou = float(roc_auc_score(yo.values, po.values))
                    except Exception:
                        auc_ou = None
        # CRPS using normal approx if columns present
        if {"pred_total","actual_total","pred_total_sigma"}.issubset(df.columns):
            mu = pd.to_numeric(df["pred_total"], errors="coerce").values
            x = pd.to_numeric(df["actual_total"], errors="coerce").values
            sigma = pd.to_numeric(df["pred_total_sigma"], errors="coerce").values
            sigma = np.clip(sigma, 1e-6, None)
            try:
                from scipy.stats import norm
                z = (x - mu) / sigma
                crps_vals = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))
                crps_total = float(np.nanmean(crps_vals))
            except Exception:
                crps_total = None
    except Exception:
        pass
    # Aggregate metrics
    mae_total_overall = float(df["mae_total"].mean()) if "mae_total" in df.columns else None
    summary = {
        "start": start, "end": end, "stake": stake, "price": dec_price,
        "days": int(len(daily)), "total_bets": total_bets, "hit_rate": hit_rate,
        "total_pnl": total_pnl,
        "mae_total": mae_total_overall,
        "brier_ats": brier_ats,
        "brier_ou": brier_ou,
        "auc_ats": auc_ats,
        "auc_ou": auc_ou,
        "crps_total": crps_total,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[green]Backtest complete[/green] -> {out_json} | {out_csv}")

@app.command(name="write-risk-config")
def write_risk_config(
    daily_loss_cap: float = typer.Option(None, help="Daily loss cap in units (optional)"),
    kelly_cap: float = typer.Option(None, help="Kelly fraction cap, e.g., 0.2 for 20% (optional)"),
    exposure_cap_units: float = typer.Option(None, help="Per-market exposure cap in units (optional)"),
    out: Path = typer.Option(settings.outputs_dir / "risk_config.json", help="Output JSON"),
):
    """Write a simple risk config JSON consumed by /api/status for UI pills.

    Provide any combination of caps; unspecified fields will be omitted.
    """
    cfg = {}
    if daily_loss_cap is not None:
        cfg["daily_loss_cap"] = float(daily_loss_cap)
    if kelly_cap is not None:
        cfg["kelly_cap"] = float(kelly_cap)
    if exposure_cap_units is not None:
        cfg["exposure_cap_units"] = float(exposure_cap_units)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[green]Wrote risk config[/green] {out} -> {cfg}")

@app.command(name="seed-priors")
def seed_priors(
    features_csv: Path = typer.Argument(..., help="Features CSV to enrich (from build-features)"),
    priors_csv: Path = typer.Option(settings.outputs_dir / "priors.csv", help="Team priors CSV (team,rating_margin,off_rating,def_rating,tempo_rating)"),
    out: Path = typer.Option(settings.outputs_dir / "features_with_priors.csv", help="Output enriched features CSV"),
):
    """Seed rating and tempo priors into a sparse features file and add derived diffs/sums.

    This helps early-season slates where rolling stats are missing by injecting per-team priors so
    downstream models see differentiated feature vectors.
    """
    if not features_csv.exists():
        print(f"[red]Missing features file:[/red] {features_csv}")
        raise typer.Exit(code=1)
    if not priors_csv.exists():
        print(f"[red]Missing priors file:[/red] {priors_csv}")
        raise typer.Exit(code=1)
    try:
        feats = pd.read_csv(features_csv)
        pri = pd.read_csv(priors_csv)
        if "game_id" in feats.columns:
            feats["game_id"] = feats["game_id"].astype(str)
        if "team" not in pri.columns:
            print("[red]priors CSV missing 'team' column[/red]")
            raise typer.Exit(code=1)

        def _map_prior(series: pd.Series, key: str) -> pd.Series:
            if key not in pri.columns:
                return pd.Series([pd.NA] * len(series))
            m = dict(zip(pri["team"].astype(str), pri[key]))
            return series.astype(str).map(m)

        # Compute mapped columns
        hm = _map_prior(feats.get("home_team", pd.Series(dtype=str)), "rating_margin")
        am = _map_prior(feats.get("away_team", pd.Series(dtype=str)), "rating_margin")
        ho = _map_prior(feats.get("home_team", pd.Series(dtype=str)), "off_rating")
        ao = _map_prior(feats.get("away_team", pd.Series(dtype=str)), "off_rating")
        hd = _map_prior(feats.get("home_team", pd.Series(dtype=str)), "def_rating")
        ad = _map_prior(feats.get("away_team", pd.Series(dtype=str)), "def_rating")
        ht = _map_prior(feats.get("home_team", pd.Series(dtype=str)), "tempo_rating")
        at = _map_prior(feats.get("away_team", pd.Series(dtype=str)), "tempo_rating")

        def _fill_or_create(col: str, values: pd.Series):
            if values is None or values.empty:
                return
            if col in feats.columns:
                feats[col] = feats[col].where(pd.notna(feats[col]), values)
            else:
                feats[col] = values

        _fill_or_create("home_rating_margin", hm)
        _fill_or_create("away_rating_margin", am)
        _fill_or_create("home_off_rating", ho)
        _fill_or_create("away_off_rating", ao)
        _fill_or_create("home_def_rating", hd)
        _fill_or_create("away_def_rating", ad)
        _fill_or_create("home_tempo_rating", ht)
        _fill_or_create("away_tempo_rating", at)

        # Derived diffs/sums
        if {"home_rating_margin","away_rating_margin"}.issubset(feats.columns):
            feats["rating_margin_diff"] = pd.to_numeric(feats["home_rating_margin"], errors="coerce") - pd.to_numeric(feats["away_rating_margin"], errors="coerce")
        if {"home_off_rating","away_off_rating"}.issubset(feats.columns):
            feats["off_rating_diff"] = pd.to_numeric(feats["home_off_rating"], errors="coerce") - pd.to_numeric(feats["away_off_rating"], errors="coerce")
        if {"home_def_rating","away_def_rating"}.issubset(feats.columns):
            feats["def_rating_diff"] = pd.to_numeric(feats["home_def_rating"], errors="coerce") - pd.to_numeric(feats["away_def_rating"], errors="coerce")
        if {"home_tempo_rating","away_tempo_rating"}.issubset(feats.columns):
            feats["tempo_rating_sum"] = pd.to_numeric(feats["home_tempo_rating"], errors="coerce") + pd.to_numeric(feats["away_tempo_rating"], errors="coerce")

        out.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(out, index=False)
        print(f"[green]Wrote features with priors to[/green] {out} ({len(feats)} rows)")
    except Exception as e:
        print(f"[red]Failed seeding priors:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(name="predict-segmented")
def predict_segmented_cmd(
    features_csv: Path = typer.Argument(..., help="Features CSV for the scoring date (e.g., outputs/features_curr.csv)"),
    models_path: Path = typer.Option(settings.outputs_dir / "models_segmented" / "segmented_team_models.jsonl", help="Path to segmented_*_models.jsonl"),
    segment: str = typer.Option("team", help="Segmentation key used during training: team|conference"),
    out: Path = typer.Option(settings.outputs_dir / "predictions_segmented.csv", help="Output CSV with segmented predictions"),
):
    """Score features with segmented models and write per-game segmented predictions."""
    try:
        if models_path.is_dir():
            # Auto-detect jsonl file inside directory
            cand = sorted(models_path.glob("segmented_*_models.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cand:
                print(f"[red]No segmented models found in[/red] {models_path}")
                raise typer.Exit(code=1)
            models_path = cand[0]
        df = score_segmented(features_csv, models_path, segment=segment)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[green]Wrote segmented predictions[/green] {out} (rows={len(df)})")
    except Exception as e:
        print(f"[red]predict-segmented failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(name="blend-segmented")
def blend_segmented_cmd(
    base_predictions_csv: Path = typer.Argument(..., help="Baseline predictions CSV (contains pred_total, pred_margin, game_id)"),
    segmented_predictions_csv: Path = typer.Argument(..., help="Segmented predictions CSV from predict-segmented"),
    out: Path = typer.Option(settings.outputs_dir / "predictions_blend.csv", help="Output blended predictions CSV"),
    min_rows: int = typer.Option(25, help="Segments must have at least this many rows to get weight > 0"),
    max_weight: float = typer.Option(0.6, help="Maximum blend weight for segmented prediction"),
):
    """Blend baseline and segmented predictions with a data-driven weight capped by max_weight.

    Produces pred_total_blend and pred_margin_blend columns; preserves originals.
    """
    try:
        base_df = pd.read_csv(base_predictions_csv)
        seg_df = pd.read_csv(segmented_predictions_csv)
        m = blend_predictions(base_df, seg_df, min_rows=min_rows, max_weight=max_weight)
        out.parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(out, index=False)
        print(f"[green]Wrote blended predictions[/green] {out} (rows={len(m)})")
    except Exception as e:
        print(f"[red]blend-segmented failed:[/red] {e}")
        raise typer.Exit(code=1)
@app.command(name="coverage-report")
def coverage_report(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    games_csv: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Games master list"),
    merged_last_csv: Path = typer.Option(settings.outputs_dir / "games_with_last.csv", help="Joined games+last odds CSV"),
    out: Path = typer.Option(settings.outputs_dir / "coverage_report.csv", help="Output coverage metrics CSV"),
):
    """Compute per-day odds coverage metrics (eligible vs covered vs partial)."""
    try:
        sd = dt.date.fromisoformat(start)
        ed = dt.date.fromisoformat(end)
    except Exception:
        print("[red]Invalid start/end date format[/red]")
        raise typer.Exit(code=1)
    if ed < sd:
        print("[red]end must be >= start[/red]")
        raise typer.Exit(code=1)
    if not games_csv.exists() or not merged_last_csv.exists():
        print("[red]Required input CSVs missing[/red]")
        raise typer.Exit(code=1)
    g = pd.read_csv(games_csv)
    g["date"] = pd.to_datetime(g.get("date"), errors="coerce")
    # Start time completeness flags
    g["_missing_start_time"] = g.get("start_time").isna() | (g.get("start_time").astype(str).str.len() == 0)
    ml = pd.read_csv(merged_last_csv)
    # Robust date column for filtering: prefer date_game if present/valid, else fall back to 'date'
    # Some historical joins may have 'date_game' as None/strings; coerce both safely.
    if "date_game" in ml.columns:
        ml["date_game"] = pd.to_datetime(ml.get("date_game"), errors="coerce")
    # Build a unified datetime column used for per-day filtering
    ml_date_fallback = pd.to_datetime(ml.get("date"), errors="coerce") if "date" in ml.columns else pd.Series(pd.NaT, index=ml.index)
    ml["_date_filter"] = ml.get("date_game") if "date_game" in ml.columns else None
    if "_date_filter" not in ml.columns or ml["_date_filter"].isna().all():
        ml["_date_filter"] = ml_date_fallback
    else:
        # Where date_game is NaT, fill from 'date'
        ml["_date_filter"] = ml["_date_filter"].where(ml["_date_filter"].notna(), ml_date_fallback)
    try:
        from .data.merge_odds import normalize_name as _norm
        d1 = pd.read_csv(settings.data_dir / "d1_conferences.csv")
        d1set = set(d1['team'].astype(str).map(_norm))
        g['_home_ok'] = g['home_team'].astype(str).map(_norm).isin(d1set)
        g['_away_ok'] = g['away_team'].astype(str).map(_norm).isin(d1set)
        g['eligible'] = g['_home_ok'] & g['_away_ok']
    except Exception:
        g['eligible'] = True
    days = pd.date_range(sd, ed, freq='D')
    rows = []
    for d in days:
        gday = g[g['date'].dt.date == d.date()].copy()
        elig = gday[gday['eligible']]
        # Filter by the robust per-row date
        try:
            ml_day = ml[ml['_date_filter'].dt.date == d.date()].copy()
        except Exception:
            # In case of unexpected types, produce empty day slice rather than crash
            ml_day = ml.iloc[0:0].copy()
        if 'game_id' in ml_day.columns:
            if 'partial_pair' in ml_day.columns:
                covered_exact = ml_day[ml_day['partial_pair'] != True]['game_id'].nunique()
                covered_partial = ml_day[ml_day['partial_pair'] == True]['game_id'].nunique()
            else:
                covered_exact = ml_day['game_id'].nunique()
                covered_partial = 0
        else:
            covered_exact = 0
            covered_partial = 0
        missing_start = int(gday["_missing_start_time"].sum()) if not gday.empty else 0
        rows.append({
            'date': d.date().isoformat(),
            'games_total': len(gday),
            'games_eligible': len(elig),
            'covered_exact_games': covered_exact,
            'covered_partial_games': covered_partial,
            'covered_any_games': len(set(ml_day['game_id'].astype(str))) if 'game_id' in ml_day.columns else 0,
            'missing_start_time_games': missing_start,
        })
    df_out = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"[green]Wrote coverage report to[/green] {out}")


@app.command(name="ort-diagnostics")
def ort_diagnostics():
    """Print ONNX Runtime availability, providers, and QNN/DML environment on this system.

    Use this to verify ARM64 and QNN setup (QNNExecutionProvider) on Windows on Snapdragon.
    """
    info = {}
    # Platform/arch
    info["platform"] = platform.platform()
    info["machine"] = platform.machine()
    info["python"] = platform.python_version()
    info["bitness"] = 64 if struct.calcsize("P") * 8 == 64 else 32

    # Env related to ORT/QNN
    env_keys = [
        "NCAAB_QNN_SDK_DIR",
        "QNN_SDK_ROOT",
        "NCAAB_QNN_BACKEND_DLL",
        "NCAAB_ORT_DLL_DIR",
        "ONNXRUNTIME_DLL_DIR",
        "ONNXRUNTIME_DIR",
        "ORT_DLL_DIR",
        "NCAAB_ORT_PY_DIR",
        "ONNXRUNTIME_PY_DIR",
    ]
    env = {k: os.getenv(k) for k in env_keys}

    # Attempt to import onnxruntime and collect providers
    try:
        from .onnx import infer as infer_mod  # ensures dll dirs are added first
        ort = infer_mod.ort  # type: ignore[attr-defined]
    except Exception:
        ort = None

    providers_available = []
    ort_version = None
    if ort is not None:
        try:
            providers_available = list(ort.get_available_providers())
            ort_version = getattr(ort, "__version__", None)
        except Exception:
            providers_available = []

    # Determine preferred provider order that OnnxPredictor would use
    preferred = []
    if providers_available:
        if "QNNExecutionProvider" in providers_available:
            preferred.append("QNNExecutionProvider")
        if "DmlExecutionProvider" in providers_available:
            preferred.append("DmlExecutionProvider")
        preferred.append("CPUExecutionProvider")

    # QNN backend DLL resolution like infer._qnn_provider_options
    qnn_report = {
        "settings.qnn_sdk_dir": str(settings.qnn_sdk_dir) if settings.qnn_sdk_dir else None,
        "settings.qnn_backend_dll": str(settings.qnn_backend_dll) if settings.qnn_backend_dll else None,
        "env.NCAAB_QNN_BACKEND_DLL": env.get("NCAAB_QNN_BACKEND_DLL"),
        "resolved_backend": None,
        "exists": None,
    }
    try:
        from .onnx.infer import _qnn_provider_options  # type: ignore
        q = _qnn_provider_options()
        if q is not None:
            qnn_report["resolved_backend"] = q[1].get("backend_path")
            qnn_report["exists"] = os.path.exists(q[1].get("backend_path"))
    except Exception:
        pass

    # Print results
    print("[bold]System[/bold]", info)
    print("[bold]Env[/bold]", env)
    print("[bold]onnxruntime[/bold]", {"available": ort is not None, "version": ort_version})
    print("[bold]Available Providers[/bold]", providers_available)
    print("[bold]Preferred Order[/bold]", preferred)
    print("[bold]QNN Backend Resolution[/bold]", qnn_report)

    # Guidance summary
    if ort is None:
        print("[yellow]onnxruntime not importable. Set NCAAB_ORT_PY_DIR to the folder or wheel containing the module, and NCAAB_ORT_DLL_DIR to the DLL folder.[/yellow]")
    elif "QNNExecutionProvider" not in providers_available:
        print("[yellow]QNNExecutionProvider not available. Ensure your onnxruntime build includes QNN EP and QNN SDK paths are configured.[/yellow]")
    elif not qnn_report.get("exists"):
        print("[yellow]QNN backend DLL path not resolved. Set NCAAB_QNN_BACKEND_DLL or NCAAB_QNN_SDK_DIR to your SDK install.[/yellow]")
    else:
        print("[green]QNN appears configured. Inference should prefer QNN -> DML -> CPU.[/green]")


@app.command(name="ort-info")
def ort_info():
    """Backward-compatible alias for ort-diagnostics (used by enable_ort_qnn.ps1)."""
    ort_diagnostics()


@app.command(name="backfill-start-times")
def backfill_start_times(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    games_path: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Games CSV to update in-place"),
    provider: str = typer.Option("espn", help="Provider to fetch for backfill: espn|ncaa|fused"),
    use_cache: bool = typer.Option(True, help="Allow cached provider responses"),
):
    """Backfill missing start_time in games_all.csv by refetching scoreboard and updating rows.

    - Iterates dates in [start, end]
    - Fetches provider games for each date
    - Updates 'start_time' for matching game_id when missing or empty
    - Writes file in-place
    """
    if not games_path.exists():
        print(f"[red]Games file not found:[/red] {games_path}")
        raise typer.Exit(code=1)
    gdf = pd.read_csv(games_path)
    if "game_id" not in gdf.columns:
        print("[red]games file missing 'game_id' column[/red]")
        raise typer.Exit(code=1)
    if "start_time" not in gdf.columns:
        gdf["start_time"] = pd.NA
    gdf["game_id"] = gdf["game_id"].astype(str)
    # Normalize date col to string
    if "date" in gdf.columns:
        try:
            gdf["date"] = pd.to_datetime(gdf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            gdf["date"] = gdf["date"].astype(str)

    sd = pd.to_datetime(start, errors="coerce")
    ed = pd.to_datetime(end, errors="coerce")
    if pd.isna(sd) or pd.isna(ed) or sd > ed:
        print("[red]Invalid start/end date range[/red]")
        raise typer.Exit(code=1)
    dates = pd.date_range(sd, ed, freq="D")
    iterator = iter_games_espn if provider.lower() == "espn" else (iter_games_ncaa if provider.lower() == "ncaa" else iter_games_espn)
    n_updated = 0
    for d in dates:
        d_iso = d.date().isoformat()
        try:
            rows: list[dict] = []
            for res in iterator(d.date(), d.date(), use_cache=use_cache):
                for game in res.games:
                    rows.append(game.model_dump())
            if not rows:
                continue
            df = pd.DataFrame(rows)
            if df.empty or "game_id" not in df.columns or "start_time" not in df.columns:
                continue
            df["game_id"] = df["game_id"].astype(str)
            m = df.set_index("game_id")["start_time"].to_dict()
            mask_date = (gdf.get("date", pd.Series(dtype=str)).astype(str) == d_iso)
            need = mask_date & (gdf["start_time"].isna() | (gdf["start_time"].astype(str) == ""))
            if need.any():
                before = need.sum()
                gdf.loc[need, "start_time"] = gdf.loc[need, "game_id"].map(m)
                after = gdf.loc[mask_date & (gdf["start_time"].isna() | (gdf["start_time"].astype(str) == ""))].shape[0]
                n_updated += max(0, before - after)
                print(f"[cyan]{d_iso}[/cyan] backfilled {max(0, before - after)} start_time values")
        except Exception as e:
            print(f"[yellow]Backfill failed for {d_iso}:[/yellow] {e}")
            continue
    # Write back
    gdf.to_csv(games_path, index=False)
    print(f"[green]Backfill complete.[/green] Updated ~{n_updated} rows in {games_path}")

@app.command(name="backfill-display")
def backfill_display(
    start: str = typer.Argument(..., help="Start date (YYYY-MM-DD) inclusive"),
    end: str = typer.Argument(..., help="End date (YYYY-MM-DD) inclusive"),
    games_pattern: str = typer.Option("games_{}{}.csv", help="Pattern for per-date games files; '{}' placeholders will be replaced with date"),
    out_dir: Path = typer.Option(settings.outputs_dir, help="Directory containing dated games CSVs and where display CSVs will be written"),
    overwrite: bool = typer.Option(False, help="Overwrite existing games_display_<date>.csv if present"),
):
    """Backfill display-filtered games across a date span producing games_display_<date>.csv files.

    Uses the D1-any rule: retain games where at least one team is Division I per data/d1_conferences.csv.
    This retroactively fixes historical days after expanding the D1 list.
    """
    import datetime as _dt
    import pandas as _pd
    from .data.merge_odds import normalize_name as _norm

    d1_df = _pd.read_csv(settings.data_dir / "d1_conferences.csv")
    # Filter rows that look like real teams (reject any accidental conference placeholder rows where team==conference)
    d1_df = d1_df[d1_df['team'] != d1_df['conference']]
    d1_set = set(d1_df['team'].astype(str).map(_norm))

    d0 = _dt.date.fromisoformat(start)
    d1 = _dt.date.fromisoformat(end)
    if d1 < d0:
        raise typer.BadParameter("end date precedes start date")

    cur = d0
    results = []
    while cur <= d1:
        iso = cur.isoformat()
        games_file = out_dir / f"games_{iso}.csv"
        if not games_file.exists():
            results.append({"date": iso, "status": "missing"})
            cur += _dt.timedelta(days=1)
            continue
        try:
            g = _pd.read_csv(games_file)
        except Exception as e:
            results.append({"date": iso, "status": f"read_error:{e}"})
            cur += _dt.timedelta(days=1)
            continue
        if not {'home_team','away_team'}.issubset(g.columns):
            results.append({"date": iso, "status": "bad_schema"})
            cur += _dt.timedelta(days=1)
            continue
        g['_home_norm'] = g['home_team'].astype(str).map(_norm)
        g['_away_norm'] = g['away_team'].astype(str).map(_norm)
        mask = g['_home_norm'].isin(d1_set) | g['_away_norm'].isin(d1_set)
        g_disp = g[mask].copy()
        out_file = out_dir / f"games_display_{iso}.csv"
        if out_file.exists() and (not overwrite):
            status = "exists"
        else:
            g_disp.to_csv(out_file, index=False)
            status = "written"
        results.append({"date": iso, "status": status, "total": int(len(g)), "display": int(len(g_disp))})
        cur += _dt.timedelta(days=1)

    # Summary CSV
    summary = _pd.DataFrame(results)
    summary_path = out_dir / "display_backfill_summary.csv"
    summary.to_csv(summary_path, index=False)
    written = (summary['status'] == 'written').sum()
    typer.secho(f"Backfill complete: {written} display files written. Summary at {summary_path}", fg=typer.colors.GREEN)

@app.command(name="ort-env-hints")
def ort_env_hints(
    qnn_sdk_dir: str = typer.Option("C:/Qualcomm/QNN_SDK", help="Path to Qualcomm QNN SDK root (contains lib/)"),
    ort_dll_dir: str = typer.Option("", help="Directory containing onnxruntime DLLs (onnxruntime.dll, providers)", show_default=False),
    ort_py_dir: str = typer.Option("", help="Directory containing Python onnxruntime package or wheel contents", show_default=False),
):
    """Emit environment variable export suggestions for enabling ONNX Runtime + QNN EP.

    This does not validate the files explicitly (use ort-diagnostics after exporting).
    """
    exports = []
    if qnn_sdk_dir:
        # Try to pick an ARM64 backend DLL variant
        candidate_dirs = [
            Path(qnn_sdk_dir) / "lib" / "arm64x-windows-msvc",
            Path(qnn_sdk_dir) / "lib" / "aarch64-windows-msvc",
            Path(qnn_sdk_dir) / "lib" / "windows-aarch64",
        ]
        backend = None
        for d in candidate_dirs:
            f = d / "QnnHtp.dll"
            if f.exists():
                backend = f
                break
        if backend:
            exports.append(("ORT_QNN_BACKEND_PATH", str(backend)))
        exports.append(("NCAAB_QNN_SDK_DIR", qnn_sdk_dir))
    if ort_dll_dir:
        exports.append(("NCAAB_ORT_DLL_DIR", ort_dll_dir))
    if ort_py_dir:
        exports.append(("NCAAB_ORT_PY_DIR", ort_py_dir))
    # Provide PATH / PYTHONPATH hints
    path_hint = None
    if ort_dll_dir:
        path_hint = f"$env:PATH='{ort_dll_dir};' + $env:PATH"
    py_hint = None
    if ort_py_dir:
        py_hint = f"$env:PYTHONPATH='{ort_py_dir};' + $env:PYTHONPATH"
    print("[bold]Suggested Environment Exports[/bold]")
    for k, v in exports:
        print(f"$env:{k}='{v}'")
    if path_hint:
        print("[bold]PATH Append[/bold]", path_hint)
    if py_hint:
        print("[bold]PYTHONPATH Append[/bold]", py_hint)
    print("Run: .\\.venv\\Scripts\\python.exe -m ncaab_model.cli ort-diagnostics after exporting.")


@app.command(name="finalize-day")
def finalize_day(
    date: str = typer.Option(..., help="Target date YYYY-MM-DD"),
    provider: str = typer.Option("espn", help="Scoreboard provider espn|ncaa|fused"),
    games_csv: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Master games list (for metadata fallback)"),
    predictions_csv: Path = typer.Option(settings.outputs_dir / "predictions_week.csv", help="Predictions CSV (for pred_total, pred_margin)"),
    odds_csv: Path = typer.Option(settings.outputs_dir / "games_with_last.csv", help="Joined odds (for market/closing lines)"),
    boxscores_csv: Path = typer.Option(settings.outputs_dir / "boxscores_prev.csv", help="Optional boxscores CSV (prev-day) to backfill final scores if provider missing"),
    out_dir: Path = typer.Option(settings.outputs_dir / "daily_results", help="Output daily_results directory"),
    overwrite: bool = typer.Option(True, help="Overwrite existing results_<date>.csv if present"),
    include_halves: bool = typer.Option(True, help="Attempt to compute halftime + 2H derived actuals when partial scores available"),
    halftime_cutoff_min: int = typer.Option(45, help="Minutes after start to consider halftime safe if explicit 1H scores missing"),
    secondary_provider: str = typer.Option("ncaa", help="Fallback provider to try if primary yields zero scores"),
    use_cache: bool = typer.Option(True, help="Use cached provider responses when available (set false to force refresh)"),
    overrides_csv: Path | None = typer.Option(None, help="Optional scores override CSV (game_id,home_score,away_score) to enforce finals"),
):
    """Finalize a slate: fetch latest scores, merge predictions & odds, compute basic outcomes, write daily_results/results_<date>.csv.

    Intended to run after games finish. Safe to re-run; will overwrite unless disabled.
    """
    try:
        d_obj = dt.date.fromisoformat(date)
    except Exception:
        print(f"[red]Invalid date format[/red] {date}")
        raise typer.Exit(code=1)
    # Fetch provider games for the day
    iterator = iter_games_espn if provider.lower() == "espn" else (iter_games_ncaa if provider.lower() == "ncaa" else iter_games_espn)
    fetched_rows: list[dict[str, Any]] = []
    try:
        for res in iterator(d_obj, d_obj, use_cache=use_cache):
            for g in res.games:
                model = g.model_dump()
                # Ensure minimal keys
                fetched_rows.append({
                    "game_id": str(model.get("game_id")),
                    "date": date,
                    "start_time": model.get("start_time"),
                    "home_team": model.get("home_team"),
                    "away_team": model.get("away_team"),
                    "home_score": model.get("home_score"),
                    "away_score": model.get("away_score"),
                    "home_score_1h": model.get("home_score_1h"),
                    "away_score_1h": model.get("away_score_1h"),
                    "home_score_2h": model.get("home_score_2h"),
                    "away_score_2h": model.get("away_score_2h"),
                    "neutral_site": model.get("neutral_site"),
                    "venue": model.get("venue"),
                    "status": model.get("status"),
                })
    except Exception as e:
        print(f"[yellow]Provider fetch failed[/yellow]: {e}")
    games_df = pd.DataFrame(fetched_rows)
    if games_df.empty:
        print("[yellow]No games fetched from primary provider; attempting secondary fetch.[/yellow]")
    games_df["game_id"] = games_df["game_id"].astype(str)

    # If no or zero scores from primary, try secondary provider for score enrichment
    try_secondary = False
    if not games_df.empty:
        hs0 = pd.to_numeric(games_df.get("home_score"), errors="coerce")
        as0 = pd.to_numeric(games_df.get("away_score"), errors="coerce")
        if ((hs0.fillna(0) <= 0) & (as0.fillna(0) <= 0)).all():
            try_secondary = True
    else:
        try_secondary = True
    # If cache was used and we have zeros, try a forced refresh from primary before switching providers
    if try_secondary and use_cache:
        try:
            ref_rows: list[dict[str, Any]] = []
            for res in iterator(d_obj, d_obj, use_cache=False):
                for g in res.games:
                    m = g.model_dump()
                    ref_rows.append({
                        "game_id": str(m.get("game_id")),
                        "date": date,
                        "home_team": m.get("home_team"),
                        "away_team": m.get("away_team"),
                        "home_score": m.get("home_score"),
                        "away_score": m.get("away_score"),
                        "status": m.get("status"),
                        "start_time": m.get("start_time"),
                        "home_score_1h": m.get("home_score_1h"),
                        "away_score_1h": m.get("away_score_1h"),
                        "home_score_2h": m.get("home_score_2h"),
                        "away_score_2h": m.get("away_score_2h"),
                    })
            if ref_rows:
                ref_df = pd.DataFrame(ref_rows); ref_df["game_id"] = ref_df["game_id"].astype(str)
                if games_df.empty:
                    games_df = ref_df
                else:
                    merged = games_df.merge(ref_df[["game_id","home_score","away_score","status","home_score_1h","away_score_1h","home_score_2h","away_score_2h"]], on="game_id", how="left", suffixes=("","_ref"))
                    for side in ("home","away"):
                        cur = pd.to_numeric(merged.get(f"{side}_score"), errors="coerce")
                        rf = pd.to_numeric(merged.get(f"{side}_score_ref"), errors="coerce")
                        merged.loc[(cur.fillna(0) <= 0) & rf.notna() & (rf>0), f"{side}_score"] = rf[(cur.fillna(0) <= 0) & rf.notna() & (rf>0)]
                    # Fill 1H/2H from refresh where missing
                    for col in ["home_score_1h","away_score_1h","home_score_2h","away_score_2h"]:
                        if col in merged.columns and f"{col}_ref" in merged.columns:
                            merged[col] = merged[col].where(merged[col].notna(), merged[f"{col}_ref"])
                    if "status_ref" in merged.columns:
                        merged["status"] = merged["status"].where(merged["status"].notna(), merged["status_ref"])
                    games_df = merged.drop(columns=[c for c in merged.columns if c.endswith("_ref")])
        except Exception as e:
            print(f"[yellow]Primary refresh fetch failed[/yellow]: {e}")
    if try_secondary and secondary_provider and secondary_provider.lower() != provider.lower():
        try:
            sec_iter = iter_games_ncaa if secondary_provider.lower() == "ncaa" else iter_games_espn
            sec_rows: list[dict[str, Any]] = []
            for res in sec_iter(d_obj, d_obj, use_cache=False if use_cache else use_cache):
                for g in res.games:
                    m = g.model_dump()
                    sec_rows.append({
                        "game_id": str(m.get("game_id")),
                        "date": date,
                        "home_team": m.get("home_team"),
                        "away_team": m.get("away_team"),
                        "home_score": m.get("home_score"),
                        "away_score": m.get("away_score"),
                        "status": m.get("status"),
                        "home_score_1h": m.get("home_score_1h"),
                        "away_score_1h": m.get("away_score_1h"),
                        "home_score_2h": m.get("home_score_2h"),
                        "away_score_2h": m.get("away_score_2h"),
                    })
            if sec_rows:
                sec_df = pd.DataFrame(sec_rows)
                sec_df["game_id"] = sec_df["game_id"].astype(str)
                if games_df.empty:
                    games_df = sec_df
                else:
                    # Coalesce scores where primary has zeros; also fill halves when missing
                    merged = games_df.merge(sec_df[["game_id","home_score","away_score","status","home_score_1h","away_score_1h","home_score_2h","away_score_2h"]], on="game_id", how="left", suffixes=("","_sec"))
                    for side in ("home","away"):
                        cur = pd.to_numeric(merged.get(f"{side}_score"), errors="coerce")
                        secv = pd.to_numeric(merged.get(f"{side}_score_sec"), errors="coerce")
                        merged.loc[(cur.fillna(0) <= 0) & secv.notna() & (secv>0), f"{side}_score"] = secv[(cur.fillna(0) <= 0) & secv.notna() & (secv>0)]
                    for col in ["home_score_1h","away_score_1h","home_score_2h","away_score_2h"]:
                        if col in merged.columns and f"{col}_sec" in merged.columns:
                            merged[col] = merged[col].where(merged[col].notna(), merged[f"{col}_sec"])
                    if "status_sec" in merged.columns:
                        merged["status"] = merged["status"].where(merged["status"].notna(), merged["status_sec"])
                    games_df = merged.drop(columns=[c for c in merged.columns if c.endswith("_sec")])
        except Exception as e:
            print(f"[yellow]Secondary provider fetch failed[/yellow]: {e}")
    # If still zero scores, attempt a fused pass combining ESPN+NCAA (no-cache) keyed by date+normalized teams
    try:
        if not games_df.empty:
            hs = pd.to_numeric(games_df.get("home_score"), errors="coerce").fillna(0)
            as_ = pd.to_numeric(games_df.get("away_score"), errors="coerce").fillna(0)
            if ((hs <= 0) & (as_ <= 0)).all():
                espn_rows, ncaa_rows = [], []
                for res in iter_games_espn(d_obj, d_obj, use_cache=False):
                    for g in res.games:
                        d = g.model_dump(); d["source"] = "espn"; espn_rows.append(d)
                for res in iter_games_ncaa(d_obj, d_obj, use_cache=False):
                    for g in res.games:
                        d = g.model_dump(); d["source"] = "ncaa"; ncaa_rows.append(d)
                def _prep(df: pd.DataFrame) -> pd.DataFrame:
                    if df.empty: return df
                    for col in ["home_team","away_team"]:
                        if col in df.columns: df[col] = df[col].astype(str)
                    df["_date"] = date
                    df["_home_key"] = df.get("home_team", pd.Series(dtype=str)).astype(str).map(lambda x: _norm(x))
                    df["_away_key"] = df.get("away_team", pd.Series(dtype=str)).astype(str).map(lambda x: _norm(x))
                    df["_fuse_key"] = df["_date"] + "|" + df["_home_key"] + "|" + df["_away_key"]
                    score_cols = [c for c in ["home_score","away_score","home_score_1h","away_score_1h"] if c in df.columns]
                    df["_score_nonnull"] = df[score_cols].notna().sum(axis=1) if score_cols else 0
                    df["_has_start"] = df.get("start_time").notna() if "start_time" in df.columns else False
                    df["_quality"] = df["_has_start"].astype(int) * 10 + df["_score_nonnull"].astype(int)
                    return df
                e = _prep(pd.DataFrame(espn_rows)); n = _prep(pd.DataFrame(ncaa_rows))
                comb = pd.concat([e, n], ignore_index=True)
                if not comb.empty and "_fuse_key" in comb.columns:
                    best = comb.sort_values(["_fuse_key","_quality"], ascending=[True, False]).drop_duplicates(subset=["_fuse_key"], keep="first")
                    # Map back to games_df by normalized teams
                    tmp = games_df.copy()
                    tmp["_home_key"] = tmp.get("home_team", pd.Series(dtype=str)).astype(str).map(lambda x: _norm(x))
                    tmp["_away_key"] = tmp.get("away_team", pd.Series(dtype=str)).astype(str).map(lambda x: _norm(x))
                    tmp["_fuse_key"] = date + "|" + tmp["_home_key"] + "|" + tmp["_away_key"]
                    tmp = tmp.merge(best[["_fuse_key","home_score","away_score","status","home_score_1h","away_score_1h","home_score_2h","away_score_2h"]], on="_fuse_key", how="left", suffixes=("","_fused"))
                    for side in ("home","away"):
                        cur = pd.to_numeric(tmp.get(f"{side}_score"), errors="coerce")
                        fv = pd.to_numeric(tmp.get(f"{side}_score_fused"), errors="coerce")
                        tmp.loc[(cur.fillna(0) <= 0) & fv.notna() & (fv>0), f"{side}_score"] = fv[(cur.fillna(0) <= 0) & fv.notna() & (fv>0)]
                    for col in ["home_score_1h","away_score_1h","home_score_2h","away_score_2h"]:
                        if col in tmp.columns and f"{col}_fused" in tmp.columns:
                            tmp[col] = tmp[col].where(tmp[col].notna(), tmp[f"{col}_fused"])
                    if "status_fused" in tmp.columns:
                        tmp["status"] = tmp["status"].where(tmp["status"].notna(), tmp["status_fused"])
                    games_df = tmp.drop(columns=[c for c in tmp.columns if c.startswith("_fuse") or c.endswith("_fused")])
    except Exception as e:
        print(f"[yellow]Fused fallback failed[/yellow]: {e}")
    if games_df.empty:
        print("[yellow]Still no games after secondary fetch; aborting finalize-day.[/yellow]")
        raise typer.Exit(code=0)
    # Fallback: enrich/override missing or zero scores from master games file (past dates)
    try:
        if games_csv.exists():
            g_master = pd.read_csv(games_csv)
            if not g_master.empty and "game_id" in g_master.columns:
                g_master["game_id"] = g_master["game_id"].astype(str)
                keep_cols = [c for c in ["game_id","home_score","away_score","status","home_score_1h","away_score_1h","home_score_2h","away_score_2h"] if c in g_master.columns]
                if keep_cols:
                    games_df = games_df.merge(g_master[keep_cols], on="game_id", how="left", suffixes=("", "_m"))
                    # Coalesce scores: prefer provider (>0) else master (>0)
                    for side in ("home","away"):
                        prov = pd.to_numeric(games_df.get(f"{side}_score"), errors="coerce")
                        mast = pd.to_numeric(games_df.get(f"{side}_score_m"), errors="coerce")
                        games_df[f"{side}_score"] = np.where((prov>0) | prov.notna(), games_df.get(f"{side}_score"), mast)
                    # Backfill status if missing
                    if "status" in games_df.columns and "status_m" in games_df.columns:
                        games_df["status"] = games_df["status"].where(games_df["status"].notna(), games_df["status_m"])
    except Exception:
        pass
    # Attach predictions (pred_total, pred_margin)
    preds = pd.DataFrame()
    tried_paths: list[Path] = []
    # 1) Try provided predictions_csv
    if predictions_csv.exists():
        tried_paths.append(predictions_csv)
        try:
            preds = pd.read_csv(predictions_csv)
            if "game_id" in preds.columns:
                preds["game_id"] = preds["game_id"].astype(str)
        except Exception:
            preds = pd.DataFrame()
    # 2) If empty or no matching game_ids, try predictions_all.csv in outputs
    if preds.empty or (not set(games_df.get("game_id", pd.Series(dtype=str)).astype(str)).intersection(set(preds.get("game_id", pd.Series(dtype=str)).astype(str)) )):
        alt = settings.outputs_dir / "predictions_all.csv"
        if alt.exists() and alt not in tried_paths:
            try:
                pa = pd.read_csv(alt)
                if "game_id" in pa.columns:
                    pa["game_id"] = pa["game_id"].astype(str)
                # Prefer rows for this date if present
                if "date" in pa.columns:
                    pa_date = pa[pa["date"].astype(str) == date]
                    if not pa_date.empty:
                        preds = pa_date
                    else:
                        preds = pa
                else:
                    preds = pa
                print(f"[green]Loaded predictions from fallback[/green] {alt} ({len(preds)} rows)")
            except Exception as _pe2:
                print(f"[yellow]Fallback predictions_all.csv load failed:[/yellow] {_pe2}")
    keep_pred_cols = [c for c in ["game_id","pred_total","pred_margin"] if c in preds.columns]
    if keep_pred_cols:
        games_df = games_df.merge(preds[keep_pred_cols], on="game_id", how="left")
    # Attach odds (market_total, closing_total, spread_home, closing_spread_home)
    odds = pd.DataFrame()
    if odds_csv.exists():
        try:
            odds = pd.read_csv(odds_csv)
            if "game_id" in odds.columns:
                odds["game_id"] = odds["game_id"].astype(str)
        except Exception:
            odds = pd.DataFrame()
    # Derive market_total from totals market median if not already present
    if not odds.empty and "game_id" in odds.columns:
        o = odds.copy()
        if {"market","total"}.issubset(o.columns):
            totals_fg = o[(o["market"].astype(str).str.lower()=="totals") & (o.get("period","full_game").astype(str).str.lower().isin(["full_game","full game","fg"]))]
            if not totals_fg.empty:
                med_tot = totals_fg.groupby("game_id")["total"].median().rename("market_total")
                games_df = games_df.merge(med_tot, on="game_id", how="left")
        if {"market","home_spread"}.issubset(o.columns):
            spreads_fg = o[(o["market"].astype(str).str.lower()=="spreads") & (o.get("period","full_game").astype(str).str.lower().isin(["full_game","full game","fg"]))]
            if not spreads_fg.empty:
                med_spread = spreads_fg.groupby("game_id")["home_spread"].median().rename("spread_home")
                games_df = games_df.merge(med_spread, on="game_id", how="left")
    # Compute actual totals/margins where scores present
    try:
        hs = pd.to_numeric(games_df.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(games_df.get("away_score"), errors="coerce")
        mask_done = (hs > 0) | (as_ > 0)
        games_df["actual_total"] = np.where(mask_done, hs + as_, 0)
        games_df["actual_margin"] = np.where(mask_done, hs - as_, 0)
    except Exception:
        games_df["actual_total"] = 0
        games_df["actual_margin"] = 0
    # ATS / OU outcomes when lines and scores available
    try:
        if {"spread_home","actual_margin"}.issubset(games_df.columns):
            sp = pd.to_numeric(games_df["spread_home"], errors="coerce")
            am = pd.to_numeric(games_df["actual_margin"], errors="coerce")
            games_df["ats_result"] = np.where(am==0, "Push", np.where(am > -sp, "Home Cover", "Away Cover"))
        if {"market_total","actual_total"}.issubset(games_df.columns):
            mt = pd.to_numeric(games_df["market_total"], errors="coerce")
            at = pd.to_numeric(games_df["actual_total"], errors="coerce")
            games_df["ou_result_full"] = np.where(at==mt, "Push", np.where(at>mt, "Over", "Under"))
    except Exception:
        pass
    # Halftime / 2H derivation: if include_halves and we have start_time + current time > cutoff or explicit half scores
    if include_halves and not games_df.empty:
        # If games_all provides half scores, merge them first
        try:
            if games_csv.exists():
                g_master = pd.read_csv(games_csv)
                if not g_master.empty and "game_id" in g_master.columns:
                    g_master["game_id"] = g_master["game_id"].astype(str)
                    half_cols = [c for c in ["home_score_1h","away_score_1h","home_score_2h","away_score_2h"] if c in g_master.columns]
                    if half_cols:
                        games_df = games_df.merge(g_master[["game_id"]+half_cols], on="game_id", how="left")
        except Exception:
            pass

    # Boxscores score backfill: if provider & master both show zero scores, attempt to pull from boxscores CSV (may contain possessions only; also merge half scores when present)
    try:
        if boxscores_csv.exists() and not games_df.empty and "game_id" in games_df.columns:
            bs = pd.read_csv(boxscores_csv)
            if not bs.empty and "game_id" in bs.columns:
                bs["game_id"] = bs["game_id"].astype(str)
                score_cols = [c for c in bs.columns if c.lower() in {"home_score","away_score","home_final","away_final"}]
                half_cols = [c for c in ["home_score_1h","away_score_1h","home_score_2h","away_score_2h"] if c in bs.columns]
                if score_cols and {"home_score","away_score"}.issubset(score_cols):
                    # Only replace when current scores are zero or NaN
                    hs_cur = pd.to_numeric(games_df.get("home_score"), errors="coerce")
                    as_cur = pd.to_numeric(games_df.get("away_score"), errors="coerce")
                    replace_mask = (hs_cur.fillna(0) <= 0) & (as_cur.fillna(0) <= 0)
                    merge_cols = ["game_id","home_score","away_score"] + half_cols if {"home_score","away_score"}.issubset(bs.columns) else []
                    if merge_cols:
                        games_df = games_df.merge(bs[merge_cols], on="game_id", how="left", suffixes=("","_bs"))
                        for side in ("home","away"):
                            cur = pd.to_numeric(games_df.get(f"{side}_score"), errors="coerce")
                            bs_val = pd.to_numeric(games_df.get(f"{side}_score_bs"), errors="coerce")
                            games_df.loc[replace_mask & bs_val.notna() & (bs_val>0), f"{side}_score"] = bs_val[replace_mask & bs_val.notna() & (bs_val>0)]
                        # Fill halves from boxscores where missing
                        for col in half_cols:
                            if f"{col}_bs" in games_df.columns:
                                games_df[col] = games_df[col].where(games_df[col].notna(), games_df[f"{col}_bs"]) 
    except Exception:
        print("[yellow]Boxscores backfill skipped due to error[/yellow]")
    # Overrides CSV: allow manual finals patches
    try:
        # Look for date-specific override if not provided
        default_override = settings.outputs_dir / f"scores_override_{date}.csv"
        cand = overrides_csv if (overrides_csv is not None and overrides_csv.exists()) else (default_override if default_override.exists() else None)
        if cand and not games_df.empty and "game_id" in games_df.columns:
            ov = pd.read_csv(cand)
            if not ov.empty and {"game_id","home_score","away_score"}.issubset(ov.columns):
                ov["game_id"] = ov["game_id"].astype(str)
                merged = games_df.merge(ov[["game_id","home_score","away_score"]], on="game_id", how="left", suffixes=("","_ov"))
                for side in ("home","away"):
                    cur = pd.to_numeric(merged.get(f"{side}_score"), errors="coerce")
                    ovv = pd.to_numeric(merged.get(f"{side}_score_ov"), errors="coerce")
                    merged.loc[(cur.fillna(0) <= 0) & ovv.notna() & (ovv>0), f"{side}_score"] = ovv[(cur.fillna(0) <= 0) & ovv.notna() & (ovv>0)]
                games_df = merged.drop(columns=[c for c in merged.columns if c.endswith("_ov")])
                print(f"[green]Applied scores override from[/green] {cand}")
    except Exception as e:
        print(f"[yellow]Overrides merge skipped[/yellow]: {e}")
    # Derive / coalesce half actuals: always attempt even if base columns exist but are NaN
    try:
        if {"home_score","away_score"}.issubset(games_df.columns):
            hs = pd.to_numeric(games_df["home_score"], errors="coerce")
            as_ = pd.to_numeric(games_df["away_score"], errors="coerce")
            # Coalesce suffixed half columns into canonical names even if canonical exists but is all NaN
            for base in ["home_score_1h","away_score_1h","home_score_2h","away_score_2h"]:
                cands = [c for c in games_df.columns if c == base or c.startswith(base + "_")]
                if cands:
                    vals = None
                    # Order: base first then suffixed so existing non-NaN canonical values win
                    for c in [base] + [c for c in cands if c != base]:
                        if c not in games_df.columns: continue
                        try:
                            s = pd.to_numeric(games_df[c], errors="coerce")
                        except Exception:
                            s = pd.Series(np.nan, index=games_df.index)
                        vals = s if vals is None else vals.where(vals.notna(), s)
                    if vals is not None:
                        games_df[base] = vals
            # Provide placeholders so downstream templates/API can rely on columns existing.
            for col in ["home_score_1h","away_score_1h","home_score_2h","away_score_2h","actual_total_1h","actual_total_2h"]:
                if col not in games_df.columns:
                    games_df[col] = np.nan
            # If we have 1H scores, compute 2H and actual totals for halves
            if {"home_score_1h","away_score_1h"}.issubset(games_df.columns):
                h1 = pd.to_numeric(games_df["home_score_1h"], errors="coerce")
                a1 = pd.to_numeric(games_df["away_score_1h"], errors="coerce")
                # 2H = final - 1H when both present
                mask_half = h1.notna() & a1.notna() & hs.notna() & as_.notna()
                games_df.loc[mask_half, "home_score_2h"] = hs[mask_half] - h1[mask_half]
                games_df.loc[mask_half, "away_score_2h"] = as_[mask_half] - a1[mask_half]
                games_df.loc[mask_half, "actual_total_1h"] = h1[mask_half] + a1[mask_half]
                games_df.loc[mask_half, "actual_total_2h"] = games_df.loc[mask_half, "home_score_2h"] + games_df.loc[mask_half, "away_score_2h"]
            # If 2H scores exist but 1H blank (edge-case), compute 1H = final - 2H
            if {"home_score_2h","away_score_2h"}.issubset(games_df.columns):
                h2 = pd.to_numeric(games_df["home_score_2h"], errors="coerce")
                a2 = pd.to_numeric(games_df["away_score_2h"], errors="coerce")
                mask_half2 = h2.notna() & a2.notna() & hs.notna() & as_.notna()
                # Only fill where 1H is NaN
                if "home_score_1h" in games_df.columns:
                    mfill_h1 = mask_half2 & games_df["home_score_1h"].isna()
                    games_df.loc[mfill_h1, "home_score_1h"] = hs[mfill_h1] - h2[mfill_h1]
                if "away_score_1h" in games_df.columns:
                    mfill_a1 = mask_half2 & games_df["away_score_1h"].isna()
                    games_df.loc[mfill_a1, "away_score_1h"] = as_[mfill_a1] - a2[mfill_a1]
                # Recompute actual totals where possible
                if {"home_score_1h","away_score_1h"}.issubset(games_df.columns):
                    h1b = pd.to_numeric(games_df["home_score_1h"], errors="coerce")
                    a1b = pd.to_numeric(games_df["away_score_1h"], errors="coerce")
                    mask_ht = h1b.notna() & a1b.notna()
                    games_df.loc[mask_ht, "actual_total_1h"] = h1b[mask_ht] + a1b[mask_ht]
                mask_ht2 = h2.notna() & a2.notna()
                games_df.loc[mask_ht2, "actual_total_2h"] = h2[mask_ht2] + a2[mask_ht2]
    except Exception:
        pass

    # Output path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{date}.csv"
    if out_path.exists() and not overwrite:
        print(f"[yellow]results file exists and overwrite disabled:[/yellow] {out_path}")
        raise typer.Exit(code=0)
    games_df.to_csv(out_path, index=False)
    print(f"[green]Wrote finalized daily results[/green] {out_path} ({len(games_df)} rows; scores_in={int(((games_df['home_score']>0)|(games_df['away_score']>0)).sum())})")

# Utility: normalize key for maps/segments (reuse odds merge normalizer)
def _norm(s: str | None) -> str:
    return normalize_name(s or "")


@app.command(name="fetch-scores")
def fetch_scores(
    date: str = typer.Option(..., help="Target date YYYY-MM-DD"),
    provider: str = typer.Option("both", help="Provider to pull scores from: espn|ncaa|both"),
    out_dir: Path = typer.Option(settings.outputs_dir, help="Directory to write scores_raw_<date>_<prov>.csv"),
):
    """Force a no-cache scoreboard fetch for a date and write raw scores CSV(s).

    Useful when provider cache returns zeros; compare ESPN vs NCAA quickly.
    Writes one file per provider requested.
    """
    try:
        d_obj = dt.date.fromisoformat(date)
    except Exception:
        print(f"[red]Invalid date format[/red] {date}")
        raise typer.Exit(code=1)
    provs = []
    p = provider.lower()
    if p in {"espn","ncaa"}:
        provs = [p]
    else:
        provs = ["espn","ncaa"]
    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    for prov in provs:
        iterator = iter_games_espn if prov == "espn" else iter_games_ncaa
        rows: list[dict[str, Any]] = []
        try:
            for res in iterator(d_obj, d_obj, use_cache=False):
                for g in res.games:
                    m = g.model_dump()
                    rows.append({
                        "game_id": str(m.get("game_id")),
                        "date": date,
                        "start_time": m.get("start_time"),
                        "home_team": m.get("home_team"),
                        "away_team": m.get("away_team"),
                        "home_score": m.get("home_score"),
                        "away_score": m.get("away_score"),
                        "home_score_1h": m.get("home_score_1h"),
                        "away_score_1h": m.get("away_score_1h"),
                        "home_score_2h": m.get("home_score_2h"),
                        "away_score_2h": m.get("away_score_2h"),
                        "status": m.get("status"),
                        "venue": m.get("venue"),
                        "neutral_site": m.get("neutral_site"),
                    })
        except Exception as e:
            print(f"[yellow]{prov} fetch failed[/yellow]: {e}")
        if rows:
            df = pd.DataFrame(rows)
            path = out_dir / f"scores_raw_{date}_{prov}.csv"
            df.to_csv(path, index=False)
            total_written += len(df)
            nonzero = int(((pd.to_numeric(df.get('home_score'), errors='coerce').fillna(0) > 0) | (pd.to_numeric(df.get('away_score'), errors='coerce').fillna(0) > 0)).sum())
            print(f"[green]Wrote {len(df)} rows to[/green] {path} (nonzero_scores={nonzero})")
        else:
            print(f"[yellow]No rows returned for {prov} on {date}[/yellow]")
    if total_written == 0:
        raise typer.Exit(code=1)


def _load_conf_map(conf_map_path: Path | None) -> dict[str, str]:
    """Load a simple team->conference mapping CSV with columns team,conference.

    Returns a dict keyed on normalized team name.
    """
    mapping: dict[str, str] = {}
    if conf_map_path is None or not conf_map_path.exists():
        return mapping
    try:
        mdf = pd.read_csv(conf_map_path)
        # Support variants of column names
        team_col = None
        conf_col = None
        for c in mdf.columns:
            lc = str(c).strip().lower()
            if lc in {"team", "name", "raw", "canonical"} and team_col is None:
                team_col = c
            if lc in {"conference", "conf"} and conf_col is None:
                conf_col = c
        if team_col is None or conf_col is None:
            return mapping
        for t, conf in zip(mdf[team_col].astype(str), mdf[conf_col].astype(str)):
            key = _norm(t)
            if key and conf and str(conf).strip():
                mapping[key] = str(conf).strip()
    except Exception:
        return {}
    return mapping


def _predict_segmented_inline(
    feats: pd.DataFrame,
    seg_mode: str,
    models_root: Path,
    conf_map_path: Path | None = None,
) -> pd.DataFrame:
    """Predict totals and margins using segmented models (team|conference).

    Fallback: any rows without a matching segment model will be scored by the global baseline in models_root.
    Returns a DataFrame with columns [game_id,date,home_team,away_team,pred_total,pred_margin].
    """
    seg_mode = seg_mode.lower()
    if seg_mode not in {"team", "conference"}:
        raise ValueError(f"Unsupported segmentation mode: {seg_mode}")

    df = feats.copy()
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)

    # Conference mapping if needed
    conf_map = _load_conf_map(conf_map_path) if seg_mode == "conference" else {}
    if seg_mode == "conference":
        df["home_conf"] = df.get("home_team", pd.Series(dtype=str)).astype(str).map(lambda x: conf_map.get(_norm(x)))
        df["away_conf"] = df.get("away_team", pd.Series(dtype=str)).astype(str).map(lambda x: conf_map.get(_norm(x)))

    # Resolve model dir per row
    model_for_idx: dict[int, Path | None] = {}
    seg_base = models_root / ("seg_team" if seg_mode == "team" else "seg_conference")
    bad_keys = {"tbd", "tba", "unknown", "na", "n/a", ""}
    for idx, r in df.iterrows():
        mdir: Path | None = None
        if seg_mode == "team":
            # Prefer home team model, then away
            for tcol in ["home_team", "away_team"]:
                if tcol in df.columns and pd.notna(r.get(tcol)):
                    key = _norm(str(r.get(tcol)))
                    if key in bad_keys:
                        continue
                    p = seg_base / key
                    if (p / "baseline_target_total.onnx").exists():
                        mdir = p
                        break
        else:
            hc = r.get("home_conf")
            ac = r.get("away_conf")
            chosen = None
            if pd.notna(hc) and pd.notna(ac) and str(hc) == str(ac):
                chosen = str(hc)
            else:
                chosen = "nonconf"
            p = seg_base / _norm(chosen)
            if (p / "baseline_target_total.onnx").exists():
                mdir = p
        model_for_idx[idx] = mdir

    # Prepare global fallback
    global_cols_path = models_root / "feature_columns.txt"
    global_total = models_root / "baseline_target_total.onnx"
    global_margin = models_root / "baseline_target_margin.onnx"
    providers = OnnxPredictor.describe_available()
    global_pred_total = OnnxPredictor(str(global_total)) if providers else NumpyLinearPredictor(str(global_total))
    global_pred_margin = OnnxPredictor(str(global_margin)) if providers else NumpyLinearPredictor(str(global_margin))
    global_cols = [c.strip() for c in global_cols_path.read_text(encoding="utf-8").splitlines() if c.strip()]

    # Initialize outputs
    out_df = df[["game_id", "date", "home_team", "away_team"]].copy()
    out_df["pred_total"] = np.nan
    out_df["pred_margin"] = np.nan
    out_df["model_used"] = ""

    # Group by model directory (including None for fallback)
    by_dir: dict[Path | None, list[int]] = {}
    for i, p in model_for_idx.items():
        by_dir.setdefault(p, []).append(i)

    # Score each group
    for mdir, idxs in by_dir.items():
        sub = df.loc[idxs]
        if mdir is None:
            # Global fallback
            X = sub.reindex(columns=global_cols).fillna(0.0).to_numpy(dtype=np.float32)
            y_t = global_pred_total.predict(X).reshape(-1)
            y_m = global_pred_margin.predict(X).reshape(-1)
            out_df.loc[idxs, "pred_total"] = y_t
            out_df.loc[idxs, "pred_margin"] = y_m
            out_df.loc[idxs, "model_used"] = "global"
            continue
        # Load model-specific columns and predictors
        cols_path = mdir / "feature_columns.txt"
        if not cols_path.exists():
            # Fallback to global if columns missing
            X = sub.reindex(columns=global_cols).fillna(0.0).to_numpy(dtype=np.float32)
            y_t = global_pred_total.predict(X).reshape(-1)
            y_m = global_pred_margin.predict(X).reshape(-1)
            out_df.loc[idxs, "pred_total"] = y_t
            out_df.loc[idxs, "pred_margin"] = y_m
            out_df.loc[idxs, "model_used"] = "global"
            continue
        cols = [c.strip() for c in cols_path.read_text(encoding="utf-8").splitlines() if c.strip()]
        X = sub.reindex(columns=cols).fillna(0.0).to_numpy(dtype=np.float32)
        total_model_path = mdir / "baseline_target_total.onnx"
        margin_model_path = mdir / "baseline_target_margin.onnx"
        pred_total = OnnxPredictor(str(total_model_path)) if providers else NumpyLinearPredictor(str(total_model_path))
        pred_margin = OnnxPredictor(str(margin_model_path)) if providers else NumpyLinearPredictor(str(margin_model_path))
        y_t = pred_total.predict(X).reshape(-1)
        y_m = pred_margin.predict(X).reshape(-1)
        out_df.loc[idxs, "pred_total"] = y_t
        out_df.loc[idxs, "pred_margin"] = y_m
        label = ("team:" if seg_mode == "team" else "conference:") + mdir.name
        out_df.loc[idxs, "model_used"] = label

    return out_df


@app.command(name="build-priors")
def build_priors(
    features_csv: Path = typer.Argument(settings.outputs_dir / "features_hist.csv", help="Historical features CSV (multi-season)"),
    out: Path = typer.Option(settings.outputs_dir / "priors.csv", help="Output priors CSV with team,rating_margin,off_rating,def_rating,tempo_rating"),
    window_days: int = typer.Option(730, help="Only include rows within the last N days for priors"),
    min_rows: int = typer.Option(3, help="Minimum occurrences per team to include in priors"),
):
    """Build team priors from historical features by averaging rating metrics per team.

    Outputs columns: team, rating_margin, off_rating, def_rating, tempo_rating
    """
    if not Path(features_csv).exists():
        print(f"[red]Missing features file:[/red] {features_csv}")
        raise typer.Exit(code=1)
    df = pd.read_csv(features_csv)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(window_days))
            df = df[df["date"] >= cutoff]
        except Exception:
            pass

    def _stack(df_: pd.DataFrame, col_home: str, col_away: str, out_name: str) -> pd.DataFrame:
        parts = []
        if col_home in df_.columns and "home_team" in df_.columns:
            a = df_[["home_team", col_home]].rename(columns={"home_team": "team", col_home: out_name})
            parts.append(a)
        if col_away in df_.columns and "away_team" in df_.columns:
            b = df_[["away_team", col_away]].rename(columns={"away_team": "team", col_away: out_name})
            parts.append(b)
        if not parts:
            return pd.DataFrame(columns=["team", out_name])
        both = pd.concat(parts, ignore_index=True)
        both["team"] = both["team"].astype(str)
        # Drop NaNs
        both = both.dropna(subset=[out_name])
        return both.groupby("team", as_index=False)[out_name].mean()

    p_margin = _stack(df, "home_rating_margin", "away_rating_margin", "rating_margin")
    p_off = _stack(df, "home_off_rating", "away_off_rating", "off_rating")
    p_def = _stack(df, "home_def_rating", "away_def_rating", "def_rating")
    p_tmp = _stack(df, "home_tempo_rating", "away_tempo_rating", "tempo_rating")

    # Count occurrences to enforce min_rows
    counts = []
    for side, team_col in [("home_team", "home_team"), ("away_team", "away_team")]:
        if team_col in df.columns:
            counts.append(df[team_col].astype(str))
    cdf = pd.concat(counts, ignore_index=True).to_frame(name="team") if counts else pd.DataFrame(columns=["team"])
    cdf = cdf[cdf["team"].notna()]
    cdf = cdf.groupby("team", as_index=False).size().rename(columns={"size": "n"}) if not cdf.empty else pd.DataFrame(columns=["team", "n"])

    pri = (((p_margin.merge(p_off, on="team", how="outer"))
                 .merge(p_def, on="team", how="outer"))
                 .merge(p_tmp, on="team", how="outer"))
    if not cdf.empty:
        pri = pri.merge(cdf, on="team", how="left")
        pri = pri[pri["n"].fillna(0) >= int(min_rows)]
        pri = pri.drop(columns=["n"], errors="ignore")
    pri = pri.dropna(how="all", subset=[c for c in ["rating_margin","off_rating","def_rating","tempo_rating"] if c in pri.columns])
    out.parent.mkdir(parents=True, exist_ok=True)
    pri.to_csv(out, index=False)
    print(f"[green]Wrote priors for {len(pri)} teams to[/green] {out}")


@app.command(name="daily-results")
def daily_results(
    date: str | None = typer.Option(None, help="Target date YYYY-MM-DD (default: yesterday)"),
    games_path: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Games CSV with final scores for the season"),
    preds_path: Path = typer.Option(settings.outputs_dir / "predictions_all.csv", help="Predictions CSV (will fallback to predictions_week.csv for the date if present)"),
    closing_merged: Path | None = typer.Option(settings.outputs_dir / "games_with_closing.csv", help="Optional merged games_with_closing.csv with totals lines"),
    picks_path: Path | None = typer.Option(settings.outputs_dir / "picks_clean.csv", help="Optional picks CSV to reconcile picks results"),
    picks_raw_path: Path | None = typer.Option(settings.outputs_dir / "picks_raw.csv", help="Optional expanded picks CSV (totals/spreads/moneyline incl. halves) to reconcile"),
    out_dir: Path = typer.Option(settings.outputs_dir / "daily_results", help="Output directory for per-day results and summary"),
    half_ratio: float = typer.Option(0.485, help="Share of full-game total expected in 1H for derived model half projections (rest goes to 2H)"),
    margin_half_ratio: float = typer.Option(0.5, help="Share of full-game margin expected in 1H for derived model half projections"),
    margin_prob_scale: float = typer.Option(7.0, help="Scale (points) to convert model margin to win probability via logistic"),
):
    """Create a per-day results CSV by merging games, predictions, strict last/closing odds, and picks with outcomes.

    Adds robust reconciliation for:
      - Totals: full game + 1H + 2H (actuals, model errors, market medians)
      - ATS: full game + 1H (home_spread vs actual margin, push handling)
      - ML: full game (model margin -> probability vs moneyline implied; Brier/logloss)
    """
    target = dt.date.fromisoformat(date) if date else (_today_local() - dt.timedelta(days=1))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load games and filter to date
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))
    for col in ["game_id", "date"]:
        if col in games.columns and col == "game_id":
            games[col] = games[col].astype(str)
    if "date" in games.columns:
        games["date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
        gday = games[games["date"] == target.isoformat()].copy()
    else:
        print("[red]games file missing 'date' column[/red]")
        raise typer.Exit(code=1)
    if gday.empty:
        print(f"[yellow]No games found on {target} in {games_path}[/yellow]")
        raise typer.Exit(code=0)

    # Load predictions; fallback chain: predictions_all -> predictions_week -> predictions_hist_team/global for the date
    preds = pd.DataFrame()
    try:
        preds = pd.read_csv(preds_path)
    except Exception:
        preds = pd.DataFrame()
    if preds.empty or "date" not in preds.columns:
        alt = settings.outputs_dir / "predictions_week.csv"
        if alt.exists():
            preds = pd.read_csv(alt)
    # Historical predictions fallback (team-segmented then global) if still missing or after filtering
    def _load_hist_preds() -> pd.DataFrame:
        for name in ("predictions_hist_team.csv", "predictions_hist_global.csv"):
            p = settings.outputs_dir / name
            if p.exists():
                try:
                    dfh = pd.read_csv(p)
                    if not dfh.empty:
                        return dfh
                except Exception:
                    continue
        return pd.DataFrame()
    if preds.empty:
        preds = _load_hist_preds()
        if preds.empty:
            print("[yellow]No predictions available; proceeding with outcomes only.[/yellow]")
            preds = pd.DataFrame(columns=["game_id","pred_total","pred_margin","date"])
    if "game_id" in preds.columns:
        preds["game_id"] = preds["game_id"].astype(str)
    if "date" in preds.columns:
        try:
            preds["date"] = pd.to_datetime(preds["date"]).dt.strftime("%Y-%m-%d")
            preds = preds[preds["date"] == target.isoformat()]
        except Exception:
            pass
    # If empty after filtering, try historical predictions once more
    if preds.empty:
        h = _load_hist_preds()
        if not h.empty and "date" in h.columns:
            try:
                h["date"] = pd.to_datetime(h["date"]).dt.strftime("%Y-%m-%d")
                preds = h[h["date"] == target.isoformat()].copy()
                if "game_id" in preds.columns:
                    preds["game_id"] = preds["game_id"].astype(str)
            except Exception:
                pass

    # Merge base
    df = gday.merge(preds[[c for c in ["game_id","pred_total","pred_margin"] if c in preds.columns]], on="game_id", how="left")
    df["actual_total"] = df[["home_score","away_score"]].sum(axis=1, min_count=2)
    df["actual_margin"] = df["home_score"] - df["away_score"]

    # Attach odds (prefer strict last if provided via closing_merged path), compute per-period medians for totals/spreads/h2h
    odds_cols = {
        "totals": "total",
        "spreads": "home_spread",
        "h2h": "moneyline_home",
    }
    def _period_mask(s: pd.Series, keys: set[str]) -> pd.Series:
        vals = s.astype(str).str.lower()
        return vals.isin(keys)
    if closing_merged is not None and closing_merged.exists():
        try:
            cm = pd.read_csv(closing_merged)
            if "game_id" in cm.columns:
                cm["game_id"] = cm["game_id"].astype(str)
            # Prepare helpers for period filters
            def agg_market_period(market: str, period_keys: set[str], out_col: str) -> pd.Series | None:
                if cm.empty or {"market","period"}.issubset(cm.columns) is False:
                    return None
                sub = cm[(cm["market"].astype(str).str.lower() == market) & _period_mask(cm["period"], period_keys)].copy()
                val_col = odds_cols.get(market)
                if val_col is None or val_col not in sub.columns:
                    return None
                if sub.empty:
                    return None
                return sub.groupby("game_id")[val_col].median().rename(out_col)

            # Totals medians
            t_fg = agg_market_period("totals", {"full_game","fg","full game"}, "market_total")
            t_1h = agg_market_period("totals", {"1h","1h_1","first_half","1st_half"}, "market_total_1h")
            t_2h = agg_market_period("totals", {"2h","second_half","2nd_half"}, "market_total_2h")
            for srs in [t_fg, t_1h, t_2h]:
                if srs is not None and not srs.empty:
                    df = df.merge(srs, on="game_id", how="left")
            # Spreads (home perspective)
            s_fg = agg_market_period("spreads", {"full_game","fg","full game"}, "spread_home")
            s_1h = agg_market_period("spreads", {"1h","1h_1","first_half","1st_half"}, "spread_home_1h")
            for srs in [s_fg, s_1h]:
                if srs is not None and not srs.empty:
                    df = df.merge(srs, on="game_id", how="left")
            # Moneyline (home price)
            m_fg = agg_market_period("h2h", {"full_game","fg","full game"}, "ml_home")
            if m_fg is not None and not m_fg.empty:
                df = df.merge(m_fg, on="game_id", how="left")
            # Back-compat: if only closing_total provided in cm (heuristic), keep it for reference
            if "closing_total" not in df.columns and "closing_total" in cm.columns:
                try:
                    closing_by_game = cm.groupby("game_id")["closing_total"].median().rename("closing_total")
                    df = df.merge(closing_by_game, on="game_id", how="left")
                except Exception:
                    pass
        except Exception as e:
            print(f"[yellow]Failed to attach odds:[/yellow] {e}")

    # Fallbacks: if halves markets were not available from the provider, derive simple approximations from full-game lines
    try:
        # Totals: split full-game line using half_ratio when 1H/2H market medians are missing
        if ("market_total" in df.columns) and (
            ("market_total_1h" not in df.columns) or df.get("market_total_1h").isna().all()
        ):
            try:
                hratio = max(0.0, min(1.0, float(half_ratio)))
            except Exception:
                hratio = 0.485
            mt = pd.to_numeric(df["market_total"], errors="coerce")
            df["market_total_1h"] = mt * hratio
            df["market_total_2h"] = mt - df["market_total_1h"]
        # Spreads: approximate 1H spread as half the full-game spread when missing
        if ("spread_home" in df.columns) and (
            ("spread_home_1h" not in df.columns) or df.get("spread_home_1h").isna().all()
        ):
            sp = pd.to_numeric(df["spread_home"], errors="coerce")
            df["spread_home_1h"] = sp * 0.5
    except Exception:
        pass

    # Errors and comparison vs closing
    if "pred_total" in df.columns and df["pred_total"].notna().any():
        df["err_model_total"] = df["pred_total"] - df["actual_total"]
        if "closing_total" in df.columns and df["closing_total"].notna().any():
            df["err_closing_total"] = df["closing_total"] - df["actual_total"]
            df["model_better_than_closing"] = (df["err_model_total"].abs() < df["err_closing_total"].abs()).astype(int)

    # Derive half actuals if available
    def _safe_num(x):
        try:
            return float(x)
        except Exception:
            return None
    # First half totals
    if {"home_score_1h","away_score_1h"}.issubset(df.columns):
        try:
            df["actual_total_1h"] = pd.to_numeric(df["home_score_1h"], errors="coerce") + pd.to_numeric(df["away_score_1h"], errors="coerce")
        except Exception:
            df["actual_total_1h"] = None
    # Second half totals (prefer explicit, else derive)
    if {"home_score_2h","away_score_2h"}.issubset(df.columns):
        try:
            df["actual_total_2h"] = pd.to_numeric(df["home_score_2h"], errors="coerce") + pd.to_numeric(df["away_score_2h"], errors="coerce")
        except Exception:
            df["actual_total_2h"] = None
    if "actual_total_2h" not in df.columns or df["actual_total_2h"].isna().all():
        try:
            df["actual_total_2h"] = pd.to_numeric(df["actual_total"], errors="coerce") - pd.to_numeric(df["actual_total_1h"], errors="coerce")
        except Exception:
            df["actual_total_2h"] = None

    # Derived model half projections (prefer calibrated totals/margins if present)
    point_total_col = "pred_total_calibrated" if "pred_total_calibrated" in df.columns else "pred_total"
    point_margin_col = "pred_margin_calibrated" if "pred_margin_calibrated" in df.columns else "pred_margin"
    if point_total_col in df.columns:
        try:
            hratio = max(0.0, min(1.0, float(half_ratio)))
        except Exception:
            hratio = 0.485
        try:
            pt_series = pd.to_numeric(df[point_total_col], errors="coerce")
            df["pred_total_1h"] = pt_series * hratio
            df["pred_total_2h"] = pt_series - df["pred_total_1h"]
        except Exception:
            df["pred_total_1h"] = None
            df["pred_total_2h"] = None
    if point_margin_col in df.columns:
        try:
            mhr = max(0.0, min(1.0, float(margin_half_ratio)))
        except Exception:
            mhr = 0.5
        try:
            pm_series = pd.to_numeric(df[point_margin_col], errors="coerce")
            df["pred_margin_1h"] = pm_series * mhr
            df["pred_margin_2h"] = pm_series - df["pred_margin_1h"]
        except Exception:
            df["pred_margin_1h"] = None
            df["pred_margin_2h"] = None

    # Attach errors for halves where available
    if {"pred_total_1h","actual_total_1h"}.issubset(df.columns):
        try:
            df["err_model_total_1h"] = pd.to_numeric(df["pred_total_1h"], errors="coerce") - pd.to_numeric(df["actual_total_1h"], errors="coerce")
        except Exception:
            df["err_model_total_1h"] = None
    if {"pred_total_2h","actual_total_2h"}.issubset(df.columns):
        try:
            df["err_model_total_2h"] = pd.to_numeric(df["pred_total_2h"], errors="coerce") - pd.to_numeric(df["actual_total_2h"], errors="coerce")
        except Exception:
            df["err_model_total_2h"] = None

    # ATS outcomes (full game + 1H) when spread lines available
    def settle_spread(actual_margin: float, spread_home: float) -> str | None:
        if actual_margin is None or spread_home is None:
            return None
        diff = actual_margin - spread_home
        if abs(diff) < 1e-9:
            return "push"
        return "home_cover" if diff > 0 else "away_cover"
    try:
        df["ats_result"] = df.apply(lambda r: settle_spread(_safe_num(r.get("actual_margin")), _safe_num(r.get("spread_home"))), axis=1)
    except Exception:
        df["ats_result"] = None
    # First half ATS requires half margin (home_1h - away_1h) and spread_home_1h
    if {"home_score_1h","away_score_1h","spread_home_1h"}.issubset(df.columns):
        try:
            df["actual_margin_1h"] = pd.to_numeric(df["home_score_1h"], errors="coerce") - pd.to_numeric(df["away_score_1h"], errors="coerce")
            df["ats_result_1h"] = df.apply(lambda r: settle_spread(_safe_num(r.get("actual_margin_1h")), _safe_num(r.get("spread_home_1h"))), axis=1)
        except Exception:
            df["ats_result_1h"] = None

    # Moneyline calibration (full game): model probability vs implied
    def american_to_implied(price: float | None) -> float | None:
        try:
            p = float(price)
        except Exception:
            return None
        if p < 0:
            return (-p) / ((-p) + 100.0)
        return 100.0 / (p + 100.0)
    def logistic(x: float, s: float) -> float:
        try:
            return 1.0 / (1.0 + np.exp(-x / s))
        except Exception:
            return None  # type: ignore
    try:
        import numpy as np  # local import if not at top
        scale = max(1e-3, float(margin_prob_scale))
        # Model p_home from predicted margin
        if "pred_margin" in df.columns:
            pm = pd.to_numeric(df["pred_margin"], errors="coerce")
            df["ml_model_p_home"] = pm.apply(lambda v: logistic(v, scale) if pd.notna(v) else None)
        # Implied from moneyline_home
        if "ml_home" in df.columns:
            df["ml_implied_home"] = df["ml_home"].apply(american_to_implied)
        # Actual outcome
        if {"home_score","away_score"}.issubset(df.columns):
            df["home_win"] = (pd.to_numeric(df["home_score"], errors="coerce") > pd.to_numeric(df["away_score"], errors="coerce")).astype(float)
        # Brier and log loss (avoid log(0))
        eps = 1e-12
        if "ml_model_p_home" in df.columns and "home_win" in df.columns:
            p = pd.to_numeric(df["ml_model_p_home"], errors="coerce")
            y = pd.to_numeric(df["home_win"], errors="coerce")
            df["brier_ml"] = (p - y) ** 2
            df["logloss_ml"] = -(y * np.log(p.clip(eps, 1 - eps)) + (1 - y) * np.log((1 - p).clip(eps, 1 - eps)))
    except Exception:
        # numpy may not be available in some environments
        pass

    # Reconcile picks if provided
    if picks_path is not None and picks_path.exists():
        try:
            picks = pd.read_csv(picks_path)
            if "game_id" in picks.columns:
                picks["game_id"] = picks["game_id"].astype(str)
            # Keep picks for the target date
            if "date" in picks.columns:
                picks["date"] = pd.to_datetime(picks["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                picks = picks[picks["date"] == target.isoformat()]
            keep = [c for c in ["game_id","book","bet","line","price","pred_total","edge"] if c in picks.columns]
            picks = picks[keep].drop_duplicates(subset=["game_id"], keep="first")
            df = df.merge(picks, on="game_id", how="left", suffixes=("", "_pick"))
            # Resolve pick outcome if we have actual_total and line
            def settle_total(row):
                if pd.isna(row.get("bet")) or pd.isna(row.get("actual_total")) or pd.isna(row.get("line")):
                    return None
                bet = str(row["bet"]).lower()
                total = float(row["actual_total"]) if pd.notna(row["actual_total"]) else None
                line = float(row["line"]) if pd.notna(row["line"]) else None
                if total is None or line is None:
                    return None
                if bet == "over":
                    if total > line: return "win"
                    if total < line: return "loss"
                    return "push"
                if bet == "under":
                    if total < line: return "win"
                    if total > line: return "loss"
                    return "push"
                return None
            df["pick_result"] = df.apply(settle_total, axis=1)
        except Exception as e:
            print(f"[yellow]Failed to reconcile picks:[/yellow] {e}")

    # Optional: flag rows without predictions for transparency
    if {"pred_total"}.issubset(df.columns):
        try:
            df["no_pred"] = df["pred_total"].isna().astype(int)
        except Exception:
            pass

    # Reconcile expanded raw picks (totals/spreads/moneyline, incl. halves)
    raw_outcomes: pd.DataFrame | None = None
    if picks_raw_path is not None and picks_raw_path.exists():
        try:
            pr = pd.read_csv(picks_raw_path)
            if pr.empty:
                raw_outcomes = None
            else:
                if "game_id" in pr.columns:
                    pr["game_id"] = pr["game_id"].astype(str)
                # date filter
                if "date" in pr.columns:
                    pr["date"] = pd.to_datetime(pr["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    pr = pr[pr["date"] == target.isoformat()].copy()
                # Merge in actuals from df
                cols_needed = [c for c in [
                    "game_id","home_team","away_team","period","market","book","pick","edge","line_value","predicted_value","fair_price"
                ] if c in pr.columns]
                m = pr[cols_needed].merge(
                    df[[c for c in [
                        "game_id","actual_total","actual_total_1h","actual_total_2h",
                        "actual_margin","actual_margin_1h","actual_margin_2h",
                        "home_score","away_score"
                    ] if c in df.columns]],
                    on="game_id", how="left"
                )
                # Helpers
                def settle_total_row(row) -> str | None:
                    per = str(row.get("period") or "").lower()
                    act = None
                    if per in ("full_game","fg","full game",""):
                        act = row.get("actual_total")
                    elif per in ("1h","first_half","1st_half"):
                        act = row.get("actual_total_1h")
                    elif per in ("2h","second_half","2nd_half"):
                        act = row.get("actual_total_2h")
                    line = row.get("line_value")
                    if pd.isna(act) or pd.isna(line):
                        return None
                    pick = str(row.get("pick") or "").lower()
                    if "over" in pick:
                        if float(act) > float(line): return "win"
                        if float(act) < float(line): return "loss"
                        return "push"
                    if "under" in pick:
                        if float(act) < float(line): return "win"
                        if float(act) > float(line): return "loss"
                        return "push"
                    return None
                def settle_spread_row(row) -> str | None:
                    per = str(row.get("period") or "").lower()
                    am = None
                    if per in ("full_game","fg","full game",""):
                        am = row.get("actual_margin")
                    elif per in ("1h","first_half","1st_half"):
                        am = row.get("actual_margin_1h")
                    elif per in ("2h","second_half","2nd_half"):
                        am = row.get("actual_margin_2h")
                    if pd.isna(am) or pd.isna(row.get("line_value")):
                        return None
                    # line_value is home_spread; determine side from pick text
                    pick_txt = str(row.get("pick") or "")
                    home = str(row.get("home_team") or "")
                    away = str(row.get("away_team") or "")
                    home_sel = pick_txt.startswith(home)
                    away_sel = pick_txt.startswith(away)
                    hs = float(row.get("line_value"))
                    # Home wins vs spread if actual_margin > -home_spread
                    if home_sel:
                        diff = float(am) - hs
                        if abs(diff) < 1e-9: return "push"
                        return "win" if diff > 0 else "loss"
                    if away_sel:
                        # Away line is -home_spread; away covers if actual_margin < -home_spread
                        if float(am) < -hs: return "win"
                        if abs(float(am) + hs) < 1e-9: return "push"
                        return "loss"
                    return None
                def settle_ml_row(row) -> str | None:
                    if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
                        return None
                    pick_txt = str(row.get("pick") or "")
                    home = str(row.get("home_team") or "")
                    away = str(row.get("away_team") or "")
                    hs = float(row.get("home_score")); as_ = float(row.get("away_score"))
                    if pick_txt.lower().endswith(" ml"):
                        if pick_txt.startswith(home):
                            return "win" if hs > as_ else ("loss" if hs < as_ else "push")
                        if pick_txt.startswith(away):
                            return "win" if as_ > hs else ("loss" if as_ < hs else "push")
                    return None
                def american_profit(a: float) -> float:
                    try:
                        a = float(a)
                    except Exception:
                        return 0.0
                    if a < 0: return 100.0/(-a)
                    return a/100.0
                # Compute result and pnl_units
                m["result"] = None
                res: list[str | None] = []
                pnl: list[float | None] = []
                for _, rrow in m.iterrows():
                    market = str(rrow.get("market") or "").lower()
                    r = None
                    if market == "totals":
                        r = settle_total_row(rrow)
                    elif market == "spreads":
                        r = settle_spread_row(rrow)
                    elif market == "moneyline":
                        r = settle_ml_row(rrow)
                    res.append(r)
                    # PnL per 1u risk
                    if r is None:
                        pnl.append(None)
                    elif r == "push":
                        pnl.append(0.0)
                    else:
                        price = rrow.get("line_value") if market == "moneyline" else -110.0
                        prf = american_profit(price)
                        pnl.append(prf if r == "win" else -1.0)
                m["result"] = res
                m["pnl_units"] = pnl
                raw_outcomes = m
                # Write per-day raw outcomes file
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"picks_raw_results_{target.isoformat()}.csv").write_text(m.to_csv(index=False))
        except Exception as e:
            print(f"[yellow]Failed to reconcile picks_raw:[/yellow] {e}")

    # Write outputs
    out_csv = out_dir / f"results_{target.isoformat()}.csv"
    df.to_csv(out_csv, index=False)
    # Summary
    summary = {
        "date": target.isoformat(),
        "n_games": int(len(df)),
        "n_with_pred": int(df["pred_total"].notna().sum()) if "pred_total" in df.columns else 0,
        "mae_model_total": float(df["err_model_total"].abs().mean()) if "err_model_total" in df.columns else None,
        "mae_closing_total": float(df["err_closing_total"].abs().mean()) if "err_closing_total" in df.columns else None,
        "beat_rate_vs_closing": float(df["model_better_than_closing"].mean()) if "model_better_than_closing" in df.columns else None,
        "n_picks": int(df["bet"].notna().sum()) if "bet" in df.columns else 0,
        "n_wins": int((df["pick_result"] == "win").sum()) if "pick_result" in df.columns else 0,
        "n_losses": int((df["pick_result"] == "loss").sum()) if "pick_result" in df.columns else 0,
        "n_pushes": int((df["pick_result"] == "push").sum()) if "pick_result" in df.columns else 0,
        # Extended metrics
        "mae_model_total_1h": float(df["err_model_total_1h"].abs().mean()) if "err_model_total_1h" in df.columns else None,
        "mae_model_total_2h": float(df["err_model_total_2h"].abs().mean()) if "err_model_total_2h" in df.columns else None,
        "ats_cover_rate_full": float((df["ats_result"] == "home_cover").mean()) if "ats_result" in df.columns else None,
        "ats_cover_rate_1h": float((df["ats_result_1h"] == "home_cover").mean()) if "ats_result_1h" in df.columns else None,
        "brier_ml": float(df["brier_ml"].mean()) if "brier_ml" in df.columns else None,
        "logloss_ml": float(df["logloss_ml"].mean()) if "logloss_ml" in df.columns else None,
    }
    # Add picks_raw summary if available
    try:
        if raw_outcomes is not None and not raw_outcomes.empty and "result" in raw_outcomes.columns:
            rr = raw_outcomes.copy()
            rr = rr[pd.notna(rr["result"])].copy()
            summary["picks_raw_n"] = int(len(rr))
            summary["picks_raw_win_rate"] = float((rr["result"].astype(str) == "win").mean()) if not rr.empty else None
            for mk in ["totals","spreads","moneyline"]:
                sub = rr[rr.get("market").astype(str).str.lower() == mk]
                summary[f"picks_raw_{mk}_n"] = int(len(sub))
                summary[f"picks_raw_{mk}_win_rate"] = float((sub["result"].astype(str) == "win").mean()) if not sub.empty else None
    except Exception:
        pass
    pd.DataFrame([summary]).to_csv(out_dir / f"summary_{target.isoformat()}.csv", index=False)
    (out_dir / f"summary_{target.isoformat()}.json").write_text(json.dumps(summary, indent=2))
    print(f"[green]Wrote daily results to[/green] {out_csv}")


@app.command(name="daily-results-range")
def daily_results_range(
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD (inclusive)"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (inclusive)"),
    games_path: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Games CSV with final scores for the season"),
    preds_path: Path = typer.Option(settings.outputs_dir / "predictions_all.csv", help="Predictions CSV over the season (fallback to predictions_week per date)"),
    closing_merged: Path | None = typer.Option(settings.outputs_dir / "games_with_closing.csv", help="Optional merged games_with_closing.csv with totals lines"),
    picks_path: Path | None = typer.Option(settings.outputs_dir / "picks_clean.csv", help="Optional picks CSV to reconcile picks results"),
    picks_raw_path: Path | None = typer.Option(settings.outputs_dir / "picks_raw.csv", help="Optional expanded picks CSV for multi-market reconciliation"),
    out_dir: Path = typer.Option(settings.outputs_dir / "daily_results", help="Output directory for per-day results and summary"),
):
    """Generate daily results for a date range by iterating unique dates from games_path.

    If start/end are omitted, uses the min/max date present in games_path.
    """
    # Load games to enumerate dates
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))
    if "date" not in games.columns:
        print("[red]games file missing 'date' column[/red]")
        raise typer.Exit(code=1)
    gdates = pd.to_datetime(games["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist()
    gdates = sorted(set(gdates))
    if not gdates:
        print("[yellow]No dates found in games file.[/yellow]")
        raise typer.Exit(code=0)
    if start:
        gdates = [d for d in gdates if d >= start]
    if end:
        gdates = [d for d in gdates if d <= end]
    if not gdates:
        print("[yellow]No dates in requested range.[/yellow]")
        raise typer.Exit(code=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for d in gdates:
        try:
            daily_results(
                date=d,
                games_path=games_path,
                preds_path=preds_path,
                closing_merged=closing_merged,
                picks_path=picks_path,
                picks_raw_path=picks_raw_path,
                out_dir=out_dir,
            )
            n_ok += 1
        except SystemExit:
            # daily_results might exit early with code 0 if no games for date
            continue
        except Exception as e:
            print(f"[yellow]Failed to build results for {d}:[/yellow] {e}")
            continue
    print(f"[green]Finished daily-results-range for {n_ok} dates.[/green]")


@app.command(name="update-branding")
def update_branding(
    out_csv: Path = typer.Option(settings.data_dir / "team_branding.csv", help="Output branding CSV path"),
    overwrite: bool = typer.Option(True, help="Overwrite existing file if present"),
    timeout: int = typer.Option(20, help="HTTP timeout (s) for ESPN request"),
):
    """Fetch all Division I team logos/colors from ESPN and write team_branding.csv.

    Columns written: team,logo,primary_color,secondary_color,text_color,espn_id,abbreviation
    Existing file will be replaced unless --overwrite False.
    """
    if out_csv.exists() and not overwrite:
        print(f"[yellow]File exists and overwrite disabled:[/yellow] {out_csv}")
        raise typer.Exit(code=0)
    try:
        print("[cyan]Fetching ESPN team branding...[/cyan]")
        df = fetch_espn_branding(timeout=timeout)
    except Exception as e:
        print(f"[red]Failed to fetch branding:[/red] {e}")
        raise typer.Exit(code=1)
    if df.empty:
        print("[red]No teams returned from ESPN API.[/red]")
        raise typer.Exit(code=1)
    write_branding_csv(out_csv, df)
    print(f"[green]Wrote branding for {len(df)} teams to[/green] {out_csv}")
    # Show a small sample for confirmation
    try:
        sample_cols = [c for c in ["team","logo","primary_color","secondary_color","text_color"] if c in df.columns]
        print(df.head(5)[sample_cols])
    except Exception:
        pass


@app.command(name="update-tuning")
def update_tuning(
    results_dir: Path = typer.Option(settings.outputs_dir / "daily_results", help="Directory with results_*.csv files"),
    window_days: int = typer.Option(7, help="Number of most-recent days to include in bias estimation"),
    min_valid_games: int = typer.Option(10, help="Require at least this many games with valid actual totals"),
    cap_abs_bias: float = typer.Option(25.0, help="Cap absolute totals bias to this value to avoid bad data drift"),
    out: Path = typer.Option(settings.outputs_dir / "model_tuning.json", help="Output JSON with tuning parameters"),
):
    """Compute simple tuning parameters (e.g., totals bias) from recent daily results.

    Valid games must have actual_total > 0 and not NaN.
    """
    if not results_dir.exists():
        print(f"[yellow]{results_dir} does not exist; nothing to tune.[/yellow]")
        raise typer.Exit(code=0)
    # Collect recent files by date in filename
    files = sorted([p for p in results_dir.glob("results_*.csv")])
    if not files:
        print("[yellow]No daily results found.[/yellow]")
        raise typer.Exit(code=0)
    # Keep last window_days by filename date
    def parse_date(p: Path) -> dt.date | None:
        try:
            s = p.stem.split("_")[-1]
            return dt.date.fromisoformat(s)
        except Exception:
            return None
    files = sorted([(parse_date(p), p) for p in files if parse_date(p) is not None], key=lambda x: x[0])
    files = [p for (d, p) in files if d is not None][-window_days:]
    if not files:
        print("[yellow]No recent results to tune from.[/yellow]")
        raise typer.Exit(code=0)
    # Compute totals bias
    errs = []
    n_valid = 0
    for p in files:
        try:
            df = pd.read_csv(p)
            if {"actual_total","pred_total"}.issubset(df.columns):
                # Only consider rows with real final scores
                mask = pd.to_numeric(df["actual_total"], errors="coerce").fillna(0) > 0
                if mask.any():
                    diffs = (pd.to_numeric(df.loc[mask, "pred_total"], errors="coerce") - pd.to_numeric(df.loc[mask, "actual_total"], errors="coerce")).dropna()
                    errs.extend(diffs.tolist())
                    n_valid += int(mask.sum())
        except Exception:
            continue
    if not errs or n_valid < int(min_valid_games):
        print("[yellow]Insufficient valid games to compute tuning; skipping write.[/yellow]")
        raise typer.Exit(code=0)
    bias = float(pd.Series(errs).mean())
    # Cap bias to avoid outliers from bad data
    if abs(bias) > cap_abs_bias:
        print(f"[yellow]Computed bias {bias:.3f} exceeds cap {cap_abs_bias}; capping.[/yellow]")
        bias = cap_abs_bias if bias > 0 else -cap_abs_bias
    tuning = {
        "totals_bias": bias,
        "window_days": window_days,
        "n_errors": len(errs),
        "n_valid_games": n_valid,
        "min_valid_games": int(min_valid_games),
        "source": "recent-daily-results",
    }
    out.write_text(json.dumps(tuning, indent=2))
    print(f"[green]Wrote model tuning to[/green] {out} -> {{'totals_bias': {bias:.3f}, 'n_valid_games': {n_valid}}}")


@app.command()
def make_dummy_model(out: Path = typer.Option(settings.outputs_dir / "dummy.onnx", help="Output ONNX path"),
                     in_dim: int = 64,
                     out_dim: int = 4):
    path = create_dummy_regression_model(out, in_dim=in_dim, out_dim=out_dim)
    print(f"[green]Wrote dummy model:[/green] {path}")


@app.command()
def predict_dummy(model: Path = typer.Option(settings.outputs_dir / "dummy.onnx"),
                  batch: int = 8,
                  in_dim: int = 64):
    providers = OnnxPredictor.describe_available()
    print("Available ONNX providers:", providers)
    if not providers:
        # Try NumPy fallback if the dummy is linear
        if NumpyLinearPredictor.can_load(str(model)):
            print("[yellow]ONNX Runtime not available; using NumPy fallback predictor.[/yellow]")
            pred = NumpyLinearPredictor(str(model))
        else:
            print(
                "onnxruntime not found. Either install a wheel if available or set NCAAB_ORT_DLL_DIR to the ORT DLL folder. "
                "For QNN NPU, use an ORT build with QNN EP and set NCAAB_QNN_SDK_DIR/NCAAB_QNN_BACKEND_DLL."
            )
            return
    else:
        pred = OnnxPredictor(str(model))
    x = np.random.randn(batch, in_dim).astype(np.float32)
    y = pred.predict(x)
    print(f"Providers used: {pred.providers}")
    print("Pred shape:", y.shape)
    print(y[: min(3, batch)])


@app.command(name="fetch-games")
def fetch_games(
    season: int = typer.Option(..., help="Season year (by calendar year of season start)"),
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD; defaults to Nov 1 of season"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD; defaults to Apr 15 following season"),
    provider: str = typer.Option("espn", help="Data provider: 'espn' (default) or 'ncaa'"),
    use_cache: bool = True,
    out: Path = typer.Option(settings.outputs_dir / "games.parquet", help="Output Parquet file"),
):
    """Fetch real NCAA D1 scoreboard data by date range and write to Parquet."""
    if start:
        start_date = dt.date.fromisoformat(start)
    else:
        start_date = dt.date(season, 11, 1)
    if end:
        end_date = dt.date.fromisoformat(end)
    else:
        end_date = dt.date(season + 1, 4, 15)

    rows = []
    total = 0
    iterator = iter_games_espn if provider.lower() == "espn" else iter_games_ncaa
    for res in iterator(start_date, end_date, use_cache=use_cache):
        g = res.games
        total += len(g)
        for game in g:
            rows.append(game.model_dump())
    if not rows:
        print("[yellow]No games found in date range.[/yellow]")
        return
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Respect .csv suffix explicitly
    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
        print(f"[green]Wrote {len(df)} games to[/green] {out}")
    else:
        try:
            df.to_parquet(out, index=False)
            print(f"[green]Wrote {len(df)} games to[/green] {out}")
        except Exception as e:
            # Fallback to CSV if parquet engine not available on this platform
            csv_out = out.with_suffix(".csv")
            df.to_csv(csv_out, index=False)
            print(
                f"[yellow]Parquet engine unavailable ({e}). Wrote CSV instead to[/yellow] {csv_out}"
            )


@app.command(name="build-features")
def build_features(
    games_path: Path = typer.Argument(..., help="Path to games CSV/Parquet produced by fetch-games"),
    out: Path = typer.Option(settings.outputs_dir / "features.csv", help="Output features file (CSV)"),
    window: int = 5,
    include_schedule: bool = typer.Option(True, help="Include rest/B2B/neutral schedule features"),
    include_adj_ratings: bool = typer.Option(True, help="Include basic opponent-adjusted margin ratings"),
    boxscores_path: Path | None = typer.Option(None, help="Optional boxscores CSV to derive four-factor rolling features"),
    include_enrichment: bool = typer.Option(True, help="Add secondary derived features (rest_diff, tempo diffs, synergies, volatility)"),
):
    """Build rolling features from real games data and write to CSV.

    Optionally merges schedule context and rolling four-factors from ESPN boxscores.
    """
    # Load CSV/Parquet flexibly
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            # fallback to CSV with same stem if parquet unavailable
            csv_alt = games_path.with_suffix(".csv")
            games = pd.read_csv(csv_alt)

    # Ensure game_id consistent string type
    if "game_id" in games.columns:
        games["game_id"] = games["game_id"].astype(str)
    # Build both short and long windows to add stable team fixed-effects style features
    # Build rolling features (+ volatility)
    from .features.build import build_team_rolling_features
    feats = build_team_rolling_features(games, windows=[5, 15], add_volatility=True)
    if "game_id" in feats.columns:
        feats["game_id"] = feats["game_id"].astype(str)

    # Schedule context
    if include_schedule:
        sched = compute_rest_days(games)
        feats = feats.merge(sched, on="game_id", how="left")

    # Four-factors from boxscores
    bs = None
    if boxscores_path is not None and boxscores_path.exists():
        try:
            from .features.factors import build_four_factor_rolling_features
            bs = pd.read_csv(boxscores_path)
            if "game_id" in bs.columns:
                bs["game_id"] = bs["game_id"].astype(str)
            ff = build_four_factor_rolling_features(games, bs, window=window)
            feats = feats.merge(ff, on="game_id", how="left")
        except Exception as e:
            print(f"[yellow]Skipping boxscore features due to error:[/yellow] {e}")

    # Opponent-adjusted ratings from margins (simple ridge)
    if include_adj_ratings:
        try:
            from .features.ratings import build_adj_rating_features
            rf = build_adj_rating_features(games)
            feats = feats.merge(rf, on="game_id", how="left")
        except Exception as e:
            print(f"[yellow]Skipping adjusted ratings due to error:[/yellow] {e}")

        # If we have boxscores, also compute opponent-adjusted Off/Def/Tempo ratings
        if bs is not None:
            try:
                from .features.ratings import build_adj_offdef_tempo_features
                odt = build_adj_offdef_tempo_features(games, bs)
                feats = feats.merge(odt, on="game_id", how="left")
            except Exception as e:
                print(f"[yellow]Skipping adjusted Off/Def/Tempo due to error:[/yellow] {e}")

    # Ensure one row per game_id (guard against upstream duplication)
    if "game_id" in feats.columns:
        before = len(feats)
        feats = feats.drop_duplicates(subset=["game_id"], keep="first")
        dropped = before - len(feats)
        if dropped > 0:
            print(f"[yellow]Dropped {dropped} duplicate feature rows by game_id.[/yellow]")

    out.parent.mkdir(parents=True, exist_ok=True)
    # Targets: total and margin derived from final scores when present
    if {"home_score", "away_score"}.issubset(games.columns):
        try:
            gsub = games[["game_id","home_score","away_score"]].copy()
            gsub["game_id"] = gsub["game_id"].astype(str)
            gsub["target_total"] = pd.to_numeric(gsub["home_score"], errors="coerce") + pd.to_numeric(gsub["away_score"], errors="coerce")
            gsub["target_margin"] = pd.to_numeric(gsub["home_score"], errors="coerce") - pd.to_numeric(gsub["away_score"], errors="coerce")
            feats = feats.merge(gsub[["game_id","target_total","target_margin"]], on="game_id", how="left")
        except Exception as e:
            print(f"[yellow]Failed to attach target_total/target_margin:[/yellow] {e}")
    # Coalesce any suffixed target columns (e.g., target_total_x, target_total_y) to canonical names
    for base in ["target_total", "target_margin"]:
        if base not in feats.columns:
            candidates = [c for c in feats.columns if c.startswith(base + "_")]
            if candidates:
                # Prefer the first non-null-rich candidate
                chosen = None
                best_nn = -1
                for c in candidates:
                    nn = feats[c].notna().sum()
                    if nn > best_nn:
                        best_nn = nn; chosen = c
                if chosen:
                    feats[base] = feats[chosen]
    # Drop duplicate suffixed target columns to reduce clutter
    for c in list(feats.columns):
        if c.startswith("target_total_") or c.startswith("target_margin_"):
            if c.replace("_x","").replace("_y","") in ["target_total","target_margin"]:
                # keep canonical only
                if c != "target_total" and c != "target_margin":
                    feats = feats.drop(columns=[c])
    # Secondary enrichment layer (pure derivations; safe if columns absent)
    if include_enrichment:
        try:
            from .features.enrichment import enrich_game_features
            feats = enrich_game_features(feats)
        except Exception as e:
            print(f"[yellow]Skipping enrichment features due to error:[/yellow] {e}")

    feats.to_csv(out, index=False)
    print(f"[green]Wrote features to[/green] {out} with {len(feats)} rows")


@app.command(name="build-schedule-features")
def build_schedule_features(
    games_path: Path = typer.Argument(..., help="Games CSV/Parquet produced by fetch-games"),
    out: Path = typer.Option(settings.outputs_dir / "schedule_features.csv", help="Output schedule features CSV"),
):
    """Compute schedule context features: rest days, B2B flags, and neutral site indicator."""
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))
    sched = compute_rest_days(games)
    out.parent.mkdir(parents=True, exist_ok=True)
    sched.to_csv(out, index=False)
    print(f"[green]Wrote schedule features to[/green] {out} ({len(sched)} rows)")


@app.command(name="train-baseline")
def train_baseline_cmd(
    features_csv: Path = typer.Argument(..., help="Features CSV from build-features"),
    out_dir: Path = typer.Option(settings.outputs_dir / "models", help="Directory to write models"),
    loss_totals: str = typer.Option("ridge", help="Loss for totals target: ridge|huber"),
    huber_delta: float = typer.Option(8.0, help="Huber delta (threshold) for totals when loss=huber"),
):
    """Train simple baseline models for totals and margin and export ONNX if possible."""
    res = train_baseline(features_csv, out_dir, alpha=1.0, loss_totals=loss_totals, huber_delta=huber_delta)
    print("[green]Training complete.[/green]")
    print(res)


@app.command(name="train-halves")
def train_halves_cmd(
    features_csv: Path = typer.Argument(..., help="Features CSV from build-features"),
    games_csv: Path = typer.Argument(..., help="Games CSV with 1H/2H scoring columns to derive half targets"),
    out_dir: Path = typer.Option(settings.outputs_dir / "models_halves", help="Directory to write half models"),
    alpha: float = typer.Option(1.0, help="Ridge regularization strength"),
):
    """Train team-level half models (1H/2H totals and margins) if half targets available."""
    try:
        res = train_half_models(features_csv, games_csv, out_dir, alpha=alpha)
        print("[green]Half-model training complete.[/green]")
        print(res)
    except Exception as e:
        print(f"[red]Failed to train half models:[/red] {e}")


@app.command(name="predict-baseline")
def predict_baseline_cmd(
    features_csv: Path = typer.Argument(..., help="Features CSV to score"),
    models_dir: Path = typer.Option(settings.outputs_dir / "models", help="Directory with baseline_* ONNX and feature_columns.txt"),
    out: Path = typer.Option(settings.outputs_dir / "predictions.csv", help="Output CSV with predictions"),
    apply_guardrails: bool = typer.Option(False, help="Apply tempo/off/def derived blending guardrail for implausibly low totals (<105 or <75% derived) and mark pred_total_adjusted"),
    half_ratio: float = typer.Option(0.485, help="Share of full-game total expected in 1H for derived half projections (rest goes to 2H)"),
    halves_models_dir: Path | None = typer.Option(None, help="Optional directory with trained half models and feature_columns_halves.txt to override ratio-based halves"),
):
    """Run ONNX baseline models on a features CSV and write predictions (total, margin)."""
    # Load features
    feats = pd.read_csv(features_csv)
    # Load feature column order
    cols_path = models_dir / "feature_columns.txt"
    if not cols_path.exists():
        print(f"[red]Missing feature_columns.txt in {models_dir}[/red]")
        raise typer.Exit(code=1)
    cols = [c.strip() for c in cols_path.read_text(encoding="utf-8").splitlines() if c.strip()]
    # Build X matrix in correct order
    X = feats.reindex(columns=cols).fillna(0.0).to_numpy(dtype=np.float32)

    providers = OnnxPredictor.describe_available()

    # Predict with totals model
    total_model_path = models_dir / "baseline_target_total.onnx"
    margin_model_path = models_dir / "baseline_target_margin.onnx"
    if providers:
        pred_total = OnnxPredictor(str(total_model_path))
        pred_margin = OnnxPredictor(str(margin_model_path))
    else:
        # Fallback to NumPy linear predictor if possible
        if NumpyLinearPredictor.can_load(str(total_model_path)) and NumpyLinearPredictor.can_load(str(margin_model_path)):
            print("[yellow]ONNX Runtime not available; using NumPy fallback predictors.[/yellow]")
            pred_total = NumpyLinearPredictor(str(total_model_path))
            pred_margin = NumpyLinearPredictor(str(margin_model_path))
        else:
            print(
                "onnxruntime not found and models are not supported by NumPy fallback. "
                "Install ORT (wheel or local build) and set NCAAB_ORT_DLL_DIR/NCAAB_ORT_PY_DIR."
            )
            raise typer.Exit(code=1)
    y_total = pred_total.predict(X).reshape(-1)
    y_margin = pred_margin.predict(X).reshape(-1)

    out_df = feats[["game_id", "date", "home_team", "away_team"]].copy()
    out_df["pred_total"] = y_total
    out_df["pred_margin"] = y_margin
    # Early filter: remove placeholder TBD games before downstream adjustments
    try:
        mask_tbd = (out_df["home_team"].astype(str).str.upper() == "TBD") | (out_df["away_team"].astype(str).str.upper() == "TBD")
        if mask_tbd.any():
            removed = int(mask_tbd.sum())
            out_df = out_df.loc[~mask_tbd].reset_index(drop=True)
            print(f"[yellow]Filtered {removed} TBD placeholder games from baseline predictions[/yellow]")
    except Exception as _e_tbd:
        print(f"[yellow]TBD filtering skipped (baseline path):[/yellow] {_e_tbd}")

    # Guardrail blending based on rating + tempo derived total
    if apply_guardrails:
        try:
            # Compute derived expected total only when sufficient rating + tempo data present.
            derived_vals = []
            adjusted_flags = []
            avail_count = 0
            missing_count = 0
            for _, r in feats.iterrows():
                try:
                    have_tempo = pd.notna(r.get("home_tempo_rating")) and pd.notna(r.get("away_tempo_rating"))
                    have_ratings = all(pd.notna(r.get(k)) for k in ["home_off_rating","away_off_rating","home_def_rating","away_def_rating"])  # noqa: E501
                    if not (have_tempo and have_ratings):
                        # Insufficient data; skip deriving to avoid uniform fallback constant (e.g., 112)
                        derived_total = np.nan
                        missing_count += 1
                    else:
                        tempo_avg = (float(r.get("home_tempo_rating")) + float(r.get("away_tempo_rating"))) / 2.0
                        off_home = float(r.get("home_off_rating"))
                        off_away = float(r.get("away_off_rating"))
                        def_home = float(r.get("home_def_rating"))
                        def_away = float(r.get("away_def_rating"))
                        exp_home_pp100 = np.clip(off_home - def_away, 65, 140)  # widen range for early season variance
                        exp_away_pp100 = np.clip(off_away - def_home, 65, 140)
                        derived_total = (exp_home_pp100 + exp_away_pp100) / 100.0 * tempo_avg
                        # Softer clamp: allow lower totals (<110) to pass; just bound extreme tails
                        derived_total = float(np.clip(derived_total, 95, 195))
                        avail_count += 1
                except Exception:
                    derived_total = np.nan
                    missing_count += 1
                derived_vals.append(derived_total)
            out_df["pred_total_raw"] = out_df["pred_total"].astype(float)
            out_df["derived_total"] = derived_vals
            blended = []
            for pt, dv in zip(out_df["pred_total_raw"].tolist(), derived_vals):
                # Skip blending when derived_total missing (avoid uniform constant)
                if np.isnan(pt) or np.isnan(dv):
                    blended.append(pt)
                    adjusted_flags.append(False)
                    continue
                if pt < 103 or (dv > 0 and pt < 0.70 * dv):
                    b = 0.45 * pt + 0.55 * dv  # bias slightly toward derived when implausibly low
                    b = float(np.clip(b, 95, 195))
                    blended.append(b)
                    adjusted_flags.append(True)
                else:
                    blended.append(pt)
                    adjusted_flags.append(False)
            out_df["pred_total"] = blended
            out_df["pred_total_adjusted"] = adjusted_flags
            try:
                out_df["derived_total_available_rows"] = avail_count
                out_df["derived_total_missing_rows"] = missing_count
            except Exception:
                pass
        except Exception:
            pass
    # Uniform prediction diagnostic
    try:
        pt_series_diag = pd.to_numeric(out_df.get("pred_total"), errors="coerce")
        if pt_series_diag.notna().sum() > 10:
            vc = pt_series_diag.value_counts()
            if (vc.iloc[0] / pt_series_diag.notna().sum()) > 0.9:
                out_df["pred_total_uniform_flag"] = True
                out_df["pred_total_uniform_value"] = vc.index[0]
                out_df["pred_total_unique_count"] = pt_series_diag.nunique()
    except Exception:
        pass

    # Half projections: prefer trained half models if provided, else ratio-based fallback
    def _ratio_fill():
        out_df["pred_total_1h"] = pd.to_numeric(out_df["pred_total"], errors="coerce") * half_ratio
        out_df["pred_total_2h"] = pd.to_numeric(out_df["pred_total"], errors="coerce") * (1.0 - half_ratio)
        out_df["pred_margin_1h"] = pd.to_numeric(out_df["pred_margin"], errors="coerce") * 0.5
        out_df["pred_margin_2h"] = pd.to_numeric(out_df["pred_margin"], errors="coerce") * 0.5

    used_half_models = False
    if halves_models_dir is not None and Path(halves_models_dir).exists():
        try:
            cols_halves_path = Path(halves_models_dir) / "feature_columns_halves.txt"
            if cols_halves_path.exists():
                cols_h = [c.strip() for c in cols_halves_path.read_text(encoding="utf-8").splitlines() if c.strip()]
                Xh = feats.reindex(columns=cols_h).fillna(0.0).to_numpy(dtype=np.float32)
                providers_h = OnnxPredictor.describe_available()
                def _load(name: str):
                    p = Path(halves_models_dir) / f"baseline_{name}.onnx"
                    if not p.exists():
                        return None
                    return OnnxPredictor(str(p)) if providers_h else (NumpyLinearPredictor(str(p)) if NumpyLinearPredictor.can_load(str(p)) else None)
                m_t1 = _load("target_total_1h")
                m_t2 = _load("target_total_2h")
                m_m1 = _load("target_margin_1h")
                m_m2 = _load("target_margin_2h")
                if any([m_t1, m_t2, m_m1, m_m2]):
                    if m_t1 is not None:
                        out_df["pred_total_1h"] = m_t1.predict(Xh).reshape(-1)
                    if m_t2 is not None:
                        out_df["pred_total_2h"] = m_t2.predict(Xh).reshape(-1)
                    if m_m1 is not None:
                        out_df["pred_margin_1h"] = m_m1.predict(Xh).reshape(-1)
                    if m_m2 is not None:
                        out_df["pred_margin_2h"] = m_m2.predict(Xh).reshape(-1)
                    used_half_models = True
        except Exception:
            used_half_models = False
    if not used_half_models:
        try:
            _ratio_fill()
        except Exception:
            out_df["pred_total_1h"] = np.nan
            out_df["pred_total_2h"] = np.nan
            out_df["pred_margin_1h"] = np.nan
            out_df["pred_margin_2h"] = np.nan
    out_df.to_csv(out, index=False)
    print(f"[green]Wrote predictions to[/green] {out}")


@app.command(name="fetch-odds")
def fetch_odds(
    season: int = typer.Option(dt.datetime.now().year, help="Season year (unused by API; for bookkeeping)"),
    region: str = typer.Option("us", help="Odds region for TheOddsAPI (e.g., us, uk, eu, au)"),
    out: Path = typer.Option(settings.outputs_dir / "odds.csv", help="Output CSV of odds"),
):
    """Fetch current odds from TheOddsAPI and write to CSV.

    Note: TheOddsAPI v4 provides current odds, not historical, without event IDs and specific plan.
    This command fetches available markets (h2h, spreads, totals) for NCAAB and writes a flat table.
    Requires NCAAB_THEODDS_API_KEY in env/.env.
    """
    adapter = TheOddsAPIAdapter(region=region)
    rows = []
    for o in adapter.iter_odds(season):
        rows.append(o.model_dump())
    if not rows:
        print("[yellow]No odds returned by TheOddsAPI. Check key/plan/region.[/yellow]")
        raise typer.Exit(code=1)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[green]Wrote {len(df)} odds rows to[/green] {out}")


@app.command(name="fetch-odds-history")
def fetch_odds_history(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    region: str = typer.Option("us", help="Odds region for TheOddsAPI (e.g., us, uk, eu, au)"),
    markets: str = typer.Option(
        "h2h,spreads,totals,spreads_1st_half,totals_1st_half,spreads_2nd_half,totals_2nd_half",
        help="Comma-separated markets to request; include halves variants if your plan supports them.",
    ),
    out_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory to write per-day CSVs"),
    mode: str = typer.Option("current", help="'current' to snapshot current odds; 'history' to call odds-history endpoint"),
):
    """Fetch expanded odds snapshots across a date range and write partitioned CSV files.

    - mode=current: uses /odds endpoint and normalizes markets (fast, good for daily snapshots)
    - mode=history: uses /odds-history for all events found on each day (requires premium access)
    """
    adapter = TheOddsAPIAdapter(region=region)
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)
    if d1 < d0:
        print("[red]End date must be >= start date[/red]")
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)

    cur = d0
    total_rows = 0
    while cur <= d1:
        date_iso = cur.isoformat()
        rows = []
        try:
            if mode == "current":
                for row in adapter.iter_current_odds_expanded(markets=markets, date_iso=date_iso):
                    rows.append(row.model_dump())
            else:
                # history mode: list events for the date, then pull odds-history
                events = adapter.list_events_by_date(date_iso)
                event_ids = [str(e.get("id")) for e in events if e.get("id")]
                # Fallback: for past dates, events endpoint may be empty; discover via current odds snapshot
                if not event_ids:
                    try:
                        discovered = []
                        for r in adapter.iter_current_odds_expanded(markets=markets, date_iso=date_iso):
                            # Each row normalizes from an event; extract id from hidden event context via quotes not available here
                            # Instead, do a light second call to events-no-date and filter by commence_time
                            discovered.append(r)
                        if discovered:
                            # As iter_current_odds_expanded yields OddsHistoryRow, we cannot extract event_id directly unless present
                            # But our normalization sets 'event_id' on the row using event.id
                            event_ids = sorted({str(x.event_id) for x in discovered if getattr(x, 'event_id', None)})
                    except Exception:
                        pass
                if event_ids:
                    for row in adapter.iter_odds_history_for_events(event_ids, markets=markets):
                        rows.append(row.model_dump())
        except Exception as e:
            # Improve guidance for common auth errors
            msg = str(e)
            if "401" in msg or "Unauthorized" in msg:
                print(
                    f"[red]Unauthorized when calling TheOddsAPI for {date_iso}.[/red] "
                    "Check NCAAB_THEODDS_API_KEY in your environment/.env. It should be just the key (no trailing '|' or spaces)."
                )
            else:
                print(f"[red]Error fetching odds for {date_iso}:[/red] {e}")
            rows = []
        if rows:
            df = pd.DataFrame(rows)
            out_path = out_dir / f"odds_{date_iso}.csv"
            df.to_csv(out_path, index=False)
            print(f"[green]Wrote {len(df)} rows for {date_iso} to[/green] {out_path}")
            total_rows += len(df)
        else:
            print(f"[yellow]No odds data for {date_iso}. Check API plan/markets/region.[/yellow]")
        cur += dt.timedelta(days=1)


@app.command(name="fetch-odds-multiregion")
def fetch_odds_multiregion(
    date: str = typer.Option(..., help="Target date YYYY-MM-DD"),
    regions: str = typer.Option("us,uk,eu,au", help="Comma-separated TheOddsAPI regions to aggregate"),
    markets: str = typer.Option(
        "h2h,spreads,totals,spreads_1st_half,totals_1st_half,spreads_2nd_half,totals_2nd_half",
        help="Markets to request; halves variants included if plan supports them.",
    ),
    out_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory to write odds_YYYY-MM-DD.csv"),
):
    """Fetch a per-day odds snapshot by aggregating multiple regions and de-duplicating.

    Writes odds_history/odds_<date>.csv. Useful when a single region has gaps.
    """
    try:
        dt.date.fromisoformat(date)
    except Exception:
        print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    reg_list = [r.strip() for r in regions.split(",") if r.strip()]
    for reg in reg_list:
        try:
            adapter = TheOddsAPIAdapter(region=reg)
            for row in adapter.iter_current_odds_expanded(markets=markets, date_iso=date):
                all_rows.append(row.model_dump())
            print(f"[green]{reg}[/green] collected {sum(1 for _ in all_rows)} rows (cumulative)")
        except Exception as e:
            print(f"[yellow]Region {reg} failed:[/yellow] {e}")
            continue
    if not all_rows:
        print("[yellow]No odds collected from any region.[/yellow]")
        raise typer.Exit(code=1)
    df = pd.DataFrame(all_rows)
    # De-duplicate across regions: prefer unique (event_id,book,market,period) when available
    subset = [c for c in ["event_id","book","market","period"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")
    else:
        # Fallback: unique by teams + book + market + period + total
        alt = [c for c in ["home_team_name","away_team_name","book","market","period","total"] if c in df.columns]
        if alt:
            df = df.drop_duplicates(subset=alt, keep="first")
    out_path = out_dir / f"odds_{date}.csv"
    df.to_csv(out_path, index=False)
    print(f"[green]Wrote {len(df)} rows to[/green] {out_path}")


@app.command(name="fetch-odds-window")
def fetch_odds_window(
    date: str = typer.Option(..., help="Anchor date YYYY-MM-DD"),
    days_before: int = typer.Option(1, help="Include this many days before the anchor"),
    days_after: int = typer.Option(1, help="Include this many days after the anchor"),
    regions: str = typer.Option("us,uk,eu,au", help="Comma-separated regions to aggregate"),
    markets: str = typer.Option(
        "h2h,spreads,totals,spreads_1st_half,totals_1st_half,spreads_2nd_half,totals_2nd_half",
        help="Markets to request; halves variants included if plan supports them.",
    ),
    out_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory for odds_YYYY-MM-DD.csv snapshots"),
):
    """Aggregate per-day odds snapshots across a small date window and de-duplicate.

    Writes combined CSV per day in the window using multi-region snapshots.
    """
    try:
        anchor = dt.date.fromisoformat(date)
    except Exception:
        print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = list(range(-abs(days_before), abs(days_after) + 1))
    for off in offsets:
        cur = anchor + dt.timedelta(days=off)
        all_rows: list[dict] = []
        for reg in [r.strip() for r in regions.split(",") if r.strip()]:
            try:
                adapter = TheOddsAPIAdapter(region=reg)
                for row in adapter.iter_current_odds_expanded(markets=markets, date_iso=cur.isoformat()):
                    all_rows.append(row.model_dump())
            except Exception as e:
                print(f"[yellow]{cur} {reg} failed:[/yellow] {e}")
                continue
        if not all_rows:
            print(f"[yellow]{cur} -> no odds rows[/yellow]")
            continue
        df = pd.DataFrame(all_rows)
        subset = [c for c in ["event_id","book","market","period"] if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="first")
        else:
            alt = [c for c in ["home_team_name","away_team_name","book","market","period","total"] if c in df.columns]
            if alt:
                df = df.drop_duplicates(subset=alt, keep="first")
        out_path = out_dir / f"odds_{cur.isoformat()}.csv"
        df.to_csv(out_path, index=False)
        print(f"[green]Wrote {len(df)} rows ->[/green] {out_path}")


@app.command(name="probe-odds-events")
def probe_odds_events(
    date: str = typer.Option(..., help="Target date YYYY-MM-DD"),
    region: str = typer.Option("us", help="Region for TheOddsAPI"),
    games_csv: Path | None = typer.Option(settings.outputs_dir / "games_all.csv", help="Optional games file to compare"),
    out: Path = typer.Option(settings.outputs_dir / "events_probe.csv", help="Output CSV of events for inspection"),
    fallback_no_date: bool = typer.Option(True, help="If date endpoint empty, also list events without date and filter by commence_time"),
    unmatched_out: Path | None = typer.Option(settings.outputs_dir / "diagnostics" / "unmatched_eligible_pairs.csv", help="If set, write unmatched eligible D1 game pairs for date"),
    d1_csv: Path | None = typer.Option(settings.data_dir / "d1_conferences.csv", help="CSV with D1 teams (column like team/school/name)"),
):
    """List provider events for a date and optionally compute intersection with games (by normalized team slugs)."""
    try:
        d = dt.date.fromisoformat(date)
    except Exception:
        print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
        raise typer.Exit(code=1)
    adapter = TheOddsAPIAdapter(region=region)
    events = []
    try:
        events = adapter.list_events_by_date(date)
    except Exception as e:
        print(f"[yellow]events-by-date failed:[/yellow] {e}")
        events = []
    if (not events) and fallback_no_date:
        try:
            all_e = adapter.list_events_no_date()
            # Filter by commence_time calendar date
            filt = []
            for ev in all_e or []:
                ct = ev.get("commence_time")
                try:
                    if isinstance(ct, str):
                        cdt = dt.datetime.fromisoformat(ct.replace("Z", "+00:00")).date()
                        if cdt == d:
                            filt.append(ev)
                except Exception:
                    pass
            events = filt
            print(f"[cyan]no-date fallback[/cyan] -> {len(events)} events on {date}")
        except Exception as e:
            print(f"[yellow]no-date events failed:[/yellow] {e}")
    if not events:
        print(f"[yellow]No provider events for {date} (region={region}).[/yellow]")
        raise typer.Exit(code=0)
    # Build output rows with normalized slugs
    rows = []
    from .data.team_normalize import canonical_slug as _canon
    for ev in events:
        eid = str(ev.get("id")) if ev.get("id") else None
        home = ev.get("home_team")
        away = ev.get("away_team")
        ct = ev.get("commence_time")
        rows.append({
            "event_id": eid,
            "home_team_name": home,
            "away_team_name": away,
            "home_slug": _canon(home or ""),
            "away_slug": _canon(away or ""),
            "commence_time": ct,
        })
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[green]Wrote {len(df)} events to[/green] {out}")

    # Optional intersection with games
    if games_csv is not None and games_csv.exists():
        g = pd.read_csv(games_csv)
        g["date"] = pd.to_datetime(g.get("date"), errors="coerce")
        gd = g[g["date"].dt.date == d].copy()
        if not gd.empty:
            gteams = set([_canon(t) for t in gd.get("home_team", pd.Series(dtype=str)).astype(str)]) | set([_canon(t) for t in gd.get("away_team", pd.Series(dtype=str)).astype(str)])
            oteams = set(df["home_slug"]) | set(df["away_slug"])
            inter = gteams & oteams
            print({"games_teams": len(gteams), "odds_teams": len(oteams), "intersection": len(inter)})
            # Eligible D1 pairs vs provider pairs
            try:
                d1set: set[str] = set()
                if d1_csv and d1_csv.exists():
                    d1df = pd.read_csv(d1_csv)
                    # find a likely team name column
                    name_col = None
                    for c in d1df.columns:
                        lc = str(c).strip().lower()
                        if lc in {"team", "school", "name", "team_name"}:
                            name_col = c
                            break
                    if name_col is None and len(d1df.columns) > 0:
                        name_col = d1df.columns[0]
                    if name_col is not None:
                        d1set = set(d1df[name_col].astype(str).map(_canon))
                # Build game pairs for the day
                gd["home_slug"] = gd.get("home_team", pd.Series(dtype=str)).astype(str).map(_canon)
                gd["away_slug"] = gd.get("away_team", pd.Series(dtype=str)).astype(str).map(_canon)
                if d1set:
                    elig_mask = gd["home_slug"].isin(d1set) & gd["away_slug"].isin(d1set)
                    gd_elig = gd[elig_mask].copy()
                else:
                    gd_elig = gd.copy()
                def pair_key(a,b):
                    a1,a2 = sorted([str(a), str(b)])
                    return f"{a1}|{a2}"
                game_pairs = set(pair_key(r["home_slug"], r["away_slug"]) for _, r in gd_elig.iterrows())
                provider_pairs = set(pair_key(r["home_slug"], r["away_slug"]) for _, r in df.iterrows())
                unmatched = game_pairs - provider_pairs
                print({"eligible_pairs": len(game_pairs), "provider_pairs": len(provider_pairs), "unmatched_pairs": len(unmatched)})
                if unmatched_out is not None:
                    rows_unmatched = []
                    for _, r in gd_elig.iterrows():
                        pk = pair_key(r["home_slug"], r["away_slug"])
                        if pk in unmatched:
                            rows_unmatched.append({
                                "date": date,
                                "game_id": r.get("game_id"),
                                "home_team": r.get("home_team"),
                                "away_team": r.get("away_team"),
                                "home_slug": r.get("home_slug"),
                                "away_slug": r.get("away_slug"),
                            })
                    if rows_unmatched:
                        unmatched_out.parent.mkdir(parents=True, exist_ok=True)
                        # allow per-date filenames by default path or append date suffix if default path used
                        out_path = unmatched_out
                        if unmatched_out.name == "unmatched_eligible_pairs.csv":
                            out_path = unmatched_out.parent / f"unmatched_eligible_pairs_{date}.csv"
                        pd.DataFrame(rows_unmatched).to_csv(out_path, index=False)
                        print(f"[green]Wrote {len(rows_unmatched)} unmatched eligible pairs ->[/green] {out_path}")
            except Exception as e:
                print(f"[yellow]Eligible-unmatched calc failed:[/yellow] {e}")
        else:
            print(f"[yellow]No games on {date} in {games_csv}[/yellow]")


@app.command(name="odds-snapshot")
def odds_snapshot(
    date: str = typer.Option(_today_local().isoformat(), help="Target date YYYY-MM-DD for odds snapshot (optional)"),
    region: str = typer.Option("us", help="Region for odds (e.g., us, uk, eu)"),
    markets: str = typer.Option("h2h,spreads,totals,spreads_1st_half,totals_1st_half", help="Markets to request"),
    out_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory to store timestamped snapshots"),
):
    """Capture a timestamped odds snapshot for the given date/region/markets."""
    try:
        dt.date.fromisoformat(date)
    except Exception:
        print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
        raise typer.Exit(code=1)
    adapter = TheOddsAPIAdapter(region=region)
    rows = []
    for row in adapter.iter_current_odds_expanded(markets=markets, date_iso=date):
        rows.append(row.model_dump())
    if not rows:
        print(f"[yellow]No odds rows returned for date {date} (region={region}).[/yellow]")
    ts = dt.datetime.now().strftime("%H%M%S")
    ddir = out_dir / date
    ddir.mkdir(parents=True, exist_ok=True)
    out_path = ddir / f"snapshot_{region}_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[green]Wrote snapshot with {len(rows)} rows ->[/green] {out_path}")
    print("Hint: run repeatedly or schedule externally to capture intraday changes.")


@app.command(name="snapshot-loop")
def snapshot_loop(
    date: str = typer.Option(_today_local().isoformat(), help="Target date YYYY-MM-DD for odds snapshot"),
    regions: str = typer.Option("us", help="Comma-separated regions (e.g., us,uk)"),
    markets: str = typer.Option("h2h,spreads,totals,spreads_1st_half,totals_1st_half", help="Markets to request"),
    out_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory to store timestamped snapshots"),
    interval_seconds: int = typer.Option(900, help="Interval between snapshots in seconds (default 15 minutes)"),
    iterations: int = typer.Option(1, help="Number of snapshots to take (set high for long-running capture)"),
):
    """Take repeated snapshots at a fixed interval. Stops after 'iterations' captures."""
    try:
        dt.date.fromisoformat(date)
    except Exception:
        print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
        raise typer.Exit(code=1)
    region_list = [r.strip() for r in regions.split(",") if r.strip()]
    for i in range(int(max(1, iterations))):
        for region in region_list:
            try:
                adapter = TheOddsAPIAdapter(region=region)
                rows = [r.model_dump() for r in adapter.iter_current_odds_expanded(markets=markets, date_iso=date)]
                ts = dt.datetime.now().strftime("%H%M%S")
                ddir = out_dir / date
                ddir.mkdir(parents=True, exist_ok=True)
                out_path = ddir / f"snapshot_{region}_{ts}.csv"
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f"[green]{region}[/green] snapshot {i+1}/{iterations}: wrote {len(rows)} rows -> {out_path}")
            except Exception as e:
                print(f"[yellow]snapshot failed ({region}):[/yellow] {e}")
        if i < iterations - 1:
            time.sleep(max(5, int(interval_seconds)))


@app.command(name="odds-history-diagnostics")
def odds_history_diagnostics(
    date: str = typer.Option(..., help="Target date YYYY-MM-DD"),
    region: str = typer.Option("us", help="Region for TheOddsAPI"),
    games_csv: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Games master list"),
    snapshots_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory with timestamped snapshots per date"),
    out_json: Path = typer.Option(settings.outputs_dir / "diagnostics" / "odds_history_diag.json", help="Summary JSON output path"),
):
    """Summarize provider events, current odds, and snapshot coverage vs games for a date.

    Produces a JSON summary with counts of events, unique team pairs in odds, intersections vs games,
    and snapshot file coverage.
    """
    try:
        d = dt.date.fromisoformat(date)
    except Exception:
        print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
        raise typer.Exit(code=1)
    adapter = TheOddsAPIAdapter(region=region)
    # Events list
    try:
        events = adapter.list_events_by_date(date)
    except Exception:
        events = []
    # Current odds pairs
    def pair_key(a,b):
        a1,a2 = sorted([str(a), str(b)])
        return f"{a1}|{a2}"
    from .data.team_normalize import canonical_slug as _canon
    event_pairs = set()
    for ev in events or []:
        event_pairs.add(pair_key(_canon(ev.get("home_team") or ""), _canon(ev.get("away_team") or "")))
    # Current expanded odds
    cur_rows = [r.model_dump() for r in adapter.iter_current_odds_expanded(date_iso=date)]
    cur_pairs = set(pair_key(_canon(r.get("home_team_name") or ""), _canon(r.get("away_team_name") or "")) for r in cur_rows)
    # Snapshot pairs
    snap_pairs = set()
    snap_files = []
    ddir = snapshots_dir / date
    if ddir.exists():
        for f in sorted(ddir.glob("snapshot_*_*.csv")):
            try:
                df = pd.read_csv(f)
                snap_files.append(f.name)
                for _, r in df.iterrows():
                    snap_pairs.add(pair_key(_canon(str(r.get("home_team_name"))), _canon(str(r.get("away_team_name")))))
            except Exception:
                continue
    # Games pairs
    g = pd.read_csv(games_csv)
    g["date"] = pd.to_datetime(g.get("date"), errors="coerce")
    gd = g[g["date"].dt.date == d].copy()
    game_pairs = set()
    if not gd.empty:
        gd["home_slug"] = gd.get("home_team", pd.Series(dtype=str)).astype(str).map(_canon)
        gd["away_slug"] = gd.get("away_team", pd.Series(dtype=str)).astype(str).map(_canon)
        for _, r in gd.iterrows():
            game_pairs.add(pair_key(r["home_slug"], r["away_slug"]))
    summary = {
        "date": date,
        "events_count": len(events or []),
        "event_pairs": len(event_pairs),
        "current_rows": len(cur_rows),
        "current_pairs": len(cur_pairs),
        "snapshot_pairs": len(snap_pairs),
        "snapshot_files": snap_files,
        "games_pairs": len(game_pairs),
        "pairs_intersections": {
            "events_vs_games": len(event_pairs & game_pairs),
            "current_vs_games": len(cur_pairs & game_pairs),
            "snapshots_vs_games": len(snap_pairs & game_pairs),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[green]Wrote diagnostics summary ->[/green] {out_json}")


@app.command(name="diagnose-odds-join")
def diagnose_odds_join(
    date: str = typer.Option(..., help="Target date YYYY-MM-DD"),
    games_csv: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Games master CSV"),
    odds_csv: Path = typer.Option(settings.outputs_dir / "last_odds.csv", help="Last odds snapshot CSV (totals/spreads)"),
    out_dir: Path = typer.Option(settings.outputs_dir / "diagnostics", help="Directory for diagnostics artifacts"),
    min_ratio: float = typer.Option(0.82, help="Minimum similarity ratio (0-1) for mapping suggestions"),
):
    """Diagnose why odds rows failed to join games for a given date and emit mapping suggestions.

    Steps:
    - Load games for date and build canonical pair keys
    - Load odds (totals full-game) and build canonical pair keys
    - Compute intersection stats, list unmatched pairs
    - Produce per-team name mapping suggestions using fuzzy similarity (difflib) against games' team set
    - Write CSV artifacts: pair_stats, unmatched_pairs, name_mapping suggestions

    Use resulting mapping file to extend normalization if systemic mismatches are found.
    """
    import difflib
    from .data.team_normalize import canonical_slug as _canon
    # Validate date
    try:
        target_date = dt.date.fromisoformat(date)
    except Exception:
        print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
        raise typer.Exit(code=1)
    if not games_csv.exists():
        print(f"[red]Games CSV not found:[/red] {games_csv}")
        raise typer.Exit(code=1)
    if not odds_csv.exists():
        print(f"[red]Odds CSV not found:[/red] {odds_csv}")
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(games_csv)
    games["date"] = pd.to_datetime(games.get("date"), errors="coerce")
    gsel = games[games["date"].dt.date == target_date].copy()
    if gsel.empty:
        print(f"[yellow]No games for {date} in {games_csv}[/yellow]")
    # Build canonical team sets
    gsel["home_team"] = gsel.get("home_team", pd.Series(dtype=str)).astype(str)
    gsel["away_team"] = gsel.get("away_team", pd.Series(dtype=str)).astype(str)
    gsel["pair_key"] = gsel.apply(lambda r: "::".join(sorted([_canon(r.home_team), _canon(r.away_team)])), axis=1)
    game_pairs = set(gsel["pair_key"].unique())
    game_teams = set(_canon(t) for t in list(gsel["home_team"].unique()) + list(gsel["away_team"].unique()))

    odds = pd.read_csv(odds_csv)
    # Normalize market/period columns flexibly
    odds["market"] = odds.get("market", pd.Series(dtype=str)).astype(str).str.lower()
    odds["period"] = odds.get("period", pd.Series(dtype=str)).astype(str).str.lower()
    mask_totals = odds["market"] == "totals"
    mask_full = odds["period"].isin(["full_game", "fg", "full game", "fullgame"])
    o_tot = odds[mask_totals & mask_full].copy()
    o_tot["home_team_name"] = o_tot.get("home_team_name", pd.Series(dtype=str)).astype(str)
    o_tot["away_team_name"] = o_tot.get("away_team_name", pd.Series(dtype=str)).astype(str)
    o_tot["pair_key"] = o_tot.apply(lambda r: "::".join(sorted([_canon(r.home_team_name), _canon(r.away_team_name)])) if r.home_team_name and r.away_team_name else "", axis=1)
    odds_pairs = set([p for p in o_tot["pair_key"].unique() if p])

    intersection = game_pairs & odds_pairs
    # Unmatched decomposition
    unmatched_games = sorted(game_pairs - intersection)
    unmatched_odds = sorted(odds_pairs - intersection)

    # Build per raw team name suggestion table for odds side teams not present in games canonical set
    raw_odds_teams = set(_canon(t) for t in list(o_tot["home_team_name"].unique()) + list(o_tot["away_team_name"].unique()))
    missing_odds_teams = sorted(raw_odds_teams - game_teams)
    suggestions = []
    game_team_raw = set(list(gsel["home_team"].unique()) + list(gsel["away_team"].unique()))
    game_team_raw_list = list(game_team_raw)
    for raw_slug in missing_odds_teams:
        # Try to find closest raw team (pre-slug canonical) by comparing slug to canonical slug of game_team_raw_list
        # We'll compare on original raw strings too for context
        best_match = None
        best_ratio = 0.0
        for candidate in game_team_raw_list:
            c_slug = _canon(candidate)
            ratio = difflib.SequenceMatcher(a=raw_slug, b=c_slug).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        if best_match and best_ratio >= min_ratio:
            suggestions.append({
                "odds_slug": raw_slug,
                "suggested_game_team": best_match,
                "suggested_game_slug": _canon(best_match),
                "similarity": round(best_ratio, 4),
            })
        else:
            suggestions.append({
                "odds_slug": raw_slug,
                "suggested_game_team": None,
                "suggested_game_slug": None,
                "similarity": round(best_ratio, 4),
            })

    # Write artifacts
    pd.DataFrame({
        "date": [date],
        "games_pairs": [len(game_pairs)],
        "odds_pairs": [len(odds_pairs)],
        "intersection": [len(intersection)],
        "unmatched_game_pairs": [len(unmatched_games)],
        "unmatched_odds_pairs": [len(unmatched_odds)],
    }).to_csv(out_dir / f"pair_stats_{date}.csv", index=False)
    pd.DataFrame({"unmatched_game_pair": unmatched_games}).to_csv(out_dir / f"unmatched_game_pairs_{date}.csv", index=False)
    pd.DataFrame({"unmatched_odds_pair": unmatched_odds}).to_csv(out_dir / f"unmatched_odds_pairs_{date}.csv", index=False)
    pd.DataFrame(suggestions).to_csv(out_dir / f"odds_name_mapping_{date}.csv", index=False)

    print("[bold]Pair Key Diagnostics[/bold]", {
        "games_pairs": len(game_pairs),
        "odds_pairs": len(odds_pairs),
        "intersection": len(intersection),
    })
    if not intersection:
        print("[yellow]No intersection found. Use odds_name_mapping CSV to augment ALIAS_MAP or adjust normalization.[/yellow]")
    else:
        print(f"[green]Found {len(intersection)} intersecting pair keys.[/green]")
    print(f"[cyan]Artifacts written to[/cyan] {out_dir}")


@app.command(name="backfill-odds-history")
def backfill_odds_history(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    games_path: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Existing games_all.csv with game_id/date/home_team/away_team"),
    region: str = typer.Option("us", help="Odds region for TheOddsAPI"),
    markets: str = typer.Option(
        "h2h,spreads,totals,spreads_1st_half,totals_1st_half,spreads_2nd_half,totals_2nd_half",
        help="Comma-separated markets (include half variants if plan supports)."
    ),
    out_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory for per-day odds_YYYY-MM-DD.csv snapshots"),
    write_last: bool = typer.Option(True, help="Also compute last pre-tip odds and merge to games_with_last.csv"),
    write_closing: bool = typer.Option(True, help="Also compute closing lines and merge to games_with_closing.csv"),
    closing_window_minutes: int = typer.Option(10, help="Pre-tip window (minutes) for highest priority closing line selection"),
    tolerance_seconds: int = typer.Option(60, help="Tolerance window after tip for last odds inclusion (seconds)"),
):
    """Backfill historical odds snapshots and merge closing / last lines to games.

    Workflow per date:
      1. List events via TheOddsAPI events endpoint
      2. Fetch odds-history for those event IDs (markets incl. halves)
      3. Write odds_YYYY-MM-DD.csv into odds_history/
    After full range:
      - If enabled, compute closing lines across all snapshots -> closing_lines.csv then join to games -> games_with_closing.csv (accumulated)
      - If enabled, compute last pre-tip odds -> last_odds.csv then join to games -> games_with_last.csv (accumulated)

    Accumulation logic: existing merged files are read, new rows concatenated, then de-duplicated on (game_id, book, market, period) keeping the latest.
    """
    try:
        d0 = dt.date.fromisoformat(start)
        d1 = dt.date.fromisoformat(end)
    except Exception:
        print("[red]Invalid start/end date format; use YYYY-MM-DD.[/red]")
        raise typer.Exit(code=1)
    if d1 < d0:
        print("[red]End date must be >= start date[/red]")
        raise typer.Exit(code=1)
    if not games_path.exists():
        print(f"[red]Games file not found:[/red] {games_path}")
        raise typer.Exit(code=1)
    try:
        games = pd.read_csv(games_path)
    except Exception as e:
        print(f"[red]Failed reading games file:[/red] {e}")
        raise typer.Exit(code=1)
    if games.empty or "game_id" not in games.columns or "date" not in games.columns:
        print("[red]games_all.csv missing required columns game_id/date[/red]")
        raise typer.Exit(code=1)
    games["game_id"] = games["game_id"].astype(str)
    # Normalize date format
    try:
        games["date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
    except Exception:
        games["date"] = games["date"].astype(str)

    adapter = TheOddsAPIAdapter(region=region)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    cur = d0
    while cur <= d1:
        date_iso = cur.isoformat()
        rows: list[dict] = []
        try:
            events = adapter.list_events_by_date(date_iso)
            event_ids = [str(e.get("id")) for e in events if e.get("id")]
            if event_ids:
                for r in adapter.iter_odds_history_for_events(event_ids, markets=markets):
                    rows.append(r.model_dump())
        except Exception as e:
            print(f"[yellow]Failed odds-history for {date_iso}:[/yellow] {e}")
        if rows:
            df = pd.DataFrame(rows)
            out_path = out_dir / f"odds_{date_iso}.csv"
            df.to_csv(out_path, index=False)
            total_rows += len(df)
            print(f"[green]{date_iso} -> wrote {len(df)} rows[/green]")
        else:
            print(f"[yellow]{date_iso} -> no odds-history rows[/yellow]")
        cur += dt.timedelta(days=1)
    if total_rows == 0:
        print("[red]No odds-history rows collected; aborting merge steps.[/red]")
        raise typer.Exit(code=1)

    # Load all snapshots for merge operations
    from .data.odds_closing import read_directory_for_dates, load_snapshots
    snap_paths = read_directory_for_dates(out_dir)
    snaps = load_snapshots(snap_paths)
    if snaps.empty:
        print("[red]Snapshots set empty after read; nothing further to do.[/red]")
        raise typer.Exit(code=1)

    if write_closing:
        try:
            closing = compute_closing_lines(snaps, window_minutes=closing_window_minutes)
            if not closing.empty:
                closing_out = settings.outputs_dir / "closing_lines.csv"
                closing.to_csv(closing_out, index=False)
                print(f"[green]Computed closing lines ->[/green] {closing_out} ({len(closing)} rows)")
                merged_closing = join_games_with_closing(games, closing)
                merged_closing_path = settings.outputs_dir / "games_with_closing.csv"
                if merged_closing_path.exists():
                    prev = pd.read_csv(merged_closing_path)
                    if "game_id" in prev.columns:
                        prev["game_id"] = prev["game_id"].astype(str)
                    merged_all = pd.concat([prev, merged_closing], ignore_index=True)
                    # Deduplicate
                    dedupe_subset = [c for c in ["game_id","book","market","period"] if c in merged_all.columns]
                    if dedupe_subset:
                        merged_all = merged_all.sort_values(dedupe_subset).drop_duplicates(subset=dedupe_subset, keep="last")
                else:
                    merged_all = merged_closing
                merged_all.to_csv(merged_closing_path, index=False)
                print(f"[green]Updated games_with_closing.csv[/green] ({len(merged_all)} rows total)")
            else:
                print("[yellow]Closing computation produced no rows.[/yellow]")
        except Exception as e:
            print(f"[yellow]Closing merge skipped:{e}[/yellow]")

    if write_last:
        try:
            last_df = compute_last_odds(snaps, tolerance_seconds=tolerance_seconds)
            if not last_df.empty:
                last_out = settings.outputs_dir / "last_odds.csv"
                last_df.to_csv(last_out, index=False)
                print(f"[green]Computed last odds ->[/green] {last_out} ({len(last_df)} rows)")
                merged_last_df = join_games_with_closing(games, last_df)
                merged_last_path = settings.outputs_dir / "games_with_last.csv"
                if merged_last_path.exists():
                    prev = pd.read_csv(merged_last_path)
                    if "game_id" in prev.columns:
                        prev["game_id"] = prev["game_id"].astype(str)
                    merged_all = pd.concat([prev, merged_last_df], ignore_index=True)
                    dedupe_subset = [c for c in ["game_id","book","market","period"] if c in merged_all.columns]
                    if dedupe_subset:
                        merged_all = merged_all.sort_values(dedupe_subset).drop_duplicates(subset=dedupe_subset, keep="last")
                else:
                    merged_all = merged_last_df
                merged_all.to_csv(merged_last_path, index=False)
                print(f"[green]Updated games_with_last.csv[/green] ({len(merged_all)} rows total)")
            else:
                print("[yellow]Last odds computation produced no rows.[/yellow]")
        except Exception as e:
            print(f"[yellow]Last odds merge skipped:{e}[/yellow]")
    print(f"[green]Backfill complete.[/green] Collected {total_rows} odds snapshots across range {start} -> {end}")
    print(f"[green]Done.[/green] Total rows: {total_rows}")


@app.command(name="make-closing-lines")
def make_closing_lines_cmd(
    in_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory with odds_YYYY-MM-DD.csv files"),
    out: Path = typer.Option(settings.outputs_dir / "closing_lines.csv", help="Output CSV of closing lines"),
):
    """Aggregate odds snapshots into closing lines per event/book/market/period.

    Selection heuristic: prefer last_update <= commence_time; else fetched_at <= commence_time; else latest observed.
    """
    try:
        path = make_closing_lines(in_dir, out)
        print(f"[green]Wrote closing lines to[/green] {path}")
    except Exception as e:
        print(f"[red]Failed to build closing lines:[/red] {e}")


@app.command(name="compute-line-movement")
def compute_line_movement_cmd(
    in_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory with odds snapshots (odds_*.csv or snapshot folders)"),
    out: Path = typer.Option(settings.outputs_dir / "line_movement.csv", help="Output CSV with open/close deltas and steam flags"),
    window_minutes: int | None = typer.Option(90, help="Closing selection window (minutes before tip); None to treat all pre-tip equally"),
):
    """Compute line movement metrics (open->close deltas, steam flags) per (event_id, book, market, period).

    Uses earliest observed snapshot row as the opening line and closing selection logic for the closing line.
    Totals: steam_total_flag when abs(delta_total) >= 1.5
    Spreads: steam_spread_flag when abs(delta_home_spread) >= 2.0
    Moneyline: steam_ml_home_flag when abs(delta_moneyline_home) >= 40 (American odds)
    """
    try:
        paths = read_directory_for_dates(in_dir)
        if not paths:
            print(f"[red]No odds snapshots found under {in_dir}[/red]")
            raise typer.Exit(code=1)
        df = load_snapshots(paths)
        if df.empty:
            print("[yellow]Loaded 0 rows from snapshots; aborting.[/yellow]")
            raise typer.Exit(code=1)
        moved = compute_closing_lines(df, window_minutes=window_minutes)
        out.parent.mkdir(parents=True, exist_ok=True)
        moved.to_csv(out, index=False)
        print(f"[green]Wrote line movement metrics to[/green] {out} ({len(moved)} rows)")
    except Exception as e:
        print(f"[red]Line movement computation failed:[/red] {e}")


@app.command(name="make-last-odds")
def make_last_odds_cmd(
    in_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory with odds_YYYY-MM-DD.csv files"),
    out: Path = typer.Option(settings.outputs_dir / "last_odds.csv", help="Output CSV of last pre-tip odds (per book)"),
    tolerance_seconds: int = typer.Option(60, help="Allow small clock skew: include snapshots up to this many seconds after commence_time"),
):
    """Select the last observed pre-tip odds per event/book/market/period (no fallback/post-tip)."""
    try:
        path = make_last_odds(in_dir, out, tolerance_seconds=tolerance_seconds)
        print(f"[green]Wrote last odds to[/green] {path}")
    except Exception as e:
        print(f"[red]Failed to build last odds:[/red] {e}")


@app.command(name="fetch-boxscores")
def fetch_boxscores(
    games_path: Path = typer.Argument(..., help="Games CSV/Parquet produced by fetch-games (must be ESPN source for event IDs)"),
    out: Path = typer.Option(settings.outputs_dir / "boxscores.csv", help="Output CSV with possessions and four factors"),
    use_cache: bool = True,
):
    """Fetch ESPN box score summaries for games and compute possessions and four factors.

    Expects the games file to contain ESPN event IDs in the 'game_id' column.
    """
    # Load games flexibly
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))

    if "game_id" not in games.columns:
        print("[red]games file missing 'game_id' column[/red]")
        raise typer.Exit(code=1)

    rows = []
    for r in iter_boxscores(games["game_id"].astype(str).tolist(), use_cache=use_cache):
        if r is None:
            continue
        rows.append(r.model_dump())
    if not rows:
        print("[yellow]No box scores fetched. Check that game_id values are ESPN event IDs.[/yellow]")
        raise typer.Exit(code=1)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[green]Wrote boxscores to[/green] {out} ({len(df)} rows)")


@app.command(name="predict-segmented-inline")
def predict_segmented_inline_cmd(
    features_csv: Path = typer.Argument(..., help="Features CSV to score"),
    segment: str = typer.Option("conference", help="Segmentation: team|conference"),
    models_dir: Path = typer.Option(settings.outputs_dir / "models", help="Directory containing segmented models (seg_team|seg_conference) and global models"),
    conf_map: Path | None = typer.Option(None, help="Optional team->conference CSV (columns: team,conference[,season])"),
    out: Path = typer.Option(settings.outputs_dir / "predictions_segmented.csv", help="Output CSV with predictions"),
    apply_guardrails: bool = typer.Option(False, help="Apply tempo/off/def derived blending guardrail for implausibly low totals (<105 or <75% derived) and mark pred_total_adjusted"),
    half_ratio: float = typer.Option(0.485, help="Share of full-game total expected in 1H for derived half projections (rest goes to 2H)"),
):
    """Score a features CSV using segmented models, falling back to the global baseline when needed."""
    feats = pd.read_csv(features_csv)
    out_df = _predict_segmented_inline(
        feats,
        seg_mode=segment,
        models_root=models_dir,
        conf_map_path=conf_map if conf_map is not None else settings.data_dir / "conferences.csv",
    )
    # Optional guardrails on segmented outputs
    if apply_guardrails:
        try:
            feats = pd.read_csv(features_csv)
            feats["game_id"] = feats.get("game_id").astype(str)
            out_df["game_id"] = out_df.get("game_id").astype(str)
            merged = out_df.merge(feats[["game_id","home_off_rating","away_off_rating","home_def_rating","away_def_rating","home_tempo_rating","away_tempo_rating","tempo_rating_sum"]], on="game_id", how="left")
            derived_vals = []
            flags = []
            for _, r in merged.iterrows():
                try:
                    tempo_avg = None
                    if pd.notna(r.get("home_tempo_rating")) and pd.notna(r.get("away_tempo_rating")):
                        tempo_avg = (float(r.get("home_tempo_rating")) + float(r.get("away_tempo_rating"))) / 2.0
                    elif pd.notna(r.get("tempo_rating_sum")):
                        tempo_avg = float(r.get("tempo_rating_sum")) / 2.0
                    else:
                        tempo_avg = 70.0
                    off_home = float(r.get("home_off_rating")) if pd.notna(r.get("home_off_rating")) else 100.0
                    off_away = float(r.get("away_off_rating")) if pd.notna(r.get("away_off_rating")) else 100.0
                    def_home = float(r.get("home_def_rating")) if pd.notna(r.get("home_def_rating")) else 100.0
                    def_away = float(r.get("away_def_rating")) if pd.notna(r.get("away_def_rating")) else 100.0
                    exp_home_pp100 = np.clip(off_home - def_away, 80, 130)
                    exp_away_pp100 = np.clip(off_away - def_home, 80, 130)
                    dv = (exp_home_pp100 + exp_away_pp100) / 100.0 * tempo_avg
                    dv = float(np.clip(dv, 110, 185))
                except Exception:
                    dv = np.nan
                derived_vals.append(dv)
                pt = r.get("pred_total")
                if pd.notna(pt) and pd.notna(dv) and (pt < 105 or pt < 0.75 * dv):
                    flags.append(True)
                else:
                    flags.append(False)
            out_df["pred_total_raw"] = out_df["pred_total"].astype(float)
            blended = []
            for pt, dv, f in zip(out_df["pred_total_raw"].tolist(), derived_vals, flags):
                if not f or np.isnan(pt) or np.isnan(dv):
                    blended.append(pt)
                else:
                    b = float(np.clip(0.5 * pt + 0.5 * dv, 110, 185))
                    blended.append(b)
            out_df["pred_total"] = blended
            out_df["pred_total_adjusted"] = flags
        except Exception:
            pass

    # Derived half projections
    try:
        out_df["pred_total_1h"] = pd.to_numeric(out_df["pred_total"], errors="coerce") * half_ratio
        out_df["pred_total_2h"] = pd.to_numeric(out_df["pred_total"], errors="coerce") * (1.0 - half_ratio)
        out_df["pred_margin_1h"] = pd.to_numeric(out_df["pred_margin"], errors="coerce") * 0.5
        out_df["pred_margin_2h"] = pd.to_numeric(out_df["pred_margin"], errors="coerce") * 0.5
    except Exception:
        pass

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"[green]Wrote segmented predictions to[/green] {out}")


@app.command(name="bankroll-optimize")
def bankroll_optimize(
    merged_csv: Path = typer.Option(settings.outputs_dir / "games_with_last.csv", help="Joined games + last odds CSV (or any CSV with pred_* and market columns)"),
    bankroll: float = typer.Option(1000.0, help="Total bankroll amount (currency units)"),
    kelly_fraction: float = typer.Option(0.5, help="Fractional Kelly to apply (0-1)"),
    include_markets: str = typer.Option("totals,spreads,h2h", help="Comma-separated markets to include: totals,spreads,h2h"),
    min_edge_total: float = typer.Option(0.5, help="Min absolute edge (points) to include totals pick"),
    min_edge_margin: float = typer.Option(0.5, help="Min absolute edge (points) to include spread pick"),
    min_kelly: float = typer.Option(0.01, help="Minimum Kelly fraction to include a pick (after sign, before fractional multiplier)"),
    max_pct_per_bet: float = typer.Option(0.03, help="Maximum stake per bet as a fraction of bankroll (0-1)"),
    max_daily_risk_pct: float = typer.Option(0.10, help="Cap total risk for a single date as a fraction of bankroll; stakes are scaled down if exceeded"),
    use_distributional: bool = typer.Option(False, help="If true and pred_total_mu/sigma available, compute totals Kelly from Normal CDF and scale stakes by uncertainty"),
    min_ev: float = typer.Option(0.0, help="Minimum expected value per unit stake (EV) to include a pick when EV is available"),
    sigma_ref: float = typer.Option(10.0, help="Reference sigma for uncertainty scaling: scale = sigma_ref / (sigma + sigma_ref)"),
    z_ref: float = typer.Option(1.0, help="Reference z-score for optional z-based scaling (unused for now)"),
    consolidate_totals: bool = typer.Option(True, help="If true, keep only best book per game & side for totals"),
    out: Path = typer.Option(settings.outputs_dir / "stake_sheet.csv", help="Output CSV with recommended stakes"),
    calibrate_probabilities: bool = typer.Option(False, help="If true, apply z-score recenter calibration to p_over/p_under when using distributional"),
    calibration_artifact: Path | None = typer.Option(None, help="Path to calibration artifact JSON (defaults to models_dist/calibration_totals.json if omitted)"),
):
    """Generate recommended stakes using fractional Kelly sizing with guardrails.

    Inputs require model outputs (pred_total, pred_margin) and market columns for totals/spreads/h2h.
    If edge columns are missing, they will be computed.
    """
    if not merged_csv.exists():
        print(f"[red]Input file not found:[/red] {merged_csv}")
        raise typer.Exit(code=1)
    try:
        df = pd.read_csv(merged_csv)
    except Exception as e:
        print(f"[red]Failed to read input CSV:[/red] {e}")
        raise typer.Exit(code=1)
    if df.empty:
        print("[yellow]No rows in input; nothing to do.[/yellow]")
        raise typer.Exit(code=0)
    # Ensure edges present
    need_cols = {"edge_total", "edge_margin", "kelly_fraction_total", "kelly_fraction_ml_home", "kelly_fraction_ml_away"}
    if need_cols - set(df.columns):
        try:
            df = compute_edges(df)
        except Exception as e:
            print(f"[red]Failed to compute edges:[/red] {e}")
            raise typer.Exit(code=1)

    markets_set = {m.strip().lower() for m in include_markets.split(',') if m.strip()}
    bankroll = float(max(0.0, bankroll))
    kelly_fraction = float(np.clip(kelly_fraction, 0.0, 1.0))
    max_pct_per_bet = float(np.clip(max_pct_per_bet, 0.0, 1.0))
    max_daily_risk_pct = float(np.clip(max_daily_risk_pct, 0.0, 1.0))

    picks: list[dict] = []

    # Helper to cap stake
    def cap_stake(raw: float) -> float:
        cap = bankroll * max_pct_per_bet
        return float(min(max(0.0, raw), cap))

    # Build totals picks (allow absence of explicit 'market' by falling back to any row with a total value)
    if "totals" in markets_set and ("total" in df.columns):
        if "market" in df.columns:
            tot_rows = df[df["market"].astype(str).str.lower() == "totals"].copy()
        else:
            tot_rows = df[df["total"].notna()].copy()
        # Helpers
        def american_to_b(odds: float) -> float:
            if pd.isna(odds):
                return np.nan
            # returns b multiplier (decimal minus 1)
            if odds < 0:
                return 100.0 / (-odds)
            else:
                return odds / 100.0
        def normal_cdf(z: float) -> float:
            # Use math.erf for portability (numpy.math may not exist in some environments)
            import math
            return 0.5 * (1.0 + float(math.erf(z / math.sqrt(2.0))))
        # Load calibration artifact once if requested
        cal = None
        if use_distributional and calibrate_probabilities:
            art_path = calibration_artifact or (settings.outputs_dir / "models_dist" / "calibration_totals.json")
            if art_path.exists():
                cal = load_artifact(art_path)
                if cal is None or getattr(cal, "method", None) != "z_recenter":
                    print("[yellow]Calibration artifact not compatible; skipping probability calibration.[/yellow]")
                    cal = None
            else:
                print(f"[yellow]Calibration artifact not found at {art_path}; skipping probability calibration.[/yellow]")
        for _, r in tot_rows.iterrows():
            line = r.get("total")
            if not np.isfinite(float(line)):
                continue
            # Distributional path
            if use_distributional and ("pred_total_mu" in df.columns and "pred_total_sigma" in df.columns):
                mu = float(r.get("pred_total_mu", np.nan))
                sig = float(r.get("pred_total_sigma", np.nan))
                if not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
                    continue
                z = (float(line) - mu) / sig
                # Apply z-score recenter calibration if available
                if cal is not None:
                    try:
                        z = (z - float(cal.z_center)) / max(float(cal.z_scale), 1e-6)
                    except Exception:
                        pass
                p_over = 1.0 - normal_cdf(z)
                p_under = normal_cdf(z)
                # Clamp for stability
                p_over = float(np.clip(p_over, 1e-4, 1.0 - 1e-4))
                p_under = float(np.clip(p_under, 1e-4, 1.0 - 1e-4))
                # Prices may be missing in prefetch snapshot; default to -110/-110 so Kelly can still evaluate
                raw_over = float(r.get("over_price", np.nan)) if "over_price" in r.index else np.nan
                raw_under = float(r.get("under_price", np.nan)) if "under_price" in r.index else np.nan
                b_over = american_to_b(raw_over)
                b_under = american_to_b(raw_under)
                if not np.isfinite(b_over):
                    b_over = american_to_b(-110.0)
                if not np.isfinite(b_under):
                    b_under = american_to_b(-110.0)
                # Expected value per unit stake (decimal-minus-1 b used): EV = p*b - (1-p)
                ev_over = np.nan if not (np.isfinite(b_over) and np.isfinite(p_over)) else (p_over * b_over - (1.0 - p_over))
                ev_under = np.nan if not (np.isfinite(b_under) and np.isfinite(p_under)) else (p_under * b_under - (1.0 - p_under))
                # Kelly for over/under
                def kelly_from_p(b: float, p: float) -> float:
                    if not np.isfinite(b) or not np.isfinite(p):
                        return np.nan
                    q = 1.0 - p
                    kf = (b * p - q) / b
                    return float(kf)
                k_over = kelly_from_p(b_over, p_over)
                k_under = kelly_from_p(b_under, p_under)
                side = None
                k = None
                price = None
                # Choose by Kelly; tie-break by EV when close
                if np.isfinite(k_over) and k_over > 0 and (not np.isfinite(k_under) or k_over >= k_under):
                    side = "over"; k = k_over; price = r.get("over_price")
                elif np.isfinite(k_under) and k_under > 0:
                    side = "under"; k = k_under; price = r.get("under_price")
                if side is None or k is None or k < float(min_kelly):
                    continue
                # Apply EV gate if available
                ev_chosen = ev_over if side == "over" else ev_under
                p_chosen = p_over if side == "over" else p_under
                if np.isfinite(min_ev) and min_ev > 0.0 and (not np.isfinite(ev_chosen) or ev_chosen < float(min_ev)):
                    continue
                # Uncertainty scaling
                scale = float(sigma_ref / (sig + sigma_ref)) if sigma_ref > 0 else 1.0
                stake = cap_stake(bankroll * kelly_fraction * k * scale)
                picks.append({
                    "date": r.get("date") or r.get("date_game") or np.nan,
                    "game_id": r.get("game_id"),
                    "event_id": r.get("event_id"),
                    "book": r.get("book"),
                    "market": "totals",
                    "period": r.get("period"),
                    "selection": side,
                    "line": line,
                    "price": price,
                    "edge": (mu - float(line)),
                    "kelly": k,
                    "fractional": kelly_fraction,
                    "uncertainty_scale": scale,
                    "p_over": p_over,
                    "p_under": p_under,
                    "b_over": b_over,
                    "b_under": b_under,
                    "ev_over": ev_over,
                    "ev_under": ev_under,
                    "prob": p_chosen,
                    "ev": ev_chosen,
                    "stake": stake,
                })
            else:
                # Edge-based heuristic path (legacy)
                if "pred_total" not in df.columns or "edge_total" not in df.columns:
                    continue
                edge = float(r.get("edge_total", np.nan))
                if not np.isfinite(edge) or abs(edge) < float(min_edge_total):
                    continue
                k = float(r.get("kelly_fraction_total", np.nan))
                if not np.isfinite(k) or k <= 0 or k < float(min_kelly):
                    continue
                side = "over" if edge > 0 else "under"
                price = r.get("over_price") if side == "over" else r.get("under_price")
                stake = cap_stake(bankroll * kelly_fraction * k)
                picks.append({
                    "date": r.get("date") or r.get("date_game") or np.nan,
                    "game_id": r.get("game_id"),
                    "event_id": r.get("event_id"),
                    "book": r.get("book"),
                    "market": "totals",
                    "period": r.get("period"),
                    "selection": side,
                    "line": line,
                    "price": price,
                    "edge": edge,
                    "kelly": k,
                    "fractional": kelly_fraction,
                    "stake": stake,
                })

    # Build spread picks
    if "spreads" in markets_set and {"market", "home_spread", "pred_margin"}.issubset(df.columns):
        sp_rows = df[(df["market"].astype(str).str.lower() == "spreads") & df["edge_margin"].notna()]
        for _, r in sp_rows.iterrows():
            edge = float(r.get("edge_margin", np.nan))
            if not np.isfinite(edge) or abs(edge) < float(min_edge_margin):
                continue
            # Approximate kelly-like fraction for spread via edge magnitude vs handicap size
            denom = max(1.0, abs(float(r.get("home_spread", 0.0))) * 2.0)
            k_proxy = float(np.clip(edge / denom, -1.0, 1.0))
            if k_proxy <= 0 or k_proxy < float(min_kelly):
                continue
            if edge > 0:
                side = "home"
                line = r.get("home_spread")
                price = r.get("home_spread_price")
            else:
                side = "away"
                line = r.get("away_spread")
                price = r.get("away_spread_price")
            stake = cap_stake(bankroll * kelly_fraction * k_proxy)
            picks.append({
                "date": r.get("date") or r.get("date_game") or np.nan,
                "game_id": r.get("game_id"),
                "event_id": r.get("event_id"),
                "book": r.get("book"),
                "market": "spreads",
                "period": r.get("period"),
                "selection": side,
                "line": line,
                "price": price,
                "edge": edge,
                "kelly": k_proxy,
                "fractional": kelly_fraction,
                "stake": stake,
            })

    # Build moneyline picks (choose best side per row)
    if "h2h" in markets_set and {"market", "moneyline_home", "moneyline_away", "pred_margin"}.issubset(df.columns):
        ml_rows = df[(df["market"].astype(str).str.lower() == "h2h")]
        for _, r in ml_rows.iterrows():
            k_home = float(r.get("kelly_fraction_ml_home", np.nan))
            k_away = float(r.get("kelly_fraction_ml_away", np.nan))
            ev_home = float(r.get("home_ml_ev", np.nan))
            ev_away = float(r.get("away_ml_ev", np.nan))
            p_home = float(r.get("home_ml_prob_fair", np.nan))
            p_away = float(r.get("away_ml_prob_fair", np.nan))
            side = None
            k = None
            line = None
            price = None
            if np.isfinite(k_home) and k_home > 0 and k_home >= float(min_kelly) and (not np.isfinite(k_away) or k_home >= k_away):
                side = "home"
                k = k_home
                line = np.nan
                price = r.get("moneyline_home")
            elif np.isfinite(k_away) and k_away > 0 and k_away >= float(min_kelly):
                side = "away"
                k = k_away
                line = np.nan
                price = r.get("moneyline_away")
            if side is None or k is None:
                continue
            # Apply EV gate if present
            ev_chosen = ev_home if side == "home" else ev_away
            p_chosen = p_home if side == "home" else p_away
            if np.isfinite(min_ev) and min_ev > 0.0 and (not np.isfinite(ev_chosen) or ev_chosen < float(min_ev)):
                continue
            stake = cap_stake(bankroll * kelly_fraction * float(k))
            picks.append({
                "date": r.get("date") or r.get("date_game") or np.nan,
                "game_id": r.get("game_id"),
                "event_id": r.get("event_id"),
                "book": r.get("book"),
                "market": "h2h",
                "period": r.get("period"),
                "selection": side,
                "line": line,
                "price": price,
                "edge": np.nan,
                "kelly": k,
                "fractional": kelly_fraction,
                "prob": p_chosen,
                "ev": ev_chosen,
                "stake": stake,
            })

    if not picks:
        print("[yellow]No picks met thresholds; stake sheet is empty.[/yellow]")
        raise typer.Exit(code=0)

    stake_df = pd.DataFrame(picks)
    # Consolidate totals duplicates (best Kelly then best EV then best price) if requested
    if consolidate_totals and not stake_df.empty and {"market","kelly","game_id","selection"}.issubset(stake_df.columns):
        # Sort by preference and keep first per (game_id,selection)
        mask_tot = stake_df["market"].astype(str).str.lower() == "totals"
        totals_df = stake_df[mask_tot].copy()
        others_df = stake_df[~mask_tot].copy()
        if not totals_df.empty:
            # Some legacy/edge-only paths may omit 'ev'; guard by filling with NaN if absent.
            if "ev" not in totals_df.columns:
                totals_df["ev"] = np.nan
            totals_df = totals_df.sort_values(by=["kelly", "ev", "price"], ascending=[False, False, False], kind="mergesort")
            best_rows = totals_df.drop_duplicates(subset=["game_id","selection"], keep="first")
            stake_df = pd.concat([best_rows, others_df], ignore_index=True)
    # Daily risk cap: scale per-date stakes if exceeding cap
    if "date" in stake_df.columns and stake_df["date"].notna().any() and max_daily_risk_pct > 0:
        grouped = []
        for d, g in stake_df.groupby(stake_df["date"].astype(str)):
            total = g["stake"].sum()
            cap_amt = bankroll * max_daily_risk_pct
            scale = 1.0
            if total > cap_amt and cap_amt > 0:
                scale = cap_amt / total
            g = g.copy()
            g["stake"] = g["stake"] * scale
            grouped.append(g)
        stake_df = pd.concat(grouped, ignore_index=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        stake_df.to_csv(out, index=False)
    except Exception as e:
        print(f"[red]Failed writing stake sheet:[/red] {e}")
        raise typer.Exit(code=1)
    print(f"[green]Wrote {len(stake_df)} picks to[/green] {out}")


@app.command(name="train-distributional")
def train_distributional(
    features_csv: Path = typer.Argument(..., help="Features CSV with target_total column"),
    out_dir: Path = typer.Option(settings.outputs_dir / "models_dist", help="Directory for distributional totals models"),
    alpha_mu: float = typer.Option(1.0, help="Ridge alpha for mean model"),
    alpha_sigma: float = typer.Option(1.0, help="Ridge alpha for sigma model"),
    min_sigma: float = typer.Option(6.0, help="Floor for predicted sigma (points)"),
    sigma_mode: str = typer.Option("log", help="Sigma modeling mode: log|linear (log usually more stable)"),
    baseline_preds_csv: Path | None = typer.Option(None, help="Optional baseline predictions CSV to merge as feature (uses pred_total)"),
    build_calibration: bool = typer.Option(False, help="If set, build z-score recenter calibration artifact after training"),
):
    """Train distributional totals models (mean & sigma)."""
    try:
        res = train_distributional_totals(
            features_csv,
            out_dir,
            alpha_mu=alpha_mu,
            alpha_sigma=alpha_sigma,
            min_sigma=min_sigma,
            sigma_mode=sigma_mode,
            baseline_preds_csv=baseline_preds_csv,
            baseline_pred_col="pred_total",
        )
        print("[green]Distributional training complete:[/green]", res)
    except Exception as e:
        print(f"[red]Distributional training failed:[/red] {e}")
        raise typer.Exit(code=1)
    # Optional calibration artifact build
    if build_calibration:
        try:
            feats = pd.read_csv(features_csv)
            # Merge in mu/sigma predictions for calibration if produced paths recorded in res
            mu_path = Path(res.get("mu_model_path", "")) if isinstance(res, dict) else None
            sigma_path = Path(res.get("sigma_model_path", "")) if isinstance(res, dict) else None
            # If training function already saved a scored file, prefer that; else require scoring step externally.
            # For simplicity here assume features_csv has target_total and later user will run predict-distributional to produce preds before calibration; we attempt naive join if columns exist.
            if {"pred_total_mu","pred_total_sigma"}.issubset(feats.columns):
                calib_df = feats
            else:
                print("[yellow]Features CSV lacks pred_total_mu/sigma; run predict-distributional then re-run calibration separately.[/yellow]")
                calib_df = None
            if calib_df is not None:
                try:
                    art = build_z_recenter_artifact(calib_df, min_rows=200)
                    art_path = out_dir / "calibration_totals.json"
                    save_artifact(art, art_path)
                    print(f"[green]Saved calibration artifact ->[/green] {art_path} (n={art.n_samples})")
                except Exception as e:
                    print(f"[yellow]Calibration build skipped: {e}[/yellow]")
        except Exception as e:
            print(f"[yellow]Calibration phase failed: {e}[/yellow]")

@app.command(name="predict-distributional")
def predict_distributional(
    features_csv: Path = typer.Argument(..., help="Features CSV to score"),
    models_dir: Path = typer.Option(settings.outputs_dir / "models_dist", help="Directory containing dist_total_mu.npz & dist_total_sigma.npz"),
    baseline_preds_csv: Path | None = typer.Option(None, help="Optional baseline predictions CSV (columns: game_id,pred_total) for mu blending"),
    blend_weight: float = typer.Option(0.0, help="Weight to blend baseline pred_total into mu (0 disables, 0.3-0.6 typical early season)"),
    global_shift: float = typer.Option(0.0, help="Additive shift to blended mu (calibration toward market means)"),
    calibrate_to_baseline: bool = typer.Option(False, help="If set, scale blended mu to match baseline mean (capped)"),
    calibration_max_ratio: float = typer.Option(3.0, help="Max multiplier when calibrating mu to baseline mean"),
    sigma_cap: float = typer.Option(25.0, help="Cap predicted sigma at this value (0 disables)"),
    out: Path = typer.Option(settings.outputs_dir / "predictions_distributional.csv", help="Output CSV with pred_total_mu & pred_total_sigma"),
):
    """Generate distributional totals predictions (mean & sigma) with optional baseline mu blending and global shift."""
    try:
        baseline_df = None
        if baseline_preds_csv and baseline_preds_csv.exists():
            try:
                baseline_df = pd.read_csv(baseline_preds_csv)
            except Exception as e:
                print(f"[yellow]Failed reading baseline_preds_csv ({baseline_preds_csv}): {e}. Continuing without blend.[/yellow]")
                baseline_df = None
        out_df = predict_distributional_totals(
            features_csv,
            models_dir,
            baseline_preds=baseline_df,
            blend_weight=float(blend_weight),
            global_shift=float(global_shift),
            calibrate_to_baseline=bool(calibrate_to_baseline),
            calibration_max_ratio=float(calibration_max_ratio),
            sigma_cap=(float(sigma_cap) if sigma_cap and sigma_cap > 0 else None),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out, index=False)
        print(f"[green]Wrote distributional predictions to[/green] {out}")
    except Exception as e:
        print(f"[red]Distributional prediction failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(name="train-segmented-onnx")
def train_segmented_onnx_cmd(
    features_csv: Path = typer.Argument(..., help="Features CSV from build-features (multi-season recommended)"),
    segment: str = typer.Option("conference", help="Segmentation: team|conference"),
    out_dir: Path = typer.Option(settings.outputs_dir / "models", help="Root output models directory"),
    conf_map: Path | None = typer.Option(None, help="Optional team->conference CSV (columns: team,conference[,season]) for conference segmentation"),
    min_rows: int = typer.Option(200, help="Minimum rows required per segment to train a model"),
):
    """Train segmented baseline models per team or per conference.

    - team: trains one model per team using games where the team appears (home or away)
    - conference: trains one model per conference using intra-conference games (home_conf == away_conf == conf)
      and a special 'nonconf' model for inter-conference games
    """
    seg = segment.lower()
    if seg not in {"team", "conference"}:
        print("[red]segment must be 'team' or 'conference'[/red]")
        raise typer.Exit(code=1)
    df = pd.read_csv(features_csv)
    if {"target_total", "target_margin"}.difference(df.columns):
        print("[red]features CSV missing target columns. Rebuild with targets.[/red]")
        raise typer.Exit(code=1)
    root = out_dir / ("seg_team" if seg == "team" else "seg_conference")
    root.mkdir(parents=True, exist_ok=True)
    trained = []
    if seg == "team":
        teams = set()
        for c in ["home_team", "away_team"]:
            if c in df.columns:
                teams.update(df[c].astype(str).dropna().unique().tolist())
        for t in sorted(teams):
            key = _norm(t)
            if key in {"tbd", "tba", "unknown", "na", "n/a", ""}:
                continue
            mask = pd.Series(False, index=df.index)
            if "home_team" in df.columns:
                mask = mask | (df["home_team"].astype(str) == t)
            if "away_team" in df.columns:
                mask = mask | (df["away_team"].astype(str) == t)
            sub = df[mask]
            sub_valid = sub.dropna(subset=["target_total", "target_margin"]) if not sub.empty else sub
            if len(sub_valid) < min_rows:
                continue
            seg_dir = root / key
            seg_dir.mkdir(parents=True, exist_ok=True)
            tmp_csv = seg_dir / "features_segment.csv"
            sub_valid.to_csv(tmp_csv, index=False)
            _ = train_baseline(tmp_csv, seg_dir, loss_totals="huber", huber_delta=8.0)
            trained.append({"segment": "team", "key": t, "rows": len(sub_valid), "path": str(seg_dir)})
    else:
        cmap = _load_conf_map(conf_map if conf_map is not None else settings.data_dir / "conferences.csv")
        if not cmap:
            print("[yellow]No conference mapping found; skipping conference training.[/yellow]")
            raise typer.Exit(code=0)
        df["home_conf"] = df.get("home_team", pd.Series(dtype=str)).astype(str).map(lambda x: cmap.get(_norm(x)))
        df["away_conf"] = df.get("away_team", pd.Series(dtype=str)).astype(str).map(lambda x: cmap.get(_norm(x)))
        confs = sorted(set([c for c in pd.concat([df["home_conf"], df["away_conf"]]).dropna().unique().tolist() if c]))
        for conf in confs:
            sub = df[(df["home_conf"] == conf) & (df["away_conf"] == conf)]
            sub_valid = sub.dropna(subset=["target_total", "target_margin"]) if not sub.empty else sub
            if len(sub_valid) < min_rows:
                continue
            seg_dir = root / _norm(conf)
            seg_dir.mkdir(parents=True, exist_ok=True)
            tmp_csv = seg_dir / "features_segment.csv"
            sub_valid.to_csv(tmp_csv, index=False)
            _ = train_baseline(tmp_csv, seg_dir, loss_totals="huber", huber_delta=8.0)
            trained.append({"segment": "conference", "key": conf, "rows": len(sub_valid), "path": str(seg_dir)})
        nonconf = df[(df["home_conf"] != df["away_conf"]) | (df["home_conf"].isna()) | (df["away_conf"].isna())]
        nonconf_valid = nonconf.dropna(subset=["target_total", "target_margin"]) if not nonconf.empty else nonconf
        if len(nonconf_valid) >= max(100, min_rows // 2):
            seg_dir = root / "nonconf"
            seg_dir.mkdir(parents=True, exist_ok=True)
            tmp_csv = seg_dir / "features_segment.csv"
            nonconf_valid.to_csv(tmp_csv, index=False)
            _ = train_baseline(tmp_csv, seg_dir, loss_totals="huber", huber_delta=8.0)
            trained.append({"segment": "conference", "key": "nonconf", "rows": len(nonconf_valid), "path": str(seg_dir)})
    if trained:
        pd.DataFrame(trained).to_csv(root / "index.csv", index=False)
        print(f"[green]Trained {len(trained)} segmented models in[/green] {root}")
    else:
        print("[yellow]No segments met min_rows; no models trained.[/yellow]")


@app.command(name="join-closing")
def join_closing(
    games_path: Path = typer.Argument(..., help="Games file from fetch-games (CSV/Parquet)"),
    closing_path: Path = typer.Argument(..., help="Closing lines CSV from make-closing-lines"),
    out: Path = typer.Option(settings.outputs_dir / "games_with_closing.csv", help="Output merged CSV"),
):
    """Join closing lines to games by normalized team pair and date.

    Produces one row per game per bookmaker and market/period.
    """
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception as e:
            csv_alt = games_path.with_suffix(".csv")
            if csv_alt.exists():
                games = pd.read_csv(csv_alt)
                print(f"[yellow]Parquet engine unavailable ({e}). Using CSV fallback {csv_alt}[/yellow]")
            else:
                print(
                    "[red]Could not read games Parquet and no CSV fallback found.[/red]\n"
                    f"Looked for: {csv_alt}.\n"
                    "Re-run fetch-games and either install a Parquet engine or write directly to CSV, e.g.:\n"
                    "python -m ncaab_model.cli fetch-games --season 2025 --start YYYY-MM-DD --end YYYY-MM-DD --provider espn --out outputs/games.csv"
                )
                raise typer.Exit(code=1)
    closing = pd.read_csv(closing_path)
    merged = join_games_with_closing(games, closing)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"[green]Wrote merged games+closing to[/green] {out} ({len(merged)} rows)")


@app.command(name="join-last-odds")
def join_last_odds(
    games_path: Path = typer.Argument(..., help="Games file from fetch-games (CSV/Parquet)"),
    last_odds_path: Path = typer.Argument(..., help="last_odds.csv from make-last-odds"),
    out: Path = typer.Option(settings.outputs_dir / "games_with_last.csv", help="Output merged CSV (last odds per book)"),
    mapping_csv: Path | None = typer.Option(None, help="Optional mapping CSV (diagnostics odds_name_mapping_<date>.csv or data/team_map.csv)"),
    filter_exhibitions: bool = typer.Option(True, help="[Deprecated] legacy toggle; see --filter-mode"),
    filter_mode: str = typer.Option("any", help="D1 filter: 'both' keep only D1 vs D1; 'any' keep games with at least one D1 team; 'none' keep all"),
    allow_partial: bool = typer.Option(True, help="If exact pair not found, include partial matches where one team slug matches; rows are tagged partial_pair=True"),
):
    """Join last odds to games by normalized team pair and date (same logic as join-closing).

    Filtering behavior:
      - filter_mode="any" (default): keep games where at least one team is in the D1 list
      - filter_mode="both": keep only games where both teams are D1 (excludes D1 vs non-D1 exhibitions)
      - filter_mode="none": no D1-based filtering
    """
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception as e:
            csv_alt = games_path.with_suffix(".csv")
            if csv_alt.exists():
                games = pd.read_csv(csv_alt)
                print(f"[yellow]Parquet engine unavailable ({e}). Using CSV fallback {csv_alt}[/yellow]")
            else:
                print("[red]Could not read games Parquet and no CSV fallback found.[/red]")
                raise typer.Exit(code=1)
    last_df = pd.read_csv(last_odds_path)
    # Optional: filter out exhibitions via D1 list per selected mode
    mode = (filter_mode or "any").strip().lower()
    if mode not in {"both", "any", "none"}:
        mode = "any"
    # Back-compat: if legacy flag is explicitly false and no mode provided, treat as none; if true and mode is default, treat as both
    try:
        if filter_exhibitions is False and filter_mode == "any":
            mode = "none"
        elif filter_exhibitions is True and filter_mode == "any":
            # Historical default was both-D1; prefer 'any' unless explicitly forced by legacy expectations.
            # No-op here to keep new default; users can pass --filter-mode both to restore old behavior.
            pass
    except Exception:
        pass

    if mode != "none":
        try:
            d1 = pd.read_csv(settings.data_dir / "d1_conferences.csv")
            team_col = None
            for c in d1.columns:
                if str(c).strip().lower() in {"team", "name"}:
                    team_col = c
                    break
            if team_col is not None:
                from .data.merge_odds import normalize_name as _norm
                d1set = set(d1[team_col].astype(str).map(_norm).tolist())
                games["_home_ok"] = games.get("home_team", pd.Series(dtype=str)).astype(str).map(_norm).isin(d1set)
                games["_away_ok"] = games.get("away_team", pd.Series(dtype=str)).astype(str).map(_norm).isin(d1set)
                before = len(games)
                if mode == "both":
                    mask = games["_home_ok"] & games["_away_ok"]
                else:  # mode == "any"
                    mask = games["_home_ok"] | games["_away_ok"]
                games = games[mask].copy()
                games.drop(columns=["_home_ok","_away_ok"], inplace=True, errors="ignore")
                if len(games) < before:
                    removed = before - len(games)
                    detail = "both-D1 required" if mode == "both" else "kept any-D1"
                    print(f"[cyan]D1 filter ({detail}):[/cyan] {removed} games removed")
        except Exception as e:
            print(f"[yellow]D1 filter skipped:[/yellow] {e}")
    merged = join_games_with_closing(games, last_df, mapping_csv=mapping_csv, allow_partial=allow_partial)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"[green]Wrote merged games+last odds to[/green] {out} ({len(merged)} rows)")
    # Optional: compute edges if predictions present
    if {"pred_total", "pred_margin"}.issubset(set(merged.columns)):
        try:
            edged = compute_edges(merged)
            edged_path = out.parent / (out.stem + "_edges.csv")
            edged.to_csv(edged_path, index=False)
            print(f"[green]Edge metrics written ->[/green] {edged_path}")
        except Exception as e:
            print(f"[yellow]Edge computation skipped:[/yellow] {e}")


@app.command(name="compute-edges")
def compute_edges_cmd(
    merged_csv: Path = typer.Argument(..., help="Merged games+odds CSV (e.g., games_with_last.csv)."),
    predictions_csv: Path | None = typer.Option(None, help="Optional predictions CSV with columns [game_id,pred_total,pred_margin] to merge before computing edges."),
    out: Path = typer.Option(None, help="Explicit output CSV path for edges (defaults to *_edges.csv)"),
):
    """Compute prediction vs market edges (totals, spreads, moneyline) and Kelly fractions.

    If predictions are not present in merged_csv, provide --predictions-csv to merge by game_id.
    """
    if not merged_csv.exists():
        print(f"[red]Merged file not found:[/red] {merged_csv}")
        raise typer.Exit(code=1)
    df = pd.read_csv(merged_csv)
    if not {"pred_total", "pred_margin"}.issubset(df.columns):
        if predictions_csv is None or not predictions_csv.exists():
            print("[red]Missing prediction columns and no predictions CSV provided. Supply --predictions-csv.[/red]")
            raise typer.Exit(code=1)
        try:
            preds = pd.read_csv(predictions_csv)
            for c in ("game_id", "pred_total", "pred_margin"):
                if c not in preds.columns:
                    print(f"[red]predictions CSV missing required column:[/red] {c}")
                    raise typer.Exit(code=1)
            df["game_id"] = df.get("game_id").astype(str)
            preds["game_id"] = preds.get("game_id").astype(str)
            df = df.merge(preds[["game_id","pred_total","pred_margin"]], on="game_id", how="left")
        except Exception as e:
            print(f"[red]Failed merging predictions:[/red] {e}")
            raise typer.Exit(code=1)
    edged = compute_edges(df)
    out_path = out if out is not None else merged_csv.parent / (merged_csv.stem + "_edges.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edged.to_csv(out_path, index=False)
    print(f"[green]Wrote edges to[/green] {out_path} ({len(edged)} rows)")


@app.command(name="align-period-preds")
def align_period_preds(
    merged_csv: Path = typer.Option(settings.outputs_dir / "games_with_last.csv", help="Merged games+odds CSV (must include 'period' column)"),
    predictions_csv: Path = typer.Option(settings.outputs_dir / "predictions_week.csv", help="Predictions CSV with columns [game_id,pred_total,pred_margin,(optional halves)]"),
    out: Path = typer.Option(settings.outputs_dir / "merged_with_period_preds.csv", help="Output CSV with pred_total/pred_margin aligned to row period"),
    half_ratio: float = typer.Option(0.485, help="If half projections missing, derive 1H total as pred_total*half_ratio and 2H as remainder"),
    margin_half_ratio: float = typer.Option(0.5, help="If half margin projections missing, derive 1H margin as pred_margin*margin_half_ratio (2H remainder)"),
    write_edges: bool = typer.Option(True, help="Also compute edges and write *_edges.csv (or to --edges-out if provided)"),
    edges_out: Path | None = typer.Option(None, help="Optional explicit output path for edges CSV"),
):
    """Attach predictions and align them to each odds row's period, optionally writing edges.

    - Inputs: merged_csv produced by join-last-odds or join-closing (columns: game_id, market, period, total/spreads/ml...)
    - Predictions: baseline with optional half projections (pred_total_1h/_2h, pred_margin_1h/_2h)
    - Output: pred_total/pred_margin aligned to period (full_game uses pred_total/pred_margin; 1H/2H use half projections or derived heuristics)
    """
    if not merged_csv.exists():
        print(f"[red]Merged odds file not found:[/red] {merged_csv}")
        raise typer.Exit(code=1)
    if not predictions_csv.exists():
        print(f"[red]Predictions file not found:[/red] {predictions_csv}")
        raise typer.Exit(code=1)
    try:
        m = pd.read_csv(merged_csv)
        p = pd.read_csv(predictions_csv)
    except Exception as e:
        print(f"[red]Failed reading inputs:[/red] {e}")
        raise typer.Exit(code=1)
    if "period" not in m.columns:
        print("[red]merged_csv missing 'period' column; cannot align by period.[/red]")
        raise typer.Exit(code=1)
    # Normalize dtypes
    for df in (m, p):
        if "game_id" in df.columns:
            df["game_id"] = df["game_id"].astype(str)
    # Merge available prediction columns
    # Support calibrated prediction columns by mapping them to canonical names if base absent
    if "pred_total" not in p.columns and "pred_total_calibrated" in p.columns:
        p.rename(columns={"pred_total_calibrated":"pred_total"}, inplace=True)
    if "pred_margin" not in p.columns and "pred_margin_calibrated" in p.columns:
        p.rename(columns={"pred_margin_calibrated":"pred_margin"}, inplace=True)
    keep = [
        "game_id",
        "pred_total",
        "pred_margin",
        "pred_total_1h",
        "pred_total_2h",
        "pred_margin_1h",
        "pred_margin_2h",
    ]
    pcols = [c for c in keep if c in p.columns]
    if not {"game_id","pred_total","pred_margin"}.issubset(pcols):
        print("[yellow]Predictions missing pred_total/pred_margin; alignment will yield empty preds.[/yellow]")
    e = m.merge(p[pcols], on="game_id", how="left")
    per = e["period"].astype(str).str.lower()
    # Start with full-game
    pt = e.get("pred_total")
    pm = e.get("pred_margin")
    # If explicit half projections exist, prefer them
    if "pred_total_1h" in e.columns:
        pt = np.where(per == "1h", e["pred_total_1h"], pt)
    if "pred_total_2h" in e.columns:
        pt = np.where(per == "2h", e["pred_total_2h"], pt)
    if "pred_margin_1h" in e.columns:
        pm = np.where(per == "1h", e["pred_margin_1h"], pm)
    if "pred_margin_2h" in e.columns:
        pm = np.where(per == "2h", e["pred_margin_2h"], pm)
    # If halves not present, derive from full-game heuristics
    try:
        hr = float(half_ratio)
    except Exception:
        hr = 0.485
    try:
        mhr = float(margin_half_ratio)
    except Exception:
        mhr = 0.5
    # Build fallback half projections only for rows where half-specific columns are missing
    # Use logical checks rather than columns.isin(...) any() misuse; just test column presence
    if "pred_total_1h" not in e.columns:
        pt = np.where(per == "1h", pd.to_numeric(e.get("pred_total"), errors="coerce") * hr, pt)
    if "pred_total_2h" not in e.columns:
        pt = np.where(per == "2h", pd.to_numeric(e.get("pred_total"), errors="coerce") * (1.0 - hr), pt)
    if "pred_margin_1h" not in e.columns:
        pm = np.where(per == "1h", pd.to_numeric(e.get("pred_margin"), errors="coerce") * mhr, pm)
    if "pred_margin_2h" not in e.columns:
        pm = np.where(per == "2h", pd.to_numeric(e.get("pred_margin"), errors="coerce") * (1.0 - mhr), pm)

    # Preserve originals if present for reference
    if "pred_total" in e.columns:
        e.rename(columns={"pred_total": "pred_total_full"}, inplace=True)
    if "pred_margin" in e.columns:
        e.rename(columns={"pred_margin": "pred_margin_full"}, inplace=True)
    # Set aligned canonical columns
    e["pred_total"] = pd.to_numeric(pt, errors="coerce")
    e["pred_margin"] = pd.to_numeric(pm, errors="coerce")

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        e.to_csv(out, index=False)
    except Exception as ex:
        print(f"[red]Failed writing output:[/red] {ex}")
        raise typer.Exit(code=1)
    print(f"[green]Wrote period-aligned predictions to[/green] {out} ({len(e)} rows)")

    if write_edges:
        try:
            edged = compute_edges(e)
            eout = edges_out if edges_out is not None else out.parent / (out.stem + "_edges.csv")
            edged.to_csv(eout, index=False)
            print(f"[green]Edges written ->[/green] {eout} ({len(edged)} rows)")
        except Exception as ex:
            print(f"[yellow]Failed to compute edges:[/yellow] {ex}")


@app.command(name="backfill-range")
def backfill_range(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD (inclusive)"),
    end: str = typer.Argument(..., help="End date YYYY-MM-DD (inclusive)"),
    provider: str = typer.Option("espn", help="Games provider: espn|ncaa|fused"),
    region: str = typer.Option("us", help="Odds region for TheOddsAPI"),
    season: int | None = typer.Option(None, help="Season year override; default derives from each date"),
    use_cache: bool = typer.Option(True, help="Use provider cache (set --no-use-cache to refresh)"),
    enable_ort: bool = typer.Option(False, help="Attempt ORT/QNN; defaults to False for backfill speed"),
    preseason_weight: float = typer.Option(0.0, help="Preseason blend weight for early-season dates"),
    preseason_only_sparse: bool = typer.Option(True, help="Only apply preseason blend for sparse features"),
    apply_guardrails: bool = typer.Option(True, help="Apply totals guardrail blend for implausibly low predictions"),
    accumulate_schedule: bool = typer.Option(True, help="Accumulate into games_all.csv"),
    accumulate_predictions: bool = typer.Option(True, help="Accumulate into predictions_all.csv"),
):
    """Run daily-run for every date in [start, end], writing dated artifacts.

    This backfills games_YYYY-MM-DD.csv, odds_YYYY-MM-DD.csv, games_with_odds_YYYY-MM-DD.csv,
    predictions_YYYY-MM-DD.csv and updates games_all.csv/predictions_all.csv.
    The join-last-odds default keeps games with at least one D1 team via filter_mode='any'.
    """
    try:
        d0 = dt.date.fromisoformat(start)
        d1 = dt.date.fromisoformat(end)
    except Exception:
        print("[red]Invalid start/end date. Use YYYY-MM-DD.[/red]")
        raise typer.Exit(code=1)
    if d1 < d0:
        print("[red]End date is before start date.[/red]")
        raise typer.Exit(code=1)
    cur = d0
    n_ok = 0
    n_err = 0
    while cur <= d1:
        try:
            print(f"[cyan]Backfill {cur}[/cyan] provider={provider} cache={use_cache}")
            daily_run(
                date=cur.isoformat(),
                season=(season if season is not None else cur.year),
                region=region,
                provider=provider,
                threshold=2.0,
                default_price=-110.0,
                retrain=False,
                segment="none",
                conf_map=None,
                use_cache=use_cache,
                preseason_weight=preseason_weight,
                preseason_only_sparse=preseason_only_sparse,
                db=settings.data_dir / "ncaab.sqlite",
                book_whitelist=None,
                target_picks=None,
                apply_guardrails=apply_guardrails,
                half_ratio=0.485,
                auto_train_halves=False,
                halves_models_dir=settings.outputs_dir / "models_halves",
                enable_ort=enable_ort,
                accumulate_schedule=accumulate_schedule,
                accumulate_predictions=accumulate_predictions,
            )
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[red]Backfill failed for {cur}:[/red] {e}")
        cur += dt.timedelta(days=1)
    print({"ok_days": n_ok, "err_days": n_err, "start": d0.isoformat(), "end": d1.isoformat(), "provider": provider})

@app.command(name="backfill-last-odds")
def backfill_last_odds(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD inclusive"),
    end: str = typer.Argument(..., help="End date YYYY-MM-DD inclusive"),
    region: str = typer.Option("us", help="Odds region (unused in backfill; assumes snapshots already exist)"),
    games_dir: Path = typer.Option(settings.outputs_dir, help="Directory containing games_<date>.csv files"),
    odds_history_dir: Path = typer.Option(settings.outputs_dir / "odds_history", help="Directory containing odds_<date>.csv snapshots"),
    out_merged: Path = typer.Option(settings.outputs_dir / "games_with_last.csv", help="Master merged output (concatenated)"),
    tolerance_seconds: int = typer.Option(60, help="Skew tolerance for last selection per date"),
    filter_mode: str = typer.Option("any", help="D1 filter mode: any|both|none applied post-merge"),
    allow_partial: bool = typer.Option(True, help="Allow partial pair matches when exact pair absent (tag partial_pair)"),
    overwrite: bool = typer.Option(False, help="Overwrite per-date games_with_last_<date>.csv if present"),
):
    """Rebuild historical last odds merges across a date range and refresh master games_with_last.csv.

    Expects odds_history/odds_<date>.csv and games_<date>.csv to exist (produce via backfill-range or daily pipeline).
    For each date:
      - Select last pre-tip odds rows (per event/book/market/period) honoring tolerance_seconds after commence_time.
      - Join to games_<date>.csv producing games_with_last_<date>.csv (respecting D1 filter).
    Concatenate all per-date merged files into a new master games_with_last.csv.
    Writes summary to games_with_last_backfill_summary.csv.
    """
    import datetime as _dt
    import pandas as _pd
    from .data.merge_odds import normalize_name as _norm
    # Generic pair/date join helper located in this module (import locally to avoid circulars)
    try:
        from .data.odds_closing import compute_last_odds  # noqa: F401
    except Exception:
        pass

    d0 = _dt.date.fromisoformat(start)
    d1 = _dt.date.fromisoformat(end)
    if d1 < d0:
        raise typer.BadParameter("end date precedes start date")
    dates = []
    cur = d0
    while cur <= d1:
        dates.append(cur)
        cur += _dt.timedelta(days=1)

    summary_rows = []
    per_paths = []
    for day in dates:
        iso = day.isoformat()
        g_path = games_dir / f"games_{iso}.csv"
        o_path = odds_history_dir / f"odds_{iso}.csv"
        if not g_path.exists():
            summary_rows.append({"date": iso, "status": "missing_games"})
            continue
        if not o_path.exists():
            summary_rows.append({"date": iso, "status": "missing_odds"})
            continue
        try:
            games_df = _pd.read_csv(g_path)
            odds_df = _pd.read_csv(o_path)
            if odds_df.empty:
                summary_rows.append({"date": iso, "status": "empty_odds"})
                continue
            # Normalize time columns
            if 'commence_time' in odds_df.columns:
                odds_df['commence_time'] = _pd.to_datetime(odds_df['commence_time'], errors='coerce')
            if 'last_update' in odds_df.columns:
                odds_df['last_update'] = _pd.to_datetime(odds_df['last_update'], errors='coerce')
            tol = _dt.timedelta(seconds=int(max(0, tolerance_seconds)))
            key_cols = [c for c in ['event_id','book','market','period'] if c in odds_df.columns]
            def _select(grp):
                ct = grp['commence_time'].iloc[0] if 'commence_time' in grp.columns else None
                if ct is not None and not _pd.isna(ct):
                    grp = grp[grp['last_update'] <= ct + tol]
                grp = grp.sort_values('last_update')
                return grp.tail(1)
            if key_cols:
                last_df = _pd.concat([_select(g) for _, g in odds_df.groupby(key_cols)], ignore_index=True)
            else:
                last_df = odds_df.copy()
            per_date_out = games_dir / f"games_with_last_{iso}.csv"
            if per_date_out.exists() and not overwrite:
                status = "exists"
                per_paths.append(per_date_out)
                summary_rows.append({"date": iso, "status": status})
                continue
            # Reuse existing join helper via closure from earlier scope (join_games_with_closing defined above with closing merge logic)
            try:
                from .data.odds_closing import compute_last_odds as _dummy  # force module load
            except Exception:
                pass
            # We replicate minimal join: normalized pair_key on games & odds then left merge by pair+date
            def _norm_pair(s: str) -> str:
                return str(s).lower().replace(' ', '').replace('.', '')
            games_df = games_df.copy()
            games_df['date'] = pd.to_datetime(games_df.get('date'), errors='coerce')
            games_df['game_date'] = games_df['date'].dt.date
            games_df['pair_key'] = games_df['away_team'].map(_norm_pair) + "::" + games_df['home_team'].map(_norm_pair)
            last_df = last_df.copy()
            if 'home_team_name' in last_df.columns and 'away_team_name' in last_df.columns:
                last_df['pair_key'] = last_df['away_team_name'].map(_norm_pair) + "::" + last_df['home_team_name'].map(_norm_pair)
            if 'commence_time' in last_df.columns:
                last_df['commence_time'] = pd.to_datetime(last_df['commence_time'], errors='coerce')
                last_df['odds_date'] = last_df['commence_time'].dt.date
            else:
                last_df['odds_date'] = pd.NaT
            merged = games_df.merge(
                last_df,
                left_on=['pair_key','game_date'],
                right_on=['pair_key','odds_date'],
                how='left',
                suffixes=("","_odds")
            )
            # D1 filter post-merge
            mode = (filter_mode or 'any').lower().strip()
            if mode in {'any','both'}:
                try:
                    d1_df = _pd.read_csv(settings.data_dir / 'd1_conferences.csv')
                    d1_teams = set(d1_df['team'].astype(str).map(_norm))
                    merged['_home_ok'] = merged['home_team'].astype(str).map(_norm).isin(d1_teams)
                    merged['_away_ok'] = merged['away_team'].astype(str).map(_norm).isin(d1_teams)
                    if mode == 'both':
                        msk = merged['_home_ok'] & merged['_away_ok']
                    else:
                        msk = merged['_home_ok'] | merged['_away_ok']
                    merged = merged[msk].copy()
                    merged.drop(columns=['_home_ok','_away_ok'], inplace=True)
                except Exception:
                    pass
            merged.to_csv(per_date_out, index=False)
            per_paths.append(per_date_out)
            summary_rows.append({"date": iso, "status": "written", "games": int(len(games_df)), "last_rows": int(len(last_df)), "merged_rows": int(len(merged))})
        except Exception as e:
            summary_rows.append({"date": iso, "status": f"error:{e}"})

    # Concatenate per-date merges
    frames = []
    for p in per_paths:
        try:
            frames.append(_pd.read_csv(p))
        except Exception:
            pass
    if frames:
        master = _pd.concat(frames, ignore_index=True)
        out_merged.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(out_merged, index=False)
        print(f"[green]Refreshed master games_with_last.csv[/green] ({len(master)} rows)")
    else:
        print("[yellow]No per-date merged files produced; master not updated.[/yellow]")
    summary_path = games_dir / "games_with_last_backfill_summary.csv"
    _pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[green]Backfill summary written ->[/green] {summary_path}")


@app.command(name="merge-predictions-odds")
def merge_predictions_odds(
    games_path: Path = typer.Argument(..., help="Games CSV for the target slate (from fetch-games)"),
    odds_path: Path = typer.Argument(..., help="Odds CSV (from fetch-odds or last/closing builders)"),
    predictions_path: Path = typer.Argument(..., help="Baseline predictions CSV (requires game_id,pred_total,pred_margin)"),
    dist_predictions_path: Path | None = typer.Option(None, help="Optional distributional predictions CSV with pred_total_mu,pred_total_sigma"),
    out: Path = typer.Option(settings.outputs_dir / "merged_odds_predictions.csv", help="Output merged CSV"),
    coverage_report: bool = typer.Option(True, help="Emit coverage diagnostics and unmatched CSV sidecars"),
    enforce_date_match: bool = typer.Option(True, help="Warn (and optionally skip attach) when prediction dates differ from games date set"),
):
    """Merge games, odds and predictions (baseline + optional distributional) into a single wide CSV.

    Result: one row per bookmaker/market/period with attached model outputs ready for compute-edges or bankroll-optimize.
    """
    for p in (games_path, odds_path, predictions_path):
        if not p.exists():
            print(f"[red]Input file missing:[/red] {p}")
            raise typer.Exit(code=1)
    try:
        games = pd.read_csv(games_path)
        odds = pd.read_csv(odds_path)
        preds = pd.read_csv(predictions_path)
        # Date consistency check
        game_dates = set(games.get("date", pd.Series(dtype=str)).astype(str).dropna().unique().tolist()) if "date" in games.columns else set()
        pred_dates = set(preds.get("date", pd.Series(dtype=str)).astype(str).dropna().unique().tolist()) if "date" in preds.columns else set()
        date_mismatch = bool(game_dates and pred_dates and game_dates.isdisjoint(pred_dates))
        if date_mismatch:
            print(f"[yellow]Prediction file dates ({sorted(list(pred_dates))[:5]}) do not overlap games dates ({sorted(list(game_dates))[:5]}). Attach will yield empty baseline columns.[/yellow]")
    except Exception as e:
        print(f"[red]Failed reading input CSVs:[/red] {e}")
        raise typer.Exit(code=1)

    def _norm(x: str) -> str:
        return str(x).lower().replace(' ', '').replace('.', '')

    # Construct pair_key in games if needed
    if {"home_team", "away_team"}.issubset(games.columns):
        games["home_team"] = games["home_team"].astype(str)
        games["away_team"] = games["away_team"].astype(str)
        games["pair_key"] = games["away_team"].map(_norm) + "::" + games["home_team"].map(_norm)
    elif "pair_key" not in games.columns:
        print("[red]games file missing team columns/pair_key; cannot construct key.")
        raise typer.Exit(code=1)

    # Build pair_key in odds from available team name columns
    home_name_col = None
    away_name_col = None
    for c in odds.columns:
        lc = str(c).lower()
        if lc in {"home_team_name", "home_team"}:
            home_name_col = c
        if lc in {"away_team_name", "away_team"}:
            away_name_col = c
    if home_name_col and away_name_col:
        odds[home_name_col] = odds[home_name_col].astype(str)
        odds[away_name_col] = odds[away_name_col].astype(str)
        odds["pair_key"] = odds[away_name_col].map(_norm) + "::" + odds[home_name_col].map(_norm)
    elif "pair_key" not in odds.columns:
        print("[red]odds file missing team name columns/pair_key; cannot construct key.")
        raise typer.Exit(code=1)

    # Merge odds -> games (attach game_id,date)
    games_sub = games[[c for c in ("game_id", "date", "pair_key") if c in games.columns]].copy()
    if "game_id" in games_sub.columns:
        games_sub["game_id"] = games_sub["game_id"].astype(str)
    merged = odds.merge(games_sub, on="pair_key", how="left")
    # Normalize game_id column post-merge (avoid _x/_y ambiguity)
    if "game_id_y" in merged.columns:
        merged["game_id"] = merged["game_id_y"]
    elif "game_id" in merged.columns:
        merged["game_id"] = merged["game_id"]
    elif "game_id_x" in merged.columns:
        merged["game_id"] = merged["game_id_x"]
    # Clean potential duplicates
    for c in ("game_id_x", "game_id_y"):
        if c in merged.columns:
            merged.drop(columns=[c], inplace=True)

    # Merge baseline predictions by game_id
    req = {"game_id", "pred_total", "pred_margin"}
    if not req.issubset(preds.columns):
        print(f"[red]Predictions CSV missing required columns {req}.[/red]")
        raise typer.Exit(code=1)
    preds["game_id"] = preds["game_id"].astype(str)
    if "game_id" in merged.columns:
        merged["game_id"] = merged["game_id"].astype(str)
    else:
        merged["game_id"] = pd.NA
    # Attach baseline predictions only if dates overlap or enforcement disabled
    if not date_mismatch or not enforce_date_match:
        merged = merged.merge(preds[["game_id", "pred_total", "pred_margin"]], on="game_id", how="left")
    else:
        # Create empty columns for clarity
        for c in ("pred_total", "pred_margin"):
            if c not in merged.columns:
                merged[c] = pd.NA
        print("[yellow]Skipped baseline attachment due to date mismatch (use --no-enforce-date-match to override).[/yellow]")

    # Fallback slug join if game_id missing widely (attempt to recover joins when upstream provider naming differs)
    if merged.get("game_id").isna().sum() > 0:
        try:
            # Build slug keys for odds and games
            def _slug(s: str) -> str:
                return str(s).lower().replace(' ', '').replace('.', '')
            if {"home_team", "away_team"}.issubset(games.columns):
                games_slugs = games[["game_id", "home_team", "away_team"]].copy()
                games_slugs["home_slug"] = games_slugs["home_team"].map(_slug)
                games_slugs["away_slug"] = games_slugs["away_team"].map(_slug)
                odds_slugs = merged[[c for c in merged.columns if c in {"home_team_name", "away_team_name", "game_id", "pair_key"}]].copy()
                # Preserve existing columns for merge
                odds_slugs["home_slug"] = odds_slugs.get("home_team_name").map(_slug)
                odds_slugs["away_slug"] = odds_slugs.get("away_team_name").map(_slug)
                # Identify rows lacking game_id to attempt recovery
                missing_mask = merged["game_id"].isna()
                rec = odds_slugs[missing_mask].merge(games_slugs[["game_id", "home_slug", "away_slug"]], on=["home_slug", "away_slug"], how="left")
                if rec["game_id_y"].notna().any():
                    # Update only rows where recovery succeeded
                    merged.loc[missing_mask, "game_id"] = rec["game_id_y"].values
                    print(f"[green]Recovered {rec['game_id_y'].notna().sum()} game_id values via slug fallback.[/green]")
        except Exception as e:
            print(f"[yellow]Slug fallback join failed: {e}[/yellow]")

    # Optional distributional merge
    if dist_predictions_path is not None and dist_predictions_path.exists():
        try:
            dist = pd.read_csv(dist_predictions_path)
            if {"game_id", "pred_total_mu", "pred_total_sigma"}.issubset(dist.columns):
                dist["game_id"] = dist["game_id"].astype(str)
                # Optional distributional date consistency similar to baseline
                dist_dates = set(dist.get("date", pd.Series(dtype=str)).astype(str).dropna().unique().tolist()) if "date" in dist.columns else set()
                dist_mismatch = bool(game_dates and dist_dates and game_dates.isdisjoint(dist_dates))
                if not dist_mismatch or not enforce_date_match:
                    merged = merged.merge(dist[["game_id", "pred_total_mu", "pred_total_sigma"]], on="game_id", how="left")
                else:
                    for c in ("pred_total_mu", "pred_total_sigma"):
                        if c not in merged.columns:
                            merged[c] = pd.NA
                    print("[yellow]Skipped distributional attachment due to date mismatch (override with --no-enforce-date-match).[/yellow]")
                print("[green]Attached distributional predictions (mu,sigma).[/green]")
            else:
                print("[yellow]Distributional predictions CSV missing mu/sigma; skipped.[/yellow]")
        except Exception as e:
            print(f"[yellow]Failed reading distributional predictions; skipped: {e}[/yellow]")

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"[green]Wrote merged odds+predictions to[/green] {out} ({len(merged)} rows)")

    if coverage_report:
        try:
            total_rows = len(merged)
            with_gid = merged["game_id"].notna().sum()
            with_preds = merged["pred_total"].notna().sum()
            with_dist = merged["pred_total_mu"].notna().sum() if "pred_total_mu" in merged.columns else 0
            unmatched_odds = merged[merged["game_id"].isna()].copy()
            unmatched_games = games_sub[~games_sub["game_id"].isin(merged["game_id"].dropna().unique())].copy()
            side_dir = out.parent
            if not unmatched_odds.empty:
                uo_path = side_dir / (out.stem + "_unmatched_odds.csv")
                unmatched_odds.to_csv(uo_path, index=False)
                print(f"[yellow]Wrote {len(unmatched_odds)} unmatched odds rows -> {uo_path}[/yellow]")
            if not unmatched_games.empty:
                ug_path = side_dir / (out.stem + "_unmatched_games.csv")
                unmatched_games.to_csv(ug_path, index=False)
                print(f"[yellow]Wrote {len(unmatched_games)} games without odds -> {ug_path}[/yellow]")
            print(f"[blue]Coverage: rows={total_rows}, rows_with_game_id={with_gid}, rows_with_baseline={with_preds}, rows_with_dist={with_dist}[/blue]")
        except Exception as e:
            print(f"[yellow]Coverage diagnostics failed: {e}[/yellow]")

    # Best-effort edges artifact alongside
    try:
        if {"pred_total", "pred_margin"}.issubset(merged.columns):
            edged = compute_edges(merged)
            edged_path = out.parent / (out.stem + "_edges.csv")
            edged.to_csv(edged_path, index=False)
            print(f"[green]Edge metrics written ->[/green] {edged_path}")
    except Exception as e:
        print(f"[yellow]Edges computation skipped: {e}[/yellow]")


@app.command(name="train-preseason")
def train_preseason(
    features_csv: Path = typer.Argument(settings.outputs_dir / "features_hist.csv", help="Historical features CSV (multi-season)"),
    out_dir: Path = typer.Option(settings.outputs_dir / "models_preseason", help="Output dir for preseason models"),
):
    """Train a priors-only (preseason) linear model for totals and margins.

    Uses rating-based features only: rating_margin_diff, off_rating_diff, def_rating_diff, tempo_rating_sum, neutral_site.
    """
    if not features_csv.exists():
        print(f"[red]Missing features file:[/red] {features_csv}")
        raise typer.Exit(code=1)
    df = pd.read_csv(features_csv)
    # Build derived priors features if not present
    def _ensure(df_: pd.DataFrame, out_col: str, expr: callable):
        if out_col not in df_.columns:
            try:
                df_[out_col] = expr(df_)
            except Exception:
                df_[out_col] = pd.NA
    _ensure(df, "rating_margin_diff", lambda d: d.get("home_rating_margin") - d.get("away_rating_margin"))
    _ensure(df, "off_rating_diff", lambda d: d.get("home_off_rating") - d.get("away_off_rating"))
    _ensure(df, "def_rating_diff", lambda d: d.get("home_def_rating") - d.get("away_def_rating"))
    _ensure(df, "tempo_rating_sum", lambda d: d.get("home_tempo_rating") + d.get("away_tempo_rating"))
    # Keep only necessary columns + targets
    keep = [
        "game_id", "date", "home_team", "away_team",
        "rating_margin_diff", "off_rating_diff", "def_rating_diff", "tempo_rating_sum",
        "neutral_site", "target_total", "target_margin",
    ]
    missing_targets = {"target_total", "target_margin"}.difference(df.columns)
    if missing_targets:
        print(f"[red]features CSV missing targets {missing_targets}. Rebuild with targets enabled.[/red]")
        raise typer.Exit(code=1)
    tdf = df[[c for c in keep if c in df.columns]].copy()
    # Coerce neutral_site to numeric 0/1 if present
    if "neutral_site" in tdf.columns:
        try:
            tdf["neutral_site"] = tdf["neutral_site"].fillna(False).astype(int)
        except Exception:
            pass
    tmp = out_dir / "features_priors_only.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(tmp, index=False)
    res = train_baseline(tmp, out_dir)
    print(f"[green]Preseason training complete.[/green] {res}")
    print(f"Models at {out_dir}")


@app.command(name="train-segmented")
def train_segmented_cmd(
    features_csv: Path = typer.Argument(settings.outputs_dir / "features_hist.csv", help="Features CSV with targets (multi-season recommended)"),
    out_dir: Path = typer.Option(settings.outputs_dir / "models_segmented", help="Output directory for per-segment JSONL models"),
    segment: str = typer.Option("team", help="Segmentation key: team|conference"),
    min_rows: int = typer.Option(25, help="Minimum rows per segment to fit a model"),
    alpha: float = typer.Option(1.0, help="Ridge regularization strength"),
):
    """Train per-segment linear models over teams or conferences.

    Produces a JSONL file with one small linear model per segment (totals and margins). Use features_hist.csv
    for best coverage. Requires target_total/target_margin columns.
    """
    if not features_csv.exists():
        print(f"[red]Missing features file:[/red] {features_csv}")
        raise typer.Exit(code=1)
    seg = (segment or "team").strip().lower()
    if seg not in {"team", "conference"}:
        print("[red]segment must be 'team' or 'conference'[/red]")
        raise typer.Exit(code=1)
    try:
        res = train_segmented(features_csv, out_dir, segment=seg, min_rows=int(min_rows), alpha=float(alpha))
        print(f"[green]Segmented training complete.[/green] {res}")
    except Exception as e:
        print(f"[red]Segmented training failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(name="backtest")
def backtest(
    games_path: Path = typer.Argument(..., help="Games file from fetch-games (CSV/Parquet)"),
    odds_path: Path = typer.Argument(..., help="Odds CSV from fetch-odds"),
    preds_path: Path = typer.Argument(..., help="Predictions CSV from predict-baseline"),
    out: Path = typer.Option(settings.outputs_dir / "backtest_bets.csv", help="Per-bet results CSV"),
    threshold: float = typer.Option(2.0, help="Bet when |pred_total - line_total| >= threshold"),
    price: float = typer.Option(-110.0, help="Assumed American price for totals if not provided"),
):
    """Backtest totals strategy using predictions vs bookmaker totals lines.

    Outputs a CSV of per-bet rows and prints a summary (bets, win rate, PnL units, ROI, avg edge).
    """
    # Load games flexibly
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))
    odds = pd.read_csv(odds_path)
    preds = pd.read_csv(preds_path)

    bets, summary = backtest_totals(games, odds, preds, threshold=threshold, default_price=price)
    out.parent.mkdir(parents=True, exist_ok=True)
    bets.to_csv(out, index=False)
    print(f"[green]Backtest complete.[/green] Wrote {len(bets)} rows to {out}")
    print({
        "n_games": summary.n_games,
        "n_books_rows": summary.n_books_rows,
        "n_bets": summary.n_bets,
        "n_resolved": summary.n_resolved,
        "win_rate": summary.win_rate,
        "pnl_units": summary.pnl_units,
        "roi": summary.roi,
        "avg_edge": summary.avg_edge,
    })

@app.command(name="produce-picks")
def produce_picks(
    preds_path: Path = typer.Argument(..., help="Predictions CSV with full + half projections (pred_total, pred_margin, pred_total_1h, pred_margin_1h, etc.)"),
    odds_merged_path: Path = typer.Argument(..., help="Merged odds file (games_with_last.csv or games_with_closing.csv) containing market, period, spreads/totals/moneylines"),
    out: Path = typer.Option(settings.outputs_dir / "picks_raw.csv", help="Output CSV with per-market picks and edges"),
    total_threshold: float = typer.Option(1.5, help="Min |pred_total - line_total| for Over/Under pick"),
    spread_threshold: float = typer.Option(1.0, help="Min |pred_margin - implied_margin| for spread pick"),
    moneyline_margin_scale: float = typer.Option(7.0, help="Scale converting margin to win prob via logistic"),
    moneyline_edge_pct: float = typer.Option(2.0, help="Min % edge (fair vs book implied prob) for moneyline pick"),
):
    """Generate picks (totals, spreads, moneyline) for full game and halves using predictions vs odds.

    odds_merged_path must include columns: game_id, book, market (h2h|spreads|totals), period (full_game|1h|2h), plus line fields.
    """
    if not preds_path.exists():
        print(f"[red]Missing predictions file {preds_path}[/red]"); raise typer.Exit(code=1)
    if not odds_merged_path.exists():
        print(f"[red]Missing merged odds file {odds_merged_path}[/red]"); raise typer.Exit(code=1)
    preds = pd.read_csv(preds_path)
    odds = pd.read_csv(odds_merged_path)
    preds["game_id"] = preds.get("game_id").astype(str)
    odds["game_id"] = odds.get("game_id").astype(str)
    # Build long-form predictions per period
    rows: list[dict] = []
    for _, r in preds.iterrows():
        gid = str(r.get("game_id"))
        base = {"game_id": gid, "date": r.get("date"), "home_team": r.get("home_team"), "away_team": r.get("away_team")}
        rows.append({**base, "period": "full_game", "model_total": r.get("pred_total"), "model_margin": r.get("pred_margin")})
        if pd.notna(r.get("pred_total_1h")):
            rows.append({**base, "period": "1h", "model_total": r.get("pred_total_1h"), "model_margin": r.get("pred_margin_1h")})
        if pd.notna(r.get("pred_total_2h")):
            rows.append({**base, "period": "2h", "model_total": r.get("pred_total_2h"), "model_margin": r.get("pred_margin_2h")})
    p_long = pd.DataFrame(rows)
    if p_long.empty or "period" not in odds.columns:
        print("[red]Predictions lack half projections or odds file missing period column.[/red]"); raise typer.Exit(code=1)
    merged = odds.merge(p_long, on=["game_id", "period"], how="left")

    def margin_to_prob(margin: float, scale: float = moneyline_margin_scale) -> float:
        try: return 1.0 / (1.0 + np.exp(-(float(margin) / float(scale))))
        except Exception: return np.nan
    def prob_to_american(p: float) -> float:
        if not (0 < p < 1): return np.nan
        return -100.0 * (p/(1-p)) if p >= 0.5 else 100.0 * ((1-p)/p)
    def american_to_prob(a: float) -> float:
        a = float(a)
        return (-a)/((-a)+100.0) if a < 0 else 100.0/(a+100.0)

    out_rows: list[dict] = []
    for _, r in merged.iterrows():
        gid = r.get("game_id"); period = r.get("period"); book = r.get("book")
        home = r.get("home_team") or r.get("home_team_name"); away = r.get("away_team") or r.get("away_team_name")
        m_total = r.get("model_total"); m_margin = r.get("model_margin")
        # Totals
        line_total = r.get("total")
        if pd.notna(m_total) and pd.notna(line_total):
            edge_t = float(m_total) - float(line_total)
            if edge_t > total_threshold:
                out_rows.append({"game_id": gid,"period": period,"market": "totals","book": book,"pick": "Over","edge": edge_t,
                                 "line_value": line_total,"predicted_value": m_total,"home_team": home,"away_team": away})
            elif edge_t < -total_threshold:
                out_rows.append({"game_id": gid,"period": period,"market": "totals","book": book,"pick": "Under","edge": abs(edge_t),
                                 "line_value": line_total,"predicted_value": m_total,"home_team": home,"away_team": away})
        # Spreads (home side)
        home_spread = r.get("home_spread")
        if pd.notna(m_margin) and pd.notna(home_spread):
            try:
                implied = -float(home_spread)
                edge_s = float(m_margin) - implied
                if edge_s > spread_threshold:
                    out_rows.append({"game_id": gid,"period": period,"market": "spreads","book": book,
                                     "pick": f"{home} {home_spread:+}","edge": edge_s,"line_value": home_spread,
                                     "predicted_value": m_margin,"home_team": home,"away_team": away})
                elif edge_s < -spread_threshold:
                    out_rows.append({"game_id": gid,"period": period,"market": "spreads","book": book,
                                     "pick": f"{away} {(-home_spread):+}","edge": abs(edge_s),"line_value": home_spread,
                                     "predicted_value": m_margin,"home_team": home,"away_team": away})
            except Exception: pass
        # Moneyline
        ml_home = r.get("moneyline_home"); ml_away = r.get("moneyline_away")
        if pd.notna(m_margin) and pd.notna(ml_home) and pd.notna(ml_away):
            p_home = margin_to_prob(m_margin); p_away = 1.0 - p_home if pd.notna(p_home) else np.nan
            fair_home = prob_to_american(p_home); fair_away = prob_to_american(p_away)
            if pd.notna(fair_home) and pd.notna(ml_home):
                pb = american_to_prob(ml_home); pf = american_to_prob(fair_home)
                edge_pct = (pf - pb)/pf * 100.0 if pf > 0 and pb > 0 else np.nan
                if pd.notna(edge_pct) and edge_pct >= moneyline_edge_pct:
                    out_rows.append({"game_id": gid,"period": period,"market": "moneyline","book": book,"pick": f"{home} ML",
                                     "edge": edge_pct,"line_value": ml_home,"predicted_value": m_margin,"fair_price": fair_home,
                                     "home_team": home,"away_team": away})
            if pd.notna(fair_away) and pd.notna(ml_away):
                pb = american_to_prob(ml_away); pf = american_to_prob(fair_away)
                edge_pct = (pf - pb)/pf * 100.0 if pf > 0 and pb > 0 else np.nan
                if pd.notna(edge_pct) and edge_pct >= moneyline_edge_pct:
                    out_rows.append({"game_id": gid,"period": period,"market": "moneyline","book": book,"pick": f"{away} ML",
                                     "edge": edge_pct,"line_value": ml_away,"predicted_value": -m_margin if pd.notna(m_margin) else m_margin,
                                     "fair_price": fair_away,"home_team": home,"away_team": away})
    if not out_rows:
        print("[yellow]No picks passed thresholds; wrote empty file.[/yellow]")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["game_id"]).to_csv(out, index=False)
        return
    picks_df = pd.DataFrame(out_rows).sort_values(["period","market","edge"], ascending=[True,True,False]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    picks_df.to_csv(out, index=False)
    print(f"[green]Wrote {len(picks_df)} picks (totals/spreads/moneyline) to[/green] {out}")


@app.command(name="backtest-closing")
def backtest_closing(
    games_path: Path = typer.Argument(..., help="Games file from fetch-games (CSV/Parquet)"),
    merged_closing_path: Path = typer.Argument(..., help="Merged games_with_closing.csv (from join-closing)"),
    preds_path: Path = typer.Argument(..., help="Predictions CSV from predict-baseline"),
    out: Path = typer.Option(settings.outputs_dir / "backtest_bets_closing.csv", help="Per-bet results CSV"),
    threshold: float = typer.Option(2.0, help="Bet when |pred_total - closing_total| >= threshold"),
):
    """Backtest totals strategy using closing lines (full-game totals) and predictions.

    Use join-closing first to attach game_id to closing lines.
    """
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))
    closing = pd.read_csv(merged_closing_path)
    preds = pd.read_csv(preds_path)

    bets, summary = backtest_totals_with_closing(games, closing, preds, threshold=threshold)
    out.parent.mkdir(parents=True, exist_ok=True)
    bets.to_csv(out, index=False)
    print(f"[green]Backtest (closing) complete.[/green] Wrote {len(bets)} rows to {out}")
    print({
        "n_games": summary.n_games,
        "n_books_rows": summary.n_books_rows,
        "n_bets": summary.n_bets,
        "n_resolved": summary.n_resolved,
        "win_rate": summary.win_rate,
        "pnl_units": summary.pnl_units,
        "roi": summary.roi,
        "avg_edge": summary.avg_edge,
    })


@app.command(name="edge-persistence")
def edge_persistence(
    games_path: Path = typer.Argument(settings.outputs_dir / "games_all.csv", help="Games master (with final scores if available)"),
    last_path: Path = typer.Argument(settings.outputs_dir / "games_with_last.csv", help="Merged games+last odds (full-game rows incl. total/spread)"),
    closing_path: Path = typer.Argument(settings.outputs_dir / "games_with_closing.csv", help="Merged games+closing odds (full-game rows)"),
    preds_path: Path = typer.Argument(settings.outputs_dir / "predictions_all.csv", help="Predictions file (multi-date) with pred_total,pred_margin"),
    start_date: str = typer.Option("", help="Optional inclusive start date YYYY-MM-DD"),
    end_date: str = typer.Option("", help="Optional inclusive end date YYYY-MM-DD"),
    total_threshold: float = typer.Option(1.5, help="Min abs early total edge to consider a potential bet"),
    spread_threshold: float = typer.Option(1.0, help="Min abs early ATS edge to consider a potential bet"),
    ml_edge_pct: float = typer.Option(2.0, help="Min % ML edge (model vs implied) to consider bet"),
    out: Path = typer.Option(settings.outputs_dir / "edge_persistence_summary.json", help="Output JSON summary"),
):
    """Analyze persistence of early (last) edges vs closing lines.

    Metrics:
      - correlation_total: Pearson corr between early total edge and closing total edge
      - sign_retention_total: fraction of games where sign(early_edge_total)==sign(closing_edge_total) (ignoring zeros)
      - magnitude_decay_total: median(|closing_edge_total|/|early_edge_total|) for qualifying early edges
      - similar stats for ATS (margin/spread) and moneyline (prob edge)
      - bet_retention_total: fraction of early total edges >= threshold that remain >= threshold at close (same direction)
      - realized_roi_total: ROI if staking 1u on each qualifying early total edge (using closing result + actual scores)
        (Only computed if final scores present.)

    Requires merged last & closing files with full-game period rows.
    """
    # Load inputs
    for pth in [games_path, last_path, closing_path, preds_path]:
        if not Path(pth).exists():
            print(f"[red]Missing required file {pth}[/red]"); raise typer.Exit(code=1)
    games = pd.read_csv(games_path)
    last_df = pd.read_csv(last_path)
    closing_df = pd.read_csv(closing_path)
    preds = pd.read_csv(preds_path)
    # Normalize date filters
    def _norm_date_col(df, col):
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                df[col] = df[col].astype(str)
        return df
    for d, c in [(games, "date"), (preds, "date"), (last_df, "date_game"), (closing_df, "date_game")]:
        _norm_date_col(d, c)
    sd = start_date.strip() or None
    ed = end_date.strip() or None
    if sd and ed and sd > ed:
        print("[red]start_date must be <= end_date[/red]"); raise typer.Exit(code=1)
    def _filter(df, col):
        if col in df.columns and (sd or ed):
            if sd:
                df = df[df[col] >= sd]
            if ed:
                df = df[df[col] <= ed]
        return df
    games = _filter(games, "date")
    preds = _filter(preds, "date")
    last_df = _filter(last_df, "date_game")
    closing_df = _filter(closing_df, "date_game")

    # Restrict to full-game totals/spreads rows for edge computations
    def _fg(df):
        if {"market","period"}.issubset(df.columns):
            vals = df["period"].astype(str).str.lower()
            return df[(vals.isin(["full_game","full game","fg"]))]
        return df
    last_fg = _fg(last_df.copy())
    closing_fg = _fg(closing_df.copy())
    # Predictions only keep needed columns and string game_id
    if "game_id" in preds.columns:
        preds["game_id"] = preds["game_id"].astype(str)
    keep_pred = [c for c in ["game_id","pred_total","pred_margin","date"] if c in preds.columns]
    preds = preds[keep_pred].drop_duplicates(subset=["game_id"], keep="last") if keep_pred else preds
    # Attach predictions to last & closing lines separately for totals & spreads
    for dfname, d in [("last", last_fg), ("closing", closing_fg)]:
        if "game_id" in d.columns and "game_id" in preds.columns:
            d["game_id"] = d["game_id"].astype(str)
    last_fg = last_fg.merge(preds, on="game_id", how="left", suffixes=("", "_p"))
    closing_fg = closing_fg.merge(preds, on="game_id", how="left", suffixes=("", "_p"))

    # Compute edges
    def _edge_total(df):
        if {"pred_total","total"}.issubset(df.columns):
            return pd.to_numeric(df["pred_total"], errors="coerce") - pd.to_numeric(df["total"], errors="coerce")
        return pd.Series([np.nan]*len(df))
    def _edge_ats(df):
        if {"pred_margin","home_spread"}.issubset(df.columns):
            market_margin = -pd.to_numeric(df["home_spread"], errors="coerce")
            return pd.to_numeric(df["pred_margin"], errors="coerce") - market_margin
        return pd.Series([np.nan]*len(df))
    # Filter to market type subsets for clarity
    last_totals = last_fg[last_fg.get("market").astype(str).str.lower()=="totals"].copy()
    last_spreads = last_fg[last_fg.get("market").astype(str).str.lower()=="spreads"].copy()
    closing_totals = closing_fg[closing_fg.get("market").astype(str).str.lower()=="totals"].copy()
    closing_spreads = closing_fg[closing_fg.get("market").astype(str).str.lower()=="spreads"].copy()
    last_totals["edge_total_early"] = _edge_total(last_totals)
    closing_totals["edge_total_close"] = _edge_total(closing_totals)
    last_spreads["edge_ats_early"] = _edge_ats(last_spreads)
    closing_spreads["edge_ats_close"] = _edge_ats(closing_spreads)

    # Merge early vs closing edges per game/book (align by game_id; ignore book mismatch for aggregated view)
    def _agg_edges(early: pd.DataFrame, close: pd.DataFrame, early_col: str, close_col: str) -> pd.DataFrame:
        if early.empty or close.empty or "game_id" not in early.columns or "game_id" not in close.columns:
            return pd.DataFrame(columns=["game_id", early_col, close_col])
        e = early.groupby("game_id")[early_col].median().rename(early_col)
        c = close.groupby("game_id")[close_col].median().rename(close_col)
        return e.to_frame().merge(c.to_frame(), on="game_id", how="inner")
    agg_tot = _agg_edges(last_totals, closing_totals, "edge_total_early", "edge_total_close")
    agg_ats = _agg_edges(last_spreads, closing_spreads, "edge_ats_early", "edge_ats_close")

    def _corr(df, a, b):
        """Robust Pearson correlation; return None if degenerate (zero variance)."""
        try:
            s = df[[a,b]].dropna()
            if s.empty:
                return None
            if s[a].nunique() <= 1 or s[b].nunique() <= 1:
                return None
            return float(s[a].corr(s[b]))
        except Exception:
            return None
    def _sign_retention(df, a, b):
        try:
            s = df[[a,b]].dropna()
            if s.empty:
                return None
            sa = np.sign(s[a])
            sb = np.sign(s[b])
            mask = (sa!=0) & (sb!=0)
            if not mask.any():
                return None
            return float((sa[mask]==sb[mask]).mean())
        except Exception:
            return None
    def _mag_decay(df, a, b, thresh):
        try:
            s = df[[a,b]].dropna()
            if s.empty:
                return None
            qa = s[a].abs()
            qb = s[b].abs()
            mask = qa>=thresh
            if not mask.any():
                return None
            ratio = (qb[mask]/qa[mask]).replace([np.inf,-np.inf], np.nan).dropna()
            if ratio.empty:
                return None
            return float(ratio.median())
        except Exception:
            return None
    def _bet_retention(df, a, b, thresh):
        try:
            s = df[[a,b]].dropna()
            if s.empty:
                return None
            qa = s[a].abs()
            dir_a = np.sign(s[a])
            dir_b = np.sign(s[b])
            mask = qa>=thresh
            if not mask.any():
                return None
            kept = (dir_a[mask]==dir_b[mask]) & (s[b].abs()[mask] >= thresh)
            return float(kept.mean())
        except Exception:
            return None

    # Totals metrics
    corr_total = _corr(agg_tot, "edge_total_early", "edge_total_close")
    sign_ret_total = _sign_retention(agg_tot, "edge_total_early", "edge_total_close")
    decay_total = _mag_decay(agg_tot, "edge_total_early", "edge_total_close", total_threshold)
    bet_ret_total = _bet_retention(agg_tot, "edge_total_early", "edge_total_close", total_threshold)
    # ATS metrics
    corr_ats = _corr(agg_ats, "edge_ats_early", "edge_ats_close")
    sign_ret_ats = _sign_retention(agg_ats, "edge_ats_early", "edge_ats_close")
    decay_ats = _mag_decay(agg_ats, "edge_ats_early", "edge_ats_close", spread_threshold)
    bet_ret_ats = _bet_retention(agg_ats, "edge_ats_early", "edge_ats_close", spread_threshold)

    # Realized ROI for totals early edges vs actual outcomes (if scores available in games)
    realized_roi_total = None
    try:
        if {"home_score","away_score"}.issubset(games.columns) and not agg_tot.empty:
            # Need line values; use last_totals median per game
            line_map = last_totals.groupby("game_id")["total"].median().to_dict()
            score_map = (games.set_index("game_id")["home_score"].astype(float) + games.set_index("game_id")["away_score"].astype(float)).to_dict()
            pnl = 0.0; n_bets = 0
            for _, r in agg_tot.iterrows():
                gid = r.get("game_id")
                early_edge = r.get("edge_total_early")
                if pd.isna(early_edge) or abs(early_edge) < total_threshold:
                    continue
                line = line_map.get(gid); actual = score_map.get(gid)
                if line is None or actual is None or actual <= 0:
                    continue
                n_bets += 1
                # Assume -110 price, stake 1u
                bet_over = early_edge > 0
                if actual == line:
                    continue  # push
                won = (bet_over and actual > line) or ((not bet_over) and actual < line)
                win_units = 1.0 if won else -110.0/100.0  # simplified payout: risk 1 to win (price/100) if positive odds else 1
                # For negative odds: risk |price|/100 to win 1 ; we standardize to unit returns: one could refine
                pnl += win_units
            realized_roi_total = (pnl / n_bets) if n_bets>0 else None
    except Exception:
        realized_roi_total = None

    summary = {
        "n_games_total": int(len(agg_tot)),
        "n_games_ats": int(len(agg_ats)),
        "correlation_total": corr_total,
        "sign_retention_total": sign_ret_total,
        "median_magnitude_ratio_total": decay_total,
        "bet_retention_total": bet_ret_total,
        "correlation_ats": corr_ats,
        "sign_retention_ats": sign_ret_ats,
        "median_magnitude_ratio_ats": decay_ats,
        "bet_retention_ats": bet_ret_ats,
        "realized_roi_total": realized_roi_total,
        "thresholds": {"total": total_threshold, "spread": spread_threshold, "moneyline_pct": ml_edge_pct},
        "date_range": {"start": sd, "end": ed},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("[green]Edge persistence summary written to[/green]", out)
    print(summary)


@app.command(name="backtest-closing-range")
def backtest_closing_range(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    games_path: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Games file (CSV/Parquet) with final scores and dates"),
    merged_closing_path: Path = typer.Option(settings.outputs_dir / "games_with_closing.csv", help="Merged games+closing (from join-closing)"),
    merged_last_path: Path = typer.Option(settings.outputs_dir / "games_with_last.csv", help="Merged games+last pre-tip odds (from join-last) for CLV proxy"),
    preds_path: Path = typer.Option(settings.outputs_dir / "predictions_all.csv", help="Predictions (multi-date) with game_id, date, pred_total"),
    threshold: float = typer.Option(2.0, help="Bet when |pred_total - closing_total| >= threshold"),
    out_dir: Path = typer.Option(settings.outputs_dir / "eval_closing", help="Directory to write summaries (CSV/JSON) and optional per-day bets"),
    write_daily_bets: bool = typer.Option(False, help="If true, write per-day bets_<date>.csv files"),
    include_spread: bool = typer.Option(True, help="Attempt to compute spread metrics if functions/columns are available"),
    include_moneyline: bool = typer.Option(True, help="Attempt to compute moneyline metrics if functions/columns are available"),
    compute_clv: bool = typer.Option(True, help="Compute CLV proxy for totals using last pre-tip lines vs closing"),
):
    """Aggregate backtest vs closing lines across a date range (totals only for v1).

    Produces:
      - per-date summary CSV (date,n_games,n_books_rows,n_bets,n_resolved,win_rate,pnl_units,roi,avg_edge)
      - overall summary JSON with macro averages
      - optional per-day bets CSVs when write_daily_bets=true
    """
    try:
        d0 = dt.date.fromisoformat(start)
        d1 = dt.date.fromisoformat(end)
    except Exception:
        print("[red]Invalid start/end; use YYYY-MM-DD[/red]"); raise typer.Exit(code=1)
    if d1 < d0:
        print("[red]end must be >= start[/red]"); raise typer.Exit(code=1)
    # Load games
    if games_path.suffix.lower() == ".csv":
        games_all = pd.read_csv(games_path)
    else:
        try:
            games_all = pd.read_parquet(games_path)
        except Exception:
            games_all = pd.read_csv(games_path.with_suffix(".csv"))
    clos = pd.read_csv(merged_closing_path)
    preds_all = pd.read_csv(preds_path)
    last_all: pd.DataFrame | None = None
    try:
        if merged_last_path and merged_last_path.exists():
            last_all = pd.read_csv(merged_last_path)
    except Exception:
        last_all = None
    # Normalize ids/dates
    for df, dcol in [(games_all, "date"), (clos, "date_game"), (preds_all, "date")]:
        if dcol in df.columns:
            try:
                df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                df[dcol] = df[dcol].astype(str)
    if "game_id" in games_all.columns:
        games_all["game_id"] = games_all["game_id"].astype(str)
    if "game_id" in clos.columns:
        clos["game_id"] = clos["game_id"].astype(str)
    if "game_id" in preds_all.columns:
        preds_all["game_id"] = preds_all["game_id"].astype(str)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_day_rows = []
    cur = d0
    while cur <= d1:
        day = cur.isoformat()
        gday = games_all[games_all.get("date").astype(str) == day].copy() if "date" in games_all.columns else games_all.copy()
        cday = clos[clos.get("date_game").astype(str) == day].copy() if "date_game" in clos.columns else clos.copy()
        pday = preds_all[preds_all.get("date").astype(str) == day].copy() if "date" in preds_all.columns else preds_all.copy()
        if gday.empty or cday.empty or pday.empty:
            per_day_rows.append({
                "date": day, "n_games": len(gday), "n_books_rows": len(cday), "n_bets": 0,
                "n_resolved": 0, "win_rate": None, "pnl_units": 0.0, "roi": None, "avg_edge": None
            })
            cur += dt.timedelta(days=1)
            continue
        # Ensure game_id types
        if "game_id" in gday.columns:
            gday["game_id"] = gday["game_id"].astype(str)
        if "game_id" in cday.columns:
            cday["game_id"] = cday["game_id"].astype(str)
        if "game_id" in pday.columns:
            pday["game_id"] = pday["game_id"].astype(str)
        # Run backtest (totals + closing)
        bets, summary = backtest_totals_with_closing(gday, cday, pday, threshold=threshold)
        if write_daily_bets and not bets.empty:
            bets.to_csv(out_dir / f"bets_{day}.csv", index=False)
        # CLV proxy using last pre-tip lines
        clv_metrics = {"clv_n": None, "clv_mean": None, "clv_median": None, "clv_pos_rate": None}
        if compute_clv and last_all is not None and not bets.empty:
            # Prepare last lines for totals at full_game level, by game
            lday = last_all[last_all.get("date_game").astype(str) == day].copy() if "date_game" in last_all.columns else last_all.copy()
            if not lday.empty:
                # If schema is closing-like (market/period), filter totals full_game
                if {"market","period"}.issubset(lday.columns):
                    ltot = lday[(lday["market"]=="totals") & (lday["period"]=="full_game")].copy()
                    # Choose representative last total per game (median across books)
                    ltot = ltot.groupby("game_id", as_index=False)["total"].median().rename(columns={"total":"last_total"}) if "total" in ltot.columns else pd.DataFrame()
                else:
                    # Generic: expect a 'total' column directly
                    ltot = lday.groupby("game_id", as_index=False)["total"].median().rename(columns={"total":"last_total"}) if "total" in lday.columns else pd.DataFrame()
                # Closing totals representative per game
                ctot = cday.copy()
                if {"market","period"}.issubset(ctot.columns):
                    ctot = ctot[(ctot["market"]=="totals") & (ctot["period"]=="full_game")]
                ctot = ctot.groupby("game_id", as_index=False)["total"].median().rename(columns={"total":"closing_total"}) if "total" in ctot.columns else pd.DataFrame()
                if not ltot.empty and not ctot.empty:
                    b = bets.merge(ltot, on="game_id", how="left").merge(ctot, on="game_id", how="left")
                    if not b.empty:
                        # Compute CLV: Over -> (closing - last), Under -> (last - closing)
                        def _clv(row):
                            if pd.isna(row.get("last_total")) or pd.isna(row.get("closing_total")):
                                return np.nan
                            if row["bet"] == "over":
                                return row["closing_total"] - row["last_total"]
                            else:
                                return row["last_total"] - row["closing_total"]
                        b["clv"] = b.apply(_clv, axis=1)
                        vc = b["clv"].dropna()
                        if not vc.empty:
                            clv_metrics = {
                                "clv_n": int(vc.shape[0]),
                                "clv_mean": float(vc.mean()),
                                "clv_median": float(vc.median()),
                                "clv_pos_rate": float((vc > 0).mean()),
                            }
        # Optional: spreads and moneyline
        spread_metrics = {"spread_n_bets": None, "spread_win_rate": None, "spread_roi": None}
        ml_metrics = {"ml_n_bets": None, "ml_win_rate": None, "ml_roi": None}
        if include_spread:
            try:
                # Lazy import to avoid hard dependency if not present
                from .eval.backtest import backtest_spread_with_closing  # type: ignore
                sbets, ssum = backtest_spread_with_closing(gday, cday, pday)
                if write_daily_bets and sbets is not None and not sbets.empty:
                    sbets.to_csv(out_dir / f"bets_spread_{day}.csv", index=False)
                if ssum is not None:
                    spread_metrics = {
                        "spread_n_bets": getattr(ssum, "n_bets", None),
                        "spread_win_rate": getattr(ssum, "win_rate", None),
                        "spread_roi": getattr(ssum, "roi", None),
                    }
            except Exception:
                pass
        if include_moneyline:
            try:
                from .eval.backtest import backtest_moneyline_with_closing  # type: ignore
                mlbets, mlsum = backtest_moneyline_with_closing(gday, cday, pday)
                if write_daily_bets and mlbets is not None and not mlbets.empty:
                    mlbets.to_csv(out_dir / f"bets_moneyline_{day}.csv", index=False)
                if mlsum is not None:
                    ml_metrics = {
                        "ml_n_bets": getattr(mlsum, "n_bets", None),
                        "ml_win_rate": getattr(mlsum, "win_rate", None),
                        "ml_roi": getattr(mlsum, "roi", None),
                    }
            except Exception:
                pass

        per_day_rows.append({
            "date": day,
            "n_games": summary.n_games,
            "n_books_rows": summary.n_books_rows,
            "n_bets": summary.n_bets,
            "n_resolved": summary.n_resolved,
            "win_rate": summary.win_rate,
            "pnl_units": summary.pnl_units,
            "roi": summary.roi,
            "avg_edge": summary.avg_edge,
            **spread_metrics,
            **ml_metrics,
            **clv_metrics,
        })
        cur += dt.timedelta(days=1)

    # Write per-day CSV and overall JSON
    df_sum = pd.DataFrame(per_day_rows)
    df_sum.to_csv(out_dir / "summary_by_date.csv", index=False)
    # Overall aggregates
    valid = df_sum[df_sum["n_bets"] > 0]
    overall = {
        "date_start": d0.isoformat(),
        "date_end": d1.isoformat(),
        "days": len(df_sum),
        "days_with_bets": int((df_sum["n_bets"] > 0).sum()),
        "total_bets": int(valid["n_bets"].sum()) if not valid.empty else 0,
        "total_pnl_units": float(valid["pnl_units"].sum()) if not valid.empty else 0.0,
        "avg_daily_roi": float(valid["roi"].dropna().mean()) if not valid.empty and valid["roi"].notna().any() else None,
        "avg_daily_win_rate": float(valid["win_rate"].dropna().mean()) if not valid.empty and valid["win_rate"].notna().any() else None,
    }
    # Optional aggregates for spreads and ML
    if "spread_n_bets" in df_sum.columns:
        vsp = df_sum[df_sum["spread_n_bets"].fillna(0) > 0]
        overall.update({
            "spread_days_with_bets": int((df_sum["spread_n_bets"].fillna(0) > 0).sum()),
            "spread_total_bets": int(vsp["spread_n_bets"].sum()) if not vsp.empty else 0,
            "spread_avg_daily_win_rate": float(vsp["spread_win_rate"].dropna().mean()) if not vsp.empty and vsp["spread_win_rate"].notna().any() else None,
            "spread_avg_daily_roi": float(vsp["spread_roi"].dropna().mean()) if not vsp.empty and vsp["spread_roi"].notna().any() else None,
        })
    if "ml_n_bets" in df_sum.columns:
        vml = df_sum[df_sum["ml_n_bets"].fillna(0) > 0]
        overall.update({
            "ml_days_with_bets": int((df_sum["ml_n_bets"].fillna(0) > 0).sum()),
            "ml_total_bets": int(vml["ml_n_bets"].sum()) if not vml.empty else 0,
            "ml_avg_daily_win_rate": float(vml["ml_win_rate"].dropna().mean()) if not vml.empty and vml["ml_win_rate"].notna().any() else None,
            "ml_avg_daily_roi": float(vml["ml_roi"].dropna().mean()) if not vml.empty and vml["ml_roi"].notna().any() else None,
        })
    # CLV aggregates
    if "clv_n" in df_sum.columns and df_sum["clv_n"].fillna(0).sum() > 0:
        vclv = df_sum[df_sum["clv_n"].fillna(0) > 0]
        overall.update({
            "clv_days_with_vals": int((df_sum["clv_n"].fillna(0) > 0).sum()),
            "clv_total_bets": int(vclv["clv_n"].sum()),
            "clv_mean_of_means": float(vclv["clv_mean"].dropna().mean()) if vclv["clv_mean"].notna().any() else None,
            "clv_mean_pos_rate": float(vclv["clv_pos_rate"].dropna().mean()) if vclv["clv_pos_rate"].notna().any() else None,
        })
    (out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"[green]Wrote per-day summary to[/green] {out_dir / 'summary_by_date.csv'}")
    print(f"[green]Wrote overall summary to[/green] {out_dir / 'overall_summary.json'}")


@app.command(name="calibrate-prob")
def calibrate_prob(
    preds_path: Path = typer.Option(..., help="Predictions CSV with a probability column and binary label"),
    prob_col: str = typer.Option("ml_prob_model", help="Column with predicted probability (0-1)"),
    label_col: str = typer.Option("home_win", help="Column with outcome label (1 if home win else 0)"),
    out_dir: Path = typer.Option(settings.outputs_dir / "calibration", help="Output directory for metrics and calibrator"),
    bins: int = typer.Option(10, help="Number of bins for reliability curve when isotonic is unavailable"),
):
    """Compute calibration metrics and fit an isotonic calibrator when available.

    Saves:
      - metrics.json with Brier and log-loss
      - reliability_curve.json with per-bin avg prob and empirical rate
      - isotonic.pkl (if scikit-learn available) for later application
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(preds_path)
    if prob_col not in df.columns or label_col not in df.columns:
        print(f"[red]Missing required columns {prob_col} and/or {label_col} in {preds_path}[/red]")
        raise typer.Exit(code=1)
    p = df[prob_col].astype(float).clip(0, 1)
    y = df[label_col].astype(int).clip(0, 1)

    # Metrics
    brier = float(((p - y) ** 2).mean())
    # Log loss with clipping for stability
    eps = 1e-12
    logloss = float(-(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))).mean())
    metrics = {"brier": brier, "logloss": logloss, "n": int(len(df))}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Reliability curve
    cuts = np.linspace(0, 1, bins + 1)
    curve = []
    for i in range(bins):
        lo, hi = cuts[i], cuts[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not mask.any():
            curve.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": 0, "p_avg": None, "rate": None})
        else:
            p_avg = float(p[mask].mean())
            rate = float(y[mask].mean())
            curve.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": int(mask.sum()), "p_avg": p_avg, "rate": rate})
    (out_dir / "reliability_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")

    # Try isotonic calibration
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
        import joblib  # type: ignore
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p.values, y.values)
        joblib.dump(iso, out_dir / "isotonic.pkl")
        print("[green]Saved isotonic calibrator to[/green]", out_dir / "isotonic.pkl")
    except Exception:
        print("[yellow]scikit-learn not available; saved reliability curve instead. Install scikit-learn to enable isotonic calibration.[/yellow]")


@app.command(name="train-segmented")
def train_segmented_cmd(
    features_csv: Path = typer.Option(settings.outputs_dir / "features_all.csv", help="Features CSV with segment columns (home_team/home_conference)"),
    out_dir: Path = typer.Option(settings.outputs_dir / "seg_models", help="Directory to write segmented model JSONL"),
    segment: str = typer.Option("team", help="Segmentation key: team|conference"),
    min_rows: int = typer.Option(25, help="Minimum rows per segment to train a model"),
    alpha: float = typer.Option(1.0, help="Ridge regularization strength"),
):
    """Train per-segment (team or conference) linear models for totals and margin.

    Produces a JSONL file with weights, bias, and standardization stats per segment.
    """
    try:
        res = train_segmented(features_csv, out_dir, segment=segment, min_rows=min_rows, alpha=alpha)
        print("[green]Segmented training complete:[/green]", res)
    except Exception as e:
        print(f"[red]Segmented training failed:[/red] {e}")
        raise typer.Exit(code=1)


def _load_segment_models(models_dir: Path, segment: str) -> dict[str, dict]:
    """Load segmented models JSONL into a dict keyed by segment_key."""
    import json
    path = models_dir / f"segmented_{segment}_models.jsonl"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            k = obj.get("segment_key")
            if k:
                out[str(k)] = obj
        except Exception:
            continue
    return out


@app.command(name="predict-segmented")
def predict_segmented_cmd(
    features_csv: Path = typer.Option(settings.outputs_dir / "features_curr.csv", help="Current features CSV"),
    models_dir: Path = typer.Option(settings.outputs_dir / "seg_models", help="Directory with segmented model JSONL"),
    segment: str = typer.Option("team", help="Segmentation key used for training (team|conference)"),
    out: Path = typer.Option(settings.outputs_dir / "predictions_segmented.csv", help="Output predictions CSV"),
    blend_weight: float = typer.Option(0.5, help="Weight for segmented model vs baseline pred_total/margin if baseline columns present"),
):
    """Generate predictions using segmented models (optionally blending with baseline predictions)."""
    df = pd.read_csv(features_csv)
    models = _load_segment_models(models_dir, segment)
    if not models:
        print("[yellow]No segmented models found; aborting segmented prediction.[/yellow]")
        raise typer.Exit(code=1)
    key_home = f"home_{segment}" if f"home_{segment}" in df.columns else ("home_team" if segment=="team" else None)
    key_away = f"away_{segment}" if f"away_{segment}" in df.columns else ("away_team" if segment=="team" else None)
    if not key_home or not key_away:
        print("[red]Missing segment columns for prediction.[/red]")
        raise typer.Exit(code=1)
    rows = []
    for _, r in df.iterrows():
        hk = str(r.get(key_home, ""))
        ak = str(r.get(key_away, ""))
        mt = models.get(hk)
        ma = models.get(ak)
        # Fallback to home segment only if away missing
        seg_models = [m for m in [mt, ma] if m]
        if not seg_models:
            continue
        # Prepare feature vector using first model's feature_columns intersection
        cols = seg_models[0].get("feature_columns", [])
        x = []
        for c in cols:
            x.append(float(r.get(c, 0.0)))
        x_arr = np.array(x, dtype=np.float32)
        # Average predictions across available segment models
        preds_tot = []
        preds_mar = []
        for m in seg_models:
            # Totals
            mu_t = np.array(m["mu_total"], dtype=np.float32)
            sg_t = np.array(m["sigma_total"], dtype=np.float32)
            w_t = np.array(m["weights_total"], dtype=np.float32)
            b_t = float(m["bias_total"])
            y_t = ((x_arr - mu_t) / sg_t) @ w_t + b_t
            preds_tot.append(float(y_t))
            # Margin
            mu_m = np.array(m["mu_margin"], dtype=np.float32)
            sg_m = np.array(m["sigma_margin"], dtype=np.float32)
            w_m = np.array(m["weights_margin"], dtype=np.float32)
            b_m = float(m["bias_margin"])
            y_m = ((x_arr - mu_m) / sg_m) @ w_m + b_m
            preds_mar.append(float(y_m))
        pred_total_seg = float(np.mean(preds_tot)) if preds_tot else None
        pred_margin_seg = float(np.mean(preds_mar)) if preds_mar else None
        # Blend with baseline predictions if present in row
        base_total = r.get("pred_total") if "pred_total" in r else None
        base_margin = r.get("pred_margin") if "pred_margin" in r else None
        if base_total is not None and pred_total_seg is not None:
            pred_total = (1 - blend_weight) * float(base_total) + blend_weight * pred_total_seg
        else:
            pred_total = pred_total_seg or base_total
        if base_margin is not None and pred_margin_seg is not None:
            pred_margin = (1 - blend_weight) * float(base_margin) + blend_weight * pred_margin_seg
        else:
            pred_margin = pred_margin_seg or base_margin
        rows.append({
            "game_id": r.get("game_id"),
            "date": r.get("date"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "pred_total": pred_total,
            "pred_margin": pred_margin,
            "segmented_total": pred_total_seg,
            "segmented_margin": pred_margin_seg,
            "blend_weight": blend_weight,
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[green]Wrote {len(rows)} segmented predictions to[/green] {out}")


@app.command(name="validate-closing-coverage")
def validate_closing_coverage(
    games_path: Path = typer.Argument(settings.outputs_dir / "games_all.csv", help="Season games CSV/Parquet (final scores or placeholders)"),
    merged_closing_path: Path = typer.Argument(settings.outputs_dir / "games_with_closing.csv", help="Merged games+closing from join-closing"),
    out: Path = typer.Option(settings.outputs_dir / "closing_coverage.csv", help="Output CSV with per-date coverage stats"),
    min_quality: int = typer.Option(0, help="Only consider games rows with _quality >= this (0 keeps all)"),
    verbose: bool = typer.Option(False, help="Print missing game_ids per date"),
    start_date: str | None = typer.Option(None, help="Optional inclusive start date (YYYY-MM-DD) for coverage window"),
    end_date: str | None = typer.Option(None, help="Optional inclusive end date (YYYY-MM-DD) for coverage window; defaults to yesterday"),
    exclude_future: bool = typer.Option(True, help="Exclude dates > yesterday even if present in games file"),
):
    """Report per-date closing line coverage (full-game totals) vs scheduled games.

    For each date present in games_path, compute:
      - n_games: number of games scheduled (rows in games_path for date)
      - n_with_closing: games having at least one closing totals row (market=totals, period=full_game/fg) in merged_closing_path
      - coverage_pct: n_with_closing / n_games
      - missing_game_ids: semicolon-separated game_ids without closing totals (optional)

    Notes:
      - Assumes merged_closing_path already joined game_id (via join-closing).
      - Games with quality < min_quality are excluded from denominator.
    """
    # Load games flexibly
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))
    if "date" not in games.columns:
        print("[red]games file missing 'date' column[/red]")
        raise typer.Exit(code=1)
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "game_id" in games.columns:
        games["game_id"] = games["game_id"].astype(str)
    # Optional quality filter
    if min_quality > 0 and "_quality" in games.columns:
        games = games[games["_quality"].fillna(0) >= int(min_quality)]
    # Load closing merged
    closing = pd.read_csv(merged_closing_path)
    if closing.empty:
        print(f"[yellow]No rows in {merged_closing_path}; all coverage 0.[/yellow]")
    if "game_id" in closing.columns:
        closing["game_id"] = closing["game_id"].astype(str)
    # Filter to totals + full_game
    for col in ["market", "period"]:
        if col not in closing.columns:
            print(f"[red]Closing file missing column '{col}' required for filtering.[/red]")
            raise typer.Exit(code=1)
    mask_tot = closing["market"].astype(str).str.lower() == "totals"
    mask_fg = closing["period"].astype(str).str.lower().isin(["full_game", "fg"])
    closing_fg = closing[mask_tot & mask_fg]
    # Determine coverage
    out_rows = []
    # Date window filtering
    # Convert to datetime for filtering precision
    g_dates = pd.to_datetime(games["date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    yesterday = today - pd.Timedelta(days=1)
    # Apply user-specified start/end
    if start_date:
        try:
            sd = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(sd):
                mask = g_dates >= sd
                games = games[mask]
                g_dates = g_dates[mask]
        except Exception:
            print(f"[yellow]Warning: invalid start_date '{start_date}' ignored.[/yellow]")
    if end_date:
        try:
            ed = pd.to_datetime(end_date, errors="coerce")
            if pd.notna(ed):
                mask = g_dates <= ed
                games = games[mask]
                g_dates = g_dates[mask]
        except Exception:
            print(f"[yellow]Warning: invalid end_date '{end_date}' ignored; will default to yesterday if exclude_future.[/yellow]")
    elif exclude_future:
        # Default end_date to yesterday when not provided
        mask = g_dates <= yesterday
        games = games[mask]
        g_dates = g_dates[mask]
    # Recompute unique ordered dates after filtering
    dates = sorted(d.strftime("%Y-%m-%d") for d in pd.to_datetime(games["date"], errors="coerce").dropna().unique())
    for d in dates:
        gday = games[games["date"] == d]
        gids = set(gday["game_id"].astype(str).tolist())
        have = set(closing_fg[closing_fg["date_game"].astype(str) == d]["game_id"].astype(str).tolist()) if "date_game" in closing_fg.columns else set(closing_fg[closing_fg["_date"].astype(str) == d]["game_id"].astype(str).tolist() if "_date" in closing_fg.columns else [])
        missing = sorted(gids - have)
        row = {
            "date": d,
            "n_games": len(gids),
            "n_with_closing": len(have & gids),
            "coverage_pct": (len(have & gids) / len(gids)) if gids else None,
        }
        if verbose:
            row["missing_game_ids"] = ";".join(missing)
        out_rows.append(row)
        if verbose and missing:
            print(f"[yellow]{d}: missing {len(missing)} closing totals[/yellow]")
    out_df = pd.DataFrame(out_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"[green]Wrote closing coverage for {len(out_df)} dates to[/green] {out}")
    try:
        overall = float(out_df["coverage_pct"].mean()) if "coverage_pct" in out_df.columns else 0.0
        print(f"Overall mean coverage: {overall:.2%}")
    except Exception:
        pass


@app.command(name="validate-last-coverage")
def validate_last_coverage(
    games_path: Path = typer.Argument(settings.outputs_dir / "games_all.csv", help="Season games CSV/Parquet (final scores or placeholders)"),
    merged_last_path: Path = typer.Argument(settings.outputs_dir / "games_with_last.csv", help="Merged games+last odds from join-last-odds"),
    out: Path = typer.Option(settings.outputs_dir / "last_coverage.csv", help="Output CSV with per-date coverage stats (any book)"),
    out_books: Path = typer.Option(settings.outputs_dir / "last_coverage_books.csv", help="Output CSV with per-date, per-book coverage stats"),
    min_quality: int = typer.Option(0, help="Only consider games rows with _quality >= this (0 keeps all)"),
    verbose: bool = typer.Option(False, help="Print missing game_ids per date"),
    start_date: str | None = typer.Option(None, help="Optional inclusive start date (YYYY-MM-DD) for coverage window"),
    end_date: str | None = typer.Option(None, help="Optional inclusive end date (YYYY-MM-DD) for coverage window; defaults to yesterday"),
    exclude_future: bool = typer.Option(True, help="Exclude dates > yesterday even if present in games file"),
):
    """Report per-date strict last-odds coverage (full-game totals) vs scheduled games, plus per-book coverage.

    For each date present in games_path, compute:
      - n_games: number of games scheduled (rows in games_path for date)
      - n_with_last: games having at least one strict last totals row (market=totals, period=full_game/fg) in merged_last_path
      - coverage_pct: n_with_last / n_games
      - missing_game_ids: semicolon-separated game_ids without last totals (optional)

    Also writes per-date, per-book coverage to out_books with columns:
      - date, book, n_games, n_with_last_book, coverage_pct_book
    """
    # Load games flexibly
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))
    if "date" not in games.columns:
        print("[red]games file missing 'date' column[/red]")
        raise typer.Exit(code=1)
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "game_id" in games.columns:
        games["game_id"] = games["game_id"].astype(str)
    # Optional quality filter
    if min_quality > 0 and "_quality" in games.columns:
        games = games[games["_quality"].fillna(0) >= int(min_quality)]

    # Load last merged
    last = pd.read_csv(merged_last_path) if merged_last_path.exists() else pd.DataFrame()
    if last.empty:
        print(f"[yellow]No rows in {merged_last_path}; all coverage 0.[/yellow]")
    if "game_id" in last.columns:
        last["game_id"] = last["game_id"].astype(str)
    # Filter to totals + full_game
    for col in ["market", "period"]:
        if col not in last.columns:
            if last.empty:
                break
            print(f"[red]Last-odds file missing column '{col}' required for filtering.[/red]")
            raise typer.Exit(code=1)
    if not last.empty:
        mask_tot = last["market"].astype(str).str.lower() == "totals"
        mask_fg = last["period"].astype(str).str.lower().isin(["full_game", "fg"])
        last_fg = last[mask_tot & mask_fg].copy()
    else:
        last_fg = last

    # Date window filtering for games
    g_dates = pd.to_datetime(games["date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    yesterday = today - pd.Timedelta(days=1)
    if start_date:
        try:
            sd = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(sd):
                mask = g_dates >= sd
                games = games[mask]
                g_dates = g_dates[mask]
        except Exception:
            print(f"[yellow]Warning: invalid start_date '{start_date}' ignored.[/yellow]")
    if end_date:
        try:
            ed = pd.to_datetime(end_date, errors="coerce")
            if pd.notna(ed):
                mask = g_dates <= ed
                games = games[mask]
                g_dates = g_dates[mask]
        except Exception:
            print(f"[yellow]Warning: invalid end_date '{end_date}' ignored; will default to yesterday if exclude_future.[/yellow]")
    elif exclude_future:
        mask = g_dates <= yesterday
        games = games[mask]
        g_dates = g_dates[mask]

    dates = sorted(d.strftime("%Y-%m-%d") for d in pd.to_datetime(games["date"], errors="coerce").dropna().unique())
    out_rows = []
    out_rows_books = []
    for d in dates:
        gday = games[games["date"] == d]
        gids = set(gday["game_id"].astype(str).tolist())
        # Determine which last rows map to this date; prefer 'date_game' from join, else compare by commence date if available
        if not last_fg.empty:
            if "date_game" in last_fg.columns:
                lf = last_fg[last_fg["date_game"].astype(str) == d]
            elif "date_line" in last_fg.columns:
                lf = last_fg[last_fg["date_line"].astype(str) == d]
            else:
                # fallback: attempt parse commence_time to date
                tmp = last_fg.copy()
                if "commence_time" in tmp.columns:
                    tmp["_date"] = pd.to_datetime(tmp["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")
                    lf = tmp[tmp["_date"].astype(str) == d]
                else:
                    lf = last_fg
        else:
            lf = last_fg

        have = set(lf["game_id"].astype(str).tolist()) if not lf.empty else set()
        missing = sorted(gids - have)
        row = {
            "date": d,
            "n_games": len(gids),
            "n_with_last": len(have & gids),
            "coverage_pct": (len(have & gids) / len(gids)) if gids else None,
            "n_books_rows": int(len(lf)) if not lf.empty else 0,
        }
        if verbose:
            row["missing_game_ids"] = ";".join(missing)
        out_rows.append(row)

        # Per-book coverage
        if not lf.empty and "book" in lf.columns:
            for book, bdf in lf.groupby("book"):
                have_b = set(bdf["game_id"].astype(str).tolist())
                out_rows_books.append({
                    "date": d,
                    "book": book,
                    "n_games": len(gids),
                    "n_with_last_book": len(have_b & gids),
                    "coverage_pct_book": (len(have_b & gids) / len(gids)) if gids else None,
                })

        if verbose and missing:
            print(f"[yellow]{d}: missing {len(missing)} last totals[/yellow]")

    out_df = pd.DataFrame(out_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"[green]Wrote last-odds coverage for {len(out_df)} dates to[/green] {out}")
    if out_books:
        bdf = pd.DataFrame(out_rows_books)
        out_books.parent.mkdir(parents=True, exist_ok=True)
        bdf.to_csv(out_books, index=False)
        print(f"[green]Wrote per-book last-odds coverage to[/green] {out_books}")
    try:
        overall = float(out_df["coverage_pct"].mean()) if "coverage_pct" in out_df.columns else 0.0
        print(f"Overall mean last-odds coverage: {overall:.2%}")
    except Exception:
        pass

@app.command(name="make-picks")
def make_picks(
    games_path: Path = typer.Argument(..., help="Games file from fetch-games (CSV/Parquet)"),
    odds_path: Path = typer.Argument(..., help="Odds CSV from fetch-odds or join-odds output"),
    preds_path: Path = typer.Argument(..., help="Predictions CSV from predict-baseline"),
    out: Path = typer.Option(settings.outputs_dir / "picks.csv", help="Output pick sheet CSV"),
    threshold: float = typer.Option(2.0, help="Bet when |pred_total - line_total| >= threshold"),
    default_price: float = typer.Option(-110.0, help="Fallback American price if missing"),
    team_map: Path | None = typer.Option(settings.team_map_path, help="Optional team map CSV for joining odds (default: data/team_map.csv)"),
    book_whitelist: str | None = typer.Option(None, help="Comma-separated list of bookmaker names to allow (filters others)"),
    target_picks: int | None = typer.Option(None, help="If provided, limit output to top-N picks by absolute edge after per-game selection"),
):
    """Generate a clean pick sheet for totals using current odds and model predictions.

    Produces one pick per game (best edge across books), with columns:
      game_id, date, home_team, away_team, book, bet, line, price, pred_total, edge
    """
    # Load games
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))

    # Load odds; if this is a joined file (already merged), use as-is; else join first
    odds = pd.read_csv(odds_path)
    preds = pd.read_csv(preds_path)

    if {"game_key", "odds_key"}.issubset(odds.columns):
        merged = odds.copy()
    else:
        # Optional mapping for robust joins
        mapping = None
        if team_map is not None and team_map.exists():
            try:
                mdf = pd.read_csv(team_map)
                mapping = {}
                def add_map(col_raw: str, col_canon: str):
                    if col_raw in mdf.columns and col_canon in mdf.columns:
                        for raw, canon in zip(mdf[col_raw].astype(str), mdf[col_canon].astype(str)):
                            if raw and canon:
                                mapping[normalize_name(raw)] = canon
                if {'raw', 'canonical'}.issubset(mdf.columns):
                    add_map('raw', 'canonical')
                else:
                    if 'espn' in mdf.columns:
                        add_map('espn', 'canonical')
                    if 'oddsapi' in mdf.columns:
                        add_map('oddsapi', 'canonical')
            except Exception as e:
                print(f"[yellow]Failed to load team map, proceeding without:[/yellow] {e}")
        from .data.merge_odds import join_odds_to_games as _join
        merged = _join(games, odds, team_map=mapping)

    # Use backtest_totals computation to assign bet/edge/price
    bets, _ = backtest_totals(games, merged, preds, threshold=threshold, default_price=default_price)
    # Optional book whitelist
    if book_whitelist:
        allow = {b.strip() for b in book_whitelist.split(',') if b.strip()}
        if 'book' in bets.columns:
            bets = bets[bets['book'].isin(allow)]
    if bets.empty:
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["game_id","date","home_team","away_team","book","bet","line","price","pred_total","edge"]).to_csv(out, index=False)
        print(f"[yellow]No picks meeting threshold {threshold}. Wrote empty sheet to[/yellow] {out}")
        return

    # Attach basic game columns for readability
    picks = bets.merge(games[["game_id", "date", "home_team", "away_team"]], on="game_id", how="left")
    # Choose one pick per game_id: maximize absolute edge, tie-break on better price for the chosen side
    # Sort and drop duplicates keeping the best per game
    picks["abs_edge"] = picks["edge"].abs()
    picks = picks.sort_values(["game_id", "abs_edge", "price"], ascending=[True, False, False])
    picks = picks.drop_duplicates(subset=["game_id"], keep="first")

    # If target_picks specified, take top-N by abs_edge
    if target_picks is not None and target_picks > 0:
        picks = picks.sort_values(["abs_edge", "price"], ascending=[False, False]).head(int(target_picks))

    cols = ["game_id", "date", "home_team", "away_team", "book", "bet", "line", "price", "pred", "edge"]
    picks = picks[cols].rename(columns={"pred": "pred_total"})

    out.parent.mkdir(parents=True, exist_ok=True)
    picks.to_csv(out, index=False)
    print(f"[green]Wrote {len(picks)} picks to[/green] {out}")


@app.command(name="init-db")
def init_db(
    db: Path = typer.Option(settings.data_dir / "ncaab.sqlite", help="SQLite database path (default: data/ncaab.sqlite)"),
):
    """Initialize a SQLite database (creates file and enables WAL)."""
    conn = sqlite_connect(db)
    conn.close()
    print(f"[green]Initialized database at[/green] {db}")


@app.command(name="ingest-csv")
def ingest_csv_cmd(
    table: str = typer.Argument(..., help="Destination table name"),
    csv_path: Path = typer.Argument(..., help="Source CSV path"),
    keys: str = typer.Option("", help="Comma-separated key columns for upsert (composite primary key)"),
    db: Path = typer.Option(settings.data_dir / "ncaab.sqlite", help="SQLite DB path"),
):
    """Ingest an arbitrary CSV into SQLite with optional upsert keys."""
    conn = sqlite_connect(db)
    key_cols = [k.strip() for k in keys.split(",") if k.strip()]
    n = sqlite_ingest(conn, table, csv_path, key_cols)
    conn.close()
    print(f"[green]Ingested {n} rows into {table}[/green]")


@app.command(name="ingest-outputs")
def ingest_outputs(
    db: Path = typer.Option(settings.data_dir / "ncaab.sqlite", help="SQLite DB path"),
    rebuild: bool = typer.Option(False, help="Drop existing target tables before ingest (schema reset)"),
):
    """Ingest common outputs CSVs into SQLite with reasonable keys.

    Tables and keys:
      - games (outputs/games_all.csv) key: game_id
      - boxscores (outputs/boxscores.csv) key: game_id
      - odds_current (outputs/odds_today.csv) key: date,book,home_team_name,away_team_name,market,period
      - closing_lines (outputs/closing_lines.csv) key: event_id,book,market,period
      - features (outputs/features_all.csv) key: game_id
      - predictions (outputs/predictions_all.csv) key: game_id
      - picks (outputs/picks_clean.csv) key: game_id,book
    """
    conn = sqlite_connect(db)
    # Optional rebuild: drop known tables first
    if rebuild:
        to_drop = ["games","boxscores","odds_current","closing_lines","features","predictions","picks"]
        for t in to_drop:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {t}")
            except Exception:
                pass
    def maybe_ingest(table: str, path: Path, keys: list[str], dynamic: bool = False):
        if not path.exists():
            print(f"[yellow]Missing {path}, skipped {table}[/yellow]")
            return
        try:
            if dynamic:
                # Inspect CSV to choose keys that exist
                df = pd.read_csv(path, nrows=1)
                keys_use = [k for k in keys if k in df.columns]
                if not keys_use:
                    # fallback shortlist
                    fallback = [c for c in ["commence_time", "book", "home_team_name", "away_team_name"] if c in df.columns]
                    keys_use = fallback
                n = sqlite_ingest(conn, table, path, keys_use)
            else:
                n = sqlite_ingest(conn, table, path, keys)
            print(f"[green]{table}[/green]: {n} rows from {path}")
        except Exception as e:
            print(f"[yellow]Skipping {table} due to error:[/yellow] {e}")

    root = settings.outputs_dir
    maybe_ingest("games", root / "games_all.csv", ["game_id"])
    maybe_ingest("boxscores", root / "boxscores.csv", ["game_id"])
    # Odds current: composite keys if columns present; fall back to subset handled by ingest
    maybe_ingest("odds_current", root / "odds_today.csv", [
        "commence_time", "book", "home_team_name", "away_team_name", "market", "period"
    ], dynamic=True)
    maybe_ingest("closing_lines", root / "closing_lines.csv", ["event_id", "book", "market", "period"])
    # Strict last odds (if generated). Same key shape as closing_lines.
    maybe_ingest("last_odds", root / "last_odds.csv", ["event_id", "book", "market", "period"])
    maybe_ingest("features", root / "features_all.csv", ["game_id"])
    maybe_ingest("predictions", root / "predictions_all.csv", ["game_id"])
    maybe_ingest("picks", root / "picks_clean.csv", ["game_id", "book"])
    conn.close()
    print(f"[green]Ingestion complete into[/green] {db}")


@app.command(name="seed-conferences")
def seed_conferences(
    out: Path = typer.Option(settings.data_dir / "conferences.csv", help="Output team->conference CSV"),
    games_paths: str = typer.Option("outputs/games_all.csv,outputs/games_curr.csv", help="Comma-separated games CSVs to scan for team names"),
    min_games: int = typer.Option(1, help="Minimum total games across files to include a team (filters exhibitions/D2)"),
):
    """Seed or update a team->conference mapping CSV by scanning games files for team names.

    The output CSV has columns: team,conference. Existing conference assignments are preserved; new teams are added with blank conference.
    """
    names: set[str] = set()
    counts: dict[str, int] = {}
    for p in [Path(x.strip()) for x in games_paths.split(',') if x.strip()]:
        if not p.exists():
            continue
        try:
            g = pd.read_csv(p)
            for col in ["home_team", "away_team"]:
                if col in g.columns:
                    vals = g[col].astype(str).dropna().tolist()
                    for v in vals:
                        names.add(v)
                        counts[v] = counts.get(v, 0) + 1
        except Exception:
            continue
    # Apply min_games filter if requested
    rows = sorted({n for n in names if n and counts.get(n, 0) >= max(1, int(min_games))})
    if not rows:
        print("[yellow]No team names found in provided games files.[/yellow]")
        return
    # Merge with existing mapping if present
    cur = {}
    if out.exists():
        try:
            m = pd.read_csv(out)
            if {"team", "conference"}.issubset(m.columns):
                cur = {str(r["team"]): str(r["conference"]) for _, r in m.iterrows()}
        except Exception:
            pass
    data = []
    for n in rows:
        data.append({"team": n, "conference": cur.get(n, "")})
    df = pd.DataFrame(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[green]Wrote conference mapping with {len(df)} teams to[/green] {out}")


@app.command(name="import-d1-conferences")
def import_d1_conferences(
    d1_csv: Path = typer.Option(settings.data_dir / "d1_conferences.csv", help="Canonical D1 team->conference mapping (team,conference)"),
    out: Path = typer.Option(settings.data_dir / "conferences.csv", help="Destination conferences.csv to write"),
    use_team_map: bool = typer.Option(True, help="Normalize names using data/team_map.csv when available"),
):
    """Import a canonical D1 mapping file into conferences.csv, optionally normalizing names via team_map.

    The d1_csv should have columns: team,conference[,espn,oddsapi]. Only team and conference are required.
    """
    if not d1_csv.exists():
        print(f"[red]Missing D1 mapping file at {d1_csv}[/red]. Place a CSV with columns team,conference.")
        raise typer.Exit(code=1)
    m = pd.read_csv(d1_csv)
    if not {"team", "conference"}.issubset(m.columns):
        print("[red]d1_conferences.csv must have columns: team,conference[/red]")
        raise typer.Exit(code=1)

    # Normalize via team_map if requested
    if use_team_map and settings.team_map_path and settings.team_map_path.exists():
        try:
            tmap = pd.read_csv(settings.team_map_path)
            canon = {}
            # Prefer explicit canonical if present
            if {"raw", "canonical"}.issubset(tmap.columns):
                for raw, c in zip(tmap["raw"].astype(str), tmap["canonical"].astype(str)):
                    canon[_norm(raw)] = c
            # Also consider provider-specific columns
            for col in ["espn", "oddsapi"]:
                if {col, "canonical"}.issubset(tmap.columns):
                    for raw, c in zip(tmap[col].astype(str), tmap["canonical"].astype(str)):
                        canon[_norm(raw)] = c
            # Map each team to canonical if we have it
            m["team"] = m["team"].astype(str).map(lambda x: canon.get(_norm(x), x))
        except Exception as e:
            print(f"[yellow]Failed to apply team_map normalization:[/yellow] {e}")

    # Deduplicate by team
    m = m.dropna(subset=["team"]).copy()
    m = m.drop_duplicates(subset=["team"]).sort_values("team")
    out.parent.mkdir(parents=True, exist_ok=True)
    m[["team", "conference"]].to_csv(out, index=False)
    print(f"[green]Imported {len(m)} D1 teams into[/green] {out}")


@app.command(name="report-conference-coverage")
def report_conference_coverage(
    games_path: Path = typer.Option(settings.outputs_dir / "games_all.csv", help="Games CSV to compare"),
    conf_csv: Path = typer.Option(settings.data_dir / "conferences.csv", help="Conferences mapping CSV (team,conference)"),
):
    """Report coverage stats: how many teams in games are present in conferences.csv and which are missing."""
    if not games_path.exists() or not conf_csv.exists():
        print("[red]Missing input files. Ensure games and conferences.csv exist.[/red]")
        raise typer.Exit(code=1)
    g = pd.read_csv(games_path)
    teams = set()
    for col in ["home_team", "away_team"]:
        if col in g.columns:
            teams.update(g[col].astype(str).dropna().unique().tolist())
    c = pd.read_csv(conf_csv)
    mapped = set(c["team"].astype(str).dropna().unique().tolist())
    missing = sorted(t for t in teams if t not in mapped)
    print({
        "teams_in_games": len(teams),
        "teams_in_conferences": len(mapped),
        "missing": len(missing),
    })
    if missing:
        miss_df = pd.DataFrame({"team": missing})
        path = conf_csv.parent / "conferences_missing_from_games.csv"
        miss_df.to_csv(path, index=False)
        print(f"[yellow]Wrote missing team list to[/yellow] {path}")


@app.command(name="daily-run")
def daily_run(
    date: str | None = typer.Option(None, help="Target date YYYY-MM-DD (default: today)"),
    season: int = typer.Option(dt.datetime.now().year, help="Season year (bookkeeping)"),
    region: str = typer.Option("us", help="Odds region for TheOddsAPI"),
    provider: str = typer.Option("espn", help="Games provider: espn|ncaa|fused"),
    threshold: float = typer.Option(2.0, help="Pick threshold for totals |pred - line| >= threshold"),
    default_price: float = typer.Option(-110.0, help="Fallback American price if missing"),
    retrain: bool = typer.Option(False, help="Retrain baseline on features_all before scoring current"),
    segment: str = typer.Option("none", help="Prediction model segmentation: none|team|conference"),
    conf_map: Path | None = typer.Option(None, help="Optional team->conference CSV for conference segmentation (columns: team,conference[,season])"),
    use_cache: bool = typer.Option(True, help="Use cached ESPN scoreboard responses when available (use --no-use-cache to force refresh)"),
    preseason_weight: float = typer.Option(0.5, help="Blend weight for preseason model (0=off, 1=preseason-only)"),
    preseason_only_sparse: bool = typer.Option(True, help="Apply preseason blend only when rolling features are sparse"),
    db: Path = typer.Option(settings.data_dir / "ncaab.sqlite", help="SQLite DB to ingest into at end"),
    book_whitelist: str | None = typer.Option(None, help="Comma-separated list of bookmaker names to allow (filters others)"),
    target_picks: int | None = typer.Option(None, help="If provided, limit output to top-N picks by absolute edge after per-game selection"),
    apply_guardrails: bool = typer.Option(True, help="Apply tempo/off/def guardrail blend for implausibly low totals"),
    half_ratio: float = typer.Option(0.485, help="Fallback 1H share when no half models available"),
    auto_train_halves: bool = typer.Option(True, help="Train half models automatically if missing and data available"),
    halves_models_dir: Path = typer.Option(settings.outputs_dir / "models_halves", help="Directory for trained half models"),
    enable_ort: bool = typer.Option(True, help="Attempt ONNX/QNN activation via scripts/enable_ort_qnn.ps1 when ORT not importable"),
    accumulate_schedule: bool = typer.Option(True, help="Append today's games into games_all.csv (dedupe by game_id, prefer rows with scores)"),
    accumulate_predictions: bool = typer.Option(True, help="Append today's predictions into predictions_all.csv (dedupe by game_id, keep latest)"),
    blend_min_rows: int = typer.Option(25, help="Minimum segment training rows before blend weight > 0"),
    blend_max_weight: float = typer.Option(0.6, help="Maximum blend weight applied to segmented prediction"),
):
    """End-to-end daily pipeline: fetch games and odds for a date, build features, predict, make picks, ingest to SQLite."""
    target_date = dt.date.fromisoformat(date) if date else _today_local()

    # 1) Fetch games for the target date -> games_curr.csv (with optional fused provider)
    prov = provider.lower()
    df_games: pd.DataFrame
    if prov == "fused":
        # Use the fusion logic inline (single-day range)
        # Reuse fetch-games-fused implementation principles without writing temp file
        espn_rows: list[dict] = []
        ncaa_rows: list[dict] = []
        for res in iter_games_espn(target_date, target_date, use_cache=use_cache):
            for g in res.games:
                d = g.model_dump(); d["source"] = "espn"; espn_rows.append(d)
        for res in iter_games_ncaa(target_date, target_date, use_cache=use_cache):
            for g in res.games:
                d = g.model_dump(); d["source"] = "ncaa"; ncaa_rows.append(d)
        if not espn_rows and not ncaa_rows:
            print(f"[yellow]No games found for {target_date} via fused providers.[/yellow]")
            df_games = pd.DataFrame(columns=["game_id","date","home_team","away_team"])
        else:
            def _prep(df: pd.DataFrame) -> pd.DataFrame:
                if df.empty: return df
                if "date" in df.columns:
                    try: df["_date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    except Exception: df["_date"] = df["date"].astype(str)
                else: df["_date"] = target_date.isoformat()
                for col in ["home_team","away_team"]:
                    if col in df.columns: df[col] = df[col].astype(str)
                df["_home_key"] = df.get("home_team", pd.Series(dtype=str)).astype(str).map(lambda x: _norm(x))
                df["_away_key"] = df.get("away_team", pd.Series(dtype=str)).astype(str).map(lambda x: _norm(x))
                df["_fuse_key"] = df["_date"] + "|" + df["_home_key"] + "|" + df["_away_key"]
                score_cols = [c for c in ["home_score","away_score","home_score_1h","away_score_1h","home_score_2h","away_score_2h"] if c in df.columns]
                df["_score_nonnull"] = df[score_cols].notna().sum(axis=1) if score_cols else 0
                df["_has_start"] = df.get("start_time").notna() if "start_time" in df.columns else False
                df["_quality"] = df["_has_start"].astype(int) * 10 + df["_score_nonnull"].astype(int)
                return df
            espn_df = _prep(pd.DataFrame(espn_rows))
            ncaa_df = _prep(pd.DataFrame(ncaa_rows))
            combined = pd.concat([espn_df, ncaa_df], ignore_index=True)
            if "_fuse_key" in combined.columns:
                combined = combined.sort_values(["_fuse_key","_quality"], ascending=[True, False]).drop_duplicates(subset=["_fuse_key"], keep="first")
            df_games = combined.drop(columns=[c for c in combined.columns if c.startswith("_")])
            print({
                "fused_rows": len(df_games),
                "espn_only": len(set(espn_df.get("_fuse_key", [])) - set(ncaa_df.get("_fuse_key", []))),
                "ncaa_only": len(set(ncaa_df.get("_fuse_key", [])) - set(espn_df.get("_fuse_key", []))),
            })
    else:
        rows = []
        iterator = iter_games_espn if prov == "espn" else iter_games_ncaa
        for res in iterator(target_date, target_date, use_cache=use_cache):
            for game in res.games:
                rows.append(game.model_dump())
        if rows:
            df_games = pd.DataFrame(rows)
            if "game_id" in df_games.columns: df_games["game_id"] = df_games["game_id"].astype(str)
            if "date" in df_games.columns:
                try: df_games["date"] = pd.to_datetime(df_games["date"]).dt.strftime("%Y-%m-%d")
                except Exception: pass
        else:
            print(f"[yellow]No games found for {target_date}[/yellow]")
            df_games = pd.DataFrame(columns=["game_id","date","home_team","away_team"])
    games_curr_path = settings.outputs_dir / "games_curr.csv"
    df_games.to_csv(games_curr_path, index=False)
    print(f"[green]Wrote {len(df_games)} games to[/green] {games_curr_path}")
    # Also write a dated copy for deterministic history navigation
    try:
        dated_games_path = settings.outputs_dir / f"games_{target_date.isoformat()}.csv"
        df_games.to_csv(dated_games_path, index=False)
        print(f"[green]Wrote dated games to[/green] {dated_games_path}")
    except Exception as _e_gd:
        print(f"[yellow]Skipping dated games write:[/yellow] {_e_gd}")

    # Optional schedule accumulation into games_all.csv so downstream daily-results has full slate
    try:
        if accumulate_schedule:
            all_path = settings.outputs_dir / "games_all.csv"
            if all_path.exists():
                g_all = pd.read_csv(all_path)
                if "game_id" in g_all.columns:
                    g_all["game_id"] = g_all["game_id"].astype(str)
            else:
                g_all = pd.DataFrame(columns=df_games.columns)
            # Ensure game_id string type
            if "game_id" in df_games.columns:
                df_games["game_id"] = df_games["game_id"].astype(str)
            combined = pd.concat([g_all, df_games], ignore_index=True)
            if "game_id" in combined.columns:
                # Prefer rows with non-null score info (finals) over scoreless schedule rows
                score_cols = [c for c in ["home_score","away_score","home_score_1h","away_score_1h","home_score_2h","away_score_2h"] if c in combined.columns]
                if score_cols:
                    combined["_score_nonnull"] = combined[score_cols].notna().sum(axis=1)
                    # Sort so higher _score_nonnull comes first; drop duplicates keeping first
                    combined = combined.sort_values(["game_id","_score_nonnull"], ascending=[True, False]).drop_duplicates(subset=["game_id"], keep="first")
                    combined = combined.drop(columns=["_score_nonnull"], errors="ignore")
            combined.to_csv(all_path, index=False)
            print(f"[green]Accumulated schedule ->[/green] {all_path} ({len(combined)} unique games)")
    except Exception as _acc_e:
        print(f"[yellow]Schedule accumulation skipped:[/yellow] {_acc_e}")

    # 2) Fetch current odds -> odds_today.csv
    adapter = TheOddsAPIAdapter(region=region)
    rows = []
    for o in adapter.iter_odds(season):
        rows.append(o.model_dump())
    if rows:
        df_odds = pd.DataFrame(rows)
    else:
        print("[yellow]No odds returned by TheOddsAPI. Check key/plan/region.[/yellow]")
        df_odds = pd.DataFrame()
    odds_today_path = settings.outputs_dir / "odds_today.csv"
    df_odds.to_csv(odds_today_path, index=False)
    print(f"[green]Wrote {len(df_odds)} odds rows to[/green] {odds_today_path}")
    # Build a per-date odds frame focused on the target date; fallback to odds_history snapshot if empty
    df_odds_day = df_odds.copy()
    try:
        if "commence_time" in df_odds_day.columns:
            dtc = pd.to_datetime(df_odds_day["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")
            df_odds_day = df_odds_day[dtc == target_date.isoformat()].copy()
    except Exception:
        pass
    hist_dir = settings.outputs_dir / "odds_history"
    hist_path = hist_dir / f"odds_{target_date.isoformat()}.csv"
    df_odds_join = df_odds_day
    if (df_odds_join is None) or df_odds_join.empty:
        try:
            if hist_path.exists():
                df_odds_join = pd.read_csv(hist_path)
                print(f"[cyan]Using historical odds snapshot for join:[/cyan] {hist_path}")
        except Exception as _e_hist:
            print(f"[yellow]Historical odds snapshot unavailable:[/yellow] {_e_hist}")
    # Also persist a root-level dated odds file for app fallbacks
    try:
        dated_odds_path = settings.outputs_dir / f"odds_{target_date.isoformat()}.csv"
        to_write = df_odds_day if (df_odds_day is not None and not df_odds_day.empty) else (df_odds_join if df_odds_join is not None else pd.DataFrame())
        if to_write is not None and not to_write.empty:
            to_write.to_csv(dated_odds_path, index=False)
            print(f"[green]Wrote dated odds to[/green] {dated_odds_path} ({len(to_write)} rows)")
        else:
            print(f"[yellow]No per-date odds rows for {target_date}; dated odds CSV skipped.[/yellow]")
    except Exception as _e_od:
        print(f"[yellow]Skipping dated odds write:[/yellow] {_e_od}")

    # 2b) Fetch expanded odds snapshot for derivatives (full + halves) into odds_history/odds_YYYY-MM-DD.csv
    try:
        markets_full = "h2h,spreads,totals,spreads_1st_half,totals_1st_half,spreads_2nd_half,totals_2nd_half"
        hist_rows = []
        for row in adapter.iter_current_odds_expanded(markets=markets_full, date_iso=target_date.isoformat()):
            hist_rows.append(row.model_dump())
        hist_dir = settings.outputs_dir / "odds_history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        hist_path = hist_dir / f"odds_{target_date.isoformat()}.csv"
        if hist_rows:
            pd.DataFrame(hist_rows).to_csv(hist_path, index=False)
            print(f"[green]Wrote {len(hist_rows)} expanded odds rows to[/green] {hist_path}")
        else:
            print("[yellow]No expanded odds rows for derivatives; check plan/markets.[/yellow]")
    except Exception as e:
        print(f"[yellow]Expanded odds snapshot failed:[/yellow] {e}")

    # 3) Join odds to games -> games_with_odds_today.csv (using team map if available)
    mapping = None
    if settings.team_map_path and settings.team_map_path.exists():
        try:
            mdf = pd.read_csv(settings.team_map_path)
            mapping = {}
            def add_map(col_raw: str, col_canon: str):
                if col_raw in mdf.columns and col_canon in mdf.columns:
                    for raw, canon in zip(mdf[col_raw].astype(str), mdf[col_canon].astype(str)):
                        if raw and canon:
                            mapping[normalize_name(raw)] = canon
            if {'raw', 'canonical'}.issubset(mdf.columns):
                add_map('raw', 'canonical')
            else:
                if 'espn' in mdf.columns:
                    add_map('espn', 'canonical')
                if 'oddsapi' in mdf.columns:
                    add_map('oddsapi', 'canonical')
        except Exception as e:
            print(f"[yellow]Failed to load team map, proceeding without:[/yellow] {e}")
    merged_odds_path = settings.outputs_dir / "games_with_odds_today.csv"
    merged = join_odds_to_games(df_games, df_odds, team_map=mapping) if not df_odds.empty else df_games.copy()
    merged.to_csv(merged_odds_path, index=False)
    print(f"[green]Wrote merged games+odds to[/green] {merged_odds_path} ({len(merged)} rows)")
    # Also write a dated merged games+odds using per-date odds or historical snapshot if needed
    try:
        merged_dated_path = settings.outputs_dir / f"games_with_odds_{target_date.isoformat()}.csv"
        if df_odds_join is not None and not df_odds_join.empty:
            merged_d = join_odds_to_games(df_games, df_odds_join, team_map=mapping)
        else:
            merged_d = merged.copy()
        merged_d.to_csv(merged_dated_path, index=False)
        print(f"[green]Wrote dated merged games+odds to[/green] {merged_dated_path} ({len(merged_d)} rows)")
    except Exception as _e_md:
        print(f"[yellow]Skipping dated merged write:[/yellow] {_e_md}")

    # 4) Build features for current games -> features_curr.csv
    feats_curr_path = settings.outputs_dir / "features_curr.csv"
    try:
        # Reuse build-features logic inline (short and long windows)
        feats = build_team_rolling_features(df_games, windows=[5, 15])
        # Schedule
        sched = compute_rest_days(df_games)
        feats = feats.merge(sched, on="game_id", how="left")
        # Boxscore-derived features if available
        bs_path = settings.outputs_dir / "boxscores.csv"
        if bs_path.exists():
            from .features.factors import build_four_factor_rolling_features
            bs = pd.read_csv(bs_path)
            if "game_id" in bs.columns:
                bs["game_id"] = bs["game_id"].astype(str)
            ff = build_four_factor_rolling_features(df_games, bs, window=5)
            feats = feats.merge(ff, on="game_id", how="left")
            from .features.ratings import build_adj_rating_features, build_adj_offdef_tempo_features
            rf = build_adj_rating_features(df_games)
            feats = feats.merge(rf, on="game_id", how="left")
            odt = build_adj_offdef_tempo_features(df_games, bs)
            feats = feats.merge(odt, on="game_id", how="left")

        # Seed opponent-adjusted rating priors from last-two-seasons if current-season history is shallow
        try:
            # Prefer precomputed priors.csv; fallback to features_last2.csv aggregation
            priors_path = settings.outputs_dir / "priors.csv"
            p: pd.DataFrame | None = None
            if priors_path.exists():
                p = pd.read_csv(priors_path)
                if "team" not in p.columns:
                    p = None
            if p is None:
                pri_path = settings.outputs_dir / "features_last2.csv"
                if pri_path.exists() and {"home_team", "away_team"}.issubset(feats.columns):
                    pri = pd.read_csv(pri_path)
                    # Build per-team priors by averaging home/away occurrences
                    def _build_prior(col_home: str, col_away: str, out_name: str) -> pd.DataFrame:
                        parts = []
                        if col_home in pri.columns:
                            a = pri[["home_team", col_home]].rename(columns={"home_team": "team", col_home: out_name})
                            parts.append(a)
                        if col_away in pri.columns:
                            b = pri[["away_team", col_away]].rename(columns={"away_team": "team", col_away: out_name})
                            parts.append(b)
                        if not parts:
                            return pd.DataFrame(columns=["team", out_name])
                        both = pd.concat(parts, ignore_index=True)
                        both["team"] = both["team"].astype(str)
                        return both.groupby("team", as_index=False)[out_name].mean()

                    p_margin = _build_prior("home_rating_margin", "away_rating_margin", "rating_margin")
                    p_off = _build_prior("home_off_rating", "away_off_rating", "off_rating")
                    p_def = _build_prior("home_def_rating", "away_def_rating", "def_rating")
                    p_tmp = _build_prior("home_tempo_rating", "away_tempo_rating", "tempo_rating")
                    # Merge priors into a single table
                    p = (((p_margin.merge(p_off, on="team", how="outer"))
                                 .merge(p_def, on="team", how="outer"))
                                 .merge(p_tmp, on="team", how="outer"))
                    p = p.dropna(how="all", subset=[c for c in ["rating_margin","off_rating","def_rating","tempo_rating"] if c in p.columns])

            # If we have priors table p, map into feats
            if p is not None and not p.empty:
                def _map_prior(series: pd.Series, key: str) -> pd.Series:
                    if key not in p.columns:
                        return pd.Series([pd.NA] * len(series))
                    m = dict(zip(p["team"], p[key]))
                    return series.astype(str).map(m)

                # Prepare seeded columns
                hm = _map_prior(feats["home_team"], "rating_margin") if "home_team" in feats.columns else pd.Series()
                am = _map_prior(feats["away_team"], "rating_margin") if "away_team" in feats.columns else pd.Series()
                ho = _map_prior(feats["home_team"], "off_rating") if "home_team" in feats.columns else pd.Series()
                ao = _map_prior(feats["away_team"], "off_rating") if "away_team" in feats.columns else pd.Series()
                hd = _map_prior(feats["home_team"], "def_rating") if "home_team" in feats.columns else pd.Series()
                ad = _map_prior(feats["away_team"], "def_rating") if "away_team" in feats.columns else pd.Series()
                ht = _map_prior(feats["home_team"], "tempo_rating") if "home_team" in feats.columns else pd.Series()
                at = _map_prior(feats["away_team"], "tempo_rating") if "away_team" in feats.columns else pd.Series()

                # Fill if missing or create if absent
                def _fill_or_create(col: str, values: pd.Series):
                    if values is None or values.empty:
                        return
                    if col in feats.columns:
                        feats[col] = feats[col].where(pd.notna(feats[col]), values)
                    else:
                        feats[col] = values

                _fill_or_create("home_rating_margin", hm)
                _fill_or_create("away_rating_margin", am)
                _fill_or_create("home_off_rating", ho)
                _fill_or_create("away_off_rating", ao)
                _fill_or_create("home_def_rating", hd)
                _fill_or_create("away_def_rating", ad)
                _fill_or_create("home_tempo_rating", ht)
                _fill_or_create("away_tempo_rating", at)

                # Derived diffs/sums
                if {"home_rating_margin","away_rating_margin"}.issubset(feats.columns):
                    feats["rating_margin_diff"] = feats["home_rating_margin"] - feats["away_rating_margin"]
                if {"home_off_rating","away_off_rating"}.issubset(feats.columns):
                    feats["off_rating_diff"] = feats["home_off_rating"] - feats["away_off_rating"]
                if {"home_def_rating","away_def_rating"}.issubset(feats.columns):
                    feats["def_rating_diff"] = feats["home_def_rating"] - feats["away_def_rating"]
                if {"home_tempo_rating","away_tempo_rating"}.issubset(feats.columns):
                    feats["tempo_rating_sum"] = feats["home_tempo_rating"] + feats["away_tempo_rating"]
        except Exception as e:
            print(f"[yellow]Skipping rating priors seeding due to error:[/yellow] {e}")

        feats.to_csv(feats_curr_path, index=False)
        print(f"[green]Wrote features to[/green] {feats_curr_path} with {len(feats)} rows")
    except Exception as e:
        print(f"[red]Failed to build features for current games:[/red] {e}")
        return

    # 5) Optionally retrain
    if retrain:
        try:
            res = train_baseline(settings.outputs_dir / "features_all.csv", settings.outputs_dir / "models")
            print("[green]Retraining complete.[/green]")
            print(res)
        except Exception as e:
            print(f"[yellow]Retrain failed, continuing with existing models:[/yellow] {e}")

    # 6) Predict current -> predictions_week.csv
    # Best-effort ONNX/QNN activation (no-op if already available)
    if enable_ort:
        try:
            if not OnnxPredictor.describe_available():
                script_path = settings.project_root / "scripts" / "enable_ort_qnn.ps1"
                if script_path.exists():
                    print(f"[yellow]Attempting ORT/QNN activation via {script_path}[/yellow]")
                    os.system(f"powershell -ExecutionPolicy Bypass -File \"{script_path.as_posix()}\" -OrtBinDir 'C:/onnxruntime_build/Release' -QnnSdkDir 'C:/Qualcomm/QNN_SDK'")
        except Exception as _eact:
            print(f"[yellow]ORT activation skipped:[/yellow] {_eact}")

    preds_out = settings.outputs_dir / "predictions_week.csv"
    def _append_week(df: pd.DataFrame, path: Path) -> None:
        """Append today's predictions into predictions_week.csv instead of overwriting.

        Keeps multiple dates across the rolling window; dedupes on (game_id,date).
        """
        try:
            prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
        except Exception:
            prev = pd.DataFrame()
        # Normalize types
        if 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str)
        if 'game_id' in prev.columns:
            prev['game_id'] = prev['game_id'].astype(str)
        merged = pd.concat([prev, df], ignore_index=True)
        if {'game_id','date'}.issubset(merged.columns):
            try:
                merged['_dt'] = pd.to_datetime(merged['date'], errors='coerce')
                merged = merged.sort_values(['game_id','_dt']).drop_duplicates(subset=['game_id','date'], keep='last').drop(columns=['_dt'])
            except Exception:
                merged = merged.drop_duplicates(subset=['game_id','date'], keep='last')
        elif 'game_id' in merged.columns:
            merged = merged.drop_duplicates(subset=['game_id'], keep='last')
        merged.to_csv(path, index=False)
        added_rows = len(df)
        date_label = df['date'].iloc[0] if 'date' in df.columns and not df.empty else 'unknown-date'
        print(f"[green]Updated predictions_week.csv[/green] {path} -> total {len(merged)} rows (added {added_rows} for {date_label})")
    try:
        feats = pd.read_csv(feats_curr_path)
        if "game_id" in feats.columns:
            feats["game_id"] = feats["game_id"].astype(str)
        # Safety: ensure one row per game_id to avoid explosive merges downstream
        try:
            if "game_id" in feats.columns:
                dup_count = int(feats.duplicated(subset=["game_id"], keep="last").sum())
                if dup_count > 0:
                    print(f"[yellow]Deduplicating features_curr: dropping {dup_count} duplicate game_id rows[/yellow]")
                    feats = feats.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)
        except Exception as _dedup_err:
            print(f"[yellow]Feature dedup check failed (continuing):[/yellow] {_dedup_err}")

        seg_mode = (segment or "none").lower()
        use_segmented = seg_mode in {"team", "conference"}

        def _apply_preseason_blend(out_df: pd.DataFrame) -> pd.DataFrame:
            """Blend predictions with preseason model predictions based on configured weight and sparsity."""
            w = float(max(0.0, min(1.0, preseason_weight)))
            if w <= 0.0:
                out_df["preseason_applied"] = False
                return out_df
            pre_dir = settings.outputs_dir / "models_preseason"
            cols_path = pre_dir / "feature_columns.txt"
            if not cols_path.exists():
                out_df["preseason_applied"] = False
                return out_df
            try:
                pre_cols = [c.strip() for c in cols_path.read_text(encoding="utf-8").splitlines() if c.strip()]
                Xp = feats.reindex(columns=pre_cols).fillna(0.0).to_numpy(dtype=np.float32)
                providers = OnnxPredictor.describe_available()
                t_path = pre_dir / "baseline_target_total.onnx"
                m_path = pre_dir / "baseline_target_margin.onnx"
                if providers:
                    p_total = OnnxPredictor(str(t_path))
                    p_margin = OnnxPredictor(str(m_path))
                else:
                    p_total = NumpyLinearPredictor(str(t_path))
                    p_margin = NumpyLinearPredictor(str(m_path))
                ypt = p_total.predict(Xp).reshape(-1)
                ypm = p_margin.predict(Xp).reshape(-1)
                pre_df = pd.DataFrame({
                    "game_id": feats["game_id"].astype(str).values,
                    "pre_total": ypt,
                    "pre_margin": ypm,
                })
                # Ensure one row per game_id to prevent cartesian growth on merge
                try:
                    pre_df = pre_df.drop_duplicates(subset=["game_id"], keep="last")
                except Exception:
                    pass
                # Determine sparsity on a set of rolling stats
                roll_cols = [c for c in [
                    "home_pf5","away_pf5","home_tot5","away_tot5",
                    "home_efg5","away_efg5","home_poss5","away_poss5",
                ] if c in feats.columns]
                if roll_cols:
                    nn = feats[roll_cols].notna().sum(axis=1)
                    pre_df["sparse"] = (nn <= 1).astype(bool)
                else:
                    pre_df["sparse"] = True
                # Merge preseason predictions into out_df (align by game_id)
                out_df = out_df.merge(pre_df, on="game_id", how="left")
                def _row_w(r):
                    if preseason_only_sparse and not bool(r.get("sparse", False)):
                        return 0.0
                    return w
                out_df["preseason_weight"] = out_df.apply(_row_w, axis=1)
                mask = out_df["preseason_weight"] > 0
                if "pre_total" in out_df.columns and mask.any():
                    base_pred = pd.to_numeric(out_df.loc[mask, "pred_total"], errors="coerce").astype("float32")
                    pre_pred = pd.to_numeric(out_df.loc[mask, "pre_total"], errors="coerce").astype("float32")
                    wts = out_df.loc[mask, "preseason_weight"].astype("float32")
                    out_df.loc[mask, "pred_total"] = ((1 - wts) * base_pred + wts * pre_pred).astype("float32")
                if "pre_margin" in out_df.columns and mask.any():
                    base_m = pd.to_numeric(out_df.loc[mask, "pred_margin"], errors="coerce").astype("float32")
                    pre_m = pd.to_numeric(out_df.loc[mask, "pre_margin"], errors="coerce").astype("float32")
                    wts = out_df.loc[mask, "preseason_weight"].astype("float32")
                    out_df.loc[mask, "pred_margin"] = ((1 - wts) * base_m + wts * pre_m).astype("float32")
                out_df["preseason_applied"] = out_df["preseason_weight"] > 0
                out_df = out_df.drop(columns=[c for c in ["pre_total","pre_margin","sparse"] if c in out_df.columns])
            except Exception as _pe:
                print(f"[yellow]Preseason blend skipped:[/yellow] {_pe}")
                out_df["preseason_applied"] = False
            return out_df

        # Optionally auto-train half models once per run if missing
        if auto_train_halves and not halves_models_dir.exists():
            try:
                feats_all = settings.outputs_dir / "features_all.csv"
                games_all = settings.outputs_dir / "games_all.csv"
                if feats_all.exists() and games_all.exists():
                    from .train.halves import train_half_models as _train_halves
                    res_halves = _train_halves(feats_all, games_all, halves_models_dir, alpha=1.0)
                    print(f"[green]Auto-trained half models:[/green] {list(res_halves.get('models', {}).keys())}")
                else:
                    print("[yellow]Half auto-train skipped (features_all.csv or games_all.csv missing).[/yellow]")
            except Exception as _he:
                print(f"[yellow]Half auto-train failed:[/yellow] {_he}")

        if use_segmented:
            # Try segmented predict first; fall back to global on failure
            try:
                # Call into the segmented predictor implemented below via a helper
                out_df = _predict_segmented_inline(
                    feats,
                    seg_mode,
                    models_root=settings.outputs_dir / "models",
                    conf_map_path=conf_map if conf_map is not None else settings.data_dir / "conferences.csv",
                )
                # Apply simple tuning bias to totals if available
                try:
                    tpath = settings.outputs_dir / "model_tuning.json"
                    if tpath.exists() and "pred_total" in out_df.columns:
                        t = json.loads(tpath.read_text(encoding="utf-8"))
                        n_valid = int(t.get("n_valid_games", 0))
                        min_valid = int(t.get("min_valid_games", 0))
                        bias = float(t.get("totals_bias", 0.0))
                        if n_valid >= max(1, min_valid) and abs(bias) > 1e-6 and abs(bias) <= float(t.get("cap_abs_bias", 1e9)):
                            out_df["pred_total"] = out_df["pred_total"] - bias
                            out_df["tuning_totals_bias"] = bias
                            out_df["tuning_applied"] = True
                        else:
                            out_df["tuning_applied"] = False
                except Exception as _te:
                    print(f"[yellow]Tuning not applied:[/yellow] {_te}")
                # Preseason blend when configured
                out_df = _apply_preseason_blend(out_df)

                # Guardrails (optional)
                if apply_guardrails and "pred_total" in out_df.columns:
                    try:
                        # Improved guardrail logic: only derive when tempo AND all off/def ratings present.
                        derived_vals: list[float] = []
                        adjusted_flags: list[bool] = []
                        avail_count = 0
                        missing_count = 0
                        for _, r in feats.iterrows():
                            try:
                                have_tempo = pd.notna(r.get("home_tempo_rating")) and pd.notna(r.get("away_tempo_rating"))
                                have_ratings = all(pd.notna(r.get(k)) for k in ["home_off_rating","away_off_rating","home_def_rating","away_def_rating"])
                                if not (have_tempo and have_ratings):
                                    dv = np.nan
                                    missing_count += 1
                                else:
                                    tempo_avg = (float(r.get("home_tempo_rating")) + float(r.get("away_tempo_rating"))) / 2.0
                                    off_home = float(r.get("home_off_rating"))
                                    off_away = float(r.get("away_off_rating"))
                                    def_home = float(r.get("home_def_rating"))
                                    def_away = float(r.get("away_def_rating"))
                                    exp_home_pp100 = np.clip(off_home - def_away, 65, 140)
                                    exp_away_pp100 = np.clip(off_away - def_home, 65, 140)
                                    dv = (exp_home_pp100 + exp_away_pp100) / 100.0 * tempo_avg
                                    dv = float(np.clip(dv, 95, 195))
                                    avail_count += 1
                            except Exception:
                                dv = np.nan
                                missing_count += 1
                            derived_vals.append(dv)
                        out_df["pred_total_raw"] = out_df["pred_total"].astype(float)
                        out_df["derived_total"] = derived_vals
                        blended: list[float] = []
                        for pt, dv in zip(out_df["pred_total_raw"].tolist(), derived_vals):
                            if np.isnan(pt) or np.isnan(dv):
                                blended.append(pt); adjusted_flags.append(False); continue
                            if pt < 103 or (dv > 0 and pt < 0.70 * dv):
                                b = 0.45 * pt + 0.55 * dv
                                b = float(np.clip(b, 95, 195))
                                blended.append(b); adjusted_flags.append(True)
                            else:
                                blended.append(pt); adjusted_flags.append(False)
                        out_df["pred_total"] = blended
                        out_df["pred_total_adjusted"] = adjusted_flags
                        try:
                            out_df["derived_total_available_rows"] = avail_count
                            out_df["derived_total_missing_rows"] = missing_count
                        except Exception:
                            pass
                    except Exception:
                        pass

                # --- Automatic baseline + segmented blending ---
                try:
                    # Baseline (global) predictions for all rows
                    base_cols_path = settings.outputs_dir / "models" / "feature_columns.txt"
                    base_cols = [c.strip() for c in base_cols_path.read_text(encoding="utf-8").splitlines() if c.strip()]
                    Xb = feats.reindex(columns=base_cols).fillna(0.0).to_numpy(dtype=np.float32)
                    providers_b = OnnxPredictor.describe_available()
                    base_total_path = settings.outputs_dir / "models" / "baseline_target_total.onnx"
                    base_margin_path = settings.outputs_dir / "models" / "baseline_target_margin.onnx"
                    base_total_model = OnnxPredictor(str(base_total_path)) if providers_b else NumpyLinearPredictor(str(base_total_path))
                    base_margin_model = OnnxPredictor(str(base_margin_path)) if providers_b else NumpyLinearPredictor(str(base_margin_path))
                    ybt = base_total_model.predict(Xb).reshape(-1)
                    ybm = base_margin_model.predict(Xb).reshape(-1)
                    out_df["pred_total_base"] = ybt
                    out_df["pred_margin_base"] = ybm

                    # Preserve segmented pre-blend
                    out_df["pred_total_seg"] = out_df["pred_total"].astype(float)
                    out_df["pred_margin_seg"] = out_df["pred_margin"].astype(float)

                    # Determine segment sample sizes (n_rows) from features_segment.csv in each model dir
                    seg_root = settings.outputs_dir / "models" / ("seg_team" if seg_mode == "team" else "seg_conference")
                    cache_counts: dict[str, int] = {}
                    def _seg_rows(name: str) -> int:
                        if name in cache_counts:
                            return cache_counts[name]
                        p = seg_root / name / "features_segment.csv"
                        if p.exists():
                            try:
                                cache_counts[name] = max(0, sum(1 for _ in p.open("r", encoding="utf-8")) - 1)
                            except Exception:
                                cache_counts[name] = 0
                        else:
                            cache_counts[name] = 0
                        return cache_counts[name]
                    seg_counts = []
                    for mu in out_df.get("model_used", pd.Series(dtype=str)).astype(str):
                        if mu.startswith("team:") or mu.startswith("conference:"):
                            seg_name = mu.split(":",1)[1]
                            seg_counts.append(_seg_rows(seg_name))
                        else:
                            seg_counts.append(0)
                    out_df["seg_n_rows"] = seg_counts

                    # Compute blend weights
                    def _w(n: float) -> float:
                        try:
                            return float(max(0.0, min(blend_max_weight, (float(n) - float(blend_min_rows)) / (float(blend_min_rows) * 3.0))))
                        except Exception:
                            return 0.0
                    out_df["blend_weight"] = out_df["seg_n_rows"].map(_w)
                    # Force weight=0 when segmented model wasn't used (model_used=='global')
                    mask_global = out_df.get("model_used", pd.Series(dtype=str)).astype(str) == "global"
                    out_df.loc[mask_global, "blend_weight"] = 0.0

                    # Apply blend
                    bt = pd.to_numeric(out_df["pred_total_base"], errors="coerce")
                    bm = pd.to_numeric(out_df["pred_margin_base"], errors="coerce")
                    st = pd.to_numeric(out_df["pred_total_seg"], errors="coerce")
                    sm = pd.to_numeric(out_df["pred_margin_seg"], errors="coerce")
                    wts = out_df["blend_weight"].astype(float)
                    out_df["pred_total_blend"] = bt.where(st.isna(), (1.0 - wts) * bt + wts * st)
                    out_df["pred_margin_blend"] = bm.where(sm.isna(), (1.0 - wts) * bm + wts * sm)
                    # Replace primary columns with blended versions for downstream edge / picks logic
                    out_df["pred_total"] = out_df["pred_total_blend"]
                    out_df["pred_margin"] = out_df["pred_margin_blend"]
                    out_df["blend_applied"] = True
                except Exception as _be:
                    print(f"[yellow]Automatic blend skipped:[/yellow] {_be}")
                    out_df["blend_applied"] = False

                # Half projections via models if available; else ratio fallback
                used_half = False
                try:
                    cols_halves = halves_models_dir / "feature_columns_halves.txt"
                    if halves_models_dir.exists() and cols_halves.exists():
                        ch = [c.strip() for c in cols_halves.read_text(encoding="utf-8").splitlines() if c.strip()]
                        Xh = feats.reindex(columns=ch).fillna(0.0).to_numpy(dtype=np.float32)
                        prov_h = OnnxPredictor.describe_available()
                        def _loadh(name: str):
                            p = halves_models_dir / f"baseline_{name}.onnx"
                            if not p.exists():
                                return None
                            return OnnxPredictor(str(p)) if prov_h else (NumpyLinearPredictor(str(p)) if NumpyLinearPredictor.can_load(str(p)) else None)
                        m_t1 = _loadh("target_total_1h"); m_t2 = _loadh("target_total_2h")
                        m_m1 = _loadh("target_margin_1h"); m_m2 = _loadh("target_margin_2h")
                        if any([m_t1, m_t2, m_m1, m_m2]):
                            if m_t1 is not None: out_df["pred_total_1h"] = m_t1.predict(Xh).reshape(-1)
                            if m_t2 is not None: out_df["pred_total_2h"] = m_t2.predict(Xh).reshape(-1)
                            if m_m1 is not None: out_df["pred_margin_1h"] = m_m1.predict(Xh).reshape(-1)
                            if m_m2 is not None: out_df["pred_margin_2h"] = m_m2.predict(Xh).reshape(-1)
                            used_half = True
                except Exception:
                    used_half = False
                if not used_half:
                    out_df["pred_total_1h"] = pd.to_numeric(out_df["pred_total"], errors="coerce") * half_ratio
                    out_df["pred_total_2h"] = pd.to_numeric(out_df["pred_total"], errors="coerce") * (1.0 - half_ratio)
                    out_df["pred_margin_1h"] = pd.to_numeric(out_df["pred_margin"], errors="coerce") * 0.5
                    out_df["pred_margin_2h"] = pd.to_numeric(out_df["pred_margin"], errors="coerce") * 0.5

                _append_week(out_df, preds_out)
                # Also write dated predictions for the target date
                try:
                    pred_dated = settings.outputs_dir / f"predictions_{target_date.isoformat()}.csv"
                    out_df.to_csv(pred_dated, index=False)
                    print(f"[green]Wrote dated predictions to[/green] {pred_dated}")
                except Exception as _e_pd:
                    print(f"[yellow]Skipping dated predictions write:[/yellow] {_e_pd}")
            except Exception as se:
                print(f"[yellow]Segmented prediction failed, falling back to global:[/yellow] {se}")
                use_segmented = False

        if not use_segmented:
            # Global baseline prediction path
            cols_path = (settings.outputs_dir / "models" / "feature_columns.txt")
            cols = [c.strip() for c in cols_path.read_text(encoding="utf-8").splitlines() if c.strip()]
            X = feats.reindex(columns=cols).fillna(0.0).to_numpy(dtype=np.float32)
            providers = OnnxPredictor.describe_available()
            total_model_path = settings.outputs_dir / "models" / "baseline_target_total.onnx"
            margin_model_path = settings.outputs_dir / "models" / "baseline_target_margin.onnx"
            if providers:
                pred_total = OnnxPredictor(str(total_model_path))
                pred_margin = OnnxPredictor(str(margin_model_path))
            else:
                pred_total = NumpyLinearPredictor(str(total_model_path))
                pred_margin = NumpyLinearPredictor(str(margin_model_path))
            y_total = pred_total.predict(X).reshape(-1)
            y_margin = pred_margin.predict(X).reshape(-1)
            out_df = feats[["game_id", "date", "home_team", "away_team"]].copy()
            out_df["pred_total"] = y_total
            out_df["pred_margin"] = y_margin
            # Early filter: remove placeholder TBD games before guardrails/tuning
            try:
                mask_tbd = (out_df["home_team"].astype(str).str.upper() == "TBD") | (out_df["away_team"].astype(str).str.upper() == "TBD")
                if mask_tbd.any():
                    removed = int(mask_tbd.sum())
                    out_df = out_df.loc[~mask_tbd].reset_index(drop=True)
                    print(f"[yellow]Filtered {removed} TBD placeholder games from global baseline predictions[/yellow]")
            except Exception as _e_tbd2:
                print(f"[yellow]TBD filtering skipped (global baseline path):[/yellow] {_e_tbd2}")
            # Apply simple tuning bias to totals if available
            try:
                tpath = settings.outputs_dir / "model_tuning.json"
                if tpath.exists() and "pred_total" in out_df.columns:
                    t = json.loads(tpath.read_text(encoding="utf-8"))
                    n_valid = int(t.get("n_valid_games", 0))
                    min_valid = int(t.get("min_valid_games", 0))
                    bias = float(t.get("totals_bias", 0.0))
                    if n_valid >= max(1, min_valid) and abs(bias) > 1e-6 and abs(bias) <= float(t.get("cap_abs_bias", 1e9)):
                        out_df["pred_total"] = out_df["pred_total"] - bias
                        out_df["tuning_totals_bias"] = bias
                        out_df["tuning_applied"] = True
                    else:
                        out_df["tuning_applied"] = False
            except Exception as _te:
                print(f"[yellow]Tuning not applied:[/yellow] {_te}")
            # Preseason blend when configured
            out_df = _apply_preseason_blend(out_df)

            # Guardrails (optional)
            if apply_guardrails and "pred_total" in out_df.columns:
                try:
                    # Unified improved guardrails: derive only when tempo + all ratings present to avoid uniform constant.
                    derived_vals: list[float] = []
                    adjusted_flags: list[bool] = []
                    avail_count = 0
                    missing_count = 0
                    for _, r in feats.iterrows():
                        try:
                            have_tempo = pd.notna(r.get("home_tempo_rating")) and pd.notna(r.get("away_tempo_rating"))
                            have_ratings = all(pd.notna(r.get(k)) for k in ["home_off_rating","away_off_rating","home_def_rating","away_def_rating"])
                            # Treat any placeholder 100.0 sets (all equal 100) as missing to avoid synthetic uniform dv
                            if have_ratings:
                                vals = [float(r.get(k)) for k in ["home_off_rating","away_off_rating","home_def_rating","away_def_rating"]]
                                if len(set(vals)) == 1 and abs(vals[0] - 100.0) < 1e-6:
                                    have_ratings = False
                            if not (have_tempo and have_ratings):
                                dv = np.nan
                                missing_count += 1
                            else:
                                tempo_avg = (float(r.get("home_tempo_rating")) + float(r.get("away_tempo_rating"))) / 2.0
                                off_home = float(r.get("home_off_rating"))
                                off_away = float(r.get("away_off_rating"))
                                def_home = float(r.get("home_def_rating"))
                                def_away = float(r.get("away_def_rating"))
                                exp_home_pp100 = np.clip(off_home - def_away, 65, 140)
                                exp_away_pp100 = np.clip(off_away - def_home, 65, 140)
                                dv = (exp_home_pp100 + exp_away_pp100) / 100.0 * tempo_avg
                                dv = float(np.clip(dv, 95, 195))
                                avail_count += 1
                        except Exception:
                            dv = np.nan
                            missing_count += 1
                        derived_vals.append(dv)
                    out_df["pred_total_raw"] = out_df["pred_total"].astype(float)
                    out_df["derived_total"] = derived_vals
                    blended: list[float] = []
                    for pt, dv in zip(out_df["pred_total_raw"].tolist(), derived_vals):
                        if np.isnan(pt) or np.isnan(dv):
                            blended.append(pt); adjusted_flags.append(False); continue
                        if pt < 103 or (dv > 0 and pt < 0.70 * dv):
                            b = 0.45 * pt + 0.55 * dv
                            b = float(np.clip(b, 95, 195))
                            blended.append(b); adjusted_flags.append(True)
                        else:
                            blended.append(pt); adjusted_flags.append(False)
                    out_df["pred_total"] = blended
                    out_df["pred_total_adjusted"] = adjusted_flags
                    try:
                        out_df["derived_total_available_rows"] = avail_count
                        out_df["derived_total_missing_rows"] = missing_count
                    except Exception:
                        pass
                except Exception:
                    pass

            # Half projections via models if available; else ratio fallback
            used_half = False
            try:
                cols_halves = halves_models_dir / "feature_columns_halves.txt"
                if halves_models_dir.exists() and cols_halves.exists():
                    ch = [c.strip() for c in cols_halves.read_text(encoding="utf-8").splitlines() if c.strip()]
                    Xh = feats.reindex(columns=ch).fillna(0.0).to_numpy(dtype=np.float32)
                    prov_h = OnnxPredictor.describe_available()
                    def _loadh(name: str):
                        p = halves_models_dir / f"baseline_{name}.onnx"
                        if not p.exists():
                            return None
                        return OnnxPredictor(str(p)) if prov_h else (NumpyLinearPredictor(str(p)) if NumpyLinearPredictor.can_load(str(p)) else None)
                    m_t1 = _loadh("target_total_1h"); m_t2 = _loadh("target_total_2h")
                    m_m1 = _loadh("target_margin_1h"); m_m2 = _loadh("target_margin_2h")
                    if any([m_t1, m_t2, m_m1, m_m2]):
                        if m_t1 is not None: out_df["pred_total_1h"] = m_t1.predict(Xh).reshape(-1)
                        if m_t2 is not None: out_df["pred_total_2h"] = m_t2.predict(Xh).reshape(-1)
                        if m_m1 is not None: out_df["pred_margin_1h"] = m_m1.predict(Xh).reshape(-1)
                        if m_m2 is not None: out_df["pred_margin_2h"] = m_m2.predict(Xh).reshape(-1)
                        used_half = True
            except Exception:
                used_half = False
            if not used_half:
                out_df["pred_total_1h"] = pd.to_numeric(out_df["pred_total"], errors="coerce") * half_ratio
                out_df["pred_total_2h"] = pd.to_numeric(out_df["pred_total"], errors="coerce") * (1.0 - half_ratio)
                out_df["pred_margin_1h"] = pd.to_numeric(out_df["pred_margin"], errors="coerce") * 0.5
                out_df["pred_margin_2h"] = pd.to_numeric(out_df["pred_margin"], errors="coerce") * 0.5

            _append_week(out_df, preds_out)
            # Also write dated predictions for the target date
            try:
                pred_dated = settings.outputs_dir / f"predictions_{target_date.isoformat()}.csv"
                out_df.to_csv(pred_dated, index=False)
                print(f"[green]Wrote dated predictions to[/green] {pred_dated}")
            except Exception as _e_pd2:
                print(f"[yellow]Skipping dated predictions write:[/yellow] {_e_pd2}")

        # Accumulate predictions into predictions_all.csv (union & dedupe by game_id)
        try:
            if accumulate_predictions and 'game_id' in out_df.columns:
                pad = settings.outputs_dir / 'predictions_all.csv'
                out_df['game_id'] = out_df['game_id'].astype(str)
                if pad.exists():
                    prev = pd.read_csv(pad)
                    if 'game_id' in prev.columns:
                        prev['game_id'] = prev['game_id'].astype(str)
                    merged_pred = pd.concat([prev, out_df], ignore_index=True)
                    # Keep latest occurrence (assume later row has updated blend/guardrails). Sort by date if present.
                    if 'date' in merged_pred.columns:
                        try:
                            merged_pred['_dt'] = pd.to_datetime(merged_pred['date'], errors='coerce')
                            merged_pred = merged_pred.sort_values(['game_id','_dt']).drop_duplicates(subset=['game_id'], keep='last').drop(columns=['_dt'])
                        except Exception:
                            merged_pred = merged_pred.sort_values(['game_id']).drop_duplicates(subset=['game_id'], keep='last')
                    else:
                        merged_pred = merged_pred.sort_values(['game_id']).drop_duplicates(subset=['game_id'], keep='last')
                else:
                    merged_pred = out_df.copy()
                merged_pred.to_csv(pad, index=False)
                print(f"[green]Accumulated predictions ->[/green] {pad} ({len(merged_pred)} unique games)")
        except Exception as _pred_acc_e:
            print(f"[yellow]Predictions accumulation skipped:[/yellow] {_pred_acc_e}")
    except Exception as e:
        print(f"[red]Prediction failed:[/red] {e}")
        return

    # Before picks: build strict last-odds aggregate and merge to games for derivatives
    try:
        last_out = settings.outputs_dir / "last_odds.csv"
        try:
            _ = make_last_odds(settings.outputs_dir / "odds_history", last_out, tolerance_seconds=60)
        except Exception as _ml:
            print(f"[yellow]make-last-odds skipped:[/yellow] {_ml}")
        merged_last = settings.outputs_dir / "games_with_last.csv"
        try:
            last_df = pd.read_csv(last_out) if last_out.exists() else pd.DataFrame()
            if not last_df.empty:
                merged_last_df = join_games_with_closing(df_games, last_df)
                merged_last_df.to_csv(merged_last, index=False)
                print(f"[green]Wrote merged games+last odds to[/green] {merged_last} ({len(merged_last_df)} rows)")
            else:
                print("[yellow]No last_odds.csv present; derivatives picks will be limited.[/yellow]")
        except Exception as _jl:
            print(f"[yellow]join-last-odds failed:[/yellow] {_jl}")
    except Exception as _dl:
        print(f"[yellow]Derivatives merge step skipped:[/yellow] {_dl}")

    # 7) Make picks -> picks_clean.csv
    picks_out = settings.outputs_dir / "picks_clean.csv"
    try:
        preds_df = pd.read_csv(preds_out)
        if "game_id" in preds_df.columns:
            preds_df["game_id"] = preds_df["game_id"].astype(str)
        bets, _ = backtest_totals(df_games, merged, preds_df, threshold=threshold, default_price=default_price)
        if book_whitelist and not bets.empty:
            allow = {b.strip() for b in book_whitelist.split(',') if b.strip()}
            if 'book' in bets.columns:
                bets = bets[bets['book'].isin(allow)]
        if bets.empty:
            pd.DataFrame(columns=["game_id","date","home_team","away_team","book","bet","line","price","pred_total","edge"]).to_csv(picks_out, index=False)
            print(f"[yellow]No picks meeting threshold {threshold}. Wrote empty sheet to[/yellow] {picks_out}")
        else:
            picks = bets.merge(df_games[["game_id","date","home_team","away_team"]], on="game_id", how="left")
            picks["abs_edge"] = picks["edge"].abs()
            picks = picks.sort_values(["game_id","abs_edge","price"], ascending=[True, False, False])
            picks = picks.drop_duplicates(subset=["game_id"], keep="first")
            if target_picks is not None and target_picks > 0:
                picks = picks.sort_values(["abs_edge", "price"], ascending=[False, False]).head(int(target_picks))
            cols = ["game_id","date","home_team","away_team","book","bet","line","price","pred","edge"]
            picks = picks[cols].rename(columns={"pred":"pred_total"})
            picks.to_csv(picks_out, index=False)
            print(f"[green]Wrote {len(picks)} picks to[/green] {picks_out}")
    except Exception as e:
        print(f"[red]Failed to make picks:[/red] {e}")
        return

    # 7b) Produce expanded picks across totals/spreads/moneyline and halves -> picks_raw.csv
    try:
        preds_df = pd.read_csv(preds_out)
        merged_last_path = settings.outputs_dir / "games_with_last.csv"
        if merged_last_path.exists() and not preds_df.empty:
            # Reuse produce_picks logic by calling the function defined above
            # Avoid circular import; the function is in this module
            picks_raw_out = settings.outputs_dir / "picks_raw.csv"
            produce_picks(
                preds_path=preds_out,
                odds_merged_path=merged_last_path,
                out=picks_raw_out,
                total_threshold=1.5,
                spread_threshold=1.0,
                moneyline_margin_scale=7.0,
                moneyline_edge_pct=2.0,
            )
        else:
            print("[yellow]Skipped produce-picks (missing merged last odds or predictions empty).[/yellow]")
    except Exception as e:
        print(f"[yellow]produce-picks failed:[/yellow] {e}")

    # 8) Ingest outputs into SQLite
    try:
        ingest_outputs(db=db)
    except Exception as e:
        print(f"[yellow]Ingestion step failed:[/yellow] {e}")


@app.command(name="eval-accuracy")
def eval_accuracy_cmd(
    games_path: Path = typer.Argument(..., help="Games CSV/Parquet with final scores"),
    preds_path: Path = typer.Argument(..., help="Predictions CSV with pred_total and pred_margin"),
    out_dir: Path = typer.Option(settings.outputs_dir / "eval", help="Output directory for reports"),
):
    """Compute model accuracy metrics (MAE, RMSE, bias, R2) for totals and margins."""
    # Load games flexibly
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))
    preds = pd.read_csv(preds_path)

    per_game, summary = compute_accuracy(games, preds)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_game.to_csv(out_dir / "accuracy_per_game.csv", index=False)
    # Write summary as CSV and JSON
    pd.DataFrame([summary.__dict__]).to_csv(out_dir / "accuracy_summary.csv", index=False)
    (out_dir / "accuracy_summary.json").write_text(json.dumps(summary.__dict__, indent=2))
    print("[green]Wrote accuracy reports to[/green]", out_dir)
    print(summary.__dict__)


@app.command(name="eval-accuracy-closing")
def eval_accuracy_closing_cmd(
    games_path: Path = typer.Argument(..., help="Games CSV/Parquet with final scores"),
    merged_closing_path: Path = typer.Argument(..., help="games_with_closing.csv from join-closing (must include game_id and totals)"),
    preds_path: Path = typer.Argument(..., help="Predictions CSV with pred_total"),
    out_dir: Path = typer.Option(settings.outputs_dir / "eval", help="Output directory for reports"),
):
    """Compare model accuracy vs closing totals (MAE delta, beat rate, edge correlation)."""
    # Load games flexibly
    if games_path.suffix.lower() == ".csv":
        games = pd.read_csv(games_path)
    else:
        try:
            games = pd.read_parquet(games_path)
        except Exception:
            games = pd.read_csv(games_path.with_suffix(".csv"))
    closing = pd.read_csv(merged_closing_path)
    preds = pd.read_csv(preds_path)

    per_game, summary = compare_vs_closing(games, closing, preds)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_game.to_csv(out_dir / "closing_eval_per_game.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "closing_eval_summary.csv", index=False)
    (out_dir / "closing_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print("[green]Wrote closing comparison reports to[/green]", out_dir)
    print(summary)

@app.command(name="seed-team-map")
def seed_team_map(
    games_path: Path = typer.Option(settings.outputs_dir / "games_curr.csv", help="Games CSV (from fetch-games) used to seed canonical names"),
    odds_path: Path = typer.Option(settings.outputs_dir / "games_with_odds_today.csv", help="Odds or joined CSV used to collect alternate raw names"),
    out: Path = typer.Option(settings.team_map_path, help="Output team map CSV (raw,canonical). Default: data/team_map.csv"),
):
    """Create or update a seed team mapping CSV by scanning games and odds files.

    Heuristic:
    - Use games' team names as canonical for each normalized key.
    - Map any alternate raw names from odds to the same canonical.
    """
    # Load games
    if games_path.exists():
        try:
            if games_path.suffix.lower() == ".csv":
                gdf = pd.read_csv(games_path)
            else:
                gdf = pd.read_parquet(games_path)
        except Exception:
            gdf = pd.read_csv(games_path.with_suffix(".csv")) if games_path.with_suffix(".csv").exists() else pd.DataFrame()
    else:
        gdf = pd.DataFrame()

    # Load odds or joined odds
    odf = pd.DataFrame()
    if odds_path.exists():
        try:
            odf = pd.read_csv(odds_path)
        except Exception:
            odf = pd.DataFrame()

    # Use enhanced canonical normalization
    from .data.team_normalize import canonical_slug as normalize_name

    # Build canonical map from games
    canonical_by_key: dict[str, str] = {}
    if not gdf.empty and {"home_team", "away_team"}.issubset(gdf.columns):
        for name in pd.concat([gdf["home_team"].astype(str), gdf["away_team"].astype(str)], ignore_index=True).dropna().unique():
            key = normalize_name(name)
            if key and key not in canonical_by_key:
                canonical_by_key[key] = name

    # Collect raw names from odds
    raw_names = set()
    for col in ["home_team_name", "away_team_name", "home_team", "away_team"]:
        if col in odf.columns:
            raw_names.update(odf[col].astype(str).dropna().unique().tolist())

    # Build rows: include games names and odds names; default canonical to games' canonical if available, else self
    rows = []
    # Include games names explicitly too (in case of minor variations)
    game_names = set()
    if not gdf.empty and {"home_team", "away_team"}.issubset(gdf.columns):
        game_names.update(gdf["home_team"].astype(str).dropna().unique().tolist())
        game_names.update(gdf["away_team"].astype(str).dropna().unique().tolist())

    for name in sorted(raw_names.union(game_names)):
        key = normalize_name(name)
        canon = canonical_by_key.get(key, name)
        rows.append({"raw": name, "canonical": canon})

    out.parent.mkdir(parents=True, exist_ok=True)
    mdf = pd.DataFrame(rows)
    # If file exists, perform an update merge (preserve any manual edits of canonical where raw matches)
    if out.exists():
        try:
            current = pd.read_csv(out)
            if {"raw", "canonical"}.issubset(current.columns):
                # Prefer existing canonical for matching raw; add new rows
                cur_map = dict(zip(current["raw"].astype(str), current["canonical"].astype(str)))
                mdf["canonical"] = mdf.apply(lambda r: cur_map.get(r["raw"], r["canonical"]), axis=1)
                # Union rows by raw
                mdf = (
                    pd.concat([current[["raw", "canonical"]], mdf[["raw", "canonical"]]], ignore_index=True)
                    .drop_duplicates(subset=["raw"], keep="first")
                    .sort_values("raw")
                )
        except Exception:
            pass

    mdf.to_csv(out, index=False)
    print(f"[green]Wrote team map with {len(mdf)} rows to[/green] {out}")


@app.command(name="evaluate-last2")
def evaluate_last2(
    season1: int = typer.Option(2023, help="First season start year (e.g., 2023 for 2023-24)"),
    season2: int = typer.Option(2024, help="Second season start year (e.g., 2024 for 2024-25)"),
    out_prefix: Path = typer.Option(settings.outputs_dir, help="Outputs directory to write combined artifacts"),
    closing: str = typer.Option("none", help="Closing lines mode: none|current|history"),
):
    """One-shot pipeline for season start: backfill last two seasons, build features, train, predict, and evaluate.

    If `closing` is 'current' or 'history', also fetch odds snapshots, build closing lines,
    join to games, and run the closing-line evaluation.
    """
    import subprocess
    import sys
    from pathlib import Path as _Path
    import pandas as _pd

    def run_cmd(args: list[str]):
        res = subprocess.run([sys.executable, "-m", "ncaab_model.cli", *args], capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise typer.Exit(code=res.returncode)
        # Mirror typical command output
        if res.stdout:
            print(res.stdout.strip())

    out_prefix = _Path(out_prefix)
    out_prefix.mkdir(parents=True, exist_ok=True)

    games1 = out_prefix / f"games_{season1}.csv"
    games2 = out_prefix / f"games_{season2}.csv"
    games_combined = out_prefix / "games_last2.csv"
    boxscores = out_prefix / "boxscores_last2.csv"
    features = out_prefix / "features_last2.csv"
    predictions = out_prefix / "predictions_last2.csv"
    eval_dir = out_prefix / "eval_last2"

    # 1) Fetch games for both seasons
    run_cmd(["fetch-games", "--season", str(season1), "--out", str(games1)])
    run_cmd(["fetch-games", "--season", str(season2), "--out", str(games2)])

    # 2) Combine to games_last2.csv
    dfs = []
    for p in (games1, games2):
        if p.exists():
            dfs.append(_pd.read_csv(p))
    if not dfs:
        print("[red]No games found for the two seasons provided.[/red]")
        raise typer.Exit(code=1)
    all_df = _pd.concat(dfs, ignore_index=True)
    if "game_id" in all_df.columns:
        all_df["game_id"] = all_df["game_id"].astype(str)
        all_df = all_df.drop_duplicates(subset=["game_id"])
    # Normalize dates to ISO date strings if present
    if "date" in all_df.columns:
        try:
            all_df["date"] = _pd.to_datetime(all_df["date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    all_df.to_csv(games_combined, index=False)
    print(f"[green]Wrote[/green] {games_combined} with {len(all_df)} rows")

    # 3) Boxscores and four factors
    run_cmd(["fetch-boxscores", str(games_combined), "--out", str(boxscores)])

    # 4) Build features (schedule + ratings + four-factors)
    run_cmd(["build-features", str(games_combined), "--boxscores-path", str(boxscores), "--out", str(features)])

    # 5) Train + predict
    run_cmd(["train-baseline", str(features)])
    run_cmd(["predict-baseline", str(features), "--out", str(predictions)])

    # 6) Evaluate overall
    eval_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["eval-accuracy", str(games_combined), str(predictions), "--out-dir", str(eval_dir)])

    # 7) Optional: closing-line evaluation
    cmode = (closing or "none").lower()
    if cmode in {"current", "history"}:
        # Date range for odds fetch from games
        min_date = max_date = None
        if "date" in all_df.columns:
            try:
                min_date = _pd.to_datetime(all_df["date"]).min().date().isoformat()
                max_date = _pd.to_datetime(all_df["date"]).max().date().isoformat()
            except Exception:
                pass
        if min_date and max_date:
            run_cmd(["fetch-odds-history", "--start", min_date, "--end", max_date, "--mode", cmode])
        else:
            print("[yellow]Could not parse date range from games; skipping odds fetch.[/yellow]")

        closing_csv = out_prefix / "closing_lines.csv"
        run_cmd(["make-closing-lines", "--in-dir", str(out_prefix / "odds_history"), "--out", str(closing_csv)])
        joined = out_prefix / "games_with_closing_last2.csv"
        run_cmd(["join-closing", str(games_combined), str(closing_csv), "--out", str(joined)])
        run_cmd(["eval-accuracy-closing", str(games_combined), str(joined), str(predictions), "--out-dir", str(eval_dir)])

    print("[green]evaluate-last2 completed.[/green]")


@app.command(name="backfill-history")
def backfill_history(
    start_season: int = typer.Option(2018, help="First season start year to include (e.g., 2018)"),
    end_season: int = typer.Option(dt.datetime.now().year, help="Last season start year to include (inclusive)"),
    out_prefix: Path = typer.Option(settings.outputs_dir, help="Outputs directory to write combined artifacts"),
):
    """Fetch games/boxscores and build features across an arbitrary season range.

    Writes:
      - outputs/games_hist.csv
      - outputs/boxscores_hist.csv
      - outputs/features_hist.csv
    """
    import subprocess
    import sys
    import pandas as _pd

    def run_cmd(args: list[str]):
        res = subprocess.run([sys.executable, "-m", "ncaab_model.cli", *args], capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise typer.Exit(code=res.returncode)
        if res.stdout:
            print(res.stdout.strip())

    out_prefix.mkdir(parents=True, exist_ok=True)
    seasons = list(range(int(start_season), int(end_season) + 1))
    game_paths = []
    for y in seasons:
        p = out_prefix / f"games_{y}.csv"
        run_cmd(["fetch-games", "--season", str(y), "--out", str(p)])
        if p.exists():
            game_paths.append(p)
    if not game_paths:
        print("[red]No games fetched for the provided range.[/red]")
        raise typer.Exit(code=1)
    # Combine
    dfs = [_pd.read_csv(p) for p in game_paths]
    all_df = _pd.concat(dfs, ignore_index=True)
    if "game_id" in all_df.columns:
        all_df["game_id"] = all_df["game_id"].astype(str)
        all_df = all_df.drop_duplicates(subset=["game_id"])
    if "date" in all_df.columns:
        try:
            all_df["date"] = _pd.to_datetime(all_df["date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    games_hist = out_prefix / "games_hist.csv"
    all_df.to_csv(games_hist, index=False)
    print(f"[green]Wrote[/green] {games_hist} with {len(all_df)} rows")

    # Boxscores and features
    boxscores = out_prefix / "boxscores_hist.csv"
    run_cmd(["fetch-boxscores", str(games_hist), "--out", str(boxscores)])
    features = out_prefix / "features_hist.csv"
    run_cmd(["build-features", str(games_hist), "--boxscores-path", str(boxscores), "--out", str(features)])
    print("[green]backfill-history completed.[/green]")

# --- New commands for fused game coverage and segmented team model verification ---

@app.command(name="fetch-games-fused")
def fetch_games_fused(
    season: int = typer.Option(..., help="Season year (by calendar year of season start)"),
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD; defaults to Nov 1 of season"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD; defaults to Apr 15 following season"),
    use_cache: bool = True,
    adjacent: bool = typer.Option(False, help="Extend range by ±1 day to capture cross-midnight slates"),
    out: Path = typer.Option(settings.outputs_dir / "games_fused.parquet", help="Output fused games file (CSV if .csv suffix)"),
):
    """Fetch ESPN + NCAA scoreboards and fuse into a unified D1 games file.

    De-duplicates by (date | home | away) normalized key, preferring rows with:
      1. A non-null start_time (tipoff) (ESPN advantage)
      2. Greater score/linescore completeness.

    Prints coverage statistics (espn_only, ncaa_only, both) and warns if any day
    in the requested range appears abnormally low (<15 games mid-season).
    """
    # Date range
    if start:
        start_date = dt.date.fromisoformat(start)
    else:
        start_date = dt.date(season, 11, 1)
    if end:
        end_date = dt.date.fromisoformat(end)
    else:
        end_date = dt.date(season + 1, 4, 15)
    fetch_start = start_date - dt.timedelta(days=1) if adjacent else start_date
    fetch_end = end_date + dt.timedelta(days=1) if adjacent else end_date

    espn_rows: list[dict] = []
    ncaa_rows: list[dict] = []
    for res in iter_games_espn(fetch_start, fetch_end, use_cache=use_cache):
        for g in res.games:
            d = g.model_dump()
            d["source"] = "espn"
            espn_rows.append(d)
    for res in iter_games_ncaa(fetch_start, fetch_end, use_cache=use_cache):
        for g in res.games:
            d = g.model_dump()
            d["source"] = "ncaa"
            ncaa_rows.append(d)

    if not espn_rows and not ncaa_rows:
        print("[yellow]No games found from either provider in range.[/yellow]")
        return

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Normalize key on date + normalized team names
        if "date" in df.columns:
            try:
                df["_date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                df["_date"] = df["date"].astype(str)
        else:
            df["_date"] = ""
        for col in ["home_team", "away_team"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df["_home_key"] = df.get("home_team", pd.Series(dtype=str)).astype(str).map(lambda x: _norm(x))
        df["_away_key"] = df.get("away_team", pd.Series(dtype=str)).astype(str).map(lambda x: _norm(x))
        df["_fuse_key"] = df["_date"] + "|" + df["_home_key"] + "|" + df["_away_key"]
        # Quality metric: start_time presence + number of non-null score fields
        score_cols = [c for c in ["home_score","away_score","home_score_1h","away_score_1h","home_score_2h","away_score_2h"] if c in df.columns]
        df["_score_nonnull"] = df[score_cols].notna().sum(axis=1) if score_cols else 0
        df["_has_start"] = df.get("start_time").notna() if "start_time" in df.columns else False
        df["_quality"] = df["_has_start"].astype(int) * 10 + df["_score_nonnull"].astype(int)
        return df

    espn_df = _prep(pd.DataFrame(espn_rows))
    ncaa_df = _prep(pd.DataFrame(ncaa_rows))
    combined = pd.concat([espn_df, ncaa_df], ignore_index=True)

    # Deduplicate by fuse key preferring highest quality
    if "_fuse_key" in combined.columns:
        combined = combined.sort_values(["_fuse_key","_quality"], ascending=[True, False])
        combined = combined.drop_duplicates(subset=["_fuse_key"], keep="first")

    # Filter back to original requested range (exclude adjacent padding) based on date
    try:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        mask = (combined["date"].dt.date >= start_date) & (combined["date"].dt.date <= end_date)
        combined = combined[mask].copy()
        # Return date column to string
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    # Coverage stats
    espn_keys = set(espn_df["_fuse_key"].tolist()) if not espn_df.empty else set()
    ncaa_keys = set(ncaa_df["_fuse_key"].tolist()) if not ncaa_df.empty else set()
    fused_keys = set(combined["_fuse_key"].tolist()) if not combined.empty else set()
    both = espn_keys & ncaa_keys
    espn_only = espn_keys - ncaa_keys
    ncaa_only = ncaa_keys - espn_keys
    stats = {
        "espn_rows": len(espn_df),
        "ncaa_rows": len(ncaa_df),
        "espn_unique": len(espn_keys),
        "ncaa_unique": len(ncaa_keys),
        "fused": len(fused_keys),
        "both_overlap": len(both),
        "espn_only": len(espn_only),
        "ncaa_only": len(ncaa_only),
    }
    print(stats)
    # Persist per-day counts & anomalies for later UI / monitoring if within <= 2 year span
    try:
        counts_csv = settings.outputs_dir / "schedule_day_counts.csv"
        per_day_df = per_day.reset_index().rename(columns={"index":"date"}) if isinstance(per_day, pd.Series) else pd.DataFrame()
        if not per_day_df.empty:
            per_day_df.to_csv(counts_csv, index=False)
    except Exception:
        pass

    # Per-day simple anomaly warning (mid-season expectation heuristic)
    try:
        per_day = combined.groupby("date").size().rename("n_games")
        warnings = []
        for date_str, n_games in per_day.items():
            try:
                d = dt.date.fromisoformat(date_str)
                if d.month in {11,12,1,2,3} and n_games < 15:  # crude heuristic
                    warnings.append(f"{date_str}: only {n_games} games")
            except Exception:
                continue
        if warnings:
            print("[yellow]Low coverage warnings:[/yellow]", warnings)
    except Exception:
        pass

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        combined.to_csv(out, index=False)
    else:
        try:
            combined.to_parquet(out, index=False)
        except Exception as e:
            csv_alt = out.with_suffix(".csv")
            combined.to_csv(csv_alt, index=False)
            print(f"[yellow]Parquet write failed ({e}); wrote CSV to[/yellow] {csv_alt}")
    print(f"[green]Wrote fused games to[/green] {out} ({len(combined)} rows)")


@app.command(name="verify-seg-team-models")
def verify_seg_team_models(
    models_root: Path = typer.Option(settings.outputs_dir / "models", help="Root models directory containing seg_team"),
    branding_csv: Path = typer.Option(settings.data_dir / "team_branding.csv", help="Branding CSV listing all D1 teams"),
    features_csv: Path = typer.Option(settings.outputs_dir / "features_hist.csv", help="Historical features to train missing team models (multi-season with targets)"),
    train_missing: bool = typer.Option(False, help="Train missing team models if enough rows available"),
    min_rows: int = typer.Option(150, help="Minimum rows required per team to train a model"),
    out: Path = typer.Option(settings.outputs_dir / "seg_team_coverage.csv", help="Output coverage report CSV"),
):
    """Verify per-team segmented model coverage; optionally train models for missing teams.

    Coverage report columns: team_key, team_name, has_model, rows_available, trained_now.
    """
    seg_dir = models_root / "seg_team"
    seg_dir.mkdir(parents=True, exist_ok=True)
    # Collect team names from branding CSV (robust across column naming variants)
    if not branding_csv.exists():
        print(f"[red]Missing branding CSV at {branding_csv}. Run update-branding first.[/red]")
        raise typer.Exit(code=1)
    bdf = pd.read_csv(branding_csv)
    # Heuristic: take any column with 'team' or 'display' in name as candidate
    team_cols = [c for c in bdf.columns if any(k in str(c).lower() for k in ["team","display","name"])] or list(bdf.columns[:1])
    raw_names = set()
    for c in team_cols:
        raw_names.update(bdf[c].astype(str).dropna().tolist())
    teams = sorted({_norm(n) for n in raw_names if n and _norm(n) not in {"", "tbd", "tba", "unknown"}})
    # Existing model keys
    existing = set()
    for p in seg_dir.glob("*"):
        if p.is_dir():
            key = p.name
            if (p / "baseline_target_total.onnx").exists():
                existing.add(key)
    # Load features once if training
    feats_df: pd.DataFrame | None = None
    if train_missing:
        if not features_csv.exists():
            print(f"[red]Features CSV {features_csv} missing; cannot train missing models.[/red]")
            train_missing = False
        else:
            feats_df = pd.read_csv(features_csv)
            for col in ["home_team","away_team"]:
                if col in feats_df.columns:
                    feats_df[col] = feats_df[col].astype(str)
            # enforce game_id string
            if "game_id" in feats_df.columns:
                feats_df["game_id"] = feats_df["game_id"].astype(str)
    rows: list[dict] = []
    trained_count = 0
    for key in teams:
        # Attempt to recover original display name (first matching branding row)
        display_name = None
        for c in team_cols:
            cand = bdf[bdf[c].astype(str).map(lambda x: _norm(x) == key)]
            if not cand.empty:
                display_name = cand[c].astype(str).iloc[0]
                break
        has_model = key in existing
        rows_available = None
        trained_now = False
        if train_missing and not has_model and feats_df is not None and {"home_team","away_team","target_total","target_margin"}.issubset(feats_df.columns):
            mask = (feats_df["home_team"].astype(str).map(_norm) == key) | (feats_df["away_team"].astype(str).map(_norm) == key)
            sub = feats_df[mask].dropna(subset=["target_total","target_margin"]) if mask.any() else pd.DataFrame()
            rows_available = len(sub)
            if rows_available >= int(min_rows):
                # Train baseline for this team
                tdir = seg_dir / key
                tdir.mkdir(parents=True, exist_ok=True)
                tmp_csv = tdir / "features_segment.csv"
                sub.to_csv(tmp_csv, index=False)
                try:
                    train_baseline(tmp_csv, tdir, loss_totals="huber", huber_delta=8.0)
                    has_model = True
                    trained_now = True
                    trained_count += 1
                except Exception as e:
                    print(f"[yellow]Training failed for {display_name or key}:[/yellow] {e}")
            else:
                # Seed a minimal placeholder model using global baseline columns if available (ensures prediction path works)
                try:
                    tdir = seg_dir / key
                    tdir.mkdir(parents=True, exist_ok=True)
                    # Build tiny synthetic feature set replicating global structure
                    cols_path = settings.outputs_dir / "models" / "feature_columns.txt"
                    if cols_path.exists():
                        cols = [c.strip() for c in cols_path.read_text(encoding="utf-8").splitlines() if c.strip()]
                        synth = pd.DataFrame({"game_id": [f"seed-{key}-0"], "date": ["1970-01-01"], "home_team": [display_name or key], "away_team": [display_name or key]})
                        for c in cols:
                            if c not in synth.columns:
                                synth[c] = 0.0
                        synth["target_total"] = 140.0
                        synth["target_margin"] = 0.0
                        tmp_csv = tdir / "features_segment.csv"
                        synth.to_csv(tmp_csv, index=False)
                        train_baseline(tmp_csv, tdir, loss_totals="ridge")
                        has_model = True
                        trained_now = True
                        trained_count += 1
                except Exception as e:
                    print(f"[yellow]Placeholder model failed for {display_name or key}:[/yellow] {e}")
        else:
            # If not training, attempt to count available rows just for report
            if feats_df is not None and {"home_team","away_team"}.issubset(feats_df.columns):
                mask = (feats_df["home_team"].astype(str).map(_norm) == key) | (feats_df["away_team"].astype(str).map(_norm) == key)
                rows_available = int(mask.sum())
        rows.append({
            "team_key": key,
            "team_name": display_name or key,
            "has_model": has_model,
            "rows_available": rows_available,
            "trained_now": trained_now,
        })
    cov_df = pd.DataFrame(rows).sort_values("team_key")
    out.parent.mkdir(parents=True, exist_ok=True)
    cov_df.to_csv(out, index=False)
    coverage_pct = 100.0 * cov_df["has_model"].sum() / max(1, len(cov_df))
    print({
        "teams_total": len(cov_df),
        "teams_with_model": int(cov_df["has_model"].sum()),
        "coverage_pct": round(coverage_pct, 2),
        "trained_now": trained_count,
        "report": str(out),
    })
    if coverage_pct < 95:
        print("[yellow]Coverage below 95%; consider lowering min_rows or expanding historical data.[/yellow]")


@app.command(name="update-conferences-espn")
def update_conferences_espn(
    out: Path = typer.Option(settings.data_dir / "conferences.csv", help="Output conferences.csv (team,conference)"),
):
    """Generate conferences.csv using ESPN teams metadata.

    Uses the same source as update-branding; prefers ESPN's group/conference field.
    """
    try:
        df = fetch_espn_branding()
        if df.empty or "team" not in df.columns:
            print("[red]Failed to fetch ESPN teams list.[/red]")
            raise typer.Exit(code=1)
        if "conference" not in df.columns:
            df["conference"] = ""
        conf = df[["team","conference"]].copy()
        conf = conf.dropna(subset=["team"]).drop_duplicates(subset=["team"]).sort_values("team")
        out.parent.mkdir(parents=True, exist_ok=True)
        conf.to_csv(out, index=False)
        print(f"[green]Wrote {len(conf)} teams with conferences to[/green] {out}")
    except Exception as e:
        print(f"[red]update-conferences-espn failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(name="backfill-history-fused")
def backfill_history_fused(
    start_season: int = typer.Option(2023, help="First season start year to include (e.g., 2023)"),
    end_season: int = typer.Option(dt.datetime.now().year, help="Last season start year to include (inclusive)"),
    out_prefix: Path = typer.Option(settings.outputs_dir, help="Outputs directory to write artifacts"),
    adjacent: bool = typer.Option(True, help="Use ±1 day padding to catch cross-midnight games"),
    train_seg_teams: bool = typer.Option(True, help="Train segmented team models after backfill"),
    min_rows: int = typer.Option(140, help="Min rows per team to train segmented model"),
):
    """Backfill fused games across seasons, build features, and train segmented team models.

    Writes games_hist_fused.csv, boxscores_hist.csv, features_hist.csv, then trains seg_team models.
    """
    import subprocess, sys
    from pathlib import Path as _Path
    import pandas as _pd

    def run_cmd(args: list[str]):
        res = subprocess.run([sys.executable, "-m", "ncaab_model.cli", *args], capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise typer.Exit(code=res.returncode)
        if res.stdout:
            print(res.stdout.strip())

    out_prefix = _Path(out_prefix)
    out_prefix.mkdir(parents=True, exist_ok=True)
    seasons = list(range(int(start_season), int(end_season) + 1))
    season_paths: list[_Path] = []
    # 1) Fused fetch per season
    for y in seasons:
        p = out_prefix / f"games_{y}_fused.csv"
        run_cmd([
            "fetch-games-fused", "--season", str(y), "--out", str(p), "--no-use-cache",
            *( ["--adjacent"] if adjacent else [] )
        ])
        if p.exists():
            season_paths.append(p)
    if not season_paths:
        print("[red]No fused games fetched for the provided range.[/red]")
        raise typer.Exit(code=1)

    # 2) Combine to games_hist_fused.csv
    dfs = [_pd.read_csv(p) for p in season_paths]
    all_df = _pd.concat(dfs, ignore_index=True)
    if "game_id" in all_df.columns:
        all_df["game_id"] = all_df["game_id"].astype(str)
        all_df = all_df.drop_duplicates(subset=["game_id"])  # keep ESPN IDs stable where available
    if "date" in all_df.columns:
        try: all_df["date"] = _pd.to_datetime(all_df["date"]).dt.strftime("%Y-%m-%d")
        except Exception: pass
    games_hist = out_prefix / "games_hist_fused.csv"
    all_df.to_csv(games_hist, index=False)
    print(f"[green]Wrote[/green] {games_hist} with {len(all_df)} rows")

    # 3) Boxscores and features
    boxscores = out_prefix / "boxscores_hist.csv"
    run_cmd(["fetch-boxscores", str(games_hist), "--out", str(boxscores)])
    features = out_prefix / "features_hist.csv"
    run_cmd(["build-features", str(games_hist), "--boxscores-path", str(boxscores), "--out", str(features)])

    # 4) Train segmented team models
    if train_seg_teams:
        run_cmd(["train-segmented", str(features), "--segment", "team", "--min-rows", str(int(min_rows))])
        # 5) Verify/optionally fill any remaining gaps
        run_cmd(["verify-seg-team-models", "--train-missing", "--min-rows", str(int(min_rows))])
    print("[green]backfill-history-fused completed.[/green]")

@app.command(name="report-schedule-coverage")
def report_schedule_coverage(
    games_path: Path = typer.Argument(settings.outputs_dir / "games_hist_fused.csv", help="Games CSV with date column"),
    out_csv: Path = typer.Option(settings.outputs_dir / "schedule_coverage.csv", help="Output CSV for per-day counts and anomaly flag"),
    min_midseason: int = typer.Option(15, help="Min expected games for Nov-Mar"),
    min_offseason: int = typer.Option(5, help="Min expected games for other months"),
):
    """Compute per-day game counts and flag anomalies when counts fall below month thresholds."""
    if not games_path.exists():
        print(f"[red]Missing games file: {games_path}[/red]")
        raise typer.Exit(code=1)
    df = pd.read_csv(games_path, usecols=["date"]) if games_path.suffix.lower() == ".csv" else pd.read_parquet(games_path)
    if "date" not in df.columns:
        print("[red]games file missing 'date' column[/red]")
        raise typer.Exit(code=1)
    try:
        dd = pd.to_datetime(df["date"], errors="coerce").dt.date
    except Exception:
        print("[red]Could not parse date column[/red]")
        raise typer.Exit(code=1)
    counts = dd.value_counts().rename_axis("date").reset_index(name="n_games")
    # Sort by date ascending
    counts = counts.sort_values("date")
    # Flag anomalies by month
    def flag(row):
        m = row["date"].month
        thresh = min_midseason if m in {11,12,1,2,3} else min_offseason
        return bool(row["n_games"] < thresh)
    counts["anomaly"] = counts.apply(flag, axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_csv, index=False)
    print(f"[green]Wrote schedule coverage to[/green] {out_csv} ({len(counts)} days)")


# ---------------- Odds Coverage & Name Matching Diagnostics -----------------

@app.command(name="odds-coverage")
def odds_coverage(
    games_path: Path = typer.Option(settings.outputs_dir / "games_with_odds_today.csv", help="Joined games+odds CSV or raw games CSV if join not yet performed"),
    odds_path: Path = typer.Option(settings.outputs_dir / "odds_today.csv", help="Raw odds snapshot CSV (from fetch-odds or fetch-odds-history)"),
    fuzzy_threshold: int = typer.Option(88, help="Minimum fuzzy score (token_set_ratio avg) to suggest candidate mapping"),
    top_k: int = typer.Option(3, help="Number of top fuzzy candidates to display per unmatched game"),
    out_report: Path = typer.Option(settings.outputs_dir / "odds_coverage_report.csv", help="Detailed per-game coverage report output"),
    suggest_aliases: bool = typer.Option(True, help="Emit suggested ALIAS_MAP additions for unmatched patterns"),
    show_per_book: bool = typer.Option(True, help="Also print per-book coverage summary"),
    out_books: Path | None = typer.Option(None, help="Optional: write per-book coverage CSV with columns [book, covered_games, coverage_pct]"),
    candidate_aliases: bool = typer.Option(True, help="Use rule-based generate_alias_candidates to propose additional alias suggestions"),
):
    """Report current odds coverage and suggest name normalizations to push toward 100%.

    Coverage Definition:
      - A game is 'covered' if at least one bookmaker row is attached (book not null).
      - Uses existing joined file if it contains 'book'; otherwise attempts an in-memory join.

    For each unmatched game, computes fuzzy similarity between (home,away) pairs and odds rows
    on the same date to surface likely naming discrepancies, producing suggested alias entries.
    """
    import math
    import pandas as _pd
    from .data.team_normalize import canonical_slug, pair_key, generate_alias_candidates
    from .data.merge_odds import join_odds_to_games as _join
    try:
        from rapidfuzz import fuzz as _fuzz  # type: ignore
    except Exception:  # pragma: no cover
        _fuzz = None

    if not games_path.exists():
        print(f"[red]Games file not found:[/red] {games_path}")
        raise typer.Exit(code=1)
    if not odds_path.exists():
        print(f"[red]Odds file not found:[/red] {odds_path}")
        raise typer.Exit(code=1)

    gdf = _pd.read_csv(games_path)
    odf = _pd.read_csv(odds_path)
    # Determine if join already present
    if 'book' not in gdf.columns or gdf['book'].isna().all():
        # Need to perform a lightweight join (only game_id,date,home_team,away_team present?)
        core_cols = ['game_id','date','home_team','away_team']
        missing_core = [c for c in core_cols if c not in gdf.columns]
        if missing_core:
            print(f"[red]Games CSV missing required columns for join:{missing_core}[/red]")
            raise typer.Exit(code=1)
        # Minimal join
        try:
            joined = _join(gdf, odf, use_fuzzy=True, fuzzy_threshold=92, date_tolerance_days=1)
            gdf = joined
        except Exception as e:
            print(f"[yellow]Join failed; proceeding with original games data:[/yellow] {e}")

    # Build report rows (unique game perspective)
    rows = []
    for gid, grp in gdf.groupby('game_id'):
        home = str(grp.iloc[0].get('home_team',''))
        away = str(grp.iloc[0].get('away_team',''))
        date = str(grp.iloc[0].get('date',''))
        covered = bool(grp.get('book').notna().any()) if 'book' in grp.columns else False
        slug_home = canonical_slug(home)
        slug_away = canonical_slug(away)
        rows.append({
            'game_id': gid,
            'date': date,
            'home_team': home,
            'away_team': away,
            'home_slug': slug_home,
            'away_slug': slug_away,
            'covered': covered,
            'n_books': int(grp['book'].nunique()) if covered and 'book' in grp.columns else 0,
        })
    rep_df = _pd.DataFrame(rows)

    # Accurate coverage stats (unique games with any odds attached)
    n_games = len(rep_df)
    n_covered = int(rep_df['covered'].sum())
    covered_pct = (n_covered / max(1, n_games)) if n_games else 0.0
    print(f"[bold]Coverage[/bold]: {n_covered}/{n_games} games => {covered_pct:.2%}")

    # Optional: per-book coverage across unique games
    books_df = None
    if show_per_book and 'book' in gdf.columns and n_games:
        by_book = gdf.dropna(subset=['book']).groupby('book')['game_id'].nunique().reset_index(name='covered_games')
        if not by_book.empty:
            by_book['coverage_pct'] = by_book['covered_games'].astype(float) / float(n_games)
            books_df = by_book.sort_values('coverage_pct', ascending=False)
            top = books_df.head(10)
            print("[bold]Per-book coverage (top 10)[/bold]:")
            for _, r in top.iterrows():
                print(f"  {r['book']}: {int(r['covered_games'])}/{n_games} ({r['coverage_pct']:.2%})")
            if out_books is not None:
                books_df.to_csv(out_books, index=False)
                print(f"[green]Wrote per-book coverage to[/green] {out_books}")

    # Fuzzy suggestions for uncovered
    suggestions = []  # fuzzy suggestions
    if _fuzz is not None and not rep_df.empty:
        uncovered = rep_df[~rep_df['covered']]
        # Prepare odds pool indexed by date
        if 'commence_time' in odf.columns:
            odf['commence_time'] = _pd.to_datetime(odf['commence_time'], errors='coerce')
            odf['odds_date'] = odf['commence_time'].dt.strftime('%Y-%m-%d')
        elif 'date' in odf.columns:
            odf['odds_date'] = odf['date'].astype(str)
        else:
            odf['odds_date'] = ''
        for _, r in uncovered.iterrows():
            pool = odf[odf['odds_date'] == str(r['date'])]
            if pool.empty:
                continue
            # Score each candidate event row
            score_map: dict[tuple[str,str], float] = {}
            for idx, prow in pool.iterrows():
                h = str(prow.get('home_team_name',''))
                a = str(prow.get('away_team_name',''))
                s_hh = _fuzz.token_set_ratio(r['home_team'], h)
                s_aa = _fuzz.token_set_ratio(r['away_team'], a)
                s_cross1 = _fuzz.token_set_ratio(r['home_team'], a)
                s_cross2 = _fuzz.token_set_ratio(r['away_team'], h)
                score_direct = (s_hh + s_aa)/2.0
                score_cross = (s_cross1 + s_cross2)/2.0
                score = max(score_direct, score_cross)
                key = (h, a)
                prev = score_map.get(key, -1)
                if score > prev:
                    score_map[key] = score
            if not score_map:
                continue
            # Sort candidates by score descending and take top_k above threshold
            sorted_cands = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
            for (h,a), score in sorted_cands[:top_k]:
                if score < fuzzy_threshold:
                    continue
                suggestions.append({
                    'game_id': r['game_id'],
                    'date': r['date'],
                    'home_team': r['home_team'],
                    'away_team': r['away_team'],
                    'candidate_home': h,
                    'candidate_away': a,
                    'fuzzy_score': score,
                })
    sug_df = _pd.DataFrame(suggestions)

    # Rule-based candidate alias suggestions (independent of fuzzy) for uncovered games
    rule_alias_rows = []
    if candidate_aliases and not rep_df.empty:
        uncovered = rep_df[~rep_df['covered']]
        # Precompute odds slugs by date for fast matching
        odf_slugs = []
        if 'commence_time' in odf.columns:
            odf['commence_time'] = _pd.to_datetime(odf['commence_time'], errors='coerce')
            odf['odds_date'] = odf['commence_time'].dt.strftime('%Y-%m-%d')
        elif 'date' in odf.columns:
            odf['odds_date'] = odf['date'].astype(str)
        else:
            odf['odds_date'] = ''
        for _, row in odf.iterrows():
            h = str(row.get('home_team_name',''))
            a = str(row.get('away_team_name',''))
            odf_slugs.append((row.get('odds_date',''), canonical_slug(h), canonical_slug(a)))
        # Build per-date set of pair slugs for odds rows
        from collections import defaultdict
        odds_pairs_by_date: dict[str, set[str]] = defaultdict(set)
        for d, hs, as_ in odf_slugs:
            pk = pair_key(hs, as_)
            odds_pairs_by_date[d].add(pk)
        for _, grow in uncovered.iterrows():
            date = str(grow['date'])
            # Generate candidate slugs for each team name
            cand_home = {canonical_slug(c) for c in generate_alias_candidates(grow['home_team'])}
            cand_away = {canonical_slug(c) for c in generate_alias_candidates(grow['away_team'])}
            # Existing canonical slugs
            base_home = canonical_slug(grow['home_team'])
            base_away = canonical_slug(grow['away_team'])
            # Try combinations of candidate replacements (cartesian) limited
            tested = 0
            for ch in sorted(cand_home | {base_home}):
                for ca in sorted(cand_away | {base_away}):
                    pk = pair_key(ch, ca)
                    tested += 1
                    if pk in odds_pairs_by_date.get(date, set()):
                        # Found a candidate combination present in odds
                        rule_alias_rows.append({
                            'game_id': grow['game_id'],
                            'date': date,
                            'home_team': grow['home_team'],
                            'away_team': grow['away_team'],
                            'candidate_home_slug': ch,
                            'candidate_away_slug': ca,
                            'base_home_slug': base_home,
                            'base_away_slug': base_away,
                        })
                    if tested > 200:  # safety cap
                        break
                if tested > 200:
                    break
    rule_df = _pd.DataFrame(rule_alias_rows)

    # Write report
    rep_df.to_csv(out_report, index=False)
    print(f"[green]Wrote coverage report to[/green] {out_report} ({len(rep_df)} games)")
    if not sug_df.empty:
        print(f"Top fuzzy suggestions (score >= {fuzzy_threshold}): {len(sug_df)}")
        preview = sug_df.sort_values('fuzzy_score', ascending=False).head(10)
        for _, prow in preview.iterrows():
            print(f"[cyan]{prow['date']}[/cyan] {prow['home_team']} vs {prow['away_team']} -> candidate: {prow['candidate_home']} / {prow['candidate_away']} (score {prow['fuzzy_score']})")
    else:
        print("[yellow]No fuzzy candidate suggestions (either full coverage or low similarity).[/yellow]")

    if suggest_aliases and (not sug_df.empty or not rule_df.empty):
        # Generate alias map suggestions: canonical slugs of candidate vs game teams if distinct
        alias_lines = []
        for _, srow in sug_df.iterrows():
            hs = canonical_slug(str(srow['home_team']))
            cs_h = canonical_slug(str(srow['candidate_home']))
            if hs != cs_h:
                alias_lines.append(f"    \"{cs_h}\": \"{hs}\",")
            as_ = canonical_slug(str(srow['away_team']))
            cs_a = canonical_slug(str(srow['candidate_away']))
            if as_ != cs_a:
                alias_lines.append(f"    \"{cs_a}\": \"{as_}\",")
        for _, rrow in rule_df.iterrows():
            # Only suggest if candidate differs from base
            if rrow['candidate_home_slug'] != rrow['base_home_slug']:
                alias_lines.append(f"    \"{rrow['candidate_home_slug']}\": \"{rrow['base_home_slug']}\",")
            if rrow['candidate_away_slug'] != rrow['base_away_slug']:
                alias_lines.append(f"    \"{rrow['candidate_away_slug']}\": \"{rrow['base_away_slug']}\",")
        alias_lines = sorted(set(alias_lines))
        if alias_lines:
            print("\n[bold]Suggested ALIAS_MAP additions[/bold] (review before adding to team_normalize.py):")
            print("ALIAS_MAP.update({")
            for l in alias_lines[:50]:  # cap output
                print(l)
            print("})")
        else:
            print("[yellow]No alias suggestions generated (slugs already aligned).[/yellow]")

    # Summary line for automation
    print(json.dumps({
        "games": int(n_games),
        "covered_games": int(n_covered),
        "coverage_pct": round(100.0 * covered_pct, 2),
        "fuzzy_suggestions": int(len(sug_df)),
        "rule_suggestions": int(len(rule_df)),
    }))


@app.command(name="diagnose-odds-missing")
def diagnose_odds_missing(
    games_path: Path = typer.Argument(settings.outputs_dir / "games_curr.csv", help="Scheduled games CSV (date, home_team, away_team)"),
    odds_paths: list[Path] = typer.Option(None, help="Repeat this option to include multiple odds CSVs (e.g., --odds-paths A.csv --odds-paths B.csv)"),
    odds_dir: Path | None = typer.Option(None, help="Directory with odds_YYYY-MM-DD.csv files to auto-include for game dates and next-day"),
    books_filter: str | None = typer.Option(None, help="Optional comma-separated list of books to focus on (e.g., DraftKings,FanDuel)"),
    show: int = typer.Option(25, help="Show up to N missing examples"),
    probe_events: bool = typer.Option(True, help="Query events endpoint by date to see if missing games exist at the provider at all"),
    region: str = typer.Option("us", help="TheOddsAPI region for event probe"),
    probe_alt_keys: bool = typer.Option(False, help="Also probe alternate sport keys and report event counts per key"),
    alt_keys: str = typer.Option(
        "basketball_ncaab,basketball_ncaab_preseason,basketball_ncaab_exhibition,basketball_ncaa",
        help="Comma-separated sport keys to probe when --probe-alt-keys is set",
    ),
    union_days: int = typer.Option(1, help="Union provider events across +/- this many days around the target date for presence checks"),
    probe_no_date: bool = typer.Option(True, help="Also fetch the unfiltered events list and locally filter by date to compare presence"),
):
    """Diagnose why scheduled games lack odds coverage.

    - Loads scheduled games and one or more odds snapshots.
    - Uses canonical normalization to find presence of each game in odds by home/away pair.
    - Reports counts and prints examples of missing pairs.
    - If books_filter is set, filters odds rows to those books first.
    """
    import pandas as _pd
    from .data.team_normalize import canonical_slug, pair_key

    if not games_path.exists():
        print(f"[red]Games file not found:[/red] {games_path}")
        raise typer.Exit(code=1)
    g = _pd.read_csv(games_path)
    need_cols = {'date','home_team','away_team'}
    if not need_cols.issubset(set(g.columns)):
        print(f"[red]Games CSV missing required columns[/red]: {need_cols - set(g.columns)}")
        raise typer.Exit(code=1)
    g['date'] = _pd.to_datetime(g['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    g['home_slug'] = g['home_team'].astype(str).map(canonical_slug)
    g['away_slug'] = g['away_team'].astype(str).map(canonical_slug)
    g['pair'] = g.apply(lambda r: pair_key(r['home_team'], r['away_team']), axis=1)

    # Load and combine odds
    candidates: list[Path] = []
    if odds_paths:
        candidates.extend(list(odds_paths))
    if odds_dir and odds_dir.exists():
        uniq_dates = sorted(set(g['date']))
        # Also include next day per date to account for UTC shifts
        for d in uniq_dates:
            candidates.append(odds_dir / f'odds_{d}.csv')
            try:
                d1 = (dt.date.fromisoformat(d) + dt.timedelta(days=1)).isoformat()
                candidates.append(odds_dir / f'odds_{d1}.csv')
            except Exception:
                pass
    if not candidates:
        # Default fallback
        uniq_dates = sorted(set(g['date']))
        for d in uniq_dates:
            candidates.append(settings.outputs_dir / 'odds_history' / f'odds_{d}.csv')
            try:
                d1 = (dt.date.fromisoformat(d) + dt.timedelta(days=1)).isoformat()
                candidates.append(settings.outputs_dir / 'odds_history' / f'odds_{d1}.csv')
            except Exception:
                pass
        candidates.append(settings.outputs_dir / 'odds_today.csv')
    odds_paths = [p for p in candidates if p and p.exists()]
    if not odds_paths:
        print("[yellow]No odds CSVs provided or found; cannot diagnose.[/yellow]")
        raise typer.Exit(code=1)

    frames = []
    for p in odds_paths:
        try:
            df = _pd.read_csv(p)
            df['_source'] = str(p)
            frames.append(df)
        except Exception as e:
            print(f"[yellow]Failed to read {p}:[/yellow] {e}")
    if not frames:
        print("[yellow]No readable odds CSVs; cannot diagnose.[/yellow]")
        raise typer.Exit(code=1)
    o = _pd.concat(frames, ignore_index=True)
    # Filter by books if requested
    if books_filter:
        keep = {b.strip() for b in books_filter.split(',') if b.strip()}
        if 'book' in o.columns:
            o = o[o['book'].isin(list(keep))]

    # Normalize and build pair keys for odds
    # Determine date column robustly
    if 'commence_time' in o.columns:
        o['date'] = _pd.to_datetime(o['commence_time'], errors='coerce').dt.strftime('%Y-%m-%d')
    elif 'date' in o.columns:
        o['date'] = _pd.to_datetime(o['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        o['date'] = ''
    o['home_team_name'] = o.get('home_team_name', '')
    o['away_team_name'] = o.get('away_team_name', '')
    o['pair'] = o.apply(lambda r: pair_key(r['home_team_name'], r['away_team_name']), axis=1)

    # Presence map by date and pair
    present = set(zip(o['date'], o['pair']))
    total = len(g)
    have = 0
    missing_rows = []
    for _, r in g.iterrows():
        k = (r['date'], r['pair'])
        if k in present:
            have += 1
        else:
            missing_rows.append({
                'date': r['date'],
                'home_team': r['home_team'],
                'away_team': r['away_team'],
                'home_slug': r['home_slug'],
                'away_slug': r['away_slug'],
            })
    pct = have / max(1, total)
    print(f"[bold]Odds presence[/bold]: {have}/{total} games in provided snapshots => {pct:.2%}")
    if missing_rows:
        print(f"[bold]Missing examples (up to {show})[/bold]:")
        for row in missing_rows[:show]:
            print(f"  {row['date']}: {row['home_team']} vs {row['away_team']} -> slugs {row['home_slug']} / {row['away_slug']}")
    else:
        print("All scheduled games present in odds snapshots.")

    # Per-book presence summary
    if 'book' in o.columns and total:
        by_book = o.groupby('book')['pair'].nunique().reset_index(name='events_present')
        # Convert to unique scheduled games covered per book by intersecting with g pairs by date
        scheduled = set(zip(g['date'], g['pair']))
        rows = []
        for b, grp in o.groupby('book'):
            have_b = set(zip(grp['date'], grp['pair'])) & scheduled
            rows.append({'book': b, 'covered_games': len(have_b), 'coverage_pct': len(have_b)/max(1,total)})
        if rows:
            rows_sorted = sorted(rows, key=lambda x: x['coverage_pct'], reverse=True)
            print("[bold]Per-book scheduled game presence[/bold] (top 10):")
            for r in rows_sorted[:10]:
                print(f"  {r['book']}: {r['covered_games']}/{total} ({r['coverage_pct']:.2%})")

    # Probe provider events endpoint to check if missing games exist there
    if probe_events:
        try:
            from .data.adapters.odds_theoddsapi import TheOddsAPIAdapter
            adapter = TheOddsAPIAdapter(region=region)
            all_events_no_date = None
            if probe_no_date:
                try:
                    all_events_no_date = adapter.list_events_no_date()
                except Exception as e:
                    print(f"[yellow]No-date events probe failed:[/yellow] {e}")
            for date_iso in sorted(set(g['date'])):
                try:
                    events = adapter.list_events_by_date(date_iso)
                except Exception as e:
                    print(f"[yellow]Events probe failed for {date_iso}:[/yellow] {e}")
                    continue
                # Build normalized pairs from events
                def _pk(ev):
                    h = str(ev.get('home_team') or '')
                    a = str(ev.get('away_team') or '')
                    return pair_key(h, a)
                ev_pairs = {_pk(ev) for ev in events}
                sched_pairs = set(g[g['date'] == date_iso]['pair'])
                present_pairs = sched_pairs & ev_pairs
                print(f"[bold]{date_iso}[/bold] provider events present: {len(present_pairs)}/{len(sched_pairs)}")
                if len(present_pairs) < len(sched_pairs):
                    missing_pairs = list(sched_pairs - ev_pairs)
                    for pk in missing_pairs[:show]:
                        row = g[(g['date']==date_iso) & (g['pair']==pk)].iloc[0]
                        print(f"  Missing in events: {row['home_team']} vs {row['away_team']}")

                # Optional: union across +/- union_days
                try:
                    ud = int(max(0, union_days))
                except Exception:
                    ud = 0
                if ud:
                    from datetime import date as _date, timedelta as _timedelta
                    d0 = _date.fromisoformat(date_iso)
                    dates = [ (d0 + _timedelta(days=k)).isoformat() for k in range(-ud, ud+1) ]
                    union_events = []
                    for di in dates:
                        try:
                            union_events.extend(adapter.list_events_by_date(di))
                        except Exception:
                            pass
                    union_pairs = {_pk(ev) for ev in union_events}
                    upresent = sched_pairs & union_pairs
                    print(f"  Union +/-{ud} days present: {len(upresent)}/{len(sched_pairs)}")

                # Optional: probe no-date full list filtered locally to target date
                if all_events_no_date:
                    from datetime import datetime as _dt
                    def _ev_date_iso(ev):
                        ct = ev.get('commence_time')
                        if isinstance(ct, str):
                            try:
                                return _dt.fromisoformat(ct.replace('Z','+00:00')).date().isoformat()
                            except Exception:
                                return None
                        return None
                    nd_pairs = {_pk(ev) for ev in all_events_no_date if _ev_date_iso(ev) == date_iso}
                    nd_present = sched_pairs & nd_pairs
                    print(f"  No-date filtered locally present: {len(nd_present)}/{len(sched_pairs)}")
            # Optional: probe alternate sport keys (date-agnostic count)
            if probe_alt_keys:
                try:
                    keys = [k.strip() for k in (alt_keys or "").split(',') if k.strip()]
                    if keys:
                        counts = adapter.try_alternate_sport_keys(keys)
                        print("[bold]Alternate sport key event counts[/bold]:")
                        for k in keys:
                            v = counts.get(k, None)
                            if v is None:
                                print(f"  {k}: n/a")
                            elif v < 0:
                                print(f"  {k}: unavailable (-1)")
                            else:
                                print(f"  {k}: {v}")
                except Exception as e:
                    print(f"[yellow]Alternate key probe failed:[/yellow] {e}")
        except Exception as e:
            print(f"[yellow]Skipping provider events probe:[/yellow] {e}")


# ---------------- Deep Provider Probe (OddsAPI) -----------------

@app.command(name="probe-oddsapi-depth")
def probe_oddsapi_depth(
    games_path: Path = typer.Argument(settings.outputs_dir / "games_curr.csv", help="Scheduled games CSV (date, home_team, away_team)"),
    date: str | None = typer.Option(None, help="ISO date to probe; defaults to unique dates in games"),
    regions: str = typer.Option("us,us2", help="Comma-separated region codes for odds endpoint (passed through as regions param)"),
    markets: str = typer.Option("h2h,spreads,totals,spreads_1st_half,totals_1st_half", help="Markets for odds endpoint"),
    union_days: int = typer.Option(1, help="Union provider events across +/- this many days"),
    use_no_date: bool = typer.Option(True, help="Fetch unfiltered events and odds, then filter locally by date"),
    show: int = typer.Option(10, help="Show up to N missing examples per section"),
):
    """Exhaustive probe of TheOddsAPI for coverage debugging on a date.

    Compares scheduled games to:
      - events endpoint (by date)
      - events endpoint union across +/- union_days
      - unfiltered events filtered locally to date (if enabled)
      - odds endpoint (current) by date for specified markets and regions
      - odds endpoint without date filtered locally to date (if enabled)
    """
    import pandas as _pd
    from datetime import date as _date, timedelta as _timedelta, datetime as _dt
    from .data.team_normalize import pair_key as _pk
    try:
        from .data.adapters.odds_theoddsapi import TheOddsAPIAdapter
    except Exception as e:
        print(f"[red]Adapter import failed:[/red] {e}")
        raise typer.Exit(code=1)

    if not games_path.exists():
        print(f"[red]Games file not found:[/red] {games_path}")
        raise typer.Exit(code=1)
    g = _pd.read_csv(games_path)
    need_cols = {'date','home_team','away_team'}
    if not need_cols.issubset(set(g.columns)):
        print(f"[red]Games CSV missing required columns[/red]: {need_cols - set(g.columns)}")
        raise typer.Exit(code=1)
    g['date'] = _pd.to_datetime(g['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    g['pair'] = g.apply(lambda r: _pk(str(r['home_team']), str(r['away_team'])), axis=1)

    dates = [date] if date else sorted(set(g['date']))
    adapter = TheOddsAPIAdapter(region=regions)
    all_events_no_date = None
    all_odds_no_date_rows = None
    if use_no_date:
        try:
            all_events_no_date = adapter.list_events_no_date()
        except Exception as e:
            print(f"[yellow]No-date events probe failed:[/yellow] {e}")
        try:
            # Pull current odds without date (expanded markets) and filter locally later
            all_rows = list(adapter.iter_current_odds_expanded(markets=markets, date_iso=None))
            all_odds_no_date_rows = all_rows
        except Exception as e:
            print(f"[yellow]No-date odds probe failed:[/yellow] {e}")

    for d in dates:
        print(f"[bold]{d}[/bold] — scheduled games: {len(g[g['date']==d])}")
        sched_pairs = set(g[g['date']==d]['pair'])

        # events by date
        try:
            ev = adapter.list_events_by_date(d)
            ev_pairs = {_pk(str(e.get('home_team') or ''), str(e.get('away_team') or '')) for e in ev}
            present = sched_pairs & ev_pairs
            print(f"  events(date): {len(present)}/{len(sched_pairs)}")
            if len(present) < len(sched_pairs):
                miss = list(sched_pairs - ev_pairs)
                for pk in miss[:show]:
                    row = g[(g['date']==d)&(g['pair']==pk)].iloc[0]
                    print(f"    missing: {row['home_team']} vs {row['away_team']}")
        except Exception as e:
            print(f"  events(date) failed: {e}")

        # union across +/- union_days
        if union_days:
            try:
                d0 = _date.fromisoformat(d)
                udates = [ (d0 + _timedelta(days=k)).isoformat() for k in range(-union_days, union_days+1) ]
                uev = []
                for di in udates:
                    try:
                        uev.extend(adapter.list_events_by_date(di))
                    except Exception:
                        pass
                u_pairs = {_pk(str(e.get('home_team') or ''), str(e.get('away_team') or '')) for e in uev}
                upresent = sched_pairs & u_pairs
                print(f"  events(union +/-{union_days}): {len(upresent)}/{len(sched_pairs)}")
            except Exception as e:
                print(f"  events(union) failed: {e}")

        # events no-date filtered locally
        if all_events_no_date:
            def _ev_date_iso(ev):
                ct = ev.get('commence_time')
                if isinstance(ct, str):
                    try:
                        return _dt.fromisoformat(ct.replace('Z','+00:00')).date().isoformat()
                    except Exception:
                        return None
                return None
            nd_pairs = {_pk(str(e.get('home_team') or ''), str(e.get('away_team') or '')) for e in all_events_no_date if _ev_date_iso(e) == d}
            nd_present = sched_pairs & nd_pairs
            print(f"  events(no-date filtered): {len(nd_present)}/{len(sched_pairs)}")

        # odds by date (expanded markets)
        try:
            rows = list(adapter.iter_current_odds_expanded(markets=markets, date_iso=d))
            # Build pair set from rows
            o_pairs = set()
            for r in rows:
                h = getattr(r, 'home_team_name', None)
                a = getattr(r, 'away_team_name', None)
                o_pairs.add(_pk(str(h or ''), str(a or '')))
            opresent = sched_pairs & o_pairs
            print(f"  odds(date, {regions}): {len(opresent)}/{len(sched_pairs)} (rows: {len(rows)})")
        except Exception as e:
            print(f"  odds(date) failed: {e}")

        # odds no-date filtered locally
        if all_odds_no_date_rows is not None:
            def _row_date_iso(r):
                ct = getattr(r, 'commence_time', None)
                if ct is None:
                    return None
                try:
                    return ct.date().isoformat()
                except Exception:
                    return None
            filt = [r for r in all_odds_no_date_rows if _row_date_iso(r) == d]
            o_pairs2 = set()
            for r in filt:
                h = getattr(r, 'home_team_name', None)
                a = getattr(r, 'away_team_name', None)
                o_pairs2.add(_pk(str(h or ''), str(a or '')))
            opresent2 = sched_pairs & o_pairs2
            print(f"  odds(no-date filtered, {regions}): {len(opresent2)}/{len(sched_pairs)} (rows: {len(filt)})")

# (moved app() invocation to end of file to ensure all commands are registered before run)

@app.command(name="synthetic-e2e")
def synthetic_e2e(
    date: str | None = typer.Option(None, help="Target date YYYY-MM-DD (default: today UTC)"),
    n_games: int = typer.Option(8, help="Number of synthetic games to generate"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    conference_seg: bool = typer.Option(True, help="Simulate conference segmentation by assigning conferences"),
    calibrate: bool = typer.Option(True, help="Apply simple synthetic calibration to raw predictions"),
    stake_sheet: bool = typer.Option(True, help="Generate synthetic stake sheet with edges"),
    min_cal_share_total: float = typer.Option(0.6, help="Minimum calibrated share (totals) for pass when calibrate=true"),
    min_cal_share_margin: float = typer.Option(0.6, help="Minimum calibrated share (margin) for pass when calibrate=true"),
    min_rows: int = typer.Option(4, help="Minimum number of games required"),
    max_mean_edge_total: float = typer.Option(20.0, help="Upper bound on mean total edge to catch runaway calibration"),
):
    """Synthetic end-to-end harness (segmentation + calibration + stake sizing).

    Writes artifacts into outputs/ without external API calls:
      games_synth_<date>.csv, odds_synth_<date>.csv, predictions_synthetic_<date>.csv,
      predictions_unified_enriched_synthetic_<date>.csv, stake_sheet_synthetic_<date>.csv (optional),
      synthetic_e2e_<date>.json summary.
    """
    import datetime as _dt, json as _json, pandas as _pd, numpy as _np
    from .config import settings as _settings
    rng = _np.random.default_rng(seed)
    d_iso = ( _dt.date.fromisoformat(date) if date else _dt.datetime.utcnow().date() ).isoformat()
    out_dir = _settings.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    teams = [
        "Duke","Kansas","Gonzaga","UConn","Arizona","Purdue","Houston","Tennessee","Baylor","North Carolina",
        "Illinois","Alabama","Creighton","Auburn","San Diego St","Iowa State","Texas","Marquette","Arkansas","Wisconsin"
    ]
    confs = ["ACC","B12","BE","SEC","P12","B10"] if conference_seg else [None]
    conf_map = {t: (confs[i % len(confs)] if conference_seg else None) for i, t in enumerate(teams)}
    games = []
    used: set[str] = set()
    for i in range(n_games * 2):
        if len(games) >= n_games: break
        h, a = rng.choice(teams, size=2, replace=False)
        key = "|".join(sorted([h,a]))
        if key in used: continue
        used.add(key)
        games.append({"game_id": f"SYN{i:03d}", "date": d_iso, "home_team": h, "away_team": a, "home_conference": conf_map[h], "away_conference": conf_map[a]})
    df_games = _pd.DataFrame(games)
    (out_dir / f"games_synth_{d_iso}.csv").write_text(df_games.to_csv(index=False))

    market_total = rng.normal(146.0, 8.0, size=len(df_games)).round(1)
    spread_home = rng.normal(-2.0, 7.0, size=len(df_games)).round(1)
    df_odds = _pd.DataFrame({"game_id": df_games.game_id, "date": d_iso, "market_total": market_total, "spread_home": spread_home})
    (out_dir / f"odds_synth_{d_iso}.csv").write_text(df_odds.to_csv(index=False))

    pred_total_raw = market_total + rng.normal(0,5.0,size=len(df_games))
    pred_margin_raw = (-spread_home) + rng.normal(0,3.5,size=len(df_games))
    df_pred = _pd.DataFrame({
        "game_id": df_games.game_id,
        "date": d_iso,
        "home_team": df_games.home_team,
        "away_team": df_games.away_team,
        "pred_total_raw": pred_total_raw,
        "pred_margin_raw": pred_margin_raw,
        "market_total": market_total,
        "spread_home": spread_home,
    })
    if conference_seg:
        adj = df_games.home_conference.map(lambda c: (hash(str(c)) % 7 - 3) * 0.25 if c else 0.0)
        df_pred["pred_total_seg"] = df_pred["pred_total_raw"] + adj
    else:
        df_pred["pred_total_seg"] = df_pred["pred_total_raw"]
    if calibrate:
        mt_mean = float(_np.mean(market_total))
        mm_mean = float(_np.mean(pred_margin_raw))
        df_pred["pred_total_cal"] = (df_pred["pred_total_seg"] - mt_mean) * 0.92 + mt_mean
        df_pred["pred_margin_cal"] = (df_pred["pred_margin_raw"] - mm_mean) * 0.90 + mm_mean
        df_pred["pred_total_basis"] = "cal"
        df_pred["pred_margin_basis"] = "cal"
    else:
        df_pred["pred_total_cal"] = _pd.NA
        df_pred["pred_margin_cal"] = _pd.NA
        df_pred["pred_total_basis"] = "raw"
        df_pred["pred_margin_basis"] = "raw"
    preds_path = out_dir / f"predictions_synthetic_{d_iso}.csv"
    preds_path.write_text(df_pred.to_csv(index=False))
    enriched_path = out_dir / f"predictions_unified_enriched_synthetic_{d_iso}.csv"
    enriched_path.write_text(df_pred.to_csv(index=False))
    stake_path = None
    if stake_sheet:
        df_pred["edge_total"] = (df_pred["pred_total_cal"].fillna(df_pred["pred_total_raw"]) - df_pred["market_total"]).abs()
        stake_df = df_pred.sort_values("edge_total", ascending=False).head(max(5, len(df_pred)//2))[ ["game_id","home_team","away_team","pred_total_cal","market_total","edge_total"] ]
        stake_path = out_dir / f"stake_sheet_synthetic_{d_iso}.csv"
        stake_path.write_text(stake_df.to_csv(index=False))
    summary = {
        "date": d_iso,
        "n_games": len(df_games),
        "preds_path": str(preds_path),
        "enriched_path": str(enriched_path),
        "stake_path": str(stake_path) if stake_path else None,
        "cal_share_total": float((df_pred["pred_total_basis"].astype(str)=="cal").mean()),
        "cal_share_margin": float((df_pred["pred_margin_basis"].astype(str)=="cal").mean()),
        "mean_pred_total": float(df_pred["pred_total_raw"].mean()),
        "mean_pred_margin": float(df_pred["pred_margin_raw"].mean()),
        "mean_edge_total": float(df_pred.get("edge_total").mean()) if "edge_total" in df_pred.columns else None,
    }
    (out_dir / f"synthetic_e2e_{d_iso}.json").write_text(_json.dumps(summary, indent=2))
    print(_json.dumps(summary, indent=2))
    # CI thresholds
    if len(df_games) < min_rows:
        raise typer.Exit(code=2)
    if calibrate:
        if summary["cal_share_total"] is not None and summary["cal_share_total"] < min_cal_share_total:
            raise typer.Exit(code=3)
        if summary["cal_share_margin"] is not None and summary["cal_share_margin"] < min_cal_share_margin:
            raise typer.Exit(code=4)
    if summary.get("mean_edge_total") is not None and abs(summary["mean_edge_total"]) > max_mean_edge_total:
        raise typer.Exit(code=5)

if __name__ == "__main__":
    app()

