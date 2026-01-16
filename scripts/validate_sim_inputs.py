import json
import os
import re
import sys
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    ok: bool
    severity: str  # 'info' | 'warn' | 'error'
    message: str
    details: dict


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            return pd.DataFrame()


def _as_iso_date(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        return series.astype(str).str.slice(0, 10)


def _pct(n: float, d: float) -> float:
    return float(0.0 if d <= 0 else 100.0 * (n / d))


def validate(date: str, out_dir: Path) -> tuple[dict, int]:
    out_dir = Path(out_dir)
    preds_path = out_dir / f"predictions_unified_enriched_{date}.csv"
    disp_path = out_dir / f"predictions_display_{date}.csv"
    feat_path = out_dir / f"features_{date}.csv"
    simq_path = out_dir / f"sim_quantiles_{date}.csv"

    results: list[CheckResult] = []

    preds = _read_csv(preds_path)
    disp = _read_csv(disp_path)
    feats = _read_csv(feat_path)

    # 1) Presence
    for name, path in [
        ("predictions_unified_enriched", preds_path),
        ("predictions_display", disp_path),
        ("features", feat_path),
    ]:
        results.append(
            CheckResult(
                ok=path.exists(),
                severity="error" if not path.exists() else "info",
                message=f"{name} file {'present' if path.exists() else 'missing'}",
                details={"path": str(path)},
            )
        )

    if preds.empty:
        results.append(
            CheckResult(
                ok=False,
                severity="error",
                message="predictions_unified_enriched is empty; cannot validate sim inputs",
                details={"path": str(preds_path)},
            )
        )
        payload = {
            "date": date,
            "out_dir": str(out_dir),
            "checks": [asdict(r) for r in results],
        }
        return payload, 2

    # Filter preds to date when column exists
    if "date" in preds.columns:
        try:
            preds = preds[_as_iso_date(preds["date"]) == str(date)].copy()
        except Exception:
            pass

    n_games = int(len(preds))

    # 2) Required columns
    req_cols = ["game_id", "home_team", "away_team"]
    missing_cols = [c for c in req_cols if c not in preds.columns]
    results.append(
        CheckResult(
            ok=(len(missing_cols) == 0),
            severity="error" if missing_cols else "info",
            message="required columns present" if not missing_cols else "missing required columns",
            details={"missing": missing_cols},
        )
    )

    # 2b) 1H market lines (optional but needed for 1H market probabilities)
    try:
        mt1_nonnull = int(pd.to_numeric(preds.get("market_total_1h"), errors="coerce").notna().sum()) if "market_total_1h" in preds.columns else 0
        sp1_nonnull = int(pd.to_numeric(preds.get("spread_home_1h"), errors="coerce").notna().sum()) if "spread_home_1h" in preds.columns else 0
        mt1_ok = bool(("market_total_1h" in preds.columns) and (mt1_nonnull >= int(0.5 * len(preds))))
        sp1_ok = bool(("spread_home_1h" in preds.columns) and (sp1_nonnull >= int(0.5 * len(preds))))

        mt_full_ok = bool(("market_total" in preds.columns) and (pd.to_numeric(preds.get("market_total"), errors="coerce").notna().sum() >= int(0.5 * len(preds))))
        sp_full_ok = bool(("spread_home" in preds.columns) and (pd.to_numeric(preds.get("spread_home"), errors="coerce").notna().sum() >= int(0.5 * len(preds))))
        derivable = bool(mt_full_ok and sp_full_ok)

        ok = bool((mt1_ok and sp1_ok) or derivable)
        severity = "info" if ok else "warn"
        if mt1_ok and sp1_ok:
            msg = "1H market lines present for most games"
        elif derivable:
            msg = "1H market lines missing in enriched preds (can derive from full-game lines for sims)"
        else:
            msg = "1H market lines missing for many games and full-game lines insufficient (1H sim market probs may be null)"

        results.append(
            CheckResult(
                ok=ok,
                severity=severity,
                message=msg,
                details={
                    "market_total_1h_present": bool("market_total_1h" in preds.columns),
                    "spread_home_1h_present": bool("spread_home_1h" in preds.columns),
                    "market_total_1h_nonnull": mt1_nonnull,
                    "spread_home_1h_nonnull": sp1_nonnull,
                    "market_total_full_ok": bool(mt_full_ok),
                    "spread_home_full_ok": bool(sp_full_ok),
                    "rows": int(len(preds)),
                },
            )
        )
    except Exception:
        pass

    # 3) Duplicates
    if "game_id" in preds.columns:
        gid = preds["game_id"].astype(str)
        dup = int(gid.duplicated().sum())
        results.append(
            CheckResult(
                ok=(dup == 0),
                severity="warn" if dup else "info",
                message=f"duplicate game_id rows: {dup}",
                details={"rows": n_games, "duplicate_game_id_rows": dup},
            )
        )

    # 4) Features file: join coverage + tempo/off/def plausibility
    if feats.empty:
        results.append(
            CheckResult(
                ok=False,
                severity="warn",
                message="features_<date>.csv missing/empty; pace sim may fall back",
                details={"path": str(feat_path)},
            )
        )
    else:
        feats_work = feats.copy()
        # Filter features to date when possible
        if "date" in feats_work.columns:
            try:
                feats_work = feats_work[_as_iso_date(feats_work["date"]) == str(date)].copy()
            except Exception:
                pass

        # Join coverage by game_id
        join_ok = True
        join_details: dict = {}
        if "game_id" in preds.columns and "game_id" in feats_work.columns:
            try:
                pg = preds["game_id"].astype(str)
                fg = feats_work["game_id"].astype(str)
                pred_ids = set(pg.dropna().tolist())
                feat_ids = set(fg.dropna().tolist())
                inter = pred_ids & feat_ids
                missing = pred_ids - feat_ids
                coverage = float(len(inter)) / float(max(1, len(pred_ids)))
                join_ok = coverage >= 0.95
                join_details = {
                    "pred_game_ids": int(len(pred_ids)),
                    "feature_game_ids": int(len(feat_ids)),
                    "matched": int(len(inter)),
                    "missing_in_features": int(len(missing)),
                    "coverage_pct": float(round(100.0 * coverage, 2)),
                    "sample_missing_game_ids": list(sorted(list(missing)))[:5],
                }
            except Exception as e:
                join_ok = False
                join_details = {"error": str(e)}
        else:
            join_ok = False
            join_details = {
                "pred_has_game_id": bool("game_id" in preds.columns),
                "features_has_game_id": bool("game_id" in feats_work.columns),
            }
        results.append(
            CheckResult(
                ok=join_ok,
                severity="warn" if not join_ok else "info",
                message="features join coverage by game_id >= 95%" if join_ok else "features join coverage by game_id is low/missing",
                details=join_details,
            )
        )

        # Tempo ratings plausibility (source of truth: features file)
        tempo_cols = [c for c in ["home_tempo_rating", "away_tempo_rating", "tempo_rating_sum"] if c in feats_work.columns]
        if tempo_cols:
            checked = 0
            missing_vals = 0
            out_of_range = 0
            # NCAA possessions per 40 typically ~60-80; allow wider.
            for col in ["home_tempo_rating", "away_tempo_rating"]:
                if col in feats_work.columns:
                    vals = pd.to_numeric(feats_work[col], errors="coerce")
                    checked += int(len(vals))
                    missing_vals += int(vals.isna().sum())
                    out_of_range += int(((vals < 50.0) | (vals > 90.0)).where(vals.notna(), False).sum())
            tempo_ok = (out_of_range == 0) and (missing_vals <= int(0.05 * max(1, checked)))
            results.append(
                CheckResult(
                    ok=tempo_ok,
                    severity="warn" if not tempo_ok else "info",
                    message="tempo ratings plausible and mostly present" if tempo_ok else "tempo ratings missing/out of range",
                    details={
                        "columns": tempo_cols,
                        "rows": int(len(feats_work)),
                        "checked": int(checked),
                        "missing": int(missing_vals),
                        "missing_pct": float(round(_pct(missing_vals, checked), 2)) if checked else None,
                        "out_of_range": int(out_of_range),
                    },
                )
            )
        else:
            results.append(
                CheckResult(
                    ok=False,
                    severity="warn",
                    message="features file lacks tempo columns required for pace sim",
                    details={"available_columns": list(feats_work.columns)[:50]},
                )
            )

        # Pace mean/sigma (new) - ensures pace-based sim uses game-specific inputs.
        try:
            pace_cols = [c for c in ["pace_game_est", "pace_sigma_game_est"] if c in feats_work.columns]
            if pace_cols:
                pace_est = pd.to_numeric(feats_work.get("pace_game_est"), errors="coerce") if "pace_game_est" in feats_work.columns else pd.Series(dtype=float)
                pace_sig = pd.to_numeric(feats_work.get("pace_sigma_game_est"), errors="coerce") if "pace_sigma_game_est" in feats_work.columns else pd.Series(dtype=float)

                nonnull_est = int(pace_est.notna().sum()) if len(pace_est) else 0
                nonnull_sig = int(pace_sig.notna().sum()) if len(pace_sig) else 0
                out_of_range_est = int(((pace_est < 50.0) | (pace_est > 90.0)).where(pace_est.notna(), False).sum()) if len(pace_est) else 0
                out_of_range_sig = int(((pace_sig < 0.5) | (pace_sig > 12.0)).where(pace_sig.notna(), False).sum()) if len(pace_sig) else 0

                ok = bool(
                    ("pace_game_est" in feats_work.columns and nonnull_est >= int(0.9 * len(feats_work)) and out_of_range_est == 0)
                    and ("pace_sigma_game_est" in feats_work.columns and nonnull_sig >= int(0.9 * len(feats_work)) and out_of_range_sig == 0)
                )
                results.append(
                    CheckResult(
                        ok=ok,
                        severity="info" if ok else "warn",
                        message="pace_game_est/pace_sigma_game_est present and plausible" if ok else "pace mean/sigma missing or out of range (pace sim may be less accurate)",
                        details={
                            "columns": pace_cols,
                            "rows": int(len(feats_work)),
                            "pace_game_est_nonnull": nonnull_est,
                            "pace_sigma_game_est_nonnull": nonnull_sig,
                            "pace_game_est_out_of_range": out_of_range_est,
                            "pace_sigma_game_est_out_of_range": out_of_range_sig,
                        },
                    )
                )
            else:
                results.append(
                    CheckResult(
                        ok=False,
                        severity="warn",
                        message="pace columns missing from features_<date>.csv (pace sim will fall back to tempo/default sigma)",
                        details={"expected": ["pace_game_est", "pace_sigma_game_est"], "present": list(feats_work.columns)},
                    )
                )
        except Exception:
            pass

        # Off/def ratings plausibility (basic sanity only)
        rating_cols = [c for c in ["home_off_rating", "away_off_rating", "home_def_rating", "away_def_rating"] if c in feats_work.columns]
        if rating_cols:
            bad = 0
            missing = 0
            for c in rating_cols:
                v = pd.to_numeric(feats_work[c], errors="coerce")
                missing += int(v.isna().sum())
                # Ratings are typically around ~90-120; allow wider.
                bad += int(((v < 70.0) | (v > 140.0)).where(v.notna(), False).sum())
            ok_r = (bad == 0) and (missing <= int(0.05 * max(1, len(feats_work) * len(rating_cols))))
            results.append(
                CheckResult(
                    ok=ok_r,
                    severity="warn" if not ok_r else "info",
                    message="off/def ratings plausible and mostly present" if ok_r else "off/def ratings missing/out of range",
                    details={"columns": rating_cols, "rows": int(len(feats_work)), "missing": int(missing), "bad": int(bad)},
                )
            )

        # Team name consistency sanity: sample mismatches between preds and features for matched game_ids
        try:
            if {"game_id", "home_team", "away_team"}.issubset(preds.columns) and {"game_id", "home_team", "away_team"}.issubset(feats_work.columns):
                p2 = preds[["game_id", "home_team", "away_team"]].copy()
                f2 = feats_work[["game_id", "home_team", "away_team"]].copy()
                p2["game_id"] = p2["game_id"].astype(str)
                f2["game_id"] = f2["game_id"].astype(str)
                j = p2.merge(f2, on="game_id", how="inner", suffixes=("_pred", "_feat"))
                if not j.empty:
                    def _norm(s: pd.Series) -> pd.Series:
                        out = s.astype(str).str.strip().str.lower()
                        # Treat common parenthetical qualifiers as formatting (e.g., "(PA)")
                        out = out.str.replace(r"\([^)]*\)", "", regex=True)
                        out = out.str.replace(r"[^a-z0-9 ]+", " ", regex=True)
                        out = out.str.replace(r"\s+", " ", regex=True).str.strip()
                        return out
                    mism = (
                        (_norm(j["home_team_pred"]) != _norm(j["home_team_feat"]))
                        | (_norm(j["away_team_pred"]) != _norm(j["away_team_feat"]))
                    )
                    n_mism = int(mism.sum())
                    ok_names = n_mism == 0
                    sample = j.loc[mism, ["game_id", "home_team_pred", "home_team_feat", "away_team_pred", "away_team_feat"]].head(5).to_dict(orient="records")
                    results.append(
                        CheckResult(
                            ok=ok_names,
                            severity="warn" if not ok_names else "info",
                            message="team names match between preds and features" if ok_names else "team name mismatch between preds and features",
                            details={"mismatches": n_mism, "sample": sample},
                        )
                    )
        except Exception:
            pass

    # 5) Sigma sanity
    sig_cols = [c for c in ["sigma_total_adj", "sigma_total", "interval_total_std", "sigma_margin_adj", "sigma_margin", "interval_margin_std"] if c in preds.columns]
    if sig_cols:
        bad = 0
        for c in sig_cols:
            v = pd.to_numeric(preds[c], errors="coerce")
            # totals sigma usually ~8-25; margin sigma ~8-20. Allow wider, flag extreme.
            bad += int(((v <= 0) | (v > 40)).where(v.notna(), False).sum())
        results.append(
            CheckResult(
                ok=(bad == 0),
                severity="warn" if bad else "info",
                message="sigma fields positive and not extreme" if bad == 0 else "sigma fields contain invalid/extreme values",
                details={"sigma_columns": sig_cols, "bad_values": bad},
            )
        )

    # 6) Mean sanity (NCAAB totals roughly 100-180; allow wider)
    mean_cols = [c for c in ["pred_total_calibrated", "pred_total", "pred_margin_calibrated", "pred_margin"] if c in preds.columns]
    if mean_cols:
        bad_mean = 0
        if "pred_total" in preds.columns or "pred_total_calibrated" in preds.columns:
            ct = "pred_total_calibrated" if "pred_total_calibrated" in preds.columns else "pred_total"
            t = pd.to_numeric(preds[ct], errors="coerce")
            bad_mean += int(((t < 80) | (t > 220)).where(t.notna(), False).sum())
        if "pred_margin" in preds.columns or "pred_margin_calibrated" in preds.columns:
            cm = "pred_margin_calibrated" if "pred_margin_calibrated" in preds.columns else "pred_margin"
            m = pd.to_numeric(preds[cm], errors="coerce")
            bad_mean += int(((m < -40) | (m > 40)).where(m.notna(), False).sum())
        results.append(
            CheckResult(
                ok=(bad_mean == 0),
                severity="warn" if bad_mean else "info",
                message="mean totals/margins within plausible bounds" if bad_mean == 0 else "mean totals/margins contain outliers",
                details={"bad_values": bad_mean},
            )
        )

    # 7) Display snapshot row parity
    if not disp.empty:
        results.append(
            CheckResult(
                ok=(len(disp) == len(preds)),
                severity="warn" if len(disp) != len(preds) else "info",
                message=f"display rows={len(disp)} vs enriched rows={len(preds)}",
                details={"display_rows": int(len(disp)), "enriched_rows": int(len(preds))},
            )
        )

    # 8) Sim artifact existence (optional)
    if simq_path.exists():
        sq = _read_csv(simq_path)
        results.append(
            CheckResult(
                ok=(not sq.empty),
                severity="warn" if sq.empty else "info",
                message="sim_quantiles exists" if not sq.empty else "sim_quantiles is empty",
                details={"path": str(simq_path), "rows": int(len(sq))},
            )
        )

    # Decide exit code
    strict = str(os.environ.get("STRICT_FEATURE_VALIDATION", "0")).strip().lower() in {"1", "true", "yes"}
    errors = [r for r in results if r.severity == "error" and not r.ok]
    warns = [r for r in results if r.severity == "warn" and not r.ok]
    exit_code = 0
    if errors:
        exit_code = 2
    elif strict and warns:
        exit_code = 3

    payload = {
        "date": date,
        "out_dir": str(out_dir),
        "rows": {"enriched": int(len(preds)), "display": int(len(disp)) if not disp.empty else 0, "features": int(len(feats)) if not feats.empty else 0},
        "checks": [asdict(r) for r in results],
        "exit_code": exit_code,
        "strict": strict,
    }
    return payload, exit_code


def main() -> int:
    # Backwards compatible CLI:
    #   python validate_sim_inputs.py <YYYY-MM-DD> [outputs_dir]
    # Also supports:
    #   python validate_sim_inputs.py --date <YYYY-MM-DD> --out-dir <dir>
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("date", nargs="?", help="Date YYYY-MM-DD (positional)")
    p.add_argument("out_dir", nargs="?", default="outputs", help="Outputs dir (positional)")
    p.add_argument("--date", dest="date_flag", help="Date YYYY-MM-DD")
    p.add_argument("--out-dir", dest="out_dir_flag", help="Outputs dir")

    ns = p.parse_args()
    date = (ns.date_flag or ns.date or "").strip()
    out_dir = Path((ns.out_dir_flag or ns.out_dir or "outputs")).expanduser()

    if not date:
        print(json.dumps({"error": "Usage: validate_sim_inputs.py <YYYY-MM-DD> [outputs_dir] OR --date <YYYY-MM-DD> --out-dir <dir>"}))
        return 2

    payload, code = validate(date, out_dir)

    # Persist a diagnostic artifact for upload/debugging
    try:
        out_path = Path(out_dir) / f"sim_inputs_diagnostic_{date}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    print(json.dumps(payload))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
