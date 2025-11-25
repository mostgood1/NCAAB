import argparse
import datetime as dt
from pathlib import Path
import pandas as pd

OUT = Path("outputs")

REASONS = [
    "ok_calibrated",
    "missing_model_prediction",
    "missing_calibration_artifact",
    "basis_not_cal",
]

def _safe_read(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def diagnose(date: str | None) -> pd.DataFrame:
    # Resolve date (default today string)
    if not date:
        date = dt.datetime.utcnow().strftime("%Y-%m-%d")
    games = _safe_read(OUT / "games_curr.csv")
    preds_unified = _safe_read(OUT / f"predictions_unified_{date}.csv")
    preds_model = _safe_read(OUT / f"predictions_model_{date}.csv")
    # Fallback broad files if specific date files missing
    if preds_unified.empty:
        preds_unified = _safe_read(OUT / "predictions_unified.csv")
    if preds_model.empty:
        preds_model = _safe_read(OUT / "predictions_model.csv")

    for df in (games, preds_unified, preds_model):
        if not df.empty and "game_id" in df.columns:
            df["game_id"] = df["game_id"].astype(str)
        if not df.empty and "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    if not games.empty and "game_id" in games.columns:
        if "date" in games.columns:
            games = games[games["date"] == date]
    game_ids = games["game_id"].astype(str).tolist() if not games.empty and "game_id" in games.columns else []
    rows = []
    for gid in game_ids:
        row = {"game_id": gid}
        if not games.empty:
            gsub = games[games.game_id.astype(str) == gid]
            if not gsub.empty:
                for col in ["home_team", "away_team", "date", "start_time"]:
                    if col in gsub.columns:
                        row[col] = gsub.iloc[0][col]
        model_sub = preds_model[preds_model.game_id.astype(str) == gid] if (not preds_model.empty and "game_id" in preds_model.columns) else pd.DataFrame()
        unified_sub = preds_unified[preds_unified.game_id.astype(str) == gid] if (not preds_unified.empty and "game_id" in preds_unified.columns) else pd.DataFrame()
        has_model_total = (not model_sub.empty) and any(c in model_sub.columns for c in ["pred_total_model", "pred_total"])
        has_model_margin = (not model_sub.empty) and any(c in model_sub.columns for c in ["pred_margin_model", "pred_margin"])
        has_cal_total = (not unified_sub.empty) and ("pred_total_calibrated" in unified_sub.columns) and pd.to_numeric(unified_sub["pred_total_calibrated"], errors="coerce").notna().any()
        has_cal_margin = (not unified_sub.empty) and ("pred_margin_calibrated" in unified_sub.columns) and pd.to_numeric(unified_sub["pred_margin_calibrated"], errors="coerce").notna().any()
        basis_total = None
        basis_margin = None
        if not unified_sub.empty:
            if "pred_total_basis" in unified_sub.columns:
                basis_total = str(unified_sub.iloc[0]["pred_total_basis"])
            if "pred_margin_basis" in unified_sub.columns:
                basis_margin = str(unified_sub.iloc[0]["pred_margin_basis"])

        # Determine reason priority
        if has_cal_total and has_cal_margin and basis_total == "cal" and basis_margin == "cal":
            reason = "ok_calibrated"
        elif not has_model_total or not has_model_margin:
            reason = "missing_model_prediction"
        elif not has_cal_total or not has_cal_margin:
            reason = "missing_calibration_artifact"
        elif (basis_total and basis_total != "cal") or (basis_margin and basis_margin != "cal"):
            reason = "basis_not_cal"
        else:
            reason = "missing_calibration_artifact"

        row["reason"] = reason
        row["has_model_total"] = has_model_total
        row["has_model_margin"] = has_model_margin
        row["has_cal_total"] = has_cal_total
        row["has_cal_margin"] = has_cal_margin
        row["basis_total"] = basis_total
        row["basis_margin"] = basis_margin
        rows.append(row)

    df = pd.DataFrame(rows)
    # Summary counts
    summary = df.groupby("reason").size().rename("count").reset_index() if not df.empty else pd.DataFrame(columns=["reason","count"])
    out_summary = OUT / f"calibration_diagnostic_summary_{date}.csv"
    out_detail = OUT / f"calibration_diagnostic_detail_{date}.csv"
    try:
        if not summary.empty:
            summary.to_csv(out_summary, index=False)
        if not df.empty:
            df.to_csv(out_detail, index=False)
    except Exception:
        pass
    return df


def main():
    parser = argparse.ArgumentParser(description="Diagnose calibrated prediction coverage for a given date.")
    parser.add_argument("--date", dest="date", default=None, help="Date YYYY-MM-DD (default today UTC)")
    args = parser.parse_args()
    diag_df = diagnose(args.date)
    if diag_df.empty:
        print("No games found for date", args.date or dt.datetime.utcnow().strftime("%Y-%m-%d"))
    else:
        print(diag_df.head(25).to_string(index=False))
        print("---")
        print("Reason counts:")
        print(diag_df.groupby("reason").size())

if __name__ == "__main__":
    main()
