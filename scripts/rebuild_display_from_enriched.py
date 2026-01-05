import argparse
import datetime as dt
from pathlib import Path

import pandas as pd


def classify_basis(series: pd.Series, default: str = "model") -> pd.Series:
    # Minimal basis classifier: mark non-null predictions as model-based
    out = pd.Series([None] * len(series), index=series.index, dtype=object)
    mask = series.notna()
    out.loc[mask] = default
    return out


def main(date_str: str | None) -> None:
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    if not date_str:
        date_str = dt.datetime.utcnow().date().isoformat()

    disp_path = out_dir / f"predictions_display_{date_str}.csv"
    enr_path = out_dir / f"predictions_unified_enriched_{date_str}.csv"
    edges_path = out_dir / f"align_period_{date_str}_edges.csv"

    if not disp_path.exists():
        raise FileNotFoundError(f"Display file not found: {disp_path}")
    if not enr_path.exists() and not edges_path.exists():
        raise FileNotFoundError(
            f"Neither enriched nor edges file found for {date_str}: {enr_path} | {edges_path}"
        )

    df = pd.read_csv(disp_path)

    # Normalize game_id type if present
    if "game_id" in df.columns:
        try:
            df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
        except Exception:
            pass

    # Try enriched join first, then edges
    joined_cols = ["market_total", "spread_home", "total", "closing_spread_home"]
    join_df = None
    if enr_path.exists():
        jdf = pd.read_csv(enr_path)
        # Ensure game_id numeric for join
        if "game_id" in jdf.columns:
            jdf["game_id"] = pd.to_numeric(jdf["game_id"], errors="coerce")
        join_df = jdf[[c for c in joined_cols if c in jdf.columns] + (["game_id"] if "game_id" in jdf.columns else [])]
    elif edges_path.exists():
        jdf = pd.read_csv(edges_path)
        if "game_id" in jdf.columns:
            jdf["game_id"] = pd.to_numeric(jdf["game_id"], errors="coerce")
        join_df = jdf[[c for c in joined_cols if c in jdf.columns] + (["game_id"] if "game_id" in jdf.columns else [])]

    if join_df is not None and "game_id" in df.columns and "game_id" in join_df.columns:
        df = df.merge(join_df, on="game_id", how="left", suffixes=("", "_j"))

        # Map market_total: prefer explicit market_total, else fallback to total
        if "market_total" not in df.columns:
            if "market_total_j" in df.columns:
                df["market_total"] = df["market_total_j"]
            elif "total" in df.columns:
                df["market_total"] = df["total"]
            elif "total_j" in df.columns:
                df["market_total"] = df["total_j"]

        # Map spread_home: prefer spread_home, else fallback to closing_spread_home
        if "spread_home" not in df.columns:
            if "spread_home_j" in df.columns:
                df["spread_home"] = df["spread_home_j"]
            elif "closing_spread_home" in df.columns:
                df["spread_home"] = df["closing_spread_home"]
            elif "closing_spread_home_j" in df.columns:
                df["spread_home"] = df["closing_spread_home_j"]

        # Drop helper columns
        for c in list(df.columns):
            if c.endswith("_j") or c == "total" or c == "closing_spread_home":
                # keep market_total/spread_home targets, drop the rest
                if c not in ("market_total", "spread_home"):
                    df.drop(columns=[c], inplace=True)

    # Fill basis labels minimally if missing
    if "pred_total_basis" not in df.columns:
        df["pred_total_basis"] = classify_basis(df["pred_total"]) if "pred_total" in df.columns else None
    if "pred_margin_basis" not in df.columns:
        df["pred_margin_basis"] = classify_basis(df["pred_margin"]) if "pred_margin" in df.columns else None

    # Write back
    df.to_csv(disp_path, index=False)
    print(f"[ok] Rebuilt display with odds+basis: {disp_path} rows={len(df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild display CSV by enriching with market odds and basis labels")
    ap.add_argument("--date", dest="date", default=None, help="Date YYYY-MM-DD; defaults to today UTC")
    args = ap.parse_args()
    main(args.date)
import argparse
from pathlib import Path
import pandas as pd


def choose_display_margin(df: pd.DataFrame) -> pd.DataFrame:
    pm = pd.to_numeric(df.get('pred_margin'), errors='coerce') if 'pred_margin' in df.columns else pd.Series([None]*len(df))
    basis_m = df.get('pred_margin_basis') if 'pred_margin_basis' in df.columns else pd.Series([None]*len(df))
    used = pd.Series([False]*len(df))
    if 'pred_margin_calibrated' in df.columns:
        pm_cal = pd.to_numeric(df['pred_margin_calibrated'], errors='coerce')
        mask = pm_cal.notna()
        pm[mask] = pm_cal[mask]
        basis_m = basis_m.where(~mask, 'cal')
        used = used | mask
    if 'pred_margin_model' in df.columns:
        pm_mod = pd.to_numeric(df['pred_margin_model'], errors='coerce')
        mask = (~used) & pm_mod.notna()
        pm[mask] = pm_mod[mask]
        basis_m = basis_m.where(~mask, df.get('pred_margin_model_basis','model'))
        used = used | mask
    if 'pred_margin_blend' in df.columns:
        pm_blend = pd.to_numeric(df['pred_margin_blend'], errors='coerce')
        mask = (~used) & pm_blend.notna()
        pm[mask] = pm_blend[mask]
        basis_m = basis_m.where(~mask, 'blend')
        used = used | mask
    if 'pred_margin_seg' in df.columns:
        pm_seg = pd.to_numeric(df['pred_margin_seg'], errors='coerce')
        mask = (~used) & pm_seg.notna()
        pm[mask] = pm_seg[mask]
        basis_m = basis_m.where(~mask, 'seg')
        used = used | mask
    sh = None
    if 'spread_home' in df.columns:
        sh = pd.to_numeric(df['spread_home'], errors='coerce')
    elif 'closing_spread_home' in df.columns:
        sh = pd.to_numeric(df['closing_spread_home'], errors='coerce')
    ea = pd.to_numeric(df['edge_ats'], errors='coerce') if 'edge_ats' in df.columns else None
    if sh is not None and ea is not None:
        pm_rec = ea + sh
        mask = (~used) & pm_rec.notna()
        pm[mask] = pm_rec[mask]
        basis_m = basis_m.where(~mask, 'reconstructed_from_edge')
        used = used | mask
    df['pred_margin'] = pm
    df['pred_margin_basis'] = basis_m
    # refresh edge_ats if spread exists
    if (('spread_home' in df.columns) or ('closing_spread_home' in df.columns)):
        sh2 = pd.to_numeric(df.get('spread_home') if 'spread_home' in df.columns else df.get('closing_spread_home'), errors='coerce')
        df['edge_ats'] = pd.to_numeric(df['pred_margin'], errors='coerce') - sh2
    return df


def main():
    ap = argparse.ArgumentParser(description='Rebuild predictions_display_<date>.csv from enriched with robust margin selection')
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    args = ap.parse_args()

    out = Path('outputs')
    enr = out / f'predictions_unified_enriched_{args.date}.csv'
    if not enr.exists():
        raise SystemExit(f"Missing {enr}")
    df = pd.read_csv(enr)
    # Minimal projection for display
    keep = [
        'game_id','home_team','away_team','pred_total','pred_margin','pred_total_basis','pred_margin_basis',
        'edge_total','edge_ats','closing_total','closing_spread_home','display_date','display_time_str',
        'start_time_display','start_time_local'
    ]
    # ensure columns present
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df = choose_display_margin(df)
    disp = df[keep].copy()
    disp_path = out / f'predictions_display_{args.date}.csv'
    disp.to_csv(disp_path, index=False)
    print(f"Wrote {disp_path} rows={len(disp)}")


if __name__ == '__main__':
    main()
