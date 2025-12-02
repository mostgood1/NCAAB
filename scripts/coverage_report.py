import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

def coverage_for_date(date_str: str):
    enriched_path = OUT / f'predictions_unified_enriched_{date_str}.csv'
    if not enriched_path.exists():
        enriched_path = OUT / f'predictions_unified_enriched_{date_str}_force_fill.csv'
    if not enriched_path.exists():
        print(f"No enriched file for {date_str}")
        return
    df = pd.read_csv(enriched_path)
    total = len(df)
    has_preds = df['pred_total'].notna().sum() if 'pred_total' in df.columns else 0
    has_margin = df['pred_margin'].notna().sum() if 'pred_margin' in df.columns else 0
    has_iso = df['start_time_iso'].notna().sum() if 'start_time_iso' in df.columns else 0
    has_disp = df['display_date'].notna().sum() if 'display_date' in df.columns else 0
    has_odds = ((df.get('home_odds_decimal').notna() if 'home_odds_decimal' in df.columns else pd.Series([False]*total)) | 
                (df.get('away_odds_decimal').notna() if 'away_odds_decimal' in df.columns else pd.Series([False]*total))).sum()
    print(f"Date {date_str}: total={total} preds={has_preds} margin={has_margin} iso={has_iso} display={has_disp} odds_any={has_odds}")
    missing = df[(df.get('pred_total').isna() if 'pred_total' in df.columns else True) | 
                 (df.get('pred_margin').isna() if 'pred_margin' in df.columns else True)]
    cols = [c for c in ['game_id','date','home_team','away_team','start_time_iso','display_date'] if c in df.columns]
    if not missing.empty:
        print("Missing prediction rows:")
        print(missing[cols].to_string(index=False))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/coverage_report.py YYYY-MM-DD [YYYY-MM-DD ...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        coverage_for_date(d)