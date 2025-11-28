import pandas as pd
from pathlib import Path
from datetime import datetime
import sys, shutil, json

OUT = Path('outputs')

def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    if len(sys.argv) > 1 and sys.argv[1].strip():
        date_str = sys.argv[1].strip()
    base = OUT / f'predictions_unified_enriched_{date_str}.csv'
    force_path = OUT / f'predictions_unified_enriched_{date_str}_force_fill.csv'
    if not force_path.exists():
        print('Force fill artifact missing:', force_path)
        return
    promote = False
    if not base.exists():
        promote = True
    else:
        try:
            df = pd.read_csv(base)
            if df.empty:
                promote = True
            else:
                # If any predictions missing, prefer promotion
                if ('pred_total' not in df.columns) or df['pred_total'].isna().any() or ('pred_margin' not in df.columns) or df['pred_margin'].isna().any():
                    promote = True
        except Exception:
            promote = True
    if promote:
        # Preserve coverage_status ordering & write summary sidecar
        try:
            df_force = pd.read_csv(force_path)
            if 'coverage_status' in df_force.columns:
                summary = df_force['coverage_status'].value_counts().to_dict()
                sidecar = OUT / f'coverage_status_summary_{date_str}.json'
                with open(sidecar,'w',encoding='utf-8') as fh:
                    json.dump({'date': date_str,'counts': summary,'rows': int(len(df_force))}, fh, indent=2)
        except Exception:
            pass
        shutil.copy2(force_path, base)
        print('Promoted force fill artifact to canonical enriched:', base)
    else:
        print('Canonical enriched already complete; no promotion performed.')

if __name__ == '__main__':
    main()
