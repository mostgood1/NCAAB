import datetime as dt
import pandas as pd
import sys, pathlib
# Ensure repository root (parent of scripts/) is on sys.path for app import
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import app  # type: ignore
from app import OUT  # type: ignore

def main():
    # Ensure snapshot populated by invoking index route if absent
    ldf = getattr(app, 'last_index_df', None)
    if ldf is None:
        try:
            with app.app.test_client() as client:  # type: ignore
                client.get('/')
            ldf = getattr(app, 'last_index_df', None)
        except Exception as e:
            print('Failed to invoke app index for snapshot:', e)
            return
    if ldf is None:
        print('No last_index_df available after invoking index')
        return
    ldf = ldf.copy()
    if 'game_id' not in ldf.columns:
        print('Runtime frame missing game_id')
        return
    if 'date' in ldf.columns and ldf['date'].notna().any():
        persist_date = str(ldf['date'].dropna().astype(str).unique()[0])
    else:
        persist_date = dt.datetime.utcnow().strftime('%Y-%m-%d')
    path_display = OUT / f'predictions_display_{persist_date}.csv'
    path_unified = OUT / f'predictions_unified_{persist_date}.csv'
    print('Date used:', persist_date)
    print('Display exists:', path_display.exists(), 'Unified exists:', path_unified.exists())
    if not path_display.exists():
        print('Display file missing')
        return
    disp = pd.read_csv(path_display)
    if 'game_id' not in disp.columns:
        print('Display file missing game_id')
        return
    disp['game_id'] = disp['game_id'].astype(str)
    ldf['game_id'] = ldf['game_id'].astype(str)
    cols_live = [c for c in ['pred_total','pred_margin','pred_total_basis','pred_margin_basis'] if c in ldf.columns]
    cols_disp = [c for c in ['pred_total','pred_margin','pred_total_basis','pred_margin_basis'] if c in disp.columns]
    cmp = ldf[['game_id'] + cols_live].merge(disp[['game_id'] + cols_disp], on='game_id', suffixes=('_live','_disp'))
    # Numeric diffs
    if 'pred_total_live' in cmp.columns and 'pred_total_disp' in cmp.columns:
        diff_total = (pd.to_numeric(cmp['pred_total_live'], errors='coerce') - pd.to_numeric(cmp['pred_total_disp'], errors='coerce')).abs()
        mism_total = cmp[diff_total > 0.01]
        print('Pred_total mismatches >0.01:', len(mism_total))
        if not mism_total.empty:
            print(mism_total[['game_id','pred_total_live','pred_total_disp']].head().to_dict(orient='records'))
    if 'pred_margin_live' in cmp.columns and 'pred_margin_disp' in cmp.columns:
        diff_margin = (pd.to_numeric(cmp['pred_margin_live'], errors='coerce') - pd.to_numeric(cmp['pred_margin_disp'], errors='coerce')).abs()
        mism_margin = cmp[diff_margin > 0.01]
        print('Pred_margin mismatches >0.01:', len(mism_margin))
        if not mism_margin.empty:
            print(mism_margin[['game_id','pred_margin_live','pred_margin_disp']].head().to_dict(orient='records'))
    # Basis diffs
    if 'pred_total_basis_live' in cmp.columns and 'pred_total_basis_disp' in cmp.columns:
        bdiff = cmp[cmp['pred_total_basis_live'] != cmp['pred_total_basis_disp']]
        print('Total basis mismatches:', len(bdiff))
        if not bdiff.empty:
            print(bdiff[['game_id','pred_total_basis_live','pred_total_basis_disp']].head().to_dict(orient='records'))
    if 'pred_margin_basis_live' in cmp.columns and 'pred_margin_basis_disp' in cmp.columns:
        bdiffm = cmp[cmp['pred_margin_basis_live'] != cmp['pred_margin_basis_disp']]
        print('Margin basis mismatches:', len(bdiffm))
        if not bdiffm.empty:
            print(bdiffm[['game_id','pred_margin_basis_live','pred_margin_basis_disp']].head().to_dict(orient='records'))

if __name__ == '__main__':
    main()
