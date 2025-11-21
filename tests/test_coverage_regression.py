import pandas as pd
import pathlib
import pytest

OUT = pathlib.Path(__file__).resolve().parents[1] / 'outputs'

def test_unified_model_coverage_today_strict():
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    path = OUT / f'predictions_unified_{today}.csv'
    if not path.exists():
        pytest.skip('Unified predictions file missing for today')
    df = pd.read_csv(path)
    assert not df.empty
    for col in ['pred_total_model','pred_margin_model','pred_total','pred_margin']:
        assert col in df.columns, f'Missing {col} column'
    miss_total = pd.to_numeric(df['pred_total_model'], errors='coerce').isna().sum()
    miss_margin = pd.to_numeric(df['pred_margin_model'], errors='coerce').isna().sum()
    assert miss_total == 0, f'Model total missing for {miss_total} rows'
    assert miss_margin == 0, f'Model margin missing for {miss_margin} rows'
