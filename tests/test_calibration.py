import pandas as pd
from pathlib import Path

OUT = Path('outputs')

# Basic test: predictions_unified should not contain blend basis when calibrated available enforcement applied

def test_blend_rows_lack_calibrated_artifact():
    files = sorted([p for p in OUT.glob('predictions_unified_*.csv')], reverse=True)
    if not files:
        return
    df = pd.read_csv(files[0])
    if 'pred_total_basis' in df.columns and 'pred_total_calibrated' in df.columns:
        blend_mask = df['pred_total_basis'].astype(str).str.contains('blend')
        if blend_mask.any():
            cal_series = pd.to_numeric(df['pred_total_calibrated'], errors='coerce')
            assert cal_series[blend_mask].isna().all(), 'Blend rows should not have calibrated total artifact'
    if 'pred_margin_basis' in df.columns and 'pred_margin_calibrated' in df.columns:
        blend_mask_m = df['pred_margin_basis'].astype(str).str.contains('blend')
        if blend_mask_m.any():
            cal_m_series = pd.to_numeric(df['pred_margin_calibrated'], errors='coerce')
            assert cal_m_series[blend_mask_m].isna().all(), 'Blend rows should not have calibrated margin artifact'


def test_cal_or_model_missing_cal_present():
    files = sorted([p for p in OUT.glob('predictions_unified_*.csv')], reverse=True)
    if not files:
        return
    df = pd.read_csv(files[0])
    if 'pred_total_basis' in df.columns:
        bases = set(df['pred_total_basis'].dropna().astype(str))
        assert any(b in bases for b in ['cal','model_raw_missing_cal','cal_est']), 'Expected calibrated or model_raw_missing_cal or cal_est basis in totals'
    if 'pred_margin_basis' in df.columns:
        bases_m = set(df['pred_margin_basis'].dropna().astype(str))
        assert any(b in bases_m for b in ['cal','model_raw_missing_cal','cal_est']), 'Expected calibrated or model_raw_missing_cal or cal_est basis in margins'
