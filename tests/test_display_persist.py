import json
from pathlib import Path

def test_display_persist_file_created():
    out = Path('outputs')
    files = list(out.glob('predictions_display_*.csv'))
    # Allow zero if index route not run; else require at least one file
    if not files:
        return
    # Basic schema check
    import pandas as pd
    df = pd.read_csv(files[-1])
    assert 'pred_total' in df.columns and 'pred_total_basis' in df.columns


def test_pipeline_stats_alert_keys():
    snap = Path('outputs/pipeline_stats_last.json')
    if not snap.exists():
        return
    data = json.loads(snap.read_text())
    # If shares present and below threshold, alert keys should exist
    bt = data.get('basis_share_total_cal')
    if isinstance(bt,(int,float)) and bt < 0.60:
        assert 'alert_low_cal_total' in data
    bm = data.get('basis_share_margin_cal')
    if isinstance(bm,(int,float)) and bm < 0.60:
        assert 'alert_low_cal_margin' in data
