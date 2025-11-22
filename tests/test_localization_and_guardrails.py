import os
import importlib
import pandas as pd

def test_start_time_local_present():
    os.environ['NCAAB_LOCAL_TZ'] = 'America/New_York'
    import app  # ensure module loads with env
    importlib.reload(app)
    df, meta = app._build_results_df('2025-11-22')
    assert 'start_time_local' in df.columns, 'start_time_local column missing'
    if 'start_time' in df.columns and not df['start_time'].empty:
        # Verify at least one localized value differs from raw hour if UTC offset present
        raw = df['start_time'].astype(str).head(5).tolist()
        loc = df['start_time_local'].astype(str).head(5).tolist()
        assert any(r[:13] != l[:13] for r,l in zip(raw, loc)), 'Localization did not adjust any timestamps'


def test_derived_total_dropped_by_default():
    # Ensure derived_total removed when env var not set
    if 'NCAAB_INCLUDE_DERIVED_TOTAL' in os.environ:
        del os.environ['NCAAB_INCLUDE_DERIVED_TOTAL']
    import app
    importlib.reload(app)
    df, meta = app._build_results_df('2025-11-22')
    assert 'derived_total' not in df.columns, 'derived_total should be dropped by default'
    assert meta.get('derived_total_dropped') is True


def test_include_derived_total_via_env():
    os.environ['NCAAB_INCLUDE_DERIVED_TOTAL'] = '1'
    import app
    importlib.reload(app)
    df, meta = app._build_results_df('2025-11-22')
    assert 'derived_total' in df.columns, 'derived_total should be retained when env var enabled'

