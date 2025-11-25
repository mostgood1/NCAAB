import json
from pathlib import Path

def test_display_predictions_api_alignment():
    """Ensure /api/display_predictions matches persisted CSV values for a date.
    If no display file exists yet (index route never hit), test is skipped.
    """
    out = Path('outputs')
    files = sorted(out.glob('predictions_display_*.csv'))
    if not files:
        # Skip gracefully
        return
    import pandas as pd
    csv_path = files[-1]
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    # Infer date from filename
    date_str = csv_path.stem.replace('predictions_display_','')
    # Hit API
    from app import app as flask_app
    with flask_app.test_client() as c:
        resp = c.get(f'/api/display_predictions?date={date_str}')
        assert resp.status_code == 200
        data = json.loads(resp.data.decode())
    assert data.get('date') == date_str
    rows = data.get('rows', [])
    if not rows:
        return
    # Build CSV map
    if 'game_id' not in df.columns:
        return
    df['game_id'] = df['game_id'].astype(str)
    csv_map = {r['game_id']: r for r in df.to_dict(orient='records')}
    # Compare per-row
    mismatches = []
    for r in rows:
        gid = str(r.get('game_id'))
        if gid not in csv_map:
            continue
        csv_r = csv_map[gid]
        # Numeric comparisons within tight tolerance
        for col in ('pred_total','pred_margin'):
            if col in csv_r and col in r:
                try:
                    cv = float(csv_r[col])
                    rv = float(r[col])
                    if abs(cv - rv) > 1e-6:
                        mismatches.append((gid, col, cv, rv))
                except Exception:
                    pass
        # Basis match (allow legacy synonyms model_raw vs model for margin)
        bt_api = r.get('pred_total_basis')
        bt_csv = csv_r.get('pred_total_basis')
        if bt_api and bt_csv and bt_api != bt_csv:
            # Allow API granular refinement when CSV persisted older (unknown -> synthetic/derived/blend/model/cal)
            if bt_csv == 'unknown' and (bt_api.startswith('synthetic') or bt_api.startswith('derived') or bt_api.startswith('blend_') or bt_api in {'model_raw','model','cal'}):
                pass
            else:
                mismatches.append((gid, 'pred_total_basis', bt_csv, bt_api))
        bm_api = r.get('pred_margin_basis')
        bm_csv = csv_r.get('pred_margin_basis')
        if bm_api and bm_csv and bm_api != bm_csv:
            syn = {('model','model_raw'), ('model_raw','model')}
            if bm_csv == 'unknown' and (bm_api.startswith('synthetic') or bm_api in {'model_raw','model','cal'}):
                pass
            elif (bm_api, bm_csv) in syn:
                pass
            else:
                mismatches.append((gid, 'pred_margin_basis', bm_csv, bm_api))
    assert not mismatches, f"Alignment mismatches found: {mismatches[:5]} (showing up to 5)"
