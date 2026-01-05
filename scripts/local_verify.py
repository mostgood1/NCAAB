import json
import os, sys
import datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import app

out = {}
with app.test_client() as c:
    # Try only lightweight JSON endpoints first
    try:
        r_ps = c.get('/api/preds-summary')
        out['preds_summary_status'] = r_ps.status_code
        try:
            out['preds_summary'] = r_ps.get_json(force=True)
        except Exception:
            out['preds_summary'] = {'error':'no_json'}
    except Exception as e:
        out['preds_summary_error'] = str(e)

    try:
        r_diag = c.get('/api/diag')
        out['diag_status'] = r_diag.status_code
        try:
            out['diag'] = r_diag.get_json(force=True)
        except Exception:
            out['diag'] = {'error':'no_json'}
    except Exception as e:
        out['diag_error'] = str(e)

    # Index HTML (may be heavier)
    try:
        r_idx = c.get('/', follow_redirects=True)
        out['index_status'] = r_idx.status_code
        html = r_idx.data.decode('utf-8', errors='ignore')
        out['index_even_count'] = html.count('Even')
        out['index_len'] = len(html)
    except Exception as e:
        out['index_error'] = str(e)

    # Index with totals=model to verify model-based totals rendering
    try:
        # Force environment override in case redirects drop query params
        os.environ['NCAAB_TOTALS_MODE'] = 'model'
        r_idx_model = c.get('/?totals=model', follow_redirects=True)
        out['index_model_status'] = r_idx_model.status_code
    except Exception as e:
        out['index_model_error'] = str(e)

    # Inspect debug snapshot written by index route for basis counts
    try:
        today_str = dt.datetime.now().strftime('%Y-%m-%d')
        debug_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', f'index_rows_debug_{today_str}.json')
        if os.path.exists(debug_path):
            with open(debug_path, 'r', encoding='utf-8') as f:
                dbg = json.load(f)
            out['index_rows_debug_basis'] = dbg.get('basis_counts')
            out['index_rows_debug_row_count'] = dbg.get('row_count')
            out['index_rows_non_market_total_rows'] = dbg.get('non_market_total_rows')
            out['index_rows_totals_override_mode'] = dbg.get('totals_override_mode')
        else:
            out['index_rows_debug_missing'] = debug_path
        # Also check for mode-specific snapshot after the '?totals=model' request
        debug_mode_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', f'index_rows_debug_{today_str}_model.json')
        if os.path.exists(debug_mode_path):
            with open(debug_mode_path, 'r', encoding='utf-8') as f:
                dbg_model = json.load(f)
            out['index_rows_debug_basis_model'] = dbg_model.get('basis_counts')
            out['index_rows_debug_row_count_model'] = dbg_model.get('row_count')
            out['index_rows_non_market_total_rows_model'] = dbg_model.get('non_market_total_rows')
            out['index_rows_totals_override_mode_model'] = dbg_model.get('totals_override_mode')
        else:
            out['index_rows_debug_model_missing'] = debug_mode_path
    except Exception as e:
        out['index_rows_debug_error'] = str(e)

print(json.dumps(out, indent=2))
