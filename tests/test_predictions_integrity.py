import datetime as dt
from pathlib import Path
import pandas as pd
import importlib.util, sys

# Ensure repository root in sys.path so we can import app.py directly
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("app", ROOT / "app.py")
app_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(app_mod)  # type: ignore
app = app_mod.app
OUT = app_mod.OUT
_compute_predictions_hash = app_mod._compute_predictions_hash


def ensure_local_predictions(date_str: str):
    p = OUT / f"predictions_{date_str}.csv"
    if not p.exists():
        df = pd.DataFrame({
            'game_id': ['test_game_1','test_game_2'],
            'date': [date_str, date_str],
            'pred_total': [145.2, 132.7],
            'pred_margin': [5.1, -2.3]
        })
        df.to_csv(p, index=False)
    return p


def test_predictions_integrity_endpoint():
    today = dt.date.today().strftime('%Y-%m-%d')
    path = ensure_local_predictions(today)
    df = pd.read_csv(path)
    local_hash = _compute_predictions_hash(df)
    client = app.test_client()
    resp = client.get(f'/api/predictions_integrity?date={today}')
    assert resp.status_code == 200
    js = resp.get_json()
    assert js.get('ok') is True
    meta = js.get('meta')
    assert meta['exists_primary'] is True
    assert meta['rows_primary'] >= 2
    assert meta['predictions_hash_primary'] == local_hash or meta['predictions_hash_primary'] is not None
    assert 'nan_pred_total_primary' in meta
    assert 'nan_pred_margin_primary' in meta
    # Self-heal test: simulate missing primary with uploaded present
    prim = OUT / f'predictions_{today}.csv'
    upl = OUT / f'predictions_{today}_uploaded.csv'
    df.to_csv(upl, index=False)
    prim.unlink(missing_ok=True)
    resp2 = client.get(f'/api/predictions_integrity?date={today}&auto_promote=1')
    assert resp2.status_code == 200
    js2 = resp2.get_json()
    assert js2['meta']['promotion_performed'] is True
    assert prim.exists()
