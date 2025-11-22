import datetime as dt
from pathlib import Path
import pandas as pd
import importlib.util, sys

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("app", ROOT / "app.py")
app_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(app_mod)  # type: ignore
app = app_mod.app
OUT = app_mod.OUT


def ensure_predictions(date_str: str):
    p = OUT / f"predictions_{date_str}.csv"
    if not p.exists():
        df = pd.DataFrame({
            'game_id': ['psnap1','psnap2'],
            'date': [date_str, date_str],
            'home_team': ['HomeA','HomeB'],
            'away_team': ['AwayA','AwayB'],
            'pred_total': [150.25, 133.75],
            'pred_margin': [5.5, -3.2],
            'pred_total_1h': [72.1, 65.4],
            'pred_total_2h': [78.15, 68.35],
            'pred_margin_1h': [2.1, -1.4],
            'pred_margin_2h': [3.4, -1.8],
            'tuning_totals_bias': [0.0, 0.0],
            'preseason_weight': [0.5, 0.5],
            'preseason_applied': [True, True],
            'pred_total_raw': [150.25, 133.75],
            'pred_total_adjusted': [False, False],
        })
        df.to_csv(p, index=False)
    return p


def test_predictions_snapshot_endpoint():
    today = dt.date.today().strftime('%Y-%m-%d')
    ensure_predictions(today)
    client = app.test_client()
    resp = client.get(f'/api/predictions_snapshot?date={today}')
    assert resp.status_code == 200
    js = resp.get_json()
    assert js.get('ok') is True
    meta = js.get('meta', {})
    assert meta.get('date') == today
    assert meta.get('rows') >= 2
    cols = meta.get('columns', [])
    # Required columns
    for c in ['game_id','pred_total','pred_margin','pred_total_canonical','pred_margin_canonical']:
        assert c in cols
    # Row content sanity
    row0 = js['rows'][0]
    assert 'game_id' in row0
    assert row0.get('pred_total') == row0.get('pred_total_canonical')
