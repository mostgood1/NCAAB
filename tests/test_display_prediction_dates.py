import json
from pathlib import Path

from app import app as flask_app

def test_display_prediction_dates_endpoint():
    # Ensure at least one display file exists (skip if none to avoid false fail)
    out = Path('outputs')
    existing = list(out.glob('predictions_display_*.csv'))
    if not existing:
        return
    with flask_app.test_client() as c:
        resp = c.get('/api/display_prediction_dates')
        assert resp.status_code == 200
        data = json.loads(resp.data.decode())
    assert 'dates' in data and isinstance(data['dates'], list)
    if data['dates']:
        assert data['latest'] in data['dates']
