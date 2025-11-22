import os
import importlib
from flask import json


def test_health_includes_timezone_and_anomaly():
    os.environ.pop('NCAAB_SCHEDULE_ANOMALY_THRESHOLD', None)
    import app
    importlib.reload(app)
    with app.app.test_request_context('/'):
        resp, status = app.api_health()
        assert status == 200
        payload = resp.get_json()
        assert 'local_timezone' in payload, 'local_timezone missing in health payload'
        assert 'schedule_anomaly' in payload, 'schedule_anomaly missing'
        sa = payload['schedule_anomaly']
        assert 'threshold' in sa and 'games_today_rows' in sa
        # ncaa_endpoint_code may be None if no anomaly; just ensure key exists
        assert 'ncaa_endpoint_code' in sa
