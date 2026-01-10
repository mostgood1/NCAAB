import json
import re

def has_digit(s: str) -> bool:
    try:
        return any(ch.isdigit() for ch in s)
    except Exception:
        return False

def test_ou_labels_include_numeric():
    try:
        from app import app as flask_app
    except Exception:
        import importlib
        flask_app = importlib.import_module('app').app

    client = flask_app.test_client()
    resp = client.get('/api/recommendations?date=2026-01-10')
    assert resp.status_code == 200
    data = resp.get_json() or {}
    rows_field = data.get('rows')
    if isinstance(rows_field, list):
        rows = rows_field
    else:
        data_field = data.get('data')
        if isinstance(data_field, list):
            rows = data_field
        else:
            recs_field = data.get('recommendations')
            rows = recs_field if isinstance(recs_field, list) else []
    # Ensure we have OU rows
    ou_rows = [r for r in rows if str(r.get('code') or r.get('rec_code') or '').upper() == 'OU']
    assert len(ou_rows) > 0, 'Expected OU recommendations to be present'
    # Validate each OU row has numeric total in label and a non-empty line
    for r in ou_rows:
        lbl = str(r.get('bet_label') or r.get('bet') or '')
        line = r.get('line')
        assert has_digit(lbl), f"OU bet_label missing numeric total: {lbl}"
        assert line is not None and str(line).strip() != '', f"OU line missing: label={lbl}"
