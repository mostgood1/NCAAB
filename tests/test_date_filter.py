import os
import importlib

def test_results_date_filter(client):
    """Ensure /api/results enforces final date filter (no leakage of other dates)."""
    # Choose a date likely present (today or sample from environment)
    from datetime import date
    target = date.today().strftime('%Y-%m-%d')
    resp = client.get(f'/api/results?date={target}')
    assert resp.status_code == 200
    js = resp.get_json()
    assert js.get('ok') is True
    meta = js.get('meta', {})
    # post_date_filter_rows should be <= n_rows and removed count non-negative
    if 'post_date_filter_rows' in meta:
        assert meta['post_date_filter_rows'] <= meta.get('n_rows', meta['post_date_filter_rows'])
        removed = meta.get('post_date_filter_removed')
        if removed is not None:
            assert removed >= 0
    # All row dates (if date column returned) must match target
    rows = js.get('rows', [])
    for r in rows:
        if 'date' in r:
            assert str(r['date']) == target

def test_health_version_stamp(client):
    resp = client.get('/api/health')
    assert resp.status_code == 200
    js = resp.get_json()
    assert js.get('status') == 'ok'
    # git_commit may be None if .git not present in deployment, but key should exist
    assert 'git_commit' in js
    assert 'app_version' in js