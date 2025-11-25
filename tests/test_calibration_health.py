import pytest


def test_calibration_health_endpoint():
    # Import app lazily to avoid side effects during collection
    from app import app  # type: ignore
    with app.test_client() as client:
        resp = client.get('/api/calibration_health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        # Basic keys expected in the payload
        assert 'threshold' in data
        assert 'trend' in data
        assert 'generated_at' in data
        # trend should be a list
        assert isinstance(data['trend'], list)
