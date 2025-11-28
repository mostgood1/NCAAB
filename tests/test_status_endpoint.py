import json
from app import app as flask_app

def test_status_endpoint_shape():
    with flask_app.test_client() as c:
        rv = c.get('/api/status')
        assert rv.status_code == 200
        data = rv.get_json()
        # Core keys present
        for key in ['results_latest','stake_latest','finalize','display_hash']:
            assert key in data
        # Finalize payload shape when present
        fin = data.get('finalize')
        if isinstance(fin, dict):
            for k in ['date','rows','final','pending','started','ready']:
                assert k in fin
