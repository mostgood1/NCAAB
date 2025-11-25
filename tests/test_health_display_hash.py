import json
from pathlib import Path
from app import app as flask_app

def test_health_contains_display_hash():
    # Skip if no display file persisted yet
    out = Path('outputs')
    disp_files = list(out.glob('predictions_display_*.csv'))
    if not disp_files:
        return
    with flask_app.test_client() as c:
        # Hit index to populate pipeline stats if not already
        c.get('/')
        r = c.get('/api/health')
        assert r.status_code == 200
        data = json.loads(r.data.decode())
    # display_hash should be present and a hex string (64 length) or hash_error
    dh = data.get('display_hash')
    assert dh is not None
    if dh != 'hash_error':
        assert isinstance(dh, str) and len(dh) == 64 and all(ch in '0123456789abcdef' for ch in dh.lower())
