import sys
import warnings
import os
import importlib.util
warnings.filterwarnings("ignore")

# Ensure workspace root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import app from app.py robustly even when running from tests/
import sys
import re
import warnings
warnings.filterwarnings("ignore")

try:
    from app import app
except Exception as e:
    print("IMPORT_ERROR:", e)
    sys.exit(1)

client = app.test_client()
urls = [
    ("/recommendations?group=1&date=2026-01-12", "grouped"),
    ("/recommendations?group=0&date=2026-01-12", "flat"),
    ("/api/recommendations?date=2026-01-12", "api"),
]

for url, name in urls:
    try:
        resp = client.get(url, follow_redirects=False)
        data_len = len(resp.get_data())
        print(f"{name.upper()} STATUS {resp.status_code} LEN {data_len} LOCATION {resp.headers.get('Location')}")
    except Exception as e:
        print("REQUEST_ERROR", url, e)
        sys.exit(2)

# Also fetch index and extract the Recommendations link target
try:
    idx = client.get('/', follow_redirects=False)
    html = idx.get_data(as_text=True)
    m = re.search(r'href=\"(/recommendations[^\"]*)\"[^>]*>Recommendations<', html)
    print('INDEX STATUS', idx.status_code, 'REC_LINK', (m.group(1) if m else 'NOT_FOUND'))
    if m:
        rr = client.get(m.group(1), follow_redirects=False)
        print('REC_FROM_INDEX STATUS', rr.status_code, 'LEN', len(rr.get_data()), 'LOC', rr.headers.get('Location'))
except Exception as e:
    print('INDEX_TEST_ERROR', e)
