import sys
from urllib.parse import urlparse
root = r"c:\\Users\\mostg\\OneDrive\\Coding\\NCAAB"
if root not in sys.path:
    sys.path.insert(0, root)
from app import app  # type: ignore
app.testing = True
client = app.test_client()

DATE = "2026-01-11"

paths = [
    ("root", "/"),
    ("recommendations", f"/recommendations?date={DATE}"),
]

for name, path in paths:
    resp = client.get(path, follow_redirects=False)
    loc = resp.headers.get("Location")
    print(f"[{name}] status={resp.status_code} location={loc}")
    try:
        text = resp.get_data(as_text=True)
    except Exception:
        text = "<binary>"
    # Print a small sample of content to identify template
    sample = (text or "")[:200].replace("\n", " ")
    print(f"[{name}] sample={sample}")
