import re
from app import app

c = app.test_client()
# Render index without date to simulate default
r = c.get('/')
print('INDEX', r.status_code, len(r.get_data()))
html = r.get_data(as_text=True)
# Use a raw string with proper escaping for double quotes
m = re.search(r'href="(/high-likelihood[^"]*)"[^>]*>Recommendations<', html)
print('REC_LINK:', m.group(1) if m else 'NOT_FOUND')
if m:
    rr = c.get(m.group(1), follow_redirects=False)
    print('REC_PAGE', rr.status_code, len(rr.get_data()), rr.headers.get('Location'))
