import sys
from pathlib import Path

# Some environments set PYTHONSAFEPATH (or equivalent) which prevents the
# repo root from being added to sys.path. Ensure tests can import `app.py`
# and the implicit namespace package `src.*`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
