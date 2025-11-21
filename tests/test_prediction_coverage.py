import os
import math
import pytest
from pathlib import Path

import sys
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from app import app  # Flask app


@pytest.fixture(scope="module")
def client():
    app.testing = True
    with app.test_client() as c:
        yield c


def _is_number(v):
    try:
        if v is None:
            return False
        f = float(v)
        return not math.isnan(f)
    except Exception:
        return False


def test_all_games_have_predictions(client):
    """Index route should return cards for all games with non-null pred_total & pred_margin.

    Uses today's date by default (auto selection logic in index). If a future/past date
    is needed, supply ?date=YYYY-MM-DD. This ensures universal fallback logic works.
    """
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    resp = client.get(f"/?date={today}&diag=1&export=1")
    assert resp.status_code == 200, "Index route failed"
    import app as app_module
    df = getattr(app_module, "_LAST_UNIFIED_FRAME", None)
    if df is None or df.empty:
        pytest.skip("No unified frame (empty slate) - skipping coverage assertion")
    for col in ["pred_total","pred_margin"]:
        assert col in df.columns, f"Missing column {col}"
        assert df[col].notna().sum() == len(df), f"Null predictions in {col}"
    # Stronger check: model prediction completeness when columns exist
    for col in ["pred_total_model","pred_margin_model"]:
        if col in df.columns:
            assert df[col].notna().sum() == len(df), f"Null model predictions in {col}"

    # Supplemental check: unified CSV if written
    today = os.environ.get("TEST_DATE_OVERRIDE")  # allow override in CI
    from datetime import datetime, timezone
    if not today:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = BASE / "outputs"
    model_csv = out_dir / f"predictions_model_{today}.csv"
    if model_csv.exists():
        import pandas as pd
        df = pd.read_csv(model_csv)
        # Expect pred_total_model & pred_margin_model columns (margin may be absent initially)
        if "pred_total_model" in df.columns:
            assert df["pred_total_model"].notna().all(), "Missing model total predictions detected"
        if "pred_margin_model" in df.columns:
            assert df["pred_margin_model"].notna().all(), "Missing model margin predictions detected"


def test_fallback_basis_labels_present(client):
    """Ensure that when fallback predictions are synthesized, basis labels appear in HTML for transparency."""
    resp = client.get("/?diag=1&export=1")
    text = resp.get_data(as_text=True)
    # If slate empty, skip
    import app as app_module
    df = getattr(app_module, "_LAST_UNIFIED_FRAME", None)
    if df is None or df.empty:
        pytest.skip("Empty slate - skipping basis label check")
    # Look for basis markers in HTML OR verify basis columns exist in DataFrame
    has_marker = any(m in text for m in ["fallback_derived","fallback_league_avg","model_raw","model_calibrated"])
    if not has_marker:
        # DataFrame basis columns fallback
        basis_cols = [c for c in df.columns if c.endswith("_basis")]
        assert basis_cols, "No basis columns found in unified frame"
    else:
        assert has_marker, "Missing basis markers"


def test_unified_frame_coverage(client):
    """After index run with export flag, global _LAST_UNIFIED_FRAME should contain non-null pred_total & pred_margin for all rows."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    resp = client.get(f"/?date={today}&diag=1&export=1")
    assert resp.status_code == 200
    import app as app_module  # access global
    df = getattr(app_module, "_LAST_UNIFIED_FRAME", None)
    if df is None or df.empty:
        pytest.skip("Empty unified frame - skipping detailed coverage checks")
    for col in ["pred_total","pred_margin"]:
        assert col in df.columns, f"Missing column {col}"
        assert df[col].notna().sum() == len(df), f"Null predictions present in {col}"
    # If model columns exist they should also be fully populated due to universal fallback
    for col in ["pred_total_model","pred_margin_model"]:
        if col in df.columns:
            assert df[col].notna().sum() == len(df), f"Null model predictions present in {col}"
    # Start time normalization guarantee
    if "start_time" in df.columns:
        st = df["start_time"].astype(str)
        assert st.str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$").any(), "No normalized start_time format detected"
