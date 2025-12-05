import argparse
import json
from pathlib import Path

import pandas as pd

# Local import from src
import sys
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meta_inference import enrich_meta_probs  # type: ignore

OUT = ROOT / "outputs"


def main():
    ap = argparse.ArgumentParser(description="Enrich predictions_unified_enriched_<date>.csv with meta probabilities")
    ap.add_argument("date", help="Target date YYYY-MM-DD")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the enriched file in place")
    args = ap.parse_args()

    enriched_path = OUT / f"predictions_unified_enriched_{args.date}.csv"
    if not enriched_path.exists():
        print(json.dumps({"status": "missing", "path": str(enriched_path)}))
        return

    df = pd.read_csv(enriched_path)
    # Strictly limit to the target slate date to avoid carryover rows
    if 'date' in df.columns:
        df = df[df['date'].astype(str).str.strip() == args.date].copy()
    df2, info = enrich_meta_probs(df)

    if args.inplace:
        df2.to_csv(enriched_path, index=False)
        status = {"status": "updated", "path": str(enriched_path), "info": info}
    else:
        out_path = OUT / f"predictions_unified_enriched_{args.date}_meta.csv"
        df2.to_csv(out_path, index=False)
        status = {"status": "written", "path": str(out_path), "info": info}

    # Write a small diagnostic sidecar
    diag = OUT / f"meta_alignment_diag_{args.date}.json"
    try:
        diag.write_text(json.dumps(status, indent=2), encoding="utf-8")
    except Exception:
        pass

    print(json.dumps(status))


if __name__ == "__main__":
    main()
