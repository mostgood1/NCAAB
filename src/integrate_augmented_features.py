import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

AUG_RECENT = OUT / 'features_augmented_recent.csv'

TARGETS = [
    OUT / 'features_curr.csv',
    OUT / 'features_all.csv',
    OUT / 'features_last2.csv',
]

AUG_SUFFIX = '_augmented'


def integrate_one(target: Path, aug: pd.DataFrame) -> Path | None:
    if not target.exists():
        return None
    try:
        base = pd.read_csv(target)
    except Exception:
        return None
    if base.empty or aug.empty:
        return None
    # Merge on game_id; avoid duplicating columns
    key = 'game_id'
    common = set(base.columns) & set(aug.columns)
    aug_cols = [c for c in aug.columns if c not in common or c == key]
    merged = base.merge(aug[aug_cols], on=key, how='left')
    out = target.with_name(target.stem + AUG_SUFFIX + target.suffix)
    merged.to_csv(out, index=False)
    return out


def run() -> dict:
    if not AUG_RECENT.exists():
        return {'status': 'no_augmented_file'}
    aug = pd.read_csv(AUG_RECENT)
    results = {}
    for t in TARGETS:
        out = integrate_one(t, aug)
        if out:
            results[t.name] = str(out)
    # Also produce a standalone enriched preview
    prev = OUT / 'features_merged_recent.csv'
    aug.to_csv(prev, index=False)
    results['preview'] = str(prev)
    results['rows'] = int(len(aug))
    return results


def main():
    res = run()
    print(res)

if __name__ == '__main__':
    main()
