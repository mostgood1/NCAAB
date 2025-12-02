import re
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUTS = Path("outputs")

PRED_FILE_PATTERN = re.compile(r"predictions_(\d{4}-\d{2}-\d{2})\.csv$")

def iter_dated_prediction_paths(root: Path) -> list[Path]:
    paths = []
    for p in root.glob("predictions_*.csv"):
        m = PRED_FILE_PATTERN.search(p.name)
        if m:
            paths.append(p)
    return sorted(paths)

def load_and_tag(path: Path) -> pd.DataFrame:
    date_match = PRED_FILE_PATTERN.search(path.name)
    date_str = date_match.group(1) if date_match else None
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if 'date' not in df.columns:
        df['date'] = date_str
    # Normalize game_id dtype and remove trailing .0 artifacts
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    return df

def union_columns(dfs: list[pd.DataFrame]) -> list[str]:
    cols = []
    for d in dfs:
        for c in d.columns:
            if c not in cols:
                cols.append(c)
    return cols

def build_history():
    pred_paths = iter_dated_prediction_paths(OUTPUTS)
    if not pred_paths:
        print("[history] No dated prediction files found.")
        return
    loaded = [load_and_tag(p) for p in pred_paths]
    loaded = [d for d in loaded if not d.empty]
    if not loaded:
        print("[history] All dated prediction files empty or unreadable.")
        return
    all_cols = union_columns(loaded)
    norm_frames = []
    for df in loaded:
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            for m in missing:
                df[m] = np.nan
        norm_frames.append(df[all_cols])
    hist = pd.concat(norm_frames, ignore_index=True)
    # De-duplicate only within the same (date, game_id); preserve multi-date evolution
    if {'date','game_id'}.issubset(hist.columns):
        hist = hist.sort_values(['date','game_id']).drop_duplicates(subset=['date','game_id'], keep='last')
    out_path = OUTPUTS / 'predictions_history.csv'
    hist.to_csv(out_path, index=False)
    print(f"[history] Wrote {len(hist)} rows -> {out_path}")

if __name__ == '__main__':
    build_history()
