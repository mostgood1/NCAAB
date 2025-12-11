import pandas as pd, hashlib, os
from datetime import datetime
OUT = os.path.join(os.getcwd(), 'outputs')
today = datetime.utcnow().strftime('%Y-%m-%d')
path = os.path.join(OUT, f'predictions_display_{today}.csv')
print({'path': path})
if not os.path.exists(path):
    print({'ok': False, 'msg': 'display file missing', 'path': path})
else:
    df = pd.read_csv(path)
    core = df[["game_id","pred_total","pred_margin"]] if set(["game_id","pred_total","pred_margin"]).issubset(df.columns) else df
    if "game_id" in core.columns:
        core = core.sort_values("game_id")
    hasher = hashlib.sha256()
    cols = list(core.columns)
    for _, row in core.iterrows():
        try:
            line_vals = [row.get(col, "") for col in cols]
        except Exception:
            line_vals = [row[col] if col in core.columns else "" for col in cols]
        line = ",".join(map(str, line_vals)) + "\n"
        hasher.update(line.encode())
    digest = hasher.hexdigest()
    print({"ok": True, "date": today, "rows": len(df), "hash": digest})
