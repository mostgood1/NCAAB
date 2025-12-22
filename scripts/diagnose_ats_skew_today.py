#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt

def main(date: str | None = None) -> int:
    date = date or dt.date.today().strftime("%Y-%m-%d")
    root = Path(__file__).resolve().parents[1]
    p = root / "outputs" / f"predictions_unified_enriched_{date}.csv"
    if not p.exists():
        print(f"[err] Missing enriched file: {p}")
        return 2
    df = pd.read_csv(p)
    # Identify columns
    cover_cols = [c for c in df.columns if 'cover' in c]
    print("[info] cover-related cols:", ", ".join(cover_cols))
    hs = pd.to_numeric(df.get("closing_spread_home", df.get("home_spread", df.get("spread_home"))), errors="coerce")
    mkt_margin = -hs
    pred = pd.to_numeric(df.get("pred_margin_market_blend", df.get("pred_margin")), errors="coerce")
    delta = pred.sub(mkt_margin)
    p_cover_col = None
    preferred_cols = [
        "p_home_cover_final",
        "p_home_cover_meta_cal",
        "p_home_cover_ensemble",
        "p_home_cover_emp",
        "p_home_cover",
        "p_cover_display",
    ]
    for c in preferred_cols:
        if c in df.columns:
            p_cover_col = c
            break
    p_cover = pd.to_numeric(df.get(p_cover_col), errors="coerce") if p_cover_col else None

    tau = 5.0
    mask_base = delta.abs().ge(tau)

    print(f"[info] date={date} base_sel_ct={int(mask_base.sum())}")
    # Distribution of delta sign within base selection
    base_delta_pos = int((delta[mask_base] > 0).sum())
    base_delta_neg = int((delta[mask_base] < 0).sum())
    print(f"[info] delta>0 (home)={base_delta_pos} delta<0 (away)={base_delta_neg}")

    if p_cover is not None:
        thr = 0.55
        mask_prob = mask_base & p_cover.notna()
        home_ok = (p_cover.ge(thr)) & mask_prob
        away_ok = (p_cover.le(1.0 - thr)) & mask_prob  # equivalently (1 - p_cover) >= thr
        print(f"[info] p_cover_col={p_cover_col} prob_mask_ct={int(mask_prob.sum())} thr={thr}")
        print(f"[info] home_ok_ct={int(home_ok.sum())} away_ok_ct={int(away_ok.sum())}")
        # Agreement between prob-side and delta sign
        agree_home = int(((delta.gt(0)) & home_ok).sum())
        agree_away = int(((delta.lt(0)) & away_ok).sum())
        print(f"[info] agree_home={agree_home} agree_away={agree_away}")
        # Quantiles for p_cover under mask
        q = p_cover[mask_prob].quantile([0.1,0.25,0.5,0.75,0.9]) if int(mask_prob.sum())>0 else None
        if q is not None:
            print("[info] p_cover quantiles (10/25/50/75/90):", ", ".join(f"{a:.3f}" for a in q.values))
        # Also inspect quantiles for all cover columns
        for c in preferred_cols:
            if c in df.columns:
                pc = pd.to_numeric(df[c], errors="coerce")
                qq = pc[mask_base & pc.notna()].quantile([0.1,0.25,0.5,0.75,0.9]) if int((mask_base & pc.notna()).sum())>0 else None
                if qq is not None:
                    print(f"[info] {c} quantiles:", ", ".join(f"{a:.3f}" for a in qq.values))

    else:
        print("[warn] No p_cover column found; selections rely on delta sign only.")

    return 0

if __name__ == "__main__":
    sys.exit(main(None))
