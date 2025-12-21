#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import math
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


def _ou_prob_column(df: pd.DataFrame) -> str | None:
    for c in [
        "p_over_final",
        "p_over_meta_cal",
        "p_over_display",
        "p_over",
        "p_over_emp",
        "p_over_dist",
        "p_over_ensemble",
    ]:
        if c in df.columns:
            return c
    return None


def _ou_sweep(df: pd.DataFrame, hi_vals: Iterable[float], lo_vals: Iterable[float]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ou_hi", "ou_lo", "totals", "overs", "unders", "balance"])
    pcol = _ou_prob_column(df)
    if not pcol:
        return pd.DataFrame(columns=["ou_hi", "ou_lo", "totals", "overs", "unders", "balance"])
    probs = pd.to_numeric(df[pcol], errors="coerce")
    # Use edge_total/pred_total vs market_total when probability is NaN for side determination, but only count when prob is confident
    edge = pd.to_numeric(df.get("edge_total"), errors="coerce")
    pt = pd.to_numeric(df.get("pred_total"), errors="coerce")
    ln = pd.to_numeric(df.get("market_total"), errors="coerce")

    def side_by_fallback(i: int) -> str:
        try:
            # Prefer pred vs market
            pti = float(pt.iloc[i]) if pd.notna(pt.iloc[i]) else math.nan
            lni = float(ln.iloc[i]) if pd.notna(ln.iloc[i]) else math.nan
            if math.isfinite(pti) and math.isfinite(lni):
                return "Over" if pti > lni else "Under"
        except Exception:
            pass
        try:
            ei = float(edge.iloc[i]) if pd.notna(edge.iloc[i]) else math.nan
            if math.isfinite(ei):
                return "Over" if ei >= 0 else "Under"
        except Exception:
            pass
        return "Over"

    rows = []
    n = len(df)
    for hi in hi_vals:
        for lo in lo_vals:
            overs = 0
            unders = 0
            for i in range(n):
                p = probs.iloc[i]
                if pd.notna(p):
                    if p >= hi:
                        overs += 1
                        continue
                    if p <= lo:
                        unders += 1
                        continue
                    # not confident → drop
                    continue
                # If probability missing, drop to avoid overs-only bias
                # side = side_by_fallback(i)  # We do not count fallback-only in sweep
            totals = overs + unders
            balance = 0.0
            if totals > 0:
                balance = min(overs, unders) / max(overs, unders) if max(overs, unders) > 0 else 0.0
            rows.append({
                "ou_hi": round(float(hi), 4),
                "ou_lo": round(float(lo), 4),
                "totals": int(totals),
                "overs": int(overs),
                "unders": int(unders),
                "balance": round(float(balance), 4),
            })
    return pd.DataFrame(rows)


def _ats_sweep(enriched: pd.DataFrame, tau_vals: Iterable[float], prob_thresholds: Iterable[float], use_closing: bool, strict_modes: Iterable[bool]) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame(columns=["tau", "prob_thr", "strict", "ats_total", "home", "away"])
    # Mirror selection logic from scripts/select_ats_picks.py
    hs = pd.to_numeric(enriched.get("closing_spread_home") if use_closing else enriched.get("home_spread", enriched.get("spread_home")), errors="coerce")
    mkt_margin = -hs
    pred_blend = pd.to_numeric(enriched.get("pred_margin_market_blend", enriched.get("pred_margin")), errors="coerce")
    p_cover = pd.to_numeric(enriched.get("p_cover_display", enriched.get("p_home_cover_emp", enriched.get("p_home_cover"))), errors="coerce")
    mismatch = enriched.get("flag_market_margin_mismatch")
    if mismatch is None:
        mismatch = pd.Series(False, index=enriched.index)
    else:
        try:
            mismatch = mismatch.fillna(False).infer_objects(copy=False).astype(bool)
        except Exception:
            mismatch = mismatch.fillna(False).map(lambda x: bool(x))

    delta = pred_blend.sub(mkt_margin)

    rows = []
    for tau in tau_vals:
        sel_base = delta.abs().ge(tau) & (~mismatch)
        for prob_thr in prob_thresholds:
            for strict in strict_modes:
                if pd.notna(p_cover).any():
                    prob_home_ok = p_cover.ge(prob_thr)
                    prob_away_ok = (1.0 - p_cover).ge(prob_thr)
                    sel_prob = sel_base & p_cover.notna() & (prob_home_ok | prob_away_ok)
                    if strict:
                        sel_final = sel_prob
                        pred_home = pd.Series(False, index=enriched.index)
                        pred_home.loc[p_cover.notna()] = p_cover.loc[p_cover.notna()].ge(prob_thr)
                    else:
                        sel_final = sel_prob | (sel_base & (~p_cover.notna()))
                        pred_home = pd.Series(False, index=enriched.index)
                        pred_home.loc[p_cover.notna()] = p_cover.loc[p_cover.notna()].ge(prob_thr)
                        pred_home.loc[~p_cover.notna()] = delta.loc[~p_cover.notna()].gt(0)
                else:
                    sel_final = sel_base
                    pred_home = delta.gt(0)

                idx = sel_final[sel_final].index
                if len(idx) == 0:
                    rows.append({
                        "tau": float(tau),
                        "prob_thr": float(prob_thr),
                        "strict": bool(strict),
                        "ats_total": 0,
                        "home": 0,
                        "away": 0,
                    })
                    continue

                home_ct = int(pred_home.loc[idx].sum())
                away_ct = int(len(idx) - home_ct)
                rows.append({
                    "tau": float(tau),
                    "prob_thr": float(prob_thr),
                    "strict": bool(strict),
                    "ats_total": int(len(idx)),
                    "home": home_ct,
                    "away": away_ct,
                })
    return pd.DataFrame(rows)


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    # Inclusive stop when aligned with step
    x = start
    while x <= stop + 1e-9:
        yield round(x, 10)
        x += step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD; default today")
    ap.add_argument("--ou-hi", type=str, default="0.50:0.58:0.02", help="hi range start:stop:step")
    ap.add_argument("--ou-lo", type=str, default="0.42:0.50:0.02", help="lo range start:stop:step")
    ap.add_argument("--ats-thresholds", type=str, default="0.50,0.52,0.54,0.56", help="prob thresholds list")
    ap.add_argument("--ats-tau", type=str, default="5.0,6.0,7.0", help="delta tau list")
    ap.add_argument("--use-closing", action="store_true", help="Use closing spread for ATS")
    ap.add_argument("--strict", action="store_true", help="Use strict prob-side for ATS")
    args = ap.parse_args()

    date = args.date or dt.date.today().strftime("%Y-%m-%d")
    enriched = _safe_read_csv(OUT / f"predictions_unified_enriched_{date}.csv")
    if enriched.empty:
        print(f"[warn] No enriched predictions found for {date} at {OUT}/predictions_unified_enriched_{date}.csv")
    # OU sweep
    def parse_range(s: str) -> Tuple[float, float, float]:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad range: {s}")
        return (float(parts[0]), float(parts[1]), float(parts[2]))

    ou_hi_s, ou_hi_e, ou_hi_step = parse_range(args.ou_hi)
    ou_lo_s, ou_lo_e, ou_lo_step = parse_range(args.ou_lo)
    ou_df = _ou_sweep(enriched, _frange(ou_hi_s, ou_hi_e, ou_hi_step), _frange(ou_lo_s, ou_lo_e, ou_lo_step))

    # ATS sweep
    def parse_list(s: str) -> Iterable[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    tau_vals = parse_list(args.ats_tau)
    thr_vals = parse_list(args.ats_thresholds)
    strict_modes = [args.strict, not args.strict]
    ats_df = _ats_sweep(enriched, tau_vals, thr_vals, use_closing=args.use_closing, strict_modes=strict_modes)

    # Write outputs
    sweeps_dir = OUT / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    ou_out = sweeps_dir / f"ou_sweep_{date}.csv"
    ats_out = sweeps_dir / f"ats_sweep_{date}.csv"
    ou_df.to_csv(ou_out, index=False)
    ats_df.to_csv(ats_out, index=False)

    # Quick summary prints
    print(f"[ok] OU sweep written: {ou_out} rows={len(ou_df)}")
    if not ou_df.empty:
        # Prefer combinations near 50/50 with reasonable coverage
        cand = ou_df.copy()
        cand["score"] = cand.apply(lambda r: (r["balance"] * 0.7) + (min(r["totals"], 20) / 20.0 * 0.3), axis=1)
        best = cand.sort_values(["score", "totals"], ascending=[False, False]).head(5)
        print("Top OU combos:")
        for _, r in best.iterrows():
            print(f"  hi={r['ou_hi']:.3f} lo={r['ou_lo']:.3f} totals={int(r['totals'])} overs={int(r['overs'])} unders={int(r['unders'])} balance={r['balance']:.3f}")

    print(f"[ok] ATS sweep written: {ats_out} rows={len(ats_df)}")
    if not ats_df.empty:
        cand2 = ats_df.copy()
        cand2["balance"] = cand2.apply(lambda r: (min(r["home"], r["away"]) / max(r["home"], r["away"]) if max(r["home"], r["away"]) > 0 else 0.0), axis=1)
        cand2["score"] = cand2.apply(lambda r: (r["balance"] * 0.6) + (min(r["ats_total"], 10) / 10.0 * 0.4), axis=1)
        best2 = cand2.sort_values(["score", "ats_total"], ascending=[False, False]).head(5)
        print("Top ATS combos:")
        for _, r in best2.iterrows():
            print(f"  tau={r['tau']:.2f} thr={r['prob_thr']:.3f} strict={bool(r['strict'])} total={int(r['ats_total'])} home={int(r['home'])} away={int(r['away'])} balance={r['balance']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
