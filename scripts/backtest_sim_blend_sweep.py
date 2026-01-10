import sys
import json
from pathlib import Path
import subprocess
import pandas as pd

# Ensure repo root on sys.path for local imports
ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def list_candidate_dates(out_dir: Path, days: int) -> list[str]:
    dr = out_dir / "daily_results"
    dates = []
    if dr.exists():
        for p in sorted(dr.glob("results_*.csv"), reverse=True):
            d = p.stem.replace("results_", "")
            dates.append(d)
    return dates[:days]


def compute_accuracy_for_blend(blend_path: Path, results_path: Path) -> tuple[int, float | None]:
    if not (blend_path.exists() and results_path.exists()):
        return 0, None
    blend = pd.read_csv(blend_path)
    results = pd.read_csv(results_path)
    # Filter date strictly when available
    try:
        if "date" in blend.columns:
            bd = pd.to_datetime(blend["date"], errors="coerce").dt.strftime('%Y-%m-%d')
            blend = blend[bd == blend_path.stem.replace("sim_blend_", "")]
        if "date" in results.columns:
            rd = pd.to_datetime(results["date"], errors="coerce").dt.strftime('%Y-%m-%d')
            results = results[rd == blend_path.stem.replace("sim_blend_", "")]
    except Exception:
        pass
    # Dedup by game_id or team pair
    try:
        if "game_id" in blend.columns:
            blend["game_id"] = blend["game_id"].astype(str)
            blend = blend.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
        else:
            blend = blend.drop_duplicates(subset=["home_team","away_team"]).reset_index(drop=True)
        if "game_id" in results.columns:
            results["game_id"] = results["game_id"].astype(str)
            results = results.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
        else:
            results = results.drop_duplicates(subset=["home_team","away_team"]).reset_index(drop=True)
    except Exception:
        pass
    # Join and compute outcome
    if "game_id" in blend.columns and "game_id" in results.columns:
        merged = blend.merge(results, on="game_id", how="left")
    else:
        merged = blend.merge(results, on=["home_team","away_team"], how="left")
    out_over = merged.get("ou_result_full")
    if out_over is not None:
        y = out_over.astype(str).str.lower().map({"over":1.0, "under":0.0})
    else:
        mt = merged.get("market_total") if "market_total" in merged.columns else merged.get("closing_total")
        y = (pd.to_numeric(merged.get("actual_total"), errors="coerce") > pd.to_numeric(mt, errors="coerce")).astype(float)
    p = pd.to_numeric(merged.get("p_over_blend"), errors="coerce")
    valid = p.notna() & y.notna()
    n = int(valid.sum())
    acc = float(((p[valid] >= 0.5).astype(float) == y[valid]).mean()) if n > 0 else None
    return n, acc


def ensure_sim_quantiles(out_dir: Path, date: str) -> None:
    sim_path = out_dir / f"sim_quantiles_{date}.csv"
    if not sim_path.exists():
        subprocess.run([str(PY), str(Path("scripts")/"run_game_simulations.py"), date, str(out_dir)], check=False)


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: backtest_sim_blend_sweep.py <days> <weights_csv> [outputs_dir]"}))
        return 1
    days = int(sys.argv[1])
    weights = [float(x) for x in sys.argv[2].split(',')]
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("outputs")

    dates = list_candidate_dates(out_dir, days)
    sweep_root = out_dir / "backtests" / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for w in weights:
        accs = []
        Ns = []
        w_dir = sweep_root / f"w_{w:.2f}"
        w_dir.mkdir(parents=True, exist_ok=True)
        for d in dates:
            ensure_sim_quantiles(out_dir, d)
            # Run blender to produce standard blend file, then copy to sweep dir to avoid overwriting across weights
            subprocess.run([str(PY), str(Path("scripts")/"blend_sim_quantiles.py"), d, str(out_dir), str(w)], check=False)
            std_blend = out_dir / f"sim_blend_{d}.csv"
            tgt_blend = w_dir / f"sim_blend_{d}.csv"
            if std_blend.exists():
                tgt_blend.write_bytes(std_blend.read_bytes())
            n, acc = compute_accuracy_for_blend(tgt_blend, out_dir / "daily_results" / f"results_{d}.csv")
            if acc is not None:
                accs.append(acc)
                Ns.append(n)
        # Aggregate by mean accuracy and by pooled accuracy (weighted by N)
        mean_acc = float(pd.Series(accs).mean()) if accs else None
        total_N = int(pd.Series(Ns).sum()) if Ns else 0
        pooled_acc = None
        if total_N > 0:
            # recompute pooled by counting all corrects across dates
            corrects = 0
            for d in dates:
                blend_path = w_dir / f"sim_blend_{d}.csv"
                if blend_path.exists():
                    n, acc = compute_accuracy_for_blend(blend_path, out_dir / "daily_results" / f"results_{d}.csv")
                    if acc is not None:
                        corrects += int(round(acc * n))
            pooled_acc = float(corrects / total_N) if total_N > 0 else None
        summaries.append({"weight": w, "days": days, "mean_accuracy": mean_acc, "pooled_accuracy": pooled_acc, "total_N": total_N})
        # Write per-weight summary CSV
        pd.DataFrame([s for s in summaries if s["weight"] == w]).to_csv(sweep_root / f"summary_{days}d_w_{w:.2f}.csv", index=False)

    # Write combined summary CSV
    out_summary = sweep_root / f"summary_{days}d_all_weights.csv"
    pd.DataFrame(summaries).to_csv(out_summary, index=False)
    print(json.dumps({"wrote": str(out_summary), "weights": weights}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
