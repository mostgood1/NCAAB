import sys
import json
from pathlib import Path
import datetime as dt
import pandas as pd

# Ensure repo root on sys.path for local imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.game_sim import run_simulations_for_date

PY = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"


def ensure_sim_blend(date: str, out_dir: Path) -> None:
    sim_path = out_dir / f"sim_quantiles_{date}.csv"
    blend_path = out_dir / f"sim_blend_{date}.csv"
    if not sim_path.exists():
        run_simulations_for_date(out_dir, date)
    if not blend_path.exists():
        # Run the blender script via python to reuse its CLI
        import subprocess
        subprocess.run([str(PY), str(Path("scripts")/"blend_sim_quantiles.py"), date, str(out_dir), "0.4"], check=False)


def list_candidate_dates(out_dir: Path, days: int) -> list[str]:
    # Use daily_results presence as ground truth of playable dates
    dr = out_dir / "daily_results"
    dates = []
    if dr.exists():
        for p in sorted(dr.glob("results_*.csv"), reverse=True):
            d = p.stem.replace("results_", "")
            dates.append(d)
    # Trim to requested window
    dates = dates[:days]
    return dates


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: backtest_sim_blend_range.py <days> [outputs_dir]"}))
        return 1
    days = int(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs")

    dates = list_candidate_dates(out_dir, days)
    rows = []
    for d in dates:
        try:
            ensure_sim_blend(d, out_dir)
            # Compute accuracy by invoking the module entrypoint
            # We can't directly receive return value from compute_acc_main, so reimplement the core here
            blend_path = out_dir / f"sim_blend_{d}.csv"
            results_path = out_dir / "daily_results" / f"results_{d}.csv"
            if not (blend_path.exists() and results_path.exists()):
                rows.append({"date": d, "n": 0, "accuracy": None})
                continue
            blend = pd.read_csv(blend_path)
            results = pd.read_csv(results_path)
            # Filter by date where possible to avoid cross-day contamination
            try:
                if "date" in blend.columns:
                    blend["date"] = pd.to_datetime(blend["date"], errors="coerce").dt.strftime('%Y-%m-%d')
                    blend = blend[blend["date"] == d]
                if "date" in results.columns:
                    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime('%Y-%m-%d')
                    results = results[results["date"] == d]
            except Exception:
                pass
            # Deduplicate per game to avoid inflated counts
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
            # Join on game_id when possible
            if "game_id" in blend.columns and "game_id" in results.columns:
                merged = blend.merge(results, on="game_id", how="left")
            else:
                merged = blend.merge(results, on=["home_team","away_team"], how="left")
            # Outcome from ou_result_full if present
            out_over = merged.get("ou_result_full")
            if out_over is not None:
                y = out_over.astype(str).str.lower().map({"over":1.0, "under":0.0})
            else:
                # Fallback compute using actual_total vs market_total
                mt = merged.get("market_total") if "market_total" in merged.columns else merged.get("closing_total")
                y = (pd.to_numeric(merged.get("actual_total"), errors="coerce") > pd.to_numeric(mt, errors="coerce")).astype(float)
            # Prediction from blended prob
            p = pd.to_numeric(merged.get("p_over_blend"), errors="coerce")
            valid = p.notna() & y.notna()
            n = int(valid.sum())
            acc = float(((p[valid] >= 0.5).astype(float) == y[valid]).mean()) if n > 0 else None
            rows.append({"date": d, "n": n, "accuracy": acc})
        except Exception as e:
            rows.append({"date": d, "error": str(e)})

    out_path = out_dir / f"backtests" / f"sim_blend_summary_last_{days}d.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(json.dumps({"wrote": str(out_path), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
