import sys
from pathlib import Path
import json
import argparse

# Ensure repository root is on sys.path so `src` imports work when invoked as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.game_sim import run_simulations_for_date


def main():
    ap = argparse.ArgumentParser(description="Run game simulations for a date")
    ap.add_argument("date", help="Slate date YYYY-MM-DD")
    ap.add_argument("outputs_dir", nargs="?", default="outputs", help="Outputs directory (default: outputs)")
    ap.add_argument("--seed", type=int, default=None, help="Deterministic simulation seed (overrides env NCAAB_SIM_SEED)")
    args = ap.parse_args(sys.argv[1:])

    date = args.date
    out_dir = Path(args.outputs_dir)
    try:
        # Optional knobs (all optional; defaults are safe)
        # - NCAAB_SIM_USE_PACE: '1'/'true' to force enable; '0'/'false' to force disable
        # - NCAAB_PACE_SIGMA: float (std dev of possessions per 40)
        # - NCAAB_INJURIES_FILE: path to CSV overrides
        use_pace_env = (str((__import__("os").environ.get("NCAAB_SIM_USE_PACE") or "")).strip().lower())
        if use_pace_env in {"1", "true", "yes", "y"}:
            use_pace = True
        elif use_pace_env in {"0", "false", "no", "n"}:
            use_pace = False
        else:
            use_pace = None

        pace_sigma_env = (__import__("os").environ.get("NCAAB_PACE_SIGMA") or "").strip()
        try:
            pace_sigma = float(pace_sigma_env) if pace_sigma_env else None
        except Exception:
            pace_sigma = None

        injuries_env = (__import__("os").environ.get("NCAAB_INJURIES_FILE") or "").strip()
        injuries_path = Path(injuries_env) if injuries_env else (ROOT / "data" / "injuries_overrides.csv")

        mean_source = (__import__("os").environ.get("NCAAB_SIM_MEAN_SOURCE") or "").strip() or "auto"
        guard_env = (__import__("os").environ.get("NCAAB_SIM_MARKET_GUARDRAILS") or "").strip().lower()
        if guard_env in {"0", "false", "no", "n"}:
            allow_market_guardrails = False
        else:
            allow_market_guardrails = True

        out_path = run_simulations_for_date(
            out_dir,
            date,
            use_pace=use_pace,
            pace_sigma=pace_sigma if pace_sigma is not None else 3.5,
            injuries_path=injuries_path,
            seed=args.seed,
            mean_source=mean_source,
            allow_market_guardrails=allow_market_guardrails,
        )
        print(json.dumps({"date": date, "sim_path": str(out_path)}))
        return 0
    except Exception as e:
        print(json.dumps({"date": date, "error": str(e)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
