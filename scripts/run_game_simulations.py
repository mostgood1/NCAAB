import sys
from pathlib import Path
import json

# Ensure repository root is on sys.path so `src` imports work when invoked as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.game_sim import run_simulations_for_date


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: run_game_simulations.py <date> [outputs_dir]"}))
        return 1
    date = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs")
    try:
        out_path = run_simulations_for_date(out_dir, date)
        print(json.dumps({"date": date, "sim_path": str(out_path)}))
        return 0
    except Exception as e:
        print(json.dumps({"date": date, "error": str(e)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
