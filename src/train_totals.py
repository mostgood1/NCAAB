from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

from .model_totals import TrainConfig, train_full_season


def main():
    ap = argparse.ArgumentParser(description="Train totals model over full-season features/results")
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    ap.add_argument("--recent", type=int, default=None, help="Use N most recent dates")
    ap.add_argument("--use-all", type=str, default="1", help="Use features_all.csv when available (1/0)")
    ap.add_argument("--name", type=str, default="totals_v1", help="Model name")
    args = ap.parse_args()
    use_all = (str(args.use_all).strip().lower() in ("1", "true", "yes"))
    cfg = TrainConfig(start=args.start, end=args.end, recent=args.recent, use_all=use_all, model_name=args.name)
    payload = train_full_season(cfg)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
