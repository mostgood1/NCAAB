from __future__ import annotations
import argparse
import sys

from .backtest_totals import BacktestConfig, run_backtest
from .factors_audit import audit_factors
from .train_totals import main as train_totals_main
from .score_totals import main as score_totals_main
from .integrate_model_totals import main as integrate_model_totals_main


def main():
    ap = argparse.ArgumentParser(prog="ncaab-cli", description="NCAAB tooling CLI")
    sub = ap.add_subparsers(dest="cmd")

    bt = sub.add_parser("backtest-totals", help="Backtest totals predictions vs actuals")
    bt.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    bt.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    bt.add_argument("--recent", type=int, default=None, help="Use N most recent dates by predictions files")
    bt.add_argument("--out-prefix", type=str, default=None, help="Output file prefix under outputs/")

    fa = sub.add_parser("audit-factors", help="Audit factor coverage in features vs actual totals correlation")
    fa.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    fa.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    fa.add_argument("--recent", type=int, default=None, help="Use N most recent feature files")

    tt = sub.add_parser("train-totals", help="Train totals model over full season features/results")
    tt.add_argument("--start", type=str, default=None)
    tt.add_argument("--end", type=str, default=None)
    tt.add_argument("--recent", type=int, default=None)
    tt.add_argument("--use-all", type=str, default="1")
    tt.add_argument("--name", type=str, default="totals_v1")

    st = sub.add_parser("score-totals", help="Score totals for a date using trained model")
    st.add_argument("--date", type=str, required=True)
    st.add_argument("--model", type=str, default=None)

    it = sub.add_parser("integrate-model-totals", help="Integrate model totals into enriched predictions for a date")
    it.add_argument("--date", type=str, required=True)

    args = ap.parse_args()
    if args.cmd == "backtest-totals":
        payload = run_backtest(BacktestConfig(start=args.start, end=args.end, recent=args.recent, out_prefix=args.out_prefix))
        print(payload)
        return 0
    elif args.cmd == "audit-factors":
        payload = audit_factors(args.start, args.end, args.recent)
        print(payload)
        return 0
    elif args.cmd == "train-totals":
        # Delegate to module main for simplicity
        return train_totals_main()
    elif args.cmd == "score-totals":
        return score_totals_main()
    elif args.cmd == "integrate-model-totals":
        return integrate_model_totals_main()
    else:
        ap.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
