from __future__ import annotations
import argparse
import sys

from .backtest_totals import BacktestConfig, run_backtest
from .factors_audit import audit_factors
from .train_totals import main as train_totals_main
from .score_totals import main as score_totals_main
from .integrate_model_totals import main as integrate_model_totals_main
from .calibrate_quantiles_segmented import main as calibrate_quantiles_segmented_main
from .calibrate_quantiles_crps import main as calibrate_quantiles_crps_main
from .evaluate_ou_accuracy import evaluate as evaluate_ou
from .build_features import main as build_features_main
from .augment_features import main as augment_features_main
from .integrate_augmented_features import main as integrate_augmented_features_main
from .train_ou_classifier import main as train_ou_lr_main
from .train_ou_classifier_hgb import main as train_ou_hgb_main
from .evaluate_ou_classifier import main as evaluate_ou_classifier_main


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

    cqs = sub.add_parser("calibrate-quantiles-segmented", help="Calibrate quantile spread scaling to target tail coverage")
    cqs.add_argument("--recent", type=int, default=14, help="Use N most recent results files")
    cqs.add_argument("--target", type=float, default=0.10, help="Target tail probability for q10/q90")

    cqc = sub.add_parser("calibrate-quantiles-crps", help="Calibrate quantiles via CRPS (global)")

    bf = sub.add_parser("build-features", help="Build derived features from enriched predictions")
    bf.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD; if omitted, builds for all")

    af = sub.add_parser("augment-features", help="Augment features from boxscores (pace/TS/3P/TO/DRB/rest)")
    af.add_argument("--recent", type=int, default=60, help="Days back to include")
    af.add_argument("--date", type=str, default="", help="Specific YYYY-MM-DD date to process")

    iaf = sub.add_parser("integrate-augmented-features", help="Integrate augmented features into consolidated feature files")

    eva = sub.add_parser("evaluate-ou", help="Evaluate over/under percent correct for predictions")
    eva.add_argument("--start", type=str, default=None)
    eva.add_argument("--end", type=str, default=None)
    eva.add_argument("--recent", type=int, default=14)

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
    elif args.cmd == "calibrate-quantiles-segmented":
        # Run with provided args
        return calibrate_quantiles_segmented_main(args.recent, args.target)
    elif args.cmd == "calibrate-quantiles-crps":
        return calibrate_quantiles_crps_main()
    elif args.cmd == "evaluate-ou":
        payload = evaluate_ou(args.start, args.end, args.recent)
        print(payload)
        return 0
    elif args.cmd == "build-features":
        return build_features_main()
    elif args.cmd == "augment-features":
        return augment_features_main()
    elif args.cmd == "integrate-augmented-features":
        return integrate_augmented_features_main()
    elif args.cmd == "train-ou-lr":
        return train_ou_lr_main()
    elif args.cmd == "train-ou-hgb":
        return train_ou_hgb_main()
    elif args.cmd == "evaluate-ou-classifier":
        return evaluate_ou_classifier_main()
    else:
        ap.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
