import argparse
from pathlib import Path
import pandas as pd
from src.modeling.hierarchical import apply_conference_shrinkage

OUT = Path('outputs')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, help='Input CSV (enriched predictions)', required=True)
    ap.add_argument('--output', type=str, help='Output CSV path (defaults overwrite input)', default=None)
    ap.add_argument('--conference-col', type=str, default='conference')
    ap.add_argument('--metrics', type=str, nargs='*', default=['pred_total','pred_margin','proj_home','proj_away'])
    ap.add_argument('--alpha', type=float, default=5.0)
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print('Input file missing', inp)
        return
    df = pd.read_csv(inp)
    df2 = apply_conference_shrinkage(df, args.conference_col, args.metrics, alpha=args.alpha)
    outp = Path(args.output) if args.output else inp
    df2.to_csv(outp, index=False)
    print('Shrinkage applied ->', outp)

if __name__ == '__main__':
    main()
