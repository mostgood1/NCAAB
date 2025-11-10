from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from ncaab_model.onnx.export import create_dummy_regression_model
from ncaab_model.onnx.infer import OnnxPredictor
from ncaab_model.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-model", action="store_true", help="Create a dummy ONNX model")
    parser.add_argument("--predict", action="store_true", help="Run a prediction against the dummy model")
    parser.add_argument("--in-dim", type=int, default=64)
    parser.add_argument("--out-dim", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default=str(settings.outputs_dir / "dummy.onnx"))
    args = parser.parse_args()

    model_path = Path(args.model)

    if args.make_model or not model_path.exists():
        print("Creating dummy ONNX model...", model_path)
        create_dummy_regression_model(model_path, in_dim=args.in_dim, out_dim=args.out_dim)

    if args.predict:
        providers = OnnxPredictor.describe_available()
        print("Available providers:", providers)
        if not providers:
            print(
                "onnxruntime not found. To run inference, install either 'onnxruntime' (CPU) or 'onnxruntime-directml' (DML).\n"
                "For Qualcomm NPU (QNN), ensure an ORT build with QNN EP is installed and NCAAB_QNN_SDK_DIR is set."
            )
            return
        predictor = OnnxPredictor(str(model_path))
        x = np.random.randn(args.batch, args.in_dim).astype(np.float32)
        y = predictor.predict(x)
        print("Providers used:", predictor.providers)
        print("Pred shape:", y.shape)
        print(y[: min(3, args.batch)])


if __name__ == "__main__":
    main()
