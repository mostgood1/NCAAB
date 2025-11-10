from pathlib import Path
import numpy as np

from ncaab_model.onnx.export import create_dummy_regression_model
from ncaab_model.onnx.infer import OnnxPredictor


def test_dummy_inference(tmp_path: Path):
    model_path = tmp_path / "dummy.onnx"
    create_dummy_regression_model(model_path, in_dim=16, out_dim=3)
    providers = OnnxPredictor.describe_available()
    if not providers:
        # onnxruntime not installed; skip runtime portion
        return
    pred = OnnxPredictor(str(model_path))
    x = np.random.randn(5, 16).astype(np.float32)
    y = pred.predict(x)
    assert y.shape == (5, 3)
