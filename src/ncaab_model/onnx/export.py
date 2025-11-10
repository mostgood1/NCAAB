from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh


def create_dummy_regression_model(out_path: str | Path, in_dim: int = 64, out_dim: int = 4, seed: int = 42) -> Path:
    """Create a tiny ONNX model: y = X @ W^T + b using a single Gemm node.

    This is for smoke testing ONNX Runtime with DirectML.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(out_dim, in_dim)).astype(np.float32)
    b = rng.normal(0, 0.1, size=(out_dim,)).astype(np.float32)

    X = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, in_dim])
    Y = oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None, out_dim])

    W_init = onh.from_array(W, name="W")
    b_init = onh.from_array(b, name="B")

    # Gemm: Y = alpha*A*B + beta*C; we'll use transB=1 to treat W as transposed
    gemm = oh.make_node(
        "Gemm",
        inputs=["input", "W", "B"],
        outputs=["output"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )

    graph = oh.make_graph(
        nodes=[gemm],
        name="dummy_regression",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init, b_init],
    )

    model = oh.make_model(graph, producer_name="ncaab_model", opset_imports=[oh.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path.as_posix())
    return out_path
