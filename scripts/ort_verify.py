import os
import sys
from pathlib import Path
import time
import json

try:
    import onnxruntime as ort  # type: ignore
except ImportError as e:
    print("onnxruntime not importable. Install the built wheel first.")
    print(e)
    sys.exit(1)

TEST_MODEL_CANDIDATES = [
    "mlp_megatron_basic_test.onnx",
    "bart_mlp_megatron_basic_test.onnx",
    "self_attention_megatron_basic_test.onnx",
]


def _find_test_model():
    # Allow override via env var
    override = os.environ.get("NCAAB_TEST_MODEL")
    if override and Path(override).exists():
        return Path(override)
    cwd = Path.cwd()
    # Search current dir, parent, and project root markers
    search_dirs = [cwd, cwd.parent]
    # If we're inside the onnxruntime submodule, parent.parent might be project root
    if cwd.name == "onnxruntime":
        search_dirs.append(cwd.parent)
    if cwd.parent.name == "onnxruntime":
        search_dirs.append(cwd.parent.parent)
    seen = set()
    for d in search_dirs:
        if d in seen:
            continue
        seen.add(d)
        for fname in TEST_MODEL_CANDIDATES:
            p = d / fname
            if p.exists():
                return p
    return None


def _random_feed(sess):
    import numpy as np  # lazy import
    feed = {}
    for inp in sess.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
        # cap overly large dimension heuristically
        shape = [min(64, s) for s in shape]
        feed[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feed


def main():
    info = {
        "ort_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "dll_dir": os.environ.get("NCAAB_ORT_DLL_DIR"),
        "qnn_sdk_root": os.environ.get("QNN_SDK_ROOT") or os.environ.get("NCAAB_QNN_SDK_DIR"),
    }
    print("ORT version:", info["ort_version"])
    print("Available providers:", info["available_providers"])
    print("NCAAB_ORT_DLL_DIR:", info["dll_dir"])
    print("QNN_SDK_ROOT/NCAAB_QNN_SDK_DIR:", info["qnn_sdk_root"])

    model_path = _find_test_model()
    if not model_path:
        print("No test model found (searched current, parent, project root). Set NCAAB_TEST_MODEL to override.")
        print("Candidates:", TEST_MODEL_CANDIDATES)
        return
    print(f"Using test model: {model_path}")

    preferred = [p for p in ["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"] if p in info["available_providers"]]
    print("Creating session with providers (preferred order):", preferred)
    sess = ort.InferenceSession(str(model_path), providers=preferred)
    session_order = sess.get_providers()
    print("Session provider order:", session_order)
    has_dml = any(p.lower().startswith("dml") for p in session_order)
    has_qnn = any(p.lower().startswith("qnn") for p in session_order)
    print("DML provider active:", has_dml)
    print("QNN provider active:", has_qnn)
    if not has_dml:
        print("WARNING: DmlExecutionProvider not active. Confirm build flags and wheel install.")
    if not has_qnn and "QNNExecutionProvider" in info["available_providers"]:
        print("NOTE: QNNExecutionProvider available but not active; may need provider options or SDK path.")

    # Benchmark small batch
    feed = _random_feed(sess)
    for _ in range(8):
        sess.run(None, feed)  # warmup
    t0 = time.time()
    runs = 32
    for _ in range(runs):
        sess.run(None, feed)
    avg_ms = (time.time() - t0) / runs * 1000.0
    info.update({
        "benchmark_model": model_path.name,
        "benchmark_runs": runs,
        "benchmark_avg_ms": round(avg_ms, 3),
        "session_providers": session_order,
    })
    print("Benchmark avg_ms:", info["benchmark_avg_ms"])
    print("JSON summary:\n" + json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
