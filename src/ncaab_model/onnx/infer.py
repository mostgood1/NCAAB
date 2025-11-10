from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np
import onnx
import onnx.numpy_helper as onh

from ..config import settings


def _maybe_add_dll_dirs() -> None:
    """On Windows, add DLL search directories before importing onnxruntime.

    This enables using a local ONNX Runtime build (e.g., with QNN EP) without pip-installed wheel.
    Set one of these env vars to the directory containing onnxruntime DLLs:
      - NCAAB_ORT_DLL_DIR
      - ONNXRUNTIME_DLL_DIR
      - ONNXRUNTIME_DIR
      - ORT_DLL_DIR
    Additionally, if QNN SDK is configured, add a few common QNN backend DLL folders.
    """
    if os.name != "nt":
        return
    def _add_dir(path: str) -> None:
        try:
            if os.path.isdir(path):
                os.add_dll_directory(path)  # type: ignore[attr-defined]
        except Exception:
            pass

    # ORT directories
    for key in ("NCAAB_ORT_DLL_DIR", "ONNXRUNTIME_DLL_DIR", "ONNXRUNTIME_DIR", "ORT_DLL_DIR"):
        val = os.getenv(key)
        if not val:
            continue
        parts = [p for p in val.split(";") if p.strip()]
        for p in parts:
            _add_dir(p.strip().strip('"'))

    # QNN SDK common DLL locations (for backend dependencies)
    qnn_root = os.getenv("NCAAB_QNN_SDK_DIR") or os.getenv("QNN_SDK_ROOT")
    if qnn_root and os.path.isdir(qnn_root):
        candidates = [
            os.path.join(qnn_root, "lib", "htp", "Windows", "Release"),
            os.path.join(qnn_root, "lib", "htp", "Windows"),
            os.path.join(qnn_root, "lib", "Windows", "htp"),
        ]
        for d in candidates:
            _add_dir(d)


def _maybe_add_python_paths() -> None:
    """Allow importing onnxruntime from a local wheel or module directory.

    Set one of these env vars (semicolon-separated supported):
      - NCAAB_ORT_PY_DIR (recommended) -> can be a folder or a .whl file path
      - ONNXRUNTIME_PY_DIR
    We append each entry to sys.path so Python can import the onnxruntime module.
    """
    for key in ("NCAAB_ORT_PY_DIR", "ONNXRUNTIME_PY_DIR"):
        val = os.getenv(key)
        if not val:
            continue
        parts = [p for p in val.split(";") if p.strip()]
        for p in parts:
            p = p.strip().strip('"')
            if os.path.exists(p) and p not in sys.path:
                sys.path.append(p)


# Attempt to prepare DLL search path, then import ORT.
try:
    _maybe_add_dll_dirs()
    _maybe_add_python_paths()
    import onnxruntime as ort  # type: ignore
except Exception:  # ImportError or other load errors
    ort = None  # type: ignore


def _qnn_provider_options() -> tuple[str, dict] | None:
    """Build QNN EP provider + options if QNN SDK is available.

    Returns (provider_name, options) or None if QNN cannot be configured.
    The exact backend DLL path may vary by SDK version; allow override via env or settings.
    """
    if settings.qnn_backend_dll is not None and settings.qnn_backend_dll.exists():
        return (
            "QNNExecutionProvider",
            {"backend_path": settings.qnn_backend_dll.as_posix()},
        )

    qnn_root = settings.qnn_sdk_dir
    if not qnn_root:
        # Also allow direct env override for DLL
        dll_env = os.getenv("NCAAB_QNN_BACKEND_DLL")
        if dll_env and os.path.exists(dll_env):
            return (
                "QNNExecutionProvider",
                {"backend_path": os.path.abspath(dll_env).replace("\\", "/")},
            )
        return None

    # Heuristic: try common backend DLL locations (adjust as needed for your SDK version)
    candidate_rel = [
        # Example paths; please adjust for your installed SDK layout
        "lib/htp/Windows/Release/qnn-htp.dll",
        "lib/htp/Windows/qnn-htp.dll",
        "lib/Windows/htp/qnn-htp.dll",
    ]
    for rel in candidate_rel:
        dll = (qnn_root / rel).resolve()
        if dll.exists():
            return (
                "QNNExecutionProvider",
                {"backend_path": dll.as_posix()},
            )
    return None


class OnnxPredictor:
    """ONNX Runtime predictor with QNN/DirectML preference and CPU fallback.

    Preference order:
    1) QNNExecutionProvider (if QNN SDK configured and provider available)
    2) DmlExecutionProvider
    3) CPUExecutionProvider
    """

    def __init__(self, model_path: str | bytes, prefer_dml: bool = True, prefer_qnn: bool = True):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not available. Either install via pip (if a wheel exists) or set NCAAB_ORT_DLL_DIR "
                "to the folder containing onnxruntime DLLs (and providers). Optionally install an ORT build with QNN EP "
                "and set NCAAB_QNN_SDK_DIR (and NCAAB_QNN_BACKEND_DLL)."
            )

        providers: list[str] = []
        provider_options: list[dict] = []

        available = set(ort.get_available_providers())

        # Try QNN first if available and configured
        if prefer_qnn and "QNNExecutionProvider" in available:
            qnn_opt = _qnn_provider_options()
            if qnn_opt is not None:
                providers.append(qnn_opt[0])
                provider_options.append(qnn_opt[1])

        # Then DirectML
        if prefer_dml and "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
            provider_options.append({})

        # Always include CPU fallback
        providers.append("CPUExecutionProvider")
        provider_options.append({})

        self.session = ort.InferenceSession(model_path, providers=providers, provider_options=provider_options)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.providers = self.session.get_providers()

    def predict(self, x: np.ndarray | dict[str, np.ndarray]) -> np.ndarray | list[np.ndarray]:
        if isinstance(x, dict):
            feed = x
        else:
            if len(self.input_names) != 1:
                raise ValueError("Model expects multiple inputs; provide a dict[str, np.ndarray].")
            feed = {self.input_names[0]: x}
        outputs = self.session.run(self.output_names, feed)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @staticmethod
    def describe_available():
        if ort is None:
            return []
        try:
            return list(getattr(ort, "get_available_providers", lambda: [])())
        except Exception:
            return []


class NumpyLinearPredictor:
    """Lightweight predictor for our exported linear ONNX (mu/sigma standardization + Gemm).

    This bypasses onnxruntime by parsing initializers and running the math in NumPy.
    It supports models exported by train.baseline._export_onnx_linear.
    """

    def __init__(self, model_path: str | bytes):
        model = onnx.load_model(model_path)
        inits = {init.name: onh.to_array(init) for init in model.graph.initializer}
        # Expected names
        try:
            self.mu = inits["mu"].astype(np.float32)
            self.sigma = inits["sigma"].astype(np.float32)
            # W saved as shape (1, d) for Gemm with transB=1; we want (d,)
            W = inits["W"].astype(np.float32)
            self.W = W.reshape(-1)
            B = inits["B"].astype(np.float32)
            self.b = float(B.reshape(-1)[0])
        except KeyError as e:
            raise ValueError(f"Model missing expected initializer {e}. Not a supported linear ONNX.")
        self.providers = ["NumpyFallback"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x_norm = (x - self.mu) / np.where(self.sigma == 0, 1.0, self.sigma)
        y = x_norm @ self.W + self.b
        return y.reshape(-1, 1).astype(np.float32)

    @staticmethod
    def can_load(model_path: str | bytes) -> bool:
        try:
            model = onnx.load_model(model_path)
            names = {init.name for init in model.graph.initializer}
            return {"mu", "sigma", "W", "B"}.issubset(names)
        except Exception:
            return False
