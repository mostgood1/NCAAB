# ONNX Runtime + QNN EP on Windows ARM64

This project prefers ONNX Runtime for inference acceleration and aims to enable the Qualcomm QNN Execution Provider (EP) on Snapdragon devices.

## 1) Install ONNX Runtime

Options:
- If a prebuilt Windows ARM64 wheel is available:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install --upgrade onnxruntime
  ```
- If using DirectML for GPU fallback on Windows:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install --upgrade onnxruntime-directml
  ```
  Note: QNN EP is separate; DirectML is a GPU-based EP from Microsoft.

If wheels are unavailable for your target, build from source following ONNX Runtime docs (with ARM64 & QNN flags).

## 2) QNN EP Binaries

You need the Qualcomm QNN SDK and an ONNX Runtime build with QNN provider enabled. Ensure the following exist:
- QNN backend DLL (e.g., `QnnHtp.dll`) located in one of:
  - `lib/arm64x-windows-msvc/`
  - `lib/aarch64-windows-msvc/`
  - `lib/windows-aarch64/`
- ONNX Runtime core + providers DLLs in your build output (e.g., `onnxruntime.dll`, `onnxruntime_providers_qnn.dll`).

## 3) Environment Variables (example)

Set env vars before launching the app:
```powershell
$env:ORT_QNN_BACKEND_PATH = "C:\\qnn\\lib\\windows-aarch64\\QnnHtp.dll"
$env:PATH = "C:\\qnn\\lib\\windows-aarch64;" + $env:PATH
# Optional: if ORT providers DLLs are not on PATH
$env:PATH = "C:\\onnxruntime\\build\\Windows-ARM64\\Release;" + $env:PATH
```
Adjust paths to match your SDK/build locations.

## 4) Verify Providers

Run diagnostics (after setting environment variables):
```powershell
.\.venv\Scripts\python.exe -m ncaab_model.cli ort-diagnostics
```
Expected output includes providers list, e.g.:
- QNNExecutionProvider
- DmlExecutionProvider (if DirectML installed)
- CPUExecutionProvider

If QNN is missing:
1. Confirm your ORT build flags included `--enable_qnn`.
2. Verify `onnxruntime_providers_qnn.dll` resides in `$env:NCAAB_ORT_DLL_DIR`.
3. Confirm `ORT_QNN_BACKEND_PATH` points to the correct `QnnHtp.dll` variant (prefer `arm64x` on modern Snapdragon Windows).
4. Start a fresh PowerShell session (PATH updates sometimes require it) and re-run diagnostics.

## 5) Using QNN in code

The CLI and internal inference module will try to use available providers in preferred order (QNN > DML > CPU). No changes needed to your commands.

## 6) Troubleshooting
- ImportError on onnxruntime: Wheel unavailable for Windows ARM64; build from source then set `NCAAB_ORT_DLL_DIR` & `NCAAB_ORT_PY_DIR` (and PATH/PYTHONPATH).
- QNN EP not showing: Confirm build includes QNN (`onnxruntime_providers_qnn.dll` exists) and backend DLL path set (`ORT_QNN_BACKEND_PATH`).
- Backend DLL path resolves but provider absent: version mismatch or missing build flag; rebuild with `--enable_qnn`.
- Need environment hints: run `.\.venv\Scripts\python.exe -m ncaab_model.cli ort-env-hints`.
- Provider ordering: `ort-diagnostics` output shows chosen precedence (QNN > DML > CPU when available).

> Fallback: When ONNX Runtime isn't present, the system uses NumPy linear predictors. Functionality is maintained with reduced performance.
