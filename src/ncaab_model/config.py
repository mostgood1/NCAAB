from pathlib import Path
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Directories
    # Note: __file__ is .../src/ncaab_model/config.py; parents[2] points to the project root (NCAAB)
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    outputs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "outputs")

    # API keys (set via env vars)
    theodds_api_key: str | None = None

    # QNN SDK (for Qualcomm NPU via ONNX Runtime QNN EP)
    # If not provided via env var, default to common install path on Windows.
    qnn_sdk_dir: Path | None = None
    qnn_backend_dll: Path | None = None  # override to point directly to backend DLL if needed

    # Team mapping (optional): CSV to harmonize team names across providers
    team_map_path: Path | None = None

    # pydantic-settings v2 configuration
    model_config = SettingsConfigDict(
        env_prefix="NCAAB_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )


def _default_qnn_dir() -> Path | None:
    # Prefer env var first
    env_dir = os.getenv("NCAAB_QNN_SDK_DIR") or os.getenv("QNN_SDK_ROOT")
    if env_dir:
        p = Path(env_dir)
        return p if p.exists() else None
    # Common default on Windows
    p = Path("C:/Qualcomm/QNN_SDK")
    return p if p.exists() else None


_defaults = {}
qnn_dir = _default_qnn_dir()
if qnn_dir is not None:
    _defaults["qnn_sdk_dir"] = qnn_dir


settings = Settings(**_defaults)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.mkdir(parents=True, exist_ok=True)

# Default team_map_path to <project_root>/data/team_map.csv if not explicitly set
if settings.team_map_path is None:
    settings.team_map_path = settings.data_dir / "team_map.csv"
