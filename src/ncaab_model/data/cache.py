from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import settings


def cache_path(*parts: str) -> Path:
    p = settings.data_dir / "cache" / Path(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
