"""Small shared utilities: file hashing, atomic artifact writes, action coercion."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    """Hash a complete file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_action(action: Any) -> Any:
    """Coerce a scalar action (possibly a tensor/ndarray) to a plain Python value."""
    return action.item() if hasattr(action, "item") else action


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a small JSON artifact in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a YAML artifact in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml", dir=path.parent, delete=False
    ) as handle:
        yaml.safe_dump(value, handle, allow_unicode=False, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)
