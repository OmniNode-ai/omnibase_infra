# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Atomic env file writer for interactive onboarding output.

Reads existing file if present, merges (preserves keys not in env_dict,
overwrites keys that are), writes atomically via tmp + rename.
Returns the merged content as a string for dry-run display.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_env_file(env_dict: dict[str, str], target_path: Path) -> str:
    existing: dict[str, str] = {}
    if target_path.exists():
        for line in target_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                existing[k] = v

    merged = {**existing, **env_dict}
    content = "".join(f"{k}={v}\n" for k, v in merged.items())

    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=target_path.parent, suffix=".tmp")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        tmp.replace(target_path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return content


__all__ = ["write_env_file"]
