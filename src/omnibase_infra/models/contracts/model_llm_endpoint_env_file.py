# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read a host settings store (``.env`` file) into a name -> value mapping (OMN-24017).

Moved here unchanged from ``scripts/check_llm_endpoint_env_contract.py`` so that
script, whose code lines the file-shape ratchet only lets shrink, carries the
overlay resolution without growing.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["load_env_file"]


def load_env_file(path: Path) -> dict[str, str]:
    """Parse ``NAME=value`` lines; blank lines, comments and lines without ``=`` are skipped."""
    env: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env
