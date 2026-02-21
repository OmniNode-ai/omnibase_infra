#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared utilities for Infisical provisioning scripts.

This module contains helpers used by both provision-infisical.py and
register-repo.py.  It is intentionally a lightweight, stdlib-only module so
that it can be imported before any project dependencies are installed.

.. versionadded:: 0.10.0
    Extracted from provision-infisical.py and register-repo.py (OMN-2287).
"""

from __future__ import annotations

from pathlib import Path


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a .env file into a key-value dict.

    Skips blank lines and comment lines (starting with ``#``).
    Handles ``export KEY=value`` syntax.
    Strips inline comments and surrounding quotes from values.

    Args:
        env_path: Path to the ``.env`` file.  Returns an empty dict if the
            file does not exist.

    Returns:
        A mapping of environment variable names to their string values.
    """
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:]
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        is_quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"')
        if is_quoted:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #")[0].strip()
        elif "#" in value and not value.startswith("#"):
            value = value.split("#")[0].strip()
        if key:
            values[key] = value
    return values
