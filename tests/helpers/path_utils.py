# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Path discovery utilities for tests.

Provides a shared ``find_project_root()`` helper so that test modules that
need to locate project-relative files (e.g. contract YAML) do not duplicate
the lookup logic.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path) -> Path:
    """Walk up from *start* to find the project root (contains pyproject.toml).

    Parameters
    ----------
    start:
        Directory to start the search from.  Callers must pass their own
        starting directory explicitly, typically
        ``Path(__file__).resolve().parent``.

    Raises
    ------
    RuntimeError
        If no ``pyproject.toml`` is found before reaching the filesystem root.
    """
    current = start
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    msg = "Could not find project root (no pyproject.toml found)"
    raise RuntimeError(msg)
