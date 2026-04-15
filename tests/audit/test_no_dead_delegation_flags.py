# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression test: dead delegation feature flags must not reappear in live config.

OMN-8779: ENABLE_DELEGATION_BRIDGE and ENABLE_LOCAL_DELEGATION were no-ops after
OMN-8746 made the Kafka delegation bridge unconditional. These flags were removed
to prevent documentation lies from misleading future contributors.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_DEAD_FLAGS = ("ENABLE_DELEGATION_BRIDGE", "ENABLE_LOCAL_DELEGATION")

_LIVE_CONFIG_GLOBS = (
    "**/*.env",
    "**/*.env.*",
    "**/*.yaml",
    "**/*.yml",
    "**/*.py",
    "**/*.sh",
    "**/*.toml",
    "**/*.cfg",
    "**/*.txt",
)

_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
}

_HISTORICAL_SUFFIXES = (
    "docs/plans",
    "docs/sessions",
    "docs/decisions",
    "CHANGELOG",
    "changelog",
)


def _is_historical(path: Path) -> bool:
    path_str = str(path)
    return any(h in path_str for h in _HISTORICAL_SUFFIXES)


_SELF = Path(__file__)


def _collect_live_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for glob in _LIVE_CONFIG_GLOBS:
        for p in root.glob(glob):
            if not p.is_file():
                continue
            if p.resolve() == _SELF.resolve():
                continue
            if any(ex in p.parts for ex in _EXCLUDED_DIRS):
                continue
            if _is_historical(p):
                continue
            found.append(p)
    return found


def test_no_dead_delegation_flags_in_live_config() -> None:
    """Assert dead delegation flags do not exist in any live config or source file."""
    repo_root = Path(__file__).parent.parent.parent
    pattern = re.compile("|".join(re.escape(f) for f in _DEAD_FLAGS))

    violations: list[str] = []
    for path in _collect_live_files(repo_root):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                violations.append(
                    f"{path.relative_to(repo_root)}:{lineno}: {line.strip()}"
                )

    assert not violations, (
        "Dead delegation flags found in live config (OMN-8779):\n"
        + "\n".join(violations)
    )
