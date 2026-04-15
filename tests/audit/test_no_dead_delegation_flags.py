# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression test: dead feature flags must not reappear in live config.

OMN-8779: ENABLE_DELEGATION_BRIDGE and ENABLE_LOCAL_DELEGATION were no-ops after
OMN-8746 made the Kafka delegation bridge unconditional.

OMN-8780: ENABLE_LOCAL_INFERENCE_PIPELINE and ENABLE_PATTERN_ENFORCEMENT violated
the no-informational-gates policy (defaulted to false = silent non-enforcement).
Removed to make both pipeline and pattern enforcement unconditional.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_DEAD_FLAGS = (
    "ENABLE_DELEGATION_BRIDGE",
    "ENABLE_LOCAL_DELEGATION",
    "ENABLE_LOCAL_INFERENCE_PIPELINE",
    "ENABLE_PATTERN_ENFORCEMENT",
)

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

# Guard tests that intentionally name the dead flags to assert their removal.
_GUARD_TEST_SUFFIXES = ("tests/integration/test_dead_flag_removal_omn_8780.py",)


def _is_guard_test(path: Path) -> bool:
    path_str = path.as_posix()
    return any(path_str.endswith(s) for s in _GUARD_TEST_SUFFIXES)


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
            if _is_guard_test(p):
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
        "Dead feature flags found in live config (OMN-8779, OMN-8780):\n"
        + "\n".join(violations)
    )
