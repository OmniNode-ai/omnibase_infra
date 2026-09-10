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


def _is_live_config_file(root: Path, path: Path) -> bool:
    """Match the existing live-config glob set without repeated traversal."""
    relative = path.relative_to(root)
    return any(
        relative.match(glob.removeprefix("**/"))
        if glob.startswith("**/")
        else relative.match(glob)
        for glob in _LIVE_CONFIG_GLOBS
    )


def _collect_live_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for directory, dirnames, filenames in root.walk():
        dirnames[:] = [name for name in dirnames if name not in _EXCLUDED_DIRS]
        for filename in filenames:
            p = directory / filename
            if not p.is_file() or not _is_live_config_file(root, p):
                continue
            if p.resolve() == _SELF.resolve():
                continue
            if _is_historical(p):
                continue
            if _is_guard_test(p):
                continue
            found.append(p)
    return found


def test_collect_live_files_prunes_excluded_tree_and_keeps_untracked_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basenames = (
        ".env",
        ".env.local",
        "name.env",
        "name.env.local",
        ".py",
        ".yaml",
        ".yml",
        ".cfg",
        ".txt",
        ".sh",
        ".toml",
        "module.py",
        "config.yaml",
        "config.yml",
        "run.sh",
        "config.toml",
        "config.cfg",
        "notes.txt",
    )
    matches = [
        parent / name
        for parent in (tmp_path, tmp_path / "nested")
        for name in basenames
    ]
    included_root = tmp_path / "module.py"
    included_nested = tmp_path / "nested" / "config.yaml"
    untracked_match = tmp_path / ".env.local"
    excluded = tmp_path / ".venv" / "ignored.py"
    for path in (included_root, included_nested, untracked_match, *matches, excluded):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")

    legacy = {
        path.relative_to(tmp_path)
        for glob in _LIVE_CONFIG_GLOBS
        for path in tmp_path.glob(glob)
        if path.is_file() and ".venv" not in path.parts
    }
    visited: list[Path] = []
    original_walk = Path.walk

    def recording_walk(path: Path) -> object:
        for entry in original_walk(path):
            visited.append(entry[0])
            yield entry

    monkeypatch.setattr(Path, "walk", recording_walk)
    found = {path.relative_to(tmp_path) for path in _collect_live_files(tmp_path)}

    assert found == legacy
    assert included_root.relative_to(tmp_path) in found
    assert included_nested.relative_to(tmp_path) in found
    assert untracked_match.relative_to(tmp_path) in found
    assert tmp_path in visited
    assert included_nested.parent in visited
    assert excluded.parent not in visited


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
