# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The unused resource manager stays removed (OMN-21699)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_THIS_FILE = Path(__file__).resolve()
_REPO = _THIS_FILE.parents[3]
_NAME = b"HandlerResourceManager"
# Every tree a reintroduction could land in: package source and config, tests,
# and the ratchet baselines. Ticket contracts and the changelog record the
# removal itself, so they are not scanned.
_SCANNED_ROOTS = ("src", "tests", ".onex_ratchets")
_SKIPPED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCANNED_ROOTS:
        base = _REPO / root
        assert base.is_dir(), f"scan root missing: {base}"
        files.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and not _SKIPPED_DIRS.intersection(path.relative_to(_REPO).parts)
            and path.resolve() != _THIS_FILE
        )
    return files


@pytest.mark.unit
def test_resources_package_is_gone() -> None:
    assert (_REPO / "src/omnibase_infra/__init__.py").is_file()
    assert not (_REPO / "src/omnibase_infra/resources").exists()
    assert importlib.util.find_spec("omnibase_infra.resources") is None


@pytest.mark.unit
def test_no_tracked_tree_names_the_resource_manager() -> None:
    files = _scanned_files()
    assert any(path.name == "store_keyed_memory.py" for path in files)
    offenders = [
        str(path.relative_to(_REPO)) for path in files if _NAME in path.read_bytes()
    ]
    assert offenders == []
