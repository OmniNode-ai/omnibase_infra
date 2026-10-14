# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lock written by the release train's bump passes ``uv lock --check`` (OMN-21084).

Runs ``uv`` itself over a throwaway project with no dependencies, so no index is
read: ``uv lock`` writes the lock at one version, the train's ``bump-version``
moves the version, and ``uv lock --check`` is the verdict. The control bumps the
manifest alone, which is what the train did before, and ``uv`` must refuse it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]

_PYPROJECT = """\
[project]
name = "fixture-pkg"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []
"""


def _release_train() -> Any:
    name = "release_train_uv_lock_check_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = _ROOT / "scripts" / "ci" / "release_train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _uv(clone: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "lock", *args, "--offline"],
        cwd=clone,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def locked_clone(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    written = _uv(tmp_path)
    assert written.returncode == 0, written.stderr
    assert _uv(tmp_path, "--check").returncode == 0, "the fixture lock must start valid"
    return tmp_path


def test_a_lock_written_by_the_bump_passes_uv_lock_check(locked_clone: Path) -> None:
    module = _release_train()
    code = module.main(
        ["bump-version", "--clone", str(locked_clone), "--version", "0.1.1"]
    )
    assert code == 0
    checked = _uv(locked_clone, "--check")
    assert checked.returncode == 0, checked.stderr


def test_a_manifest_only_bump_is_refused_by_uv_lock_check(locked_clone: Path) -> None:
    """Control: the bump this replaces leaves the lock stale, and ``uv`` says so."""
    pyproject = locked_clone / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace('"0.1.0"', '"0.1.1"'),
        encoding="utf-8",
    )
    checked = _uv(locked_clone, "--check")
    assert checked.returncode != 0
    assert "needs to be updated" in checked.stderr
