# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Adversarial source-target tests for the architecture-layer script (OMN-17793)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "check_architecture.sh"


def _source_tree(tmp_path: Path) -> Path:
    package = tmp_path / "omnibase_core" / "src" / "omnibase_core"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__all__ = []\n")
    (tmp_path / "omnibase_core" / "pyproject.toml").write_text(
        "[project]\nname = 'omnibase-core'\nversion = '0.0.0'\n"
    )
    return package


def _run(
    path: Path, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--no-color", "--path", str(path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.unit
def test_valid_source_package_is_accepted(tmp_path: Path) -> None:
    result = _run(_source_tree(tmp_path))
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["missing", "arbitrary", "empty", "installed"])
def test_explicit_invalid_targets_fail_without_fallback(
    tmp_path: Path, kind: str
) -> None:
    if kind == "missing":
        target = tmp_path / "missing"
    elif kind == "arbitrary":
        target = tmp_path / "arbitrary"
        target.mkdir()
    elif kind == "empty":
        target = tmp_path / "omnibase_core" / "src" / "omnibase_core"
        target.mkdir(parents=True)
    else:
        target = (
            tmp_path
            / ".venv"
            / "lib"
            / "python3.12"
            / "site-packages"
            / "omnibase_core"
        )
        target.mkdir(parents=True)
        (target / "__init__.py").write_text("\n")

    result = _run(target)
    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
def test_invalid_environment_override_does_not_fall_back(tmp_path: Path) -> None:
    invalid = tmp_path / "empty"
    invalid.mkdir()
    env = os.environ.copy()
    env["OMNIBASE_CORE_PATH"] = str(invalid)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--no-color"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert "OMNIBASE_CORE_PATH" in result.stderr


@pytest.mark.unit
def test_zero_python_file_guard_is_fail_closed() -> None:
    text = SCRIPT.read_text()
    assert "JSON_EXIT_CODE=2" in text
    assert "No Python files found in source target" in text
    assert "exit 2" in text
