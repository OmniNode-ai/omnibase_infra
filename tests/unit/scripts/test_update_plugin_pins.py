# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for update-plugin-pins.py (OMN-3287)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Load the hyphen-named script as a module
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "update_plugin_pins",
        _SCRIPTS_DIR / "update-plugin-pins.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["update_plugin_pins"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_script()
rewrite_content = _mod.rewrite_content
fetch_latest_version = _mod.fetch_latest_version
main = _mod.main


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_DOCKERFILE_RANGE_PINS = """\
FROM python:3.12-slim AS builder
RUN --mount=type=cache,target=/root/.cache/uv \\
    uv pip install --no-deps \\
    "omninode-claude>=0.3.0,<0.5.0"
RUN --mount=type=cache,target=/root/.cache/uv \\
    uv pip install --no-deps \\
    "omninode-memory>=0.6.0,<0.8.0"
CMD ["onex-runtime"]
"""

_DOCKERFILE_EXACT_PINS = """\
FROM python:3.12-slim AS builder
RUN --mount=type=cache,target=/root/.cache/uv \\
    uv pip install --no-deps \\
    "omninode-claude==1.2.3"
RUN --mount=type=cache,target=/root/.cache/uv \\
    uv pip install --no-deps \\
    "omninode-memory==2.3.4"
CMD ["onex-runtime"]
"""

_VERSIONS = {
    "omninode-claude": "1.2.3",
    "omninode-memory": "2.3.4",
}


# ---------------------------------------------------------------------------
# test_pin_rewrite
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pin_rewrite() -> None:
    """Given known version strings, rewrite_content rewrites Dockerfile lines."""
    result = rewrite_content(_DOCKERFILE_RANGE_PINS, _VERSIONS)

    assert '"omninode-claude==1.2.3"' in result
    assert '"omninode-memory==2.3.4"' in result
    # Original range pins should be gone
    assert ">=0.3.0,<0.5.0" not in result
    assert ">=0.6.0,<0.8.0" not in result
    # Unrelated lines unchanged
    assert "FROM python:3.12-slim AS builder" in result
    assert 'CMD ["onex-runtime"]' in result


# ---------------------------------------------------------------------------
# test_dry_run_no_write
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dry_run_no_write(tmp_path: Path) -> None:
    """--dry-run does not modify the Dockerfile."""
    dockerfile = tmp_path / "Dockerfile.runtime"
    dockerfile.write_text(_DOCKERFILE_RANGE_PINS, encoding="utf-8")

    with patch.object(
        _mod,
        "fetch_latest_version",
        side_effect=lambda pkg: _VERSIONS[pkg],
    ):
        exit_code = main(["--dry-run", "--dockerfile", str(dockerfile)])

    assert exit_code == 0
    # File must be untouched
    assert dockerfile.read_text(encoding="utf-8") == _DOCKERFILE_RANGE_PINS


# ---------------------------------------------------------------------------
# test_no_change_when_current
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_change_when_current(tmp_path: Path) -> None:
    """No-op when the Dockerfile already has exact pins matching latest versions."""
    dockerfile = tmp_path / "Dockerfile.runtime"
    dockerfile.write_text(_DOCKERFILE_EXACT_PINS, encoding="utf-8")

    with patch.object(
        _mod,
        "fetch_latest_version",
        side_effect=lambda pkg: _VERSIONS[pkg],
    ):
        exit_code = main(["--dockerfile", str(dockerfile)])

    assert exit_code == 0
    # Content unchanged
    assert dockerfile.read_text(encoding="utf-8") == _DOCKERFILE_EXACT_PINS


# ---------------------------------------------------------------------------
# test_pypi_fetch_failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pypi_fetch_failure(tmp_path: Path) -> None:
    """main() returns non-zero exit code on PyPI network error."""
    import urllib.error

    dockerfile = tmp_path / "Dockerfile.runtime"
    dockerfile.write_text(_DOCKERFILE_RANGE_PINS, encoding="utf-8")

    with patch.object(
        _mod,
        "fetch_latest_version",
        side_effect=RuntimeError(
            "Failed to fetch PyPI metadata for 'omninode-claude': <URLError>"
        ),
    ):
        exit_code = main(["--dockerfile", str(dockerfile)])

    assert exit_code != 0
    # File must not have been modified
    assert dockerfile.read_text(encoding="utf-8") == _DOCKERFILE_RANGE_PINS
