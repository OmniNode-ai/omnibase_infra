# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for update-plugin-pins.py (OMN-3287)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

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
    "omninode-claude>=0.3.0,<2.0.0"
RUN --mount=type=cache,target=/root/.cache/uv \\
    uv pip install --no-deps \\
    "omninode-memory>=0.6.0,<3.0.0"
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
    "omninode-intelligence": "3.4.5",
    "omninode-memory": "2.3.4",
}


# ---------------------------------------------------------------------------
# test_pin_rewrite
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pin_rewrite() -> None:
    """A bounded range keeps its form: the floor advances, the ceiling stays.

    OMN-18596: the runtime Dockerfile moved to bounded ranges under OMN-18595
    (omnibase_infra#3749) because exact ``--no-deps`` pins fail the Dockerfile
    plugin pin validator as soon as a newer plugin publishes. Rewriting a range
    into ``==`` reverted that decision on every cascade (omnibase_infra#4168).
    """
    result = rewrite_content(_DOCKERFILE_RANGE_PINS, _VERSIONS)

    assert '"omninode-claude>=1.2.3,<2.0.0"' in result
    assert '"omninode-memory>=2.3.4,<3.0.0"' in result
    assert "omninode-claude==" not in result
    assert "omninode-memory==" not in result
    # Original floors should be gone
    assert ">=0.3.0,<2.0.0" not in result
    assert ">=0.6.0,<3.0.0" not in result
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


# ---------------------------------------------------------------------------
# test_intelligence_in_plugins_list (OMN-5374)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_intelligence_in_plugins_list() -> None:
    """omninode-intelligence must be in the PLUGINS list."""
    assert "omninode-intelligence" in _mod.PLUGINS


@pytest.mark.unit
def test_pin_re_matches_intelligence() -> None:
    """_PIN_RE must match omninode-intelligence package references."""
    line = 'RUN pip install "omninode-intelligence==0.5.0"'
    m = _mod._PIN_RE.search(line)
    assert m is not None
    assert m.group("pkg") == "omninode-intelligence"


@pytest.mark.unit
def test_fetch_sends_cache_bust_headers() -> None:
    """fetch_latest_version must send Cache-Control and Pragma headers (OMN-6811)."""
    import urllib.request

    captured_request: list[urllib.request.Request] = []
    original_urlopen = urllib.request.urlopen

    def mock_urlopen(req: urllib.request.Request, **kwargs: object) -> object:  # type: ignore[override]
        captured_request.append(req)
        raise urllib.error.URLError("test — not a real request")

    with patch.object(urllib.request, "urlopen", side_effect=mock_urlopen):
        with pytest.raises(RuntimeError):
            fetch_latest_version("omninode-claude")

    assert len(captured_request) == 1
    req = captured_request[0]
    assert isinstance(req, urllib.request.Request)
    assert "no-cache" in req.get_header("Cache-control")
    assert req.get_header("Pragma") == "no-cache"


@pytest.mark.unit
def test_rewrite_updates_all_three_plugins() -> None:
    """rewrite_content updates claude, intelligence, and memory pins."""
    content = (
        'RUN pip install "omninode-claude==0.1.0"\n'
        'RUN pip install "omninode-intelligence==0.2.0"\n'
        'RUN pip install "omninode-memory==0.3.0"\n'
    )
    result = rewrite_content(content, _VERSIONS)
    assert '"omninode-claude==1.2.3"' in result
    assert '"omninode-intelligence==3.4.5"' in result
    assert '"omninode-memory==2.3.4"' in result


# ---------------------------------------------------------------------------
# OMN-18596: bounded ranges keep their ceiling, and a release above it refuses
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_live_runtime_range_shape_advances_only_the_floor() -> None:
    content = (
        "    uv-with-retry pip install --no-deps \\\n"
        '    "omninode-claude>=0.25.1,<1.0.0" \\\n'
        '    "omninode-memory>=0.18.2,<1.0.0" \\\n'
        '    "omninode-intelligence>=0.24.0,<1.0.0"\n'
    )
    versions = {
        "omninode-claude": "0.26.0",
        "omninode-memory": "0.18.3",
        "omninode-intelligence": "0.24.0",
    }
    result = rewrite_content(content, versions)
    assert '"omninode-claude>=0.26.0,<1.0.0"' in result
    assert '"omninode-memory>=0.18.3,<1.0.0"' in result
    assert '"omninode-intelligence>=0.24.0,<1.0.0"' in result
    assert "==" not in result


@pytest.mark.unit
def test_a_release_at_or_above_the_ceiling_refuses() -> None:
    content = '    "omninode-memory>=0.18.2,<1.0.0"\n'
    with pytest.raises(ValueError, match="ceiling"):
        rewrite_content(content, {"omninode-memory": "1.0.0"})


@pytest.mark.unit
def test_ceiling_refusal_fails_the_run_without_writing(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile.runtime"
    original = '    "omninode-memory>=0.18.2,<1.0.0"\n'
    dockerfile.write_text(original, encoding="utf-8")
    with patch.object(_mod, "fetch_latest_version", return_value="1.0.0"):
        exit_code = _mod.main(["--dockerfile", str(dockerfile)])
    assert exit_code != 0
    assert dockerfile.read_text(encoding="utf-8") == original


@pytest.mark.unit
def test_exact_pins_stay_exact() -> None:
    result = rewrite_content(_DOCKERFILE_EXACT_PINS, {"omninode-claude": "1.2.4"})
    assert '"omninode-claude==1.2.4"' in result
