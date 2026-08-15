# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for the PyPI pin-resolvability release gate (OMN-14070).

``omnimarket`` 0.4.6 was published pinning a nonexistent
``omnibase-compat==0.5.1`` (OMN-14064) because no gate anywhere in the 5
PyPI-publishing repos ever resolved a package's declared
``[project.dependencies]`` pins against the real PyPI index. These tests
prove ``verify_pypi_pin_resolvability``:

* fails closed (unit) when ``dist/`` doesn't contain exactly one wheel,
* fails RED (integration, real PyPI) on a deliberately-broken pin -- the
  exact OMN-14064 failure shape (a version that does not exist on PyPI), and
* passes GREEN (integration, real PyPI) once the pin is a resolvable range.

OMN-16047 adds the timeout dimension. The gate installs the whole transitive
closure with ``--no-cache``, so on a saturated fleet it can exceed its wall-clock
budget without any pin being wrong. The original code let
``subprocess.TimeoutExpired`` escape, which (a) discarded the partial ``uv``
output and (b) presented a throughput failure in language reserved for an
unresolvable pin. These tests pin the distinction.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess  # nosec B404 - invokes `uv build` with a fixed argv in tests
from pathlib import Path

import pytest

import scripts.ci.verify_pypi_pin_resolvability as pin_gate
from scripts.ci.verify_pypi_pin_resolvability import (
    PinResolveTimeoutError,
    find_single_wheel,
    verify_pin_resolvability,
)

_UV_BIN = shutil.which("uv")

# ---------------------------------------------------------------------------
# find_single_wheel -- unit, no network
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_single_wheel_raises_when_dist_is_empty(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no wheel"):
        find_single_wheel(tmp_path)


@pytest.mark.unit
def test_find_single_wheel_raises_on_more_than_one_wheel(tmp_path: Path) -> None:
    (tmp_path / "a-1.0-py3-none-any.whl").touch()
    (tmp_path / "b-1.0-py3-none-any.whl").touch()
    with pytest.raises(SystemExit, match="expected exactly one wheel"):
        find_single_wheel(tmp_path)


@pytest.mark.unit
def test_find_single_wheel_ignores_sdist_and_returns_the_wheel(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    wheel.write_bytes(b"")
    (tmp_path / "pkg-1.0.tar.gz").write_bytes(b"")
    assert find_single_wheel(tmp_path) == wheel


# ---------------------------------------------------------------------------
# verify_pin_resolvability -- integration, hits the real PyPI index
# ---------------------------------------------------------------------------


def _build_fixture_wheel(fixture_dir: Path, dependency_pin: str) -> Path:
    """Build a real, minimal wheel declaring exactly one dependency pin."""
    (fixture_dir / "pyproject.toml").write_text(
        f"""
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "omn14070-pin-fixture"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["{dependency_pin}"]

[tool.hatch.build.targets.wheel]
packages = ["src/omn14070_pin_fixture"]
"""
    )
    pkg_dir = fixture_dir / "src" / "omn14070_pin_fixture"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")

    assert _UV_BIN is not None, "`uv` not found on PATH"
    subprocess.run(  # nosec B603 - fixed argv, no shell, fully-qualified uv path, test-only
        [_UV_BIN, "build", "--wheel"],
        cwd=fixture_dir,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return find_single_wheel(fixture_dir / "dist")


@pytest.mark.integration
def test_broken_pin_fails_red_reproduces_omn_14064(tmp_path: Path) -> None:
    """A pin on a version that does not exist on PyPI must fail this gate --
    this is structurally the omnimarket 0.4.6 / omnibase-compat==0.5.1 shape.
    """
    wheel = _build_fixture_wheel(tmp_path, "requests==99.99.99")

    ok, log = verify_pin_resolvability(wheel)

    assert ok is False
    assert "requests" in log.lower()
    assert "99.99.99" in log


@pytest.mark.integration
def test_resolvable_pin_passes_green(tmp_path: Path) -> None:
    """Once the pin names a version range that actually exists on PyPI, the
    same gate passes cleanly.
    """
    wheel = _build_fixture_wheel(tmp_path, "requests>=2,<3")

    ok, log = verify_pin_resolvability(wheel)

    assert ok is True, log


# ---------------------------------------------------------------------------
# Timeout handling (OMN-16047) -- unit, no network
# ---------------------------------------------------------------------------


def _stub_runs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    install_raises: subprocess.TimeoutExpired | None,
) -> None:
    """Replace the module's ``subprocess.run`` so no real ``uv`` is invoked.

    ``uv venv`` always succeeds (and its side effect of creating the venv
    directory is irrelevant to these assertions); the ``uv pip install`` call
    raises whatever the caller supplies.
    """

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "venv" in argv:
            return subprocess.CompletedProcess(argv, 0, "venv-created\n", "")
        if install_raises is not None:
            raise install_raises
        return subprocess.CompletedProcess(argv, 0, "installed\n", "")

    monkeypatch.setattr(pin_gate.subprocess, "run", fake_run)


@pytest.mark.unit
def test_install_timeout_raises_typed_error_and_keeps_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow install must surface as ``PinResolveTimeoutError`` carrying whatever
    ``uv`` had already emitted -- not a bare ``TimeoutExpired`` traceback that
    throws the diagnostic output away. This is the v0.38.4 failure shape.
    """
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    wheel.write_bytes(b"")
    _stub_runs(
        monkeypatch,
        install_raises=subprocess.TimeoutExpired(
            cmd=["uv", "pip", "install"],
            timeout=pin_gate._INSTALL_TIMEOUT_SECONDS,
            output=b"Resolved 135 packages\nDownloading grpcio\n",
        ),
    )

    with pytest.raises(PinResolveTimeoutError) as caught:
        verify_pin_resolvability(wheel)

    assert caught.value.step == "uv pip install"
    assert caught.value.budget_seconds == pin_gate._INSTALL_TIMEOUT_SECONDS
    assert "Downloading grpcio" in caught.value.partial_output
    assert "venv-created" in caught.value.prior_output


@pytest.mark.unit
def test_timeout_is_not_reported_as_an_unresolvable_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole point of this gate is to name a bad pin. A timeout says nothing
    about the pins, so the report must not borrow that language -- otherwise a
    saturated runner reads as a broken release, which is what sent v0.38.4's
    diagnosis down the wrong path.
    """
    (tmp_path / "pkg-1.0-py3-none-any.whl").write_bytes(b"")
    _stub_runs(
        monkeypatch,
        install_raises=subprocess.TimeoutExpired(
            cmd=["uv", "pip", "install"], timeout=1800
        ),
    )

    assert pin_gate.main([str(tmp_path)]) == 1

    out = capsys.readouterr().out
    assert "THROUGHPUT failure" in out
    assert "do not resolve" not in out
    assert pin_gate._INSTALL_TIMEOUT_ENV_VAR in out


@pytest.mark.unit
def test_venv_creation_has_its_own_smaller_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``uv venv`` is purely local work. It must not be able to consume the
    install's (much larger) network budget before the install even starts.
    """
    assert pin_gate._VENV_TIMEOUT_SECONDS < pin_gate._INSTALL_TIMEOUT_SECONDS

    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    wheel.write_bytes(b"")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == pin_gate._VENV_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(
            cmd=argv, timeout=pin_gate._VENV_TIMEOUT_SECONDS
        )

    monkeypatch.setattr(pin_gate.subprocess, "run", fake_run)

    with pytest.raises(PinResolveTimeoutError) as caught:
        verify_pin_resolvability(wheel)

    assert caught.value.step == "uv venv"


@pytest.mark.unit
def test_install_budget_is_overridable_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fleet's throughput is not a property of this repo, so the budget has
    to be tunable from the workflow without editing the script.
    """
    monkeypatch.setenv(pin_gate._INSTALL_TIMEOUT_ENV_VAR, "2400")
    reloaded = importlib.reload(pin_gate)
    try:
        assert reloaded._INSTALL_TIMEOUT_SECONDS == 2400
    finally:
        monkeypatch.delenv(pin_gate._INSTALL_TIMEOUT_ENV_VAR, raising=False)
        importlib.reload(pin_gate)


@pytest.mark.unit
def test_default_install_budget_clears_the_measured_fleet_floor() -> None:
    """A *cached* uv sync on the omnibase-ci fleet was measured at 110-402s
    (OMN-16047). An uncached 242 MB / 135-package install cannot fit inside the
    old 300s ceiling, so the default must stay well clear of it.
    """
    assert pin_gate._INSTALL_TIMEOUT_SECONDS >= 1800
