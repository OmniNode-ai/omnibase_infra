# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/check-pinned-wheels.py (OMN-9331)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check-pinned-wheels.py"


def _load_module() -> Any:
    """Load the script as a module without executing __main__."""
    mod_name = "check_pinned_wheels"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module so @dataclass can resolve cls.__module__
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_mod = _load_module()


# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_version_strips_v_prefix() -> None:
    assert _mod._parse_version("v1.2.3") == (1, 2, 3)


@pytest.mark.unit
def test_parse_version_no_prefix() -> None:
    assert _mod._parse_version("0.41.0") == (0, 41, 0)


@pytest.mark.unit
def test_parse_version_short() -> None:
    assert _mod._parse_version("v1.0") == (1, 0)


@pytest.mark.unit
def test_parse_version_invalid_returns_empty() -> None:
    assert _mod._parse_version("abc") == ()


# ---------------------------------------------------------------------------
# _minor_behind
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_minor_behind_same_version() -> None:
    assert _mod._minor_behind("v0.41.0", "v0.41.0") == 0


@pytest.mark.unit
def test_minor_behind_one_minor() -> None:
    assert _mod._minor_behind("v0.40.0", "v0.41.0") == 1


@pytest.mark.unit
def test_minor_behind_two_minor() -> None:
    assert _mod._minor_behind("v0.39.0", "v0.41.0") == 2


@pytest.mark.unit
def test_minor_behind_major_diff_returns_999() -> None:
    assert _mod._minor_behind("v0.41.0", "v1.0.0") == 999


@pytest.mark.unit
def test_minor_behind_pinned_ahead_returns_zero() -> None:
    assert _mod._minor_behind("v0.42.0", "v0.41.0") == 0


# ---------------------------------------------------------------------------
# _parse_dockerfile_args
# ---------------------------------------------------------------------------

_SAMPLE_DOCKERFILE = """
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.6.14
ARG OMNIBASE_COMPAT_REF="c1a878f1339d396a7f11dee42ea9f0e30fb9c1d1"
ARG OMNIBASE_COMPAT_SOURCE="https://github.com/OmniNode-ai/omnibase_compat/archive/c1a878f1339d396a7f11dee42ea9f0e30fb9c1d1.tar.gz"
ARG ONEX_CHANGE_CONTROL_REF=main
ARG OMNIMARKET_REF=main
"""


@pytest.mark.unit
def test_parse_dockerfile_args_extracts_known_pins(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE)
    result = _mod._parse_dockerfile_args(df)
    assert result["OMNIBASE_COMPAT_REF"] == "c1a878f1339d396a7f11dee42ea9f0e30fb9c1d1"
    assert result["ONEX_CHANGE_CONTROL_REF"] == "main"
    assert result["OMNIMARKET_REF"] == "main"
    assert result["UV_VERSION"] == "0.6.14"


# ---------------------------------------------------------------------------
# _parse_uv_sources
# ---------------------------------------------------------------------------

_SAMPLE_PYPROJECT = """
[tool.uv.sources]
omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", rev = "7631ef3321e1a6561f4b2c70019ee57d24160620" }
omnibase-spi = { git = "https://github.com/OmniNode-ai/omnibase_spi.git", tag = "v0.21.0" }
onex-change-control = { git = "https://github.com/OmniNode-ai/onex_change_control.git", branch = "main" }

[other.section]
foo = "bar"
"""


@pytest.mark.unit
def test_parse_uv_sources_extracts_rev() -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp:
        tmp.write(_SAMPLE_PYPROJECT)
        tmp_path = Path(tmp.name)
    try:
        result = _mod._parse_uv_sources(tmp_path)
        assert "omnibase-core" in result
        assert (
            result["omnibase-core"]["rev"] == "7631ef3321e1a6561f4b2c70019ee57d24160620"
        )
        assert result["omnibase-spi"]["tag"] == "v0.21.0"
        assert result["onex-change-control"]["branch"] == "main"
    finally:
        tmp_path.unlink()


@pytest.mark.unit
def test_parse_uv_sources_missing_section(tmp_path: Path) -> None:
    pj = tmp_path / "pyproject.toml"
    pj.write_text("[project]\nname = 'foo'\n")
    assert _mod._parse_uv_sources(pj) == {}


# ---------------------------------------------------------------------------
# _check_dockerfile_args
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_dockerfile_args_flags_main_ref(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE)
    df_args = _mod._parse_dockerfile_args(df)

    with patch.object(_mod, "_gh_latest_release", return_value="v1.2.0"):
        results = _mod._check_dockerfile_args(df_args, threshold=1)

    main_results = [r for r in results if r.pinned == "main"]
    assert all(r.drifted for r in main_results), "floating 'main' refs must be drifted"
    names = {r.name for r in main_results}
    assert "ONEX_CHANGE_CONTROL_REF" in names
    assert "OMNIMARKET_REF" in names


@pytest.mark.unit
def test_check_dockerfile_args_sha_with_release_flags_drift(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE)
    df_args = _mod._parse_dockerfile_args(df)

    # Latest release exists; the pinned SHA is not the tag — drift expected
    with patch.object(_mod, "_gh_latest_release", return_value="v0.5.0"):
        results = _mod._check_dockerfile_args(df_args, threshold=1)

    compat_results = [r for r in results if r.name == "OMNIBASE_COMPAT_REF"]
    assert len(compat_results) == 1
    assert compat_results[0].drifted is True


@pytest.mark.unit
def test_check_dockerfile_args_github_unavailable(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE)
    df_args = _mod._parse_dockerfile_args(df)

    with patch.object(_mod, "_gh_latest_release", return_value=None):
        results = _mod._check_dockerfile_args(df_args, threshold=1)

    # All results should mark check_failed when GitHub is unreachable
    assert all(r.check_failed for r in results)


# ---------------------------------------------------------------------------
# _check_uv_sources
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_uv_sources_branch_is_always_drifted() -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp:
        tmp.write(_SAMPLE_PYPROJECT)
        tmp_path = Path(tmp.name)
    try:
        sources = _mod._parse_uv_sources(tmp_path)
        with patch.object(_mod, "_gh_latest_release", return_value="v0.5.0"):
            results = _mod._check_uv_sources(sources, threshold=1)

        occ = next(r for r in results if r.name == "onex-change-control")
        assert occ.drifted is True
        assert "branch" in occ.pinned
    finally:
        tmp_path.unlink()


@pytest.mark.unit
def test_check_uv_sources_tag_current_not_drifted() -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp:
        tmp.write(_SAMPLE_PYPROJECT)
        tmp_path = Path(tmp.name)
    try:
        sources = _mod._parse_uv_sources(tmp_path)
        # Latest release matches the pinned tag exactly
        with patch.object(_mod, "_gh_latest_release", return_value="v0.21.0"):
            results = _mod._check_uv_sources(sources, threshold=1)

        spi = next(r for r in results if r.name == "omnibase-spi")
        assert spi.drifted is False
    finally:
        tmp_path.unlink()


@pytest.mark.unit
def test_check_uv_sources_tag_behind_threshold_drifted() -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp:
        tmp.write(_SAMPLE_PYPROJECT)
        tmp_path = Path(tmp.name)
    try:
        sources = _mod._parse_uv_sources(tmp_path)
        # Latest is v0.23.0 — three minors ahead of pinned v0.21.0
        with patch.object(_mod, "_gh_latest_release", return_value="v0.23.0"):
            results = _mod._check_uv_sources(sources, threshold=1)

        spi = next(r for r in results if r.name == "omnibase-spi")
        assert spi.drifted is True
        assert spi.minor_behind == 2
    finally:
        tmp_path.unlink()


@pytest.mark.unit
def test_check_uv_sources_pre_release_sha_flagged() -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp:
        tmp.write(_SAMPLE_PYPROJECT)
        tmp_path = Path(tmp.name)
    try:
        sources = _mod._parse_uv_sources(tmp_path)
        with patch.object(_mod, "_gh_latest_release", return_value="v0.42.0"):
            results = _mod._check_uv_sources(sources, threshold=1)

        core = next(r for r in results if r.name == "omnibase-core")
        # Pre-release commit SHA when a tagged release exists → drifted
        assert core.drifted is True
        assert "pre-release commit" in core.message
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# _check_pypi_range_pins
# ---------------------------------------------------------------------------

_SAMPLE_DOCKERFILE_PYPI = """
RUN uv pip install --no-deps \\
    "omninode-claude>=0.9.0,<2.0.0" \\
    "omninode-memory>=0.9.0,<2.0.0" \\
    "omninode-intelligence>=0.15.0,<2.0.0"
"""


@pytest.mark.unit
def test_check_pypi_range_pins_current(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE_PYPI)

    # Use a same-major latest so minor_behind < threshold (0.9.x → 0.10.0 = 1 minor)
    with patch.object(_mod, "_pypi_latest", return_value="0.10.0"):
        results = _mod._check_pypi_range_pins(df, threshold=1)

    assert len(results) == 3
    assert all(not r.drifted for r in results)


@pytest.mark.unit
def test_check_pypi_range_pins_lower_bound_far_behind(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE_PYPI)

    # PyPI latest is 0.15.0; lower bound >=0.9.0 is 6 minors behind
    with patch.object(_mod, "_pypi_latest", return_value="0.15.0"):
        results = _mod._check_pypi_range_pins(df, threshold=1)

    claude = next(r for r in results if r.name == "omninode-claude")
    assert claude.drifted is True
    assert claude.minor_behind >= 2


@pytest.mark.unit
def test_check_pypi_range_pins_unavailable(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE_PYPI)

    with patch.object(_mod, "_pypi_latest", return_value=None):
        results = _mod._check_pypi_range_pins(df, threshold=1)

    assert all(r.check_failed for r in results)


# ---------------------------------------------------------------------------
# main() integration — JSON output and exit codes
# ---------------------------------------------------------------------------


_SAMPLE_DOCKERFILE_NO_FLOATING = """
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.6.14
ARG OMNIBASE_COMPAT_REF="c1a878f1339d396a7f11dee42ea9f0e30fb9c1d1"
ARG OMNIBASE_COMPAT_SOURCE="https://github.com/OmniNode-ai/omnibase_compat/archive/c1a878f1339d396a7f11dee42ea9f0e30fb9c1d1.tar.gz"
ARG ONEX_CHANGE_CONTROL_REF=abc123def456789012345678901234567890abcd
ARG OMNIMARKET_REF=deadbeef1234567890abcdef1234567890abcdef
"""

# pyproject with no floating branch refs — all pinned to tags or SHAs
_SAMPLE_PYPROJECT_NO_FLOATING = """
[tool.uv.sources]
omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", rev = "7631ef3321e1a6561f4b2c70019ee57d24160620" }
omnibase-spi = { git = "https://github.com/OmniNode-ai/omnibase_spi.git", tag = "v0.21.0" }
onex-change-control = { git = "https://github.com/OmniNode-ai/onex_change_control.git", tag = "v0.3.0" }
"""


@pytest.mark.unit
def test_main_json_output_network_down_exits_2(tmp_path: Path) -> None:
    # No floating refs; all checks fail due to no network → exit 2 (not 1)
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE_NO_FLOATING)
    pj = tmp_path / "pyproject.toml"
    pj.write_text(_SAMPLE_PYPROJECT_NO_FLOATING)

    with (
        patch.object(_mod, "_gh_latest_release", return_value=None),
        patch.object(_mod, "_pypi_latest", return_value=None),
    ):
        rc = _mod.main(
            [
                "--dockerfile",
                str(df),
                "--pyproject",
                str(pj),
                "--json",
            ]
        )

    # All checks failed (network down), no confirmed drift → exit 2
    assert rc == 2


@pytest.mark.unit
def test_main_exits_1_on_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE)
    pj = tmp_path / "pyproject.toml"
    pj.write_text(_SAMPLE_PYPROJECT)

    with (
        patch.object(_mod, "_gh_latest_release", return_value="v99.0.0"),
        patch.object(_mod, "_pypi_latest", return_value="99.0.0"),
    ):
        rc = _mod.main(
            [
                "--dockerfile",
                str(df),
                "--pyproject",
                str(pj),
            ]
        )

    assert rc == 1


@pytest.mark.unit
def test_main_json_is_valid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    df = tmp_path / "Dockerfile.runtime"
    df.write_text(_SAMPLE_DOCKERFILE + _SAMPLE_DOCKERFILE_PYPI)
    pj = tmp_path / "pyproject.toml"
    pj.write_text(_SAMPLE_PYPROJECT)

    with (
        patch.object(_mod, "_gh_latest_release", return_value="v0.21.0"),
        patch.object(_mod, "_pypi_latest", return_value="1.0.0"),
    ):
        _mod.main(
            [
                "--dockerfile",
                str(df),
                "--pyproject",
                str(pj),
                "--json",
            ]
        )

    out = capsys.readouterr().out
    data = json.loads(out)
    assert "results" in data
    assert isinstance(data["results"], list)
    assert "drifted" in data


@pytest.mark.unit
def test_main_missing_dockerfile(tmp_path: Path) -> None:
    rc = _mod.main(
        [
            "--dockerfile",
            str(tmp_path / "nonexistent.Dockerfile"),
            "--pyproject",
            str(tmp_path / "pyproject.toml"),
        ]
    )
    assert rc == 1


@pytest.mark.unit
def test_main_help_exits_zero() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--dockerfile" in result.stdout
    assert "--json" in result.stdout
    assert "--threshold" in result.stdout
