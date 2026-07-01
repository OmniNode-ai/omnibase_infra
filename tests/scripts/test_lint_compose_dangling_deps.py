# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/lint_compose_dangling_deps.py.

OMN-13037 — Retro B-9: compose-config lint for dangling credential/address
env vars injected by an overlay for a service that same overlay disables.

Fail-closed proof:
  - known-bad fixture (disabled infisical + INFISICAL_ADDR injected) → exit 1
    naming the offending container+env in stdout.
  - clean fixture (infisical disabled, no dangling env) → exit 0.
  - real overlay files pass the linter (no regressions introduced).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "lint_compose_dangling_deps.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    *args: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=cwd or _REPO_ROOT,
        check=False,
        text=True,
        capture_output=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


# ---------------------------------------------------------------------------
# Fixture YAML builders
# ---------------------------------------------------------------------------

_CLEAN_OVERLAY = """\
name: test-clean-overlay
services:
  infisical:
    profiles: !override ["test-disabled"]
  omninode-runtime:
    environment:
      ONEX_ENVIRONMENT: test
      INFISICAL_ADDR: ""
"""

_BAD_OVERLAY_HARDCODED_VALUE = """\
name: test-bad-overlay-hardcoded
services:
  infisical:
    profiles: !override ["test-disabled"]
  omninode-runtime:
    environment:
      ONEX_ENVIRONMENT: test
      INFISICAL_ADDR: "http://infisical:8080"
"""

_BAD_OVERLAY_HOSTNAME_ONLY = """\
name: test-bad-overlay-hostname
services:
  vault:
    profiles: !override ["test-disabled"]
  some-service:
    environment:
      VAULT_ADDR: "http://vault:8200"
      OTHER_HOST: vault
"""

_CLEAN_OVERLAY_EMPTY_DEFAULT = """\
name: test-clean-empty-default
services:
  infisical:
    profiles: !override ["test-disabled"]
  omninode-runtime:
    environment:
      INFISICAL_ADDR: ${ONEX_RUNTIME_INFISICAL_ADDR:-}
      INFISICAL_CLIENT_ID: ${ONEX_RUNTIME_INFISICAL_CLIENT_ID:-}
"""

_CLEAN_OVERLAY_NO_DISABLED = """\
name: test-no-disabled-services
services:
  infisical:
    environment:
      SOME_VAR: value
  omninode-runtime:
    environment:
      INFISICAL_ADDR: "http://infisical:8080"
"""


# ---------------------------------------------------------------------------
# Tests: fail-closed proof (STEP D DoD item 4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bad_overlay_hardcoded_infisical_addr_exits_nonzero(tmp_path: Path) -> None:
    """Disabled infisical + hardcoded INFISICAL_ADDR → exit 1, names container+env.

    This is the OMN-12966 INFISICAL_ADDR pattern the linter must catch.
    probe_stdout captured verbatim (verifier != runner by test isolation).
    """
    overlay = tmp_path / "compose.yml"
    overlay.write_text(_BAD_OVERLAY_HARDCODED_VALUE, encoding="utf-8")

    result = _run(str(overlay))

    assert result.returncode == 1, (
        f"Expected exit 1 for bad overlay, got {result.returncode}\n"
        f"Output:\n{result.stdout}"
    )
    # Must name the offending container
    assert "omninode-runtime" in result.stdout, (
        f"Expected offending container 'omninode-runtime' in output:\n{result.stdout}"
    )
    # Must name the offending env var
    assert "INFISICAL_ADDR" in result.stdout, (
        f"Expected offending env var 'INFISICAL_ADDR' in output:\n{result.stdout}"
    )
    # Must name the disabled service
    assert "infisical" in result.stdout, (
        f"Expected disabled service 'infisical' in output:\n{result.stdout}"
    )


@pytest.mark.unit
def test_bad_overlay_vault_hostname_exits_nonzero(tmp_path: Path) -> None:
    """Disabled vault + env var value containing 'vault' hostname → exit 1."""
    overlay = tmp_path / "compose.yml"
    overlay.write_text(_BAD_OVERLAY_HOSTNAME_ONLY, encoding="utf-8")

    result = _run(str(overlay))

    assert result.returncode == 1, (
        f"Expected exit 1 for vault hostname overlay, got {result.returncode}\n"
        f"Output:\n{result.stdout}"
    )
    assert "vault" in result.stdout


# ---------------------------------------------------------------------------
# Tests: clean cases exit 0
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clean_overlay_disabled_infisical_no_ref_exits_zero(tmp_path: Path) -> None:
    """Disabled infisical, INFISICAL_ADDR='', no reference → exit 0.

    probe_stdout: exit 0 confirms clean.
    """
    overlay = tmp_path / "compose.yml"
    overlay.write_text(_CLEAN_OVERLAY, encoding="utf-8")

    result = _run(str(overlay))

    assert result.returncode == 0, (
        f"Expected exit 0 for clean overlay, got {result.returncode}\n"
        f"Output:\n{result.stdout}"
    )


@pytest.mark.unit
def test_clean_overlay_empty_default_interpolation_exits_zero(tmp_path: Path) -> None:
    """Disabled infisical, INFISICAL_ADDR=${VAR:-} empty default → exit 0.

    Shell-default-to-empty pattern is safe: no concrete value references
    the disabled service.
    """
    overlay = tmp_path / "compose.yml"
    overlay.write_text(_CLEAN_OVERLAY_EMPTY_DEFAULT, encoding="utf-8")

    result = _run(str(overlay))

    assert result.returncode == 0, (
        f"Expected exit 0 for empty-default overlay, got {result.returncode}\n"
        f"Output:\n{result.stdout}"
    )


@pytest.mark.unit
def test_clean_overlay_no_disabled_services_exits_zero(tmp_path: Path) -> None:
    """No disabled services → nothing to check → exit 0 even with INFISICAL_ADDR."""
    overlay = tmp_path / "compose.yml"
    overlay.write_text(_CLEAN_OVERLAY_NO_DISABLED, encoding="utf-8")

    result = _run(str(overlay))

    assert result.returncode == 0, (
        f"Expected exit 0 when no services disabled, got {result.returncode}\n"
        f"Output:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# Tests: real overlay files (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "overlay_name",
    [
        "docker-compose.stability-test.yml",
        "docker-compose.prod.yml",
        "docker-compose.judge.yml",
    ],
)
def test_real_overlay_passes_lint(overlay_name: str) -> None:
    """Real lane overlay files must pass the linter with exit 0.

    This is a regression guard — if a future overlay introduces a dangling
    credential/address env var for a disabled service, this test catches it.
    """
    overlay = _REPO_ROOT / "docker" / overlay_name

    if not overlay.exists():
        pytest.skip(f"Overlay file not found: {overlay}")

    result = _run(str(overlay))

    assert result.returncode == 0, (
        f"Real overlay {overlay_name} failed the dangling-deps lint.\n"
        f"Output:\n{result.stdout}"
    )


@pytest.mark.unit
def test_nonexistent_file_exits_nonzero() -> None:
    """A non-existent file path should fail with exit 1."""
    result = _run("/nonexistent-path/does-not-exist-omn13037.yml")

    assert result.returncode == 1
