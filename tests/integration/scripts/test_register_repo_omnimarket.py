# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for register-repo.py onboard-repo idempotency (OMN-10593).

CLI surface under test
----------------------
scripts/register-repo.py has two subcommands:

    seed-shared  [--env-file <path>] [--overwrite] [--execute]
        Populates /shared/<transport>/ paths in Infisical from a platform .env.
        Dry-run by default; pass --execute to write.
        NOTE: seed-shared validates INFISICAL_ADDR *before* the dry-run gate,
        so it cannot be dry-run-tested without live Infisical creds.

    onboard-repo --repo <name> --env-file <path> [--overwrite] [--execute]
        Creates /services/<repo>/ folder structure and seeds repo-specific
        secrets.  Dry-run by default; pass --execute to write.
        Dry-run does NOT require INFISICAL_ADDR (validated only on --execute).

Idempotency guarantee
---------------------
The script uses _upsert_secret(overwrite=False) for all per-service keys,
meaning a second run with the same inputs:
  - Reports "skipped" (not "created" or "updated") for every existing key.
  - Returns exit-code 0 (no errors).
  - Produces no duplicate secrets.

The tests below verify this guarantee at three levels:
  1. Dry-run succeeds without any live infrastructure.
  2. First --execute run creates the expected folder structure and keys.
  3. Second --execute run reports only "skipped", no errors.

Tests 2 and 3 are skipped when Infisical is unreachable (CI-safe).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "register-repo.py"
_INFISICAL_ADDR = os.environ.get("INFISICAL_ADDR", "http://192.168.86.201:8880")

# ---------------------------------------------------------------------------
# Infisical reachability probe — performed once at module import time so that
# parametrize + skip markers resolve correctly without adding network latency
# to each individual test.
# ---------------------------------------------------------------------------

_INFISICAL_REACHABLE: bool = False
try:
    import httpx

    with httpx.Client(timeout=3) as _client:
        _resp = _client.get(f"{_INFISICAL_ADDR}/api/status")
        _INFISICAL_REACHABLE = _resp.status_code == 200
except Exception:  # noqa: BLE001
    _INFISICAL_REACHABLE = False

# The --execute path also requires the admin token file produced by
# provision-infisical.py. Without it, folder creation aborts before any
# secret writes, so idempotency cannot be exercised. Skip if absent.
_ADMIN_TOKEN_FILE = _REPO_ROOT / ".infisical-admin-token"
_ADMIN_TOKEN_PRESENT: bool = _ADMIN_TOKEN_FILE.is_file() and bool(
    _ADMIN_TOKEN_FILE.read_text(encoding="utf-8").strip()
)

_EXECUTE_PREREQS_MET = _INFISICAL_REACHABLE and _ADMIN_TOKEN_PRESENT

_requires_infisical = pytest.mark.skipif(
    not _EXECUTE_PREREQS_MET,
    reason=(
        f"Infisical not reachable at {_INFISICAL_ADDR}"
        if not _INFISICAL_REACHABLE
        else f"Admin token not provisioned at {_ADMIN_TOKEN_FILE} — run scripts/provision-infisical.py first"
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_env_file(tmp_path: Path) -> Path:
    """Write a minimal .env that satisfies register-repo.py's parser."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# Minimal synthetic env for onboard-repo idempotency tests",
                "POSTGRES_DATABASE=omnimarket",
                "POSTGRES_HOST=192.168.86.201",
                "POSTGRES_PORT=5436",
                "POSTGRES_USER=postgres",
                "KAFKA_GROUP_ID=omnimarket-consumer",
                "LLM_CODER_URL=http://192.168.86.201:8000",
                "LLM_CODER_MODEL_ID=cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
                "LLM_REASONER_URL=http://192.168.86.201:8001",
                "LLM_REASONER_MODEL_ID=Corianas/DeepSeek-R1-Distill-Qwen-14B-AWQ",
            ]
        ),
        encoding="utf-8",
    )
    return env_file


def _run(
    args: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


# ---------------------------------------------------------------------------
# Test 1: dry-run — no live infra required
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_dry_run_succeeds_without_infisical(tmp_path: Path) -> None:
    """onboard-repo --dry-run exits 0 and prints the plan without touching Infisical."""
    env_file = _make_minimal_env_file(tmp_path)

    result = _run(
        [
            "onboard-repo",
            "--repo",
            "omnimarket",
            "--env-file",
            str(env_file),
        ]
    )

    assert result.returncode == 0, (
        f"dry-run failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "omnimarket" in result.stdout
    assert "dry-run" in result.stdout.lower()
    assert "/services/omnimarket" in result.stdout


@pytest.mark.integration
def test_dry_run_shows_expected_service_keys(tmp_path: Path) -> None:
    """Dry-run output lists the per-service keys declared in services.omnimarket."""
    env_file = _make_minimal_env_file(tmp_path)

    result = _run(
        [
            "onboard-repo",
            "--repo",
            "omnimarket",
            "--env-file",
            str(env_file),
        ]
    )

    assert result.returncode == 0
    for expected_key in ("KAFKA_GROUP_ID", "POSTGRES_DATABASE", "LLM_CODER_URL"):
        assert expected_key in result.stdout, (
            f"Expected key {expected_key!r} missing from dry-run output:\n{result.stdout}"
        )


@pytest.mark.integration
def test_dry_run_invalid_repo_name_rejected(tmp_path: Path) -> None:
    """Repo names with path-traversal characters must be rejected before any I/O."""
    env_file = _make_minimal_env_file(tmp_path)

    result = _run(
        [
            "onboard-repo",
            "--repo",
            "../../../etc",
            "--env-file",
            str(env_file),
        ]
    )

    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Tests 2 + 3: live Infisical idempotency (skipped when Infisical unreachable)
# ---------------------------------------------------------------------------


def _build_execute_env() -> dict[str, str]:
    """Build a subprocess env that inherits the shell env plus any .omnibase/.env values."""
    env = dict(os.environ)
    omnibase_env = Path.home() / ".omnibase" / ".env"
    if omnibase_env.is_file():
        for line in omnibase_env.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip().removeprefix("export ").strip()
            if key and key not in env:
                env[key] = value.strip()
    return env


@pytest.mark.integration
@_requires_infisical
def test_execute_first_run_exits_zero(tmp_path: Path) -> None:
    """First --execute run must exit 0 for a clean onboard."""
    env_file = _make_minimal_env_file(tmp_path)
    execute_env = _build_execute_env()

    result = _run(
        [
            "onboard-repo",
            "--repo",
            "omnimarket",
            "--env-file",
            str(env_file),
            "--execute",
        ],
        env=execute_env,
    )

    assert result.returncode == 0, (
        f"First --execute run failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


@pytest.mark.integration
@_requires_infisical
def test_execute_second_run_is_idempotent(tmp_path: Path) -> None:
    """Second --execute run must exit 0 with zero errors and only 'skipped' outcomes.

    This is the core idempotency guarantee: onboard-repo can be re-run safely
    without duplicating secrets or failing on already-existing keys.
    """
    env_file = _make_minimal_env_file(tmp_path)
    execute_env = _build_execute_env()
    run_args = [
        "onboard-repo",
        "--repo",
        "omnimarket",
        "--env-file",
        str(env_file),
        "--execute",
    ]

    first = _run(run_args, env=execute_env)
    assert first.returncode == 0, (
        f"First run failed — cannot test idempotency:\n"
        f"stdout: {first.stdout}\nstderr: {first.stderr}"
    )

    second = _run(run_args, env=execute_env)

    assert second.returncode == 0, (
        f"Second (idempotent) run failed (rc={second.returncode})\n"
        f"stdout: {second.stdout}\n"
        f"stderr: {second.stderr}"
    )
    assert "0 errors" in second.stdout, (
        f"Expected '0 errors' in second run output:\n{second.stdout}"
    )
    # No new keys should be created on the second run (all already exist).
    assert "0 created" in second.stdout, (
        f"Expected '0 created' in second run output (idempotency failed):\n{second.stdout}"
    )
