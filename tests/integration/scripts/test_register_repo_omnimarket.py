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

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "register-repo.py"


def _load_omnibase_env_into_process() -> None:
    """Load local OmniBase env defaults before module-level infra probes."""
    omnibase_env = Path.home() / ".omnibase" / ".env"
    if not omnibase_env.is_file():
        return
    for line in omnibase_env.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


_load_omnibase_env_into_process()
_INFISICAL_ADDR = os.environ.get("INFISICAL_ADDR", "http://192.168.86.201:8880")

# ---------------------------------------------------------------------------
# Infisical reachability probe — performed once at module import time so that
# parametrize + skip markers resolve correctly without adding network latency
# to each individual test.
# ---------------------------------------------------------------------------

# OMN-18795: the two `--execute` tests that lived here were DELETED, and with
# them the live-Infisical reachability probe and the `_requires_infisical`
# skip marker that gated them.
#
# They asserted the idempotency guarantee of `onboard-repo --execute`: a second
# run reports only `skipped`, `0 created`, `0 errors`. They could only ever run
# against a reachable Infisical PLUS an admin token file produced by
# scripts/provision-infisical.py, and CI has neither. Providing them would mean
# standing up a secret store in CI and minting an admin credential for it --
# a credential-posture decision of its own, not a test-wiring one -- so the
# suite was registered in config/env_gated_test_skips.yaml as having no
# execution path, which is a description of the gap rather than a fix.
#
# WHAT IS GIVEN UP: `_upsert_secret(overwrite=False)` idempotency proven against
# a REAL Infisical. No CI gate covers that today; the gap is recorded on
# OMN-18795 rather than left implied by a permanently-skipped test. What
# remains below needs no infrastructure at all and runs on every pull request:
# the dry-run path, the service-key set it reports, and the repo-name refusal.

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
