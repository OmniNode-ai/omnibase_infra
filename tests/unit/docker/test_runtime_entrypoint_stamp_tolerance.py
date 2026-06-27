# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for entrypoint schema-fingerprint stamp tolerance (OMN-13666).

Policy under test:
    * The runtime's OWN database (omnibase_infra) stamp is REQUIRED -- a failure
      after all retries aborts boot (exit 1) so the crash cause is loud and the
      kernel never starts with a NULL/stale fingerprint.
    * A SECONDARY / non-owned database (omniintelligence) stamp is BEST-EFFORT --
      a failure (e.g. "permission denied for table db_metadata") warns and boot
      proceeds (exit 0).

The behavioral tests execute the real ``docker/entrypoint-runtime.sh`` with a
stubbed ``python`` on PATH whose exit code is driven per-manifest by env vars,
so no Docker, Postgres, or privilege drop is involved. The CMD that the
entrypoint exec's at the end is ``true`` (exit 0), so a clean run exits 0.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]

ENTRYPOINT = DOCKER_DIR / "entrypoint-runtime.sh"

# Stub "python" that emulates util_schema_fingerprint stamp exit codes.
# It parses the --manifest value out of its args and looks up an exit code from
# an env var (STUB_RC_<MANIFEST>), defaulting to 0 (success). Any non-stamp
# invocation (e.g. render modules, which the entrypoint guards behind unset env
# vars and therefore never calls here) also exits 0.
_PYTHON_STUB = """#!/bin/sh
manifest=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--manifest" ]; then
    manifest="$arg"
  fi
  prev="$arg"
done
case "$manifest" in
  omnibase_infra) exit "${STUB_RC_OMNIBASE_INFRA:-0}" ;;
  omniintelligence) exit "${STUB_RC_OMNIINTELLIGENCE:-0}" ;;
  *) exit 0 ;;
esac
"""


def _run_entrypoint(
    tmp_path: Path,
    *,
    infra_rc: int = 0,
    intel_rc: int = 0,
    with_intel_db: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run the real entrypoint with a stubbed python and controlled stamp RCs."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "python"
    stub.write_text(_PYTHON_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    env = {
        # Minimal PATH: stub python first, then real sh/sed/echo/sleep tools.
        "PATH": f"{bindir}:/usr/bin:/bin",
        "OMNIBASE_INFRA_DB_URL": "postgresql://u:p@db:5432/omnibase_infra",
        "STUB_RC_OMNIBASE_INFRA": str(infra_rc),
        "STUB_RC_OMNIINTELLIGENCE": str(intel_rc),
    }
    if with_intel_db:
        env["OMNIINTELLIGENCE_DB_URL"] = "postgresql://u:p@db:5432/omniintelligence"

    # CMD the entrypoint exec's at the end; "true" exits 0 on a clean boot path.
    return subprocess.run(
        ["sh", str(ENTRYPOINT), "true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_clean_stamp_both_dbs_boots() -> None:
    """Both stamps succeed -> entrypoint reaches exec and exits 0."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        result = _run_entrypoint(Path(td), infra_rc=0, intel_rc=0)
    assert result.returncode == 0, result.stderr
    assert "Schema fingerprint stamped for omnibase_infra." in result.stdout
    assert "Schema fingerprint stamped for omniintelligence." in result.stdout
    assert "Starting runtime kernel..." in result.stdout


def test_secondary_db_stamp_failure_is_nonfatal() -> None:
    """omniintelligence stamp fails (perm denied) -> WARNING, boot continues (exit 0)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # rc=1 emulates "permission denied for table db_metadata" general error.
        result = _run_entrypoint(Path(td), infra_rc=0, intel_rc=1)
    assert result.returncode == 0, result.stderr
    assert "Schema fingerprint stamped for omnibase_infra." in result.stdout
    assert (
        "WARNING: omniintelligence (secondary/non-owned DB) fingerprint stamp "
        "did not succeed -- continuing best-effort" in result.stdout
    )
    # Boot proceeds past the stamp section.
    assert "Starting runtime kernel..." in result.stdout


def test_primary_db_stamp_failure_is_fatal() -> None:
    """omnibase_infra stamp fails -> entrypoint aborts boot (exit 1), no kernel start."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        result = _run_entrypoint(Path(td), infra_rc=1, intel_rc=0)
    assert result.returncode == 1
    assert (
        "ERROR: omnibase_infra (PRIMARY/owned DB) fingerprint stamp failed "
        "-- aborting boot" in result.stderr
    )
    # Must NOT reach kernel start when the primary DB stamp fails.
    assert "Starting runtime kernel..." not in result.stdout


def test_secondary_db_optional_when_db_url_unset() -> None:
    """No OMNIINTELLIGENCE_DB_URL -> stamp skipped, boot proceeds (exit 0)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        result = _run_entrypoint(Path(td), infra_rc=0, with_intel_db=False)
    assert result.returncode == 0, result.stderr
    assert (
        "OMNIINTELLIGENCE_DB_URL not set -- skipping omniintelligence "
        "fingerprint stamp" in result.stdout
    )
    assert "Starting runtime kernel..." in result.stdout


# ---------------------------------------------------------------------------
# Static guards on the script source -- keep the required/optional contract.
# ---------------------------------------------------------------------------


def test_entrypoint_marks_primary_required_and_secondary_optional() -> None:
    source = ENTRYPOINT.read_text()
    assert (
        'stamp_fingerprint "omnibase_infra" "${OMNIBASE_INFRA_DB_URL}" "required"'
        in source
    )
    assert (
        'stamp_fingerprint "omniintelligence" "${OMNIINTELLIGENCE_DB_URL}" "optional"'
        in source
    )


def test_required_stamp_failure_exits_nonzero_in_source() -> None:
    source = ENTRYPOINT.read_text()
    # The required branch must exit non-zero; the optional branch must only warn.
    assert 'if [ "${REQUIRED}" = "required" ]; then' in source
    assert "exit 1" in source.split('if [ "${REQUIRED}" = "required" ]; then', 1)[1]


def test_shellcheck_clean_if_available() -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck not installed")
    result = subprocess.run(
        [shellcheck, str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ},
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
