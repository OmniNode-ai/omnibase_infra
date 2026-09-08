# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for boot schema-fingerprint stamp tolerance (OMN-13666).

Policy under test (unchanged by OMN-17372 — only its host moved):
    * The runtime's OWN database (omnibase_infra) stamp is REQUIRED -- a failure
      after all retries aborts boot (exit 1) so the crash cause is loud and the
      kernel never starts with a NULL/stale fingerprint.
    * A SECONDARY / non-owned database (omniintelligence) stamp is BEST-EFFORT --
      a failure (e.g. "permission denied for table db_metadata") warns and boot
      proceeds (exit 0).

Before OMN-17372 this policy lived in ``docker/entrypoint-runtime.sh`` and was
exercised by running that script with a stubbed ``python`` on PATH. The stamp,
the two renders and the kernel start are now ONE warm interpreter
(``omnibase_infra.runtime.entrypoint_preflight``), so the tests drive that
module directly with a stubbed ``stamp_manifest_fingerprint`` — same policy,
same operator-visible text, one process instead of four.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.runtime import entrypoint_preflight, util_schema_fingerprint
from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]

ENTRYPOINT = DOCKER_DIR / "entrypoint-runtime.sh"

INFRA_DSN = "postgresql://u:pw@db:5432/omnibase_infra"
INTEL_DSN = "postgresql://u:pw@db:5432/omniintelligence"


def _stub_stamp(
    monkeypatch: pytest.MonkeyPatch, *, rc_by_manifest: dict[str, int]
) -> list[str]:
    """Replace the real stamp with a per-manifest exit code. Returns the call log."""
    calls: list[str] = []

    def _fake(*, manifest_name: str, db_url: str) -> int:
        calls.append(manifest_name)
        return rc_by_manifest.get(manifest_name, 0)

    monkeypatch.setattr(
        util_schema_fingerprint, "stamp_manifest_fingerprint", _fake, raising=True
    )
    # The retry loop must not sleep in a unit test.
    monkeypatch.setattr(entrypoint_preflight, "STAMP_RETRY_SLEEP_SECONDS", 0.0)
    monkeypatch.setattr(entrypoint_preflight.time, "sleep", lambda _s: None)
    return calls


def _env(*, with_intel_db: bool = True) -> dict[str, str]:
    env = {"OMNIBASE_INFRA_DB_URL": INFRA_DSN}
    if with_intel_db:
        env["OMNIINTELLIGENCE_DB_URL"] = INTEL_DSN
    return env


def test_clean_stamp_both_dbs_boots(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both stamps succeed -> preflight returns 0 and boot may proceed."""
    _stub_stamp(monkeypatch, rc_by_manifest={})

    rc = entrypoint_preflight.stamp_all_fingerprints(_env())

    out = capsys.readouterr().out
    assert rc == 0
    assert "Schema fingerprint stamped for omnibase_infra." in out
    assert "Schema fingerprint stamped for omniintelligence." in out


def test_secondary_db_stamp_failure_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """omniintelligence stamp fails (perm denied) -> WARNING, boot continues."""
    _stub_stamp(monkeypatch, rc_by_manifest={"omniintelligence": 1})

    rc = entrypoint_preflight.stamp_all_fingerprints(_env())

    out = capsys.readouterr().out
    assert rc == 0
    assert "Schema fingerprint stamped for omnibase_infra." in out
    assert (
        "WARNING: omniintelligence (secondary/non-owned DB) fingerprint stamp "
        "did not succeed -- continuing best-effort" in out
    )


def test_primary_db_stamp_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """omnibase_infra stamp fails -> boot aborts (rc 1), kernel never starts."""
    calls = _stub_stamp(monkeypatch, rc_by_manifest={"omnibase_infra": 1})

    rc = entrypoint_preflight.run_preflight(_env())

    captured = capsys.readouterr()
    assert rc == 1
    assert (
        "ERROR: omnibase_infra (PRIMARY/owned DB) fingerprint stamp failed "
        "-- aborting boot" in captured.err
    )
    # Retried the documented number of times, then gave up.
    assert calls.count("omnibase_infra") == entrypoint_preflight.MAX_STAMP_ATTEMPTS
    # And never reached the secondary DB.
    assert "omniintelligence" not in calls


def test_mismatch_exit_code_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 2 is a schema mismatch: retrying cannot help, so it must not retry."""
    calls = _stub_stamp(monkeypatch, rc_by_manifest={"omnibase_infra": 2})

    rc = entrypoint_preflight.stamp_all_fingerprints(_env(with_intel_db=False))

    out = capsys.readouterr().out
    # Exit 2 breaks the retry loop but leaves stamp_ok False, so the REQUIRED
    # database still aborts boot -- the kernel must never start on a mismatch.
    assert rc == 1
    assert calls.count("omnibase_infra") == 1
    assert (
        "WARNING: omnibase_infra fingerprint mismatch (exit 2) -- not retrying" in out
    )


def test_secondary_db_optional_when_db_url_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No OMNIINTELLIGENCE_DB_URL -> stamp skipped, boot proceeds."""
    calls = _stub_stamp(monkeypatch, rc_by_manifest={})

    rc = entrypoint_preflight.stamp_all_fingerprints(_env(with_intel_db=False))

    out = capsys.readouterr().out
    assert rc == 0
    assert calls == ["omnibase_infra"]
    assert (
        "OMNIINTELLIGENCE_DB_URL not set -- skipping omniintelligence "
        "fingerprint stamp" in out
    )


def test_primary_db_optional_when_db_url_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No OMNIBASE_INFRA_DB_URL -> stamp skipped entirely, as the shell did."""
    calls = _stub_stamp(monkeypatch, rc_by_manifest={})

    rc = entrypoint_preflight.stamp_all_fingerprints({})

    out = capsys.readouterr().out
    assert rc == 0
    assert calls == []
    assert "OMNIBASE_INFRA_DB_URL not set -- skipping fingerprint stamp" in out


def test_dsn_credentials_are_never_logged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The log line shows host:port/db only -- same as the shell's sed did."""
    _stub_stamp(monkeypatch, rc_by_manifest={})

    entrypoint_preflight.stamp_all_fingerprints(
        {"OMNIBASE_INFRA_DB_URL": "postgresql://someuser:s3cret@db:5432/omnibase_infra"}
    )

    out = capsys.readouterr().out
    assert "s3cret" not in out
    assert "someuser" not in out
    assert "(db: db:5432/omnibase_infra, required)" in out


def test_dsn_without_userinfo_is_echoed_unchanged() -> None:
    """sed's substitution needs an `@`; without one the DSN is printed as-is."""
    assert (
        entrypoint_preflight._safe_dsn("postgresql://db:5432/omnibase_infra")
        == "postgresql://db:5432/omnibase_infra"
    )


# ---------------------------------------------------------------------------
# Static guards -- the required/optional contract, now in Python
# ---------------------------------------------------------------------------


def test_preflight_marks_primary_required_and_secondary_optional() -> None:
    assert entrypoint_preflight.__file__ is not None
    source = Path(entrypoint_preflight.__file__).read_text()
    assert (
        'manifest_name="omnibase_infra", db_url=infra_db_url, required=True' in source
    )
    assert (
        'manifest_name="omniintelligence", db_url=intel_db_url, required=False'
        in source
    )


def test_shellcheck_clean_if_available() -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck not installed")
    result = subprocess.run(
        [shellcheck, str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


__all__: list[str] = []
