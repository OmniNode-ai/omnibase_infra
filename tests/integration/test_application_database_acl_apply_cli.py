# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end refusals of the shipped ACL apply entry point (OMN-15355).

The unit tests drive the gate functions directly. This module drives the
command a change window would actually type -- a real subprocess, the real
repository-committed matrix, real files on disk, and the real process exit
code -- because a gate that is correct as a function and broken as a CLI still
lets a live privilege change through.

Nothing here connects to a database, and nothing here can: every assertion is
about a refusal that lands strictly before the first connection is opened. The
admin DSN is a deliberately unroutable placeholder, and a run that reached it
would fail differently, which is itself part of the proof.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "apply_application_database_acl.py"
_MATRIX = (
    _REPO_ROOT
    / "docker"
    / "application-acl-proof"
    / "generated"
    / "candidate-matrix.yaml"
)

_CONSENT_ROW = (
    "2026-09-14T12:11:31Z | OPERATOR-CONSENT | lane=omn15355-change-window | "
    'approved_by=operator | "authorized" | APPROVED SCOPE: the OMN-15355 '
    "change window, live GRANT/REVOKE per the generated ACL matrix | "
    "OUT OF SCOPE: production | This row is the durable authorization evidence"
)


def _run(
    *arguments: str,
    ledger: Path,
    snapshot: Path,
    ticket: str = "OMN-15355",
    citation_line: int = 1,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    # Unroutable by construction: a run that reaches a connection fails with a
    # different message than any refusal this module asserts.
    environment["ADMIN_DSN"] = "postgresql://acl-apply-must-not-connect/unused"
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--matrix",
            str(_MATRIX),
            "--ticket",
            ticket,
            "--consent-citation",
            f"{ledger}:{citation_line}",
            "--ledger-root",
            str(ledger.parent),
            "--snapshot-out",
            str(snapshot),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
        env=environment,
        timeout=300,
    )


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    path = tmp_path / "LEDGER.md"
    path.write_text(_CONSENT_ROW + "\n", encoding="utf-8")
    return path


def test_the_repository_still_ships_the_matrix_this_module_drives() -> None:
    """Positive control on the inputs, so an absent file cannot read as a pass."""
    assert _SCRIPT.is_file()
    assert _MATRIX.is_file()
    assert _MATRIX.stat().st_size > 0


def test_the_shipped_cli_refuses_the_committed_matrix_and_names_its_blockers(
    ledger: Path, tmp_path: Path
) -> None:
    """The end-to-end refusal, exit code included."""
    snapshot = tmp_path / "prechange.json"

    result = _run("--execute", ledger=ledger, snapshot=snapshot)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "status=REFUSED" in result.stderr
    assert "BLOCKED" in result.stderr
    assert "cross-domain privileges" in result.stderr
    # The refusal lands before any mutation, so no snapshot was written and no
    # connection to the unroutable admin DSN was attempted.
    assert not snapshot.exists()
    assert "could not translate host name" not in result.stderr


def test_the_shipped_cli_refuses_a_consent_row_that_does_not_name_the_ticket(
    ledger: Path, tmp_path: Path
) -> None:
    """A grant for other work is not a grant for this one."""
    result = _run(
        "--execute",
        ledger=ledger,
        snapshot=tmp_path / "prechange.json",
        ticket="OMN-99999",
    )

    assert result.returncode == 1
    assert "APPROVED SCOPE does not name OMN-99999" in result.stderr


def test_the_shipped_cli_refuses_a_citation_past_the_end_of_the_ledger(
    ledger: Path, tmp_path: Path
) -> None:
    result = _run(
        "--execute",
        ledger=ledger,
        snapshot=tmp_path / "prechange.json",
        citation_line=99,
    )

    assert result.returncode == 1
    assert "past the end of" in result.stderr


def test_the_shipped_cli_refuses_a_row_that_is_not_a_consent_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "LEDGER.md"
    path.write_text(
        "2026-09-14T00:00:00Z | CLAIM | lane=x | ticket=OMN-15355\n", encoding="utf-8"
    )

    result = _run("--execute", ledger=path, snapshot=tmp_path / "prechange.json")

    assert result.returncode == 1
    assert "is not an OPERATOR-CONSENT row" in result.stderr


def test_the_shipped_cli_refuses_when_no_admin_dsn_is_supplied(
    ledger: Path, tmp_path: Path
) -> None:
    """A missing admin DSN is a refusal, never a silently skipped apply."""
    environment = dict(os.environ)
    environment.pop("ADMIN_DSN", None)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--matrix",
            str(_MATRIX),
            "--ticket",
            "OMN-15355",
            "--consent-citation",
            f"{ledger}:1",
            "--ledger-root",
            str(ledger.parent),
            "--snapshot-out",
            str(tmp_path / "prechange.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
        env=environment,
        timeout=300,
    )

    assert result.returncode == 2
    assert "ADMIN_DSN_absent" in result.stderr


def test_a_dry_run_is_still_refused_when_the_matrix_is_ineligible(
    ledger: Path, tmp_path: Path
) -> None:
    """Omitting --execute does not downgrade a refusal into a warning."""
    result = _run(ledger=ledger, snapshot=tmp_path / "prechange.json")

    assert result.returncode == 1
    assert "status=REFUSED" in result.stderr
