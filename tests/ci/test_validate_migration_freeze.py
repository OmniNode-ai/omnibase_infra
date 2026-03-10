# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for migration freeze validation with freeze age gate.

Covers the freeze_date= expiry logic added in OMN-3533:
  - WARNING at 30+ days
  - ERROR (expired) at 60+ days
  - No age check when freeze_date= is absent
  - Normal violation detection still works alongside age gate

Ticket: OMN-3533
"""

from __future__ import annotations

import textwrap
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.validation.validate_migration_freeze import (
    FreezeAgeStatus,
    FreezeValidationResult,
    _compute_freeze_age,
    _parse_freeze_date,
    generate_report,
    validate_migration_freeze,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_freeze_file(tmp_path: Path, content: str) -> Path:
    """Write a .migration_freeze file to tmp_path and return its path."""
    freeze = tmp_path / ".migration_freeze"
    freeze.write_text(textwrap.dedent(content), encoding="utf-8")
    return freeze


def _today() -> date:
    """Return today's date in UTC."""
    return datetime.now(tz=UTC).date()


def _date_ago(days: int) -> str:
    """Return an ISO date string for `days` ago from today (UTC)."""
    return (_today() - timedelta(days=days)).isoformat()


# ── _parse_freeze_date ────────────────────────────────────────────────────────


class TestParseFreezeDate:
    def test_parses_valid_freeze_date(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(
            tmp_path,
            """\
            # Comment
            freeze_date=2026-02-10
            ticket=OMN-2073
            """,
        )
        result = _parse_freeze_date(freeze)
        assert result == date(2026, 2, 10)

    def test_returns_none_when_field_absent(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(
            tmp_path,
            """\
            # Migration Freeze
            # Created: 2026-02-10
            """,
        )
        assert _parse_freeze_date(freeze) is None

    def test_returns_none_when_field_commented_out(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(
            tmp_path,
            """\
            # freeze_date=2026-02-10
            """,
        )
        assert _parse_freeze_date(freeze) is None

    def test_returns_none_on_invalid_date_format(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(
            tmp_path,
            """\
            freeze_date=not-a-date
            """,
        )
        assert _parse_freeze_date(freeze) is None

    def test_handles_whitespace_around_value(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(
            tmp_path,
            "freeze_date=  2026-02-10  \n",
        )
        result = _parse_freeze_date(freeze)
        assert result == date(2026, 2, 10)


# ── _compute_freeze_age ───────────────────────────────────────────────────────


class TestComputeFreezeAge:
    def test_fresh_freeze_no_warning(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(tmp_path, f"freeze_date={_date_ago(5)}\n")
        status = _compute_freeze_age(freeze)
        assert status.has_date
        assert status.age_days == 5
        assert not status.is_warning
        assert not status.is_expired

    def test_30_day_freeze_is_warning(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(tmp_path, f"freeze_date={_date_ago(30)}\n")
        status = _compute_freeze_age(freeze)
        assert status.is_warning
        assert not status.is_expired

    def test_60_day_freeze_is_expired(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(tmp_path, f"freeze_date={_date_ago(60)}\n")
        status = _compute_freeze_age(freeze)
        assert status.is_warning
        assert status.is_expired

    def test_61_day_freeze_is_expired(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(tmp_path, f"freeze_date={_date_ago(61)}\n")
        status = _compute_freeze_age(freeze)
        assert status.is_expired

    def test_no_freeze_date_returns_empty_status(self, tmp_path: Path) -> None:
        freeze = _write_freeze_file(tmp_path, "# No freeze_date field\n")
        status = _compute_freeze_age(freeze)
        assert not status.has_date
        assert status.age_days is None
        assert not status.is_warning
        assert not status.is_expired


# ── validate_migration_freeze (full integration with age gate) ────────────────


class TestValidateMigrationFreezeAgeGate:
    def test_no_freeze_file_is_valid(self, tmp_path: Path) -> None:
        result = validate_migration_freeze(tmp_path)
        assert result.is_valid
        assert not result.freeze_active

    def test_fresh_freeze_no_staged_files_is_valid(self, tmp_path: Path) -> None:
        # Freeze file present, but freeze is young (no age violation)
        _write_freeze_file(tmp_path, f"freeze_date={_date_ago(5)}\n")
        result = validate_migration_freeze(tmp_path, check_staged=True)
        # No staged files to check, but freeze is active
        assert result.freeze_active
        assert result.is_valid
        assert not result.age_status.is_expired

    def test_expired_freeze_is_not_valid(self, tmp_path: Path) -> None:
        _write_freeze_file(tmp_path, f"freeze_date={_date_ago(65)}\n")
        result = validate_migration_freeze(tmp_path, check_staged=True)
        assert result.freeze_active
        assert result.age_status.is_expired
        assert not result.is_valid

    def test_warning_freeze_is_still_valid(self, tmp_path: Path) -> None:
        _write_freeze_file(tmp_path, f"freeze_date={_date_ago(35)}\n")
        result = validate_migration_freeze(tmp_path, check_staged=True)
        assert result.freeze_active
        assert result.age_status.is_warning
        assert not result.age_status.is_expired
        # Warning does NOT make it invalid — only expired does
        assert result.is_valid

    def test_absent_freeze_date_still_valid(self, tmp_path: Path) -> None:
        _write_freeze_file(tmp_path, "# Migration Freeze\n# Created: 2026-02-10\n")
        result = validate_migration_freeze(tmp_path, check_staged=True)
        assert result.freeze_active
        assert not result.age_status.has_date
        assert result.is_valid


# ── generate_report ───────────────────────────────────────────────────────────


class TestGenerateReport:
    def test_inactive_freeze_report(self, tmp_path: Path) -> None:
        result = FreezeValidationResult(freeze_active=False)
        report = generate_report(result, tmp_path)
        assert "inactive" in report

    def test_fresh_freeze_pass_report(self, tmp_path: Path) -> None:
        result = FreezeValidationResult(
            freeze_active=True,
            age_status=FreezeAgeStatus(
                freeze_date=_today() - timedelta(days=5),
                age_days=5,
                is_warning=False,
                is_expired=False,
            ),
        )
        report = generate_report(result, tmp_path)
        assert "PASS" in report

    def test_warning_freeze_report_contains_warning(self, tmp_path: Path) -> None:
        result = FreezeValidationResult(
            freeze_active=True,
            age_status=FreezeAgeStatus(
                freeze_date=_today() - timedelta(days=35),
                age_days=35,
                is_warning=True,
                is_expired=False,
            ),
        )
        report = generate_report(result, tmp_path)
        assert "WARNING" in report
        assert "approaching expiry" in report

    def test_expired_freeze_report_contains_error(self, tmp_path: Path) -> None:
        result = FreezeValidationResult(
            freeze_active=True,
            age_status=FreezeAgeStatus(
                freeze_date=_today() - timedelta(days=65),
                age_days=65,
                is_warning=True,
                is_expired=True,
            ),
        )
        report = generate_report(result, tmp_path)
        assert "EXPIRED" in report
        assert not result.is_valid
