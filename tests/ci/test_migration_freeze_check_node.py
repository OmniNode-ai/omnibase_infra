# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for node_migration_freeze_check_compute and its runtime (OMN-20568).

Ports every case of the former tests/ci/test_validate_migration_freeze.py, which
covered scripts/validation/validate_migration_freeze.py (OMN-3533): the
freeze_date= parse, the 30 day warning and 60 day expiry, the no-date case, and the
report wording.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationReport,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute import (
    NodeMigrationFreezeCheckCompute,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute import (
    runtime_migration_freeze_check as runtime,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute.handler import (
    parse_freeze_date,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute.models import (
    ModelMigrationFreezeCheckInput,
)

TODAY = date(2026, 10, 5)


def _freeze_ago(days: int) -> str:
    return f"freeze_date={(TODAY - timedelta(days=days)).isoformat()}\n"


def _check(
    freeze_text: str, added: tuple[str, ...] = (), active: bool = True
) -> ModelValidationReport:
    return NodeMigrationFreezeCheckCompute().handle(
        ModelMigrationFreezeCheckInput(
            freeze_active=active,
            freeze_text=freeze_text,
            today=TODAY,
            added_paths=added,
        )
    )


@pytest.mark.unit
class TestParseFreezeDate:
    def test_parses_valid_freeze_date(self) -> None:
        text = "# Comment\nfreeze_date=2026-02-10\nticket=OMN-2073\n"
        assert parse_freeze_date(text) == date(2026, 2, 10)

    def test_returns_none_when_field_absent(self) -> None:
        assert parse_freeze_date("# Migration Freeze\n# Created: 2026-02-10\n") is None

    def test_returns_none_when_field_commented_out(self) -> None:
        assert parse_freeze_date("# freeze_date=2026-02-10\n") is None

    def test_returns_none_on_invalid_date_format(self) -> None:
        assert parse_freeze_date("freeze_date=not-a-date\n") is None

    def test_handles_whitespace_around_value(self) -> None:
        assert parse_freeze_date("freeze_date=  2026-02-10  \n") == date(2026, 2, 10)


@pytest.mark.unit
class TestFreezeAge:
    def test_fresh_freeze_no_warning(self) -> None:
        report = _check(_freeze_ago(5))
        assert report.overall_status == "PASS"
        assert report.findings == ()

    def test_30_day_freeze_is_warning(self) -> None:
        report = _check(_freeze_ago(30))
        assert report.overall_status == "WARN"
        assert [f.rule_id for f in report.findings] == ["freeze-expiring"]

    def test_60_day_freeze_is_expired(self) -> None:
        report = _check(_freeze_ago(60))
        assert report.overall_status == "FAIL"
        assert [f.rule_id for f in report.findings] == ["freeze-expired"]

    def test_61_day_freeze_is_expired(self) -> None:
        assert _check(_freeze_ago(61)).overall_status == "FAIL"

    def test_no_freeze_date_has_no_age_finding(self) -> None:
        report = _check("# No freeze_date field\n")
        assert report.overall_status == "PASS"
        assert report.findings == ()


@pytest.mark.unit
class TestFreezeDecision:
    def test_inactive_freeze_never_fails_even_with_added_migrations(self) -> None:
        report = _check(
            "", added=("docker/migrations/forward/200_x.sql",), active=False
        )
        assert report.overall_status == "PASS"

    def test_fresh_freeze_without_new_files_is_valid(self) -> None:
        assert _check(_freeze_ago(5)).overall_status == "PASS"

    def test_expired_freeze_is_not_valid(self) -> None:
        assert _check(_freeze_ago(65)).overall_status == "FAIL"

    def test_warning_freeze_is_still_valid(self) -> None:
        # Only expiry makes the freeze invalid; a warning does not.
        assert _check(_freeze_ago(35)).overall_status == "WARN"

    def test_new_migration_file_is_a_violation(self) -> None:
        report = _check(
            _freeze_ago(5), added=("docker/migrations/forward/201_x.sql", "src/a.py")
        )
        assert report.overall_status == "FAIL"
        assert [f.location for f in report.findings] == [
            "docker/migrations/forward/201_x.sql"
        ]

    def test_new_file_outside_migration_dirs_is_allowed(self) -> None:
        assert _check(_freeze_ago(5), added=("src/a.py",)).overall_status == "PASS"


@pytest.mark.unit
class TestRuntimeWording:
    """The report wording the former generate_report tests pinned."""

    def test_inactive_freeze_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert runtime.main([]) == 0
        assert "inactive" in capsys.readouterr().out

    def test_pass_report(self) -> None:
        assert _check(_freeze_ago(5)).overall_status == "PASS"

    def test_warning_report_contains_warning(self) -> None:
        text = runtime._render(_check(_freeze_ago(35)))
        assert "WARNING" in text
        assert "approaching expiry" in text

    def test_expired_report_contains_error(self) -> None:
        text = runtime._render(_check(_freeze_ago(65)))
        assert "EXPIRED" in text
