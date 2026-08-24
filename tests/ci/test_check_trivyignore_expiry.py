# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/ci/check_trivyignore_expiry.py (OMN-16229)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

from scripts.ci.check_trivyignore_expiry import evaluate_trivyignore, main

_TODAY = date(2026, 8, 18)


def _valid_block(
    cve: str = "CVE-2024-12345", ticket: str = "OMN-12345", expires: str = "2026-12-31"
) -> str:
    return (
        f"# CVE: {cve}\n"
        "# reason: no-upstream-fix\n"
        f"# ticket: {ticket}\n"
        f"# expires: {expires}\n"
        f"{cve}\n"
    )


class TestEvaluateTrivyignore:
    def test_empty_file_passes(self) -> None:
        verdict = evaluate_trivyignore("", today=_TODAY)
        assert verdict.passed
        assert verdict.entries == ()

    def test_header_only_file_passes(self) -> None:
        text = (
            "# This file lists CVE ignores.\n# See docs/patterns/security_patterns.md\n"
        )
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert verdict.passed
        assert verdict.entries == ()

    def test_valid_entry_passes(self) -> None:
        verdict = evaluate_trivyignore(_valid_block(), today=_TODAY)
        assert verdict.passed
        assert len(verdict.entries) == 1

    def test_valid_entry_expiring_today_fails(self) -> None:
        # <= today is expired, not just < today -- the day it expires, it is
        # no longer valid to rely on without re-triage.
        text = _valid_block(expires="2026-08-18")
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed

    def test_expired_entry_fails(self) -> None:
        text = _valid_block(expires="2026-01-01")
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed
        assert "expired" in verdict.violations[0].reason

    def test_future_entry_passes(self) -> None:
        text = _valid_block(expires="2027-01-01")
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert verdict.passed

    def test_bare_id_with_no_metadata_fails(self) -> None:
        verdict = evaluate_trivyignore("CVE-2024-99999\n", today=_TODAY)
        assert not verdict.passed
        assert "no preceding" in verdict.violations[0].reason

    def test_cve_field_mismatch_fails(self) -> None:
        text = (
            "# CVE: CVE-2024-00001\n"
            "# reason: no-upstream-fix\n"
            "# ticket: OMN-12345\n"
            "# expires: 2027-01-01\n"
            "CVE-2024-00002\n"
        )
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed
        assert "drift" in verdict.violations[0].reason

    def test_wrong_reason_fails(self) -> None:
        text = (
            "# CVE: CVE-2024-12345\n"
            "# reason: not-exploitable\n"
            "# ticket: OMN-12345\n"
            "# expires: 2027-01-01\n"
            "CVE-2024-12345\n"
        )
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed
        assert "reason" in verdict.violations[0].reason

    def test_missing_ticket_fails(self) -> None:
        text = (
            "# CVE: CVE-2024-12345\n"
            "# reason: no-upstream-fix\n"
            "# expires: 2027-01-01\n"
            "CVE-2024-12345\n"
        )
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed
        assert "ticket" in verdict.violations[0].reason

    def test_malformed_ticket_fails(self) -> None:
        text = _valid_block(ticket="JIRA-123")
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed
        assert "ticket" in verdict.violations[0].reason

    def test_malformed_expires_date_fails(self) -> None:
        text = (
            "# CVE: CVE-2024-12345\n"
            "# reason: no-upstream-fix\n"
            "# ticket: OMN-12345\n"
            "# expires: not-a-date\n"
            "CVE-2024-12345\n"
        )
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed
        assert "expires" in verdict.violations[0].reason

    def test_invalid_calendar_date_fails(self) -> None:
        text = _valid_block(expires="2026-02-30")
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed

    def test_blank_line_breaks_block_attachment(self) -> None:
        text = (
            "# CVE: CVE-2024-12345\n"
            "# reason: no-upstream-fix\n"
            "# ticket: OMN-12345\n"
            "\n"
            "# expires: 2027-01-01\n"
            "CVE-2024-12345\n"
        )
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed

    def test_multiple_entries_one_expired_one_valid(self) -> None:
        text = _valid_block(
            cve="CVE-2024-11111", ticket="OMN-1", expires="2026-01-01"
        ) + _valid_block(cve="CVE-2024-22222", ticket="OMN-2", expires="2027-01-01")
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed
        assert len(verdict.entries) == 2
        assert len(verdict.violations) == 1

    def test_ghsa_id_supported(self) -> None:
        text = _valid_block(cve="GHSA-f2ff-p2ww-7p4p", expires="2027-01-01")
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert verdict.passed

    def test_inline_trivy_expiry_syntax_fails_closed(self) -> None:
        text = (
            "# CVE: CVE-2024-12345\n"
            "# reason: no-upstream-fix\n"
            "# ticket: OMN-12345\n"
            "# expires: 2027-01-01\n"
            "CVE-2024-12345 exp:2027-01-01\n"
        )
        verdict = evaluate_trivyignore(text, today=_TODAY)
        assert not verdict.passed
        assert "unsupported .trivyignore line" in verdict.violations[0].reason

    def test_default_today_uses_utc_date(self) -> None:
        class FixedDateTime:
            @classmethod
            def now(cls, tz: object = None) -> datetime:
                assert tz is not None
                return datetime(2026, 8, 18)

        text = _valid_block(expires="2026-08-18")
        with patch("scripts.ci.check_trivyignore_expiry.datetime", FixedDateTime):
            verdict = evaluate_trivyignore(text)

        assert not verdict.passed
        assert "today is 2026-08-18" in verdict.violations[0].reason

    def test_trivy_report_unfixed_entry_passes(self) -> None:
        report = {
            "Results": [
                {
                    "Vulnerabilities": [
                        {
                            "VulnerabilityID": "CVE-2024-12345",
                            "FixedVersion": "",
                        }
                    ]
                }
            ]
        }
        verdict = evaluate_trivyignore(
            _valid_block(expires="2027-01-01"),
            today=_TODAY,
            trivy_report=report,
        )
        assert verdict.passed

    def test_trivy_report_fixed_entry_fails(self) -> None:
        report = {
            "Results": [
                {
                    "Vulnerabilities": [
                        {
                            "VulnerabilityID": "CVE-2024-12345",
                            "FixedVersion": "1.2.3",
                        }
                    ]
                }
            ]
        }
        verdict = evaluate_trivyignore(
            _valid_block(expires="2027-01-01"),
            today=_TODAY,
            trivy_report=report,
        )
        assert not verdict.passed
        assert "fixed version" in verdict.violations[0].reason

    def test_trivy_report_missing_entry_fails_as_stale(self) -> None:
        report = {"Results": [{"Vulnerabilities": []}]}
        verdict = evaluate_trivyignore(
            _valid_block(expires="2027-01-01"),
            today=_TODAY,
            trivy_report=report,
        )
        assert not verdict.passed
        assert "not present" in verdict.violations[0].reason


class TestCliMain:
    def test_missing_file_passes(self, tmp_path: Path) -> None:
        assert main([str(tmp_path / "missing.trivyignore")]) == 0

    def test_valid_file_exits_0(self, tmp_path: Path) -> None:
        f = tmp_path / ".trivyignore"
        f.write_text(_valid_block(expires="2099-01-01"))
        assert main([str(f)]) == 0

    def test_empty_file_exits_0(self, tmp_path: Path) -> None:
        f = tmp_path / ".trivyignore"
        f.write_text("")
        assert main([str(f)]) == 0

    def test_expired_file_exits_1(self, tmp_path: Path) -> None:
        f = tmp_path / ".trivyignore"
        f.write_text(_valid_block(expires="2020-01-01"))
        assert main([str(f)]) == 1
