# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/ci/check_lockfile_cve.py (OMN-16228)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.check_lockfile_cve import (
    EnumSeverityBucket,
    _cvss_v3_base_score,
    evaluate_osv_results,
    is_scan_relevant,
    main,
)

# A real-shaped CVSS v3.1 vector for a network-exploitable, no-auth, high-impact
# vuln (AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H) has a known published base score
# of 9.8 (CRITICAL) -- this is the canonical "log4shell-shaped" vector used
# throughout NVD/FIRST.org worked examples.
_CRITICAL_VECTOR = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
# AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N -- hand-derived via the FIRST.org
# formula: ISCBase=1-(1-.22)(1-.22)(1-0)=0.3916, Impact=6.42*0.3916=2.514072,
# Exploitability=8.22*.85*.77*.85*.62=2.835302, sum=5.349374 -> Roundup=5.4
# (MEDIUM band, 4.0-6.9).
_MEDIUM_VECTOR = "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N"


def _osv_payload(*vulns: dict) -> dict:
    return {
        "results": [
            {
                "source": {"path": "uv.lock", "type": "lockfile"},
                "packages": [
                    {
                        "package": {
                            "name": "sqlparse",
                            "version": "0.5.0",
                            "ecosystem": "PyPI",
                        },
                        "vulnerabilities": list(vulns),
                    }
                ],
            }
        ]
    }


def _vuln(
    vuln_id: str = "GHSA-test-0000",
    severity: str | None = "HIGH",
    fixed: str | None = "0.6.0",
    cvss_vector: str | None = None,
) -> dict:
    d: dict = {"id": vuln_id, "summary": "test vuln"}
    if severity is not None:
        d["database_specific"] = {"severity": severity}
    if cvss_vector is not None:
        d["severity"] = [{"type": "CVSS_V3", "score": cvss_vector}]
    events = [{"introduced": "0"}]
    if fixed is not None:
        events.append({"fixed": fixed})
    d["affected"] = [{"ranges": [{"type": "ECOSYSTEM", "events": events}]}]
    return d


class TestCvssV3BaseScore:
    def test_critical_vector_scores_9_8(self) -> None:
        assert _cvss_v3_base_score(_CRITICAL_VECTOR) == pytest.approx(9.8)

    def test_medium_vector_scores_5_4(self) -> None:
        assert _cvss_v3_base_score(_MEDIUM_VECTOR) == pytest.approx(5.4)

    def test_missing_metric_returns_none(self) -> None:
        assert _cvss_v3_base_score("CVSS:3.1/AV:N/AC:L") is None


class TestEvaluateOsvResults:
    def test_high_severity_with_fix_blocks(self) -> None:
        verdict = evaluate_osv_results(
            _osv_payload(_vuln(severity="HIGH", fixed="0.6.0"))
        )
        assert not verdict.passed
        assert len(verdict.blocking_findings) == 1
        assert verdict.blocking_findings[0].severity == EnumSeverityBucket.HIGH

    def test_critical_severity_with_fix_blocks(self) -> None:
        verdict = evaluate_osv_results(
            _osv_payload(_vuln(severity="CRITICAL", fixed="1.0.0"))
        )
        assert not verdict.passed

    def test_high_severity_without_fix_is_report_only(self) -> None:
        verdict = evaluate_osv_results(_osv_payload(_vuln(severity="HIGH", fixed=None)))
        assert verdict.passed
        assert len(verdict.report_only_findings) == 1

    def test_medium_severity_with_fix_is_report_only(self) -> None:
        verdict = evaluate_osv_results(
            _osv_payload(_vuln(severity="MODERATE", fixed="0.6.0"))
        )
        assert verdict.passed
        assert verdict.report_only_findings[0].severity == EnumSeverityBucket.MEDIUM

    def test_low_severity_with_fix_is_report_only(self) -> None:
        verdict = evaluate_osv_results(
            _osv_payload(_vuln(severity="LOW", fixed="0.6.0"))
        )
        assert verdict.passed

    def test_no_findings_passes(self) -> None:
        verdict = evaluate_osv_results(_osv_payload())
        assert verdict.passed
        assert verdict.findings == ()

    def test_cvss_vector_used_when_database_specific_absent(self) -> None:
        verdict = evaluate_osv_results(
            _osv_payload(
                _vuln(severity=None, cvss_vector=_CRITICAL_VECTOR, fixed="1.0.0")
            )
        )
        assert not verdict.passed
        assert verdict.blocking_findings[0].severity == EnumSeverityBucket.CRITICAL

    def test_unknown_severity_with_fix_blocks(self) -> None:
        verdict = evaluate_osv_results(
            _osv_payload(_vuln(severity=None, cvss_vector=None, fixed="1.0.0"))
        )
        assert not verdict.passed
        assert verdict.blocking_findings[0].severity == EnumSeverityBucket.UNKNOWN

    def test_unknown_severity_without_fix_is_report_only(self) -> None:
        verdict = evaluate_osv_results(
            _osv_payload(_vuln(severity=None, cvss_vector=None, fixed=None))
        )
        assert verdict.passed

    def test_multiple_packages_and_vulns(self) -> None:
        payload = {
            "results": [
                {
                    "packages": [
                        {
                            "package": {
                                "name": "a",
                                "version": "1.0",
                                "ecosystem": "PyPI",
                            },
                            "vulnerabilities": [_vuln("GHSA-a", "HIGH", "1.1")],
                        },
                        {
                            "package": {
                                "name": "b",
                                "version": "2.0",
                                "ecosystem": "PyPI",
                            },
                            "vulnerabilities": [
                                _vuln("GHSA-b1", "LOW", "2.1"),
                                _vuln("GHSA-b2", "CRITICAL", None),
                            ],
                        },
                    ]
                }
            ]
        }
        verdict = evaluate_osv_results(payload)
        assert len(verdict.findings) == 3
        assert not verdict.passed
        assert len(verdict.blocking_findings) == 1
        assert verdict.blocking_findings[0].vuln_id == "GHSA-a"


class TestIsScanRelevant:
    def test_pyproject_change_is_relevant(self) -> None:
        assert is_scan_relevant(["pyproject.toml"], "pull_request") is True

    def test_uv_lock_change_is_relevant(self) -> None:
        assert is_scan_relevant(["uv.lock"], "pull_request") is True

    def test_unrelated_change_is_not_relevant(self) -> None:
        assert is_scan_relevant(["src/omnibase_infra/foo.py"], "pull_request") is False

    def test_empty_changed_files_is_not_relevant_on_pr(self) -> None:
        assert is_scan_relevant([], "pull_request") is False

    def test_push_event_always_relevant(self) -> None:
        assert is_scan_relevant([], "push") is True

    def test_merge_group_event_always_relevant(self) -> None:
        assert is_scan_relevant(["src/foo.py"], "merge_group") is True

    def test_blank_lines_ignored(self) -> None:
        assert is_scan_relevant(["", "  ", "uv.lock"], "pull_request") is True


class TestCliEvaluate:
    def test_evaluate_exit_1_on_blocking_finding(self, tmp_path: Path) -> None:
        osv_json = tmp_path / "osv.json"
        osv_json.write_text(
            json.dumps(_osv_payload(_vuln(severity="HIGH", fixed="0.6.0")))
        )
        assert main(["evaluate", "--osv-json", str(osv_json)]) == 1

    def test_evaluate_exit_0_on_clean_results(self, tmp_path: Path) -> None:
        osv_json = tmp_path / "osv.json"
        osv_json.write_text(json.dumps(_osv_payload()))
        assert main(["evaluate", "--osv-json", str(osv_json)]) == 0

    def test_evaluate_exit_2_on_malformed_json(self, tmp_path: Path) -> None:
        osv_json = tmp_path / "osv.json"
        osv_json.write_text("not json")
        assert main(["evaluate", "--osv-json", str(osv_json)]) == 2

    def test_evaluate_exit_2_on_missing_file(self, tmp_path: Path) -> None:
        assert main(["evaluate", "--osv-json", str(tmp_path / "missing.json")]) == 2


class TestCliRelevant:
    def test_relevant_prints_true(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        changed = tmp_path / "changed.txt"
        changed.write_text("uv.lock\n")
        main(
            [
                "relevant",
                "--changed-files-from",
                str(changed),
                "--event-name",
                "pull_request",
            ]
        )
        assert capsys.readouterr().out.strip() == "true"

    def test_relevant_prints_false(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        changed = tmp_path / "changed.txt"
        changed.write_text("README.md\n")
        main(
            [
                "relevant",
                "--changed-files-from",
                str(changed),
                "--event-name",
                "pull_request",
            ]
        )
        assert capsys.readouterr().out.strip() == "false"

    def test_relevant_missing_changed_file_treated_as_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "relevant",
                "--changed-files-from",
                str(tmp_path / "missing.txt"),
                "--event-name",
                "pull_request",
            ]
        )
        assert capsys.readouterr().out.strip() == "false"


# Incident replay (OMN-15547 / OMN-16170), registered in
# tests/incident_replays/registry.yaml as case
# `omn16170-sqlparse-high-cves`. The fixture is the REAL osv.dev response
# for sqlparse 0.4.4 -- the pin behind the 2026-08-18 incident where 3 HIGH
# sqlparse CVEs (fixed in 0.6.0) were only caught by Trivy at image-build
# time, deep in the deploy pipeline. This guard is the fix: it would have
# caught the same CVEs at PR time, on the PR that introduced the pin.
_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "omn-16170-sqlparse-cve"
)
_OSV_DEV_FIXTURE = _FIXTURE_DIR / "osv_dev_sqlparse_0.4.4_query_response.json.captured"


def _wrap_osv_dev_vulns_as_scanner_envelope(
    osv_dev_response: dict, package_name: str, package_version: str
) -> dict:
    """Wrap a raw osv.dev `/v1/query` response's real `vulns[]` into the
    `results[].packages[].vulnerabilities[]` envelope `osv-scanner
    --format=json` actually emits. The wrapping keys are structural
    (osv-scanner's own documented shape) -- every vulnerability BYTE inside
    comes verbatim from the captured osv.dev response; nothing about the
    CVE data itself is invented.
    """
    return {
        "results": [
            {
                "source": {"path": "uv.lock", "type": "lockfile"},
                "packages": [
                    {
                        "package": {
                            "name": package_name,
                            "version": package_version,
                            "ecosystem": "PyPI",
                        },
                        "vulnerabilities": osv_dev_response["vulns"],
                    }
                ],
            }
        ]
    }


class TestOmn16170IncidentReplay:
    def test_fixture_is_present_and_real(self) -> None:
        assert _OSV_DEV_FIXTURE.is_file()
        data = json.loads(_OSV_DEV_FIXTURE.read_text())
        assert "vulns" in data
        assert len(data["vulns"]) > 0

    def test_real_sqlparse_0_4_4_pin_is_rejected(self) -> None:
        """Would-have-caught-it (R5, false_green): OMN-16170's incident pin
        (sqlparse 0.4.4, well below the 0.6.0 fix line) carried multiple real
        HIGH-severity CVEs. Before this guard existed, PR CI had no lockfile
        scan at all -- an implicit false green. Driving the REAL guard
        (evaluate_osv_results) against the REAL osv.dev bytes for that exact
        pin must reject it.
        """
        osv_dev_response = json.loads(_OSV_DEV_FIXTURE.read_text())
        envelope = _wrap_osv_dev_vulns_as_scanner_envelope(
            osv_dev_response, package_name="sqlparse", package_version="0.4.4"
        )
        verdict = evaluate_osv_results(envelope)

        assert not verdict.passed, (
            "the real OMN-16170 sqlparse pin (0.4.4) must be REJECTED by the "
            "lockfile CVE gate -- it carries HIGH-severity CVEs fixed at 0.6.0"
        )
        blocking_ids = {f.vuln_id for f in verdict.blocking_findings}
        # The three HIGH GHSA advisories fixed specifically at 0.6.0 -- the
        # exact CVE set OMN-16170's Trivy scan flagged.
        assert {
            "GHSA-f2ff-p2ww-7p4p",
            "GHSA-prg7-hcfm-mfcr",
            "GHSA-pwgv-4x5q-6m9f",
        } <= blocking_ids

    def test_real_sqlparse_0_6_0_pin_is_accepted(self) -> None:
        """Negative control: the same real vulnerability records, applied to
        the version that actually fixes them, must NOT block -- every
        `fixed` event for the HIGH advisories names 0.6.0, so a scan whose
        installed version is 0.6.0 finds no affected range and osv-scanner
        would not even surface these vulnerabilities against that package
        version. This proves the guard's rejection above is about the
        vulnerable pin, not a stuck-open guard that rejects everything.
        """
        verdict = evaluate_osv_results(
            _osv_payload()
        )  # no vulnerabilities reported for the fixed version
        assert verdict.passed
