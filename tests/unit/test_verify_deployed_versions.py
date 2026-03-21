# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/verify_deployed_versions.py [OMN-5608].

Tests the version verification logic using a mock subprocess runner
so no Docker daemon is needed.
"""

from __future__ import annotations

# Import the module under test by path -- the script lives under scripts/,
# not in the installed package, so we import it via importlib.
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "verify_deployed_versions.py"
)
_spec = importlib.util.spec_from_file_location("verify_deployed_versions", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["verify_deployed_versions"] = _mod
_spec.loader.exec_module(_mod)

VersionCheckResult = _mod.VersionCheckResult
VerificationReport = _mod.VerificationReport
verify_versions = _mod.verify_versions
get_installed_version = _mod.get_installed_version
parse_versions_string = _mod.parse_versions_string
format_report_text = _mod.format_report_text
main = _mod.main


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_runner(
    outputs: dict[str, str],
    returncode: int = 0,
) -> Any:
    """Create a mock subprocess runner that returns canned pip-show output.

    Args:
        outputs: Mapping of package name to ``uv pip show`` stdout text.
        returncode: Exit code to return for all calls.
    """

    def runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # cmd is: docker exec <container> uv pip show <package>
        package = cmd[-1]
        stdout = outputs.get(package, "")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    return runner


def _pip_show_output(name: str, version: str) -> str:
    """Generate realistic ``uv pip show`` output."""
    return (
        f"Name: {name}\n"
        f"Version: {version}\n"
        f"Location: /usr/local/lib/python3.12/site-packages\n"
    )


# ─── Tests: parse_versions_string ──────────────────────────────────────────


class TestParseVersionsString:
    def test_simple(self) -> None:
        result = parse_versions_string("omniintelligence=0.16.0")
        assert result == {"omniintelligence": "0.16.0"}

    def test_multiple(self) -> None:
        result = parse_versions_string(
            "omniintelligence=0.16.0,omninode-claude=0.4.0,omninode-memory=0.6.1"
        )
        assert result == {
            "omniintelligence": "0.16.0",
            "omninode-claude": "0.4.0",
            "omninode-memory": "0.6.1",
        }

    def test_whitespace_tolerance(self) -> None:
        result = parse_versions_string(" pkg=1.0.0 , other=2.0.0 ")
        assert result == {"pkg": "1.0.0", "other": "2.0.0"}

    def test_empty_string(self) -> None:
        result = parse_versions_string("")
        assert result == {}

    def test_invalid_token(self) -> None:
        with pytest.raises(ValueError, match="Invalid version token"):
            parse_versions_string("no-equals-sign")


# ─── Tests: get_installed_version ──────────────────────────────────────────


class TestGetInstalledVersion:
    def test_success(self) -> None:
        runner = _make_runner(
            {"omniintelligence": _pip_show_output("omniintelligence", "0.16.0")}
        )
        version, error = get_installed_version(
            "test-container", "omniintelligence", runner=runner
        )
        assert version == "0.16.0"
        assert error is None

    def test_package_not_found(self) -> None:
        runner = _make_runner({}, returncode=1)
        version, error = get_installed_version(
            "test-container", "nonexistent-pkg", runner=runner
        )
        assert version is None
        assert error is not None
        assert "docker exec failed" in error

    def test_no_version_line(self) -> None:
        runner = _make_runner({"pkg": "Name: pkg\nSummary: something\n"})
        version, error = get_installed_version("test-container", "pkg", runner=runner)
        assert version is None
        assert error is not None
        assert "No 'Version:' line" in error

    def test_timeout(self) -> None:
        def timeout_runner(cmd: list[str], **kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd, 30)

        version, error = get_installed_version(
            "test-container", "pkg", runner=timeout_runner
        )
        assert version is None
        assert error is not None
        assert "Timed out" in error


# ─── Tests: verify_versions ───────────────────────────────────────────────


class TestVerifyVersions:
    def test_all_match(self) -> None:
        runner = _make_runner(
            {
                "omniintelligence": _pip_show_output("omniintelligence", "0.16.0"),
                "omninode-claude": _pip_show_output("omninode-claude", "0.4.0"),
            }
        )
        report = verify_versions(
            "test-container",
            {"omniintelligence": "0.16.0", "omninode-claude": "0.4.0"},
            runner=runner,
        )
        assert report.all_match is True
        assert len(report.results) == 2
        assert len(report.mismatches) == 0

    def test_version_mismatch(self) -> None:
        """Core test for OMN-5608: mismatch detection path."""
        runner = _make_runner(
            {
                "omniintelligence": _pip_show_output("omniintelligence", "0.15.0"),
            }
        )
        report = verify_versions(
            "test-container",
            {"omniintelligence": "0.16.0"},
            runner=runner,
        )
        assert report.all_match is False
        assert len(report.mismatches) == 1

        mismatch = report.mismatches[0]
        assert mismatch.package == "omniintelligence"
        assert mismatch.expected == "0.16.0"
        assert mismatch.actual == "0.15.0"
        assert mismatch.match is False
        assert mismatch.error is None

    def test_mixed_match_and_mismatch(self) -> None:
        runner = _make_runner(
            {
                "omniintelligence": _pip_show_output("omniintelligence", "0.16.0"),
                "omninode-claude": _pip_show_output("omninode-claude", "0.3.0"),
            }
        )
        report = verify_versions(
            "test-container",
            {"omniintelligence": "0.16.0", "omninode-claude": "0.4.0"},
            runner=runner,
        )
        assert report.all_match is False
        assert len(report.mismatches) == 1
        assert report.mismatches[0].package == "omninode-claude"

    def test_container_unreachable(self) -> None:
        runner = _make_runner({}, returncode=1)
        report = verify_versions(
            "nonexistent-container",
            {"omniintelligence": "0.16.0"},
            runner=runner,
        )
        assert report.all_match is False
        assert report.results[0].error is not None

    def test_empty_versions(self) -> None:
        runner = _make_runner({})
        report = verify_versions("test-container", {}, runner=runner)
        assert report.all_match is False  # no results means not all_match
        assert len(report.results) == 0


# ─── Tests: format_report_text ─────────────────────────────────────────────


class TestFormatReportText:
    def test_all_match_output(self) -> None:
        report = VerificationReport(
            container="test",
            results=[
                VersionCheckResult("pkg", "1.0.0", "1.0.0", True),
            ],
        )
        text = format_report_text(report)
        assert "ALL VERSIONS MATCH" in text
        assert "OK" in text

    def test_mismatch_output(self) -> None:
        report = VerificationReport(
            container="test",
            results=[
                VersionCheckResult("pkg", "2.0.0", "1.0.0", False),
            ],
        )
        text = format_report_text(report)
        assert "MISMATCH" in text
        assert "FAIL" in text
        assert "expected 2.0.0" in text

    def test_error_output(self) -> None:
        report = VerificationReport(
            container="test",
            results=[
                VersionCheckResult("pkg", "1.0.0", None, False, error="container down"),
            ],
        )
        text = format_report_text(report)
        assert "container down" in text


# ─── Tests: main (CLI) ────────────────────────────────────────────────────


class TestMainCLI:
    def test_exit_0_on_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = _make_runner(
            {"omniintelligence": _pip_show_output("omniintelligence", "0.16.0")}
        )
        monkeypatch.setattr(_mod, "subprocess", MagicMock())

        # Patch verify_versions to use our runner
        original_verify = _mod.verify_versions

        def patched_verify(
            container: str, expected: dict[str, str], **kwargs: Any
        ) -> VerificationReport:
            return original_verify(container, expected, runner=runner)

        monkeypatch.setattr(_mod, "verify_versions", patched_verify)
        exit_code = main(["--versions", "omniintelligence=0.16.0"])
        assert exit_code == 0

    def test_exit_1_on_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = _make_runner(
            {"omniintelligence": _pip_show_output("omniintelligence", "0.15.0")}
        )
        original_verify = _mod.verify_versions

        def patched_verify(
            container: str, expected: dict[str, str], **kwargs: Any
        ) -> VerificationReport:
            return original_verify(container, expected, runner=runner)

        monkeypatch.setattr(_mod, "verify_versions", patched_verify)
        exit_code = main(["--versions", "omniintelligence=0.16.0"])
        assert exit_code == 1

    def test_exit_2_on_infra_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = _make_runner({}, returncode=1)
        original_verify = _mod.verify_versions

        def patched_verify(
            container: str, expected: dict[str, str], **kwargs: Any
        ) -> VerificationReport:
            return original_verify(container, expected, runner=runner)

        monkeypatch.setattr(_mod, "verify_versions", patched_verify)
        exit_code = main(["--versions", "omniintelligence=0.16.0"])
        assert exit_code == 2


# ─── Tests: VerificationReport serialization ───────────────────────────────


class TestVerificationReportSerialization:
    def test_to_dict(self) -> None:
        report = VerificationReport(
            container="test",
            results=[
                VersionCheckResult("pkg", "1.0.0", "1.0.0", True),
                VersionCheckResult("pkg2", "2.0.0", "1.9.0", False),
            ],
        )
        d = report.to_dict()
        assert d["container"] == "test"
        assert d["all_match"] is False
        assert len(d["results"]) == 2  # type: ignore[arg-type]
