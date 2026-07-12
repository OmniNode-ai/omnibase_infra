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
# RT-6 (OMN-14469): image-revision readback surface.
RevisionCheckResult = _mod.RevisionCheckResult
get_image_revision = _mod.get_image_revision
verify_revision = _mod.verify_revision


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


# ─── Tests: image-revision readback (RT-6 OMN-14469) ───────────────────────


_REVISION_LABEL = "org.opencontainers.image.revision"


def _make_inspect_runner(
    revision: str,
    returncode: int = 0,
) -> Any:
    """Mock runner emulating ``docker inspect --format '{{...revision}}'``.

    ``revision`` is what the running container's revision label would print
    (Go template prints a trailing newline, which the parser strips).
    """

    def runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # cmd is: docker inspect <container> --format {{index .Config.Labels "..."}}
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=returncode,
            stdout=f"{revision}\n",
            stderr="" if returncode == 0 else "No such container",
        )

    return runner


class TestGetImageRevision:
    def test_success(self) -> None:
        runner = _make_inspect_runner("abc123def456")
        revision, error = get_image_revision("omninode-runtime", runner=runner)
        assert revision == "abc123def456"
        assert error is None

    def test_container_down(self) -> None:
        runner = _make_inspect_runner("", returncode=1)
        revision, error = get_image_revision("omninode-runtime", runner=runner)
        assert revision is None
        assert error is not None
        assert "docker inspect failed" in error

    def test_missing_label_no_value(self) -> None:
        """Go prints '<no value>' when the label key is absent -> fail-closed."""
        runner = _make_inspect_runner("<no value>")
        revision, error = get_image_revision("omninode-runtime", runner=runner)
        assert revision is None
        assert error is not None
        assert _REVISION_LABEL in error

    def test_missing_label_empty(self) -> None:
        runner = _make_inspect_runner("")
        revision, error = get_image_revision("omninode-runtime", runner=runner)
        assert revision is None
        assert error is not None

    def test_docker_not_found(self) -> None:
        def missing_docker(cmd: list[str], **kwargs: Any) -> None:
            raise FileNotFoundError("docker")

        revision, error = get_image_revision("c", runner=missing_docker)
        assert revision is None
        assert error is not None
        assert "docker command not found" in error


class TestVerifyRevision:
    def test_exact_match(self) -> None:
        runner = _make_inspect_runner("abc123def456")
        result = verify_revision("omninode-runtime", "abc123def456", runner=runner)
        assert result.match is True
        assert result.actual == "abc123def456"
        assert result.error is None

    def test_short_vs_full_sha_prefix_match(self) -> None:
        """A --short=12 label must match a full-40 intended SHA (and vice versa)."""
        full = "abc123def456789012345678901234567890abcd"
        runner = _make_inspect_runner("abc123def456")  # short label
        result = verify_revision("omninode-runtime", full, runner=runner)
        assert result.match is True

    def test_stale_container_goes_red(self) -> None:
        """ACCEPTANCE (RT-6, exists-but-WRONG): a stale running container whose
        image revision label != the intended git SHA must FAIL. Before this
        wiring, deploy-runtime.sh only warned and returned 0 -- this is the
        exact silent-pass the readback closes."""
        stale = "0000stale0000deadbeefcafef00dbaadf00dbaad"
        intended = "abc123def456"
        runner = _make_inspect_runner(stale)
        result = verify_revision("omninode-runtime", intended, runner=runner)
        assert result.match is False
        assert result.error is None  # a genuine mismatch, not an infra error
        assert result.actual == stale

    def test_container_down_is_infra_error(self) -> None:
        runner = _make_inspect_runner("", returncode=1)
        result = verify_revision("omninode-runtime", "abc123def456", runner=runner)
        assert result.match is False
        assert result.error is not None


class TestMainCLIRevision:
    """main() exit contract for the image-revision readback path.

    main() -> verify_revision() -> get_image_revision(): patching the module-level
    ``get_image_revision`` to inject a mock docker-inspect runner exercises the
    real verify_revision + exit-code aggregation without a Docker daemon.
    """

    @staticmethod
    def _patch_revision(monkeypatch: pytest.MonkeyPatch, runner: Any) -> None:
        monkeypatch.setattr(
            _mod,
            "get_image_revision",
            lambda container, **_kw: get_image_revision(container, runner=runner),
        )

    def test_exit_0_on_revision_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_revision(monkeypatch, _make_inspect_runner("abc123def456"))
        assert main(["--expected-revision", "abc123def456"]) == 0

    def test_exit_1_on_stale_revision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RED-on-stale at the CLI boundary -- the deploy readback exit contract."""
        self._patch_revision(monkeypatch, _make_inspect_runner("0000stale0000deadbeef"))
        assert main(["--expected-revision", "abc123def456"]) == 1

    def test_exit_2_on_container_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_revision(monkeypatch, _make_inspect_runner("", returncode=1))
        assert main(["--expected-revision", "abc123def456"]) == 2

    def test_exit_2_when_nothing_requested(self) -> None:
        """No --versions and no --expected-revision is a no-op readback -> refuse."""
        assert main([]) == 2

    def test_both_dimensions_must_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A matching version but a stale revision must still fail (RED)."""
        version_runner = _make_runner(
            {"omnibase-infra": _pip_show_output("omnibase-infra", "0.38.4")}
        )
        monkeypatch.setattr(
            _mod,
            "verify_versions",
            lambda container, expected, **_kw: verify_versions(
                container, expected, runner=version_runner
            ),
        )
        self._patch_revision(monkeypatch, _make_inspect_runner("0000stale0000"))
        exit_code = main(
            [
                "--versions",
                "omnibase-infra=0.38.4",
                "--expected-revision",
                "abc123def456",
            ]
        )
        assert exit_code == 1
