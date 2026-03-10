# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for scripts/compare-environments.py — parity checker."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


@pytest.mark.unit
def test_parity_report_serializes_to_json() -> None:
    from compare_environments import (
        ModelParityFinding,
        ModelParityReport,
        ModelParitySummary,
    )

    finding = ModelParityFinding(
        check_id="credential_parity",
        severity="CRITICAL",
        title="Wrong POSTGRES_USER for omniintelligence-credentials",
        detail="k8s secret has 'postgres'; expected 'role_omniintelligence'",
        local_value="role_omniintelligence",
        cloud_value="postgres",
        auto_fixable=False,
        fix_hint="Re-seed /dev/omniintelligence/ in Infisical and force-resync the InfisicalSecret",
    )
    report = ModelParityReport(
        run_id="abc123",
        generated_at="2026-03-10T00:00:00Z",
        mode="check",
        checks_run=["credential_parity"],
        findings=[finding],
        summary=ModelParitySummary(
            critical_count=1, warning_count=0, info_count=0, checks_skipped=[]
        ),
    )
    raw = report.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["findings"][0]["severity"] == "CRITICAL"
    assert parsed["summary"]["critical_count"] == 1


@pytest.mark.unit
def test_ssm_runner_skips_when_aws_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    from compare_environments import SsmRunner

    monkeypatch.setattr(shutil, "which", lambda _x: None)
    result = SsmRunner("i-test", "us-east-1", timeout=5).run("echo hi")
    assert result.skipped is True
    assert "aws CLI not found" in result.skip_reason


@pytest.mark.unit
def test_ssm_runner_skips_on_expired_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    import subprocess

    from compare_environments import SsmRunner

    monkeypatch.setattr(shutil, "which", lambda _x: "/usr/bin/aws")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_kw: type(
            "R",
            (),
            {"returncode": 255, "stdout": "", "stderr": "ExpiredTokenException"},
        )(),
    )
    result = SsmRunner("i-test", "us-east-1", timeout=5).run("echo hi")
    assert result.skipped is True
    assert "SSO session expired" in result.skip_reason


@pytest.mark.unit
def test_ssm_runner_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """SsmRunner.run() must return SsmResult on all failure paths, never raise."""
    import shutil

    from compare_environments import SsmRunner

    monkeypatch.setattr(shutil, "which", lambda _x: "/usr/bin/aws")
    monkeypatch.setattr(
        "subprocess.run", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("broken"))
    )
    result = SsmRunner("i-test", "us-east-1", timeout=5).run("echo hi")
    assert result.skipped is True


@pytest.mark.unit
def test_detects_wrong_postgres_user() -> None:
    from compare_environments import check_credential_parity

    # Simulates: omniintelligence-credentials has postgres instead of role_omniintelligence
    cloud_secrets = {
        "onex-runtime-credentials": {
            "OMNIINTELLIGENCE_DB_URL": "postgresql://role_omniintelligence:pass@host/db"
        },
        "omniintelligence-credentials": {
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "wrong",
        },
        "omnidash-credentials": {
            "POSTGRES_USER": "role_omnidash",
            "POSTGRES_PASSWORD": "ok",
        },
    }
    findings = check_credential_parity(cloud_secrets)
    critical = [f for f in findings if f.severity == "CRITICAL"]
    assert len(critical) >= 1
    assert any(
        "POSTGRES_USER" in f.title and "omniintelligence" in f.title.lower()
        for f in critical
    )
    # Correct service should produce no CRITICAL finding for omnidash POSTGRES_USER
    assert not any(
        "omnidash" in f.title.lower() and "POSTGRES_USER" in f.title for f in critical
    )


@pytest.mark.unit
def test_infisical_path_missing(httpserver: object) -> None:
    from compare_environments import probe_infisical_paths

    # httpserver fixture from pytest-httpserver
    httpserver.expect_request("/api/v1/secrets").respond_with_data(  # type: ignore[attr-defined]
        "", status=404
    )
    findings = probe_infisical_paths(
        infisical_addr=httpserver.url_for("/"),  # type: ignore[attr-defined]
        project_id="proj-id",
        paths=[("/dev/omniweb/", "dev", "omniweb-infisical-secret")],
        token="tok",
    )
    assert any(
        f.severity == "CRITICAL" and "/dev/omniweb/" in f.title for f in findings
    )
    assert any(f.auto_fixable is True for f in findings)
