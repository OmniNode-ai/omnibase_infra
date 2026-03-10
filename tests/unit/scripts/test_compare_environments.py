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
