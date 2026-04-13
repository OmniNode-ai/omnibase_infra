# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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
def test_detects_missing_ecr_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    import subprocess

    from compare_environments import check_ecr_tag_validity

    monkeypatch.setattr(shutil, "which", lambda _x: "/usr/bin/aws")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_kw: type(
            "R", (), {"returncode": 1, "stderr": "ImageNotFoundException", "stdout": ""}
        )(),
    )
    findings = check_ecr_tag_validity(
        deployments={
            "omniintelligence": "123.dkr.ecr.us-east-1.amazonaws.com/omniintelligence:stale"
        },
        region="us-east-1",
    )
    assert any(f.severity == "CRITICAL" and "stale" in f.detail for f in findings)


@pytest.mark.unit
def test_skips_ecr_check_when_aws_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    from compare_environments import check_ecr_tag_validity

    monkeypatch.setattr(shutil, "which", lambda _x: None)
    findings = check_ecr_tag_validity({"svc": "123.ecr/repo:tag"}, region="us-east-1")
    assert all(f.severity == "INFO" for f in findings)


@pytest.mark.unit
def test_detects_schema_drift_cloud_ahead() -> None:
    from compare_environments import check_db_schema_parity

    findings = check_db_schema_parity(
        local_migration="2026-02-01", cloud_migration="2026-03-10"
    )
    assert any(
        f.severity == "CRITICAL" and "cloud ahead" in f.detail.lower() for f in findings
    )


@pytest.mark.unit
def test_detects_schema_drift_local_ahead() -> None:
    from compare_environments import check_db_schema_parity

    findings = check_db_schema_parity(
        local_migration="2026-03-10", cloud_migration="2026-02-01"
    )
    assert any(
        f.severity == "WARNING" and "local ahead" in f.detail.lower() for f in findings
    )


@pytest.mark.unit
def test_schema_parity_clean_when_equal() -> None:
    from compare_environments import check_db_schema_parity

    findings = check_db_schema_parity(
        local_migration="2026-03-10", cloud_migration="2026-03-10"
    )
    assert findings == []


@pytest.mark.unit
def test_package_version_mismatch_is_warning() -> None:
    from compare_environments import check_package_version_parity

    findings = check_package_version_parity(
        local_versions={"omnibase-core": "1.0.0", "omnibase-spi": "0.15.2"},
        cloud_versions={"omnibase-core": "1.1.0", "omnibase-spi": "0.15.2"},
    )
    assert any(
        f.severity == "WARNING" and "omnibase-core" in f.detail for f in findings
    )
    assert not any("omnibase-spi" in f.detail for f in findings)


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


# ---------------------------------------------------------------------------
# Task 6: Service deployment and feature flag parity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_service_parity_local_only_is_warning() -> None:
    from compare_environments import check_service_deployment_parity

    findings = check_service_deployment_parity(
        local_services={"omninode-runtime", "omnibase-intelligence-api"},
        cloud_services={"omnibase-intelligence-api"},
    )
    assert any(
        f.severity == "WARNING" and "omninode-runtime" in f.title for f in findings
    )
    assert not any(f.severity == "CRITICAL" for f in findings)


@pytest.mark.unit
def test_service_parity_cloud_only_is_info() -> None:
    from compare_environments import check_service_deployment_parity

    findings = check_service_deployment_parity(
        local_services={"omninode-runtime"},
        cloud_services={"omninode-runtime", "omnibase-intelligence-api"},
    )
    assert any(
        f.severity == "INFO" and "omnibase-intelligence-api" in f.title
        for f in findings
    )


@pytest.mark.unit
def test_feature_flag_mismatch_is_warning() -> None:
    from compare_environments import check_feature_flag_consistency

    findings = check_feature_flag_consistency(
        local_flags={"ENABLE_REAL_TIME_EVENTS": "true"},
        cloud_flags={"ENABLE_REAL_TIME_EVENTS": "false"},
    )
    assert any(
        f.severity == "WARNING" and "ENABLE_REAL_TIME_EVENTS" in f.title
        for f in findings
    )


# ---------------------------------------------------------------------------
# Task 7: Kafka topic parity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_kafka_topic_local_only_is_warning() -> None:
    from compare_environments import check_kafka_topic_parity

    findings = check_kafka_topic_parity(
        local_topics={"agent.routing.requested.v1", "agent-actions"},
        cloud_topics={"agent-actions"},
    )
    assert any(
        f.severity == "WARNING" and "agent.routing.requested.v1" in f.detail
        for f in findings
    )


@pytest.mark.unit
def test_kafka_internal_topics_filtered() -> None:
    from compare_environments import check_kafka_topic_parity

    findings = check_kafka_topic_parity(
        local_topics={"__consumer_offsets", "agent-actions"},
        cloud_topics={"agent-actions"},
    )
    assert not any("__consumer_offsets" in f.title for f in findings)


# ---------------------------------------------------------------------------
# Task 8: Preflight and fix mode restriction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fix_mode_only_fixes_infisical_paths() -> None:
    from compare_environments import AUTO_FIXABLE_CHECKS

    assert {"infisical_path_completeness"} == AUTO_FIXABLE_CHECKS


@pytest.mark.unit
def test_preflight_returns_none_when_ssm_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    from compare_environments import SsmRunner, preflight_namespace_check

    monkeypatch.setattr(shutil, "which", lambda _x: None)
    ssm = SsmRunner("i-test", "us-east-1", timeout=5)
    result = preflight_namespace_check(ssm, "onex-dev")
    assert result is None


# ---------------------------------------------------------------------------
# Task 9: End-to-end smoke test with mocked SSM
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_parity_check_end_to_end_with_mocked_ssm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full pipeline: mocked SSM returning credential drift → CRITICAL finding in report."""
    import shutil

    from compare_environments import SsmResult, SsmRunner, run_parity_check

    def mock_ssm_run(self: SsmRunner, _command: str) -> SsmResult:
        payload = {
            "onex_runtime_credentials": {
                "OMNIINTELLIGENCE_DB_URL": "postgresql://role_omniintelligence:pass@host/db"
            },
            "omniintelligence_credentials": {
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "bad",
            },
            "omnidash_credentials": {
                "POSTGRES_USER": "role_omnidash",
                "POSTGRES_PASSWORD": "ok",
            },
        }
        return SsmResult(returncode=0, stdout=json.dumps(payload))

    monkeypatch.setattr(shutil, "which", lambda _x: "/usr/bin/aws")
    monkeypatch.setattr(SsmRunner, "run", mock_ssm_run)

    report = run_parity_check(
        mode="check",
        checks=["credential"],
        namespace="onex-dev",
        instance_id="i-mock",
        region="us-east-1",
        timeout=5,
    )
    assert report.summary.critical_count >= 1
    assert report.findings[0].check_id == "credential_parity"
    assert report.findings[0].severity == "CRITICAL"
