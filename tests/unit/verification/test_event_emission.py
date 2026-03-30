# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for verification event emission."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.event_emission import (
    _build_event_payload,
    _load_publish_topic,
    emit_verification_result,
)
from omnibase_infra.verification.models import (
    ModelContractCheckResult,
    ModelContractVerificationReport,
)


def _make_report(
    contract_name: str = "test_orch",
    verdict: EnumValidationVerdict = EnumValidationVerdict.PASS,
    fail_count: int = 0,
) -> ModelContractVerificationReport:
    """Build a test report with the specified overall verdict."""
    checks: list[ModelContractCheckResult] = []

    # Add passing checks
    checks.append(
        ModelContractCheckResult(
            check_type=EnumContractCheckType.REGISTRATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence="ok",
            contract_name=contract_name,
            message="pass",
        )
    )

    # Add failing checks if requested
    for i in range(fail_count):
        checks.append(
            ModelContractCheckResult(
                check_type=EnumContractCheckType.SUBSCRIPTION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.FAIL,
                evidence=f"fail {i}",
                contract_name=contract_name,
                message=f"fail {i}",
            )
        )

    checks_tuple = tuple(checks)
    return ModelContractVerificationReport(
        contract_name=contract_name,
        node_type="ORCHESTRATOR_GENERIC",
        checks=checks_tuple,
        overall_verdict=verdict,
        checked_at=datetime(2026, 3, 30, 12, 0, 0, tzinfo=UTC),
        duration_ms=150,
        report_fingerprint=ModelContractVerificationReport.compute_fingerprint(
            checks_tuple
        ),
    )


@pytest.mark.unit
class TestLoadPublishTopic:
    def test_loads_topic_from_contract(self) -> None:
        topic = _load_publish_topic()
        assert topic == "onex.evt.platform.contract-verification-result.v1"

    def test_topic_follows_naming_convention(self) -> None:
        topic = _load_publish_topic()
        assert topic.startswith("onex.evt.")
        assert ".v1" in topic


@pytest.mark.unit
class TestBuildEventPayload:
    def test_payload_has_required_fields(self) -> None:
        report = _make_report()
        payload = _build_event_payload(report)

        assert payload["contract_name"] == "test_orch"
        assert payload["overall_verdict"] == "pass"
        assert payload["check_count"] == 1
        assert payload["fail_count"] == 0
        assert payload["duration_ms"] == 150
        assert "checked_at" in payload
        assert "emitted_at" in payload
        assert "report_fingerprint" in payload

    def test_payload_fail_count_accurate(self) -> None:
        report = _make_report(verdict=EnumValidationVerdict.FAIL, fail_count=2)
        payload = _build_event_payload(report)
        assert payload["fail_count"] == 2
        assert payload["check_count"] == 3  # 1 pass + 2 fails

    def test_payload_fingerprint_matches_report(self) -> None:
        report = _make_report()
        payload = _build_event_payload(report)
        assert payload["report_fingerprint"] == report.report_fingerprint

    def test_payload_checked_at_is_iso(self) -> None:
        report = _make_report()
        payload = _build_event_payload(report)
        # Should parse as ISO datetime without error
        datetime.fromisoformat(payload["checked_at"])

    def test_payload_emitted_at_is_iso(self) -> None:
        report = _make_report()
        payload = _build_event_payload(report)
        datetime.fromisoformat(payload["emitted_at"])


@pytest.mark.unit
class TestEmitVerificationResult:
    def test_calls_publish_fn_with_topic_and_payload(self) -> None:
        report = _make_report()
        publish_fn = MagicMock()

        emit_verification_result(report, publish_fn)

        publish_fn.assert_called_once()
        args = publish_fn.call_args
        topic = args[0][0]
        payload = args[0][1]
        assert topic == "onex.evt.platform.contract-verification-result.v1"
        assert payload["contract_name"] == "test_orch"

    def test_fire_and_forget_on_publish_error(self) -> None:
        report = _make_report()

        def failing_publish(topic: str, payload: dict[str, Any]) -> None:
            raise RuntimeError("Kafka down")

        # Should not raise
        emit_verification_result(report, failing_publish)

    def test_emits_for_failing_report(self) -> None:
        report = _make_report(verdict=EnumValidationVerdict.FAIL, fail_count=3)
        publish_fn = MagicMock()

        emit_verification_result(report, publish_fn)

        payload = publish_fn.call_args[0][1]
        assert payload["overall_verdict"] == "fail"
        assert payload["fail_count"] == 3

    def test_does_not_modify_report(self) -> None:
        report = _make_report()
        publish_fn = MagicMock()

        emit_verification_result(report, publish_fn)

        # Report is frozen (Pydantic), so this is a sanity check
        assert report.contract_name == "test_orch"
        assert report.overall_verdict == EnumValidationVerdict.PASS
