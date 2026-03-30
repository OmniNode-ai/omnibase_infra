# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for contract verification data models [OMN-7042]."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.models.model_contract_check_result import (
    ModelContractCheckResult,
)
from omnibase_infra.verification.models.model_contract_verification_report import (
    ModelContractVerificationReport,
)


@pytest.mark.unit
class TestEnumContractCheckType:
    """Tests for EnumContractCheckType enum."""

    def test_has_all_expected_members(self) -> None:
        expected = {
            "REGISTRATION",
            "SUBSCRIPTION",
            "PUBLICATION",
            "HANDLER_EXECUTION",
            "PROJECTION_STATE",
            "FSM_STATE",
        }
        actual = {m.name for m in EnumContractCheckType}
        assert actual == expected

    def test_is_str_enum(self) -> None:
        assert isinstance(EnumContractCheckType.REGISTRATION, str)
        assert EnumContractCheckType.REGISTRATION == "registration"

    def test_str_returns_value(self) -> None:
        assert str(EnumContractCheckType.SUBSCRIPTION) == "subscription"


@pytest.mark.unit
class TestModelContractCheckResult:
    """Tests for ModelContractCheckResult model."""

    def test_create_valid_result(self) -> None:
        result = ModelContractCheckResult(
            check_type=EnumContractCheckType.REGISTRATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence="Node found in registration_projections with status ACTIVE",
            contract_name="node_registration_orchestrator",
            message="Registration check passed",
        )
        assert result.check_type == EnumContractCheckType.REGISTRATION
        assert result.severity == EnumCheckSeverity.REQUIRED
        assert result.verdict == EnumValidationVerdict.PASS

    def test_frozen(self) -> None:
        result = ModelContractCheckResult(
            check_type=EnumContractCheckType.REGISTRATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence="test",
            contract_name="test_contract",
            message="test",
        )
        with pytest.raises(Exception):
            result.verdict = EnumValidationVerdict.FAIL  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            ModelContractCheckResult(
                check_type=EnumContractCheckType.REGISTRATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.PASS,
                evidence="test",
                contract_name="test_contract",
                message="test",
                extra_field="not allowed",  # type: ignore[call-arg]
            )


@pytest.mark.unit
class TestModelContractVerificationReport:
    """Tests for ModelContractVerificationReport model."""

    @pytest.fixture
    def passing_checks(self) -> tuple[ModelContractCheckResult, ...]:
        return (
            ModelContractCheckResult(
                check_type=EnumContractCheckType.REGISTRATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.PASS,
                evidence="Node registered",
                contract_name="node_registration_orchestrator",
                message="Registration OK",
            ),
            ModelContractCheckResult(
                check_type=EnumContractCheckType.SUBSCRIPTION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.PASS,
                evidence="All topics subscribed",
                contract_name="node_registration_orchestrator",
                message="Subscription OK",
            ),
        )

    @pytest.fixture
    def failing_checks(self) -> tuple[ModelContractCheckResult, ...]:
        return (
            ModelContractCheckResult(
                check_type=EnumContractCheckType.REGISTRATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.FAIL,
                evidence="Node not found in registration_projections",
                contract_name="node_registration_orchestrator",
                message="Registration FAIL",
            ),
        )

    def test_bool_true_on_pass(
        self, passing_checks: tuple[ModelContractCheckResult, ...]
    ) -> None:
        report = ModelContractVerificationReport(
            contract_name="node_registration_orchestrator",
            node_type="ORCHESTRATOR",
            checks=passing_checks,
            overall_verdict=EnumValidationVerdict.PASS,
        )
        assert bool(report) is True

    def test_bool_false_on_fail(
        self, failing_checks: tuple[ModelContractCheckResult, ...]
    ) -> None:
        report = ModelContractVerificationReport(
            contract_name="node_registration_orchestrator",
            node_type="ORCHESTRATOR",
            checks=failing_checks,
            overall_verdict=EnumValidationVerdict.FAIL,
        )
        assert bool(report) is False

    def test_bool_false_on_quarantine(
        self, passing_checks: tuple[ModelContractCheckResult, ...]
    ) -> None:
        report = ModelContractVerificationReport(
            contract_name="node_registration_orchestrator",
            node_type="ORCHESTRATOR",
            checks=passing_checks,
            overall_verdict=EnumValidationVerdict.QUARANTINE,
        )
        assert bool(report) is False

    def test_frozen(self, passing_checks: tuple[ModelContractCheckResult, ...]) -> None:
        report = ModelContractVerificationReport(
            contract_name="node_registration_orchestrator",
            node_type="ORCHESTRATOR",
            checks=passing_checks,
            overall_verdict=EnumValidationVerdict.PASS,
        )
        with pytest.raises(Exception):
            report.overall_verdict = EnumValidationVerdict.FAIL  # type: ignore[misc]

    def test_extra_forbid(
        self, passing_checks: tuple[ModelContractCheckResult, ...]
    ) -> None:
        with pytest.raises(Exception):
            ModelContractVerificationReport(
                contract_name="test",
                node_type="ORCHESTRATOR",
                checks=passing_checks,
                overall_verdict=EnumValidationVerdict.PASS,
                bogus="not allowed",  # type: ignore[call-arg]
            )

    def test_defaults(
        self, passing_checks: tuple[ModelContractCheckResult, ...]
    ) -> None:
        report = ModelContractVerificationReport(
            contract_name="test",
            node_type="ORCHESTRATOR",
            checks=passing_checks,
            overall_verdict=EnumValidationVerdict.PASS,
        )
        assert report.probe_mode == "primary"
        assert report.degraded_probes == ()
        assert report.runtime_target == "localhost:8085"
        assert report.duration_ms == 0
        assert report.report_fingerprint == ""
        assert isinstance(report.checked_at, datetime)

    def test_compute_fingerprint_deterministic(
        self, passing_checks: tuple[ModelContractCheckResult, ...]
    ) -> None:
        fp1 = ModelContractVerificationReport.compute_fingerprint(passing_checks)
        fp2 = ModelContractVerificationReport.compute_fingerprint(passing_checks)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_compute_fingerprint_order_independent(self) -> None:
        check_a = ModelContractCheckResult(
            check_type=EnumContractCheckType.REGISTRATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence="a",
            contract_name="alpha",
            message="a",
        )
        check_b = ModelContractCheckResult(
            check_type=EnumContractCheckType.SUBSCRIPTION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence="b",
            contract_name="alpha",
            message="b",
        )
        fp_ab = ModelContractVerificationReport.compute_fingerprint((check_a, check_b))
        fp_ba = ModelContractVerificationReport.compute_fingerprint((check_b, check_a))
        assert fp_ab == fp_ba

    def test_from_attributes(self) -> None:
        """Verify from_attributes=True allows dict-like construction."""
        data = {
            "contract_name": "test",
            "node_type": "ORCHESTRATOR",
            "checks": (),
            "overall_verdict": EnumValidationVerdict.PASS,
            "checked_at": datetime.now(UTC),
            "probe_mode": "primary",
            "degraded_probes": (),
            "runtime_target": "localhost:8085",
            "duration_ms": 100,
            "report_fingerprint": "abc123",
        }
        report = ModelContractVerificationReport.model_validate(data)
        assert report.contract_name == "test"
        assert report.duration_ms == 100
