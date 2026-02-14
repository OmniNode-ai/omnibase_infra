# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for flake detection (OMN-2151).

Tests:
- FlakeDetector rerun-once rule
- ModelFlakeRecord model
- ModelFlakeDetectionResult aggregate
- should_quarantine_verdict adjustment
- is_promotion_blocked enforcement
- Edge cases (all pass, all fail, mixed)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumCheckSeverity, EnumValidationVerdict
from omnibase_infra.models.validation.model_check_result import ModelCheckResult
from omnibase_infra.validation.service_flake_detector import (
    FlakeDetector,
    ModelFlakeDetectionResult,
    ModelFlakeRecord,
    is_promotion_blocked,
    should_quarantine_verdict,
)

pytestmark = pytest.mark.unit


# ============================================================================
# Helpers
# ============================================================================


def _make_check_result(
    check_code: str = "CHECK-TEST-001",
    passed: bool = True,
    skipped: bool = False,
    severity: EnumCheckSeverity = EnumCheckSeverity.REQUIRED,
) -> ModelCheckResult:
    """Create a check result for testing."""
    return ModelCheckResult(
        check_code=check_code,
        label="Test check",
        severity=severity,
        passed=passed,
        skipped=skipped,
        message="test",
        executed_at=datetime.now(tz=UTC),
    )


# ============================================================================
# ModelFlakeRecord
# ============================================================================


class TestModelFlakeRecord:
    """Tests for ModelFlakeRecord model."""

    def test_basic_record(self) -> None:
        """ModelFlakeRecord can be created with required fields."""
        record = ModelFlakeRecord(
            check_code="CHECK-TEST-001",
            first_passed=False,
        )
        assert record.check_code == "CHECK-TEST-001"
        assert record.first_passed is False
        assert record.rerun_passed is None
        assert record.is_flake_suspected is False
        assert record.rerun_count == 0

    def test_flake_suspected_record(self) -> None:
        """ModelFlakeRecord with flake suspected flag."""
        record = ModelFlakeRecord(
            check_code="CHECK-TEST-001",
            first_passed=False,
            rerun_passed=True,
            is_flake_suspected=True,
            rerun_count=1,
        )
        assert record.is_flake_suspected is True
        assert record.rerun_passed is True
        assert record.rerun_count == 1

    def test_frozen(self) -> None:
        """ModelFlakeRecord is frozen (immutable)."""
        record = ModelFlakeRecord(check_code="X", first_passed=True)
        with pytest.raises(ValidationError):
            record.first_passed = False  # type: ignore[misc]


# ============================================================================
# ModelFlakeDetectionResult
# ============================================================================


class TestModelFlakeDetectionResult:
    """Tests for ModelFlakeDetectionResult aggregate model."""

    def test_empty_result(self) -> None:
        """Empty result has no flakes."""
        result = ModelFlakeDetectionResult()
        assert result.has_suspected_flakes is False
        assert result.quarantine_check_codes == ()
        assert result.quarantine_reasons == ()

    def test_result_with_flake(self) -> None:
        """Result with a suspected flake has quarantine data."""
        record = ModelFlakeRecord(
            check_code="CHECK-TEST-001",
            first_passed=False,
            rerun_passed=True,
            is_flake_suspected=True,
            rerun_count=1,
        )
        result = ModelFlakeDetectionResult(
            records=(record,),
            has_suspected_flakes=True,
            quarantine_check_codes=("CHECK-TEST-001",),
        )
        assert result.has_suspected_flakes is True
        assert len(result.quarantine_reasons) == 1
        assert "CHECK-TEST-001" in result.quarantine_reasons[0]

    def test_result_without_flake(self) -> None:
        """Result without flakes has empty quarantine data."""
        record = ModelFlakeRecord(
            check_code="CHECK-TEST-001",
            first_passed=False,
            rerun_passed=False,
            is_flake_suspected=False,
            rerun_count=1,
        )
        result = ModelFlakeDetectionResult(
            records=(record,),
            has_suspected_flakes=False,
            quarantine_check_codes=(),
        )
        assert result.quarantine_reasons == ()


# ============================================================================
# FlakeDetector
# ============================================================================


class TestFlakeDetector:
    """Tests for FlakeDetector rerun-once logic."""

    def test_should_rerun_failed_check(self) -> None:
        """Failed required check should be rerun."""
        detector = FlakeDetector()
        result = _make_check_result(passed=False)
        assert detector.should_rerun(result) is True

    def test_should_not_rerun_passed_check(self) -> None:
        """Passed check should not be rerun."""
        detector = FlakeDetector()
        result = _make_check_result(passed=True)
        assert detector.should_rerun(result) is False

    def test_should_not_rerun_skipped_check(self) -> None:
        """Skipped check should not be rerun."""
        detector = FlakeDetector()
        result = _make_check_result(passed=False, skipped=True)
        assert detector.should_rerun(result) is False

    def test_should_not_rerun_informational_check(self) -> None:
        """Informational check should not be rerun."""
        detector = FlakeDetector()
        result = _make_check_result(
            passed=False, severity=EnumCheckSeverity.INFORMATIONAL
        )
        assert detector.should_rerun(result) is False

    def test_rerun_once_limit(self) -> None:
        """Check is only rerun once (rerun-once rule)."""
        detector = FlakeDetector(max_reruns_per_check=1)
        first = _make_check_result(check_code="C1", passed=False)
        rerun = _make_check_result(check_code="C1", passed=True)

        # First rerun should be allowed
        assert detector.should_rerun(first) is True
        detector.record_rerun(first, rerun)

        # Second rerun should be blocked
        assert detector.should_rerun(first) is False

    def test_record_first_run(self) -> None:
        """record_first_run stores the initial result."""
        detector = FlakeDetector()
        result = _make_check_result(check_code="C1", passed=False)
        detector.record_first_run(result)

        detection = detector.get_result()
        assert len(detection.records) == 1
        assert detection.records[0].check_code == "C1"
        assert detection.records[0].first_passed is False

    def test_record_rerun_flake_suspected(self) -> None:
        """Rerun with different result suspects a flake."""
        detector = FlakeDetector()
        first = _make_check_result(check_code="C1", passed=False)
        rerun = _make_check_result(check_code="C1", passed=True)

        record = detector.record_rerun(first, rerun)
        assert record.is_flake_suspected is True
        assert record.first_passed is False
        assert record.rerun_passed is True

    def test_record_rerun_consistent_failure(self) -> None:
        """Rerun with same failure result does not suspect a flake."""
        detector = FlakeDetector()
        first = _make_check_result(check_code="C1", passed=False)
        rerun = _make_check_result(check_code="C1", passed=False)

        record = detector.record_rerun(first, rerun)
        assert record.is_flake_suspected is False

    def test_get_result_aggregate(self) -> None:
        """get_result aggregates all recorded flake data."""
        detector = FlakeDetector()

        # Record a flake
        first_fail = _make_check_result(check_code="C1", passed=False)
        rerun_pass = _make_check_result(check_code="C1", passed=True)
        detector.record_rerun(first_fail, rerun_pass)

        # Record a consistent failure
        first_fail2 = _make_check_result(check_code="C2", passed=False)
        rerun_fail2 = _make_check_result(check_code="C2", passed=False)
        detector.record_rerun(first_fail2, rerun_fail2)

        result = detector.get_result()
        assert result.has_suspected_flakes is True
        assert "C1" in result.quarantine_check_codes
        assert "C2" not in result.quarantine_check_codes

    def test_multiple_checks_independent(self) -> None:
        """Different checks maintain independent rerun counts."""
        detector = FlakeDetector(max_reruns_per_check=1)

        c1 = _make_check_result(check_code="C1", passed=False)
        c2 = _make_check_result(check_code="C2", passed=False)

        assert detector.should_rerun(c1) is True
        assert detector.should_rerun(c2) is True

        detector.record_rerun(c1, _make_check_result(check_code="C1", passed=True))

        # C1 is now exhausted, C2 still available
        assert detector.should_rerun(c1) is False
        assert detector.should_rerun(c2) is True


# ============================================================================
# Verdict Adjustment Functions
# ============================================================================


class TestShouldQuarantineVerdict:
    """Tests for should_quarantine_verdict function."""

    def test_fail_stays_fail(self) -> None:
        """FAIL verdict is never changed (even with flakes)."""
        flake_result = ModelFlakeDetectionResult(
            has_suspected_flakes=True,
            quarantine_check_codes=("C1",),
        )
        assert (
            should_quarantine_verdict(EnumValidationVerdict.FAIL, flake_result)
            == EnumValidationVerdict.FAIL
        )

    def test_pass_with_flakes_becomes_quarantine(self) -> None:
        """PASS verdict with suspected flakes becomes QUARANTINE."""
        flake_result = ModelFlakeDetectionResult(
            has_suspected_flakes=True,
            quarantine_check_codes=("C1",),
        )
        assert (
            should_quarantine_verdict(EnumValidationVerdict.PASS, flake_result)
            == EnumValidationVerdict.QUARANTINE
        )

    def test_pass_without_flakes_stays_pass(self) -> None:
        """PASS verdict without flakes stays PASS."""
        flake_result = ModelFlakeDetectionResult()
        assert (
            should_quarantine_verdict(EnumValidationVerdict.PASS, flake_result)
            == EnumValidationVerdict.PASS
        )

    def test_quarantine_with_flakes_stays_quarantine(self) -> None:
        """QUARANTINE verdict stays QUARANTINE regardless of flakes."""
        flake_result = ModelFlakeDetectionResult(
            has_suspected_flakes=True,
            quarantine_check_codes=("C1",),
        )
        assert (
            should_quarantine_verdict(EnumValidationVerdict.QUARANTINE, flake_result)
            == EnumValidationVerdict.QUARANTINE
        )


class TestIsPromotionBlocked:
    """Tests for is_promotion_blocked function."""

    def test_pass_not_blocked(self) -> None:
        """PASS verdict does not block promotion."""
        assert is_promotion_blocked(EnumValidationVerdict.PASS) is False

    def test_fail_blocked(self) -> None:
        """FAIL verdict blocks promotion."""
        assert is_promotion_blocked(EnumValidationVerdict.FAIL) is True

    def test_quarantine_blocked(self) -> None:
        """QUARANTINE verdict blocks promotion."""
        assert is_promotion_blocked(EnumValidationVerdict.QUARANTINE) is True
