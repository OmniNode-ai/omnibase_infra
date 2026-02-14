# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Flake detection for the validation pipeline.

Implements the rerun-once rule for non-deterministic failures:

    1. When a check fails, the flake detector offers to rerun it once.
    2. If the rerun passes, the failure is classified as a suspected flake.
    3. Suspected flakes produce a QUARANTINE verdict (not PASS or FAIL).
    4. Pattern promotion is blocked while a candidate is in QUARANTINE.

Flake Detection Protocol:
    - First failure: ``is_flake_suspected = False``
    - Rerun result differs from first run: ``is_flake_suspected = True``
    - Both runs fail identically: ``is_flake_suspected = False``
    - Both runs pass: Not a flake (normal PASS)

The detector tracks rerun history per check code to prevent
infinite rerun loops. Each check gets at most one rerun opportunity.

Usage:
    detector = FlakeDetector()
    first_result = await executor.run(check)
    detector.record_first_run(first_result)
    if not first_result.passed:
        should_rerun = detector.should_rerun(first_result)
        if should_rerun:
            rerun_result = await executor.run(check)
            detector.record_rerun(first_result, rerun_result)

    detection_result = detector.get_result()

Ticket: OMN-2151
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumCheckSeverity, EnumValidationVerdict
from omnibase_infra.models.validation.model_check_result import ModelCheckResult
from omnibase_infra.models.validation.model_flake_detection_result import (
    ModelFlakeDetectionResult,
)
from omnibase_infra.models.validation.model_flake_record import ModelFlakeRecord

logger = logging.getLogger(__name__)


class FlakeDetector:
    """Detects non-deterministic (flaky) check failures.

    Implements the rerun-once rule: each failing check is rerun once.
    If the rerun result differs from the first run, the check is
    classified as a suspected flake and the candidate enters QUARANTINE.

    Note:
        Instances accumulate internal state (rerun counts and flake
        records). Create a fresh instance per validation run rather
        than reusing across runs.

    Usage:
        detector = FlakeDetector()
        first_result = await executor.run(check)
        detector.record_first_run(first_result)
        if not first_result.passed:
            should_rerun = detector.should_rerun(first_result)
            if should_rerun:
                rerun_result = await executor.run(check)
                detector.record_rerun(first_result, rerun_result)

        detection_result = detector.get_result()
    """

    def __init__(self, max_reruns_per_check: int = 1) -> None:
        """Initialize the flake detector.

        Args:
            max_reruns_per_check: Maximum number of reruns per check code.
                Default is 1 (the rerun-once rule).
        """
        self._max_reruns = max_reruns_per_check
        self._records: dict[str, ModelFlakeRecord] = {}
        self._rerun_counts: dict[str, int] = {}

    def should_rerun(self, result: ModelCheckResult) -> bool:
        """Determine if a failed check should be rerun.

        A check should be rerun if:
        - It failed (not passed, not skipped)
        - It has not exceeded the max rerun count
        - It is not INFORMATIONAL severity (no point rerunning)

        Args:
            result: The check result from the first run.

        Returns:
            True if the check should be rerun.
        """
        if result.passed or result.skipped:
            return False
        if result.severity == EnumCheckSeverity.INFORMATIONAL:
            return False

        current_reruns = self._rerun_counts.get(result.check_code, 0)
        return current_reruns < self._max_reruns

    def record_first_run(self, result: ModelCheckResult) -> None:
        """Record the first run result for a check.

        Args:
            result: The check result from the first run.
        """
        self._records[result.check_code] = ModelFlakeRecord(
            check_code=result.check_code,
            first_passed=result.passed,
            rerun_passed=None,
            is_flake_suspected=False,
            rerun_count=0,
        )

    def record_rerun(
        self,
        first_result: ModelCheckResult,
        rerun_result: ModelCheckResult,
    ) -> ModelFlakeRecord:
        """Record a rerun result and determine flake status.

        A flake is suspected when:
        - First run failed and rerun passed (most common)
        - First run passed and rerun failed (less common but possible)

        Args:
            first_result: The original check result.
            rerun_result: The rerun check result.

        Returns:
            Updated ModelFlakeRecord with flake detection status.
        """
        check_code = first_result.check_code
        rerun_count = self._rerun_counts.get(check_code, 0) + 1
        self._rerun_counts[check_code] = rerun_count

        # Flake suspected if results differ
        is_flake = first_result.passed != rerun_result.passed

        record = ModelFlakeRecord(
            check_code=check_code,
            first_passed=first_result.passed,
            rerun_passed=rerun_result.passed,
            is_flake_suspected=is_flake,
            rerun_count=rerun_count,
        )
        self._records[check_code] = record

        if is_flake:
            logger.warning(
                "Flake suspected for %s: first=%s rerun=%s",
                check_code,
                first_result.passed,
                rerun_result.passed,
            )

        return record

    def get_result(self) -> ModelFlakeDetectionResult:
        """Build the aggregate flake detection result.

        Returns:
            ModelFlakeDetectionResult with all recorded flake analysis.
        """
        records = tuple(self._records.values())
        quarantine_codes = tuple(r.check_code for r in records if r.is_flake_suspected)

        return ModelFlakeDetectionResult(
            records=records,
            has_suspected_flakes=len(quarantine_codes) > 0,
            quarantine_check_codes=quarantine_codes,
        )


def should_quarantine_verdict(
    verdict: EnumValidationVerdict,
    flake_result: ModelFlakeDetectionResult,
) -> EnumValidationVerdict:
    """Apply flake detection to adjust the verdict.

    If the verdict is PASS but flakes were suspected, downgrade
    to QUARANTINE. If the verdict is already FAIL, it stays FAIL.

    Args:
        verdict: Original verdict from the adjudicator.
        flake_result: Flake detection analysis.

    Returns:
        Adjusted verdict (possibly QUARANTINE).
    """
    if verdict == EnumValidationVerdict.FAIL:
        return EnumValidationVerdict.FAIL

    if flake_result.has_suspected_flakes:
        logger.info(
            "Verdict adjusted to QUARANTINE due to suspected flakes: %s",
            ", ".join(flake_result.quarantine_check_codes),
        )
        return EnumValidationVerdict.QUARANTINE

    return verdict


def is_promotion_blocked(verdict: EnumValidationVerdict) -> bool:
    """Check if pattern promotion is blocked by the verdict.

    Promotion is blocked for FAIL and QUARANTINE verdicts.
    Only PASS allows promotion to proceed.

    Args:
        verdict: The validation verdict.

    Returns:
        True if promotion is blocked.
    """
    return verdict != EnumValidationVerdict.PASS


__all__: list[str] = [
    "FlakeDetector",
    "is_promotion_blocked",
    "should_quarantine_verdict",
]
