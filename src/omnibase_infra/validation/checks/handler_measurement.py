# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Measurement check executors (CHECK-COST-001, CHECK-TIME-001).

CHECK-COST-001: Token delta vs baseline
    Measures token usage delta compared to a baseline. Informational
    only -- does not block the verdict.

CHECK-TIME-001: Wall-clock delta vs baseline
    Measures wall-clock execution time delta compared to a baseline.
    Informational only -- does not block the verdict.

These checks are always INFORMATIONAL and never cause a FAIL verdict.
They provide data for trend analysis and cost tracking.

Ticket: OMN-2151
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.models.validation.model_check_result import ModelCheckResult
from omnibase_infra.validation.checks.handler_check_executor import (
    HandlerCheckExecutor,
    ModelCheckExecutorConfig,
)

# Average estimated tokens per changed file. Used as a rough heuristic
# until real LLM token tracking is integrated (CHECK-COST-001).
_ESTIMATED_TOKENS_PER_FILE = 150

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_validation_orchestrator.models.model_pattern_candidate import (
        ModelPatternCandidate,
    )


class HandlerCostTokenDelta(HandlerCheckExecutor):
    """CHECK-COST-001: Token delta vs baseline.

    Measures token consumption and compares against a stored baseline.
    Always passes (informational) but records the delta for trend analysis.
    """

    def __init__(self, baseline_tokens: int = 0) -> None:
        """Initialize with a token baseline.

        Args:
            baseline_tokens: Baseline token count for comparison.
                Zero means no baseline available.
        """
        self._baseline_tokens = baseline_tokens

    @property
    def check_code(self) -> str:
        """Return check code."""
        return "CHECK-COST-001"

    @property
    def label(self) -> str:
        """Return check label."""
        return "Token delta vs baseline"

    @property
    def severity(self) -> EnumCheckSeverity:
        """Return check severity."""
        return EnumCheckSeverity.INFORMATIONAL

    async def execute(
        self,
        candidate: ModelPatternCandidate,
        config: ModelCheckExecutorConfig,
    ) -> ModelCheckResult:
        """Measure token usage delta.

        In the current implementation, this is a placeholder that
        estimates token cost based on the number of changed files
        (each file is assumed to consume ``_ESTIMATED_TOKENS_PER_FILE``
        tokens). A future version will integrate with actual LLM token
        tracking.

        Args:
            candidate: Pattern candidate to measure.
            config: Executor configuration.

        Returns:
            Informational check result with cost data.
        """
        start = time.monotonic()

        # Estimate token cost from file count (placeholder heuristic)
        estimated_tokens = len(candidate.changed_files) * _ESTIMATED_TOKENS_PER_FILE

        if self._baseline_tokens > 0:
            delta = estimated_tokens - self._baseline_tokens
            delta_pct = (delta / self._baseline_tokens) * 100
            message = (
                f"Estimated tokens: {estimated_tokens} "
                f"(baseline: {self._baseline_tokens}, "
                f"delta: {delta:+d}, {delta_pct:+.1f}%)"
            )
        else:
            message = f"Estimated tokens: {estimated_tokens} (no baseline available)"

        duration_ms = (time.monotonic() - start) * 1000.0

        return self._make_result(
            passed=True,
            message=message,
            duration_ms=duration_ms,
        )


class HandlerTimeWallClockDelta(HandlerCheckExecutor):
    """CHECK-TIME-001: Wall-clock delta vs baseline.

    Measures the total validation wall-clock time and compares against
    a stored baseline. Always passes (informational).
    """

    def __init__(
        self,
        baseline_ms: float = 0.0,
        wall_clock_seconds: float | None = None,
    ) -> None:
        """Initialize with a wall-clock baseline.

        Args:
            baseline_ms: Baseline duration in milliseconds.
                Zero means no baseline available.
            wall_clock_seconds: Pre-measured wall-clock duration in seconds.
                When provided, this value is used instead of the near-zero
                self-measurement. Typically injected by the orchestrator
                after the full validation run completes.
        """
        self._baseline_ms = baseline_ms
        self._wall_clock_seconds = wall_clock_seconds

    @property
    def check_code(self) -> str:
        """Return check code."""
        return "CHECK-TIME-001"

    @property
    def label(self) -> str:
        """Return check label."""
        return "Wall-clock delta vs baseline"

    @property
    def severity(self) -> EnumCheckSeverity:
        """Return check severity."""
        return EnumCheckSeverity.INFORMATIONAL

    async def execute(
        self,
        candidate: ModelPatternCandidate,
        config: ModelCheckExecutorConfig,
    ) -> ModelCheckResult:
        """Record wall-clock delta.

        When ``wall_clock_seconds`` was provided at construction time,
        the check reports that value as the true elapsed duration.
        Otherwise it falls back to a near-zero self-measurement.

        Args:
            candidate: Pattern candidate.
            config: Executor configuration.

        Returns:
            Informational check result with timing data.
        """
        if self._wall_clock_seconds is not None:
            elapsed_ms = self._wall_clock_seconds * 1000.0
        else:
            start = time.monotonic()
            elapsed_ms = (time.monotonic() - start) * 1000.0

        if self._baseline_ms > 0:
            delta_ms = elapsed_ms - self._baseline_ms
            delta_pct = (delta_ms / self._baseline_ms) * 100 if self._baseline_ms else 0
            message = (
                f"Wall-clock: {elapsed_ms:.0f}ms "
                f"(baseline: {self._baseline_ms:.0f}ms, "
                f"delta: {delta_ms:+.0f}ms, {delta_pct:+.1f}%)"
            )
        else:
            message = f"Wall-clock: {elapsed_ms:.0f}ms (no baseline available)"

        return self._make_result(
            passed=True,
            message=message,
            duration_ms=elapsed_ms,
        )


__all__: list[str] = [
    "HandlerCostTokenDelta",
    "HandlerTimeWallClockDelta",
]
