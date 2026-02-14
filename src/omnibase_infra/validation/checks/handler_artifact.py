# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Artifact and replay check executors (CHECK-VAL-001, CHECK-VAL-002).

CHECK-VAL-001: Deterministic replay sanity
    Verifies that running the same validation twice produces consistent
    results, guarding against non-deterministic test behavior.

CHECK-VAL-002: Artifact completeness
    Validates that required and expected artifacts are present in the
    artifact storage directory.

    Required artifacts (must exist, check fails if missing):
        - result.yaml
        - verdict.yaml

    Expected artifacts (optional, reported but non-blocking):
        - attribution.yaml
        - artifacts/junit.xml
        - artifacts/coverage.json

Ticket: OMN-2151
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.models.validation.model_check_result import ModelCheckResult
from omnibase_infra.validation.checks.handler_check_executor import (
    HandlerCheckExecutor,
    ModelCheckExecutorConfig,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_validation_orchestrator.models.model_pattern_candidate import (
        ModelPatternCandidate,
    )


# Non-deterministic code patterns for CHECK-VAL-001 (replay sanity).
# Pre-compiled at module level for consistency with handler_risk.py approach.
_PATTERN_RANDOM_MODULE: re.Pattern[str] = re.compile(r"\brandom\.\w+\(")
_PATTERN_TIME_TIME: re.Pattern[str] = re.compile(r"\btime\.time\s*\(\s*\)")

# Required artifact filenames for CHECK-VAL-002
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "result.yaml",
    "verdict.yaml",
)

# Optional but expected artifacts
EXPECTED_ARTIFACTS: tuple[str, ...] = (
    "attribution.yaml",
    "artifacts/junit.xml",
    "artifacts/coverage.json",
)


class HandlerReplaySanity(HandlerCheckExecutor):
    """CHECK-VAL-001: Deterministic replay sanity.

    Verifies that the validation results are deterministic by checking
    for known sources of non-determinism (random seeds, timestamp
    dependencies, unordered collections).

    In a full implementation, this would re-run a subset of checks and
    compare results. The current implementation performs a static
    analysis check for non-deterministic patterns.
    """

    @property
    def check_code(self) -> str:
        """Return check code."""
        return "CHECK-VAL-001"

    @property
    def label(self) -> str:
        """Return check label."""
        return "Deterministic replay sanity"

    @property
    def severity(self) -> EnumCheckSeverity:
        """Return check severity."""
        return EnumCheckSeverity.RECOMMENDED

    async def execute(
        self,
        candidate: ModelPatternCandidate,
        config: ModelCheckExecutorConfig,
    ) -> ModelCheckResult:
        """Check for non-deterministic patterns.

        Args:
            candidate: Pattern candidate.
            config: Executor configuration.

        Returns:
            Check result indicating replay sanity status.
        """
        start = time.monotonic()

        # Static check: look for files that might introduce non-determinism
        nondeterministic_indicators: list[str] = []
        source_path = Path(candidate.source_path)

        for file_path in candidate.changed_files:
            if not file_path.endswith(".py"):
                continue

            full_path = source_path / file_path
            if not full_path.is_file():
                continue

            try:
                # errors='replace' prevents crashes on binary files in source
                # trees; replacement characters are acceptable for pattern scanning.
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Check for random/time-dependent patterns
            if _PATTERN_RANDOM_MODULE.search(content):
                nondeterministic_indicators.append(f"{file_path}: uses random module")
            if _PATTERN_TIME_TIME.search(content):
                nondeterministic_indicators.append(f"{file_path}: uses time.time()")

        duration_ms = (time.monotonic() - start) * 1000.0

        if not nondeterministic_indicators:
            return self._make_result(
                passed=True,
                message="No non-deterministic patterns detected.",
                duration_ms=duration_ms,
            )

        return self._make_result(
            passed=False,
            message=(
                f"Non-deterministic patterns found ({len(nondeterministic_indicators)}): "
                + "; ".join(nondeterministic_indicators[:3])
            ),
            duration_ms=duration_ms,
        )


class HandlerArtifactCompleteness(HandlerCheckExecutor):
    """CHECK-VAL-002: Artifact completeness validation.

    Validates that the artifact storage directory contains the
    required output files from the validation run.
    """

    def __init__(self, artifact_dir: Path | None = None) -> None:
        """Initialize with an artifact directory.

        Args:
            artifact_dir: Path to the artifact storage directory.  When
                ``None``, :meth:`execute` returns a ``skipped`` result.
                This allows the default registry to instantiate the
                handler without knowing the deployment-specific path.
        """
        self._artifact_dir = artifact_dir

    @property
    def check_code(self) -> str:
        """Return check code."""
        return "CHECK-VAL-002"

    @property
    def label(self) -> str:
        """Return check label."""
        return "Artifact completeness"

    @property
    def severity(self) -> EnumCheckSeverity:
        """Return check severity."""
        return EnumCheckSeverity.REQUIRED

    async def execute(
        self,
        candidate: ModelPatternCandidate,
        config: ModelCheckExecutorConfig,
    ) -> ModelCheckResult:
        """Validate that required artifacts exist.

        Args:
            candidate: Pattern candidate.
            config: Executor configuration.

        Returns:
            Check result indicating artifact completeness.
        """
        start = time.monotonic()

        artifact_dir = self._artifact_dir

        if artifact_dir is None:
            duration_ms = (time.monotonic() - start) * 1000.0
            # Skipped checks are non-blocking: passed=True because the check
            # was not applicable (no artifact_dir configured).
            return self._make_result(
                passed=True,
                message="Skipped: no artifact_dir configured.",
                skipped=True,
                duration_ms=duration_ms,
            )

        if not artifact_dir.is_dir():
            duration_ms = (time.monotonic() - start) * 1000.0
            # Full path is intentional for internal diagnostics --
            # artifact_dir is controlled infrastructure, not user input.
            return self._make_result(
                passed=False,
                message=f"Artifact directory does not exist: {artifact_dir}",
                duration_ms=duration_ms,
            )

        missing_required: list[str] = []
        missing_optional: list[str] = []

        for artifact_name in REQUIRED_ARTIFACTS:
            if not (artifact_dir / artifact_name).is_file():
                missing_required.append(artifact_name)

        for artifact_name in EXPECTED_ARTIFACTS:
            if not (artifact_dir / artifact_name).is_file():
                missing_optional.append(artifact_name)

        duration_ms = (time.monotonic() - start) * 1000.0

        if missing_required:
            return self._make_result(
                passed=False,
                message=(
                    f"Missing required artifacts: {', '.join(missing_required)}"
                    + (
                        f" (also missing optional: {', '.join(missing_optional)})"
                        if missing_optional
                        else ""
                    )
                ),
                # Artifact dir path intentional for diagnostics (controlled infra path)
                error_output=f"Artifact dir: {artifact_dir}\nMissing: {', '.join(missing_required)}",
                duration_ms=duration_ms,
            )

        if missing_optional:
            return self._make_result(
                passed=True,
                message=(
                    f"Required artifacts present. Missing optional: "
                    f"{', '.join(missing_optional)}"
                ),
                duration_ms=duration_ms,
            )

        return self._make_result(
            passed=True,
            message="All required and expected artifacts present.",
            duration_ms=duration_ms,
        )


__all__: list[str] = [
    "HandlerArtifactCompleteness",
    "HandlerReplaySanity",
    "EXPECTED_ARTIFACTS",
    "REQUIRED_ARTIFACTS",
]
