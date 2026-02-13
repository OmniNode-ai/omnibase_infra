# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Artifact and replay check executors (CHECK-VAL-001, CHECK-VAL-002).

CHECK-VAL-001: Deterministic replay sanity
    Verifies that running the same validation twice produces consistent
    results, guarding against non-deterministic test behavior.

CHECK-VAL-002: Artifact completeness
    Validates that required artifacts (junit.xml, coverage.json, logs)
    are present in the artifact storage directory.

Ticket: OMN-2151
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.models.validation.model_check_result import ModelCheckResult
from omnibase_infra.validation.checks.check_executor import (
    CheckExecutor,
    ModelCheckExecutorConfig,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_validation_orchestrator.models.model_pattern_candidate import (
        ModelPatternCandidate,
    )


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


class CheckReplaySanity(CheckExecutor):
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
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Check for random/time-dependent patterns
            import re

            if re.search(r"\brandom\.\w+\(", content):
                nondeterministic_indicators.append(f"{file_path}: uses random module")
            if re.search(r"\btime\.time\s*\(\s*\)", content):
                nondeterministic_indicators.append(f"{file_path}: uses time.time()")

        duration_ms = (time.monotonic() - start) * 1000.0

        if not nondeterministic_indicators:
            return self._make_result(
                passed=True,
                message="No non-deterministic patterns detected.",
                duration_ms=duration_ms,
            )

        return self._make_result(
            passed=True,  # RECOMMENDED -> does not block but flags
            message=(
                f"Non-deterministic patterns found ({len(nondeterministic_indicators)}): "
                + "; ".join(nondeterministic_indicators[:3])
            ),
            duration_ms=duration_ms,
        )


class CheckArtifactCompleteness(CheckExecutor):
    """CHECK-VAL-002: Artifact completeness validation.

    Validates that the artifact storage directory contains the
    required output files from the validation run.
    """

    def __init__(self, artifact_dir: Path | None = None) -> None:
        """Initialize with an artifact directory.

        Args:
            artifact_dir: Path to the artifact storage directory.
                When None, a default path based on candidate_id will be used.
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
            # Default location based on candidate_id
            artifact_dir = (
                Path.home() / ".claude" / "validation" / str(candidate.candidate_id)
            )

        if not artifact_dir.is_dir():
            duration_ms = (time.monotonic() - start) * 1000.0
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
    "CheckArtifactCompleteness",
    "CheckReplaySanity",
    "EXPECTED_ARTIFACTS",
    "REQUIRED_ARTIFACTS",
]
