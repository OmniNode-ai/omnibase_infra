# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Risk-gating check executors (CHECK-RISK-001 through CHECK-RISK-003).

CHECK-RISK-001: Sensitive paths -> stricter bar
    Detects changes to security-sensitive paths and enforces stricter
    validation requirements when those paths are modified.

CHECK-RISK-002: Diff size threshold
    Flags candidates with diffs exceeding a configurable threshold
    (default: 500 changed lines) for additional review.

CHECK-RISK-003: Unsafe operations detector
    Scans changed files for dangerous patterns like eval(), exec(),
    subprocess with shell=True, pickle.loads(), etc.

Ticket: OMN-2151
"""

from __future__ import annotations

import re
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


# Paths considered security-sensitive
SENSITIVE_PATH_PATTERNS: tuple[str, ...] = (
    r".*/(auth|security|crypto|secrets|vault|credentials)/.*",
    r".*/\.env.*",
    r".*/(password|token|key|cert).*\.py$",
    r".*/migrations/.*",
    r".*/docker-compose.*\.ya?ml$",
    r".*/Dockerfile.*",
    r".*/(config|settings)\.py$",
)

# Unsafe operation patterns in Python source
UNSAFE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\beval\s*\(", "eval() call detected"),
    (r"\bexec\s*\(", "exec() call detected"),
    (r"subprocess\.\w+\([\s\S]*?shell\s*=\s*True", "subprocess with shell=True"),
    (r"\bpickle\.loads?\s*\(", "pickle.load/loads() call detected"),
    (r"\b__import__\s*\(", "__import__() call detected"),
    (r"\bos\.system\s*\(", "os.system() call detected"),
    (r"\bcompile\s*\(.*\bexec\b", "compile() with exec mode"),
    (r"\byaml\.load\s*\((?!.*Loader)", "yaml.load() without explicit Loader"),
)

# Default diff size threshold (number of changed files)
DEFAULT_DIFF_SIZE_THRESHOLD: int = 500


class CheckRiskSensitivePaths(CheckExecutor):
    """CHECK-RISK-001: Detect changes to security-sensitive paths.

    When sensitive paths are modified, this check flags them so that
    downstream checks can enforce a stricter validation bar.
    """

    @property
    def check_code(self) -> str:
        """Return check code."""
        return "CHECK-RISK-001"

    @property
    def label(self) -> str:
        """Return check label."""
        return "Sensitive paths -> stricter bar"

    @property
    def severity(self) -> EnumCheckSeverity:
        """Return check severity."""
        return EnumCheckSeverity.REQUIRED

    async def execute(
        self,
        candidate: ModelPatternCandidate,
        config: ModelCheckExecutorConfig,
    ) -> ModelCheckResult:
        """Check if any changed files match sensitive path patterns.

        This check passes if no sensitive paths are detected, or if
        sensitive paths are detected and the candidate has appropriate
        risk tags. It fails only if sensitive paths are found without
        corresponding risk acknowledgment.

        Args:
            candidate: Pattern candidate with changed_files list.
            config: Executor configuration.

        Returns:
            Check result indicating sensitive path detection status.
        """
        start = time.monotonic()

        compiled_patterns = [re.compile(p) for p in SENSITIVE_PATH_PATTERNS]
        sensitive_hits: list[str] = []

        for file_path in candidate.changed_files:
            for pattern in compiled_patterns:
                if pattern.match(file_path):
                    sensitive_hits.append(file_path)
                    break

        duration_ms = (time.monotonic() - start) * 1000.0

        if not sensitive_hits:
            return self._make_result(
                passed=True,
                message="No sensitive paths detected in changed files.",
                duration_ms=duration_ms,
            )

        # Check if risk tags acknowledge the sensitive changes
        has_security_tag = any(
            tag in ("security", "auth", "credentials", "infrastructure")
            for tag in candidate.risk_tags
        )

        if has_security_tag:
            return self._make_result(
                passed=True,
                message=(
                    f"Sensitive paths detected ({len(sensitive_hits)} files) "
                    f"with appropriate risk tags: {', '.join(candidate.risk_tags)}"
                ),
                duration_ms=duration_ms,
            )

        return self._make_result(
            passed=False,
            message=(
                f"Sensitive paths detected without risk acknowledgment: "
                f"{', '.join(sensitive_hits[:5])}"
                + (
                    f" (+{len(sensitive_hits) - 5} more)"
                    if len(sensitive_hits) > 5
                    else ""
                )
            ),
            error_output="\n".join(sensitive_hits),
            duration_ms=duration_ms,
        )


class CheckRiskDiffSize(CheckExecutor):
    """CHECK-RISK-002: Diff size threshold check.

    Flags candidates with an excessive number of changed files,
    which correlates with higher risk of introducing regressions.
    """

    def __init__(self, threshold: int = DEFAULT_DIFF_SIZE_THRESHOLD) -> None:
        """Initialize with configurable threshold.

        Args:
            threshold: Maximum number of changed files before flagging.
        """
        self._threshold = threshold

    @property
    def check_code(self) -> str:
        """Return check code."""
        return "CHECK-RISK-002"

    @property
    def label(self) -> str:
        """Return check label."""
        return "Diff size threshold"

    @property
    def severity(self) -> EnumCheckSeverity:
        """Return check severity."""
        return EnumCheckSeverity.RECOMMENDED

    async def execute(
        self,
        candidate: ModelPatternCandidate,
        config: ModelCheckExecutorConfig,
    ) -> ModelCheckResult:
        """Check if the diff exceeds the configured threshold.

        Args:
            candidate: Pattern candidate with changed_files list.
            config: Executor configuration.

        Returns:
            Check result indicating whether the diff size is acceptable.
        """
        start = time.monotonic()

        file_count = len(candidate.changed_files)
        duration_ms = (time.monotonic() - start) * 1000.0

        if file_count <= self._threshold:
            return self._make_result(
                passed=True,
                message=f"Diff size ({file_count} files) within threshold ({self._threshold}).",
                duration_ms=duration_ms,
            )

        return self._make_result(
            passed=False,
            message=(
                f"Diff size ({file_count} files) exceeds threshold "
                f"({self._threshold}). Consider splitting the change."
            ),
            duration_ms=duration_ms,
        )


class CheckRiskUnsafeOperations(CheckExecutor):
    """CHECK-RISK-003: Unsafe operations detector.

    Scans changed Python files for dangerous patterns like eval(),
    exec(), subprocess with shell=True, pickle.loads(), etc.
    """

    @property
    def check_code(self) -> str:
        """Return check code."""
        return "CHECK-RISK-003"

    @property
    def label(self) -> str:
        """Return check label."""
        return "Unsafe operations detector"

    @property
    def severity(self) -> EnumCheckSeverity:
        """Return check severity."""
        return EnumCheckSeverity.REQUIRED

    async def execute(
        self,
        candidate: ModelPatternCandidate,
        config: ModelCheckExecutorConfig,
    ) -> ModelCheckResult:
        """Scan changed files for unsafe operation patterns.

        Args:
            candidate: Pattern candidate with changed_files and source_path.
            config: Executor configuration.

        Returns:
            Check result indicating whether unsafe operations were detected.
        """
        start = time.monotonic()

        compiled_patterns = [(re.compile(p), desc) for p, desc in UNSAFE_PATTERNS]
        violations: list[str] = []

        python_files = [f for f in candidate.changed_files if f.endswith(".py")]

        for file_path in python_files:
            full_path = Path(candidate.source_path) / file_path
            if not full_path.is_file():
                continue
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for pattern, description in compiled_patterns:
                matches = pattern.findall(content)
                if matches:
                    violations.append(
                        f"{file_path}: {description} ({len(matches)} occurrence(s))"
                    )

        duration_ms = (time.monotonic() - start) * 1000.0

        if not violations:
            return self._make_result(
                passed=True,
                message=f"No unsafe operations detected in {len(python_files)} Python files.",
                duration_ms=duration_ms,
            )

        return self._make_result(
            passed=False,
            message=(
                f"Unsafe operations detected ({len(violations)} issue(s)): "
                + "; ".join(violations[:3])
                + (f" (+{len(violations) - 3} more)" if len(violations) > 3 else "")
            ),
            error_output="\n".join(violations),
            duration_ms=duration_ms,
        )


__all__: list[str] = [
    "CheckRiskDiffSize",
    "CheckRiskSensitivePaths",
    "CheckRiskUnsafeOperations",
    "DEFAULT_DIFF_SIZE_THRESHOLD",
    "SENSITIVE_PATH_PATTERNS",
    "UNSAFE_PATTERNS",
]
