# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Risk-gating check executors (CHECK-RISK-001 through CHECK-RISK-003).

CHECK-RISK-001: Sensitive paths -> stricter bar
    Detects changes to security-sensitive paths and enforces stricter
    validation requirements when those paths are modified.

CHECK-RISK-002: Diff size threshold
    Flags candidates whose number of changed files exceeds a configurable
    threshold (default: 500 changed files) for additional review.

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
from omnibase_infra.validation.checks.handler_check_executor import (
    HandlerCheckExecutor,
    ModelCheckExecutorConfig,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_validation_orchestrator.models.model_pattern_candidate import (
        ModelPatternCandidate,
    )


# Paths considered security-sensitive.
# Patterns use ``(?:.*/)?`` so they match both root-level files (e.g. ".env")
# and nested paths (e.g. "deploy/.env.production") when used with ``re.match``.
# Pre-compiled at module level to avoid per-invocation overhead.
SENSITIVE_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:.*/)?(?:auth|security|crypto|secrets|vault|credentials)/.*"),
    re.compile(r"(?:.*/)?\.env.*"),
    re.compile(r"(?:.*/)?(?:password|token|key|cert).*\.py$"),
    re.compile(r"(?:.*/)?migrations/.*"),
    re.compile(r"(?:.*/)?docker-compose.*\.ya?ml$"),
    re.compile(r"(?:.*/)?Dockerfile.*"),
    re.compile(r"(?:.*/)?(?:config|settings)\.py$"),
)

# Unsafe operation patterns in Python source.
# Pre-compiled at module level to avoid per-invocation overhead.
UNSAFE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\beval\s*\("), "eval() call detected"),
    (re.compile(r"\bexec\s*\("), "exec() call detected"),
    # The {0,200} bound is a heuristic to keep regex backtracking bounded.
    # It may miss shell=True when the argument list exceeds 200 characters
    # or uses nested parentheses that cause the [^)] class to stop early.
    (
        re.compile(r"subprocess\.\w+\([^)]{0,200}shell\s*=\s*True"),
        "subprocess with shell=True",
    ),
    (re.compile(r"\bpickle\.loads?\s*\("), "pickle.load/loads() call detected"),
    (re.compile(r"\b__import__\s*\("), "__import__() call detected"),
    (re.compile(r"\bos\.system\s*\("), "os.system() call detected"),
    (re.compile(r"\bcompile\s*\([^)]{0,200}\bexec\b"), "compile() with exec mode"),
    (
        re.compile(r"\byaml\.load\s*\((?![^)]{0,200}Loader)"),
        "yaml.load() without explicit Loader",
    ),
)

# Default diff size threshold (number of changed files)
DEFAULT_DIFF_SIZE_THRESHOLD: int = 500


class HandlerRiskSensitivePaths(HandlerCheckExecutor):
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

        sensitive_hits: list[str] = []

        for file_path in candidate.changed_files:
            for pattern in SENSITIVE_PATH_PATTERNS:
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


class HandlerRiskDiffSize(HandlerCheckExecutor):
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


class HandlerRiskUnsafeOperations(HandlerCheckExecutor):
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

            for pattern, description in UNSAFE_PATTERNS:
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
    "HandlerRiskDiffSize",
    "HandlerRiskSensitivePaths",
    "HandlerRiskUnsafeOperations",
    "DEFAULT_DIFF_SIZE_THRESHOLD",
    "SENSITIVE_PATH_PATTERNS",
    "UNSAFE_PATTERNS",
]
