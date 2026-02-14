# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Check executor registry mapping check codes to executor instances.

Provides a centralized registry of all check executors in the catalog
and a factory function to retrieve executors by check code.

Check Catalog (12 checks):
    CHECK-PY-001    Typecheck (mypy)                     HandlerSubprocessCheckExecutor
    CHECK-PY-002    Lint/format (ruff)                   HandlerSubprocessCheckExecutor
    CHECK-TEST-001  Unit tests (fast)                    HandlerSubprocessCheckExecutor
    CHECK-TEST-002  Targeted integration tests           HandlerSubprocessCheckExecutor
    CHECK-VAL-001   Deterministic replay sanity          HandlerReplaySanity
    CHECK-VAL-002   Artifact completeness                HandlerArtifactCompleteness
    CHECK-RISK-001  Sensitive paths -> stricter bar      HandlerRiskSensitivePaths
    CHECK-RISK-002  Diff size threshold                  HandlerRiskDiffSize
    CHECK-RISK-003  Unsafe operations detector           HandlerRiskUnsafeOperations
    CHECK-OUT-001   CI equivalent pass rate              HandlerSubprocessCheckExecutor
    CHECK-COST-001  Token delta vs baseline              HandlerCostTokenDelta
    CHECK-TIME-001  Wall-clock delta vs baseline         HandlerTimeWallClockDelta

Ticket: OMN-2151
"""

from __future__ import annotations

from types import MappingProxyType

from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.validation.checks.handler_artifact import (
    HandlerArtifactCompleteness,
    HandlerReplaySanity,
)
from omnibase_infra.validation.checks.handler_check_executor import (
    HandlerCheckExecutor,
    HandlerSubprocessCheckExecutor,
)
from omnibase_infra.validation.checks.handler_measurement import (
    HandlerCostTokenDelta,
    HandlerTimeWallClockDelta,
)
from omnibase_infra.validation.checks.handler_risk import (
    HandlerRiskDiffSize,
    HandlerRiskSensitivePaths,
    HandlerRiskUnsafeOperations,
)


def _build_registry() -> dict[str, HandlerCheckExecutor]:
    """Build the default check executor registry.

    Returns:
        Mapping from check code to executor instance.

    Note:
        CHECK-VAL-002 (artifact completeness) is instantiated without an
        ``artifact_dir``, which means it will return ``skipped=True`` at
        runtime.  This is intentional: the default registry cannot know
        the deployment-specific artifact directory.  Callers that need
        CHECK-VAL-002 to run must replace the registry entry with an
        instance configured via ``HandlerArtifactCompleteness(artifact_dir=...)``.
    """
    return {
        # --- Subprocess checks (mypy, ruff, pytest, CI) ---
        "CHECK-PY-001": HandlerSubprocessCheckExecutor(
            check_code="CHECK-PY-001",
            label="Typecheck (mypy)",
            severity=EnumCheckSeverity.REQUIRED,
            command="poetry run mypy src/",
        ),
        "CHECK-PY-002": HandlerSubprocessCheckExecutor(
            check_code="CHECK-PY-002",
            label="Lint/format (ruff)",
            severity=EnumCheckSeverity.REQUIRED,
            command="poetry run ruff check src/ tests/",
        ),
        "CHECK-TEST-001": HandlerSubprocessCheckExecutor(
            check_code="CHECK-TEST-001",
            label="Unit tests (fast)",
            severity=EnumCheckSeverity.REQUIRED,
            command="poetry run pytest tests/ -m unit --timeout=60",
        ),
        "CHECK-TEST-002": HandlerSubprocessCheckExecutor(
            check_code="CHECK-TEST-002",
            label="Targeted integration tests",
            severity=EnumCheckSeverity.RECOMMENDED,
            command="poetry run pytest tests/ -m integration --timeout=120",
        ),
        "CHECK-OUT-001": HandlerSubprocessCheckExecutor(
            check_code="CHECK-OUT-001",
            label="CI equivalent pass rate",
            severity=EnumCheckSeverity.REQUIRED,
            command="poetry run pytest tests/ --timeout=180",
        ),
        # --- Analysis checks (no subprocess, inspect candidate) ---
        "CHECK-VAL-001": HandlerReplaySanity(),
        # CHECK-VAL-002 requires artifact_dir to be provided by the caller.
        # The default instance (artifact_dir=None) returns skipped=True.
        # To enable this check, callers must supply a deployment-specific
        # artifact directory when constructing HandlerArtifactCompleteness.
        "CHECK-VAL-002": HandlerArtifactCompleteness(),
        "CHECK-RISK-001": HandlerRiskSensitivePaths(),
        "CHECK-RISK-002": HandlerRiskDiffSize(),
        "CHECK-RISK-003": HandlerRiskUnsafeOperations(),
        # --- Measurement checks (informational) ---
        "CHECK-COST-001": HandlerCostTokenDelta(),
        "CHECK-TIME-001": HandlerTimeWallClockDelta(),
    }


# Module-level singleton registry (read-only view).
# The mutable dict is scoped inside the MappingProxyType call to prevent
# accidental mutation via module-level access.
CHECK_REGISTRY: MappingProxyType[str, HandlerCheckExecutor] = MappingProxyType(
    _build_registry()
)

# Ordered check codes matching the catalog order
CHECK_CATALOG_ORDER: tuple[str, ...] = (
    "CHECK-PY-001",
    "CHECK-PY-002",
    "CHECK-TEST-001",
    "CHECK-TEST-002",
    "CHECK-VAL-001",
    "CHECK-VAL-002",
    "CHECK-RISK-001",
    "CHECK-RISK-002",
    "CHECK-RISK-003",
    "CHECK-OUT-001",
    "CHECK-COST-001",
    "CHECK-TIME-001",
)


def get_check_executor(check_code: str) -> HandlerCheckExecutor | None:
    """Retrieve a check executor by its code.

    Args:
        check_code: Check identifier (e.g., CHECK-PY-001).

    Returns:
        The executor instance, or None if no executor is registered
        for the given check code.
    """
    return CHECK_REGISTRY.get(check_code)


__all__: list[str] = [
    "CHECK_CATALOG_ORDER",
    "CHECK_REGISTRY",
    "get_check_executor",
]
