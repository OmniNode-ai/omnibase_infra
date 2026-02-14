# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Check executor registry mapping check codes to executor instances.

Provides a centralized registry of all check executors in the catalog
and a factory function to retrieve executors by check code.

Check Catalog (12 checks):
    CHECK-PY-001    Typecheck (mypy)                     SubprocessCheckExecutor
    CHECK-PY-002    Lint/format (ruff)                   SubprocessCheckExecutor
    CHECK-TEST-001  Unit tests (fast)                    SubprocessCheckExecutor
    CHECK-TEST-002  Targeted integration tests           SubprocessCheckExecutor
    CHECK-VAL-001   Deterministic replay sanity          CheckReplaySanity
    CHECK-VAL-002   Artifact completeness                CheckArtifactCompleteness
    CHECK-RISK-001  Sensitive paths -> stricter bar      CheckRiskSensitivePaths
    CHECK-RISK-002  Diff size threshold                  CheckRiskDiffSize
    CHECK-RISK-003  Unsafe operations detector           CheckRiskUnsafeOperations
    CHECK-OUT-001   CI equivalent pass rate              SubprocessCheckExecutor
    CHECK-COST-001  Token delta vs baseline              CheckCostTokenDelta
    CHECK-TIME-001  Wall-clock delta vs baseline         CheckTimeWallClockDelta

Ticket: OMN-2151
"""

from __future__ import annotations

from types import MappingProxyType

from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.validation.checks.check_artifact import (
    CheckArtifactCompleteness,
    CheckReplaySanity,
)
from omnibase_infra.validation.checks.check_executor import (
    CheckExecutor,
    SubprocessCheckExecutor,
)
from omnibase_infra.validation.checks.check_measurement import (
    CheckCostTokenDelta,
    CheckTimeWallClockDelta,
)
from omnibase_infra.validation.checks.check_risk import (
    CheckRiskDiffSize,
    CheckRiskSensitivePaths,
    CheckRiskUnsafeOperations,
)


def _build_registry() -> dict[str, CheckExecutor]:
    """Build the default check executor registry.

    Returns:
        Mapping from check code to executor instance.
    """
    return {
        # --- Subprocess checks (mypy, ruff, pytest, CI) ---
        "CHECK-PY-001": SubprocessCheckExecutor(
            check_code="CHECK-PY-001",
            label="Typecheck (mypy)",
            severity=EnumCheckSeverity.REQUIRED,
            command="poetry run mypy src/",
        ),
        "CHECK-PY-002": SubprocessCheckExecutor(
            check_code="CHECK-PY-002",
            label="Lint/format (ruff)",
            severity=EnumCheckSeverity.REQUIRED,
            command="poetry run ruff check src/ tests/",
        ),
        "CHECK-TEST-001": SubprocessCheckExecutor(
            check_code="CHECK-TEST-001",
            label="Unit tests (fast)",
            severity=EnumCheckSeverity.REQUIRED,
            command="poetry run pytest tests/ -m unit --timeout=60",
        ),
        "CHECK-TEST-002": SubprocessCheckExecutor(
            check_code="CHECK-TEST-002",
            label="Targeted integration tests",
            severity=EnumCheckSeverity.RECOMMENDED,
            command="poetry run pytest tests/ -m integration --timeout=120",
        ),
        "CHECK-OUT-001": SubprocessCheckExecutor(
            check_code="CHECK-OUT-001",
            label="CI equivalent pass rate",
            severity=EnumCheckSeverity.REQUIRED,
            command="poetry run pytest tests/ --timeout=180",
        ),
        # --- Analysis checks (no subprocess, inspect candidate) ---
        "CHECK-VAL-001": CheckReplaySanity(),
        "CHECK-VAL-002": CheckArtifactCompleteness(),
        "CHECK-RISK-001": CheckRiskSensitivePaths(),
        "CHECK-RISK-002": CheckRiskDiffSize(),
        "CHECK-RISK-003": CheckRiskUnsafeOperations(),
        # --- Measurement checks (informational) ---
        "CHECK-COST-001": CheckCostTokenDelta(),
        "CHECK-TIME-001": CheckTimeWallClockDelta(),
    }


# Module-level singleton registry (read-only view)
_CHECK_REGISTRY: dict[str, CheckExecutor] = _build_registry()
CHECK_REGISTRY: MappingProxyType[str, CheckExecutor] = MappingProxyType(_CHECK_REGISTRY)

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


def get_check_executor(check_code: str) -> CheckExecutor | None:
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
