# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Check executor registry mapping check codes to executor instances.

Provides a centralized registry of all check executors in the catalog
and a factory function to retrieve executors by check code. The catalog is
declared in ``check_catalog.yaml``; this module builds one executor per entry.

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

from pathlib import Path
from types import MappingProxyType

from omnibase_infra.models.validation.model_check_catalog_entry import (
    ModelCheckCatalogEntry,
)
from omnibase_infra.validation.checks.handler_check_executor import (
    HandlerCheckExecutor,
)


def _build_registry() -> dict[str, HandlerCheckExecutor]:
    """Build the default check executor registry from the declared catalog.

    Returns:
        Mapping from check code to executor instance.
    """
    return {
        entry.check_code: entry.build_executor()
        for entry in ModelCheckCatalogEntry.load_catalog()
    }


# Module-level singleton registry (read-only view).
# The mutable dict is scoped inside the MappingProxyType call to prevent
# accidental mutation via module-level access.
CHECK_REGISTRY: MappingProxyType[str, HandlerCheckExecutor] = MappingProxyType(
    _build_registry()
)

# Ordered check codes matching the catalog order
CHECK_CATALOG_ORDER: tuple[str, ...] = tuple(
    entry.check_code for entry in ModelCheckCatalogEntry.load_catalog()
)


def get_check_executor(
    check_code: str,
    *,
    artifact_dir: Path | None = None,
) -> HandlerCheckExecutor | None:
    """Retrieve a check executor by its code.

    Args:
        check_code: Check identifier (e.g., ``CHECK-PY-001``).
        artifact_dir: Optional artifact storage directory.  When provided
            and *check_code* is ``CHECK-VAL-002``, the returned executor
            is configured with this directory so the artifact-completeness
            check can actually run instead of returning ``skipped=True``.
            For all other check codes this parameter is ignored.

    Returns:
        The executor instance, or ``None`` if no executor is registered
        for the given check code.
    """
    if check_code == "CHECK-VAL-002" and artifact_dir is not None:
        entry = next(
            e
            for e in ModelCheckCatalogEntry.load_catalog()
            if e.check_code == check_code
        )
        return entry.build_executor(artifact_dir=artifact_dir)
    return CHECK_REGISTRY.get(check_code)


__all__: list[str] = [
    "CHECK_CATALOG_ORDER",
    "CHECK_REGISTRY",
    "get_check_executor",
]
