# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation check implementations for the full check catalog.

Provides individual check executors for all 12 checks defined in the
MVP check catalog (OMN-2147) with actual subprocess execution and
analysis logic (OMN-2151).

Check Catalog:
    CHECK-PY-001    Typecheck (mypy)                     required
    CHECK-PY-002    Lint/format (ruff)                   required
    CHECK-TEST-001  Unit tests (fast)                    required
    CHECK-TEST-002  Targeted integration tests           recommended
    CHECK-VAL-001   Deterministic replay sanity          recommended
    CHECK-VAL-002   Artifact completeness                required
    CHECK-RISK-001  Sensitive paths -> stricter bar       required
    CHECK-RISK-002  Diff size threshold                  recommended
    CHECK-RISK-003  Unsafe operations detector           required
    CHECK-OUT-001   CI equivalent pass rate              required
    CHECK-COST-001  Token delta vs baseline              informational
    CHECK-TIME-001  Wall-clock delta vs baseline         informational
"""

from omnibase_infra.validation.checks.handler_check_executor import (
    HandlerCheckExecutor,
    ModelCheckExecutorConfig,
)
from omnibase_infra.validation.checks.registry_infra_check import (
    CHECK_REGISTRY,
    get_check_executor,
)

__all__: list[str] = [
    "CHECK_REGISTRY",
    "HandlerCheckExecutor",
    "ModelCheckExecutorConfig",
    "get_check_executor",
]
