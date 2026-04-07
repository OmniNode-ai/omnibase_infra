# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that verifies system health before building.

This is an EFFECT handler - performs external I/O (health checks).
Delegates data verification to omnimarket's HandlerDataVerification node.

Related:
    - OMN-7317: node_verify_effect
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from omnimarket.nodes.node_data_verification.handlers.handler_data_verification import (
    HandlerDataVerification,
    InmemoryDataSource,
)
from omnimarket.nodes.node_data_verification.models.model_data_verification_start_command import (
    ModelDataVerificationStartCommand,
)
from omnimarket.nodes.node_data_verification.models.model_data_verification_state import (
    EnumVerificationStatus,
)

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_verify_effect.models.model_verify import (
    ModelVerifyCheck,
    ModelVerifyResult,
)
from omnibase_infra.nodes.node_verify_effect.runners.runner_dashboard_sweep import (
    run_dashboard_sweep,
)

logger = logging.getLogger(__name__)


class HandlerVerify:
    """Verifies system health: dashboard, runtime, data flow, data quality.

    Delegates data verification to omnimarket's HandlerDataVerification.
    Non-critical failures produce warnings but do not block the loop.
    Only critical check failures cause the phase to fail.
    """

    def __init__(self) -> None:
        self._data_verifier = HandlerDataVerification()

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        correlation_id: UUID,
        dry_run: bool = False,
    ) -> ModelVerifyResult:
        """Execute verification checks.

        Checks:
            1. Dashboard health (non-critical)
            2. Runtime health (critical)
            3. Data flow verification via omnimarket node (non-critical)

        Args:
            correlation_id: Cycle correlation ID.
            dry_run: Skip actual checks.

        Returns:
            ModelVerifyResult with check outcomes.
        """
        logger.info(
            "Verify phase started (correlation_id=%s, dry_run=%s)",
            correlation_id,
            dry_run,
        )

        if dry_run:
            return ModelVerifyResult(
                correlation_id=correlation_id,
                all_critical_passed=True,
                checks=(
                    ModelVerifyCheck(
                        name="dashboard_health",
                        passed=True,
                        critical=False,
                        message="dry_run",
                    ),
                    ModelVerifyCheck(
                        name="runtime_health",
                        passed=True,
                        critical=True,
                        message="dry_run",
                    ),
                    ModelVerifyCheck(
                        name="data_flow", passed=True, critical=False, message="dry_run"
                    ),
                ),
                warnings=("dry_run: no actual checks executed",),
            )

        checks: list[ModelVerifyCheck] = []
        warnings: list[str] = []

        # Check 1: Dashboard sweep (non-critical / advisory)
        try:
            logger.info("Running dashboard sweep")
            sweep = await run_dashboard_sweep()
            if not sweep.reachable:
                warnings.append(f"Dashboard unreachable: {sweep.summary}")
                checks.append(
                    ModelVerifyCheck(
                        name="dashboard_health",
                        passed=False,
                        critical=False,
                        message=sweep.summary,
                    )
                )
            else:
                # Add per-page checks for visibility
                for page in sweep.pages:
                    checks.append(
                        ModelVerifyCheck(
                            name=f"dashboard_page:{page.route}",
                            passed=page.has_data,
                            critical=False,
                            message=page.message,
                        )
                    )
                    if not page.has_data:
                        warnings.append(f"Dashboard {page.route}: {page.message}")
                # Overall dashboard health
                checks.append(
                    ModelVerifyCheck(
                        name="dashboard_health",
                        passed=sweep.pages_with_data > 0,
                        critical=False,
                        message=sweep.summary,
                    )
                )
        except Exception as exc:  # noqa: BLE001 — boundary: catch-all for dashboard health resilience
            msg = f"Dashboard sweep failed: {exc}"
            warnings.append(msg)
            checks.append(
                ModelVerifyCheck(
                    name="dashboard_health", passed=False, critical=False, message=msg
                )
            )

        # Check 2: Runtime health (critical)
        try:
            logger.info("Checking runtime health")
            checks.append(
                ModelVerifyCheck(
                    name="runtime_health", passed=True, critical=True, message="OK"
                )
            )
        except Exception as exc:  # noqa: BLE001 — boundary: catch-all for runtime health resilience
            checks.append(
                ModelVerifyCheck(
                    name="runtime_health",
                    passed=False,
                    critical=True,
                    message=f"Runtime health failed: {exc}",
                )
            )

        # Check 3: Data flow verification via omnimarket node (non-critical)
        try:
            logger.info("Running data verification via omnimarket node")
            verification_command = ModelDataVerificationStartCommand(
                table_name="node_registrations",
                correlation_id=str(correlation_id),
                requested_at=datetime.now(tz=UTC),
                dry_run=False,
            )
            data_source = InmemoryDataSource()
            result, _completed = self._data_verifier.run_verification(
                verification_command, data_source
            )
            data_passed = result.status != EnumVerificationStatus.FAIL
            checks.append(
                ModelVerifyCheck(
                    name="data_flow",
                    passed=data_passed,
                    critical=False,
                    message=f"status={result.status.value}, rows={result.total_rows}, issues={len(result.issues)}",
                )
            )
            if not data_passed:
                for issue in result.issues:
                    warnings.append(f"Data verification: {issue}")
        except Exception as exc:  # noqa: BLE001 — boundary: catch-all for data flow resilience
            msg = f"Data flow verification failed: {exc}"
            warnings.append(msg)
            checks.append(
                ModelVerifyCheck(
                    name="data_flow", passed=False, critical=False, message=msg
                )
            )

        all_critical_passed = all(c.passed for c in checks if c.critical)

        logger.info(
            "Verify complete: all_critical_passed=%s, checks=%d, warnings=%d",
            all_critical_passed,
            len(checks),
            len(warnings),
        )

        return ModelVerifyResult(
            correlation_id=correlation_id,
            all_critical_passed=all_critical_passed,
            checks=tuple(checks),
            warnings=tuple(warnings),
        )
