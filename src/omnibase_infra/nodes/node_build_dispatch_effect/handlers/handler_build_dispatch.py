# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that dispatches ticket-pipeline builds via delegation.

This is an EFFECT handler - performs external I/O (delegation dispatch).

Related:
    - OMN-7318: node_build_dispatch_effect
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_outcome import (
    ModelBuildDispatchOutcome,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_result import (
    ModelBuildDispatchResult,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_target import (
    ModelBuildTarget,
)

logger = logging.getLogger(__name__)


class HandlerBuildDispatch:
    """Dispatches ticket-pipeline builds for AUTO_BUILDABLE tickets via delegation.

    Each ticket is dispatched as an independent delegation request.
    Failures on individual tickets do not block other dispatches.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        correlation_id: UUID,
        targets: tuple[ModelBuildTarget, ...],
        dry_run: bool = False,
    ) -> ModelBuildDispatchResult:
        """Dispatch builds for each target ticket.

        Args:
            correlation_id: Cycle correlation ID.
            targets: Tickets to dispatch.
            dry_run: Skip actual dispatch.

        Returns:
            ModelBuildDispatchResult with per-ticket outcomes.
        """
        logger.info(
            "Build dispatch: %d targets (correlation_id=%s, dry_run=%s)",
            len(targets),
            correlation_id,
            dry_run,
        )

        outcomes: list[ModelBuildDispatchOutcome] = []
        total_dispatched = 0
        total_failed = 0

        for target in targets:
            if dry_run:
                outcomes.append(
                    ModelBuildDispatchOutcome(
                        ticket_id=target.ticket_id,
                        dispatched=True,
                        error=None,
                    )
                )
                total_dispatched += 1
                continue

            try:
                # TODO: Wire to actual delegation orchestrator invocation
                logger.info(
                    "Dispatching ticket-pipeline for %s: %s",
                    target.ticket_id,
                    target.title,
                )
                outcomes.append(
                    ModelBuildDispatchOutcome(
                        ticket_id=target.ticket_id,
                        dispatched=True,
                        error=None,
                    )
                )
                total_dispatched += 1
            except Exception as exc:  # noqa: BLE001 — boundary: catch-all converts dispatch failure to outcome record
                logger.warning(
                    "Failed to dispatch %s: %s",
                    target.ticket_id,
                    exc,
                )
                outcomes.append(
                    ModelBuildDispatchOutcome(
                        ticket_id=target.ticket_id,
                        dispatched=False,
                        error=str(exc),
                    )
                )
                total_failed += 1

        logger.info(
            "Build dispatch complete: %d dispatched, %d failed",
            total_dispatched,
            total_failed,
        )

        return ModelBuildDispatchResult(
            correlation_id=correlation_id,
            outcomes=tuple(outcomes),
            total_dispatched=total_dispatched,
            total_failed=total_failed,
        )
