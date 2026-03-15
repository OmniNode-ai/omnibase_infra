# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that converts a ModelManualReconcileCommand into a ModelUpdateTrigger.

When a user or agent issues ``omni-infra artifact-reconcile``, the CLI
publishes a ``ModelManualReconcileCommand`` to ``onex.cmd.artifact.reconcile.v1``.
This handler consumes that command and emits a ``ModelUpdateTrigger`` with
``trigger_type="manual_plan_request"``.

When ``command.changed_files`` is empty the trigger also has empty
``changed_files``, which the downstream COMPUTE node interprets as a
full-repo reconciliation (all registered artifacts are evaluated).

Handler Purity:
    This handler does NOT publish events. It returns a ``ModelUpdateTrigger``
    for the node shell / runtime to publish.

Related Tickets:
    - OMN-3940: Task 5 — Change Detector EFFECT Node
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_manual_reconcile_command import (
    ModelManualReconcileCommand,
)
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)

logger = logging.getLogger(__name__)

__all__ = ["HandlerManualTrigger"]


class HandlerManualTrigger:
    """Handler for the ``artifact.manual_trigger`` operation.

    Converts a ``ModelManualReconcileCommand`` (from the CLI or a skill)
    into a ``ModelUpdateTrigger`` with ``trigger_type="manual_plan_request"``.

    The ``trigger_id`` is a new UUID4 assigned at handle time.
    The ``command_id`` is stored in ``reason`` for traceability.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: NODE_HANDLER."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (command ingestion)."""
        return EnumHandlerTypeCategory.EFFECT

    def build_reconcile_trigger(
        self, command: ModelManualReconcileCommand
    ) -> ModelUpdateTrigger:
        """Convert a manual reconcile command to a ModelUpdateTrigger.

        Args:
            command: The manual reconcile command to convert.

        Returns:
            A ``ModelUpdateTrigger`` with ``trigger_type="manual_plan_request"``.
        """
        reason = command.reason or f"Manual reconcile command_id={command.command_id}"

        trigger = ModelUpdateTrigger(
            trigger_id=uuid4(),
            trigger_type="manual_plan_request",
            source_repo=command.source_repo,
            source_ref=None,
            changed_files=list(command.changed_files),
            ticket_ids=[],
            actor=command.actor,
            reason=reason,
            timestamp=datetime.now(tz=UTC),
        )

        logger.debug(
            "Manual reconcile trigger: repo=%s command_id=%s trigger_id=%s files=%d",
            command.source_repo,
            command.command_id,
            trigger.trigger_id,
            len(trigger.changed_files),
        )
        return trigger
