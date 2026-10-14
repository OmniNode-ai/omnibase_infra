# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for ingesting GitHub PR webhook events and mapping them to ModelUpdateTrigger.

Maps the ``action`` field from a ``ModelPRWebhookEvent`` to the appropriate
``trigger_type`` on a ``ModelUpdateTrigger``:

    opened       → pr_opened
    synchronize  → pr_updated
    closed+merged → pr_merged
    closed (not merged), reopened, edited → pr_updated

Handler Purity:
    This handler does NOT publish events. It returns a ``ModelUpdateTrigger``
    for the node shell / runtime to publish to ``onex.evt.artifact.change-detected.v1``.

Related Tickets:
    - OMN-3940: Task 5 — Change Detector EFFECT Node
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumUpdateTriggerType,
)
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_pr_webhook_event import (
    ModelPRWebhookEvent,
)
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)

logger = logging.getLogger(__name__)

# Action → trigger_type mapping table
_ACTION_TO_TRIGGER_TYPE: dict[str, EnumUpdateTriggerType] = {
    "opened": EnumUpdateTriggerType.PR_OPENED,
    "synchronize": EnumUpdateTriggerType.PR_UPDATED,
    "reopened": EnumUpdateTriggerType.PR_UPDATED,
    "edited": EnumUpdateTriggerType.PR_UPDATED,
}

__all__ = ["HandlerPRWebhookIngestion"]


class HandlerPRWebhookIngestion:
    """Handler for the ``artifact.ingest_pr_webhook`` operation.

    Consumes a ``ModelPRWebhookEvent`` and maps it to a ``ModelUpdateTrigger``
    for downstream processing by the impact analyzer COMPUTE node.

    Mapping rules:
        - ``opened``          → ``trigger_type="pr_opened"``
        - ``synchronize``     → ``trigger_type="pr_updated"``
        - ``reopened``        → ``trigger_type="pr_updated"``
        - ``edited``          → ``trigger_type="pr_updated"``
        - ``closed`` + merged → ``trigger_type="pr_merged"``
        - ``closed`` (unmerged) → ``trigger_type="pr_updated"``

    The ``source_ref`` is set to ``refs/pull/{pr_number}/head``.
    The ``trigger_id`` is a new UUID4 assigned at ingestion time.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: NODE_HANDLER."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (external event ingestion)."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(self, event: ModelPRWebhookEvent) -> ModelUpdateTrigger:
        """Map a PR webhook event to a ModelUpdateTrigger.

        Canonical definition-B entrypoint: the runtime validates the wire payload
        into ``ModelPRWebhookEvent`` and publishes the returned trigger to
        ``onex.evt.artifact.change-detected.v1``.

        Args:
            event: The incoming PR webhook event.

        Returns:
            A ``ModelUpdateTrigger`` with trigger_type derived from the action.
        """
        trigger_type: EnumUpdateTriggerType
        if event.action == "closed":
            trigger_type = (
                EnumUpdateTriggerType.PR_MERGED
                if event.merged
                else EnumUpdateTriggerType.PR_UPDATED
            )
        else:
            trigger_type = _ACTION_TO_TRIGGER_TYPE.get(
                event.action, EnumUpdateTriggerType.PR_UPDATED
            )

        trigger = ModelUpdateTrigger(
            trigger_id=uuid4(),
            trigger_type=trigger_type,
            source_repo=event.repo,
            source_ref=f"refs/pull/{event.pr_number}/head",
            changed_files=list(event.changed_files),
            ticket_ids=list(event.ticket_ids),
            actor=event.actor,
            reason=f"PR #{event.pr_number} action={event.action}",
            timestamp=datetime.now(tz=UTC),
        )

        logger.debug(
            "Ingested PR webhook: repo=%s pr=%d action=%s → trigger_type=%s trigger_id=%s",
            event.repo,
            event.pr_number,
            event.action,
            trigger_type,
            trigger.trigger_id,
        )
        return trigger
