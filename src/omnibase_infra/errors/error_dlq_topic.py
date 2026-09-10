# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""DLQ topic resolution refusal (OMN-18084).

``get_dlq_topic_for_original`` resolves a message category from a topic's own
segments. On a DLQ topic that is a FIXED POINT: ``onex.dlq.omnibase-infra.events.v1``
contains the segment ``events``, classifies as EVENT, and rebuilds the identical
name. Every caller that hands it a topic it just consumed from gets back the
instruction to write the record onto the topic it came from.

Measured on the .201 dev lane 2026-09-09: the auto-wiring boundary did exactly
that on its ``HANDLER_ERROR`` leg, at 193.8 records/s and ~151 GB/day, against a
mount with 590 GB free shared by the prod, stability-test and judge lanes.

A per-call-site guard closes one hole; refusing at the resolver closes the class.
The refusal is a typed error rather than ``None`` on purpose: ``None`` is already
this resolver's answer for "category could not be determined", and a caller that
falls back on ``None`` would treat a loop as an unremarkable miss.
"""

from __future__ import annotations

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class DlqTopicFixedPointError(RuntimeHostError):
    """Raised when a dead-letter topic is offered as an ORIGINAL topic.

    Callers must decide what a failure ON a dead-letter sink means before they
    ask where to route it. The record is already durably captured, so the answer
    is evidence and a stop — never another dead-letter write.

    Example:
        >>> raise DlqTopicFixedPointError(
        ...     "cannot resolve a DLQ topic for a topic that is already a DLQ "
        ...     "topic: onex.dlq.omnibase-infra.events.v1",
        ...     original_topic="onex.dlq.omnibase-infra.events.v1",
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize DlqTopicFixedPointError.

        Args:
            message: Human-readable error message naming the refused topic
            context: Bundled infrastructure context
            **extra_context: Additional context (original_topic, ...)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_INPUT,
            context=context,
            **extra_context,
        )
