# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane-mirror record refusal error (OMN-17919).

The lane mirror crosses no trust boundary: it republishes a record byte-for-byte
from one lane broker to another on the same operator-controlled host. The one
thing it must read out of a record is the identity it keys its durable
idempotency marker on, and that identity is the mandatory ``message_id`` Kafka
header stamped by ``event_bus_kafka._model_headers_to_kafka`` on every publish.

A record that carries no parseable ``message_id`` cannot be mirrored safely --
without a stable key, a redelivery would republish it and the leg's exactly-once
promise would be a coin flip. Refusing it is correct; refusing it *silently* is
not, which is what this typed error exists to prevent. It pairs with
``NodeLaneMirror.refused_record_count`` so a systematically malformed producer
reads as a rising counter and a classifiable ERROR, not as a mirror that
mysteriously moves nothing.
"""

from __future__ import annotations

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class LaneMirrorRecordRefusedError(RuntimeHostError):
    """Raised when a source record carries no usable lane-mirror identity.

    Used for:
    - A record with no ``message_id`` header at all
    - A ``message_id`` header whose value is not a UUID

    NOT used for a record whose topic is outside the contract-declared
    ``lane_mirror.topics`` set -- that is deny-by-default working as designed,
    not a malformed record, and it deliberately does not tick the refusal
    counter.

    Example:
        >>> raise LaneMirrorRecordRefusedError(
        ...     "lane mirror record carries no message_id header",
        ...     topic="onex.evt.omniclaude.prompt-submitted.v1",
        ...     partition=2,
        ...     offset=285,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize LaneMirrorRecordRefusedError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context (topic, partition, offset, ...)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_INPUT,
            context=context,
            **extra_context,
        )


__all__ = ["LaneMirrorRecordRefusedError"]
