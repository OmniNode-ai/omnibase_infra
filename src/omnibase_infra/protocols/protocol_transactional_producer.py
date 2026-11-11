# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A producer that publishes batches transactionally under a fixed id (OMN-21228).

Acceptance and publication can only be one atomic act if the publish side can
say "all of this batch, or none of it" and can be fenced. A producer is opened
under a caller-fixed ``transactional_id``; opening a newer producer under the
same id fences every older one, whose next publish fails and leaves nothing
behind.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.models.event_bus.model_transactional_record import (
    ModelTransactionalRecord,
)


@runtime_checkable
class ProtocolTransactionalProducer(Protocol):
    """One producer, one ``transactional_id``, one transaction per ``publish``."""

    @property
    def transactional_id(self) -> str:
        """The caller-fixed id this producer is fenced by."""
        ...

    async def publish(
        self, records: Sequence[ModelTransactionalRecord]
    ) -> tuple[ModelPublishReceipt, ...]:
        """Publish ``records`` as one transaction, returning one receipt each.

        Raises:
            InfraProducerFencedError: A newer producer holds the same id. Nothing
                of this batch is visible to a committed reader.
            InfraUnavailableError: The producer or its bus is closed.
            InfraConnectionError: The transaction failed for any other reason
                and was aborted.
            ProtocolConfigurationError: The batch is empty or a record's topic
                is invalid.
        """
        ...

    async def close(self) -> None:
        """Release the producer. Safe to call more than once."""
        ...
