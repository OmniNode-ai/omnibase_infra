# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One record of a transactional publish batch (OMN-21228)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders


class ModelTransactionalRecord(BaseModel):
    """A record handed to ``ProtocolTransactionalProducer.publish``.

    The same four fields ``publish`` takes, carried as one value so a batch can
    be validated and staged before any of it is sent.

    Attributes:
        topic: Canonical topic name (the bus applies its own namespace).
        key: Optional message key.
        value: Message payload.
        headers: Optional event headers; derived from the body when absent,
            exactly as for a plain ``publish``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = Field(..., min_length=1)
    key: bytes | None = None
    value: bytes
    headers: ModelEventHeaders | None = None
