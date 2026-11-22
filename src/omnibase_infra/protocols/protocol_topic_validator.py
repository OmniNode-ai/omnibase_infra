# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel's ``TopicStartupValidator``, as the auto-create pass uses it (OMN-26359)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.event_bus.model_topic_validation_result import (
        ModelTopicValidationResult,
    )

__all__ = ["ProtocolTopicValidator"]


class ProtocolTopicValidator(Protocol):
    """The kernel's ``TopicStartupValidator``, as the auto-create pass uses it."""

    async def validate(
        self, correlation_id: UUID | None = None, *, log_missing: bool = True
    ) -> ModelTopicValidationResult: ...
