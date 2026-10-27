# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-owned local transport supplied by the kernel to Kafka recovery."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.models import ModelNodeIdentity


class ProtocolRuntimeFallbackBus(Protocol):
    """Infra-shaped local delivery; the kernel owns adapter construction."""

    async def start(self) -> None:
        """Start local delivery."""
        ...

    async def close(self) -> None:
        """Stop local delivery and release subscriptions."""
        ...

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
    ) -> ModelPublishReceipt:
        """Publish locally, preserving the receipt's transport identity."""
        ...

    async def subscribe(
        self,
        topic: str,
        node_identity: ModelNodeIdentity | None = None,
        on_message: Callable[[ModelEventMessage], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
        required_for_readiness: bool = False,
    ) -> Callable[[], Awaitable[None]]:
        """Register local delivery and return its unsubscribe callback."""
        ...

    async def health_check(self) -> dict[str, object]:
        """Report whether local delivery is operational."""
        ...
