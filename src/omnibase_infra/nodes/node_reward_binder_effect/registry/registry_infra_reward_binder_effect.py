# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Dependency injection registry for the RewardBinder effect node.

Provides factory methods for creating ``NodeRewardBinderEffect`` instances
wired with a Kafka publisher.

Usage::

    from omnibase_infra.nodes.node_reward_binder_effect.registry import (
        RegistryInfraRewardBinderEffect,
    )

    container = ModelONEXContainer()
    node = RegistryInfraRewardBinderEffect.create_with_publisher(
        container=container,
        publisher=publisher.publish,
    )

Ticket: OMN-2552
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_reward_binder_effect.node import (
        NodeRewardBinderEffect,
    )

logger = logging.getLogger(__name__)

_HANDLER_STORAGE: dict[str, object] = {}
_PROTOCOL_METADATA: dict[str, dict[str, object]] = {}


class RegistryInfraRewardBinderEffect:
    """Registry for RewardBinder effect node handlers."""

    HANDLER_KEY = "handler_reward_binder"

    @staticmethod
    def register(_container: ModelONEXContainer) -> None:
        """Register RewardBinder effect metadata with the container."""
        _PROTOCOL_METADATA[RegistryInfraRewardBinderEffect.HANDLER_KEY] = {
            "handler": "HandlerRewardBinder",
            "module": "omnibase_infra.nodes.node_reward_binder_effect.handlers.handler_reward_binder",
            "description": "Handler for reward event emission to Kafka",
            "capabilities": [
                "reward.emit.run_evaluated",
                "reward.emit.reward_assigned",
                "reward.emit.policy_state_updated",
            ],
        }

    @staticmethod
    def create_with_publisher(
        container: ModelONEXContainer,
        publisher: Callable[..., Awaitable[bool]],
    ) -> NodeRewardBinderEffect:
        """Create a NodeRewardBinderEffect wired with a Kafka publisher.

        Args:
            container: ONEX dependency injection container.
            publisher: Async callable matching PublisherTopicScoped.publish
                signature: ``async (event_type, payload, topic, correlation_id) -> bool``.

        Returns:
            Configured NodeRewardBinderEffect instance.

        Raises:
            ProtocolConfigurationError: If publisher is not callable.
        """
        if not callable(publisher):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="create_with_publisher",
                target_name="reward_binder_handler",
            )
            raise ProtocolConfigurationError(
                "publisher must be a callable async function",
                context=context,
            )

        from omnibase_infra.nodes.node_reward_binder_effect.handlers.handler_reward_binder import (
            HandlerRewardBinder,
        )
        from omnibase_infra.nodes.node_reward_binder_effect.node import (
            NodeRewardBinderEffect,
        )

        handler = HandlerRewardBinder(container=container, publisher=publisher)
        node = NodeRewardBinderEffect(container=container)

        _HANDLER_STORAGE[RegistryInfraRewardBinderEffect.HANDLER_KEY] = handler
        logger.info("RewardBinder effect node created with publisher")
        return node

    @staticmethod
    def get_handler(handler_type: str = "reward_binder") -> object | None:
        """Retrieve a registered handler by type.

        Args:
            handler_type: Handler type key (default: ``"reward_binder"``).

        Returns:
            Handler instance, or None if not registered.
        """
        return _HANDLER_STORAGE.get(
            f"{RegistryInfraRewardBinderEffect.HANDLER_KEY}.{handler_type}"
        ) or _HANDLER_STORAGE.get(RegistryInfraRewardBinderEffect.HANDLER_KEY)

    @staticmethod
    def reset() -> None:
        """Clear all registered handlers (for testing only)."""
        _HANDLER_STORAGE.clear()
        _PROTOCOL_METADATA.clear()


__all__: list[str] = ["RegistryInfraRewardBinderEffect"]
