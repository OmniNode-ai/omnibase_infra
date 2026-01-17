# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Domain plugin configuration model.

This module provides the ModelDomainPluginConfig dataclass for passing
configuration to domain plugins during lifecycle hooks.

Design Pattern:
    The kernel creates this config and passes it to each domain plugin
    during bootstrap, providing all context needed for initialization
    and handler wiring.

Thread Safety:
    This is a mutable dataclass. The dispatch_engine field may be set
    after initial construction when the MessageDispatchEngine is created.

Example:
    >>> from omnibase_infra.runtime.models import ModelDomainPluginConfig
    >>> from uuid import uuid4
    >>>
    >>> config = ModelDomainPluginConfig(
    ...     container=container,
    ...     event_bus=event_bus,
    ...     correlation_id=uuid4(),
    ...     input_topic="requests",
    ...     output_topic="responses",
    ...     consumer_group="onex-runtime",
    ... )
    >>> result = await plugin.initialize(config)

Related:
    - OMN-1346: Registration Code Extraction
    - OMN-888: Registration Orchestrator
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
    from omnibase_infra.runtime import MessageDispatchEngine


@dataclass
class ModelDomainPluginConfig:
    """Configuration passed to domain plugins during lifecycle hooks.

    This model provides all the context a domain plugin needs to initialize
    its resources and wire its handlers. The kernel creates this config
    and passes it to each plugin during bootstrap.

    Attributes:
        container: The ONEX container for dependency injection.
        event_bus: The event bus instance (InMemoryEventBus or KafkaEventBus).
        correlation_id: Correlation ID for distributed tracing.
        input_topic: The input topic for event consumers.
        output_topic: The output topic for event publishers.
        consumer_group: The consumer group for Kafka consumers.
        dispatch_engine: The MessageDispatchEngine for dispatcher wiring
            (set after engine creation, may be None).

    Example:
        ```python
        config = ModelDomainPluginConfig(
            container=container,
            event_bus=event_bus,
            correlation_id=uuid4(),
            input_topic="requests",
            output_topic="responses",
            consumer_group="onex-runtime",
        )
        result = await plugin.initialize(config)
        ```
    """

    container: ModelONEXContainer
    event_bus: InMemoryEventBus | KafkaEventBus
    correlation_id: UUID
    input_topic: str
    output_topic: str
    consumer_group: str

    # Optional: MessageDispatchEngine for dispatcher wiring (set after engine creation)
    dispatch_engine: MessageDispatchEngine | None = None


__all__ = ["ModelDomainPluginConfig"]
