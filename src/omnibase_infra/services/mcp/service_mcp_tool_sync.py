# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP Tool Sync Service - Kafka listener for hot reload with idempotency.

This service subscribes to node registration events on Kafka and updates
the MCP tool registry in real-time. It supports:
- Hot reload: New/updated orchestrators appear as tools without restart
- Deregistration: Removed orchestrators are removed from tool registry
- Idempotency: Duplicate/out-of-order events are handled correctly

Event Topic: Uses SUFFIX_NODE_REGISTRATION (onex.evt.platform.node-registration.v1)
Event Types:
    - registered: New node registered → upsert tool
    - updated: Node updated → upsert tool
    - deregistered: Node deregistered → remove tool
    - expired: Node liveness expired → remove tool
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_core.container import ModelONEXContainer
from omnibase_core.types import JsonType
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.mcp.model_mcp_tool_definition import (
    ModelMCPToolDefinition,
)
from omnibase_infra.topics import SUFFIX_NODE_REGISTRATION

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models import ModelEventMessage
    from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
        ServiceMCPToolDiscovery,
    )
    from omnibase_infra.services.mcp.service_mcp_tool_registry import (
        ServiceMCPToolRegistry,
    )

logger = logging.getLogger(__name__)


class ServiceMCPToolSync:
    """Kafka listener for MCP tool hot reload with idempotency.

    This service subscribes to node registration events and updates the
    tool registry accordingly. It handles:
    - registered/updated events → upsert tool in registry
    - deregistered/expired events → remove tool from registry

    Idempotency:
        Uses event_id (from event payload or Kafka offset) to ensure
        out-of-order and duplicate events are handled correctly.

    Consul Fallback:
        When registration events don't contain full contract info,
        the service falls back to Consul discovery to re-fetch the
        tool definition.

    Attributes:
        _registry: Tool registry for storing tool definitions.
        _discovery: Consul discovery service for fallback lookups.
        _bus: Kafka event bus for subscriptions.
        _unsubscribe: Callback to unsubscribe from topic.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> # Ensure services are registered in container first
        >>> sync = ServiceMCPToolSync(container)
        >>> await sync.start()
        >>> # ... process events ...
        >>> await sync.stop()
    """

    # Topic for node registration events (uses platform suffix constant)
    TOPIC = SUFFIX_NODE_REGISTRATION

    # MCP tag constants
    TAG_MCP_ENABLED = "mcp-enabled"
    TAG_NODE_TYPE_ORCHESTRATOR = "node-type:orchestrator"
    TAG_PREFIX_MCP_TOOL = "mcp-tool:"

    # Event types
    EVENT_TYPE_REGISTERED = "registered"
    EVENT_TYPE_UPDATED = "updated"
    EVENT_TYPE_DEREGISTERED = "deregistered"
    EVENT_TYPE_EXPIRED = "expired"

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        *,
        registry: ServiceMCPToolRegistry | None = None,
        discovery: ServiceMCPToolDiscovery | None = None,
        bus: EventBusKafka | None = None,
    ) -> None:
        """Initialize the sync service.

        Supports two initialization patterns:
        1. Container-based DI: Pass a ModelONEXContainer to resolve dependencies
        2. Direct injection: Pass registry, discovery, and bus directly

        Args:
            container: Optional ONEX container for dependency injection.
            registry: Tool registry (used if container not provided)
            discovery: Discovery service for Consul fallback (used if container not provided)
            bus: Kafka event bus (used if container not provided)

        Raises:
            ValueError: If neither container nor all direct dependencies are provided.
        """
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
            ServiceMCPToolDiscovery,
        )
        from omnibase_infra.services.mcp.service_mcp_tool_registry import (
            ServiceMCPToolRegistry,
        )

        self._container = container

        if container is not None:
            # Resolve from container
            self._registry: ServiceMCPToolRegistry = container.get_service(
                ServiceMCPToolRegistry
            )
            self._discovery: ServiceMCPToolDiscovery = container.get_service(
                ServiceMCPToolDiscovery
            )
            self._bus: EventBusKafka = container.get_service(EventBusKafka)
        elif registry is not None and discovery is not None and bus is not None:
            # Use directly provided dependencies
            self._registry = registry
            self._discovery = discovery
            self._bus = bus
        else:
            raise ValueError(
                "Must provide either container or all of: registry, discovery, bus"
            )

        self._unsubscribe: Callable[[], Awaitable[None]] | None = None
        self._started = False

        logger.debug(
            "ServiceMCPToolSync initialized",
            extra={
                "topic": self.TOPIC,
            },
        )

    @property
    def is_running(self) -> bool:
        """Return True if the sync service is running."""
        return self._started

    async def start(self) -> None:
        """Start the Kafka subscription for hot reload.

        Subscribes to the node registration topic and begins processing
        events. The subscription is idempotent - calling start() multiple
        times has no effect.
        """
        if self._started:
            logger.debug("ServiceMCPToolSync already started")
            return

        correlation_id = uuid4()

        # OMN-1602: Typed node identity for consumer group derivation.
        #
        # Identity Field Rationale:
        # - env: From bus.environment for deployment-specific consumer groups
        # - service="mcp": MCP is a singleton per environment
        # - node_name="tool_sync": Unique consumer within the MCP service
        # - version="v1": Consumer protocol version (increment on breaking changes)
        sync_identity = ModelNodeIdentity(
            env=self._bus.environment,
            service="mcp",
            node_name="tool_sync",
            version="v1",
        )

        logger.info(
            "Starting MCP tool sync",
            extra={
                "topic": self.TOPIC,
                "node_identity": {
                    "env": sync_identity.env,
                    "service": sync_identity.service,
                    "node_name": sync_identity.node_name,
                    "version": sync_identity.version,
                },
                "correlation_id": str(correlation_id),
            },
        )

        self._unsubscribe = await self._bus.subscribe(
            topic=self.TOPIC,
            node_identity=sync_identity,
            on_message=self._on_message,
        )

        self._started = True

        logger.info(
            "MCP tool sync started",
            extra={
                "topic": self.TOPIC,
                "correlation_id": str(correlation_id),
            },
        )

    async def stop(self) -> None:
        """Stop the Kafka subscription.

        Unsubscribes from the topic and stops processing events.
        Idempotent - safe to call multiple times.
        """
        if not self._started:
            logger.debug("ServiceMCPToolSync already stopped")
            return

        correlation_id = uuid4()

        logger.info(
            "Stopping MCP tool sync",
            extra={"correlation_id": str(correlation_id)},
        )

        if self._unsubscribe is not None:
            await self._unsubscribe()
            self._unsubscribe = None

        self._started = False

        logger.info(
            "MCP tool sync stopped",
            extra={"correlation_id": str(correlation_id)},
        )

    async def _on_message(self, msg: ModelEventMessage) -> None:
        """Process a registration event message.

        Args:
            msg: Event message from Kafka.
        """
        correlation_id = msg.headers.correlation_id

        try:
            # Parse event payload
            event = self._parse_event(msg)
            if event is None:
                logger.debug(
                    "Skipping non-JSON message",
                    extra={"correlation_id": str(correlation_id)},
                )
                return

            # Extract event metadata
            event_type = event.get("event_type", "")
            tags_raw = event.get("tags", [])
            # Ensure tags is a list of strings for type safety
            tags: list[str] = (
                [str(t) for t in tags_raw]
                if isinstance(tags_raw, (list, tuple))
                else []
            )
            node_id = event.get("node_id")
            service_id = event.get("service_id")

            # Use event_id from payload or fall back to Kafka offset
            # Numeric offsets are zero-padded to 20 digits for lexicographic ordering
            event_id_raw = event.get("event_id") or msg.offset
            if event_id_raw is None:
                event_id = str(uuid4())
            else:
                event_id_str = str(event_id_raw)
                event_id = (
                    event_id_str.zfill(20) if event_id_str.isdigit() else event_id_str
                )

            # Check if this is an MCP-enabled orchestrator
            if not self._is_mcp_orchestrator(tags):
                logger.debug(
                    "Ignoring non-MCP event",
                    extra={
                        "event_type": event_type,
                        "tags": tags,
                        "correlation_id": str(correlation_id),
                    },
                )
                return

            # Route to appropriate handler
            if event_type in (self.EVENT_TYPE_REGISTERED, self.EVENT_TYPE_UPDATED):
                await self._handle_upsert_event(event, event_id, correlation_id)
            elif event_type in (self.EVENT_TYPE_DEREGISTERED, self.EVENT_TYPE_EXPIRED):
                await self._handle_remove_event(tags, event_id, correlation_id)
            else:
                logger.debug(
                    "Unknown event type",
                    extra={
                        "event_type": event_type,
                        "correlation_id": str(correlation_id),
                    },
                )

        except Exception as e:
            logger.exception(
                "Error processing registration event",
                extra={
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                },
            )

    def _parse_event(self, msg: ModelEventMessage) -> dict[str, JsonType] | None:
        """Parse event payload from message.

        Args:
            msg: Event message.

        Returns:
            Parsed event dict or None if not JSON.
        """
        if msg.value is None:
            return None

        try:
            value = msg.value
            if isinstance(value, bytes):
                value_str = value.decode("utf-8")
            elif isinstance(value, str):
                value_str = value
            else:
                return None
            parsed: dict[str, JsonType] = json.loads(value_str)
            return parsed
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _is_mcp_orchestrator(self, tags: Sequence[str]) -> bool:
        """Check if event is for an MCP-enabled orchestrator.

        Args:
            tags: List of tags from the event.

        Returns:
            True if the event is for an MCP-enabled orchestrator.
        """
        return self.TAG_MCP_ENABLED in tags and self.TAG_NODE_TYPE_ORCHESTRATOR in tags

    def _extract_tool_name(self, tags: Sequence[str]) -> str | None:
        """Extract the MCP tool name from tags.

        Args:
            tags: List of tags from the event.

        Returns:
            The tool name if found, None otherwise.
        """
        for tag in tags:
            if tag.startswith(self.TAG_PREFIX_MCP_TOOL):
                return tag[len(self.TAG_PREFIX_MCP_TOOL) :]
        return None

    def _extract_tags_list(self, tags_raw: object) -> list[str]:
        """Safely extract tags as a list of strings.

        Args:
            tags_raw: Raw tags value from event (could be any type).

        Returns:
            List of string tags, empty list if input is invalid.
        """
        if isinstance(tags_raw, (list, tuple)):
            return [str(t) for t in tags_raw]
        return []

    async def _handle_upsert_event(
        self,
        event: dict[str, JsonType],
        event_id: str,
        correlation_id: object,
    ) -> None:
        """Handle registered/updated events by upserting tool.

        Args:
            event: Parsed event payload (JSON-compatible values).
            event_id: Unique event identifier for idempotency.
            correlation_id: Correlation ID for tracing.
        """
        tags_raw = event.get("tags", [])
        tags: list[str] = (
            [str(t) for t in tags_raw] if isinstance(tags_raw, (list, tuple)) else []
        )
        tool_name = self._extract_tool_name(tags)

        if not tool_name:
            logger.warning(
                "MCP event missing tool name tag",
                extra={
                    "tags": tags,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        # Try to build tool from event data
        tool = self._build_tool_from_event(event, tool_name)

        # Fallback: if event lacks full info, re-fetch from Consul
        if tool is None:
            service_id = event.get("service_id")
            if service_id and isinstance(service_id, str):
                logger.debug(
                    "Event lacks full info, falling back to Consul",
                    extra={
                        "service_id": service_id,
                        "correlation_id": str(correlation_id),
                    },
                )
                tool = await self._discovery.discover_by_service_id(service_id)

        if tool is None:
            logger.warning(
                "Could not build tool definition from event or Consul",
                extra={
                    "tool_name": tool_name,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        # Upsert in registry
        updated = await self._registry.upsert_tool(tool, event_id)
        if updated:
            logger.info(
                "Tool upserted from event",
                extra={
                    "tool_name": tool_name,
                    "event_id": event_id,
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            logger.debug(
                "Tool upsert skipped (stale event)",
                extra={
                    "tool_name": tool_name,
                    "event_id": event_id,
                    "correlation_id": str(correlation_id),
                },
            )

    async def _handle_remove_event(
        self,
        tags: Sequence[str],
        event_id: str,
        correlation_id: object,
    ) -> None:
        """Handle deregistered/expired events by removing tool.

        Args:
            tags: Tags from the event (used to extract tool name).
            event_id: Unique event identifier for idempotency.
            correlation_id: Correlation ID for tracing.
        """
        tool_name = self._extract_tool_name(tags)
        if not tool_name:
            logger.debug(
                "Remove event missing tool name tag",
                extra={
                    "tags": tags,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        removed = await self._registry.remove_tool(tool_name, event_id)
        if removed:
            logger.info(
                "Tool removed from registry",
                extra={
                    "tool_name": tool_name,
                    "event_id": event_id,
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            logger.debug(
                "Tool removal skipped (stale event or not found)",
                extra={
                    "tool_name": tool_name,
                    "event_id": event_id,
                    "correlation_id": str(correlation_id),
                },
            )

    def _build_tool_from_event(
        self,
        event: dict[str, JsonType],
        tool_name: str,
    ) -> ModelMCPToolDefinition | None:
        """Build a tool definition from event data.

        Args:
            event: Parsed event payload (JSON-compatible values).
            tool_name: Extracted tool name.

        Returns:
            Tool definition if event contains enough info, None otherwise.
        """
        # Check if event has the minimum required fields
        service_name = event.get("service_name")
        if not service_name:
            return None

        # Extract optional fields
        service_id = event.get("service_id")
        node_id = event.get("node_id")
        endpoint = event.get("endpoint")
        description = event.get("description")
        timeout_seconds = event.get("timeout_seconds", 30)

        # Validate timeout
        if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
            timeout_seconds = 30

        return ModelMCPToolDefinition(
            name=tool_name,
            description=str(description)
            if description
            else f"ONEX orchestrator: {service_name}",
            version="1.0.0",
            parameters=[],
            input_schema={"type": "object", "properties": {}},
            orchestrator_node_id=str(node_id) if node_id else None,
            orchestrator_service_id=str(service_id) if service_id else None,
            endpoint=str(endpoint) if endpoint else None,
            timeout_seconds=timeout_seconds,
            metadata={
                "service_name": str(service_name),
                "tags": self._extract_tags_list(event.get("tags", [])),
                "source": "kafka_event",
            },
        )

    def describe(self) -> dict[str, object]:
        """Return service metadata for observability."""
        return {
            "service_name": "ServiceMCPToolSync",
            "topic": self.TOPIC,
            "group_id_derived": True,  # Group ID derived from ModelNodeIdentity
            "is_running": self._started,
        }


__all__ = ["ServiceMCPToolSync"]
