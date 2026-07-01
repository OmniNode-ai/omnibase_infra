# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""MCP Tool Registry - Event-loop safe in-memory cache of MCP tool definitions.

A thread-safe registry for MCP tool definitions, supporting:
- Event-driven updates from Kafka (hot reload)
- Idempotent operations with version tracking
- Concurrent access within a single event loop

The registry uses asyncio.Lock for coroutine-safe access. It is NOT thread-safe
across multiple threads/event loops - use within a single async context.

Version Tracking:
    Each tool has an associated version (event_id) to handle out-of-order
    Kafka messages. Operations only succeed if the event_id is newer than
    the last recorded version for that tool.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from uuid import UUID, uuid4

from omnibase_infra.models.mcp.model_mcp_generated_tool_registration import (
    ModelMCPGeneratedToolRegistration,
)
from omnibase_infra.models.mcp.model_mcp_tool_definition import (
    ModelMCPToolDefinition,
)

logger = logging.getLogger(__name__)


class ServiceMCPToolRegistry:
    """Event-loop safe in-memory cache of MCP tool definitions.

    Uses asyncio.Lock for coroutine-safe access within a single event loop.
    NOT thread-safe across multiple threads/event loops.

    Attributes:
        _tools: Dictionary mapping tool names to tool definitions.
        _versions: Dictionary mapping tool names to their last event_id.
        _lock: asyncio.Lock for coroutine-safe access.

    Version Tracking:
        The registry tracks event_id for each tool to handle idempotency:
        - Kafka events may arrive out of order
        - Duplicate events may be delivered
        - Only newer events (higher event_id) should update the registry

        Event IDs should be monotonically increasing (e.g., Kafka offset,
        timestamp-based UUID, or sequential counter).

    Example:
        >>> registry = ServiceMCPToolRegistry()
        >>> tool = ModelMCPToolDefinition(name="my_tool", description="...")
        >>> await registry.upsert_tool(tool, event_id="event-001")
        True
        >>> await registry.get_tool("my_tool")
        ModelMCPToolDefinition(name='my_tool', ...)
    """

    def __init__(self) -> None:
        """Initialize the tool registry with empty state."""
        self._tools: dict[str, ModelMCPToolDefinition] = {}
        self._versions: dict[str, str] = {}  # tool_name → last_event_id (normalized)
        self._lock: asyncio.Lock = asyncio.Lock()

        logger.debug("ServiceMCPToolRegistry initialized")

    def _normalize_event_id(self, event_id: str) -> str:
        """Normalize event_id for correct lexicographic comparison.

        Numeric IDs (e.g., Kafka offsets) are zero-padded to 20 digits to ensure
        correct lexicographic ordering. Non-numeric IDs are returned unchanged.

        Args:
            event_id: The event identifier to normalize.

        Returns:
            Normalized event_id suitable for lexicographic comparison.

        Example:
            >>> registry = ServiceMCPToolRegistry()
            >>> registry._normalize_event_id("9")
            '00000000000000000009'
            >>> registry._normalize_event_id("10")
            '00000000000000000010'
            >>> registry._normalize_event_id("event-001")
            'event-001'
        """
        if event_id.isdigit():
            return event_id.zfill(20)
        return event_id

    @property
    def tool_count(self) -> int:
        """Return the number of registered tools.

        Note: This is a snapshot and may change immediately after reading.
        """
        return len(self._tools)

    # MCP exposure tags required by ServiceMCPToolSync._is_mcp_exposable for a
    # generated COMPUTE node (the SEA self-extension loop output, OMN-12827 B2).
    _TAG_MCP_ENABLED = "mcp-enabled"
    _TAG_NODE_TYPE_COMPUTE = "node-type:compute"
    _TAG_GENERATED = "generated"
    _TAG_PREFIX_MCP_TOOL = "mcp-tool:"

    @staticmethod
    def _artifact_hash(text: str) -> str:
        """Return a ``sha256:<hex>`` digest binding a generated artifact."""
        return "sha256:" + hashlib.sha256(text.encode()).hexdigest()

    async def register_generated_tool(
        self,
        *,
        node_name: str,
        description: str,
        contract_yaml: str,
        handler_source: str,
        correlation_id: UUID,
    ) -> ModelMCPGeneratedToolRegistration:
        """Register a generated COMPUTE node as an MCP tool on the canonical registry.

        This is the canonical replacement for the bespoke SEA
        ``ToolRegistry.register`` (the in-process generated-tool store that the
        SEA self-extension loop used). It routes the *registration semantics* of
        a generated tool onto this canonical registry: the generated node surfaces
        through the one MCP exposure path (``ServiceMCPToolRegistry`` /
        ``ServiceMCPToolSync``) instead of a parallel bespoke store.

        The MCP server in ``omnibase_infra`` is the protocol-infra owner of tool
        registration (feedback_mcp_server_in_omnibase_infra.md). In-process
        ``exec()``/``invoke()`` of the handler is deliberately NOT part of this
        method: that hot-load sandbox concern belongs to the Phase 0 generation
        executor (OMN-13605), not to MCP registration.

        The built tool carries the tags the canonical exposure rule requires for a
        generated compute node (``mcp-enabled`` + ``node-type:compute`` +
        ``generated`` + ``mcp-tool:<name>``), so it is surfaced exactly as a
        hot-reloaded generated node from the bus would be.

        Args:
            node_name: Generated node name; becomes the canonical MCP tool name.
            description: AI-friendly description surfaced to MCP clients.
            contract_yaml: The generating contract YAML (hashed for provenance).
            handler_source: The generating handler source (hashed for provenance).
            correlation_id: Correlation id of the generation request.

        Returns:
            A typed ``ModelMCPGeneratedToolRegistration`` binding the artifact
            hashes and the registry version key used for the upsert.

        Raises:
            ValueError: If ``node_name`` is empty (fail fast; no silent default).
        """
        if not node_name:
            raise ValueError("node_name must be a non-empty generated node name")

        contract_hash = self._artifact_hash(contract_yaml)
        handler_hash = self._artifact_hash(handler_source)

        # Version key binds to the artifact content so re-registering the same
        # generated artifact is idempotent (same hashes -> same version), while a
        # revised contract/handler mints a newer version that updates the registry.
        registry_event_id = self._artifact_hash(contract_hash + handler_hash)

        tags = [
            self._TAG_MCP_ENABLED,
            self._TAG_NODE_TYPE_COMPUTE,
            self._TAG_GENERATED,
            f"{self._TAG_PREFIX_MCP_TOOL}{node_name}",
        ]

        tool = ModelMCPToolDefinition(
            name=node_name,
            description=description,
            version="1.0.0",
            parameters=[],
            input_schema={"type": "object", "properties": {}},
            orchestrator_node_id=None,
            orchestrator_service_id=None,
            endpoint=None,
            timeout_seconds=30,
            metadata={
                "tags": tags,
                "node_kind": "generated compute node",
                "generated_contract_hash": contract_hash,
                "generated_handler_hash": handler_hash,
                "generation_correlation_id": str(correlation_id),
                "source": "generated_tool_registration",
            },
        )

        await self.upsert_tool(tool, registry_event_id)

        logger.info(
            "Generated tool registered on canonical MCP registry",
            extra={
                "tool_name": node_name,
                "registry_event_id": registry_event_id,
                "correlation_id": str(correlation_id),
            },
        )

        return ModelMCPGeneratedToolRegistration(
            registration_id=uuid4(),
            name=node_name,
            description=description,
            generated_contract_hash=contract_hash,
            generated_handler_hash=handler_hash,
            generation_correlation_id=correlation_id,
            registry_event_version=registry_event_id,
        )

    async def upsert_tool(
        self,
        tool: ModelMCPToolDefinition,
        event_id: str,
    ) -> bool:
        """Upsert tool if event_id is newer. Returns True if updated.

        This method is idempotent - calling with the same event_id multiple
        times will only update the registry once. Out-of-order events with
        older event_ids are ignored.

        Args:
            tool: The tool definition to upsert.
            event_id: Unique event identifier for version tracking.
                Should be monotonically increasing (e.g., Kafka offset).

        Returns:
            True if the tool was updated (event_id was newer).
            False if the event was stale (existing event_id >= new event_id).

        Example:
            >>> registry = ServiceMCPToolRegistry()
            >>> tool = ModelMCPToolDefinition(name="my_tool", ...)
            >>> await registry.upsert_tool(tool, "event-002")
            True
            >>> await registry.upsert_tool(tool, "event-001")  # Older event
            False
        """
        correlation_id = uuid4()
        normalized_event_id = self._normalize_event_id(event_id)

        async with self._lock:
            existing_version = self._versions.get(tool.name)

            # Stale event check: ignore if normalized event_id <= existing version
            if existing_version and normalized_event_id <= existing_version:
                logger.debug(
                    "Ignoring stale event for tool",
                    extra={
                        "tool_name": tool.name,
                        "event_id": event_id,
                        "normalized_event_id": normalized_event_id,
                        "existing_version": existing_version,
                        "correlation_id": str(correlation_id),
                    },
                )
                return False

            # Update tool and version (store normalized form)
            self._tools[tool.name] = tool
            self._versions[tool.name] = normalized_event_id

            logger.info(
                "Tool upserted in registry",
                extra={
                    "tool_name": tool.name,
                    "event_id": event_id,
                    "normalized_event_id": normalized_event_id,
                    "previous_version": existing_version,
                    "correlation_id": str(correlation_id),
                },
            )
            return True

    async def remove_tool(self, tool_name: str, event_id: str) -> bool:
        """Remove tool if event_id is newer. Returns True if removed.

        This method is idempotent - calling with the same event_id multiple
        times will only remove the tool once. Out-of-order events with
        older event_ids are ignored.

        Args:
            tool_name: Name of the tool to remove.
            event_id: Unique event identifier for version tracking.

        Returns:
            True if the tool was removed (event_id was newer).
            False if the event was stale or tool didn't exist.

        Example:
            >>> registry = ServiceMCPToolRegistry()
            >>> await registry.remove_tool("my_tool", "event-003")
            True
            >>> await registry.remove_tool("my_tool", "event-002")  # Older
            False
        """
        correlation_id = uuid4()
        normalized_event_id = self._normalize_event_id(event_id)

        async with self._lock:
            existing_version = self._versions.get(tool_name)

            # Stale event check: ignore if normalized event_id <= existing version
            if existing_version and normalized_event_id <= existing_version:
                logger.debug(
                    "Ignoring stale remove event for tool",
                    extra={
                        "tool_name": tool_name,
                        "event_id": event_id,
                        "normalized_event_id": normalized_event_id,
                        "existing_version": existing_version,
                        "correlation_id": str(correlation_id),
                    },
                )
                return False

            # Remove tool if it exists
            removed = self._tools.pop(tool_name, None) is not None
            # Always update version to prevent re-adding with older event (store normalized form)
            self._versions[tool_name] = normalized_event_id

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
                    "Tool not found in registry for removal",
                    extra={
                        "tool_name": tool_name,
                        "event_id": event_id,
                        "correlation_id": str(correlation_id),
                    },
                )

            return removed

    async def get_tool(self, tool_name: str) -> ModelMCPToolDefinition | None:
        """Get a tool definition by name.

        Args:
            tool_name: Name of the tool to retrieve.

        Returns:
            The tool definition if found, None otherwise.

        Example:
            >>> registry = ServiceMCPToolRegistry()
            >>> tool = await registry.get_tool("my_tool")
            >>> if tool:
            ...     print(tool.description)
        """
        async with self._lock:
            return self._tools.get(tool_name)

    async def list_tools(self) -> list[ModelMCPToolDefinition]:
        """List all registered tool definitions.

        Returns:
            List of all tool definitions in the registry.
            The list is a snapshot - modifications after this call
            won't affect the returned list.

        Example:
            >>> registry = ServiceMCPToolRegistry()
            >>> tools = await registry.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        async with self._lock:
            return list(self._tools.values())

    async def clear(self) -> None:
        """Clear all tools and versions from the registry.

        This is useful for testing or server restart scenarios.
        """
        correlation_id = uuid4()

        async with self._lock:
            tool_count = len(self._tools)
            self._tools.clear()
            self._versions.clear()

            logger.info(
                "Registry cleared",
                extra={
                    "cleared_tool_count": tool_count,
                    "correlation_id": str(correlation_id),
                },
            )

    async def has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists in the registry.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool exists, False otherwise.
        """
        async with self._lock:
            return tool_name in self._tools

    async def get_tool_version(self, tool_name: str) -> str | None:
        """Get the last event_id for a tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            The last event_id (normalized) if found, None otherwise.
            Numeric IDs are zero-padded to 20 digits.
        """
        async with self._lock:
            return self._versions.get(tool_name)

    def describe(self) -> dict[str, object]:
        """Return registry metadata for observability.

        Returns:
            Dictionary with registry state information.
        """
        return {
            "service_name": "ServiceMCPToolRegistry",
            "tool_count": len(self._tools),
            "version_count": len(self._versions),
        }


__all__ = ["ServiceMCPToolRegistry"]
