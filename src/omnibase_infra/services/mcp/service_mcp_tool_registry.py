# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP Tool Registry - Event-loop safe in-memory cache of MCP tool definitions.

This service provides a thread-safe registry for MCP tool definitions, supporting:
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
import logging
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
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
