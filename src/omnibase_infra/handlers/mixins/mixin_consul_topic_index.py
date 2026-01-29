# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Topic Index operations mixin.

This mixin provides topic index management for the ONEX node registry,
storing event bus configuration and maintaining a reverse index from
topics to subscriber node IDs.

Consul KV Structure:
    onex/nodes/{node_id}/event_bus/subscribe_topics     # [topic strings] for routing
    onex/nodes/{node_id}/event_bus/publish_topics       # [topic strings]
    onex/nodes/{node_id}/event_bus/subscribe_entries    # [full entries] for tooling
    onex/nodes/{node_id}/event_bus/publish_entries      # [full entries]
    onex/topics/{topic}/subscribers                     # [node_ids] reverse index

Design Decisions:
    1. Consul KV is the authoritative routing index
    2. Index updates are idempotent (safe for restarts/retries)
    3. Topic strings stored separately from full entries for routing efficiency
    4. Delta computation for efficient index updates

Operations:
    - Store event bus config for a node
    - Update topic → node_id reverse index (idempotent)
    - Add/remove subscriber from topic index
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar, cast
from uuid import UUID

from omnibase_infra.constants_topic_patterns import TOPIC_NAME_PATTERN
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConsulError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.models.registration import ModelNodeEventBusConfig

if TYPE_CHECKING:
    import consul as consul_lib

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ProtocolConsulTopicIndexDependencies(Protocol):
    """Protocol defining required dependencies for topic index operations.

    HandlerConsul must provide these attributes/methods for the mixin to work.
    """

    _client: consul_lib.Consul | None
    _config: object | None

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute operation with retry logic."""
        ...


class MixinConsulTopicIndex:
    """Mixin providing Consul topic index management operations.

    This mixin extracts topic index operations from HandlerConsul to support
    ONEX node registry with event bus configuration storage and reverse indexing.

    Required Dependencies (from host class):
        - _client: consul.Consul client instance
        - _config: Handler configuration
        - _execute_with_retry: Retry execution method

    KV Structure:
        Node event bus config:
            onex/nodes/{node_id}/event_bus/subscribe_topics     # JSON array of topic strings
            onex/nodes/{node_id}/event_bus/publish_topics       # JSON array of topic strings
            onex/nodes/{node_id}/event_bus/subscribe_entries    # JSON array of full entries
            onex/nodes/{node_id}/event_bus/publish_entries      # JSON array of full entries

        Topic reverse index:
            onex/topics/{topic}/subscribers                     # JSON array of node_ids
    """

    # Instance attribute declarations for type checking
    _client: consul_lib.Consul | None
    _config: object | None

    # Methods from host class (abstract stubs for type checking)
    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute operation with retry logic - provided by host class."""
        raise NotImplementedError("Must be provided by implementing class")  # type: ignore[return-value]

    def _validate_topic_format(self, topic: str, correlation_id: UUID) -> None:
        """Validate topic format for safe use in Consul KV paths.

        Topics are interpolated into Consul KV paths like `onex/topics/{topic}/subscribers`.
        Invalid characters (especially slashes) could create unexpected key hierarchies
        or bypass ACL prefix matching.

        Args:
            topic: The topic string to validate.
            correlation_id: Correlation ID for error context.

        Raises:
            ProtocolConfigurationError: If topic contains invalid characters.
                Valid characters: alphanumeric (a-zA-Z0-9), periods (.), underscores (_),
                and hyphens (-).
        """
        if not TOPIC_NAME_PATTERN.match(topic):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="validate_topic_format",
                target_name="consul_handler",
            )
            raise ProtocolConfigurationError(
                f"Topic name '{topic}' contains invalid characters. "
                "Only alphanumeric characters, periods (.), underscores (_), "
                "and hyphens (-) are allowed. Slashes and other special "
                "characters are prohibited to prevent Consul KV path traversal.",
                context=context,
                parameter="topic",
                value=topic,
            )

    async def _kv_get_raw(
        self,
        key: str,
        correlation_id: UUID,
    ) -> str | None:
        """Get raw string value from Consul KV store.

        This is a simplified internal method for topic index operations.
        Unlike _kv_get, it returns the raw string value or None.

        Args:
            key: KV key path
            correlation_id: Correlation ID for tracing

        Returns:
            The stored value as a string, or None if key doesn't exist.

        Raises:
            InfraConsulError: If Consul client not initialized or operation fails.
        """
        # Early validation - fail fast before entering retry machinery
        if self._client is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_get_raw",
                target_name="consul_handler",
            )
            raise InfraConsulError(
                "Consul client not initialized",
                context=context,
                consul_key=key,
            )

        def get_func() -> tuple[int, dict[str, object] | None]:
            # NOTE: This inner check is NOT redundant. It's required because:
            # 1. Type narrowing: Outer check narrows self._client in method scope,
            #    but closures capture `self`, not the narrowed type. Without this
            #    check, mypy would error on self._client.kv access.
            # 2. Race protection: Between outer check and thread pool execution
            #    (via run_in_executor), _client could become None during cleanup.
            if self._client is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_get_raw",
                    target_name="consul_handler",
                )
                raise InfraConsulError(
                    "Consul client not initialized",
                    context=ctx,
                    consul_key=key,
                )
            index, data = self._client.kv.get(key, recurse=False)
            return index, data

        KVGetResult = tuple[int, dict[str, object] | None]
        result = await self._execute_with_retry(
            "consul.kv_get_raw",
            get_func,
            correlation_id,
        )
        _, data = cast("KVGetResult", result)

        if data is None:
            return None

        value = data.get("Value")
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        return None

    async def _kv_put_raw(
        self,
        key: str,
        value: str,
        correlation_id: UUID,
    ) -> bool:
        """Put raw string value to Consul KV store.

        This is a simplified internal method for topic index operations.
        Unlike _kv_put, it takes a string value directly.

        Args:
            key: KV key path
            value: String value to store
            correlation_id: Correlation ID for tracing

        Returns:
            True if the operation succeeded.

        Raises:
            InfraConsulError: If Consul client not initialized or operation fails.
        """
        # Early validation - fail fast before entering retry machinery
        if self._client is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_put_raw",
                target_name="consul_handler",
            )
            raise InfraConsulError(
                "Consul client not initialized",
                context=context,
                consul_key=key,
            )

        def put_func() -> bool:
            # NOTE: This inner check is NOT redundant. It's required because:
            # 1. Type narrowing: Outer check narrows self._client in method scope,
            #    but closures capture `self`, not the narrowed type. Without this
            #    check, mypy would error on self._client.kv access.
            # 2. Race protection: Between outer check and thread pool execution
            #    (via run_in_executor), _client could become None during cleanup.
            if self._client is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_put_raw",
                    target_name="consul_handler",
                )
                raise InfraConsulError(
                    "Consul client not initialized",
                    context=ctx,
                    consul_key=key,
                )
            result: bool = self._client.kv.put(key, value)
            return result

        result = await self._execute_with_retry(
            "consul.kv_put_raw",
            put_func,
            correlation_id,
        )
        return cast("bool", result)

    async def _store_node_event_bus(
        self,
        node_id: str,
        event_bus: ModelNodeEventBusConfig,
        correlation_id: UUID,
    ) -> None:
        """Store event_bus config in Consul KV.

        Stores both topic strings (for efficient routing lookups) and full
        entries (for tooling and introspection) separately.

        Args:
            node_id: The node identifier.
            event_bus: The resolved event bus configuration.
            correlation_id: Correlation ID for tracing.

        Raises:
            InfraConsulError: If Consul client not initialized or operation fails.
        """
        logger.debug(
            "Storing event_bus config for node %s",
            node_id,
            extra={"correlation_id": str(correlation_id), "node_id": node_id},
        )

        # Store topic strings (for routing)
        await self._kv_put_raw(
            key=f"onex/nodes/{node_id}/event_bus/subscribe_topics",
            value=json.dumps(event_bus.subscribe_topic_strings),
            correlation_id=correlation_id,
        )
        await self._kv_put_raw(
            key=f"onex/nodes/{node_id}/event_bus/publish_topics",
            value=json.dumps(event_bus.publish_topic_strings),
            correlation_id=correlation_id,
        )

        # Store full entries (for tooling)
        await self._kv_put_raw(
            key=f"onex/nodes/{node_id}/event_bus/subscribe_entries",
            value=json.dumps([e.model_dump() for e in event_bus.subscribe_topics]),
            correlation_id=correlation_id,
        )
        await self._kv_put_raw(
            key=f"onex/nodes/{node_id}/event_bus/publish_entries",
            value=json.dumps([e.model_dump() for e in event_bus.publish_topics]),
            correlation_id=correlation_id,
        )

        logger.info(
            "Stored event_bus config for node %s: %d subscribe topics, %d publish topics",
            node_id,
            len(event_bus.subscribe_topics),
            len(event_bus.publish_topics),
            extra={"correlation_id": str(correlation_id), "node_id": node_id},
        )

    async def _update_topic_index(
        self,
        node_id: str,
        event_bus: ModelNodeEventBusConfig,
        correlation_id: UUID,
    ) -> None:
        """Idempotent update of topic -> node_id reverse index.

        Computes the delta between previously registered topics and new topics,
        then adds/removes the node_id from the appropriate topic subscriber lists.

        This method is idempotent - calling it multiple times with the same
        parameters produces the same result.

        Args:
            node_id: The node identifier.
            event_bus: The resolved event bus configuration.
            correlation_id: Correlation ID for tracing.

        Raises:
            InfraConsulError: If Consul client not initialized or operation fails.

        Note:
            This operation is NOT atomic. In high-concurrency scenarios with multiple
            nodes updating the same topic's subscriber list simultaneously, race
            conditions may occur. For MVP, this is an accepted limitation.

            For production with high concurrency, consider:
            - Using Consul transactions (txn endpoint) for atomic read-modify-write
            - Implementing optimistic locking with Consul's ModifyIndex
        """
        # NOTE: Non-atomic read-modify-write. See docstring for concurrency notes.
        logger.debug(
            "Updating topic index for node %s",
            node_id,
            extra={"correlation_id": str(correlation_id), "node_id": node_id},
        )

        # Get previous registrations
        previous_key = f"onex/nodes/{node_id}/event_bus/subscribe_topics"
        previous_result = await self._kv_get_raw(previous_key, correlation_id)
        try:
            old_topics = set(json.loads(previous_result) if previous_result else [])
        except (json.JSONDecodeError, TypeError) as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_get_raw",
                target_name="consul_handler",
            )
            raise InfraConsulError(
                "Invalid JSON in Consul topic index",
                context=context,
                consul_key=previous_key,
            ) from e
        new_topics = set(event_bus.subscribe_topic_strings)

        # Compute delta
        topics_to_add = new_topics - old_topics
        topics_to_remove = old_topics - new_topics

        logger.debug(
            "Topic index delta for node %s: +%d topics, -%d topics",
            node_id,
            len(topics_to_add),
            len(topics_to_remove),
            extra={
                "correlation_id": str(correlation_id),
                "node_id": node_id,
                "topics_added": len(topics_to_add),
                "topics_removed": len(topics_to_remove),
            },
        )

        # Update index - add to new topics
        for topic in topics_to_add:
            await self._add_subscriber_to_topic(topic, node_id, correlation_id)

        # Update index - remove from old topics
        for topic in topics_to_remove:
            await self._remove_subscriber_from_topic(topic, node_id, correlation_id)

        logger.info(
            "Updated topic index for node %s: added to %d topics, removed from %d topics",
            node_id,
            len(topics_to_add),
            len(topics_to_remove),
            extra={"correlation_id": str(correlation_id), "node_id": node_id},
        )

    async def _add_subscriber_to_topic(
        self,
        topic: str,
        node_id: str,
        correlation_id: UUID,
    ) -> None:
        """Add node_id to topic's subscriber list (idempotent).

        If the node_id is already in the list, this is a no-op.
        The subscriber list is stored as a sorted JSON array.

        Args:
            topic: The topic string.
            node_id: The node identifier to add.
            correlation_id: Correlation ID for tracing.

        Raises:
            InfraConsulError: If Consul client not initialized or operation fails.
            ProtocolConfigurationError: If topic contains invalid characters.

        Note:
            Non-atomic read-modify-write. See _update_topic_index docstring for
            concurrency notes. Accepted MVP limitation.
        """
        # Validate topic format before KV path interpolation to prevent path traversal
        self._validate_topic_format(topic, correlation_id)

        # NOTE: Non-atomic read-modify-write. See _update_topic_index for details.
        key = f"onex/topics/{topic}/subscribers"
        existing = await self._kv_get_raw(key, correlation_id)
        try:
            subscribers = set(json.loads(existing) if existing else [])
        except (json.JSONDecodeError, TypeError) as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_get_raw",
                target_name="consul_handler",
            )
            raise InfraConsulError(
                "Invalid JSON in Consul topic subscriber list",
                context=context,
                consul_key=key,
            ) from e
        subscribers.add(node_id)
        await self._kv_put_raw(key, json.dumps(sorted(subscribers)), correlation_id)

        logger.debug(
            "Added node %s to topic %s subscribers",
            node_id,
            topic,
            extra={
                "correlation_id": str(correlation_id),
                "node_id": node_id,
                "topic": topic,
            },
        )

    async def _remove_subscriber_from_topic(
        self,
        topic: str,
        node_id: str,
        correlation_id: UUID,
    ) -> None:
        """Remove node_id from topic's subscriber list.

        If the node_id is not in the list, this is a no-op.
        The subscriber list is stored as a sorted JSON array.

        Args:
            topic: The topic string.
            node_id: The node identifier to remove.
            correlation_id: Correlation ID for tracing.

        Raises:
            InfraConsulError: If Consul client not initialized or operation fails.
            ProtocolConfigurationError: If topic contains invalid characters.

        Note:
            Non-atomic read-modify-write. See _update_topic_index docstring for
            concurrency notes. Accepted MVP limitation.
        """
        # Validate topic format before KV path interpolation to prevent path traversal
        self._validate_topic_format(topic, correlation_id)

        # NOTE: Non-atomic read-modify-write. See _update_topic_index for details.
        key = f"onex/topics/{topic}/subscribers"
        existing = await self._kv_get_raw(key, correlation_id)
        if existing:
            try:
                subscribers = set(json.loads(existing))
            except (json.JSONDecodeError, TypeError) as e:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_get_raw",
                    target_name="consul_handler",
                )
                raise InfraConsulError(
                    "Invalid JSON in Consul topic subscriber list",
                    context=context,
                    consul_key=key,
                ) from e
            subscribers.discard(node_id)
            await self._kv_put_raw(key, json.dumps(sorted(subscribers)), correlation_id)

            logger.debug(
                "Removed node %s from topic %s subscribers",
                node_id,
                topic,
                extra={
                    "correlation_id": str(correlation_id),
                    "node_id": node_id,
                    "topic": topic,
                },
            )

    async def _get_topic_subscribers(
        self,
        topic: str,
        correlation_id: UUID,
    ) -> list[str]:
        """Get list of node_ids subscribed to a topic.

        Args:
            topic: The topic string.
            correlation_id: Correlation ID for tracing.

        Returns:
            List of node_ids subscribed to the topic, empty list if none.

        Raises:
            InfraConsulError: If Consul client not initialized or operation fails.
            ProtocolConfigurationError: If topic contains invalid characters.
        """
        # Validate topic format before KV path interpolation to prevent path traversal
        self._validate_topic_format(topic, correlation_id)

        key = f"onex/topics/{topic}/subscribers"
        existing = await self._kv_get_raw(key, correlation_id)
        if existing:
            try:
                return sorted(json.loads(existing))
            except (json.JSONDecodeError, TypeError) as e:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_get_raw",
                    target_name="consul_handler",
                )
                raise InfraConsulError(
                    "Invalid JSON in Consul topic subscriber list",
                    context=context,
                    consul_key=key,
                ) from e
        return []


__all__: list[str] = ["MixinConsulTopicIndex"]
